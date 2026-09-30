"""Train a route-risk ensemble (CNN-GRU, one model per seed) and write checkpoints, logits and a JSON record.

The final shared rigid/soil model with the history encoder (released as ``deploy_a1_haux_gru``), from the repo root:

    PYTHONPATH=src python -m nedm.traversing.training.train_risk_model --out runs/risk --tag deploy_a1_haux_gru \\
        --mode deploy --cond hist_aux --domain-filter both --seeds 5 --seed0 0

Without ``--ds`` it reads the three released training files at the downloader's restore paths (DEFAULT_DS). The
single-ground planners of the arena and vehicle studies use one ``--ds`` file each, ``--cond none`` and
``--domain-filter rigid|crm`` (see README.md). Per seed s: ``<out>/<tag>_s<s>.pt`` (checkpoint, see
risk_model.load_risk_model); per run: ``<tag>_logits.npz`` (route logits of every fitted/evaluated row) and
``<tag>.json`` (arguments, rows, per-seed and ensemble metrics, see risk_metrics.py).

Training: AdamW (lr 2e-3, weight decay 1e-4), one-cycle schedule over epochs * (n_fit // bs) steps, gradient norm
clipped at 5, survival loss on the per-station hazards (+ aux-weight x ground-type BCE on rows with a visible
history for ``--cond hist_aux``; each row's whole history window is masked with probability ``--hist-drop``).
Seeding: torch.manual_seed(seed) for the weights and dropout, a CPU generator seeded with the seed for the batch
permutations and the history drop.
"""

from __future__ import annotations

import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # read at the first CUDA allocation

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nedm.traversing.training.risk_data import PRED_BS, Batches, RiskData
from nedm.traversing.training.risk_metrics import all_metrics, domain_head_metrics
from nedm.traversing.training.risk_model import (CONDS, DOMAIN_NAME, DOMAIN_VOCAB, GEOM_COLS, WIDTH, RiskModel, encode_history,
                        load_risk_model, route_logit, score, survival_nll)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DS = ['artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor_plus_branch_both.npz',
              'artifacts/traverse/crm_improve_20260922/datasets/short_anchor.npz',
              'artifacts/traverse/crm_improve_20260922/datasets/anchor_k60.npz']
LR, WD = 2e-3, 1e-4
HIST_LAYOUT = ('hist[:, t, :12] = state[k-T+1+t, hist_cols_state]; hist[:, t, 12:] = action[k-T+t, hist_cols_action]; '
               'hmask True where the frame exists; the model consumes the newest hist_T frames of which the newest '
               'hist_valid_T are visible (older ones masked)')


# ----------------------------------------------------------------------------- training
def risk_loss(model, D, a, k, hm):
    x, c, h, m = D.inputs(k, hm)
    out = model(x, c, h, m)
    loss = survival_nll(out['haz'], D.ev[k]).mean()
    if model.use_hist:
        vis = m.any(1)   # ground-type BCE only on rows whose window is (at least partly) visible
        if vis.any():
            bce = F.binary_cross_entropy_with_logits(out['dom'][vis], D.dom[k][vis].to(torch.float32))
            loss = loss + a.aux_weight * bce
    return loss


def run_epochs(model, D, a, gen, hist_drop, log):
    """AdamW + one-cycle over epochs * (n_fit // bs) steps; returns the per-step losses."""
    n = len(D.fit_idx)
    bs = min(a.bs, n)
    steps = max(a.epochs * (n // bs), 1)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    batches = Batches(D.fit_idx, D.dom_np, bs, a.crm_batch_frac, a.domain_filter == 'both', gen, D.device)
    losses = []
    for step in range(steps):
        model.train()
        k = batches.next()
        hm = None
        if D.hmask is not None:
            hm = D.hmask[k]
            if hist_drop > 0:   # drawn after the batch permutation, from the same generator
                drop = torch.rand(len(k), generator=gen, device='cpu').to(D.device) < hist_drop
                hm = hm & ~drop[:, None]
        loss = risk_loss(model, D, a, k, hm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if step < steps - 1:
            sch.step()
        losses.append(loss.item())
        assert np.isfinite(losses[-1]), f'non-finite loss at step {step}'
    log.update(steps=steps, first_loss=losses[0], final_loss=losses[-1], mean_loss_last50=float(np.mean(losses[-50:])),
               batch_quota=batches.quota(D.dom_np))
    return losses


def predict(model, D, bs=PRED_BS):
    """(route logits, ground-type logits or None) of every row, eval mode."""
    model.eval()
    z, dl = [], []
    with torch.no_grad():
        for i in range(0, D.n, bs):
            k = torch.arange(i, min(i + bs, D.n), device=D.device)
            o = model(*D.inputs(k))
            z.append(route_logit(o['haz']).float().cpu().numpy())
            if 'dom' in o:
                dl.append(o['dom'].float().cpu().numpy())
    return np.concatenate(z), (np.concatenate(dl) if dl else None)


def train_one(D, a, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cuda = str(D.device).startswith('cuda')
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = RiskModel(a.cond, D.X.shape[1], D.nctx).to(D.device)
    t0 = time.time()
    s1 = {}
    run_epochs(model, D, a, gen, hist_drop=(a.hist_drop if model.use_hist else 0.0), log=s1)
    if cuda:
        torch.cuda.synchronize()
    took = time.time() - t0
    if cuda:
        torch.cuda.empty_cache()
    t1 = time.time()
    z, dl = predict(model, D)
    tp = time.time() - t1
    row = dict(arch=a.arch, hist_enc=a.hist_enc, hist_T=D.hist_T, hist_valid_T=D.hist_T, hist_window=a.hist_window,
               ctx=a.ctx, cond=a.cond, domain_filter=a.domain_filter, split_eval=a.split_eval, mode=a.mode,
               hist_drop=a.hist_drop, crm_batch_frac=a.crm_batch_frac, lr=LR, wd=WD, epochs=a.epochs, seed=seed,
               params=sum(p.numel() for p in model.parameters()), secs=round(took, 1), secs_predict=round(tp, 1),
               sec_per_step=round(took / max(s1['steps'], 1), 4), n_fit=int(D.fit.sum()), steps=s1['steps'],
               final_loss=s1['final_loss'], stages=dict(train=s1),
               dev=all_metrics(z, D.d, D.dev), heldout=all_metrics(z, D.d, D.test))
    if dl is not None:
        row['domain_head'] = dict(dev=domain_head_metrics(dl, D.d, D.dev), heldout=domain_head_metrics(dl, D.d, D.test))
    if cuda:
        row['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        row['gpu_reserved_peak_gb'] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    return model, row, z


# ----------------------------------------------------------------------------- checkpoint
def checkpoint_dict(model, D, a, seed, tag):
    """The released checkpoint format (model_kind 'ci_train'); fields of options this port does not offer keep the
    value every released model was trained with."""
    return dict(
        model_kind='ci_train', arch='gru', arch_kind='gru', tx_d=0, tx_layers=0, hist_enc='gru', hist_tx=None,
        cond=model.cond, ctx_mode='geom', ctx_names=['goal_dx', 'goal_dy', 'goal_dist', 'start_yaw', 'route_len'],
        geom_cols=GEOM_COLS, vel_hist_channels=[0, 6], domain_vocab=DOMAIN_VOCAB,
        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        cin=int(D.X.shape[1]), nctx=int(D.nctx), zdim=int(model.zdim), width=WIDTH,
        hist_cols=list(D.hist_cols), hist_cols_state=list(D.hist_cols[:12]), hist_cols_action=list(D.hist_cols[12:]),
        hist_T=int(D.hist_T), hist_valid_T=int(D.hist_T), hist_window='mask', hist_T_stored=int(D.hist_T),
        hist_layout=HIST_LAYOUT,
        norm=dict(mu=np.asarray(D.norm['mu']), sd=np.asarray(D.norm['sd']), cont_index=list(D.norm['cont_index']),
                  channels=list(D.norm['channels'])),
        ctx_mu=np.asarray(D.ctx_mu, np.float32), ctx_sd=np.asarray(D.ctx_sd, np.float32),
        hist_mu=np.asarray(D.hist_mu, np.float32), hist_sd=np.asarray(D.hist_sd, np.float32),
        train_rows=int(D.fit.sum()),
        train_rows_by_domain={DOMAIN_NAME[x]: int((D.dom_np[D.fit] == x).sum()) for x in (0, 1)},
        split_hash=D.split_hash, split_eval=a.split_eval, mode=a.mode, domain_filter=a.domain_filter,
        startup_only=False,
        hist_drop=float(a.hist_drop), crm_batch_frac=float(a.crm_batch_frac), data_frac_crm=1.0, data_frac_rigid=1.0,
        data_seed=0, row_weight={}, lr=LR, wd=WD, epochs=a.epochs, seed=int(seed), tag=tag,
        ds=[f['path'] for f in D.files], args=vars(a))


def roundtrip(path, D, zref):
    """Reload the checkpoint and score the raw sample: equal to predict() (< 1e-5, batch-aligned); shared-window,
    startup and precomputed-z paths agree (< 1e-3, other batch shapes)."""
    model, ck = load_risk_model(path, D.device)
    r = D.raw
    s = score(model, ck, r['X'], r['geom'], r['hist'], r['hmask'], bs=PRED_BS)
    out = dict(n=int(len(s)), max_abs_diff=float(np.abs(s - zref[r['idx']]).max()))
    u = np.sort(np.random.default_rng(1).choice(len(s), min(len(s), 300), replace=False))
    su = score(model, ck, r['X'][u], r['geom'][u], None if r['hist'] is None else r['hist'][u],
               None if r['hmask'] is None else r['hmask'][u])
    loose = dict(unaligned_diff=float(np.abs(su - zref[r['idx'][u]]).max()))
    if r['hist'] is not None:
        k = min(8, len(s))
        X8, g8 = r['X'][:k], r['geom'][:k]
        a1 = score(model, ck, X8, g8, r['hist'][:1], r['hmask'][:1])
        a2 = score(model, ck, X8, g8, np.repeat(r['hist'][:1], k, 0), np.repeat(r['hmask'][:1], k, 0))
        a3 = score(model, ck, X8, g8, z=encode_history(model, ck, r['hist'][:1], r['hmask'][:1]))
        s0 = score(model, ck, X8, g8, None, None)
        T = ck['hist_T_stored']
        s0b = score(model, ck, X8, g8, np.zeros((k, T, 15), np.float32), np.zeros((k, T), bool))
        s0c = score(model, ck, X8, g8, z=encode_history(model, ck, None, None))
        out.update(precomputed_z_diff=float(np.abs(a1 - a3).max()), startup_z_diff=float(np.abs(s0 - s0c).max()))
        loose.update(shared_window_diff=float(np.abs(a1 - a2).max()),
                     startup_zero_window_diff=float(np.abs(s0 - s0b).max()))
    bad = {k: v for k, v in out.items() if k != 'n' and v > 1e-5}
    bad.update({k: v for k, v in loose.items() if v > 1e-3})
    assert not bad, f'round trip failed (aligned 1e-5, batch-shape checks 1e-3): {bad}'
    out.update(loose)
    return out


# ----------------------------------------------------------------------------- main
def fmt(m, key):
    r = m.get(key, {})
    return ' '.join(f'{k}={r[k]:.3f}' for k in ('W_unsafe', 'P_unsafe', 'S_unsafe', 'pick_fail', 'brier_unsafe')
                    if k in r and np.isfinite(r[k])) + f" n={r.get('n', 0)}"


def print_block(prefix, h):
    for dk in [k for k in ('rigid', 'crm') if k in h['all']]:
        b = h['all'][dk]
        print(f"{prefix} {dk:5s} startup[{fmt(b, 'startup')}] established[{fmt(b, 'established')}]", flush=True)
    for name, b in list(h['by_anchor'].items()) + [(f'src:{k}', v) for k, v in h['by_source'].items()]:
        cells = ' | '.join(f"{dk} {fmt(b[dk], 'all')}" for dk in ('rigid', 'crm') if dk in b)
        print(f'{prefix}   {name:14s} {cells}', flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ds', action='append', default=None,
                    help='npz dataset; repeat to concatenate several files (default: the three DEFAULT_DS files)')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--tag', default=None, help='file-name prefix (default: built from the options)')
    ap.add_argument('--cond', choices=CONDS, default='none',
                    help='none: route geometry only; hist_aux: + 2 s history code z and the ground-type head')
    ap.add_argument('--domain-filter', choices=['crm', 'rigid', 'both'], default='both', help='ground types kept')
    ap.add_argument('--crm-batch-frac', type=float, default=0.5,
                    help='soil share of each batch when both grounds are fitted')
    ap.add_argument('--hist-drop', type=float, default=0.2,
                    help="probability of masking a training row's whole history")
    ap.add_argument('--aux-weight', type=float, default=0.5, help='weight of the ground-type BCE (hist_aux)')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--roundtrip-check', action='store_true',
                    help='reload each checkpoint and compare score() with predict()')
    ap.add_argument('--device', default=None, help='default cuda when available, else cpu')
    # accepted so the recorded commands run unchanged; each has the one value every released model used
    ap.add_argument('--mode', choices=['deploy'], default='deploy', help='fit every train row')
    ap.add_argument('--split-eval', choices=['val'], default='val', help='split of the held-out rows')
    ap.add_argument('--arch', choices=['gru'], default='gru')
    ap.add_argument('--hist-enc', choices=['gru'], default='gru')
    ap.add_argument('--hist-window', choices=['mask'], default='mask')
    ap.add_argument('--ctx', choices=['geom'], default='geom')
    return ap


def main():
    a = build_parser().parse_args()
    if a.ds is None:
        a.ds = [str(REPO_ROOT / p) for p in DEFAULT_DS]
    a.device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    assert 0.0 <= a.crm_batch_frac <= 1.0
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or (f'ci_{a.arch}_{a.hist_enc}_{a.cond}_{a.ctx}_Tall_{a.domain_filter}_{a.mode}_{a.split_eval}'
                    + (f'_cb{a.crm_batch_frac:g}' if a.crm_batch_frac != 0.5 else ''))
    D = RiskData(a.ds, a, a.device, roundtrip_sample=a.roundtrip_check)
    dom_counts = {DOMAIN_NAME[x]: int((D.dom_np[D.fit] == x).sum()) for x in (0, 1)}
    src = D.d['source'].astype(str)
    src_counts = {str(x): int((src[D.fit] == x).sum()) for x in sorted(np.unique(src[D.fit]))}
    gpu_data = round(torch.cuda.memory_allocated() / 2 ** 30, 2) if str(a.device).startswith('cuda') else None
    print(f'=== {tag}: files {len(D.files)} rows used {D.n} fit {int(D.fit.sum())} {dom_counts} {src_counts} '
          f'dev {int(D.dev.sum())} heldout({a.split_eval}) {int(D.test.sum())} nctx {D.nctx} hist frames {D.hist_T} '
          f'split_hash {D.split_hash[:8]} load {D.load_secs}s data on gpu {gpu_data} GB'
          + (f' dropped keys {D.dropped_keys}' if D.dropped_keys else ''), flush=True)
    rows, logits = [], []
    for s in range(a.seed0, a.seed0 + a.seeds):
        model, row, z = train_one(D, a, s)
        rows.append(row)
        logits.append(z)
        print_block(f'  s{s}', row['heldout'])
        extra = f"  domain_head {row['domain_head']['heldout']}" if 'domain_head' in row else ''
        print(f"  s{s} loss {row['final_loss']:.4f} {row['secs']}s ({row['sec_per_step']} s/step, "
              f"{row['steps']} steps) {row['params']}p gpu_peak {row.get('gpu_peak_gb', 'n/a')} GB{extra}", flush=True)
        p = f'{a.out}/{tag}_s{s}.pt'
        torch.save(checkpoint_dict(model, D, a, s, tag), p)
        if a.roundtrip_check:
            row['roundtrip'] = roundtrip(p, D, z)
            print(f"  s{s} roundtrip {row['roundtrip']}", flush=True)
    ens = np.mean(logits, 0)
    np.savez_compressed(f'{a.out}/{tag}_logits.npz', id=D.d['id'], ensemble_logit=ens, member_logits=np.stack(logits),
                        fit=D.fit, dev=D.dev, heldout=D.test, domain=D.d['domain'], anchor_frame=D.d['anchor_frame'],
                        group=D.d['group'], source=D.d['source'], unsafe=D.d['unsafe'], fail=D.d['fail'],
                        file_index=D.d['file_index'])
    summary = dict(tag=tag, args=vars(a), files=D.files, dropped_keys=D.dropped_keys, n_rows_used=int(D.n),
                   n_fit=int(D.fit.sum()), fit_by_domain=dom_counts, fit_by_source=src_counts, split_hash=D.split_hash,
                   load_secs=D.load_secs, gpu_data_gb=gpu_data, members=rows,
                   ensemble=dict(dev=all_metrics(ens, D.d, D.dev), heldout=all_metrics(ens, D.d, D.test)))
    with open(f'{a.out}/{tag}.json', 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print_block(f'ENSEMBLE {tag}', summary['ensemble']['heldout'])


if __name__ == '__main__':
    main()
