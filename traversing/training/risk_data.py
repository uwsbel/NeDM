"""Route-risk training data: dataset files -> rows, splits, standardised device tensors and balanced batches.

Each ``--ds`` file is an npz of route rows: X (n,5,96,32) f16 corridor, ctx (n,22), id, group, split, fail, unsafe,
event_idx [, domain (0 rigid, 1 soil), anchor_frame, profile, source, episode, hist (n,40,15) f16, hmask (n,40)].
Several files are concatenated in memory (ids must be unique; per-row keys missing from some file are dropped and
listed). Rows:
  fit       split == 'train' (``--mode deploy``: every train row is fitted)
  dev       train rows whose group is in the dev fold (md5(group) % 5 == 0; in-sample, reported only)
  held-out  split == 'val' (``--split-eval val``)
``--domain-filter rigid|crm`` keeps one ground type. Evaluation-suite groups may never be fit rows (hard error).
Only fit, dev and held-out rows are kept. Corridor channels 0-3, the 5 geometry ctx columns and the history channels
are standardised on the fit rows. ``Batches`` draws half rigid / half soil batches when both grounds are fitted.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import time

import numpy as np
import torch

from risk_model import DOMAIN_NAME, GEOM_COLS, HIST_ACTION_COLS, HIST_COLS, prep_hist, prep_x

REQUIRED = ('X', 'ctx', 'id', 'group', 'split', 'fail', 'unsafe', 'event_idx')
HEAVY = ('X', 'hist', 'hmask', 'E', 'T')           # never kept whole in the row dict
META_KEYS = ('hist_cols', 'priv_names')            # per-file arrays that must agree between files
PRED_BS = 512                                      # predict() batch; the round-trip check scores whole batches
ROUNDTRIP_N = 2048                                 # rows in the round-trip sample
# Evaluation-suite groups (f104 planner/tracker suites, the unseen-arena and vehicle suites): never trained on.
SUITE_GROUPS = ['f104_crm_eval_group_*', 'f104_g1_test_group_*', 'f104_pair_group_*',
                'g260_test_group_*', 'g271_test_group_*', 'g251_test_group_*', 'g247_test_group_*',
                'g203_heldout_group_*', 'g228_heldout_group_*', 'g217_dev_group_*']


def dev_group(g: str) -> bool:
    return int(hashlib.md5(g.encode()).hexdigest(), 16) % 5 == 0


def is_suite_group(g: str) -> bool:
    return any(fnmatch.fnmatch(g, p) for p in SUITE_GROUPS)


def chan_stats(parts, rows_per_part, cont, chunk=4096):
    """Per-channel mean / sd over the given rows of several f16 arrays (float64 sums, chunked)."""
    s = np.zeros(len(cont))
    s2 = np.zeros(len(cont))
    n = 0
    for X16, rows in zip(parts, rows_per_part):
        for i in range(0, len(rows), chunk):
            x = X16[rows[i:i + chunk]][:, cont].astype(np.float64)
            s += x.sum((0, 2, 3))
            s2 += (x ** 2).sum((0, 2, 3))
            n += x.shape[0] * x.shape[2] * x.shape[3]
    mu = s / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


class RiskData:
    """All fitted/evaluated rows on ``device``: X (n,6,96,32) f32, ctx (n,5), ev, dom [, hist (n,40,15), hmask].
    Row metadata in ``d``; boolean row masks ``fit``, ``dev``, ``test``; fit statistics in norm, ctx_mu/sd and
    hist_mu/sd."""

    def __init__(self, paths, a, device, roundtrip_sample=False):
        t0 = time.time()
        self.device = device
        self.need_hist = a.cond == 'hist_aux'
        zs = [np.load(p, allow_pickle=True) for p in paths]
        ns = [len(z['id']) for z in zs]
        keysets = [set(z.files) for z in zs]
        for p, ks in zip(paths, keysets):
            miss = [k for k in REQUIRED + (('hist', 'hmask') if self.need_hist else ()) if k not in ks]
            assert not miss, f'{p}: missing keys {miss}'
        common = set.intersection(*keysets)
        self.dropped_keys = sorted(set.union(*keysets) - common)
        self.meta = {}
        for k in META_KEYS:
            if k in common:
                vals = [np.asarray(z[k]) for z in zs]
                assert all(v.shape == vals[0].shape and (v == vals[0]).all() for v in vals[1:]), \
                    f'{k} differs between --ds files'
                self.meta[k] = vals[0]
        d = {}
        for k in sorted(common - set(HEAVY) - set(META_KEYS)):
            parts = [z[k] for z in zs]
            if all(getattr(v, 'ndim', 0) >= 1 and v.shape[0] == n for v, n in zip(parts, ns)):
                d[k] = np.concatenate(parts) if len(parts) > 1 else parts[0]
            else:
                self.meta[k] = parts[0]
        fidx = np.repeat(np.arange(len(zs)), ns)
        loc = np.concatenate([np.arange(n) for n in ns])
        n_all = len(fidx)
        ids = d['id'].astype(str)
        u, c = np.unique(ids, return_counts=True)
        assert (c == 1).all(), f'{int((c > 1).sum())} duplicate ids across --ds files, e.g. {u[c > 1][:5].tolist()}'
        if 'domain' not in d:
            assert all(i.endswith(('@crm', '@rigid')) for i in ids), \
                'no domain array and ids are not suffixed @crm/@rigid'
            d['domain'] = np.array([1 if i.endswith('@crm') else 0 for i in ids], np.int8)
        if 'anchor_frame' not in d:
            d['anchor_frame'] = np.zeros(n_all, np.int32)
        if 'profile' not in d:
            d['profile'] = np.full(n_all, -1, np.int8)
        if 'source' not in d:
            d['source'] = np.array(['unknown'] * n_all, object)
        d['file_index'] = fidx.astype(np.int16)

        # rows: ground-type filter, then the fit / dev / held-out masks
        dom = d['domain'].astype(int)
        keep = np.ones(n_all, bool)
        if a.domain_filter != 'both':
            keep &= dom == (1 if a.domain_filter == 'crm' else 0)
        K = np.where(keep)[0]
        d = {k: v[K] for k, v in d.items()}
        fidx, loc = fidx[K], loc[K]
        sp = d['split'].astype(str)
        grp = d['group'].astype(str)
        isdev = np.array([dev_group(g) for g in grp])
        fit = sp == 'train'   # --mode deploy
        dev = fit & isdev
        test = sp == 'val'    # --split-eval val
        assert fit.sum() > 0, 'no training rows'
        assert test.sum() > 0, 'no rows with split == val'
        bad = sorted({g for g in np.unique(grp[fit]) if is_suite_group(g)})
        assert not bad, f'suite groups among the fit rows (evaluation data, never trained on): {bad[:5]}'
        bad_eval = sorted({g for g in np.unique(grp[dev | test]) if is_suite_group(g)})
        if bad_eval:
            print(f'WARNING: {len(bad_eval)} suite groups among the evaluated rows, e.g. {bad_eval[:3]}', flush=True)
        U = np.where(fit | dev | test)[0]
        self.d = {k: v[U] for k, v in d.items()}
        fidx, loc = fidx[U], loc[U]
        self.fit, self.dev, self.test = fit[U], dev[U], test[U]
        self.n = len(U)
        self.split_hash = hashlib.md5('\n'.join(sorted(self.d['id'][self.fit].astype(str))).encode()).hexdigest()
        self.files = [dict(path=os.path.abspath(p), rows=int(n), rows_used=int((fidx == i).sum()),
                           fit=int(self.fit[fidx == i].sum())) for i, (p, n) in enumerate(zip(paths, ns))]

        # heavy arrays, per file, used rows only
        Xp, Hp, Mp, offs = [], [], [], []
        for i, z in enumerate(zs):
            sel = loc[fidx == i]
            if len(sel) == 0:
                continue
            offs.append(int(np.flatnonzero(fidx == i)[0]))
            full = len(sel) == ns[i] and (sel == np.arange(ns[i])).all()
            Xf = z['X']
            assert Xf.shape[1:] == (5, 96, 32), Xf.shape
            Xp.append(Xf if full else Xf[sel])
            del Xf
            if self.need_hist:
                Hf, Mf = z['hist'], z['hmask'].astype(bool)
                assert Hf.shape[2] == 15 and Mf.shape == Hf.shape[:2], (Hf.shape, Mf.shape)
                Hp.append(Hf if full else Hf[sel])
                Mp.append(Mf if full else Mf[sel])
                del Hf, Mf
        self.fit_idx = np.where(self.fit)[0]
        dom = self.d['domain'].astype(int)
        self.dom_np = dom

        # corridor: channels 0-3 standardised on the fit rows, ones plane appended
        fit_rows = [np.where(self.fit[o:o + len(x)])[0] for o, x in zip(offs, Xp)]
        mu, sd = chan_stats(Xp, fit_rows, [0, 1, 2, 3])
        self.norm = dict(mu=mu, sd=sd, cont_index=[0, 1, 2, 3],
                         channels=['elev_rel', 'grade', 'cross', 'speed', 'valid'])
        self.X = torch.empty((self.n, 6, 96, 32), dtype=torch.float32, device=device)
        for o, x in zip(offs, Xp):
            for i in range(0, len(x), 4096):
                v = prep_x(torch.from_numpy(x[i:i + 4096].astype(np.float32)).to(device), self.norm)
                self.X[o + i:o + i + len(v)] = v
        H_all = (np.concatenate(Hp) if len(Hp) > 1 else Hp[0]) if self.need_hist else None
        M_all = (np.concatenate(Mp) if len(Mp) > 1 else Mp[0]) if self.need_hist else None

        # raw sample for the checkpoint round trip: whole predict() batches (random blocks, sorted)
        self.raw = None
        if roundtrip_sample:
            nb = (self.n + PRED_BS - 1) // PRED_BS
            blocks = np.sort(np.random.default_rng(0).permutation(nb)[:max(1, int(math.ceil(ROUNDTRIP_N / PRED_BS)))])
            rs = np.concatenate([np.arange(b * PRED_BS, min((b + 1) * PRED_BS, self.n)) for b in blocks])
            part_of = np.searchsorted(np.array(offs), rs, side='right') - 1
            self.raw = dict(idx=rs, X=np.stack([Xp[p][r - offs[p]].astype(np.float32) for p, r in zip(part_of, rs)]),
                            geom=self.d['ctx'][rs][:, GEOM_COLS].astype(np.float32),
                            hist=(H_all[rs].astype(np.float32) if self.need_hist else None),
                            hmask=(M_all[rs] if self.need_hist else None))
        del Xp

        self.hist_T = int(H_all.shape[1]) if self.need_hist else 40   # frames stored = frames the model consumes
        hc = [int(c) for c in self.meta['hist_cols']] if 'hist_cols' in self.meta else list(HIST_COLS)
        if len(hc) == 12:
            hc = hc + HIST_ACTION_COLS   # files store the 12 state columns; the 3 action channels follow
        assert len(hc) == 15, hc
        self.hist_cols = hc

        # context: 5 geometry columns standardised on the fit rows
        geom = self.d['ctx'].astype(np.float32)[:, GEOM_COLS]
        self.ctx_mu, self.ctx_sd = geom[self.fit].mean(0), geom[self.fit].std(0) + 1e-6
        ctx = (geom - self.ctx_mu) / self.ctx_sd
        self.ctx = torch.from_numpy(ctx.astype(np.float32)).to(device)
        self.nctx = ctx.shape[1]
        self.ev = torch.from_numpy(self.d['event_idx'].astype(np.int64)).to(device)
        self.dom = torch.from_numpy(dom).to(device)

        # history: per channel over the valid frames of the fit rows; masked frames zeroed
        if self.need_hist:
            H, M = H_all, M_all
            Hf = np.nan_to_num(H[self.fit].astype(np.float32))
            Mf = M[self.fit]
            cnt = Mf.sum()
            if cnt > 0:
                self.hist_mu = (Hf * Mf[..., None]).sum((0, 1)) / cnt
                self.hist_sd = np.sqrt(((Hf - self.hist_mu) ** 2 * Mf[..., None]).sum((0, 1)) / cnt) + 1e-6
            else:
                self.hist_mu = np.zeros(15, np.float32)
                self.hist_sd = np.ones(15, np.float32)
            del Hf, Mf
            self.hist = torch.empty((self.n, H.shape[1], 15), dtype=torch.float32, device=device)
            self.hmask = torch.from_numpy(np.ascontiguousarray(M)).to(device)
            for i in range(0, self.n, 8192):
                self.hist[i:i + 8192], _ = prep_hist(torch.from_numpy(H[i:i + 8192].astype(np.float32)).to(device),
                                                     self.hmask[i:i + 8192], self.hist_mu, self.hist_sd)
        else:
            self.hist = self.hmask = None
            self.hist_mu = np.zeros(15, np.float32)
            self.hist_sd = np.ones(15, np.float32)
        del H_all, M_all
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()   # release the upload temporaries
        self.load_secs = round(time.time() - t0, 1)

    def inputs(self, k, hm=None):
        """Batch tensors for rows k: X, ctx, hist, hmask (hm overrides the stored mask, e.g. after history drop)."""
        if self.hist is None:
            return self.X[k], self.ctx[k], None, None
        return self.X[k], self.ctx[k], self.hist[k], (self.hmask[k] if hm is None else hm)


class Batches:
    """Batch stream over the fit rows. With both grounds fitted (balanced) each batch is floor(bs*(1-F)) rigid rows
    + the rest soil (F = --crm-batch-frac), drawn from two independent per-ground permutations (re-drawn when
    exhausted); otherwise one permutation. Permutations use the CPU generator ``gen``."""

    def __init__(self, fit_idx, dom_np, bs, crm_frac, balanced, gen, device):
        self.bs, self.gen, self.device = bs, gen, device
        self.streams, self.q = [], []
        if balanced:
            q_r = int(math.floor(bs * (1.0 - crm_frac) + 1e-9))
            parts = [(fit_idx[dom_np[fit_idx] == 0], q_r), (fit_idx[dom_np[fit_idx] == 1], bs - q_r)]
            parts = [(p, q) for p, q in parts if len(p) > 0]
            if len(parts) == 1:
                parts = [(parts[0][0], bs)]
            parts = [(p, q) for p, q in parts if q > 0]
        else:
            parts = [(fit_idx, bs)]
        for p, q in parts:
            self.streams.append([torch.from_numpy(p).to(device), None, 0])
            self.q.append(q)

    def next(self):
        out = []
        for st, q in zip(self.streams, self.q):
            idx, perm, cur = st
            if perm is None or cur + q > len(perm):
                st[1] = perm = idx[torch.randperm(len(idx), generator=self.gen, device='cpu').to(self.device)]
                st[2] = cur = 0
            out.append(perm[cur:cur + q])
            st[2] = cur + q
        return torch.cat(out)

    def quota(self, dom_np):
        """{ground name: rows per batch} for the training record."""
        return dict(zip([DOMAIN_NAME.get(int(dom_np[s[0][0].item()]), '?') for s in self.streams], self.q))
