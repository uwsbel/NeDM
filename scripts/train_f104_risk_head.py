"""Overfit-one-arena risk model for the f104 planner, plus the baselines it must beat.

Inputs are strictly pre-drive (static RGB-D corridor, start state, planned route).
Evaluation is by held-out START/GOAL GROUP, using the collection's own 90/5/5 split,
so no route from a training group can appear in val/test.

The headline metric is route selection: every route in a group was actually driven in
Chrono, so choosing one and reading its recorded outcome is an exact counterfactual.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909'
DATA = ROOT + '/planner_dataset_v1.npz'
OUTD = ROOT + '/risk_head_v1'
os.makedirs(OUTD, exist_ok=True)
SEED = int(os.environ.get('SEED', '0'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load():
    d = np.load(DATA, allow_pickle=True)
    return {k: d[k] for k in d.files}


class RiskNet(nn.Module):
    def __init__(self, n_scal, n_state):
        super().__init__()
        ch = [4, 32, 64, 128, 128]
        blocks = []
        for i in range(4):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2 if i else (2, 1), padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.GELU()]
        self.cnn = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(n_scal + n_state, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(128 + 128, 128), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(128, 64), nn.GELU())
        self.head_fail = nn.Linear(64, 1)
        self.head_time = nn.Linear(64, 1)

    def forward(self, patch, vec):
        z = torch.cat([self.cnn(patch), self.mlp(vec)], 1)
        z = self.fuse(z)
        return self.head_fail(z).squeeze(1), self.head_time(z).squeeze(1)


# ---------------- metrics ----------------
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.min() == y.max():
        return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def within_group_pairwise(groups, y, s):
    """Fraction of (success, failure) pairs inside a group ranked correctly."""
    ok = tot = 0
    for g in np.unique(groups):
        m = groups == g
        yy, ss = y[m], s[m]
        if yy.min() == yy.max():
            continue
        for a in ss[yy == 0]:
            for b in ss[yy == 1]:
                tot += 1
                ok += (a < b) + 0.5 * (a == b)
    return (ok / tot if tot else float('nan')), tot


def selection(groups, y, score, times, rng=None, mode='argmin'):
    """Pick one route per group by score; return realized failure rate and mean time."""
    fails, ts = [], []
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        if mode == 'random':
            i = m[rng.integers(len(m))]
        elif mode == 'oracle':
            cand = m[y[m] == 0]
            i = cand[np.argmin(times[cand])] if len(cand) else m[0]
        else:
            i = m[np.argmin(score[m])]
        fails.append(y[i])
        if y[i] == 0:
            ts.append(times[i])
    return float(np.mean(fails)), (float(np.mean(ts)) if ts else float('nan')), len(np.unique(groups))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = load()
    split = d['split'].astype(str); group = d['group'].astype(str)
    y = d['failure'].astype(np.float32)
    tgoal = d['goal_time_s'].astype(np.float32)

    # hygiene: a group must live in exactly one split
    gs = {}
    for g, s in zip(group, split):
        gs.setdefault(g, set()).add(s)
    bad = [g for g, s in gs.items() if len(s) > 1]
    assert not bad, f'group leakage across splits: {bad[:5]}'
    print(f'groups {len(gs)}, episodes {len(y)}, failure rate {y.mean():.4f}')
    for s in ['train', 'val', 'test']:
        m = split == s
        print(f'  {s:6s} eps {m.sum():6d}  groups {len(set(group[m])):5d}  fail {y[m].mean():.4f}')

    patch = d['patch'].astype(np.float32)
    scal = d['scal'].astype(np.float32)
    state0 = d['state0'].astype(np.float32)
    # goal-relative geometry the planner would know
    gxy, sxy = d['goal_xy'].astype(np.float32), d['start_xy'].astype(np.float32)
    rel = gxy - sxy
    extra = np.stack([rel[:, 0], rel[:, 1], np.linalg.norm(rel, axis=1), d['start_yaw'].astype(np.float32)], 1)
    vec = np.concatenate([scal, state0, extra], 1)

    tr, va, te = split == 'train', split == 'val', split == 'test'
    mu, sd = vec[tr].mean(0), vec[tr].std(0) + 1e-6
    vecn = (vec - mu) / sd
    pmu, psd = patch[tr].mean((0, 2, 3), keepdims=True), patch[tr].std((0, 2, 3), keepdims=True) + 1e-6
    patchn = (patch - pmu) / psd

    # ---------------- baselines ----------------
    print('\n--- BASELINES (test split) ---')
    keys = list(d['scal_keys'].astype(str))
    base = {}
    for k in ['max_abs_grade', 'p95_abs_grade', 'mean_abs_grade', 'max_rough', 'route_len_m', 'max_abs_cross']:
        base[f'single:{k}'] = scal[:, keys.index(k)]
    # logistic regression on all terrain/route scalars = the terrain-profile predictor
    # torch baselines: same optimiser machinery as the learned model, so the only
    # difference is whether the RGB-D corridor is used at all.
    def fit_scalar_model(hidden, tag, epochs=300, lr=3e-3):
        torch.manual_seed(SEED)
        layers = []
        din = vecn.shape[1]
        for h in hidden:
            layers += [nn.Linear(din, h), nn.GELU()]
            din = h
        layers += [nn.Linear(din, 1)]
        m = nn.Sequential(*layers).to(DEV)
        o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        Xtr = torch.tensor(vecn[tr], device=DEV); Yt = torch.tensor(y[tr], device=DEV)
        Xva = torch.tensor(vecn[va], device=DEV)
        pwl = torch.tensor(float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)), device=DEV)
        bestv, bstate = -1, None
        for e in range(epochs):
            m.train()
            perm = torch.randperm(len(Xtr), device=DEV)
            for i in range(0, len(Xtr) - 255, 256):
                idx = perm[i:i + 256]
                loss = F.binary_cross_entropy_with_logits(m(Xtr[idx]).squeeze(1), Yt[idx], pos_weight=pwl)
                o.zero_grad(); loss.backward(); o.step()
            m.eval()
            with torch.no_grad():
                av = auc(y[va], torch.sigmoid(m(Xva).squeeze(1)).cpu().numpy())
            if av > bestv:
                bestv, bstate = av, {k: v.detach().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(bstate); m.eval()
        with torch.no_grad():
            out = torch.sigmoid(m(torch.tensor(vecn, device=DEV)).squeeze(1)).cpu().numpy()
        print(f'    [{tag}] best val AUC {bestv:.4f}')
        return out

    base['logreg:all_scalars'] = fit_scalar_model([], 'logreg')
    base['mlp:all_scalars(no image)'] = fit_scalar_model([128, 128], 'scalar-MLP')

    results = {}
    for name, s in base.items():
        a = auc(y[te], s[te]); w, n = within_group_pairwise(group[te], y[te], s[te])
        results[name] = dict(auc=a, pairwise=w, pairs=n)
        print(f'  {name:26s} AUC {a:.4f}  within-group {w:.4f} ({n} pairs)')

    # ---------------- model ----------------
    P = lambda m: torch.tensor(patchn[m], device=DEV)
    V = lambda m: torch.tensor(vecn[m], device=DEV)
    Y = lambda m: torch.tensor(y[m], device=DEV)
    T = lambda m: torch.tensor(tgoal[m], device=DEV)

    net = RiskNet(vec.shape[1] - 0, 0).to(DEV) if False else RiskNet(vec.shape[1], 0).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    ntr = int(tr.sum()); bs = 256
    epochs = int(os.environ.get('EPOCHS', '60'))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * max(1, ntr // bs))
    pw = torch.tensor(float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)), device=DEV)
    Ptr, Vtr, Ytr, Ttr = P(tr), V(tr), Y(tr), T(tr)
    best = (-1, None)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(ntr, device=DEV)
        tot = 0.0
        for i in range(0, ntr - bs + 1, bs):
            idx = perm[i:i + bs]
            lf, lt = net(Ptr[idx], Vtr[idx])
            loss = F.binary_cross_entropy_with_logits(lf, Ytr[idx], pos_weight=pw)
            ok = Ytr[idx] == 0
            if ok.any():
                loss = loss + 0.05 * F.huber_loss(lt[ok], Ttr[idx][ok])
            opt.zero_grad(); loss.backward(); opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss)
        net.eval()
        with torch.no_grad():
            sv = torch.sigmoid(net(P(va), V(va))[0]).cpu().numpy()
        a = auc(y[va], sv)
        if a > best[0]:
            best = (a, {k: v.detach().clone() for k, v in net.state_dict().items()})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'  epoch {ep:3d} loss {tot/max(1,ntr//bs):.4f}  val AUC {a:.4f}')
    net.load_state_dict(best[1]); net.eval()
    print(f'  best val AUC {best[0]:.4f}')

    with torch.no_grad():
        s_all = np.zeros(len(y), dtype=np.float32); t_all = np.zeros(len(y), dtype=np.float32)
        for i in range(0, len(y), 1024):
            sl = slice(i, i + 1024)
            lf, lt = net(torch.tensor(patchn[sl], device=DEV), torch.tensor(vecn[sl], device=DEV))
            s_all[sl] = torch.sigmoid(lf).cpu().numpy(); t_all[sl] = lt.cpu().numpy()

    a = auc(y[te], s_all[te]); w, n = within_group_pairwise(group[te], y[te], s_all[te])
    results['learned:risk_head'] = dict(auc=a, pairwise=w, pairs=n)
    print(f'\n--- LEARNED (test split) ---\n  {"learned:risk_head":26s} AUC {a:.4f}  within-group {w:.4f} ({n} pairs)')

    # ---------------- route selection: the planner metric ----------------
    print('\n--- ROUTE SELECTION on held-out test groups (recorded Chrono outcomes) ---')
    rng = np.random.default_rng(SEED)
    gte, yte, tte = group[te], y[te], tgoal[te]
    sel = {}
    r = [selection(gte, yte, None, tte, rng, 'random')[0] for _ in range(200)]
    sel['random_route'] = (float(np.mean(r)), float(np.std(r)))
    print(f'  {"random route":28s} failure {np.mean(r):.4f} +- {np.std(r):.4f}')
    for name in ['single:max_abs_grade', 'logreg:all_scalars', 'mlp:all_scalars(no image)']:
        f, mt, ng = selection(gte, yte, base[name][te], tte)
        sel[name] = (f, mt)
        print(f'  {name:28s} failure {f:.4f}   mean time {mt:.2f}s   ({ng} groups)')
    f, mt, ng = selection(gte, yte, s_all[te], tte)
    sel['learned:risk_head'] = (f, mt)
    print(f'  {"learned:risk_head":28s} failure {f:.4f}   mean time {mt:.2f}s   ({ng} groups)')
    fo, mto, _ = selection(gte, yte, None, tte, mode='oracle')
    sel['oracle'] = (fo, mto)
    print(f'  {"oracle (best possible)":28s} failure {fo:.4f}   mean time {mto:.2f}s')
    ceil = 1.0 - np.mean([ (yte[gte==g]==0).any() for g in np.unique(gte) ])
    print(f'  groups with NO feasible route (hard floor): {ceil:.4f}')

    json.dump(dict(results=results, selection={k: list(np.atleast_1d(v).astype(float))
                                               for k, v in sel.items()},
                   seed=SEED, n_test_groups=int(len(np.unique(gte))),
                   failure_rate=float(y.mean()), hard_floor=float(ceil)),
              open(f'{OUTD}/metrics_seed{SEED}.json', 'w'), indent=2)
    torch.save({'state_dict': net.state_dict(), 'mu': mu, 'sd': sd, 'pmu': pmu, 'psd': psd,
                'scal_keys': keys}, f'{OUTD}/risk_head_seed{SEED}.pt')
    np.savez_compressed(f'{OUTD}/scores_seed{SEED}.npz', score=s_all, tpred=t_all,
                        group=d['group'], split=d['split'], failure=d['failure'],
                        goal_time_s=d['goal_time_s'], ep=d['ep'])
    print(f'\nwrote {OUTD}/metrics_seed{SEED}.json')


if __name__ == '__main__':
    main()
