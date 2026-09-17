"""Multi-head route model for the demo: P(any failure), P(stuck), and time-to-goal.

'Stuck' is prolonged_blockage_terminated, the dominant real failure mode on this
arena. There are no obstacles in f104 (assets is empty in every case), so contact
is not a modelled mode.
"""
import json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909'
OUTD = ROOT + '/multihead_v1'; os.makedirs(OUTD, exist_ok=True)
SEED = int(os.environ.get('SEED', '0'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHead(nn.Module):
    def __init__(self, n_vec):
        super().__init__()
        ch = [4, 32, 64, 128, 128]; b = []
        for i in range(4):
            b += [nn.Conv2d(ch[i], ch[i+1], 3, stride=2 if i else (2, 1), padding=1),
                  nn.BatchNorm2d(ch[i+1]), nn.GELU()]
        self.cnn = nn.Sequential(*b, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(n_vec, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(128, 64), nn.GELU())
        self.fail = nn.Linear(64, 1); self.stuck = nn.Linear(64, 1); self.time = nn.Linear(64, 1)

    def forward(self, p, v):
        z = self.fuse(torch.cat([self.cnn(p), self.mlp(v)], 1))
        return self.fail(z).squeeze(1), self.stuck(z).squeeze(1), self.time(z).squeeze(1)


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.min() == y.max(): return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1*(n1+1)/2) / (n0*n1))


def build_vec(d):
    scal = d['scal'].astype(np.float32); st = d['state0'].astype(np.float32)
    rel = d['goal_xy'].astype(np.float32) - d['start_xy'].astype(np.float32)
    extra = np.stack([rel[:, 0], rel[:, 1], np.linalg.norm(rel, axis=1),
                      d['start_yaw'].astype(np.float32)], 1)
    return np.concatenate([scal, st, extra], 1)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = np.load(ROOT + '/planner_dataset_v1.npz', allow_pickle=True)
    split = d['split'].astype(str); status = d['status'].astype(str)
    y_fail = (status != 'goal_reached').astype(np.float32)
    y_stuck = (status == 'prolonged_blockage_terminated').astype(np.float32)
    t = d['goal_time_s'].astype(np.float32)
    patch = d['patch'].astype(np.float32); vec = build_vec(d)
    tr, va, te = split == 'train', split == 'val', split == 'test'
    mu, sd = vec[tr].mean(0), vec[tr].std(0) + 1e-6
    pmu = patch[tr].mean((0, 2, 3), keepdims=True); psd = patch[tr].std((0, 2, 3), keepdims=True) + 1e-6
    vn = (vec - mu) / sd; pn = (patch - pmu) / psd
    tmu, tsd = float(t[tr & (y_fail == 0)].mean()), float(t[tr & (y_fail == 0)].std() + 1e-6)

    net = MultiHead(vec.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    ntr = int(tr.sum()); bs = 256; ep_n = int(os.environ.get('EPOCHS', '60'))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ep_n*max(1, ntr//bs))
    T = lambda a, m: torch.tensor(a[m], device=DEV)
    P, V = T(pn, tr), T(vn, tr)
    Yf, Ys, Tt = T(y_fail, tr), T(y_stuck, tr), T((t - tmu)/tsd, tr)
    ok_tr = T((y_fail == 0).astype(np.float32), tr)
    wf = torch.tensor(float((y_fail[tr] == 0).sum()/max(y_fail[tr].sum(), 1)), device=DEV)
    ws = torch.tensor(float((y_stuck[tr] == 0).sum()/max(y_stuck[tr].sum(), 1)), device=DEV)
    best = (-1, None)
    for e in range(ep_n):
        net.train(); perm = torch.randperm(ntr, device=DEV)
        for i in range(0, ntr-bs+1, bs):
            k = perm[i:i+bs]
            lf, ls, lt = net(P[k], V[k])
            loss = (F.binary_cross_entropy_with_logits(lf, Yf[k], pos_weight=wf)
                    + F.binary_cross_entropy_with_logits(ls, Ys[k], pos_weight=ws))
            m = ok_tr[k] > 0
            if m.any(): loss = loss + 0.3*F.huber_loss(lt[m], Tt[k][m])
            opt.zero_grad(); loss.backward(); opt.step()
            if sch.last_epoch < sch.total_steps-1: sch.step()
        net.eval()
        with torch.no_grad():
            lf, ls, _ = net(T(pn, va), T(vn, va))
            a = 0.5*(auc(y_fail[va], torch.sigmoid(lf).cpu().numpy())
                     + auc(y_stuck[va], torch.sigmoid(ls).cpu().numpy()))
        if a > best[0]: best = (a, {k: v.detach().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        lf, ls, lt = net(T(pn, te), T(vn, te))
        pf = torch.sigmoid(lf).cpu().numpy(); ps = torch.sigmoid(ls).cpu().numpy()
        pt = lt.cpu().numpy()*tsd + tmu
    print(f'TEST  AUC(any failure) {auc(y_fail[te], pf):.4f}   AUC(stuck) {auc(y_stuck[te], ps):.4f}')
    m = y_fail[te] == 0
    print(f'      time-to-goal on successes: r={np.corrcoef(pt[m], t[te][m])[0,1]:.3f}  '
          f'MAE {np.abs(pt[m]-t[te][m]).mean():.2f}s')
    torch.save({'state_dict': net.state_dict(), 'mu': mu, 'sd': sd, 'pmu': pmu, 'psd': psd,
                'tmu': tmu, 'tsd': tsd, 'n_vec': vec.shape[1]}, f'{OUTD}/multihead_seed{SEED}.pt')
    json.dump(dict(auc_fail=auc(y_fail[te], pf), auc_stuck=auc(y_stuck[te], ps),
                   time_r=float(np.corrcoef(pt[m], t[te][m])[0,1]),
                   time_mae=float(np.abs(pt[m]-t[te][m]).mean())),
              open(f'{OUTD}/metrics_seed{SEED}.json', 'w'), indent=2)
    print('saved', f'{OUTD}/multihead_seed{SEED}.pt')


if __name__ == '__main__':
    main()
