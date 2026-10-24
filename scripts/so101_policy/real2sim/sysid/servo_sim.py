"""Batched servo-model simulator (torch) on the real time base of the robot logs. Copy of
/tmp/claude_servo_fit/sim.py (the servo fit of ~/so101_r2s_claude/sim_debug/scout/tr_servo.md, 2026-10-23) for the v8
sysid check: gravity torques passed in (servo_check.gravity), no cache files, CPU by default. Model unchanged.

Per joint j, continuous time, the command u (sent, after the guard incl. the integral correction) acts piecewise
constant from its write time tw[k] + delay on. Model (exact per 2 ms substep for a constant input):
    r   = u(t - delay) + o,   o = -(kg * bias(q) + kh * cube(q) * held) + b          (equilibrium of q)
    e   = r - q
    q'  = 0                                   if |e| <= db          (dead band: no motion)
    q'  = clip((e - db sign e) / tau, +-vmax) otherwise              (first-order lag + rate limit)
bias = MuJoCo gravity torque at the measured pose (grav.py), cube = holding torque per kg at the grasp point.
Free run: q_m(0) = q_meas(0), never reset (arm). Jaw: reset to the measured q while the jaw is blocked by the cube
(those samples are excluded from the error). ``reset_query``: also reset every joint to the measured q at every
query read (decision-horizon prediction, 1-6 steps ahead).
"""
import math
import numpy as np
import torch

H = 0.002            # substep (s)
DMAX = 0.250         # largest delay (s)
ND = int(round(DMAX / H)) + 1
PARAMS = ("delay", "tau", "vmax", "db", "kg", "kh", "b", "dbs", "tau2", "db1")
EPS = 1.7e-4        # rad: a moving joint stops within this of its dead-band edge


class Data:
    def __init__(self, Es, G, blocked, dev="cpu"):
        self.names = [E["name"] for E in Es]
        self.dev = dev
        Ne = len(Es)
        S = max(int(np.ceil((E["tr"][E["ok"]][-1] - E["tr"][0]) / H)) + 2 for E in Es)
        K = max(E["n"] for E in Es)
        self.S, self.K, self.E = S, K, Ne
        U = np.zeros((Ne, K + 1, 6), np.float32)          # index 0 = the command before the first write (= q[0])
        kc = np.zeros((Ne, S, ND), np.int32)
        bias = np.zeros((Ne, S, 6), np.float32)
        cube = np.zeros((Ne, S, 6), np.float32)
        held = np.zeros((Ne, S), np.float32)
        rd = np.full((Ne, S), -1, np.int64)               # read index recorded at the end of this substep
        qm = np.zeros((Ne, K, 6), np.float32)
        mask = np.zeros((Ne, K, 6), np.float32)
        jaw_reset = np.zeros((Ne, S, 6), bool)          # reset a joint to the measured q (blocked)
        qrs = np.zeros((Ne, S, 6), np.float32)            # measured q to reset to (at the substep of a read)
        is_read = np.zeros((Ne, S), bool)
        is_query = np.zeros((Ne, S), bool)
        q0 = np.zeros((Ne, 6), np.float32)
        for e, E in enumerate(Es):
            n = int(E["ok"].sum())
            tr, tw = E["tr"][:n], E["tw"][:n]
            t0 = tr[0]
            ts = t0 + H * np.arange(S)
            U[e, 0] = E["q"][0]
            U[e, 1:n + 1] = E["u"][:n]
            U[e, n + 1:] = E["u"][n - 1]
            for di in range(ND):
                kc[e, :, di] = np.searchsorted(tw, ts - di * H, side="right")    # 0 = before the first write
            ksub = np.clip(np.floor((tr - t0) / H).astype(int), 0, S - 1)
            # substep s covers [ts[s], ts[s] + H): the read k is recorded at the START of substep ksub (state at ts)
            rd[e, ksub] = np.arange(n)
            lat = np.clip(np.searchsorted(tr, ts, side="right") - 1, 0, n - 1)   # latest read before ts
            b, c = G[E["name"] + "|bias"][:n], G[E["name"] + "|cube"][:n]
            bias[e], cube[e] = b[lat], c[lat]
            held[e] = E["held"][:n][lat]
            qm[e, :n] = E["q"][:n]
            blk = blocked(E)[:n]
            blk[:, 5] |= E["jaw_block"][:n]
            mask[e, :n] = (~blk).astype(np.float32)
            jaw_reset[e, ksub] = blk
            qrs[e, ksub] = E["q"][:n]
            is_read[e, ksub] = True
            is_query[e, ksub] = E["query"][:n]
            mask[e, 0] = 0.0
            # the end of the episode: substeps past the last read keep their state (no reads, no error)
            q0[e] = E["q"][0]
        T = lambda x: torch.as_tensor(x, device=dev)
        self.U, self.kc, self.bias, self.cube, self.held = T(U), T(kc).long(), T(bias), T(cube), T(held)
        self.rd, self.qm, self.mask, self.jaw_reset, self.qrs = T(rd), T(qm), T(mask), T(jaw_reset), T(qrs)
        self.is_read, self.is_query, self.q0 = T(is_read), T(is_query), T(q0)
        self.nread = [int(E["ok"].sum()) for E in Es]
        nq = np.zeros((Ne, K, 6), np.float32)             # 1 = not a query read (decision-horizon error samples)
        for e, E in enumerate(Es):
            n = int(E["ok"].sum())
            nq[e, :n] = (~E["query"][:n]).astype(np.float32)[:, None]
        self.mask_dec = self.mask * T(nq)


@torch.no_grad()
def simulate(D: Data, P: dict, reset_query=False, record_all=False, reset_all=False):
    """P: {name: tensor [C, 6]} -> predicted q at the reads [E, C, K, 6]."""
    dev = D.dev
    C = P["tau"].shape[0]
    E, S, K = D.E, D.S, D.K
    di = torch.round(P["delay"] / H).long().clamp(0, ND - 1)                      # [C, 6]
    tau = P["tau"].clamp_min(1e-4)
    vmax, db = P["vmax"], P["db"]
    dbs = P["dbs"].clamp_min(0.0) if "dbs" in P else torch.zeros_like(db)          # extra break-away threshold
    a2 = (1.0 - torch.exp(-H / P["tau2"].clamp_min(1e-4))) if "tau2" in P else torch.ones_like(db)
    a2 = torch.where(P["tau2"] > 0, a2, torch.ones_like(a2)) if "tau2" in P else a2
    kg, kh, b = P["kg"], P["kh"], P["b"]
    db1 = P.get("db1")
    q = D.q0[:, None, :].expand(E, C, 6).clone()
    f = None                                                                         # command pre-filter state
    moving = torch.zeros(E, C, 6, dtype=torch.bool, device=dev)
    out = torch.zeros(E, C, K, 6, device=dev)
    jidx = torch.arange(6, device=dev)
    vt = vmax * tau
    Uf = D.U                                                                         # [E, K+1, 6]
    ar = torch.arange(E, device=dev)
    for s in range(S):
        # record / reset at a read (state at the start of the substep); no host sync
        r = D.rd[:, s]
        has = (r >= 0)[:, None, None]
        rc = r.clamp(min=0)
        if reset_all:                 # one step ahead: record the prediction, then reset to the measured q
            out[ar, :, rc] = torch.where(has, q, out[ar, :, rc])
            q = torch.where(D.is_read[:, s, None, None], D.qrs[:, s, None, :], q)
        if reset_query:
            q = torch.where(D.is_query[:, s, None, None], D.qrs[:, s, None, :], q)
        q = torch.where(D.jaw_reset[:, s, None, :], D.qrs[:, s, None, :], q)
        if not reset_all:
            out[ar, :, rc] = torch.where(has, q, out[ar, :, rc])
        kk = D.kc[:, s, :][:, di]                                                    # [E, C, 6]
        u = Uf[ar[:, None, None], kk, jidx]                                          # [E, C, 6]
        f = u if f is None else f + a2 * (u - f)
        o = b - (kg * D.bias[:, s, None, :] + kh * D.cube[:, s, None, :] * D.held[:, s, None, None])
        rr = f + o
        e = rr - q
        ae = e.abs()
        sg = torch.sign(e)
        dbe = db if db1 is None else db + db1 * D.bias[:, s, None, :].abs()          # load-dependent dead band
        mv = torch.where(moving, ae > dbe + EPS, ae > dbe + dbs)
        rp = rr - dbe * sg                                                           # shifted target
        ep = (rp - q).abs()
        t1 = (ep - vt) / vmax                                                        # linear phase length
        lin_all = t1 >= H
        q_lin = q + sg * vmax * H
        tb = t1.clamp(0.0, H)
        qb = torch.where(ep > vt, rp - sg * vt, q)                                   # state after the linear phase
        q_exp = rp - (rp - qb) * torch.exp(-(H - tb) / tau)
        qn = torch.where(lin_all, q_lin, q_exp)
        q = torch.where(mv, qn, q)
        moving = mv
    return out


def rms_deg(D: Data, out, idx=None, decision=False):
    """per (candidate, joint) RMS error in deg over episodes idx (default all) -> [C, 6]. ``decision``: skip the query
    reads (the reset points of a reset_query run)."""
    idx = list(range(D.E)) if idx is None else list(idx)
    m = (D.mask_dec if decision else D.mask)[idx][:, None]                                                         # [e, 1, K, 6]
    err = (out[idx] - D.qm[idx][:, None]) * m
    se = (err ** 2).sum((0, 2))
    n = m.sum((0, 2)).clamp_min(1)
    return torch.rad2deg(torch.sqrt(se / n))
