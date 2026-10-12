"""PPO pieces (src/nedm/so101_policy/ppo.py): GAE and the losses against RSL-RL 2.2.4, masked likelihood / entropy /
KL, the running normaliser, std groups (floor, cap), the split optimizer, the time-out bootstrap, masked dims that
get no gradient, checkpoint round trip, and a short learning run on a toy env; the abort group (7 actions: 3-group
configs extended), the env record and the episode counters in the checkpoint (review R1, R10).

CPU, a few seconds.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_ppo -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from nedm.so101_policy import ppo as P

torch.set_num_threads(2)


class ToyEnv:
    """Minimal vec env with the PickPlaceVecEnv interface: reward = -(tanh(a0) - target)^2 with target = obs[0];
    dim 5 is never used (mask False); episodes time out after ``horizon`` decisions."""

    def __init__(self, B=64, D=4, horizon=5, seed=0):
        self.B, self.num_envs, self.num_obs, self.num_actions = B, B, D, 6
        self.steps_per_decision, self.dev = 5, torch.device("cpu")
        self.gen = torch.Generator().manual_seed(seed)
        self.horizon = horizon
        self.t = torch.zeros(B, dtype=torch.long)
        self.obs = None

    def _new(self, n):
        return torch.rand(n, self.num_obs, generator=self.gen) * 1.6 - 0.8

    def reset(self):
        self.obs = self._new(self.B)
        self.t.zero_()
        return self.obs

    def step(self, a):
        r = -(torch.tanh(a[:, 0]) - self.obs[:, 0]) ** 2
        self.t += 1
        to = self.t >= self.horizon
        final = self.obs.clone()
        obs = self.obs.clone()
        obs[to] = self._new(int(to.sum()))
        self.t[to] = 0
        self.obs = obs
        mask = torch.ones(self.B, 6, dtype=torch.bool)
        mask[:, 5] = False
        z = torch.zeros(self.B, dtype=torch.long)
        ep = dict(end=torch.where(to, 2, 0), success=torch.zeros(self.B, dtype=torch.bool),
                  success10=torch.zeros(self.B, dtype=torch.bool), lifted=torch.zeros(self.B, dtype=torch.bool),
                  end_xy=torch.zeros(self.B, dtype=torch.float64), end_yaw=torch.zeros(self.B, dtype=torch.float64),
                  ret=r.double(), length=z + 1, steps=z + 5)
        return obs, r, to, dict(time_outs=to, final_obs=final, action_mask=mask, episode=ep, world_steps=self.B * 5)


def cfg(**over):
    c = P.train_cfg_default(1)
    c["policy"]["actor_hidden_dims"] = [32, 32]
    c["policy"]["critic_hidden_dims"] = [32, 32]
    c["num_steps_per_env"] = 8
    c["algorithm"]["num_mini_batches"] = 2
    return P.merge(c, over)


class Gae(unittest.TestCase):
    def test_against_rsl_rl(self):
        try:
            from rsl_rl.storage import RolloutStorage
        except ImportError:
            self.skipTest("rsl_rl not installed")
        T_, B = 7, 5
        g = torch.Generator().manual_seed(0)
        r, v = torch.randn(T_, B, generator=g), torch.randn(T_, B, generator=g)
        d = torch.rand(T_, B, generator=g) < 0.3
        last = torch.randn(B, generator=g)
        st = RolloutStorage(B, T_, [3], [3], [6], None, "cpu")
        st.rewards[:] = r[..., None]
        st.values[:] = v[..., None]
        st.dones[:] = d[..., None].to(st.dones.dtype)
        st.compute_returns(last[:, None], 0.99, 0.95, normalize_advantage=False)
        ret, adv = P.gae(r, v, d, last, 0.99, 0.95)
        self.assertLess(float((ret - st.returns[..., 0]).abs().max()), 1e-6)
        self.assertLess(float((adv - st.advantages[..., 0]).abs().max()), 1e-6)

    def test_hand_case(self):
        r = torch.tensor([[1.0], [2.0]])
        v = torch.tensor([[0.5], [0.25]])
        d = torch.tensor([[True], [False]])
        ret, adv = P.gae(r, v, d, torch.tensor([4.0]), 0.9, 0.8)
        a1 = 2.0 + 0.9 * 4.0 - 0.25                  # last step: bootstrap from last value
        a0 = 1.0 - 0.5                               # done at step 0: no bootstrap, no carry
        self.assertAlmostEqual(float(adv[1, 0]), a1, places=6)
        self.assertAlmostEqual(float(adv[0, 0]), a0, places=6)
        self.assertAlmostEqual(float(ret[0, 0]), a0 + 0.5, places=6)


class Likelihood(unittest.TestCase):
    def test_masked_log_prob_entropy_kl(self):
        g = torch.Generator().manual_seed(1)
        mu, a = torch.randn(9, 6, generator=g), torch.randn(9, 6, generator=g)
        std = torch.rand(6, generator=g) + 0.1
        mask = torch.rand(9, 6, generator=g) < 0.5
        mask[0] = False
        n = torch.distributions.Normal(mu, std)
        ref = (n.log_prob(a) * mask).sum(-1)
        self.assertLess(float((P.log_prob(mu, std, a, mask.float()) - ref).abs().max()), 1e-5)
        self.assertEqual(float(P.log_prob(mu, std, a, mask.float())[0]), 0.0)
        ent = (n.entropy() * mask).sum(-1)
        self.assertLess(float((P.entropy(std, mask.float()) - ent).abs().max()), 1e-5)
        self.assertLess(float(P.kl_div(mu, std, mu, std, mask.float()).abs().max()), 1e-4)
        mu2, std2 = mu + 0.1, std * 1.2
        kl = torch.distributions.kl_divergence(n, torch.distributions.Normal(mu2, std2))
        self.assertLess(float((P.kl_div(mu, std, mu2, std2, mask.float()) - (kl * mask).sum(-1)).abs().max()), 1e-3)


class Norm(unittest.TestCase):
    def test_running_stats_and_clip(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(1000, 3, generator=g) * torch.tensor([1.0, 5.0, 0.01]) + torch.tensor([2.0, -1.0, 0.3])
        rn = P.RunningNorm(3, clip=3.0)
        for c in x.split(137):
            rn.update(c)
        self.assertLess(float((rn.mean - x.double().mean(0)).abs().max()), 1e-9)
        self.assertLess(float((rn.var - x.double().var(0, unbiased=False)).abs().max()), 1e-9)
        y = rn(torch.tensor([[1e6, 0.0, 0.0]]))
        self.assertEqual(float(y[0, 0]), 3.0)
        self.assertEqual(y.dtype, torch.float32)


class PolicyAndOptim(unittest.TestCase):
    def test_init_bias_gain_std_groups(self):
        pol = P.Policy(4, 6, (4, 1, 1), (16,), (16,), init_std=(0.3, 1.0, 0.5), action_bias=[0, 0, 0, 0, -2, 0])
        out = pol.actor(torch.zeros(3, 4))
        self.assertLess(float((out[:, 4] + 2).abs().max().detach()), 0.5)
        self.assertEqual([round(x, 6) for x in pol.std.tolist()], [0.3] * 4 + [1.0, 0.5])
        self.assertEqual([round(x, 6) for x in pol.std_groups()], [0.3, 1.0, 0.5])

    def test_std_floor_cap_and_split_lr(self):
        env = ToyEnv()
        pol = P.Policy(env.num_obs, 6, (4, 1, 1), (32, 32), (32, 32), init_std=(0.3, 1.0, 0.5))
        c = cfg(algorithm=dict(entropy_coef=1.0))                       # entropy pushes every std up
        ppo = P.PPO(env, pol, c)
        for _ in range(3):
            ppo.iterate()
        cap = P.per_action(c["std_cap"], (4, 1, 1))
        floor = P.per_action(c["std_floor"], (4, 1, 1))
        self.assertTrue(bool((pol.std <= cap + 1e-7).all()) and bool((pol.std >= floor - 1e-7).all()))
        self.assertAlmostEqual(float(pol.std[0].detach()), 0.3, places=6)  # pushed to the move cap
        self.assertEqual(ppo.opt.param_groups[1]["lr"], c["critic_lr"])
        ppo.set_std([0.1, 0.1, 0.1])
        self.assertTrue(bool((pol.std >= floor - 1e-7).all()))           # set below the floor -> floor

    def test_masked_dim_gets_no_gradient(self):
        env = ToyEnv()
        pol = P.Policy(env.num_obs, 6, (4, 1, 1), (32, 32), (32, 32))
        w5 = pol.actor[-1].weight[5].detach().clone()
        w0 = pol.actor[-1].weight[0].detach().clone()
        s5 = float(pol.std[5].detach())
        ppo = P.PPO(env, pol, cfg(algorithm=dict(entropy_coef=0.0)))
        ppo.iterate()
        self.assertTrue(torch.equal(pol.actor[-1].weight[5].detach(), w5))   # dim 5 never used: untouched
        self.assertEqual(float(pol.std[5].detach()), s5)
        self.assertFalse(torch.equal(pol.actor[-1].weight[0].detach(), w0))              # a used dim learns

    def test_time_out_bootstrap(self):
        env = ToyEnv(B=8, horizon=2)
        pol = P.Policy(env.num_obs, 6, (4, 1, 1), (8,), (8,))
        ppo = P.PPO(env, pol, cfg(num_steps_per_env=4))
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in pol.critic.parameters():
                p.copy_(torch.randn(p.shape, generator=g))
            pol.norm.count.fill_(1e15)                # frozen normaliser (identity): exact values below
        env2 = ToyEnv(B=8, horizon=2)          # twin: the same observations
        ppo.collect()
        R = ppo.roll
        self.assertTrue(bool(R.dones[1].all()) and not bool(R.dones[0].any()))
        # step 1 times out: reward = toy reward + gamma V(final observation = the observation before the reset)
        o0 = env2.reset()
        o1, _, _, _ = env2.step(R.act[0])
        r1 = -(torch.tanh(R.act[1, :, 0]) - o1[:, 0]) ** 2
        with torch.no_grad():
            vf = pol.value(pol.norm(o1))
            o2, _, _, _ = env2.step(R.act[1])
            v_new = pol.value(pol.norm(o2))           # the observation after the reset must NOT be used
        self.assertLess(float((R.rewards[1] - r1 - 0.99 * vf).abs().max()), 1e-5)
        self.assertGreater(float((R.rewards[1] - r1 - 0.99 * v_new).abs().max()), 1e-3)
        self.assertLess(float((R.rewards[0] + (torch.tanh(R.act[0, :, 0]) - o0[:, 0]) ** 2).abs().max()), 1e-6)

    def test_learns_toy_and_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        env = ToyEnv(B=128)
        pol = P.Policy(env.num_obs, 6, (4, 1, 1), (32, 32), (32, 32))
        ppo = P.PPO(env, pol, cfg(num_steps_per_env=8, algorithm=dict(learning_rate=1e-3)))
        first = ppo.iterate()["reward_per_decision"]
        for _ in range(40):
            last = ppo.iterate()["reward_per_decision"]
        self.assertGreater(last, first + 0.05)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            ppo.save(path)
            pol2 = P.load_policy(path)
            o = torch.randn(5, env.num_obs)
            self.assertTrue(torch.equal(pol2.act_inference(o), pol.act_inference(o)))
            pol3 = P.Policy(env.num_obs, 6, (4, 1, 1), (32, 32), (32, 32))
            ppo3 = P.PPO(ToyEnv(B=128), pol3, cfg(num_steps_per_env=8))
            ppo3.load(path)
            self.assertEqual(ppo3.iteration, ppo.iteration)
            self.assertEqual(ppo3.world_steps, ppo.world_steps)
            self.assertAlmostEqual(ppo3.lr, ppo.lr)

    def test_abort_group_values_and_checkpoint_env_record(self):
        self.assertEqual(P.group_values([0.3, 1.0, 0.5], [4, 1, 1, 1], "init_std"), [0.3, 1.0, 0.5, 0.5])
        self.assertEqual(P.group_values([0.15, 0.3, 0.2, 0.25], [4, 1, 1, 1], "std_floor"), [0.15, 0.3, 0.2, 0.25])
        with self.assertRaises(ValueError):
            P.group_values([0.3, 1.0], [4, 1, 1, 1], "init_std")
        self.assertEqual(P.action_bias_for([0, 0, 0, 0, -1, 0], 7), [0, 0, 0, 0, -1, 0, -3.0])
        self.assertEqual(P.action_bias_for([0] * 6, 6), [0] * 6)
        env = ToyEnv(B=16)
        env.ep_count = torch.arange(16)
        env.resolved = lambda: dict(decoder=dict(carry_macro=True), task=dict(obs_rates=True))
        pol = P.Policy(env.num_obs, 6, (4, 1, 1), (8,), (8,))
        ppo = P.PPO(env, pol, cfg())
        ppo.iterate()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            ppo.save(path)
            pol2, rec = P.load_policy(path, with_env=True)
            self.assertEqual(rec["decoder"], dict(carry_macro=True))
            env2 = ToyEnv(B=16)
            env2.ep_count = torch.zeros(16, dtype=torch.long)
            ppo2 = P.PPO(env2, P.Policy(env.num_obs, 6, (4, 1, 1), (8,), (8,)), cfg())
            ppo2.load(path)
            self.assertEqual(env2.ep_count.tolist(), list(range(16)))      # macro-seed streams continue (R10)

    def test_context_std(self):
        """hold_std: rows with obs[ctx_index] > 0.5 sample and score with the second std vector; its own floor / cap;
        a promotion's set_std leaves it; the checkpoint round trip keeps it."""
        env = ToyEnv(B=64)
        pol = P.Policy(env.num_obs, 6, (4, 1, 1), (16,), (16,), init_std=(0.3, 1.0, 0.5), ctx_index=1,
                       ctx_init_std=(0.05, 1.0, 0.5))
        c = cfg(hold_std=dict(init=[0.05, 1.0, 0.5], floor=[0.02, 0.3, 0.2], cap=[0.1, 2.0, 1.0]))
        ppo = P.PPO(env, pol, c)
        o = torch.tensor([[0.0, 1.0, 0, 0], [0.0, 0.0, 0, 0]])
        st = pol.std_of(pol.ctx(o))
        self.assertAlmostEqual(float(st[0, 0]), 0.05, places=6)
        self.assertAlmostEqual(float(st[1, 0]), 0.3, places=6)
        ppo.collect()
        R = ppo.roll
        ctx = R.ctx
        self.assertTrue(bool(ctx.any()) and bool((~ctx).any()))
        self.assertTrue(torch.allclose(R.std[ctx][:, 0], torch.full((int(ctx.sum()),), 0.05)))
        n = torch.distributions.Normal(R.mu, R.std)
        ref = (n.log_prob(R.act) * R.mask).sum(-1)
        self.assertLess(float((R.logp - ref).abs().max()), 1e-4)
        ppo.update()
        self.assertTrue(bool((pol.std_ctx[:4] >= 0.02 - 1e-7).all()) and bool((pol.std_ctx[:4] <= 0.1 + 1e-7).all()))
        before = pol.std_ctx.detach().clone()
        ppo.set_std([0.3, 1.0, 0.5], at_least=True)
        self.assertTrue(torch.equal(pol.std_ctx.detach(), before))
        with tempfile.TemporaryDirectory() as d:
            ppo.save(Path(d) / "m.pt")
            pol2 = P.load_policy(Path(d) / "m.pt")
            self.assertTrue(torch.equal(pol2.std_ctx.detach(), pol.std_ctx.detach()))

    def test_curve_writer_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.csv"
            w = P.CurveWriter(p)
            w.write(dict(iteration=1, a=0.5, b="x"))
            w2 = P.CurveWriter(p)                    # resumed: same header, new keys dropped
            w2.write(dict(iteration=2, a=0.25, c=3.0))
            c = P.read_curve(p)
            self.assertEqual(c["iteration"].tolist(), [1.0, 2.0])
            self.assertEqual(c["a"].tolist(), [0.5, 0.25])
            self.assertNotIn("c", c)


if __name__ == "__main__":
    unittest.main()
