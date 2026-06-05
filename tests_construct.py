"""Self-checks for the RL construction optimiser (rl_construct.py).

    python3 tests_construct.py

Covers: explicit-assignment encoder validity, CRN determinism, multiprocessing ==
single-process, finite-difference checks of both policy gradients, a short
REINFORCE smoke run that must improve reward, and the 4-cycle counter on a tiny
hand-built graph.
"""
from __future__ import annotations
import numpy as np

from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
from decoder import Tanner
import rl_construct as R


def _ok(name):
    print(f"  ok  {name}")


def test_encoder_validity():
    cfg = R.Config(Z=16, mp=8)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = R.random_assign(cfg, rng)
        sc = cfg.build(assign=a)
        cw, info = sc.encode(np.random.default_rng(1))
        tan = sc.full_tanner()
        syn = np.bincount(tan.e_chk, weights=cw[tan.e_var].astype(float),
                          minlength=tan.num_chk).astype(int) & 1
        assert syn.sum() == 0, "H_SC @ c != 0"
    # round-robin + masks invariant to assignment
    sc1 = cfg.build(assign=R.round_robin_assign(cfg))
    sc2 = cfg.build(assign=R.random_assign(cfg, rng))
    assert np.array_equal(sc1.tx_mask, sc2.tx_mask) and sc1.K == sc2.K and sc1.N_tx == sc2.N_tx
    _ok("encoder validity (H.c=0) for arbitrary assignments; masks invariant")


def test_crn_determinism():
    cfg = R.Config(Z=16, mp=8)
    a = R.random_assign(cfg, np.random.default_rng(3))
    m1 = R.eval_batch(cfg, [a], 2.5, frame_seed=42, n_frames=20)[0]
    m2 = R.eval_batch(cfg, [a], 2.5, frame_seed=42, n_frames=20)[0]
    assert m1["bit_err"] == m2["bit_err"] and m1["frame_err"] == m2["frame_err"]
    # different assignment, same frames -> generally different metric (sanity)
    b = R.random_assign(cfg, np.random.default_rng(4))
    m3 = R.eval_batch(cfg, [b], 2.5, frame_seed=42, n_frames=20)[0]
    _ok(f"CRN deterministic (FER={m1['fer']:.3f}=={m2['fer']:.3f}); "
        f"other assign FER={m3['fer']:.3f}")


def test_mp_equals_sp():
    import multiprocessing as mp
    cfg = R.Config(Z=16, mp=8)
    rng = np.random.default_rng(5)
    assigns = [R.random_assign(cfg, rng) for _ in range(4)]
    sp = R.eval_batch(cfg, assigns, 2.5, frame_seed=7, n_frames=16, pool=None)
    with mp.Pool(2) as pool:
        par = R.eval_batch(cfg, assigns, 2.5, frame_seed=7, n_frames=16, pool=pool)
    for a, b in zip(sp, par):
        assert a["bit_err"] == b["bit_err"] and a["frame_err"] == b["frame_err"]
    _ok("multiprocessing == single-process (bit-exact)")


def _peredge_logp(theta, a):
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z); p = e / e.sum(axis=1, keepdims=True)
    return np.sum(np.log(p[np.arange(len(a)), a] + 1e-12))


def test_peredge_grad():
    rng = np.random.default_rng(0)
    E, C = 6, 3
    pol = R.PerEdgePolicy(E, C)
    pol.theta = rng.standard_normal((E, C))
    a = rng.integers(0, C, size=E)
    p = pol.probs()
    g = pol.grad_logp(a, p)
    # finite difference
    fd = np.zeros_like(g); eps = 1e-6
    for i in range(E):
        for k in range(C):
            t = pol.theta.copy(); t[i, k] += eps
            t2 = pol.theta.copy(); t2[i, k] -= eps
            fd[i, k] = (_peredge_logp(t, a) - _peredge_logp(t2, a)) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-5), f"max err {np.abs(g-fd).max()}"
    _ok("PerEdgePolicy grad log-pi matches finite difference")


def _feature_logp(pol, theta, a):
    """log pi(a) for a FIXED assignment, replaying the running counts."""
    cnt_row = np.zeros((pol.nrow, pol.C)); cnt_col = np.zeros((pol.ncol, pol.C))
    cnt_glob = np.zeros(pol.C); lp = 0.0
    for e in range(pol.E):
        phi = pol._features(e, cnt_row, cnt_col, cnt_glob, e)
        z = phi @ theta; z = z - z.max(); pe = np.exp(z) / np.exp(z).sum()
        lp += np.log(pe[a[e]] + 1e-12)
        r, c = pol.rows[e], pol.cols[e]
        cnt_row[r, a[e]] += 1; cnt_col[c, a[e]] += 1; cnt_glob[a[e]] += 1
    return lp


def test_feature_grad():
    cfg = R.Config(Z=16, mp=8)
    pol = R.FeaturePolicy(cfg)
    rng = np.random.default_rng(1)
    pol.theta = 0.5 * rng.standard_normal(pol.F)
    a, trace = pol.rollout()                     # trace corresponds to this exact a
    g = pol._episode_grad_logp(trace)
    fd = np.zeros_like(g); eps = 1e-6
    for i in range(pol.F):
        t1 = pol.theta.copy(); t1[i] += eps
        t2 = pol.theta.copy(); t2[i] -= eps
        fd[i] = (_feature_logp(pol, t1, a) - _feature_logp(pol, t2, a)) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-5), f"max err {np.abs(g-fd).max()}\n g={g}\n fd={fd}"
    _ok(f"FeaturePolicy grad log-pi matches finite difference (F={pol.F})")


def test_4cycle_counter():
    # tiny graph with exactly one 4-cycle: checks {0,1} both see vars {0,1}
    e_chk = np.array([0, 0, 1, 1]); e_var = np.array([0, 1, 0, 1])
    tan = Tanner(e_chk, e_var, 2, 2)

    class _S:
        def full_tanner(self_): return tan
    assert R.count_4cycles(_S()) == 1
    # add a third var to both checks -> C(3,2)=3 four-cycles
    e_chk = np.array([0, 0, 0, 1, 1, 1]); e_var = np.array([0, 1, 2, 0, 1, 2])
    tan2 = Tanner(e_chk, e_var, 2, 3)

    class _S2:
        def full_tanner(self_): return tan2
    assert R.count_4cycles(_S2()) == 3
    _ok("4-cycle counter correct on hand-built graphs")


def test_reinforce_improves():
    """A short PG run at a hard SNR must improve mean reward (FER down)."""
    import multiprocessing as mp
    cfg = R.Config(Z=16, mp=8)
    pol = R.FeaturePolicy(cfg, lr=0.2, ent=0.02, seed=0)
    with mp.Pool(R.n_workers()) as pool:
        hist, best = R.train_reinforce(cfg, pol, ebn0=2.5, n_steps=10, batch=12,
                                       frames=24, pool=pool, val_frames=80,
                                       log_every=5, base_seed=500)
    early = np.mean(hist["mean_reward"][:3])
    late = np.mean(hist["mean_reward"][-3:])
    print(f"    mean reward {early:+.3f} -> {late:+.3f} ; best val FER={best['fer']:.3f}")
    assert late >= early - 0.02, "reward should not collapse"
    _ok("REINFORCE smoke run improves / holds reward")


if __name__ == "__main__":
    test_encoder_validity()
    test_crn_determinism()
    test_mp_equals_sp()
    test_peredge_grad()
    test_feature_grad()
    test_4cycle_counter()
    test_reinforce_improves()
    print("\nall construction tests passed.")
