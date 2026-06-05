"""Self-checks for the RL-controlled windowed decoder.  Run: python tests_rl.py

These guard the two fidelity properties the experiments rely on:
  * WindowBP reproduces Tanner.decode bit-for-bit (so baseline and RL share an
    identical BP engine), and
  * the fixed-iteration controller inside the env reproduces sc.decode_windowed
    (so the env's "baseline" really is the classic window decoder).
Plus basic MDP sanity and a tiny learning check.
"""
import numpy as np

from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
import channel as ch
from rl_decoder import (WindowBP, SCWindowEnv, TabularQAgent, fixed_controller,
                        default_bins, train)


def _sc():
    return SCLDPCCode(NRLDPCCode(2, 0, 8, mp=8), w=2, L=30, seed=0)


def test_windowbp_fidelity():
    sc = _sc(); W = 6; sc._window_layout(W)
    rng = np.random.default_rng(3)
    for tpos in (0, 7, 15, 29):
        tan = sc._win_cache[W][tpos][0]
        llr = rng.standard_normal(tan.num_var) * 0.5     # low SNR -> no early stop
        for N, al in [(5, 0.7), (8, 0.8), (12, 0.9)]:
            ref = tan.decode(llr, max_iter=N, alpha=al)
            eng = WindowBP(tan); eng.reset(llr)
            for _ in range(N):
                eng.step(al)
            assert np.array_equal(eng.hard, ref), f"WindowBP != Tanner.decode (t={tpos},N={N})"
    print("PASS  WindowBP reproduces Tanner.decode bit-for-bit")


def test_fixed_controller_matches_decode_windowed():
    sc = _sc(); W = 6
    env = SCWindowEnv(sc, W=W, alpha_set=(0.8,), ebn0_range=(3.0, 3.0), max_iter_cap=30)
    for trial in range(4):
        r1 = np.random.default_rng(50 + trial); r2 = np.random.default_rng(50 + trial)
        m = env.rollout(fixed_controller(max_iter=20, alpha=0.8), r1, ebn0=3.0)
        sigma = ch.ebn0_to_sigma(3.0, sc.rate)
        cw, info = sc.encode(r2); llr = sc.make_llr(cw, sigma, r2)
        hard = sc.decode_windowed(llr, W=W, max_iter=20, alpha=0.8)
        ref = int((sc.extract_info(hard) != info).sum())
        assert m["bit_err"] == ref, f"env fixed controller != decode_windowed ({m['bit_err']} vs {ref})"
    print("PASS  env fixed controller reproduces sc.decode_windowed")


def test_env_mdp_sanity():
    sc = _sc()
    env = SCWindowEnv(sc, W=6, alpha_set=(0.7, 0.8, 0.9), ebn0_range=(3.0, 3.0))
    rng = np.random.default_rng(0)
    s = env.reset(rng, ebn0=3.0)
    assert s.shape == (env.n_state_features,)
    commits = 0
    done = False
    steps = 0
    while not done and steps < 100000:
        a = env.n_actions - 1 if env.eng.unsat else 0   # iterate at top alpha else commit
        s, r, done, info = env.step(a)
        commits += int(info["commit"])
        steps += 1
    assert commits == sc.L, f"episode should commit L={sc.L} positions, got {commits}"
    assert env.dec_bits.shape[0] == sc.num_var
    print(f"PASS  MDP episode commits all L={sc.L} positions, state dim={env.n_state_features}")


def test_qagent_learns():
    sc = _sc()
    env = SCWindowEnv(sc, W=6, alpha_set=(0.7, 0.8, 0.9), ebn0_range=(2.5, 3.5),
                      max_iter_cap=25, err_cost=600.0)
    agent = TabularQAgent(env.n_actions, default_bins(), seed=1)
    # baseline: mean return of a random policy
    rng = np.random.default_rng(9)
    def ret_of(greedy):
        rs = []
        for _ in range(30):
            s = env.reset(rng); done = False; R = 0.0
            while not done:
                a = agent.act(s, greedy=greedy)
                s, r, done, _ = env.step(a); R += r
            rs.append(R)
        return float(np.mean(rs))
    agent.eps = 1.0
    before = ret_of(greedy=True)        # untrained greedy == arbitrary
    train(env, agent, n_episodes=2500, eps_decay_frac=0.7, seed=2, log_every=10000)
    after = ret_of(greedy=True)
    assert after > before + 5, f"agent did not improve (before={before:.1f}, after={after:.1f})"
    print(f"PASS  Q-learning improves mean return ({before:.1f} -> {after:.1f})")


if __name__ == "__main__":
    test_windowbp_fidelity()
    test_fixed_controller_matches_decode_windowed()
    test_env_mdp_sanity()
    test_qagent_learns()
    print("\nAll RL tests passed.")
