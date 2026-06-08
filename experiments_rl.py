"""Experiments for RL-controlled windowed SC-LDPC decoding.

Three decoders share one BP engine and differ only in the commit policy:
  (1) standard fixed window decoder  (commit on whole-window syndrome / max_iter),
  (2) fixed-threshold early commit    (commit when min|LLR_target| > T)  -- this work,
  (3) RL-controlled                   (state-adaptive threshold, tabular Q-learning),
plus an ORACLE (commit when the target equals the truth) as an upper bound.

Headline figures (dependency-free SVG via plotting.py):
  * exp_rl_pareto.svg : at a fixed Eb/N0, info BER vs average BP iterations/frame.
                        (1) traces a curve; (2) sits to its lower-left; (3) should
                        match/beat (2) and approach the oracle point.
  * exp_rl_ber.svg    : info BER vs Eb/N0 for (1) and (3) at matched complexity.

Run:
    python experiments_rl.py all       # train + evaluate + plot  (~15 min)
    python experiments_rl.py plot      # re-render figures from results_rl.json
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np

import outpaths as OP
from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
from rl_decoder import (SCWindowEnv, TabularQAgent, fixed_controller,
                        threshold_controller, oracle_controller,
                        default_bins, train, evaluate)
import plotting

# ---- code / decoder configuration (a small but real 5G NR SC-LDPC) ---------- #
BG, ILS, Z, MP = 2, 0, 8, 8
L, Wc, WIN = 30, 2, 6
MAX_ITER_CAP, MIN_ITER = 25, 2
TRAIN_SNR = (2.5, 4.0)            # episodes sample Eb/N0 uniformly across the operating band
PARETO_SNR = 3.0                  # fixed SNR for the BER-vs-complexity figure
BER_SNRS = [2.5, 3.0, 3.5, 4.0]   # SNR sweep for the BER-vs-SNR figure
N_EPISODES = 8000
FIXED_MAXITERS = [6, 8, 10, 12, 15, 20]
THRESHOLDS = [5, 7, 9, 11, 13]
RL_ERR_COSTS = [2500, 4000, 6000]
N_PARETO = 400                    # frames per Pareto point
N_BER = 500                       # frames per BER-vs-SNR point


def build_sc():
    return SCLDPCCode(NRLDPCCode(BG, ILS, Z, mp=MP), w=Wc, L=L, seed=0)


def make_env(sc, err_cost):
    # alpha fixed at 0.8 (2-action: COMMIT / CONTINUE) -- the agent learns *when*
    # to commit; alpha=0.8 is near-optimal so an alpha action is just noise.
    return SCWindowEnv(sc, W=WIN, alpha_set=(0.8,), ebn0_range=TRAIN_SNR,
                       max_iter_cap=MAX_ITER_CAP, min_iter=MIN_ITER, iter_cost=1.0,
                       err_cost=err_cost, shape_cost=10.0)


def train_policy(sc, err_cost, n_episodes=N_EPISODES, seed=3):
    env = make_env(sc, err_cost)
    agent = TabularQAgent(env.n_actions, default_bins(), lr=0.2, gamma=0.97,
                          eps_end=0.03, seed=2)
    print(f"\n[train] err_cost={err_cost}  ({n_episodes} episodes)", flush=True)
    train(env, agent, n_episodes=n_episodes, eps_decay_frac=0.7, seed=seed,
          log_every=max(1, n_episodes // 3))
    return env, agent


def _pt(env, ctrl, snr, n):
    r = evaluate(env, ctrl, [snr], n_frames=n, target_ferr=250, min_frames=n // 2)
    return r["iters"][0], r["ber"][0], r["fer"][0]


def run_all():
    t0 = time.time()
    sc = build_sc()
    print(f"SC-LDPC: BG{BG} ils{ILS} Z{Z} mp{MP}  L={L} w={Wc} W={WIN}  rate={sc.rate:.3f}")
    benv = make_env(sc, err_cost=1000)   # baselines only use the shared BP engine

    # -------- Pareto at fixed SNR ------------------------------------------- #
    print(f"\n=== Pareto @ {PARETO_SNR} dB ===")
    base = {"iters": [], "ber": [], "tag": []}
    for mi in FIXED_MAXITERS:
        it, ber, _ = _pt(benv, fixed_controller(mi, 0.8), PARETO_SNR, N_PARETO)
        base["iters"].append(it); base["ber"].append(ber); base["tag"].append(f"mi={mi}")
        print(f"  fixed   mi={mi:2d}: iters={it:6.1f}  BER={ber:.3e}", flush=True)

    thr = {"iters": [], "ber": [], "tag": []}
    for T in THRESHOLDS:
        it, ber, _ = _pt(benv, threshold_controller(T, 0.8), PARETO_SNR, N_PARETO)
        thr["iters"].append(it); thr["ber"].append(ber); thr["tag"].append(f"T={T}")
        print(f"  thresh  T={T:2d}: iters={it:6.1f}  BER={ber:.3e}", flush=True)

    o_it, o_ber, _ = _pt(benv, oracle_controller(0.8), PARETO_SNR, N_PARETO)
    print(f"  ORACLE     : iters={o_it:6.1f}  BER={o_ber:.3e}", flush=True)

    rl = {"iters": [], "ber": [], "tag": []}
    policies = {}
    for ec in RL_ERR_COSTS:
        env, agent = train_policy(sc, ec)
        policies[ec] = (env, agent)
        it, ber, _ = _pt(env, agent.greedy_controller(), PARETO_SNR, N_PARETO)
        rl["iters"].append(it); rl["ber"].append(ber); rl["tag"].append(f"ec={ec}")
        print(f"  RL  ec={ec:4d}: iters={it:6.1f}  BER={ber:.3e}", flush=True)

    # -------- BER vs SNR at matched complexity ------------------------------ #
    # RL is matched in average complexity to the RL policy itself; the two heuristic
    # baselines are tuned at PARETO_SNR and then held fixed across the SNR sweep, so
    # the figure shows the single RL policy adapting where a fixed rule cannot.
    ref_iters = float(np.median(rl["iters"]))
    best_ec = RL_ERR_COSTS[int(np.argmin([abs(it - ref_iters) for it in rl["iters"]]))]
    rl_ref_iters = rl["iters"][RL_ERR_COSTS.index(best_ec)]
    ref_mi = FIXED_MAXITERS[int(np.argmin([abs(it - rl_ref_iters) for it in base["iters"]]))]
    ref_T = THRESHOLDS[int(np.argmin([abs(it - rl_ref_iters) for it in thr["iters"]]))]
    env_b, agent_b = policies[best_ec]
    print(f"\n=== BER vs SNR  (RL ec={best_ec} ~{rl_ref_iters:.0f} it  vs standard mi={ref_mi}, "
          f"threshold T={ref_T}, complexity-matched at {PARETO_SNR} dB) ===")
    base_curve = evaluate(benv, fixed_controller(ref_mi, 0.8), BER_SNRS,
                          n_frames=N_BER, target_ferr=150, min_frames=N_BER // 3)
    thr_curve = evaluate(benv, threshold_controller(ref_T, 0.8), BER_SNRS,
                         n_frames=N_BER, target_ferr=150, min_frames=N_BER // 3)
    rl_curve = evaluate(env_b, agent_b.greedy_controller(), BER_SNRS,
                        n_frames=N_BER, target_ferr=150, min_frames=N_BER // 3)
    for i, snr in enumerate(BER_SNRS):
        print(f"  {snr:+.1f}dB  standard: {base_curve['ber'][i]:.3e}({base_curve['iters'][i]:.0f}it)  "
              f"thr-T{ref_T}: {thr_curve['ber'][i]:.3e}({thr_curve['iters'][i]:.0f}it)  "
              f"RL: {rl_curve['ber'][i]:.3e}({rl_curve['iters'][i]:.0f}it)", flush=True)

    res = {"config": {"bg": BG, "ils": ILS, "Z": Z, "mp": MP, "L": L, "w": Wc, "W": WIN,
                      "rate": sc.rate, "pareto_snr": PARETO_SNR},
           "base": base, "thr": thr, "oracle": {"iters": o_it, "ber": o_ber}, "rl": rl,
           "ber_snrs": BER_SNRS, "ref_mi": ref_mi, "ref_T": ref_T, "best_ec": best_ec,
           "base_curve": base_curve, "thr_curve": thr_curve, "rl_curve": rl_curve}
    json.dump(res, open(OP.route("results_rl.json"), "w"), indent=1)
    make_plots(res)
    print(f"\nDone in {time.time()-t0:.0f}s.  Wrote results_rl.json, exp_rl_pareto.svg, exp_rl_ber.svg")


def make_plots(res):
    o = res["oracle"]
    plotting.semilogy(
        [{"x": res["base"]["iters"], "y": res["base"]["ber"],
          "label": "(1) standard window BP", "color": "#1f77b4"},
         {"x": res["thr"]["iters"], "y": res["thr"]["ber"],
          "label": "(2) min-LLR threshold [ours]", "color": "#ff7f0e"},
         {"x": res["rl"]["iters"], "y": res["rl"]["ber"],
          "label": "(3) RL-controlled [ours]", "color": "#d62728"},
         {"x": [o["iters"]], "y": [o["ber"]],
          "label": "oracle (upper bound)", "color": "#2ca02c"}],
        xlabel="average BP iterations / frame", ylabel="info BER",
        title=f"SC-LDPC windowed decoding: complexity-BER tradeoff @ {res['config']['pareto_snr']} dB",
        path="exp_rl_pareto.svg")
    plotting.semilogy(
        [{"x": res["base_curve"]["x"], "y": res["base_curve"]["ber"],
          "label": f"(1) standard window BP (mi={res['ref_mi']})", "color": "#1f77b4"},
         {"x": res["thr_curve"]["x"], "y": res["thr_curve"]["ber"],
          "label": f"(2) fixed threshold T={res['ref_T']}", "color": "#ff7f0e"},
         {"x": res["rl_curve"]["x"], "y": res["rl_curve"]["ber"],
          "label": f"(3) RL-controlled (ec={res['best_ec']})", "color": "#d62728"}],
        xlabel="Eb/N0 [dB]", ylabel="info BER",
        title="SC-LDPC: RL vs fixed heuristics at matched complexity",
        path="exp_rl_ber.svg")


def run_plot():
    make_plots(json.load(open(OP.route("results_rl.json"))))
    print("re-rendered exp_rl_pareto.svg and exp_rl_ber.svg")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"all": run_all, "plot": run_plot}[cmd]()
