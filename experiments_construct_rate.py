"""RL construction optimisation across code rates 0.5-0.9 (breadth study).

Companion to experiments_construct.py (which is a deep study at R~0.6).  Here we
sweep the code rate -- low rates on 5G-NR BG2, high rates on BG1 (the repo's
documented split) -- and, at each rate, learn the edge-spreading construction
with the policy-gradient RL feature-policy and compare it against equal-budget
random search, the round-robin heuristic and the repo-default random seed.  The
operating SNR is picked automatically per rate (where a typical random
construction sits at FER ~ 0.3) so the construction spread is always visible.

    python3 experiments_construct_rate.py sweep   # run all rates -> results_construct_rate.json
    python3 experiments_construct_rate.py plot     # render figures from the json
    python3 experiments_construct_rate.py check     # validate configs (rate, H.c=0, speed)

Figures:
  exp_construct_rate_gain.svg     FER vs code rate for optimized / random-search / round-robin / default
  exp_construct_rate_ber_<R>.svg  BER vs Eb/N0 at each rate (4 constructions)
"""
from __future__ import annotations
import json
import sys
import time
import multiprocessing as mp
import numpy as np

from nr_ldpc import NRLDPCCode
import plotting
import rl_construct as R

# rate target -> (base graph, parity rows mp, candidate SNR grid for auto-pick).
# BG2 (Kb=10) for the low rate; BG1 (Kb=22) for >=0.6 (matches the repo).
# BG2 (Kb=10) for low rates, BG1 (Kb=22) for the high rates 0.7/0.8/0.9 -- matching
# the repo's heatmap/PAM4 convention.  (R0.6 stays on BG2: same fast code as the deep
# study; BG1 at 0.6 would be a 4x-slower mp=17 graph for no extra insight.)
RATE_PLAN = [
    ("R0.5", R.Config(bg=2, ils=0, Z=16, mp=12, w=2, L=30, W=6),
     [1.0, 1.5, 2.0, 2.5, 3.0]),
    ("R0.6", R.Config(bg=2, ils=0, Z=16, mp=8, w=2, L=30, W=6),
     [1.5, 2.0, 2.5, 3.0, 3.5]),
    ("R0.7", R.Config(bg=1, ils=0, Z=16, mp=11, w=2, L=30, W=6),
     [2.0, 2.5, 3.0, 3.5, 4.0]),
    ("R0.8", R.Config(bg=1, ils=0, Z=16, mp=8, w=2, L=30, W=6),
     [2.5, 3.0, 3.5, 4.0, 4.5]),
    ("R0.9", R.Config(bg=1, ils=0, Z=16, mp=4, w=2, L=30, W=6),
     [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]),
]

# budgets for a 64-core node: batch == cores so each PG step is one parallel wave
# (max evaluations per wall-clock).  18 steps x 64 = 1152 evals/method -- 3x the
# earlier Mac run, enough to train cleanly even on the big high-rate codes.
RL_STEPS, RL_BATCH, RL_FRAMES = 18, 64, 28
VAL_FRAMES, FINAL_FRAMES = 150, 1200
PROBE_FRAMES, PROBE_N = 28, 8
BER_OFFSETS = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
BER_FRAMES = [250, 350, 500, 650, 850, 1050]


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


def pick_snr(cfg, candidates, pool):
    """Operating SNR where a typical (random) construction sits near FER 0.3."""
    rng = np.random.default_rng(0)
    assigns = [R.random_assign(cfg, rng) for _ in range(PROBE_N)]
    best, best_gap = candidates[len(candidates) // 2], 1e9
    for snr in candidates:
        ms = R.eval_batch(cfg, assigns, snr, 1, PROBE_FRAMES, pool=pool)
        med = float(np.median([m["fer"] for m in ms]))
        if abs(med - 0.30) < best_gap:
            best_gap, best = abs(med - 0.30), snr
    return best


def ber_curve(cfg, assign, snrs, pool):
    xs, bers, fers = [], [], []
    for snr, nf in zip(snrs, BER_FRAMES):
        m = R.eval_assignments(cfg, [assign], snr, 2025, nf, pool=pool,
                               frame_chunks=R.n_workers())[0]
        xs.append(round(snr, 2)); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def run_one_rate(label, cfg, candidates, pool, transfer_theta=None):
    t0 = time.time()
    rate = cfg.build().rate
    _, _, E = R.edge_meta(cfg)
    snr = pick_snr(cfg, candidates, pool)
    print(f"\n=== {label}: bg{cfg.bg} mp{cfg.mp} rate={rate:.3f}  E={E}  "
          f"train@{snr} dB ===", flush=True)

    # RL feature policy (the transferable, sequential-MDP learner).  Keep the full
    # training history (per-step mean_reward = policy-gradient return, the RL "loss",
    # and best_val_fer) so the learning curve is persisted, not just the champion.
    pol = R.FeaturePolicy(cfg, lr=0.15, ent=0.02, seed=0)
    h_rl, best_rl = R.train_reinforce(cfg, pol, snr, RL_STEPS, RL_BATCH, RL_FRAMES,
                                      pool=pool, val_frames=VAL_FRAMES, val_seed=99,
                                      base_seed=1000, log_every=20)
    # RL per-edge-logit policy
    pol_e = R.PerEdgePolicy(E, cfg.w + 1, lr=0.2, ent=0.01, seed=0)
    h_e, best_e = R.train_reinforce(cfg, pol_e, snr, RL_STEPS, RL_BATCH, RL_FRAMES,
                                    pool=pool, val_frames=VAL_FRAMES, val_seed=99,
                                    base_seed=4000, log_every=20)
    # CEM (cross-entropy method)
    h_c, best_c = R.train_cem(cfg, snr, RL_STEPS, RL_BATCH, RL_FRAMES, pool=pool,
                              val_frames=VAL_FRAMES, val_seed=99, base_seed=2000, log_every=20)
    # equal-budget random search
    h_rnd, best_rnd = R.train_random(cfg, snr, RL_STEPS, RL_BATCH, RL_FRAMES, pool=pool,
                                     val_frames=VAL_FRAMES, val_seed=99, base_seed=3000,
                                     log_every=20)
    hist = {"rl": h_rl, "rl_peredge": h_e, "cem": h_c, "random": h_rnd}
    champions = {
        "rl": best_rl["assign"],
        "rl_peredge": best_e["assign"],
        "cem": best_c["assign"],
        "random_search": best_rnd["assign"],
        "round_robin": R.round_robin_assign(cfg),
        "seed0_default": R.random_assign(cfg, np.random.default_rng(0)),
    }
    if transfer_theta is not None:                       # zero-shot R~0.6 policy
        tp = R.FeaturePolicy(cfg); tp.theta = np.asarray(transfer_theta, float).copy()
        champions["rl_transfer"] = tp.greedy_assign()

    # high-precision final FER + mechanism + BER curve
    finals, stats, curves = {}, {}, {}
    snrs = [snr + d for d in BER_OFFSETS]
    for name, a in champions.items():
        a = np.asarray(a, dtype=np.int64)
        m = R.eval_assignments(cfg, [a], snr, 12345, FINAL_FRAMES, pool=pool,
                               frame_chunks=R.n_workers())[0]
        finals[name] = {"fer": m["fer"], "ber": m["ber"]}
        st = R.construction_stats(cfg, a)
        stats[name] = {"n4": st["n4"], "comp_load": st["comp_load"]}
        if name not in ("rl_transfer", "rl_peredge"):   # finals-only bonus columns
            curves[name] = ber_curve(cfg, a, snrs, pool)
        print(f"  {name:14s} FER={m['fer']:.4f} BER={m['ber']:.3e} n4={st['n4']}", flush=True)
    return {"rate": rate, "bg": cfg.bg, "mp": cfg.mp, "E": int(E), "train_snr": snr,
            "finals": finals, "stats": stats, "curves": curves, "hist": hist,
            "theta_rl": pol.theta.tolist(),
            "champions": {k: _arr(v) for k, v in champions.items()}}


def run_sweep():
    t0 = time.time()
    # reuse the deep-study R~0.6 policy for a zero-shot transfer column, if present
    transfer_theta = None
    try:
        transfer_theta = json.load(open("results_construct_search.json"))["theta_feature"]
        print("(loaded R~0.6 feature policy for a zero-shot transfer column)")
    except Exception:
        pass
    out = {"plan": [(lbl, R.asdict(cfg)) for lbl, cfg, _ in RATE_PLAN], "rates": {}}
    with mp.Pool(R.n_workers()) as pool:
        for label, cfg, cand in RATE_PLAN:
            try:
                out["rates"][label] = run_one_rate(label, cfg, cand, pool, transfer_theta)
            except Exception as e:                       # don't lose other rates
                print(f"  !! {label} failed: {e}", flush=True)
            json.dump(out, open("results_construct_rate.json", "w"), indent=1)
    make_plots(out)
    print(f"\nsaved results_construct_rate.json + figures ({time.time()-t0:.0f}s)")


LAB = {"rl": "RL feature [ours]", "rl_peredge": "RL per-edge [ours]",
       "cem": "CEM", "rl_transfer": "RL zero-shot (R~0.6 policy)",
       "random_search": "random search", "round_robin": "round-robin",
       "seed0_default": "random default (seed0)"}
COL = {"rl": "#d62728", "rl_peredge": "#ff7f0e", "cem": "#9467bd",
       "rl_transfer": "#e377c2", "random_search": "#1f77b4",
       "round_robin": "#8c564b", "seed0_default": "#7f7f7f"}
LAB["random"] = "random search"
COL["random"] = "#1f77b4"
SUMMARY_METHODS = ["rl", "cem", "random_search", "round_robin", "seed0_default"]


def _thr(curve, lvl=0.1, key="fer"):
    """Eb/N0 where `key` crosses `lvl` (log-interp); None if it never does."""
    x = np.asarray(curve["x"]); y = np.asarray(curve[key])
    for i in range(1, len(x)):
        if y[i - 1] > lvl >= y[i] and y[i] > 0:
            t = (np.log10(lvl) - np.log10(y[i - 1])) / (np.log10(y[i]) - np.log10(y[i - 1]))
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return None


def make_plots(out=None):
    if out is None:
        out = json.load(open("results_construct_rate.json"))
    rates_d = out["rates"]
    labels = [l for l, _, _ in RATE_PLAN if l in rates_d]
    xrate = [rates_d[l]["rate"] for l in labels]
    # primary summary: waterfall threshold (Eb/N0 @ FER=0.1) vs code rate -- robust
    # across rates (single-point FER mixes different operating SNRs).
    thr_series = []
    for name in SUMMARY_METHODS:
        thr_series.append({"x": xrate,
                           "y": [_thr(rates_d[l]["curves"][name], 0.1) or float("nan")
                                 for l in labels],
                           "label": LAB[name], "color": COL[name]})
    plotting.linear(thr_series, xlabel="code rate R",
                    ylabel="waterfall threshold Eb/N0 @ FER=0.1 [dB]",
                    title="RL-optimized SC-LDPC construction across rates (lower=better)",
                    path="exp_construct_rate_threshold.svg")
    # secondary: FER @ each rate's operating SNR
    series = []
    for name in SUMMARY_METHODS:
        y = [max(rates_d[l]["finals"][name]["fer"], 5e-4) for l in labels]
        series.append({"x": xrate, "y": y, "label": LAB[name], "color": COL[name]})
    plotting.semilogy(series, xlabel="code rate R", ylabel="FER @ operating SNR",
                      title="RL-optimized SC-LDPC construction across rates 0.5-0.9",
                      path="exp_construct_rate_gain.svg")
    # per-rate BER curves
    for l in labels:
        cur = rates_d[l]["curves"]
        ser = [{"x": cur[n]["x"], "y": cur[n]["y"], "label": LAB[n], "color": COL[n]}
               for n in SUMMARY_METHODS if n in cur]
        r = rates_d[l]["rate"]
        plotting.semilogy(ser, xlabel="Eb/N0 [dB]", ylabel="info BER",
                          title=f"Construction at R={r:.2f} (bg{rates_d[l]['bg']}, windowed)",
                          path=f"exp_construct_rate_ber_{l[1:].replace('.','')}.svg")
    # per-rate learning curves (RL "loss": training trajectory): best validation FER
    # vs #evaluations for the four optimizers (only if training history was saved)
    floor = 0.5 / VAL_FRAMES
    for l in labels:
        h = rates_d[l].get("hist")
        if not h:
            continue
        ser = [{"x": h[m]["evals"], "y": [max(f, floor) for f in h[m]["best_val_fer"]],
                "label": LAB.get(m, m), "color": COL.get(m, "#333")}
               for m in ["rl", "rl_peredge", "cem", "random"] if m in h]
        plotting.semilogy(ser, xlabel="# construction evaluations", ylabel="best validation FER",
                          title=f"Learning curves at R={rates_d[l]['rate']:.2f} (sample efficiency)",
                          path=f"exp_construct_rate_learn_{l[1:].replace('.','')}.svg")
    print("wrote exp_construct_rate_{gain,threshold,ber_*,learn_*}.svg")


def run_check():
    """Validate each rate config: rate, encoder H.c=0, and decode speed."""
    import channel as ch
    for label, cfg, cand in RATE_PLAN:
        sc = cfg.build(assign=R.random_assign(cfg, np.random.default_rng(0)))
        cw, info = sc.encode(np.random.default_rng(1))
        tan = sc.full_tanner()
        syn = np.bincount(tan.e_chk, weights=cw[tan.e_var].astype(float),
                          minlength=tan.num_chk).astype(int) & 1
        _, _, E = R.edge_meta(cfg)
        t = time.time()
        sc.decode_windowed(sc.make_llr(cw, ch.ebn0_to_sigma(cand[len(cand)//2], sc.rate),
                                       np.random.default_rng(2)), W=cfg.W, max_iter=cfg.max_iter)
        dt = (time.time() - t) * 1000
        print(f"  {label}: bg{cfg.bg} mp{cfg.mp} rate={sc.rate:.3f} E={E} "
              f"num_var={sc.num_var} | H.c syn={int(syn.sum())} | decode {dt:.0f} ms/frame")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    {"sweep": run_sweep, "plot": make_plots, "check": run_check}[cmd]()
