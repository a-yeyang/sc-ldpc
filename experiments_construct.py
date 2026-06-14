"""End-to-end RL *construction* optimisation of SC-LDPC codes -- experiments.

Learns the edge-spreading assignment (which component B_0..B_w each systematic
5G-NR base edge goes to) that minimises end-to-end frame-error rate, and compares
the policy-gradient RL learner against an equal-budget random search and the
cross-entropy method.

    python3 experiments_construct.py search     # train RL/CEM/random  -> results_construct_search.json
    python3 experiments_construct.py curves     # BER-vs-SNR of the champions + baselines
    python3 experiments_construct.py analysis   # cycles / girth / balance mechanism
    python3 experiments_construct.py transfer   # zero-shot transfer of the learned policy
    python3 experiments_construct.py plot        # (re)render all SVGs from the JSONs
    python3 experiments_construct.py all         # everything, in order  (~40 min)

Figures (dependency-free SVG):
  exp_construct_learn.svg   best validation FER vs #construction-evaluations (RL vs random vs CEM)
  exp_construct_ber.svg     BER vs Eb/N0 for the RL champion vs baselines
  exp_construct_transfer.svg zero-shot transfer of the learned policy to a new code
"""
from __future__ import annotations
import json
import os
import sys
import time
import multiprocessing as mp
import numpy as np

import outpaths as OP
from nr_ldpc import NRLDPCCode
from decoder import Tanner
import channel as ch
import plotting
import rl_construct as R

# --------------------------------------------------------------------------- #
#  configuration & budgets
# --------------------------------------------------------------------------- #
CFG = R.Config(bg=2, ils=0, Z=16, mp=8, w=2, L=30, W=6, max_iter=12, alpha=0.8)
TRAIN_SNR = 2.5                      # waterfall point with a clean construction spread

N_STEPS = 32                         # policy-gradient / search steps
BATCH = 20                           # constructions evaluated per step (2 waves x 10 cores)
FRAMES = 36                          # CRN frames per construction during search
VAL_FRAMES = 200                     # per-step champion re-validation (fixed seed)
FINAL_VAL = 1500                     # high-precision final FER of each champion
VAL_SEED = 99

CURVE_SNRS = [1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4]
CURVE_FRAMES = [400, 450, 500, 600, 800, 1000, 1200]


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


# --------------------------------------------------------------------------- #
#  phase 1: search  (RL feature-policy, RL per-edge, CEM, random) -- equal budget
# --------------------------------------------------------------------------- #
def run_search():
    t0 = time.time()
    print(f"config: {CFG}  rate={CFG.build().rate:.4f}  train@{TRAIN_SNR}dB")
    _, _, E = R.edge_meta(CFG)
    budget = N_STEPS * BATCH
    print(f"search space {CFG.w+1}^{E}  | budget = {N_STEPS}x{BATCH} = {budget} evals/method")
    out = {"config": {**R.asdict(CFG), "train_snr": TRAIN_SNR, "E": int(E),
                       "n_steps": N_STEPS, "batch": BATCH, "frames": FRAMES,
                       "val_frames": VAL_FRAMES}}

    with mp.Pool(R.n_workers()) as pool:
        # --- RL: feature policy (sequential constructive MDP, generalises) ----
        print("\n[1/4] RL  feature-policy (REINFORCE)")
        pol_f = R.FeaturePolicy(CFG, lr=0.15, ent=0.02, seed=0)
        h_f, best_f = R.train_reinforce(CFG, pol_f, TRAIN_SNR, N_STEPS, BATCH, FRAMES,
                                        pool=pool, val_frames=VAL_FRAMES, val_seed=VAL_SEED,
                                        base_seed=1000, log_every=5)
        # --- RL: per-edge logits (structured bandit, no transfer) -------------
        print("\n[2/4] RL  per-edge-logit policy (REINFORCE)")
        pol_e = R.PerEdgePolicy(E, CFG.w + 1, lr=0.2, ent=0.01, seed=0)
        h_e, best_e = R.train_reinforce(CFG, pol_e, TRAIN_SNR, N_STEPS, BATCH, FRAMES,
                                        pool=pool, val_frames=VAL_FRAMES, val_seed=VAL_SEED,
                                        base_seed=4000, log_every=5)
        # --- CEM (strong non-RL distribution baseline) ------------------------
        print("\n[3/4] CEM  (cross-entropy method)")
        h_c, best_c = R.train_cem(CFG, TRAIN_SNR, N_STEPS, BATCH, FRAMES, pool=pool,
                                  val_frames=VAL_FRAMES, val_seed=VAL_SEED,
                                  base_seed=2000, log_every=5)
        # --- random search (the key equal-budget baseline) --------------------
        print("\n[4/4] random search")
        h_r, best_r = R.train_random(CFG, TRAIN_SNR, N_STEPS, BATCH, FRAMES, pool=pool,
                                     val_frames=VAL_FRAMES, val_seed=VAL_SEED,
                                     base_seed=3000, log_every=5)

        # --- reference constructions ------------------------------------------
        seed0 = R.random_assign(CFG, np.random.default_rng(0))       # repo default
        rr = R.round_robin_assign(CFG)                                # human heuristic
        champions = {"rl_feature": best_f["assign"], "rl_peredge": best_e["assign"],
                     "cem": best_c["assign"], "random_search": best_r["assign"],
                     "seed0_default": seed0, "round_robin": rr}
        # high-precision final FER of every champion (same large CRN bank)
        print(f"\n=== final FER @ {TRAIN_SNR} dB over {FINAL_VAL} frames ===")
        finals = {}
        for name, a in champions.items():
            m = R.eval_assignments(CFG, [a], TRAIN_SNR, 12345, FINAL_VAL, pool=pool,
                                   frame_chunks=R.n_workers())[0]
            finals[name] = {"fer": m["fer"], "ber": m["ber"]}
            print(f"  {name:16s}  FER={m['fer']:.4f}  BER={m['ber']:.3e}")

    out.update({
        "hist": {"rl_feature": h_f, "rl_peredge": h_e, "cem": h_c, "random": h_r},
        "best": {"rl_feature": best_f, "rl_peredge": best_e, "cem": best_c, "random_search": best_r},
        "champions": {k: _arr(v) for k, v in champions.items()},
        "finals": finals,
        "theta_feature": pol_f.theta.tolist(),
    })
    # numpy arrays in best{} -> lists
    for k in out["best"]:
        out["best"][k] = {"fer": out["best"][k]["fer"], "ber": out["best"][k]["ber"],
                          "assign": _arr(out["best"][k]["assign"])}
    json.dump(out, open(OP.route("results_construct_search.json"), "w"), indent=1)
    print(f"\nsaved results_construct_search.json  ({time.time()-t0:.0f}s)")
    return out


# --------------------------------------------------------------------------- #
#  phase 2: BER-vs-SNR curves of the champions + baselines
# --------------------------------------------------------------------------- #
def _ber_curve_assign(cfg, assign, pool):
    xs, bers, fers = [], [], []
    for snr, nf in zip(CURVE_SNRS, CURVE_FRAMES):
        m = R.eval_assignments(cfg, [assign], snr, 2025, nf, pool=pool,
                               frame_chunks=R.n_workers())[0]
        xs.append(snr); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def _ber_curve_component(cfg):
    """Uncoupled 5G-NR component code at the same rate matching (full-graph BP)."""
    code = NRLDPCCode(cfg.bg, cfg.ils, cfg.Z, mp=cfg.mp)
    tan = Tanner(*code.edges(), code.M, code.N)
    punct = np.zeros(code.N, dtype=bool); punct[:code.n_punct] = True
    tx = ~punct
    xs, bers, fers = [], [], []
    for snr, nf in zip(CURVE_SNRS, CURVE_FRAMES):
        sigma = ch.ebn0_to_sigma(snr, code.rate)
        rng = np.random.default_rng(2025)
        be = bits = fe = 0
        for _ in range(nf):
            msg = rng.integers(0, 2, code.K).astype(np.uint8)
            cw = code.encode(msg)
            y = ch.awgn(ch.bpsk(cw[tx]), sigma, rng)
            llr = np.zeros(code.N); llr[tx] = ch.llr_awgn(y, sigma)
            hard = tan.decode(llr, max_iter=30, alpha=0.8)
            err = int((hard[:code.K] != msg).sum())
            be += err; bits += code.K; fe += int(err > 0)
        xs.append(snr); bers.append(be / bits); fers.append(fe / nf)
    return {"x": xs, "y": bers, "fer": fers, "rate": code.rate}


def run_curves():
    t0 = time.time()
    res = json.load(open(OP.route("results_construct_search.json")))
    champs = res["champions"]
    curves = {}
    with mp.Pool(R.n_workers()) as pool:
        for name in ["rl_feature", "rl_peredge", "cem", "random_search",
                     "round_robin", "seed0_default"]:
            a = np.asarray(champs[name], dtype=np.int64)
            curves[name] = _ber_curve_assign(CFG, a, pool)
            print(f"  curve {name:16s} BER={['%.1e'%b for b in curves[name]['y']]}", flush=True)
    print("  component (uncoupled 5G-NR)...", flush=True)
    curves["component"] = _ber_curve_component(CFG)
    res["curves"] = curves
    json.dump(res, open(OP.route("results_construct_search.json"), "w"), indent=1)
    print(f"saved curves into results_construct_search.json ({time.time()-t0:.0f}s)")
    return res


# --------------------------------------------------------------------------- #
#  phase 3: mechanism analysis (cycles / girth / balance / degree)
# --------------------------------------------------------------------------- #
def run_analysis():
    res = json.load(open(OP.route("results_construct_search.json")))
    champs = res["champions"]
    stats = {}
    print("=== construction mechanism (4-cycles, balance, degree, girth) ===")
    for name in ["rl_feature", "rl_peredge", "cem", "random_search",
                 "round_robin", "seed0_default"]:
        a = np.asarray(champs[name], dtype=np.int64)
        s = R.construction_stats(CFG, a)
        s["girth"] = R.girth(CFG.build(assign=a), max_g=10, n_seeds=400, seed=1)
        stats[name] = s
        print(f"  {name:16s} n4={s['n4']:5d}  girth={s['girth']:2d}  "
              f"load={s['comp_load']} (std={s['comp_load_std']:.2f})  "
              f"chkdeg(mean={s['chk_deg_mean']:.1f},max={s['chk_deg_max']})")
    # also average over many *random* constructions for reference
    rng = np.random.default_rng(123)
    n4s = []
    for _ in range(40):
        a = R.random_assign(CFG, rng)
        n4s.append(R.count_4cycles(CFG.build(assign=a)))
    stats["_random_n4"] = {"mean": float(np.mean(n4s)), "min": int(np.min(n4s)),
                           "max": int(np.max(n4s)), "n": len(n4s)}
    print(f"  [random ref] n4 mean={np.mean(n4s):.0f} min={np.min(n4s)} max={np.max(n4s)}")
    res["stats"] = stats
    json.dump(res, open(OP.route("results_construct_search.json"), "w"), indent=1)
    print("saved analysis into results_construct_search.json")
    return res


# --------------------------------------------------------------------------- #
#  phase 4: zero-shot transfer of the learned feature policy to a new code
# --------------------------------------------------------------------------- #
# 5G-NR BG2 connectivity is identical across lifting sets (only shift values differ),
# so the base-graph-structure features transfer; Z=24 ships with ils=1 in data/.
TRANSFER_CFGS = {
    "Z=24 (longer blocks)": R.Config(bg=2, ils=1, Z=24, mp=8, w=2, L=30, W=6),
    "mp=6 (higher rate)":   R.Config(bg=2, ils=0, Z=16, mp=6, w=2, L=30, W=6),
}


def run_transfer():
    t0 = time.time()
    res = json.load(open(OP.route("results_construct_search.json")))
    theta = np.asarray(res["theta_feature"], dtype=np.float64)
    transfer = {}
    with mp.Pool(R.n_workers()) as pool:
        for label, tcfg in TRANSFER_CFGS.items():
            pol = R.FeaturePolicy(tcfg)
            assert pol.theta.shape == theta.shape, "policy must transfer (same w)"
            pol.theta = theta.copy()
            learned = pol.greedy_assign()                      # zero-shot construction
            rng = np.random.default_rng(7)
            # baseline: distribution over random constructions at this new config
            rand_assigns = [R.random_assign(tcfg, rng) for _ in range(12)]
            mr = R.eval_assignments(tcfg, rand_assigns, TRAIN_SNR, 8888, 250, pool=pool)
            rfers = sorted(m["fer"] for m in mr)
            ml = R.eval_assignments(tcfg, [learned], TRAIN_SNR, 8888, 250, pool=pool,
                                    frame_chunks=R.n_workers())[0]
            rr = R.round_robin_assign(tcfg)
            mrr = R.eval_assignments(tcfg, [rr], TRAIN_SNR, 8888, 250, pool=pool,
                                     frame_chunks=R.n_workers())[0]
            transfer[label] = {
                "rate": tcfg.build().rate,
                "learned_fer": ml["fer"], "learned_ber": ml["ber"],
                "round_robin_fer": mrr["fer"],
                "random_fers": rfers,
                "random_median": float(np.median(rfers)),
                "random_best": float(min(rfers)),
                "learned_assign": _arr(learned),
            }
            print(f"  transfer {label:22s}: learned FER={ml['fer']:.3f}  "
                  f"random[median={np.median(rfers):.3f},best={min(rfers):.3f}]  "
                  f"round_robin={mrr['fer']:.3f}", flush=True)
    res["transfer"] = transfer
    json.dump(res, open(OP.route("results_construct_search.json"), "w"), indent=1)
    print(f"saved transfer into results_construct_search.json ({time.time()-t0:.0f}s)")
    return res


# --------------------------------------------------------------------------- #
#  plotting
# --------------------------------------------------------------------------- #
LABELS = {"rl_feature": "RL feature-policy [ours]", "rl_peredge": "RL per-edge [ours]",
          "cem": "CEM (cross-entropy)", "random_search": "random search",
          "random": "random search", "round_robin": "round-robin heuristic",
          "seed0_default": "random (repo default seed 0)", "component": "uncoupled 5G-NR"}
COLORS = {"rl_feature": "#d62728", "rl_peredge": "#ff7f0e", "cem": "#9467bd",
          "random": "#1f77b4", "random_search": "#1f77b4", "round_robin": "#8c564b",
          "seed0_default": "#7f7f7f", "component": "#2ca02c"}


def make_plots(res=None):
    if res is None:
        res = json.load(open(OP.route("results_construct_search.json")))
    vf = res["config"]["val_frames"]
    floor = 0.5 / vf
    # ---- learning curve: best validation FER vs #evaluations -----------------
    lc = []
    for name in ["rl_feature", "rl_peredge", "cem", "random"]:
        h = res["hist"][name]
        y = [max(f, floor) for f in h["best_val_fer"]]
        lc.append({"x": h["evals"], "y": y, "label": LABELS[name], "color": COLORS[name]})
    plotting.semilogy(lc, xlabel="# construction evaluations", ylabel="best validation FER",
                      title=f"RL construction search vs baselines @ {res['config']['train_snr']} dB",
                      path="exp_construct_learn.svg")
    # ---- BER vs SNR ----------------------------------------------------------
    # Prefer the high-precision floor sweep (4000 frames/point, SNR grid to 4.0 dB)
    # if present; fall back to the 2000-frame search curves otherwise.
    order = ["rl_feature", "cem", "random_search", "round_robin", "seed0_default", "component"]
    cur = None
    try:
        floor_path = os.path.join(OP.ROOT, "results", "construct", "results_floor_snr.json")
        fl = json.load(open(floor_path))
        grid = fl["snr_grid"]
        cur = [{"x": grid,
                "y": [fl["curves"][n][f"{x}"]["ber"] for x in grid],
                "label": LABELS[n], "color": COLORS[n]}
               for n in order if n in fl["curves"]]
    except (FileNotFoundError, KeyError):
        cur = None
    if cur is None and "curves" in res:
        cur = [{"x": res["curves"][n]["x"], "y": res["curves"][n]["y"],
                "label": LABELS[n], "color": COLORS[n]}
               for n in order if n in res["curves"]]
    if cur:
        plotting.semilogy(cur, xlabel="Eb/N0 [dB]", ylabel="info BER",
                          title="RL-optimised SC-LDPC construction vs baselines (BPSK/AWGN, windowed)",
                          path="exp_construct_ber.svg")
    # ---- transfer bar-ish (semilogy points) ----------------------------------
    if "transfer" in res:
        tr = res["transfer"]
        labels = list(tr.keys())
        xs = list(range(1, len(labels) + 1))
        series = [
            {"x": xs, "y": [tr[l]["learned_fer"] for l in labels],
             "label": "learned policy (zero-shot)", "color": "#d62728"},
            {"x": xs, "y": [tr[l]["random_median"] for l in labels],
             "label": "random (median)", "color": "#1f77b4"},
            {"x": xs, "y": [tr[l]["random_best"] for l in labels],
             "label": "random (best of 12)", "color": "#7f7f7f"},
            {"x": xs, "y": [tr[l]["round_robin_fer"] for l in labels],
             "label": "round-robin", "color": "#8c564b"},
        ]
        plotting.semilogy(series, xlabel="transfer target (1,2)", ylabel="FER @ train SNR",
                          title="Zero-shot transfer of the learned construction policy",
                          path="exp_construct_transfer.svg")
    print("wrote exp_construct_learn.svg, exp_construct_ber.svg, exp_construct_transfer.svg")


def run_robust(n=2000, seed=54321):
    """Remove champion-selection noise: (a) score the *deterministic greedy*
    construction implied by the learned policy (no sampling luck), and (b) re-rank
    every champion on a fresh, independent CRN bank."""
    res = json.load(open(OP.route("results_construct_search.json")))
    theta = np.asarray(res["theta_feature"], dtype=np.float64)
    pol = R.FeaturePolicy(CFG); pol.theta = theta.copy()
    greedy = pol.greedy_assign()
    champs = dict(res["champions"]); champs["rl_feature_greedy"] = _arr(greedy)
    robust = {}
    with mp.Pool(R.n_workers()) as pool:
        print(f"=== independent re-rank @ {TRAIN_SNR} dB over {n} fresh frames ===")
        for name, a in champs.items():
            a = np.asarray(a, dtype=np.int64)
            m = R.eval_assignments(CFG, [a], TRAIN_SNR, seed, n, pool=pool,
                                   frame_chunks=R.n_workers())[0]
            st = R.construction_stats(CFG, a)
            robust[name] = {"fer": m["fer"], "ber": m["ber"], "n4": st["n4"]}
            print(f"  {name:20s} FER={m['fer']:.4f}  BER={m['ber']:.3e}  n4={st['n4']}", flush=True)
    res["robust"] = robust
    res["champions"]["rl_feature_greedy"] = _arr(greedy)
    json.dump(res, open(OP.route("results_construct_search.json"), "w"), indent=1)
    print("saved robust re-rank into results_construct_search.json")
    return res


def run_all():
    run_search()
    run_curves()
    run_analysis()
    run_transfer()
    run_robust()
    make_plots()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"search": run_search, "curves": run_curves, "analysis": run_analysis,
     "transfer": run_transfer, "robust": run_robust, "plot": make_plots,
     "all": run_all}[cmd]()
