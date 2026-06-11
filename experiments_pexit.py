"""Driver for the protograph-PEXIT threshold analysis (gap-filler (1), TCOM_PLAN S5).

Frame-free, analysis-grounded decoding-threshold study for the RL edge-spreading
SC-LDPC paper.  Produces results_pexit.json with:

  1. CORRECTNESS GATE  -- (3,6)/(4,8)-regular BIAWGN PEXIT thresholds vs textbook
     (1.10 dB), and a coupled (3,6)-SC chain threshold demonstrating threshold
     saturation (BP threshold lifts toward the MAP threshold sigma~0.95).
  2. PER-CELL THRESHOLDS -- for each rate cell (R0.5..R0.9, the S5.6 sweep) the
     PEXIT threshold of round_robin + several random constructions, and the
     within-cell spread (does PEXIT separate constructions, or is it ~invariant
     because of threshold saturation?).
  3. SPEARMAN -- per-cell PEXIT threshold vs the Monte-Carlo waterfall FER=0.1
     threshold (from results/construct/results_construct_rate.json).
  4. PEXIT-REWARD RL -- only run if step 2 shows PEXIT separates constructions
     within a cell; otherwise reported as a (valid, expected) null result.

Pure NumPy, no scipy.  Designed to run serially on a CPU pod in a few minutes.

Run:
    python3 experiments_pexit.py                 # full run
    python3 experiments_pexit.py --smoke         # gate + one cell only (fast)
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
import argparse
import numpy as np

from rl_construct import Config, round_robin_assign, random_assign, edge_meta
from pexit import (pexit_threshold, regular_threshold_db, sc_protograph,
                   _bisect_sigma_threshold, sigma_to_ebn0, sc_rate,
                   component_protograph)

OUT = "results_pexit.json"

# Rate cells = the S5.6 full-rate sweep (mirrors results_construct_rate.json 'plan')
RATE_CELLS = [
    ("R0.5", Config(bg=2, ils=0, Z=16, mp=12, w=2, L=30)),
    ("R0.6", Config(bg=2, ils=0, Z=16, mp=8,  w=2, L=30)),
    ("R0.7", Config(bg=1, ils=0, Z=16, mp=11, w=2, L=30)),
    ("R0.8", Config(bg=1, ils=0, Z=16, mp=8,  w=2, L=30)),
    ("R0.9", Config(bg=1, ils=0, Z=16, mp=4,  w=2, L=30)),
]


# --------------------------------------------------------------------------- #
#  utilities (no scipy)
# --------------------------------------------------------------------------- #
def spearman(x, y):
    """Spearman rank correlation (no scipy).  Average-rank ties; returns nan if
    fewer than 3 finite pairs or zero variance."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size)
        r[order] = np.arange(a.size, dtype=float)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(cnt.size)
        np.add.at(sums, inv, r)
        avg = sums / cnt
        return avg[inv]

    rx, ry = rank(x), rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def mc_fer_threshold(curve, fer_target=0.1):
    """Eb/N0 at which a Monte-Carlo FER curve crosses fer_target, by log-linear
    interpolation in (Eb/N0, log10 FER).  `curve` has 'x' (dB) and 'fer'."""
    x = np.asarray(curve["x"], float)
    fer = np.asarray(curve["fer"], float)
    # work left->right; find the bracket where fer crosses target
    lf = np.log10(np.clip(fer, 1e-9, 1.0))
    lt = np.log10(fer_target)
    thr = None
    for k in range(len(x) - 1):
        a, b = lf[k], lf[k + 1]
        if (a - lt) * (b - lt) <= 0 and a != b:
            frac = (lt - a) / (b - a)
            thr = x[k] + frac * (x[k + 1] - x[k])
            break
    if thr is None:
        # no crossing: if all below target, threshold <= min x; if all above, > max x
        if np.all(fer <= fer_target):
            thr = x[0]
        else:
            thr = x[-1]
    return float(thr)


# --------------------------------------------------------------------------- #
#  1. correctness gate
# --------------------------------------------------------------------------- #
def regular_sc_chain_threshold(dv_per_pos=1, nbp=2, w=2, L=30, max_iter=4000):
    """A faithful rate-1/2 (3,6)-regular SC chain protograph (one edge per
    component, nbp var-types/position) with termination; returns BP threshold
    sigma_awgn.  Demonstrates threshold saturation vs the uncoupled (3,6) code."""
    mbp = 1
    comps = [np.ones((mbp, nbp), dtype=np.int64) for _ in range(w + 1)]
    MB = (L + w) * mbp
    NB = L * nbp
    B = np.zeros((MB, NB), dtype=np.int64)
    for a in range(w + 1):
        for t in range(L):
            i = t + a
            if i > L + w - 1:
                continue
            B[i * mbp:(i + 1) * mbp, t * nbp:(t + 1) * nbp] += comps[a]
    var_kind = ["rx"] * NB
    for t in range(L - w, L):
        for j in range(nbp):
            var_kind[t * nbp + j] = "known"
    proto = {"Bproto": B, "var_kind": var_kind, "nb": NB, "mb": MB}
    return _bisect_sigma_threshold(proto, sig_lo=0.5, sig_hi=1.3,
                                   n_iter=30, max_iter=max_iter)


def run_gate():
    print("=== CORRECTNESS GATE ===", flush=True)
    g = {}
    db36, s36 = regular_threshold_db(3, 6)
    db48, s48 = regular_threshold_db(4, 8)
    g["reg_3_6"] = {"ebn0_db": db36, "sigma": s36, "textbook_db": 1.10,
                    "err_db": abs(db36 - 1.10)}
    g["reg_4_8"] = {"ebn0_db": db48, "sigma": s48, "textbook_db": 1.62}
    print(f"  (3,6)-regular: {db36:.3f} dB  (sigma {s36:.4f})  "
          f"[textbook 1.10 dB / 0.88;  err {abs(db36-1.10):.3f} dB]", flush=True)
    print(f"  (4,8)-regular: {db48:.3f} dB  (sigma {s48:.4f})  [textbook ~1.62 dB]",
          flush=True)
    gate_pass_reg = abs(db36 - 1.10) <= 0.15
    print(f"  GATE-A (3,6) within 0.15 dB: {'PASS' if gate_pass_reg else 'FAIL'}",
          flush=True)

    # threshold saturation: coupled (3,6)-SC chain BP threshold > uncoupled
    s_sc = regular_sc_chain_threshold(L=30)
    g["sc_3_6_chain"] = {"sigma_bp": s_sc, "uncoupled_sigma": s36,
                         "map_sigma_ref": 0.948}
    sat = (s_sc is not None) and (s_sc > s36 + 1e-3)
    print(f"  (3,6)-SC coupled chain BP threshold sigma {s_sc:.4f}  "
          f"vs uncoupled {s36:.4f}  (MAP ref ~0.948)", flush=True)
    print(f"  GATE-B threshold saturation (coupled sigma > uncoupled): "
          f"{'PASS' if sat else 'FAIL'}", flush=True)

    # 5G-NR coupled vs its own uncoupled component (the real ensemble)
    cfg = RATE_CELLS[0][1]
    proto_sc = sc_protograph(cfg, round_robin_assign(cfg))
    sig_sc = _bisect_sigma_threshold(proto_sc, n_iter=26, max_iter=400)
    Bc, vk = component_protograph(cfg)
    proto_unc = {"Bproto": Bc, "var_kind": vk, "nb": Bc.shape[1], "mb": Bc.shape[0]}
    sig_unc = _bisect_sigma_threshold(proto_unc, n_iter=26, max_iter=400)
    db_sc = sigma_to_ebn0(sig_sc, sc_rate(cfg, round_robin_assign(cfg)))
    db_unc = sigma_to_ebn0(sig_unc, cfg.component().rate)
    g["nr_coupled_vs_uncoupled"] = {
        "coupled_db": db_sc, "uncoupled_db": db_unc,
        "coupled_sigma": sig_sc, "uncoupled_sigma": sig_unc}
    print(f"  5G-NR R0.5: coupled {db_sc:.3f} dB (sig {sig_sc:.4f}) vs "
          f"uncoupled {db_unc:.3f} dB (sig {sig_unc:.4f}) -> coupled better: "
          f"{db_sc <= db_unc + 1e-3}", flush=True)
    g["gate_pass"] = bool(gate_pass_reg and sat)
    return g


# --------------------------------------------------------------------------- #
#  2. per-cell PEXIT thresholds + within-cell construction spread
# --------------------------------------------------------------------------- #
def run_cells(n_random=8, max_iter=300, seed=0):
    print("\n=== PER-CELL PEXIT THRESHOLDS ===", flush=True)
    cells = {}
    for name, cfg in RATE_CELLS:
        t0 = time.time()
        _, _, E = edge_meta(cfg)
        rate = sc_rate(cfg, round_robin_assign(cfg))
        rr = pexit_threshold(cfg, round_robin_assign(cfg), max_iter=max_iter)
        rng = np.random.default_rng(seed + hash(name) % 1000)
        rand_th, rand_assigns = [], []
        for _ in range(n_random):
            a = random_assign(cfg, rng)
            th = pexit_threshold(cfg, a, max_iter=max_iter)
            rand_th.append(th)
            rand_assigns.append(np.asarray(a).tolist())
        rand_th = np.array([t for t in rand_th if t is not None])
        all_th = np.concatenate([[rr] if rr is not None else [], rand_th])
        spread = float(all_th.max() - all_th.min()) if all_th.size else float("nan")
        cells[name] = {
            "rate": float(rate), "E": int(E),
            "bg": cfg.bg, "mp": cfg.mp,
            "rr_threshold_db": rr,
            "random_thresholds_db": rand_th.tolist(),
            "rr_assign": np.asarray(round_robin_assign(cfg)).tolist(),
            "random_assigns": rand_assigns,
            "spread_db": spread,
            "std_db": float(all_th.std()) if all_th.size else float("nan"),
            "mean_db": float(all_th.mean()) if all_th.size else float("nan"),
        }
        print(f"  {name} (R={rate:.3f}, BG{cfg.bg}, E={E}): "
              f"rr={rr:.4f} dB  rand[min={rand_th.min():.4f} "
              f"max={rand_th.max():.4f}]  spread={spread:.4f} dB  "
              f"std={all_th.std():.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return cells


# --------------------------------------------------------------------------- #
#  3. Spearman: per-cell PEXIT threshold vs MC waterfall FER=0.1 threshold
# --------------------------------------------------------------------------- #
def run_spearman(cells, mc_path="results/construct/results_construct_rate.json"):
    print("\n=== SPEARMAN: PEXIT threshold vs MC waterfall threshold ===",
          flush=True)
    if not os.path.exists(mc_path):
        print(f"  (skipped: {mc_path} not found)", flush=True)
        return {"available": False}
    mc = json.load(open(mc_path))["rates"]
    rows = []
    for name, info in cells.items():
        if name not in mc:
            continue
        cell = mc[name]
        # PEXIT threshold to compare = round_robin's (a fixed reference construction)
        pexit_rr = info["rr_threshold_db"]
        pexit_mean = info["mean_db"]
        # MC waterfall threshold of the *same* round_robin construction, FER=0.1
        mc_rr = None
        if "round_robin" in cell["curves"]:
            mc_rr = mc_fer_threshold(cell["curves"]["round_robin"], 0.1)
        # also the BEST (rl) MC curve threshold (the optimised construction)
        mc_rl = mc_fer_threshold(cell["curves"]["rl"], 0.1) if "rl" in cell["curves"] else None
        rows.append({"cell": name, "rate": info["rate"],
                     "pexit_rr_db": pexit_rr, "pexit_mean_db": pexit_mean,
                     "mc_rr_db": mc_rr, "mc_rl_db": mc_rl})
        print(f"  {name}: PEXIT(rr)={pexit_rr:.3f}  PEXIT(mean)={pexit_mean:.3f}  "
              f"MC_rr@FER0.1={mc_rr}  MC_rl@FER0.1={mc_rl}", flush=True)
    px_rr = [r["pexit_rr_db"] for r in rows]
    px_mean = [r["pexit_mean_db"] for r in rows]
    mc_rr = [r["mc_rr_db"] for r in rows]
    mc_rl = [r["mc_rl_db"] for r in rows]
    rho_rr = spearman(px_rr, mc_rr)
    rho_rl = spearman(px_mean, mc_rl)
    print(f"  Spearman(PEXIT_rr, MC_rr@FER0.1)  = {rho_rr:+.3f}  (n={len(rows)})",
          flush=True)
    print(f"  Spearman(PEXIT_mean, MC_rl@FER0.1)= {rho_rl:+.3f}  (n={len(rows)})",
          flush=True)
    return {"available": True, "rows": rows,
            "spearman_pexit_rr_vs_mc_rr": rho_rr,
            "spearman_pexit_mean_vs_mc_rl": rho_rl}


# --------------------------------------------------------------------------- #
#  4. (conditional) PEXIT-reward REINFORCE -- only if PEXIT separates within cell
# --------------------------------------------------------------------------- #
def run_pexit_rl(cfg, n_steps=20, batch=16, max_iter=200, seed=0, baseline=True):
    """Self-contained REINFORCE loop reusing rl_construct.FeaturePolicy with
    reward = -pexit_threshold(cfg, assign).  Does NOT modify rl_construct.
    Returns the champion assignment + its PEXIT threshold, vs round_robin."""
    from rl_construct import FeaturePolicy
    print("\n=== PEXIT-REWARD REINFORCE (champion vs round_robin) ===", flush=True)
    pol = FeaturePolicy(cfg, seed=seed)
    best = {"thr": np.inf, "assign": None}
    for step in range(1, n_steps + 1):
        traces, assigns, rewards = [], [], []
        for _ in range(batch):
            a, tr = pol.rollout()
            thr = pexit_threshold(cfg, a, max_iter=max_iter)
            r = -(thr if thr is not None else 10.0)
            traces.append(tr); assigns.append(a); rewards.append(r)
            if thr is not None and thr < best["thr"]:
                best = {"thr": thr, "assign": np.asarray(a).copy()}
        rewards = np.array(rewards)
        adv = rewards - rewards.mean()
        if adv.std() > 1e-9:
            adv = adv / (adv.std() + 1e-9)
        pol.update(list(zip(traces, adv)))
        if step % 5 == 0 or step == 1:
            print(f"  [PG] step {step:2d}  meanR={rewards.mean():+.4f}  "
                  f"best_thr={best['thr']:.4f} dB", flush=True)
    rr_thr = pexit_threshold(cfg, round_robin_assign(cfg), max_iter=max_iter)
    print(f"  champion PEXIT thr {best['thr']:.4f} dB  vs round_robin "
          f"{rr_thr:.4f} dB", flush=True)
    return {"champion_thr_db": best["thr"],
            "champion_assign": best["assign"].tolist() if best["assign"] is not None else None,
            "round_robin_thr_db": rr_thr}


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="gate + one rate cell only (fast local check)")
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force-rl", action="store_true",
                    help="run the PEXIT-reward RL loop even if no within-cell separation")
    args = ap.parse_args()

    t_start = time.time()
    res = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "smoke": args.smoke}}

    # 1. gate
    res["gate"] = run_gate()
    json.dump(res, open(args.out, "w"), indent=1)
    if not res["gate"]["gate_pass"]:
        print("\n!!! CORRECTNESS GATE FAILED -- thresholds are not trustworthy. "
              "Stopping.", flush=True)
        res["stopped"] = "gate_failed"
        json.dump(res, open(args.out, "w"), indent=1)
        return

    if args.smoke:
        # one cell only
        name, cfg = RATE_CELLS[0]
        cells = {}
        rr = pexit_threshold(cfg, round_robin_assign(cfg), max_iter=args.max_iter)
        rng = np.random.default_rng(0)
        rand = [pexit_threshold(cfg, random_assign(cfg, rng), max_iter=args.max_iter)
                for _ in range(3)]
        all_th = np.array([rr] + rand)
        cells[name] = {"rate": float(sc_rate(cfg, round_robin_assign(cfg))),
                       "rr_threshold_db": rr, "random_thresholds_db": rand,
                       "spread_db": float(all_th.max() - all_th.min())}
        print(f"\n[SMOKE] {name}: rr={rr:.4f} dB  rand={[round(x,4) for x in rand]}  "
              f"spread={all_th.max()-all_th.min():.4f} dB", flush=True)
        res["cells"] = cells
        res["elapsed_s"] = time.time() - t_start
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"\nSMOKE done in {res['elapsed_s']:.1f}s -> {args.out}", flush=True)
        return

    # 2. per-cell thresholds + spread
    res["cells"] = run_cells(n_random=args.n_random, max_iter=args.max_iter)
    json.dump(res, open(args.out, "w"), indent=1)

    # 3. Spearman vs MC waterfall
    res["spearman"] = run_spearman(res["cells"])
    json.dump(res, open(args.out, "w"), indent=1)

    # 4. conditional PEXIT-reward RL
    max_spread = max(c["spread_db"] for c in res["cells"].values())
    separates = max_spread > 0.10        # dB: needs >0.1 dB to be a useful reward
    res["within_cell_separates"] = bool(separates)
    res["max_within_cell_spread_db"] = float(max_spread)
    if separates or args.force_rl:
        # pick the cell with the largest spread (most room for PEXIT reward)
        best_cell = max(res["cells"].items(), key=lambda kv: kv[1]["spread_db"])[0]
        cfg = dict(RATE_CELLS)[best_cell]
        res["pexit_rl"] = run_pexit_rl(cfg)
    else:
        print("\n=== PEXIT-REWARD RL: SKIPPED ===", flush=True)
        print(f"  within-cell PEXIT spread is only {max_spread:.4f} dB "
              f"(<0.10 dB) across all cells -> threshold saturation makes PEXIT "
              f"~construction-invariant, so it is NOT a useful per-construction "
              f"reward. This NULL result is the theory contribution (explains why "
              f"the MC waterfall threshold is construction-robust).", flush=True)
        res["pexit_rl"] = {"skipped": True,
                           "reason": "no within-cell separation (threshold saturation)"}

    res["elapsed_s"] = time.time() - t_start
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nDONE in {res['elapsed_s']:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
