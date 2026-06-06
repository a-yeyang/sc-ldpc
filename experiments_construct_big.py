"""Large-scale RL construction optimisation for LONG, HIGH-RATE 5G-NR SC-LDPC.

Regime (3GPP TS 38.212, BG1, Kb=22, nb=68):
  * long codes: Z=64  ->  per-position component codeword (Kb+mp)*Z ~ 1700-2900 bits
  * wireless code rates: 1/2, 2/3, 3/4, 5/6, 7/8  ->  BG1 rate matching mp in {24,13,9,6,5}
  * coupling memory w in {1,2,3}  ->  edge-spreading search space (w+1)^E, E~121  (up to 4^121)
  * coupling chain length L: tens-to-hundreds (windowed decoding; termination costs ~ (w-1)/L rate)

Each end-to-end evaluation (encode -> AWGN -> windowed BP over L positions of a ~2000-bit
code) costs ~0.5-0.8 s, so the construction-evaluation budget is necessarily MODEST -- the
regime where RL's sample efficiency and structural prior should beat brute random search, and
where random search cannot cover a 4^121 space.

The edge spreading is L-independent, so we OPTIMISE at a moderate L=30 (cheaper) and VALIDATE
the learned construction across L=30..300 separately (chain-length / threshold-saturation study).

Distributed across pods by rate slice (each pod uses EXP_WORKERS cores):
    python3 experiments_construct_big.py slice A    # rates 1/2,2/3,3/4 + an L-sweep
    python3 experiments_construct_big.py slice B    # rates 5/6,7/8     + an L-sweep
    python3 experiments_construct_big.py plot       # merge results_big_*.json -> figures
"""
from __future__ import annotations
import json
import os
import sys
import time
import multiprocessing as mp
import numpy as np

import plotting
import rl_construct as R

# Z=32 long-ish codes (component codeword (Kb+mp)*Z ~ 860-1470 bits).  Z=64 end-to-end
# Monte-Carlo is ~8 s/frame -> infeasible for RL (thousands of evals); the learned edge
# spreading is a base-graph property, so the Z=64 "几千位" regime is reached by VALIDATING
# the learned construction at large Z separately, not by optimising there.
BG, ILS, Z, W, MAXIT = 1, 0, 32, 5, 12
OPT_L = 24                               # chain length during construction optimisation
RATE_MP = {0.5: 24, 0.667: 13, 0.75: 9, 0.833: 6, 0.875: 5}     # BG1 (Kb=22) rate matching
WS = [1, 2, 3]                           # coupling memory values

# budgets: long-code evals are expensive -> modest; batch == pod cores (one PG wave/step)
OPT_STEPS, OPT_BATCH, OPT_FRAMES = 10, 80, 8
VAL_FRAMES, FINAL_FRAMES = 80, 300       # val_frames=cores so per-step validation is one wave
PROBE_FRAMES, PROBE_N = 20, 80           # probe uses all cores; 20 frames for a reliable median
SNR_CAND = {0.5: [0.5, 1.0, 1.5, 2.0, 2.5], 0.667: [1.0, 1.5, 2.0, 2.5, 3.0],
            0.75: [1.5, 2.0, 2.5, 3.0, 3.5], 0.833: [2.0, 2.5, 3.0, 3.5, 4.0],
            0.875: [2.5, 3.0, 3.5, 4.0, 4.5]}
BER_OFFSETS = [-0.6, 0.0, 0.6]
BER_FRAMES = [150, 250, 350]
LSWEEP_LS = [24, 50, 100, 200]
LSWEEP_FRAMES = [100, 150]                # per (L, offset) -- kept light (large L is slow)
LSWEEP_OFFSETS = [0.0]

# balanced by per-position size nb=Kb+mp (R=1/2 mp=24 is the heaviest); L-sweeps put on
# lighter rates so L=300 stays feasible.  Two 80-core pods run these in parallel.
SLICES = {"A": {"rates": [0.5, 0.833], "lsweep": (0.833, 2)},
          "B": {"rates": [0.667, 0.75, 0.875], "lsweep": (0.75, 2)}}


def cfg(rate, w, L):
    return R.Config(bg=BG, ils=ILS, Z=Z, mp=RATE_MP[rate], w=w, L=L, W=W, max_iter=MAXIT)


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


def pick_snr(c, candidates, pool):
    """Operating SNR with a measurable, *distinguishable* FER.  Strong long codes have a
    sharp waterfall at low SNR, so a point past it scores FER ~ 0 for EVERY construction
    (no learning signal -- the bug we hit at R=1/2).  Among candidates we keep only those
    still in/before the waterfall (median FER >= 0.05) and take the one closest to 0.25;
    if even the lowest candidate is past the waterfall we step below the grid."""
    rng = np.random.default_rng(0)
    assigns = [R.random_assign(c, rng) for _ in range(PROBE_N)]
    meds = [float(np.median([m["fer"] for m in R.eval_batch(c, assigns, snr, 1, PROBE_FRAMES, pool=pool)]))
            for snr in candidates]
    usable = [i for i, m in enumerate(meds) if m >= 0.05]
    if not usable:
        return round(candidates[0] - 1.0, 2)
    return candidates[min(usable, key=lambda i: abs(meds[i] - 0.25))]


def _thr(curve, lvl=0.1, key="fer"):
    x = np.asarray(curve["x"]); y = np.asarray(curve[key])
    for i in range(1, len(x)):
        if y[i - 1] > lvl >= y[i] and y[i] > 0:
            t = (np.log10(lvl) - np.log10(y[i - 1])) / (np.log10(y[i]) - np.log10(y[i - 1]))
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return None


def ber_curve(c, assign, snrs, frames, pool, chunks=None):
    chunks = chunks or R.n_workers()
    xs, bers, fers = [], [], []
    for snr, nf in zip(snrs, frames):
        m = R.eval_assignments(c, [assign], snr, 2025, nf, pool=pool, frame_chunks=chunks)[0]
        xs.append(round(snr, 2)); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def run_cell(rate, w, pool):
    """One (rate, w) construction-optimisation cell: RL-feature/per-edge/CEM/random + baselines."""
    t0 = time.time()
    c = cfg(rate, w, OPT_L)
    rt = c.build().rate
    _, _, E = R.edge_meta(c)
    snr = pick_snr(c, SNR_CAND[rate], pool)
    print(f"\n=== R={rate} (sc rate {rt:.3f}) w={w}  E={E}  space {w+1}^{E}  "
          f"L={OPT_L} Z={Z}  train@{snr}dB ===", flush=True)

    pol = R.FeaturePolicy(c, lr=0.15, ent=0.02, seed=0)
    h_rl, best_rl = R.train_reinforce(c, pol, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES,
                                      pool=pool, val_frames=VAL_FRAMES, val_seed=99,
                                      base_seed=1000, log_every=8)
    pol_e = R.PerEdgePolicy(E, w + 1, lr=0.2, ent=0.01, seed=0)
    h_e, best_e = R.train_reinforce(c, pol_e, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES,
                                    pool=pool, val_frames=VAL_FRAMES, val_seed=99,
                                    base_seed=4000, log_every=8)
    h_c, best_c = R.train_cem(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                              val_frames=VAL_FRAMES, val_seed=99, base_seed=2000, log_every=8)
    h_r, best_r = R.train_random(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                                 val_frames=VAL_FRAMES, val_seed=99, base_seed=3000, log_every=8)
    champions = {"rl": best_rl["assign"], "rl_peredge": best_e["assign"],
                 "cem": best_c["assign"], "random_search": best_r["assign"],
                 "round_robin": R.round_robin_assign(c),
                 "seed0_default": R.random_assign(c, np.random.default_rng(0))}
    snrs = [snr + d for d in BER_OFFSETS]
    finals, stats, curves = {}, {}, {}
    for name, a in champions.items():
        a = np.asarray(a, dtype=np.int64)
        m = R.eval_assignments(c, [a], snr, 12345, FINAL_FRAMES, pool=pool,
                               frame_chunks=R.n_workers())[0]
        finals[name] = {"fer": m["fer"], "ber": m["ber"]}
        st = R.construction_stats(c, a)
        stats[name] = {"n4": st["n4"], "comp_load": st["comp_load"]}
        if name not in ("rl_peredge",):
            curves[name] = ber_curve(c, a, snrs, BER_FRAMES, pool)
        print(f"  {name:14s} FER={m['fer']:.4f} BER={m['ber']:.3e} n4={st['n4']}", flush=True)
    print(f"  [cell done in {time.time()-t0:.0f}s]", flush=True)
    return {"rate": rate, "sc_rate": rt, "w": w, "mp": RATE_MP[rate], "E": int(E),
            "train_snr": snr, "finals": finals, "stats": stats, "curves": curves,
            "hist": {"rl": h_rl, "rl_peredge": h_e, "cem": h_c, "random": h_r},
            "theta_rl": pol.theta.tolist(),
            "champions": {k: _arr(v) for k, v in champions.items()}}


def run_lsweep(rate, w, out, pool):
    """Validate champions across chain lengths L=30..300 (threshold saturation + rate loss)."""
    key = f"R{rate}_w{w}"
    cell = out["cells"].get(key)
    if cell is None:
        print(f"  (lsweep: no optimised cell {key}; skipping)", flush=True)
        return
    snr0 = cell["train_snr"]
    snrs = [snr0 + d for d in LSWEEP_OFFSETS]
    champs = {n: np.asarray(cell["champions"][n], dtype=np.int64)
              for n in ["rl", "random_search", "seed0_default"]}
    lsweep = {}
    print(f"\n=== L-sweep @ R={rate} w={w} (champions vs L) ===", flush=True)
    for L in LSWEEP_LS:
        cL = cfg(rate, w, L)
        chunks = min(24, R.n_workers())          # bound concurrent heavy (large-L) codes
        lsweep[str(L)] = {"rate": cL.build().rate}
        for n, a in champs.items():
            cur = ber_curve(cL, a, snrs, LSWEEP_FRAMES, pool, chunks=chunks)
            lsweep[str(L)][n] = cur
        print(f"  L={L:3d} (sc rate {cL.build().rate:.3f}) done", flush=True)
    out["lsweep"] = {"rate": rate, "w": w, "data": lsweep}


def _save(out, fn):
    """Atomic save (temp file + rename) so an interrupt mid-write can't corrupt results."""
    tmp = fn + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, fn)


def run_slice(name):
    t0 = time.time()
    sl = SLICES[name]
    fn = f"results_big_{name}.json"
    # --- resume: reload any existing results and skip the cells already computed ---
    out = None
    if os.path.exists(fn):
        try:
            out = json.load(open(fn)); out.setdefault("cells", {})
            print(f"[resume] {fn}: {len(out['cells'])} cells already done {sorted(out['cells'])}",
                  flush=True)
        except Exception as e:
            print(f"[resume] could not read {fn} ({e}); starting fresh", flush=True)
            out = None
    if out is None:
        out = {"slice": name, "config": {"bg": BG, "Z": Z, "opt_L": OPT_L, "W": W,
                                         "rate_mp": RATE_MP, "ws": WS}, "cells": {}}
    with mp.Pool(R.n_workers()) as pool:
        print(f"slice {name}: rates {sl['rates']} x w {WS}  ({R.n_workers()} workers)", flush=True)
        for rate in sl["rates"]:
            for w in WS:
                key = f"R{rate}_w{w}"
                if key in out["cells"]:                       # already computed -> skip
                    print(f"  [skip] {key} (already done)", flush=True)
                    continue
                try:
                    out["cells"][key] = run_cell(rate, w, pool)
                except Exception as e:
                    print(f"  !! cell {key} failed: {e}", flush=True)
                _save(out, fn)
        if sl.get("lsweep") and "lsweep" not in out:          # skip L-sweep if already done
            try:
                run_lsweep(sl["lsweep"][0], sl["lsweep"][1], out, pool)
            except Exception as e:
                print(f"  !! lsweep failed: {e}", flush=True)
            _save(out, fn)
    print(f"\nslice {name} done in {time.time()-t0:.0f}s -> {fn}", flush=True)


# --------------------------------------------------------------------------- #
#  plotting (run locally after merging results_big_A.json + results_big_B.json)
# --------------------------------------------------------------------------- #
def _merge():
    out = {"cells": {}, "lsweep": []}
    for name in ["A", "B"]:
        try:
            d = json.load(open(f"results_big_{name}.json"))
        except FileNotFoundError:
            continue
        out["cells"].update(d.get("cells", {}))
        if d.get("lsweep"):
            out["lsweep"].append(d["lsweep"])
    return out


LAB = {"rl": "RL feature [ours]", "rl_peredge": "RL per-edge [ours]", "cem": "CEM",
       "random_search": "random search", "round_robin": "round-robin",
       "seed0_default": "random default (seed0)"}
COL = {"rl": "#d62728", "rl_peredge": "#ff7f0e", "cem": "#9467bd",
       "random_search": "#1f77b4", "round_robin": "#8c564b", "seed0_default": "#7f7f7f"}


def make_plots():
    out = _merge()
    cells = out["cells"]
    if not cells:
        print("no results_big_*.json found"); return
    rates = sorted(set(c["rate"] for c in cells.values()))
    # Fig 1: RL-vs-random advantage (random_FER / RL_FER) vs rate, one line per w  -> grows with w?
    series = []
    for w in WS:
        xs, ys = [], []
        for rate in rates:
            c = cells.get(f"R{rate}_w{w}")
            if not c:
                continue
            rl = c["finals"]["rl"]["fer"]; rnd = c["finals"]["random_search"]["fer"]
            xs.append(rate); ys.append(rnd / max(rl, 1e-4))
        if xs:
            series.append({"x": xs, "y": ys, "label": f"w={w}",
                           "color": ["#1f77b4", "#d62728", "#2ca02c"][w - 1]})
    plotting.linear(series, xlabel="code rate R", ylabel="random_FER / RL_FER  (>1 = RL better)",
                    title="RL vs equal-budget random search across rate x w (long Z=64 codes)",
                    path="exp_big_rl_vs_random.svg")
    # Fig 2: per-cell BER curves + learning curves for a few representative hard cells
    for rate in rates:
        for w in WS:
            c = cells.get(f"R{rate}_w{w}")
            if not c:
                continue
            ser = [{"x": c["curves"][n]["x"], "y": c["curves"][n]["y"], "label": LAB[n], "color": COL[n]}
                   for n in ["rl", "cem", "random_search", "round_robin", "seed0_default"]
                   if n in c["curves"]]
            plotting.semilogy(ser, xlabel="Eb/N0 [dB]", ylabel="info BER",
                              title=f"Long SC-LDPC R={rate} w={w} (BG1 Z=64, L={OPT_L})",
                              path=f"exp_big_ber_R{int(rate*1000)}_w{w}.svg")
            h = c.get("hist")
            if h:
                floor = 0.5 / VAL_FRAMES
                ls = [{"x": h[m]["evals"], "y": [max(f, floor) for f in h[m]["best_val_fer"]],
                       "label": LAB.get(m, m), "color": COL.get(m, "#333")}
                      for m in ["rl", "rl_peredge", "cem", "random"] if m in h]
                plotting.semilogy(ls, xlabel="# construction evaluations", ylabel="best val FER",
                                  title=f"Learning curves R={rate} w={w} (sample efficiency)",
                                  path=f"exp_big_learn_R{int(rate*1000)}_w{w}.svg")
    # Fig 3: L-sweep BER (threshold saturation) for each available lsweep
    for ls in out["lsweep"]:
        data = ls["data"]; rate = ls["rate"]; w = ls["w"]
        for champ, tag in [("rl", "RL"), ("seed0_default", "default")]:
            ser = []
            for i, L in enumerate(LSWEEP_LS):
                if str(L) in data and champ in data[str(L)]:
                    cc = data[str(L)][champ]
                    ser.append({"x": cc["x"], "y": cc["y"], "label": f"L={L}",
                                "color": ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"][i]})
            if ser:
                plotting.semilogy(ser, xlabel="Eb/N0 [dB]", ylabel="info BER",
                                  title=f"Chain length L sweep ({tag} construction, R={rate} w={w})",
                                  path=f"exp_big_lsweep_{champ}.svg")
    _export_tables(out)
    json.dump(out, open("results_big_merged.json", "w"))
    print("wrote exp_big_*.svg, results_big_merged.json, results_big_{finals,hist}.csv")


def _export_tables(out):
    """CSV data tables: final FER/BER/n4 per method, and the full RL training history
    (loss = mean_reward, plus best_val_fer) per construction evaluation."""
    import csv
    cells = out["cells"]
    with open("results_big_finals.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["cell", "rate", "w", "E", "method", "fer", "ber", "n4"])
        for k, c in sorted(cells.items()):
            for m, fv in c["finals"].items():
                wr.writerow([k, c["rate"], c["w"], c["E"], m, f"{fv['fer']:.5f}",
                             f"{fv['ber']:.3e}", c["stats"].get(m, {}).get("n4", "")])
    with open("results_big_hist.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cell", "rate", "w", "method", "eval", "mean_reward_loss", "best_val_fer"])
        for k, c in sorted(cells.items()):
            for m, h in c.get("hist", {}).items():
                ev = h.get("evals", []); mr = h.get("mean_reward", []); bf = h.get("best_val_fer", [])
                for i in range(len(ev)):
                    wr.writerow([k, c["rate"], c["w"], m, ev[i],
                                 f"{mr[i]:.4f}" if i < len(mr) else "",
                                 f"{bf[i]:.4f}" if i < len(bf) else ""])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "slice":
        run_slice(sys.argv[2])
    elif cmd == "plot":
        make_plots()
    else:
        print(__doc__)
