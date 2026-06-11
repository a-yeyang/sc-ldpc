"""STRONG classical-baseline driver for SC-LDPC edge-spreading construction.

Companion to ``experiments_ci.py`` (RL / CEM / random search, multi-seed + CI).
A strict IEEE T-COM reviewer will demand the construction wins be measured not
only against *weak* baselines (uniform random, round-robin, default seed0) but
against established *classical construction algorithms*.  This driver builds the
three classical edge-spreading baselines from ``classical_baseline.py`` --

  * ``peg``      -- PEG-style greedy (maximise local girth / minimise new short
                    cycles, Hu/Eleftheriou/Arnold philosophy),
  * ``ace``      -- ACE-style greedy (Tian/Jones/Villasenor/Wesel 2004, penalise
                    low-ACE short cycles),
  * ``greedy4c`` -- harmful-object elimination local search (Battaglioni-style):
                    greedily reassign single edges to minimise the EXACT lifted
                    4-cycle count,

on the SAME cells as ``experiments_ci.CELLS``, re-evaluates each on the SAME large
final frame bank (same ``final`` count and ``FINAL_SEED`` as experiments_ci so the
FER numbers are directly comparable), and records the full
FER/BER/bit_err/n_info/frame_err/n_frames + n4/girth + construction wall-time.

These baselines are DETERMINISTIC graph-structure heuristics: no RL, no Monte-Carlo
in the construction loop (Monte-Carlo is used ONLY to *score* the finished code,
exactly as for the RL/CEM champions -- a fair, identical evaluator).

Usage (run on a CPU pod, EXP_WORKERS = pod cores):
  EXP_WORKERS=96 python3 experiments_classical.py cell R0.6_deep   # one cell
  EXP_WORKERS=96 python3 experiments_classical.py all              # every cell
  python3 experiments_classical.py agg                             # juxtaposed table
Checkpoints to results_classical_<label>.json after EVERY baseline so an evicted
pod loses at most one build+rescore.
"""
from __future__ import annotations
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import rl_construct as R
import classical_baseline as CB
import experiments_ci as CI

# Reuse experiments_ci's cells, final frame counts, SNR-freeze and FINAL_SEED so
# classical FER is directly comparable to results_ci_<label>.json.
CELLS = CI.CELLS
FINAL_SEED = CI.FINAL_SEED
BASELINES = ["peg", "ace", "greedy4c"]


def _champ_stats(cfg, assign):
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return {"n4": R.count_4cycles(sc), "girth": R.girth(sc, n_seeds=200)}


def _build_baseline(cfg, name):
    """Construct one classical baseline; return (assign, build_seconds)."""
    t0 = time.time()
    a = CB.build_baseline(cfg, name)
    return np.asarray(a, dtype=np.int64), round(time.time() - t0, 3)


def run_cell(label, pool):
    spec = CELLS[label]
    cfg = spec["cfg"]
    fname = f"results_classical_{label}.json"
    if os.path.exists(fname):
        out = json.load(open(fname))
    else:
        out = {"label": label, "cfg": CI._cfg_d(cfg),
               "budget": {"final": spec["final"], "final_seed": FINAL_SEED},
               "records": []}
    # freeze SNR identically to experiments_ci (same op-point => comparable FER)
    snr = spec["snr"]
    if snr is None:
        # reuse the experiments_ci checkpoint's frozen SNR if present, else re-pick
        ci_fname = f"results_ci_{label}.json"
        if os.path.exists(ci_fname):
            snr = json.load(open(ci_fname)).get("snr")
        if snr is None:
            snr = out.get("snr") or CI._freeze_snr(cfg, spec["cand"], pool)
    out["snr"] = snr
    rate = cfg.build().rate
    out["rate"] = rate
    done = {r["method"] for r in out["records"]}
    E = R.edge_meta(cfg)[2]
    print(f"=== {label}: rate={rate:.3f} eval@{snr}dB E={E} final={spec['final']} "
          f"({len(done)}/{len(BASELINES)} done) ===", flush=True)
    for name in BASELINES:
        if name in done:
            continue
        t0 = time.time()
        assign, build_sec = _build_baseline(cfg, name)
        # high-precision re-eval on the SAME large independent frame bank as CI
        m = R.eval_assignments(cfg, [assign], snr, FINAL_SEED, spec["final"],
                               pool=pool, frame_chunks=R.n_workers())[0]
        st = _champ_stats(cfg, assign)
        rec = {"method": name, "snr": snr, "build_sec": build_sec,
               "fer": m["fer"], "ber": m["ber"], "bit_err": m["bit_err"],
               "n_info": m["n_info"], "frame_err": m["frame_err"], "n_frames": m["n_frames"],
               "n4": st["n4"], "girth": st["girth"],
               "assign": np.asarray(assign, dtype=np.int64).tolist(),
               "sec": round(time.time() - t0, 1)}
        out["records"].append(rec)
        json.dump(out, open(fname, "w"))           # checkpoint after EVERY baseline
        print(f"  {name:9s}: FER={m['fer']:.4f} BER={m['ber']:.2e} "
              f"n4={st['n4']} girth={st['girth']} build={build_sec}s ({rec['sec']}s)",
              flush=True)
    print(f"--- {label} done -> {fname}", flush=True)
    return out


# ----------------------------------------------------------------------------- #
# aggregation: classical FER juxtaposed with RL/CEM/random from results_ci_*.json
# ----------------------------------------------------------------------------- #
def aggregate():
    summary = {}
    for label in CELLS:
        fname = f"results_classical_{label}.json"
        if not os.path.exists(fname):
            continue
        out = json.load(open(fname))
        cls = {r["method"]: r for r in out["records"]}
        # pull RL/CEM/random reference numbers (mean FER over seeds) if present
        ref = {}
        ci_fname = f"results_ci_{label}.json"
        if os.path.exists(ci_fname):
            ci = json.load(open(ci_fname))
            for method in CI.METHODS:
                fers = [r["fer"] for r in ci["records"] if r["method"] == method]
                n4s = [r["n4"] for r in ci["records"] if r["method"] == method]
                if fers:
                    ref[method] = {"mean_fer": float(np.mean(fers)),
                                   "min_fer": float(np.min(fers)),
                                   "n": len(fers),
                                   "n4_median": float(np.median(n4s))}
        summary[label] = {"rate": out.get("rate"), "snr": out.get("snr"),
                          "classical": {k: {"fer": v["fer"], "ber": v["ber"],
                                            "n4": v["n4"], "girth": v["girth"],
                                            "build_sec": v["build_sec"]}
                                        for k, v in cls.items()},
                          "ref": ref}
    json.dump(summary, open("results_classical_summary.json", "w"), indent=2)
    # printed juxtaposition table
    print("\n=== Classical baselines vs RL/CEM/random (champion FER) ===")
    for label, s in summary.items():
        rt = s["rate"]; rt = f"{rt:.3f}" if rt is not None else "?"
        print(f"\n[{label}] rate={rt} @ {s['snr']}dB")
        print(f"  {'method':12s} {'FER':>9s} {'BER':>10s} {'n4':>6s} {'girth':>5s} {'build_s':>8s}")
        for name in BASELINES:
            c = s["classical"].get(name)
            if c:
                print(f"  {('cls:'+name):12s} {c['fer']:9.4f} {c['ber']:10.2e} "
                      f"{c['n4']:6d} {c['girth']:5d} {c['build_sec']:8.2f}")
        for method in CI.METHODS:
            r = s["ref"].get(method)
            if r:
                print(f"  {('rl:'+method):12s} {r['mean_fer']:9.4f} {'(min '+format(r['min_fer'],'.4f')+')':>10s} "
                      f"{int(r['n4_median']):6d} {'-':>5s} {'-':>8s}  [n={r['n']} seeds]")
        if not s["ref"]:
            print("  (no results_ci_*.json reference found for this cell)")
    print("\n-> results_classical_summary.json", flush=True)
    return summary


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "agg":
        aggregate(); return
    nw = R.n_workers()
    print(f"workers={nw}", flush=True)
    with Pool(nw) as pool:
        if cmd == "cell":
            run_cell(sys.argv[2], pool)
        elif cmd == "all":
            for label in CELLS:
                run_cell(label, pool)
        else:
            raise SystemExit(f"unknown cmd {cmd} (use: cell <label> | all | agg)")


if __name__ == "__main__":
    main()
