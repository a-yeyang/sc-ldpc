"""vs-MCMC head-to-head (audit gap): compare the RL edge-spreading construction
against a MARKOV-CHAIN-MONTE-CARLO sampler -- the recent finite-length SC-LDPC
design paradigm of Tanrikulu-Yildirim-Hareedy (arXiv:2504.16071) / Hareedy et al.
"Adapt or Regress" (arXiv:2509.21112).  Reviewers will ask why RL beats the
established MCMC/simulated-annealing optimizer, not just PEG/ACE/random.

We implement a standard simulated-annealing Metropolis chain over the edge-spreading
assignment (single-edge component flips, geometric cooling), at the SAME evaluation
budget, evaluator, operating SNR, and independent champion re-test as the RL/CEM/
random methods of experiments_ci.py, on two cells:
  * R0.6_deep   (BG2 Z16 w2)  -- where RL~CEM~random tie (5 seeds): does MCMC also tie?
  * big_R0.667_w3 (BG1 Z32 w3) -- the strongest RL-over-random win cell (10/10, p=0.002,
                                  10 seeds): does MCMC ALSO beat random, i.e. is RL's
                                  edge specific to RL or to any structured search?
Two objectives (mirrors the cutting-vector study):
  mcmc_fer : Metropolis on the end-to-end FER (same objective as RL) -- the fair head-to-head.
  mcmc_n4  : Metropolis on the lifted-graph 4-cycle count (cheap, the literature's
             structural objective) -- re-scored by FER for comparison.
Each (method, seed) chain runs single-threaded; chains are parallelised across the
pool.  Champions are re-evaluated on the SAME independent 2000/1200-frame bank
(FINAL_SEED) used by experiments_ci, so the numbers drop straight into Table tab:ci.

Usage (CPU pod, EXP_WORKERS = pod cores):
  EXP_WORKERS=48 python3 experiments_mcmc.py run
  python3 experiments_mcmc.py agg
Checkpoints after every (cell, method) group.
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import rl_construct as R

CELLS = {
    "R0.6_deep":     dict(cfg=R.Config(bg=2, ils=0, Z=16, mp=8,  w=2, L=30, W=6, max_iter=12),
                          snr=2.5, final=2000, seeds=[0, 1, 2, 3, 4]),
    "big_R0.667_w3": dict(cfg=R.Config(bg=1, ils=0, Z=32, mp=13, w=3, L=24, W=5, max_iter=12),
                          snr=3.0, final=1200, seeds=list(range(10))),
}
N_EVALS, FRAMES, VAL = 512, 24, 80     # budget == experiments_ci STEPS*BATCH, frames, val
CHAIN_FRAME_SEED = 7777                # fixed CRN surface within a chain (low-variance accept/reject)
FINAL_SEED = 987654                    # SAME re-test bank as experiments_ci -> directly comparable
T0, TEND = 0.10, 0.005                 # SA temperature schedule (FER scale)


def _fer1(cfg, assign, snr, frames, frame_seed):
    """Single-construction FER over `frames` CRN frames, single-threaded (pool=None)."""
    return R.eval_assignments(cfg, [assign], snr, frame_seed, frames, pool=None)[0]["fer"]


def _n4(cfg, assign):
    return R.count_4cycles(cfg.build(assign=np.asarray(assign, dtype=np.int64)))


def _chain(task):
    """One simulated-annealing Metropolis chain -> champion assignment (single-threaded)."""
    cfg_d, snr, objective, seed = task
    cfg = R.Config(**cfg_d)
    E = R.edge_meta(cfg)[2]; C = R.n_components(cfg)
    rng = np.random.default_rng(1234 + seed * 97)
    cur = rng.integers(0, C, size=E)

    def energy(a):
        return _fer1(cfg, a, snr, FRAMES, CHAIN_FRAME_SEED) if objective == "fer" else float(_n4(cfg, a))

    cur_e = energy(cur)
    best_a, best_e = cur.copy(), cur_e
    for t in range(N_EVALS - 1):
        T = T0 * (TEND / T0) ** (t / max(N_EVALS - 2, 1))
        prop = cur.copy()
        e_idx = int(rng.integers(E))
        prop[e_idx] = int((prop[e_idx] + 1 + rng.integers(C - 1)) % C)   # flip to a DIFFERENT component
        pe = energy(prop)
        if pe <= cur_e or rng.random() < math.exp(-(pe - cur_e) / max(T, 1e-9)):
            cur, cur_e = prop, pe
        if pe < best_e:
            best_a, best_e = prop.copy(), pe
    return np.asarray(best_a, dtype=np.int64).tolist()


def run():
    fname = "results_mcmc.json"
    nw = R.n_workers()
    print(f"workers={nw}", flush=True)
    out = json.load(open(fname)) if os.path.exists(fname) else {"budget": {"n_evals": N_EVALS, "frames": FRAMES},
                                                                "cells": {}}
    with Pool(nw) as pool:
        for label, spec in CELLS.items():
            cfg = spec["cfg"]; snr = spec["snr"]; final = spec["final"]; seeds = spec["seeds"]
            cell = out["cells"].setdefault(label, {"cfg": _cfg_d(cfg), "snr": snr, "final": final,
                                                    "rate": cfg.build().rate, "records": []})
            done = {(r["method"], r["seed"]) for r in cell["records"]}
            tasks = [(_cfg_d(cfg), snr, obj, s) for obj in ("fer", "n4") for s in seeds
                     if (f"mcmc_{obj}", s) not in done]
            if not tasks:
                continue
            print(f"=== {label}: {len(tasks)} chains (rate={cfg.build().rate:.3f} @{snr}dB) ===", flush=True)
            t0 = time.time()
            champs = pool.map(_chain, tasks)                         # all chains in parallel
            # high-precision champion re-eval on the SHARED independent bank
            metrics = R.eval_assignments(cfg, champs, snr, FINAL_SEED, final, pool=pool,
                                         frame_chunks=1)
            for (cfg_d, _snr, obj, s), a, m in zip(tasks, champs, metrics):
                st = {"n4": _n4(cfg, a), "girth": R.girth(cfg.build(assign=np.asarray(a, dtype=np.int64)), n_seeds=200)}
                cell["records"].append({"method": f"mcmc_{obj}", "seed": s, "snr": snr,
                                        "fer": m["fer"], "ber": m["ber"], "bit_err": m["bit_err"],
                                        "n_info": m["n_info"], "frame_err": m["frame_err"],
                                        "n_frames": m["n_frames"], "n4": st["n4"], "girth": st["girth"],
                                        "assign": a})
                json.dump(out, open(fname, "w"))
            print(f"  {label} done ({time.time()-t0:.0f}s)", flush=True)
    agg()


def agg():
    fname = "results_mcmc.json"
    if not os.path.exists(fname):
        print("no results_mcmc.json"); return
    out = json.load(open(fname))
    print("\n=== vs-MCMC: champion FER (mean +/- 95% CI over seeds) ===")
    for label, cell in out["cells"].items():
        print(f"\n[{label}] rate={cell['rate']:.3f} @ {cell['snr']}dB")
        recs = cell["records"]
        for m in ["mcmc_fer", "mcmc_n4"]:
            fers = [r["fer"] for r in recs if r["method"] == m]
            n4s = [r["n4"] for r in recs if r["method"] == m]
            if not fers:
                continue
            n = len(fers); mean = np.mean(fers)
            ci = (2.776 if n == 5 else 2.262) * np.std(fers, ddof=1) / np.sqrt(n) if n > 1 else 0
            print(f"  {m:9s} n={n}  FER {mean:.4f} +/- {ci:.4f}  [{min(fers):.4f},{max(fers):.4f}]  "
                  f"n4_med={int(np.median(n4s))}")
    print("\n(compare to tab:ci: deep RL-feat 0.093, random 0.106; "
          "big_R0.667_w3 RL-feat 0.0090, random 0.0348)", flush=True)


def _cfg_d(cfg):
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "agg": agg}.get(cmd, lambda: (_ for _ in ()).throw(SystemExit("usage: run|agg")))()
