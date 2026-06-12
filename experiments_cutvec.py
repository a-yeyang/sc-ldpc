"""CPU experiment #2 -- benchmark against a NAMED literature SC-LDPC construction:
the CUTTING-VECTOR edge spreading of Mitchell-Lentmaier-Costello (IEEE T-IT 2015,
the paper's own [mitchell2015scldpc] reference).

Why this experiment: the reviewer's sharpest fair objection to the current draft is
"you compare RL only to random / CEM / PEG / ACE -- never to a *published SC-LDPC
edge-spreading design method*."  The canonical such method is the cutting vector:
a per-column row-threshold that deterministically splits the base matrix B into the
coupling components B_0..B_w.  This driver implements it faithfully and drops it
into the SAME evaluator / operating points / re-test protocol as Tables I-III, so
it slots in as a literature column.

A cutting vector assigns systematic base edge (r, c) to component
  comp(r,c) = #{ k in 1..w : zeta_k[c] <= r },
i.e. the band of column c into which check-base-row r falls, for non-decreasing
per-column thresholds zeta_1[c] <= ... <= zeta_w[c] in [0, m_b].  We report three
strengths so the comparison cannot be accused of handicapping the baseline:
  cv_uniform   : the textbook deterministic equal-band cut (zeta_k[c] = k*m_b/(w+1)).
  cv_best_n4   : best cutting vector by MIN 4-cycle count (the faithful literature
                 objective -- CVs are chosen for good cycle/distance properties).
  cv_best_fer  : best cutting vector by end-to-end VALIDATION FER over the same
                 family -- i.e. the STRONGEST possible CV, given RL's own objective
                 but restricted to the cutting-vector family.
If even cv_best_fer (CV family + FER objective) trails RL/CEM (full assign family),
that is a strong statement that the structural CV family itself is the limiter.

Candidates = a scalar-cut grid (column-independent zeta) UNION K random per-column
cutting vectors.  n_4 is computed for ALL candidates (cheap, frame-free); FER is
evaluated only for the lowest-n_4 shortlist (bounded cost).

Usage (CPU pod, EXP_WORKERS = pod cores):
  EXP_WORKERS=48 python3 experiments_cutvec.py run   # all cells -> results_cutvec_<label>.json
  python3 experiments_cutvec.py agg                  # printed comparison table
Checkpoints after every cell.
"""
from __future__ import annotations
import json
import os
import sys
import time
from itertools import combinations_with_replacement
from multiprocessing import Pool

import numpy as np

import rl_construct as R

# Cells + frozen operating points -- IDENTICAL to experiments_ci.py / classical so
# the CV numbers are directly comparable (FINAL_SEED, final frames, SNR all match).
FINAL_SEED = 987654
VAL_SEED = 987655
VAL_FRAMES = 120
N_RANDOM_CV = 300          # random per-column cutting vectors sampled
FER_SHORTLIST = 40         # FER-evaluate only the K lowest-n4 candidates

CELLS = {
    "R0.6_deep":     dict(cfg=R.Config(bg=2, ils=0, Z=16, mp=8,  w=2, L=30, W=6, max_iter=12), snr=2.5, final=2000),
    "big_R0.5_w3":   dict(cfg=R.Config(bg=1, ils=0, Z=32, mp=24, w=3, L=24, W=5, max_iter=12), snr=2.5, final=1200),
    "big_R0.667_w2": dict(cfg=R.Config(bg=1, ils=0, Z=32, mp=13, w=2, L=24, W=5, max_iter=12), snr=2.5, final=1200),
    "big_R0.667_w3": dict(cfg=R.Config(bg=1, ils=0, Z=32, mp=13, w=3, L=24, W=5, max_iter=12), snr=3.0, final=1200),
    "big_R0.75_w3":  dict(cfg=R.Config(bg=1, ils=0, Z=32, mp=9,  w=3, L=24, W=5, max_iter=12), snr=3.5, final=1200),
}


def _cv_assign(rows, cols, cuts):
    """cuts: array (w, n_cols) of non-decreasing per-column thresholds in [0, m_b].
    Return assign[e] = #{k : cuts[k, cols[e]] <= rows[e]}."""
    w = cuts.shape[0]
    a = np.zeros(len(rows), dtype=np.int64)
    for k in range(w):
        a += (cuts[k, cols] <= rows).astype(np.int64)
    return a


def _uniform_cuts(w, mb, n_cols):
    """Textbook equal-band scalar cutting vector (same threshold every column)."""
    thr = np.array([round((k + 1) * mb / (w + 1)) for k in range(w)], dtype=np.int64)
    return np.repeat(thr[:, None], n_cols, axis=1)


def _scalar_grid_cuts(w, mb, n_cols, n_levels=9):
    """Column-independent scalar cuts over a bounded threshold grid -> list of (w,n_cols)."""
    grid = sorted(set(int(round(x)) for x in np.linspace(0, mb, min(mb + 1, n_levels))))
    out = []
    for tup in combinations_with_replacement(grid, w):     # non-decreasing
        thr = np.array(tup, dtype=np.int64)
        out.append(np.repeat(thr[:, None], n_cols, axis=1))
    return out


def _random_percol_cuts(w, mb, n_cols, k, seed=0):
    """K random per-column cutting vectors (the general, generous CV family)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        cuts = np.sort(rng.integers(0, mb + 1, size=(w, n_cols)), axis=0)   # non-decreasing per col
        out.append(cuts.astype(np.int64))
    return out


def _n4_of(cfg, assign):
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return R.count_4cycles(sc)


def _stats(cfg, assign):
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return {"n4": R.count_4cycles(sc), "girth": R.girth(sc, n_seeds=200)}


def _eval(cfg, assign, snr, n_frames, pool):
    m = R.eval_assignments(cfg, [np.asarray(assign, dtype=np.int64)], snr, FINAL_SEED,
                           n_frames, pool=pool, frame_chunks=R.n_workers())[0]
    st = _stats(cfg, assign)
    return {"fer": m["fer"], "ber": m["ber"], "bit_err": m["bit_err"], "n_info": m["n_info"],
            "frame_err": m["frame_err"], "n_frames": m["n_frames"], "n4": st["n4"], "girth": st["girth"]}


def run_cell(label, pool):
    spec = CELLS[label]; cfg = spec["cfg"]; snr = spec["snr"]; final = spec["final"]
    rows, cols, E = R.edge_meta(cfg)
    mb = int(rows.max()) + 1
    n_cols = int(cols.max()) + 1
    w = cfg.w
    t0 = time.time()
    print(f"=== {label}: rate={cfg.build().rate:.3f} @ {snr}dB  E={E} mb={mb} w={w} ===", flush=True)

    # --- candidate cutting vectors -----------------------------------------
    uniform = _uniform_cuts(w, mb, n_cols)
    cands = [uniform] + _scalar_grid_cuts(w, mb, n_cols) + _random_percol_cuts(w, mb, n_cols, N_RANDOM_CV)
    # dedup by assign signature
    seen = {}
    for cuts in cands:
        a = _cv_assign(rows, cols, cuts)
        seen.setdefault(a.tobytes(), (cuts, a))
    uniq = list(seen.values())
    print(f"  {len(uniq)} unique cutting-vector constructions", flush=True)

    # --- n4 for ALL candidates (cheap, parallel) ---------------------------
    n4s = pool.starmap(_n4_of, [(cfg, a) for _, a in uniq])
    order = np.argsort(n4s)

    # cv_uniform
    uni_assign = _cv_assign(rows, cols, uniform)
    cv_uniform = _eval(cfg, uni_assign, snr, final, pool)

    # cv_best_n4 = min-n4 candidate (literature: CV chosen for cycle properties)
    bi = int(order[0])
    cv_best_n4 = _eval(cfg, uniq[bi][1], snr, final, pool)

    # cv_best_fer = best validation FER among the lowest-n4 shortlist, re-eval champ
    short = [int(i) for i in order[:FER_SHORTLIST]]
    val = R.eval_assignments(cfg, [uniq[i][1] for i in short], snr, VAL_SEED, VAL_FRAMES, pool=pool)
    vbest = short[int(np.argmin([m["fer"] for m in val]))]
    cv_best_fer = _eval(cfg, uniq[vbest][1], snr, final, pool)

    out = {"label": label, "cfg": _cfg_d(cfg), "snr": snr, "final": final,
           "rate": cfg.build().rate, "mb": mb, "E": E, "w": w,
           "n_candidates": len(uniq), "n4_min": int(min(n4s)), "n4_max": int(max(n4s)),
           "cv_uniform": cv_uniform, "cv_best_n4": cv_best_n4, "cv_best_fer": cv_best_fer,
           "sec": round(time.time() - t0, 1)}
    json.dump(out, open(f"results_cutvec_{label}.json", "w"), indent=1)
    print(f"  cv_uniform  FER={cv_uniform['fer']:.3f} n4={cv_uniform['n4']} girth={cv_uniform['girth']}", flush=True)
    print(f"  cv_best_n4  FER={cv_best_n4['fer']:.3f} n4={cv_best_n4['n4']} girth={cv_best_n4['girth']}", flush=True)
    print(f"  cv_best_fer FER={cv_best_fer['fer']:.3f} n4={cv_best_fer['n4']} girth={cv_best_fer['girth']}  ({out['sec']}s)", flush=True)
    return out


def aggregate():
    print("\n=== CPU#2: literature CUTTING-VECTOR baseline vs RL/CEM/PEG (FER) ===")
    print(f"{'cell':15} {'rate':>5} {'cv_unif':>8} {'cv_bN4':>8} {'cv_bFER':>8}   (compare to tab:ci RL/CEM/PEG)")
    for label in CELLS:
        f = f"results_cutvec_{label}.json"
        if not os.path.exists(f):
            print(f"{label:15} -- not run --"); continue
        d = json.load(open(f))
        print(f"{label:15} {d['rate']:>5.3f} {d['cv_uniform']['fer']:>8.3f} "
              f"{d['cv_best_n4']['fer']:>8.3f} {d['cv_best_fer']['fer']:>8.3f}   "
              f"[n4 {d['cv_best_n4']['n4']}/{d['cv_best_fer']['n4']}, "
              f"cand {d['n_candidates']}, n4range {d['n4_min']}-{d['n4_max']}]")


def _cfg_d(cfg):
    from dataclasses import asdict
    return asdict(cfg)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "agg":
        aggregate(); return
    nw = R.n_workers()
    print(f"workers={nw}", flush=True)
    with Pool(nw) as pool:
        if cmd == "run":
            for label in CELLS:
                if os.path.exists(f"results_cutvec_{label}.json"):
                    print(f"[{label}] cached, skip", flush=True); continue
                run_cell(label, pool)
        elif cmd == "cell":
            run_cell(sys.argv[2], pool)
        else:
            raise SystemExit("usage: experiments_cutvec.py [run|cell <label>|agg]")
    aggregate()


if __name__ == "__main__":
    main()
