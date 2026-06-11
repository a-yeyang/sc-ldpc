"""S2 -- numerical validation of the combinatorial lemma:

  "Two systematic base-edges in the SAME base column assigned to the SAME component
   B_a produce 4-cycles in the lifted coupled Tanner graph; spreading a column's
   edges across distinct components removes the dominant 4-cycle population."

We make this quantitative.  For a construction with per-edge component assignment
`assign`, define the same-column co-assignment predictor

    P_col = sum_{column c} sum_{component a} C(k_{c,a}, 2),
            k_{c,a} = # systematic edges in column c assigned to component a,

i.e. the number of same-column edge PAIRS that share a component (the lemma's
mechanism).  Control predictors P_row (same, by base ROW) and P_glob (by global
component load) isolate that it is the COLUMN structure that matters.

Claims validated here (all cheap, pure-CPU, exact via rl_construct.count_4cycles):
  (1) Spearman(P_col, n4) ~ 1, while Spearman(P_row, n4) and Spearman(P_glob, n4)
      are weak -> the column predictor is THE driver of the 4-cycle count.
  (2) A max-column-spread construction drives P_col -> 0 and n4 -> 0; round-robin /
      repo-default (seed0) leave a large same-column population and a large n4
      (reproducing the section-5.1 ~930 vs 0).
  (3) round-robin n4 scales with the lifting factor Z -> the lemma's count form.

Usage (pure NumPy; runs anywhere):
  python3 lemma_check.py run    # -> results_lemma.json + printed summary
"""
from __future__ import annotations
import json
import sys
import numpy as np

import rl_construct as R


# ---- dependency-free Spearman (rank then Pearson) --------------------------- #
def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


# ---- predictors ------------------------------------------------------------- #
def predictors(cols, rows, assign, C):
    assign = np.asarray(assign)
    P_col = P_row = 0
    for grp in (cols, rows):
        for g in np.unique(grp):
            cnt = np.bincount(assign[grp == g], minlength=C)
            val = int((cnt * (cnt - 1) // 2).sum())
            if grp is cols:
                P_col += val
            else:
                P_row += val
    glob = np.bincount(assign, minlength=C)
    P_glob = int((glob * (glob - 1) // 2).sum())
    return P_col, P_row, P_glob


def max_column_spread(cols):
    """Assign edge -> (its within-column index), maximising per-column component
    diversity; mod C applied by caller."""
    order = np.zeros(len(cols), dtype=np.int64)
    seen = {}
    for e in range(len(cols)):
        c = int(cols[e]); k = seen.get(c, 0); order[e] = k; seen[c] = k + 1
    return order


def _n4_girth(cfg, assign):
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return R.count_4cycles(sc), R.girth(sc, n_seeds=200)


# ---- experiment 1: correlation of predictors with measured n4 --------------- #
def correlation_study(cfg, n_random=80, label=""):
    rows, cols, E = R.edge_meta(cfg)
    C = R.n_components(cfg)
    rng = np.random.default_rng(0)
    samples = []
    # random constructions
    for _ in range(n_random):
        samples.append(("random", rng.integers(0, C, size=E)))
    # structured references
    samples.append(("round_robin", np.arange(E) % C))
    samples.append(("seed0_default", cfg.build(seed=0).assign.copy()
                    if hasattr(cfg.build(seed=0), "assign") else R.random_assign(cfg, np.random.default_rng(0))))
    samples.append(("col_spread", max_column_spread(cols) % C))
    P_col, P_row, P_glob, N4, GIRTH, names = [], [], [], [], [], []
    for name, a in samples:
        pc, pr, pg = predictors(cols, rows, a, C)
        n4, g = _n4_girth(cfg, a)
        P_col.append(pc); P_row.append(pr); P_glob.append(pg)
        N4.append(n4); GIRTH.append(g); names.append(name)
    P_col, P_row, P_glob, N4 = map(np.array, (P_col, P_row, P_glob, N4))
    res = {
        "label": label, "rate": cfg.build().rate, "E": int(E), "C": int(C),
        "n_samples": len(samples),
        "spearman_Pcol_n4": spearman(P_col, N4),
        "spearman_Prow_n4": spearman(P_row, N4),
        "spearman_Pglob_n4": spearman(P_glob, N4),
        "pearson_Pcol_n4_log": spearman(P_col, N4),   # (rank-based; robust)
        "refs": {name: {"P_col": int(pc), "n4": int(n4), "girth": int(g)}
                 for name, pc, n4, g in zip(names[-3:], P_col[-3:], N4[-3:], GIRTH[-3:])},
        "random_n4_mean": float(N4[:n_random].mean()),
        "random_n4_min": int(N4[:n_random].min()),
        "random_n4_max": int(N4[:n_random].max()),
        "random_Pcol_zero_frac": float((P_col[:n_random] == 0).mean()),
    }
    return res


# ---- experiment 2: round-robin n4 scaling with lifting factor Z -------------- #
def scaling_study(bg=2, mp=8, w=2, L=30, W=6, Zs=(16, 32, 64)):
    out = []
    for Z in Zs:
        cfg = R.Config(bg=bg, ils=0, Z=Z, mp=mp, w=w, L=L, W=W, max_iter=12)
        rows, cols, E = R.edge_meta(cfg)
        C = R.n_components(cfg)
        rr = np.arange(E) % C
        cs = max_column_spread(cols) % C
        n4_rr, g_rr = _n4_girth(cfg, rr)
        n4_cs, g_cs = _n4_girth(cfg, cs)
        pc_rr = predictors(cols, rows, rr, C)[0]
        out.append({"Z": Z, "E": int(E),
                    "round_robin": {"P_col": int(pc_rr), "n4": int(n4_rr), "girth": int(g_rr)},
                    "col_spread": {"n4": int(n4_cs), "girth": int(g_cs)}})
        print(f"  Z={Z}: round_robin n4={n4_rr} (P_col={pc_rr}, girth {g_rr})  "
              f"col_spread n4={n4_cs} (girth {g_cs})", flush=True)
    return out


def run():
    results = {"correlation": [], "scaling": []}
    grids = [
        ("BG2_Z16_mp8_w2", R.Config(bg=2, ils=0, Z=16, mp=8, w=2, L=30, W=6, max_iter=12)),
        ("BG1_Z16_mp8_w2", R.Config(bg=1, ils=0, Z=16, mp=8, w=2, L=30, W=6, max_iter=12)),
        ("BG2_Z16_mp12_w2", R.Config(bg=2, ils=0, Z=16, mp=12, w=2, L=30, W=6, max_iter=12)),
    ]
    print("=== correlation study (predictor vs measured n4) ===", flush=True)
    for label, cfg in grids:
        res = correlation_study(cfg, n_random=80, label=label)
        results["correlation"].append(res)
        print(f"[{label}] rate={res['rate']:.3f} E={res['E']}  "
              f"Spearman(P_col,n4)={res['spearman_Pcol_n4']:+.3f}  "
              f"(P_row {res['spearman_Prow_n4']:+.3f}, P_glob {res['spearman_Pglob_n4']:+.3f})  "
              f"random n4 mean={res['random_n4_mean']:.0f} [{res['random_n4_min']},{res['random_n4_max']}]",
              flush=True)
        for name, d in res["refs"].items():
            print(f"    {name:16s} P_col={d['P_col']:5d}  n4={d['n4']:5d}  girth={d['girth']}", flush=True)
    print("\n=== scaling study (round-robin n4 vs Z) ===", flush=True)
    results["scaling"] = scaling_study()
    json.dump(results, open("results_lemma.json", "w"), indent=2)
    print("\n-> results_lemma.json", flush=True)
    return results


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    else:
        raise SystemExit("usage: python3 lemma_check.py run")
