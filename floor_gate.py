"""GO/NO-GO gate for the RL-SC-LDPC research program.

Premise under test
------------------
For SC-LDPC, the edge-spreading CONSTRUCTION (which systematic base edge goes to
which coupling component B_0..B_w) barely moves the BP waterfall / threshold, but
moves the finite-length ERROR FLOOR (absorbing-set / trapping-set structure) by
orders of magnitude.  If true, the right objective for "RL-optimised SC-LDPC
construction" is the error floor (via absorbing sets), NOT the waterfall -- which
is why prior waterfall-based RL found RL approx CEM.

What this script does, for a single FIXED small code (BG1, Z=16, w=3, L=24,
mp=9 -> R~0.73), over ~55 constructions (round_robin + ~50 random + cem_waterfall
+ cem_floor):
  1. WATERFALL: required Eb/N0 @ BER=1e-3 (interp a short waterfall).  Spread should
     be SMALL.
  2. HARMFUL-STRUCTURE SPECTRUM of the lifted coupled Tanner graph:
       - short-cycle spectrum: #4-cycles, #6-cycles, #8-cycles, girth
       - a bounded (a,b) ELEMENTARY ABSORBING-SET count for small a (see
         `absorbing_set_spectrum` docstring for the exact definition / enumerator).
     Spread should be LARGE (orders of magnitude).
  3. ERROR FLOOR: BER at several high Eb/N0 (4.0 / 4.5 / 5.0 dB, past the waterfall).
     Spread should be LARGE.

Key analyses:
  - waterfall spread (dB)  vs  floor BER spread (orders of magnitude)
  - Spearman( absorbing-set count , floor BER ) across constructions
  - does cem_floor beat round_robin / random / cem_waterfall ON THE FLOOR?

numpy-only for the heavy path; scipy only for the final Spearman.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import argparse
import numpy as np
import multiprocessing as mp

from rl_construct import (Config, round_robin_assign, random_assign, eval_assignments,
                          train_cem, n_workers, count_4cycles, girth)
from sc_ldpc import SCLDPCCode


# --------------------------------------------------------------------------- #
#  the fixed operating point
# --------------------------------------------------------------------------- #
def make_cfg():
    # BG1, Z=16, w=3, L=24, mid-high rate (mp=9 -> R~0.73), window W=max(6,w+2)=6,
    # max_iter=20.  Short -> MC-reachable floor ~1e-6..1e-7.
    return Config(bg=1, ils=0, Z=16, mp=9, w=3, L=24, W=max(6, 3 + 2), max_iter=20,
                  alpha=0.8)


# =========================================================================== #
#  GRAPH STRUCTURE METRICS
# =========================================================================== #
def _bipartite_adj(sc):
    """Return per-variable check-neighbour lists and per-check variable-neighbour
    lists of the lifted coupled Tanner graph (simple graph; QC lifts give no
    parallel edges between a given (var,check) pair)."""
    tan = sc.full_tanner()
    var_chk = [[] for _ in range(tan.num_var)]
    chk_var = [[] for _ in range(tan.num_chk)]
    for c, v in zip(tan.e_chk.tolist(), tan.e_var.tolist()):
        var_chk[v].append(c)
        chk_var[c].append(v)
    return tan, var_chk, chk_var


def short_cycle_spectrum(sc):
    """Exact #4-cycles and #6-cycles, plus girth, of the lifted Tanner graph.

    #4-cycles : sum over variable pairs of C(#shared checks, 2)  (= count_4cycles).
    #6-cycles : number of 6-cycles (v0-c0-v1-c1-v2-c2-v0) counted via length-2
                var-var walks.  We count ordered closed walks of length 6 that are
                genuine 6-cycles (3 distinct vars, 3 distinct checks) and divide by
                the symmetry factor 12 (6 rotations x 2 directions).
    girth     : shortest cycle length (BFS from variable nodes; capped at 12).
    """
    tan, var_chk, chk_var = _bipartite_adj(sc)
    nv = tan.num_var
    # var-var adjacency via shared checks, with multiplicity = #shared checks
    # build dict {(u<v): m}
    pair_m = {}
    for vs in chk_var:
        vs = sorted(set(vs))
        for i in range(len(vs)):
            ui = vs[i]
            for j in range(i + 1, len(vs)):
                k = (ui, vs[j])
                pair_m[k] = pair_m.get(k, 0) + 1
    n4 = int(sum(m * (m - 1) // 2 for m in pair_m.values()))
    # ---- 6-cycles : triangles in the var-var multigraph weighted by shared-check
    # multiplicities, EXCLUDING degenerate ones that reuse a check.
    # Build adjacency lists of the var-var graph with multiplicities and the
    # *set of shared checks* per pair so we can ensure the 3 checks are distinct.
    shared = {}
    for c, vs in enumerate(chk_var):
        vs = sorted(set(vs))
        for i in range(len(vs)):
            ui = vs[i]
            for j in range(i + 1, len(vs)):
                shared.setdefault((ui, vs[j]), []).append(c)
    adj = {}
    for (u, v) in shared:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    def sc_set(a, b):
        return shared[(a, b)] if (a, b) in shared else shared[(b, a)]

    n6 = 0
    nodes = sorted(adj)
    for u in nodes:
        nu = [w for w in adj[u] if w > u]
        for ii in range(len(nu)):
            a = nu[ii]
            for jj in range(ii + 1, len(nu)):
                b = nu[jj]
                if b not in adj[a]:
                    continue
                # triangle u<a<b ; each edge can use any of its shared checks, but
                # the three checks must be distinct (else the "cycle" folds to a
                # shorter one).  Count distinct (c_ua, c_ub, c_ab) triples.
                cua, cub, cab = sc_set(u, a), sc_set(u, b), sc_set(a, b)
                cnt = 0
                for x in cua:
                    for y in cub:
                        if y == x:
                            continue
                        for z in cab:
                            if z == x or z == y:
                                continue
                            cnt += 1
                n6 += cnt
    g = girth(sc, max_g=12, n_seeds=None)
    return {"n4": n4, "n6": int(n6), "girth": int(g)}


def absorbing_set_spectrum(sc, a_max=6, beam=4000, seed=0):
    """Bounded enumeration of small (a,b) ELEMENTARY absorbing sets of the lifted
    coupled Tanner graph, the dominant finite-length error-floor structures.

    Definition (Dolecek et al.; standard).  Let D be a set of a variable nodes.
    For the subgraph induced by D and its neighbouring checks, split the neighbour
    checks by their degree *into D*:
        O(D) = checks with an ODD number of neighbours in D  (unsatisfied), |O|=b
        E(D) = checks with an EVEN (>=2) number of neighbours in D (satisfied)
    D is an (a,b) ABSORBING SET iff every v in D has strictly MORE neighbours in
    E(D) than in O(D)  (so bit-flipping on D is a stable fixed point of the
    decoder -- each variable "sees" a majority of satisfied checks).  D is
    ELEMENTARY iff every neighbouring check has degree 1 or 2 into D (the
    structures that dominate LDPC floors).

    Enumerator (bounded, justified): we grow CONNECTED elementary candidate sets.
    A check that already has degree 2 into D is "closed"; degree-1 checks are the
    only legal expansion handles (adding a var through a closed check would make a
    degree-3 check -> non-elementary, pruned).  We expand breadth-first up to size
    a_max keeping at most `beam` candidate sets per size (ranked by fewest
    unsatisfied checks b, then by most satisfied checks -- i.e. the most
    "absorbing-like" partial sets), and at each size record those candidates that
    satisfy the full absorbing condition.  This is the well-known elementary-AS
    growth heuristic; with a beam it is a lower-bound *count of distinct small
    elementary absorbing sets found* (we de-duplicate by the frozenset of vars).
    We report, per construction, the total count for a in [4, a_max] and the size
    of the smallest absorbing set found (a proxy for the minimum distance of the
    floor-dominating structure).

    Returns dict: as_count (total small elementary AS found), as_by_a (per-size),
    as_min_a (smallest a with >=1 AS, or a_max+1 if none), as_min_b (b of that
    smallest set).
    """
    tan, var_chk, chk_var = _bipartite_adj(sc)
    var_chk = [np.array(x, dtype=np.int64) for x in var_chk]
    nv = tan.num_var
    rng = np.random.default_rng(seed)

    # A candidate is represented by (frozenset(vars), check_deg dict) but for speed
    # we carry: vars (tuple sorted), and a dict check->deg(into D) only for the
    # checks touched.  We expand via the degree-1 ("open") checks.
    # seed candidates: every single variable node.  (a=1 sets, then grow.)
    # To bound cost we seed from a random subset if nv is large, but here nv~1e4
    # and we cap the beam at each level so full seeding is fine.

    def check_degs(vars_):
        d = {}
        for v in vars_:
            for c in var_chk[v].tolist():
                d[c] = d.get(c, 0) + 1
        return d

    def is_absorbing(vars_, d):
        # elementary already guaranteed by construction (no deg>=3).
        # b = #odd checks; each var must have #even-nbrs > #odd-nbrs.
        odd = set(c for c, dd in d.items() if dd % 2 == 1)
        b = len(odd)
        for v in vars_:
            cs = var_chk[v]
            n_odd = int(sum(1 for c in cs.tolist() if c in odd))
            n_even = cs.size - n_odd
            if not (n_even > n_odd):
                return None
        return b

    # level a=1
    cur = []
    for v in range(nv):
        vs = (v,)
        d = {int(c): 1 for c in var_chk[v].tolist()}
        cur.append((vs, d))
    found = {}            # frozenset(vars) -> b , for accepted absorbing sets
    by_a = {a: 0 for a in range(1, a_max + 1)}

    seen_sets = set()
    for a in range(1, a_max + 1):
        # record absorbing sets at this size
        if a >= 3:        # absorbing sets of interest have a>=3 (a<3 elementary
                          # cannot satisfy the strict-majority condition for dv>=2)
            for vs, d in cur:
                fs = frozenset(vs)
                if fs in found:
                    continue
                b = is_absorbing(vs, d)
                if b is not None:
                    found[fs] = b
                    by_a[a] += 1
        if a == a_max:
            break
        # expand: from each candidate, for each OPEN (deg-1) check, add a new var
        # on that check (var not already in D), keeping elementary (no deg-3).
        nxt = {}
        for vs, d in cur:
            vset = set(vs)
            open_checks = [c for c, dd in d.items() if dd == 1]
            for c in open_checks:
                for v2 in chk_var[c]:
                    if v2 in vset:
                        continue
                    # check elementary: adding v2 must not push any check to deg 3
                    ok = True
                    for c2 in var_chk[v2].tolist():
                        if d.get(c2, 0) >= 2:
                            ok = False
                            break
                    if not ok:
                        continue
                    nvs = tuple(sorted(vs + (v2,)))
                    if nvs in seen_sets:
                        continue
                    seen_sets.add(nvs)
                    nd = dict(d)
                    for c2 in var_chk[v2].tolist():
                        nd[c2] = nd.get(c2, 0) + 1
                    # score: prefer fewer odd checks (closer to absorbing), then
                    # more closed (even) checks.
                    n_odd = sum(1 for x in nd.values() if x % 2 == 1)
                    n_even = sum(1 for x in nd.values() if x % 2 == 0)
                    nxt[nvs] = (nvs, nd, n_odd, -n_even)
        # beam-prune
        items = list(nxt.values())
        items.sort(key=lambda t: (t[2], t[3]))
        items = items[:beam]
        cur = [(t[0], t[1]) for t in items]
        if not cur:
            break

    total = sum(by_a[a] for a in range(3, a_max + 1))
    min_a = a_max + 1
    min_b = -1
    for fs, b in found.items():
        if len(fs) >= 3 and len(fs) < min_a:
            min_a = len(fs)
            min_b = b
    return {"as_count": int(total),
            "as_by_a": {int(a): int(by_a[a]) for a in range(3, a_max + 1)},
            "as_min_a": int(min_a),
            "as_min_b": int(min_b)}


# =========================================================================== #
#  WATERFALL + FLOOR measurement (Monte-Carlo, parallel)
# =========================================================================== #
def measure_ber_curve(cfg, assigns, ebn0_list, n_frames, frame_seed, pool,
                      frame_chunks):
    """BER/FER for every assignment at every Eb/N0 (shared CRN frames per SNR).
    Returns array [n_assign, n_snr] of BER and of FER and of frame_err."""
    na, ns = len(assigns), len(ebn0_list)
    ber = np.zeros((na, ns))
    fer = np.zeros((na, ns))
    ferr = np.zeros((na, ns), dtype=np.int64)
    nfr = np.zeros((na, ns), dtype=np.int64)
    for si, ebn0 in enumerate(ebn0_list):
        res = eval_assignments(cfg, assigns, ebn0, frame_seed=frame_seed + si * 101,
                               n_frames=n_frames, pool=pool, frame_chunks=frame_chunks)
        for ai, m in enumerate(res):
            ber[ai, si] = m["ber"]
            fer[ai, si] = m["fer"]
            ferr[ai, si] = m["frame_err"]
            nfr[ai, si] = m["n_frames"]
    return ber, fer, ferr, nfr


def required_ebn0_at_ber(ebn0_list, ber_row, target=1e-3):
    """Linear interpolation in (Eb/N0, log10 BER) to find Eb/N0 at target BER."""
    x = np.asarray(ebn0_list, float)
    y = np.asarray(ber_row, float)
    # use only finite, positive BERs for the log interp
    good = y > 0
    if good.sum() < 2:
        return np.nan
    xs, ys = x[good], np.log10(y[good])
    lt = np.log10(target)
    # find a bracketing pair where ber crosses target
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - lt) * (y1 - lt) <= 0 and y1 != y0:
            f = (lt - y0) / (y1 - y0)
            return float(xs[i] + f * (xs[i + 1] - xs[i]))
    # no bracket: extrapolate from the two points nearest the target
    j = int(np.argmin(np.abs(ys - lt)))
    j2 = min(max(j, 1), len(xs) - 1)
    y0, y1 = ys[j2 - 1], ys[j2]
    if y1 == y0:
        return float(xs[j2])
    f = (lt - y0) / (y1 - y0)
    return float(xs[j2 - 1] + f * (xs[j2] - xs[j2 - 1]))


# =========================================================================== #
#  MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_random", type=int, default=50)
    ap.add_argument("--wf_frames", type=int, default=600,
                    help="frames per SNR for the (cheap) waterfall sweep")
    ap.add_argument("--floor_frames", type=int, default=8000,
                    help="frames per SNR for the floor sweep")
    ap.add_argument("--cem_steps", type=int, default=20)
    ap.add_argument("--cem_batch", type=int, default=24)
    ap.add_argument("--cem_frames", type=int, default=120)
    ap.add_argument("--a_max", type=int, default=6, help="max absorbing-set size a")
    ap.add_argument("--beam", type=int, default=4000)
    ap.add_argument("--out_prefix", type=str, default="floor_gate")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    if args.quick:
        args.n_random = 4
        args.wf_frames = 120
        args.floor_frames = 300
        args.cem_steps = 3
        args.cem_batch = 8
        args.cem_frames = 40

    cfg = make_cfg()
    nw = n_workers()
    sc0 = cfg.build(seed=0)
    print(f"[cfg] BG1 Z={cfg.Z} w={cfg.w} L={cfg.L} mp={cfg.mp} W={cfg.W} "
          f"max_iter={cfg.max_iter} R={sc0.rate:.4f} K={sc0.K} N_tx={sc0.N_tx} "
          f"n_sys_edges={sc0.n_sys_edges} num_var={sc0.num_var}", flush=True)
    print(f"[workers] {nw}", flush=True)

    WF_SNRS = [2.5, 3.0, 3.25, 3.5, 4.0]          # waterfall bracket (BER 1e-3 ~3.x)
    FLOOR_SNRS = [4.0, 4.5, 5.0]                    # floor (past the knee)
    t_start = time.time()

    pool = mp.get_context("spawn").Pool(nw)
    try:
        # ---- build constructions ---------------------------------------- #
        names, assigns = [], []
        names.append("round_robin"); assigns.append(round_robin_assign(cfg))
        rng = np.random.default_rng(12345)
        for s in range(args.n_random):
            names.append(f"random_{s}")
            assigns.append(random_assign(cfg, np.random.default_rng(1000 + s)))

        # cem_waterfall : minimise BER at a WATERFALL SNR (3.25 dB ~ near threshold)
        print("\n[CEM-waterfall] minimising FER at 3.25 dB (waterfall) ...", flush=True)
        _, best_wf = train_cem(cfg, ebn0=3.25, n_steps=args.cem_steps,
                               batch=args.cem_batch, frames=args.cem_frames,
                               pool=pool, val_frames=400, val_seed=4242,
                               seed=21, log_every=5, verbose=True)
        names.append("cem_waterfall"); assigns.append(best_wf["assign"])

        # cem_floor : minimise the harmful-structure metric (absorbing-set count).
        # CEM here optimises a CHEAP combinatorial reward (no channel sim), so it
        # demonstrates the floor is optimisable from the surrogate alone.
        print("\n[CEM-floor] minimising elementary-absorbing-set count ...", flush=True)
        best_floor = cem_floor_construction(cfg, n_steps=args.cem_steps,
                                             batch=args.cem_batch, a_max=args.a_max,
                                             beam=args.beam, pool=pool, seed=77)
        names.append("cem_floor"); assigns.append(best_floor)

        na = len(assigns)
        print(f"\n[total constructions] {na}", flush=True)

        # ---- 1. WATERFALL ----------------------------------------------- #
        print("\n[measure] waterfall sweep ...", flush=True)
        t0 = time.time()
        wf_ber, wf_fer, wf_ferr, wf_nfr = measure_ber_curve(
            cfg, assigns, WF_SNRS, args.wf_frames, frame_seed=900, pool=pool,
            frame_chunks=nw)
        req = np.array([required_ebn0_at_ber(WF_SNRS, wf_ber[ai], 1e-3)
                        for ai in range(na)])
        print(f"  waterfall sweep done in {time.time()-t0:.0f}s", flush=True)

        # ---- 2. STRUCTURE ----------------------------------------------- #
        print("\n[measure] structure spectra ...", flush=True)
        t0 = time.time()
        struct = []
        for ai in range(na):
            sc = cfg.build(assign=assigns[ai])
            cyc = short_cycle_spectrum(sc)
            ab = absorbing_set_spectrum(sc, a_max=args.a_max, beam=args.beam)
            struct.append({**cyc, **ab})
            if ai % 10 == 0 or ai == na - 1:
                print(f"  [{ai+1}/{na}] {names[ai]:14s} n4={cyc['n4']} "
                      f"n6={cyc['n6']} g={cyc['girth']} AS={ab['as_count']} "
                      f"minA={ab['as_min_a']}", flush=True)
        print(f"  structure done in {time.time()-t0:.0f}s", flush=True)

        # ---- 3. FLOOR --------------------------------------------------- #
        print("\n[measure] floor sweep ...", flush=True)
        t0 = time.time()
        fl_ber, fl_fer, fl_ferr, fl_nfr = measure_ber_curve(
            cfg, assigns, FLOOR_SNRS, args.floor_frames, frame_seed=5000, pool=pool,
            frame_chunks=nw)
        print(f"  floor sweep done in {time.time()-t0:.0f}s", flush=True)

    finally:
        pool.close(); pool.join()

    # ===================== ANALYSIS ===================== #
    # represent a "floor BER" as BER at the deepest SNR with at least one error
    # across the set, plus a censored value (use 0.5/n_frames as an upper bound
    # when a construction has zero errors at a given SNR).
    def floor_metric(si):
        b = fl_ber[:, si].copy()
        nf = fl_nfr[:, si].astype(float)
        K = sc0.K
        # censoring floor: if 0 errors, true BER < 0.5/(nf*K); use that as a proxy
        cens = 0.5 / (nf * K)
        b = np.where(b > 0, b, cens)
        return b

    deepest = len(FLOOR_SNRS) - 1
    # pick the deepest SNR where the WORST construction still has errors (so the
    # spread is a true measured spread, not censoring-limited)
    floor_si = deepest
    for si in range(len(FLOOR_SNRS) - 1, -1, -1):
        if (fl_ferr[:, si] > 0).sum() >= max(2, len(assigns) // 2):
            floor_si = si
            break
    floor_ber = floor_metric(floor_si)

    wf_spread = float(np.nanmax(req) - np.nanmin(req))
    floor_log_spread = float(np.log10(np.nanmax(floor_ber)) - np.log10(np.nanmin(floor_ber)))

    AS = np.array([s["as_count"] for s in struct], float)
    N4 = np.array([s["n4"] for s in struct], float)
    N6 = np.array([s["n6"] for s in struct], float)

    def spearman(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return float("nan")
        try:
            from scipy.stats import spearmanr
            return float(spearmanr(x[m], y[m]).statistic)
        except Exception:
            # rank-correlation fallback (Pearson on ranks)
            rx = np.argsort(np.argsort(x[m])).astype(float)
            ry = np.argsort(np.argsort(y[m])).astype(float)
            rx -= rx.mean(); ry -= ry.mean()
            den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
            return float((rx*ry).sum() / den) if den > 0 else float("nan")

    sp_as = spearman(AS, floor_ber)
    sp_n4 = spearman(N4, floor_ber)
    sp_n6 = spearman(N6, floor_ber)
    sp_as_log = spearman(AS, np.log10(floor_ber))

    idx = {n: i for i, n in enumerate(names)}
    def floorval(name):
        return float(floor_ber[idx[name]]) if name in idx else float("nan")

    rr = floorval("round_robin")
    rnd = np.array([floor_ber[idx[n]] for n in names if n.startswith("random_")])
    cwf = floorval("cem_waterfall")
    cfl = floorval("cem_floor")
    rnd_median = float(np.median(rnd))
    rnd_best = float(np.min(rnd))

    print("\n================ RESULTS ================", flush=True)
    print(f"operating point: BG1 Z={cfg.Z} w={cfg.w} L={cfg.L} mp={cfg.mp} "
          f"R={sc0.rate:.3f}, {na} constructions", flush=True)
    print(f"WATERFALL required Eb/N0@1e-3: "
          f"min={np.nanmin(req):.3f} max={np.nanmax(req):.3f} "
          f"SPREAD={wf_spread:.3f} dB", flush=True)
    print(f"FLOOR @ {FLOOR_SNRS[floor_si]} dB BER: "
          f"min={np.nanmin(floor_ber):.2e} max={np.nanmax(floor_ber):.2e} "
          f"SPREAD={floor_log_spread:.2f} orders", flush=True)
    print(f"Spearman(absorbing_set_count, floor_BER) = {sp_as:.3f}  "
          f"(log: {sp_as_log:.3f})", flush=True)
    print(f"Spearman(n4, floor_BER) = {sp_n4:.3f} ; Spearman(n6, floor_BER) = {sp_n6:.3f}",
          flush=True)
    print(f"floor BER  round_robin={rr:.2e}  random[median]={rnd_median:.2e} "
          f"random[best]={rnd_best:.2e}  cem_waterfall={cwf:.2e}  cem_floor={cfl:.2e}",
          flush=True)
    if cfl > 0 and rr > 0:
        print(f"  cem_floor vs round_robin: {rr/cfl:.1f}x lower; "
              f"vs random-median: {rnd_median/cfl:.1f}x ; "
              f"vs cem_waterfall: {cwf/cfl:.1f}x", flush=True)
    print(f"total runtime {time.time()-t_start:.0f}s", flush=True)

    # ---- persist ---------------------------------------------------------- #
    out = {
        "cfg": {"bg": 1, "Z": cfg.Z, "w": cfg.w, "L": cfg.L, "mp": cfg.mp,
                "W": cfg.W, "max_iter": cfg.max_iter, "rate": sc0.rate,
                "K": sc0.K, "N_tx": sc0.N_tx, "n_sys_edges": sc0.n_sys_edges},
        "wf_snrs": WF_SNRS, "floor_snrs": FLOOR_SNRS,
        "floor_si": floor_si, "floor_snr": FLOOR_SNRS[floor_si],
        "names": names,
        "assigns": [np.asarray(a).tolist() for a in assigns],
        "required_ebn0": req.tolist(),
        "wf_ber": wf_ber.tolist(), "wf_fer": wf_fer.tolist(),
        "wf_ferr": wf_ferr.tolist(), "wf_nfr": wf_nfr.tolist(),
        "floor_ber": fl_ber.tolist(), "floor_fer": fl_fer.tolist(),
        "floor_ferr": fl_ferr.tolist(), "floor_nfr": fl_nfr.tolist(),
        "floor_ber_metric": floor_ber.tolist(),
        "struct": struct,
        "summary": {
            "waterfall_spread_db": wf_spread,
            "floor_log_spread": floor_log_spread,
            "spearman_as_floor": sp_as,
            "spearman_as_floor_log": sp_as_log,
            "spearman_n4_floor": sp_n4,
            "spearman_n6_floor": sp_n6,
            "floor_round_robin": rr,
            "floor_random_median": rnd_median,
            "floor_random_best": rnd_best,
            "floor_cem_waterfall": cwf,
            "floor_cem_floor": cfl,
        },
    }
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(out, f, indent=1)
    # CSV
    import csv
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        hdr = (["name", "required_ebn0_1e-3", "n4", "n6", "girth", "as_count",
                "as_min_a", "as_min_b"]
               + [f"floor_ber_{s}" for s in FLOOR_SNRS]
               + [f"floor_ferr_{s}" for s in FLOOR_SNRS]
               + ["floor_ber_metric"])
        wcsv.writerow(hdr)
        for ai in range(na):
            row = ([names[ai], req[ai], struct[ai]["n4"], struct[ai]["n6"],
                    struct[ai]["girth"], struct[ai]["as_count"],
                    struct[ai]["as_min_a"], struct[ai]["as_min_b"]]
                   + [fl_ber[ai, si] for si in range(len(FLOOR_SNRS))]
                   + [fl_ferr[ai, si] for si in range(len(FLOOR_SNRS))]
                   + [floor_ber[ai]])
            wcsv.writerow(row)
    print(f"[saved] {args.out_prefix}.json , {args.out_prefix}.csv", flush=True)

    # ---- plots (best-effort; matplotlib may be absent) -------------------- #
    try:
        make_plots(args.out_prefix, names, FLOOR_SNRS, fl_ber, fl_ferr, req,
                   AS, floor_ber, WF_SNRS, wf_ber, sp_as, wf_spread,
                   floor_log_spread, FLOOR_SNRS[floor_si])
        print(f"[saved] plots {args.out_prefix}_*.png", flush=True)
    except Exception as e:
        print(f"[plots skipped] {e}", flush=True)

    return out


def cem_floor_construction(cfg, n_steps, batch, a_max, beam, pool, seed=77):
    """CEM over per-edge categoricals minimising the elementary-absorbing-set count
    (a CHEAP combinatorial surrogate -- no channel sim).  Demonstrates the floor is
    optimisable from the structural surrogate alone.  Evaluates the AS metric of a
    batch of candidate assignments in parallel."""
    from rl_construct import edge_meta, n_components
    _, _, E = edge_meta(cfg)
    C = n_components(cfg)
    rng = np.random.default_rng(seed)
    p = np.full((E, C), 1.0 / C)
    n_elite = max(2, int(batch * 0.3))
    best = (np.inf, None)
    for step in range(1, n_steps + 1):
        assigns = [np.array([rng.choice(C, p=p[e]) for e in range(E)]) for _ in range(batch)]
        # parallel AS-count scoring
        tasks = [(cfg_to_dict(cfg), np.asarray(a, np.int64), a_max, beam) for a in assigns]
        scores = pool.map(_as_score_worker, tasks)
        scores = np.array(scores, float)
        elite = np.argsort(scores)[:n_elite]
        freq = np.zeros((E, C))
        for idx in elite:
            freq[np.arange(E), assigns[idx]] += 1.0
        freq /= n_elite
        p = 0.7 * p + 0.3 * freq
        p = np.clip(p, 1e-3, None); p /= p.sum(axis=1, keepdims=True)
        bi = int(np.argmin(scores))
        if scores[bi] < best[0]:
            best = (float(scores[bi]), np.asarray(assigns[bi]).copy())
        if step % 5 == 0 or step == 1:
            print(f"  [CEM-floor] step {step:3d}  batch min AS={scores.min():.0f} "
                  f"best AS={best[0]:.0f}", flush=True)
    return best[1]


def cfg_to_dict(cfg):
    from dataclasses import asdict
    return asdict(cfg)


def _as_score_worker(task):
    cfg_d, assign, a_max, beam = task
    cfg = Config(**cfg_d)
    sc = cfg.build(assign=assign)
    ab = absorbing_set_spectrum(sc, a_max=a_max, beam=beam)
    # tie-break by n4 so the surrogate is smooth when AS counts are equal/zero
    n4 = count_4cycles(sc)
    return ab["as_count"] * 1e6 + n4


def make_plots(prefix, names, floor_snrs, fl_ber, fl_ferr, req, AS, floor_ber,
               wf_snrs, wf_ber, sp_as, wf_spread, floor_log_spread, floor_snr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    na = len(names)
    # ---- plot 1: BER vs SNR for all constructions (floor curves) --------- #
    fig, ax = plt.subplots(figsize=(8, 6))
    allsnr = sorted(set(wf_snrs) | set(floor_snrs))
    for ai in range(na):
        nm = names[ai]
        if nm == "round_robin":
            style = dict(color="k", lw=2.5, marker="o", zorder=5, label="round_robin")
        elif nm == "cem_floor":
            style = dict(color="tab:green", lw=2.5, marker="s", zorder=6, label="cem_floor")
        elif nm == "cem_waterfall":
            style = dict(color="tab:red", lw=2.5, marker="^", zorder=6, label="cem_waterfall")
        else:
            style = dict(color="0.7", lw=0.8, alpha=0.6)
        # stitch wf + floor (use floor where overlapping)
        snrs = list(wf_snrs) + [s for s in floor_snrs if s not in wf_snrs]
        bers = list(wf_ber[ai]) + [fl_ber[ai, j] for j, s in enumerate(floor_snrs) if s not in wf_snrs]
        order = np.argsort(snrs)
        snrs = np.array(snrs)[order]; bers = np.array(bers)[order]
        # floor SNRs that ARE in wf: prefer floor measurement (more frames)
        for j, s in enumerate(floor_snrs):
            if s in wf_snrs:
                k = list(snrs).index(s)
                bers[k] = fl_ber[ai, j] if fl_ber[ai, j] > 0 else bers[k]
        bplot = np.where(bers > 0, bers, np.nan)
        ax.semilogy(snrs, bplot, **style)
    ax.axhline(1e-3, color="b", ls=":", alpha=0.5)
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("BER")
    ax.set_title(f"BER vs SNR, all {na} constructions (floor spread "
                 f"~{floor_log_spread:.1f} orders @ {floor_snr}dB)")
    ax.legend(loc="lower left", fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_ber_curves.png", dpi=130); plt.close(fig)

    # ---- plot 2: waterfall spread vs floor spread ------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(req[np.isfinite(req)], bins=20, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("required Eb/N0 @ BER=1e-3 (dB)")
    axes[0].set_ylabel("# constructions")
    axes[0].set_title(f"WATERFALL location (spread {wf_spread:.2f} dB)")
    lb = np.log10(floor_ber)
    axes[1].hist(lb[np.isfinite(lb)], bins=20, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel(f"log10 floor BER @ {floor_snr} dB")
    axes[1].set_ylabel("# constructions")
    axes[1].set_title(f"FLOOR location (spread {floor_log_spread:.1f} orders)")
    fig.tight_layout(); fig.savefig(f"{prefix}_spread.png", dpi=130); plt.close(fig)

    # ---- plot 3: absorbing-set count vs floor BER scatter ---------------- #
    fig, ax = plt.subplots(figsize=(7, 6))
    col = []
    for nm in names:
        col.append("k" if nm == "round_robin" else
                   "tab:green" if nm == "cem_floor" else
                   "tab:red" if nm == "cem_waterfall" else "tab:blue")
    ax.scatter(AS, floor_ber, c=col, s=40, alpha=0.8, edgecolor="w", linewidth=0.5)
    for nm, x, y in zip(names, AS, floor_ber):
        if nm in ("round_robin", "cem_floor", "cem_waterfall"):
            ax.annotate(nm, (x, y), fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("elementary absorbing-set count (a<=%d)" % (AS.size and 6))
    ax.set_ylabel(f"floor BER @ {floor_snr} dB")
    ax.set_title(f"absorbing-set count vs floor BER  (Spearman={sp_as:.2f})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_as_scatter.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
