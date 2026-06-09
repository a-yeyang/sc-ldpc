"""Experiment A -- "Construction optimises the FLOOR".

Thesis (Q1/CCF-A target)
------------------------
RL / search optimisation of the SC-LDPC edge-spreading CONSTRUCTION minimises the
finite-length ERROR FLOOR (the absorbing-set / trapping-set population), and
*construction* is the dominant lever there -- the BP waterfall threshold is
~construction-invariant (which is why prior waterfall-based RL found RL approx CEM).

What this script does
---------------------
For a few small, floor-reachable codes (BG1, Z=16, w=3, L=24, mp in {24->R~0.5,
9->R~0.76}; window W=max(6,w+2)=6, max_iter=20), for each cell:

  1. Build FIVE constructions of the systematic edge-spreading `assign`:
       - round_robin   : deterministic balanced spreading e -> e mod (w+1).
       - random_best   : best of N_RANDOM i.i.d. random assigns, ranked by the
                         elementary-absorbing-set (AS) count (a cheap structural
                         metric -- NO channel sim).
       - greedy        : a classical PEG-like heuristic -- assign systematic edges
                         one at a time to the component that least increases the
                         (4-cycle, AS) harmful-structure count.
       - cem           : cross-entropy method over per-edge categoricals whose
                         reward is the AS count (cheap metric; CEM over structure).
       - rl            : FeaturePolicy REINFORCE whose terminal reward is -AS count
                         (cheap metric -> the RL loop is fast, ~30 steps x batch 24).

  2. MEASURE the error FLOOR: BER/FER for each construction at 3-4 HIGH Eb/N0 points
     (past the knee, [4.0, 4.5, 5.0, 5.5] dB) with enough frames to reach ~1e-5
     (a few thousand frames, all cores).  Also record the WATERFALL point
     (Eb/N0 @ BER~1e-3) to confirm it is ~construction-invariant.

  3. HEADLINE per cell:
       - table {construction -> AS count, floor BER, waterfall Eb/N0}
       - floor-BER reduction (orders of magnitude) of rl/cem/greedy vs round_robin
       - rl vs cem vs greedy on the floor (does RL beat the classical greedy + CEM?)
     Plots: floor BER curves per method per cell; AS-count vs floor-BER scatter.

Checkpointing
-------------
A results JSON (floor_opt_results.json) is rewritten after EVERY cell, so partial
results are pullable even if the run is interrupted.  A per-cell JSON + CSV + PNGs
are also written.  Progress is logged verbosely with timestamps.

numpy-only heavy path; matplotlib only for the (best-effort) plots; scipy only for
the optional Spearman.  Reuses floor_gate.py and rl_construct.py.
"""
from __future__ import annotations
import os
# pin BLAS to 1 thread/process BEFORE numpy import (parallelism is over constructions)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import argparse
import numpy as np
import multiprocessing as mp

from rl_construct import (Config, FeaturePolicy, round_robin_assign, random_assign,
                          edge_meta, n_components, n_workers, count_4cycles, _softmax)
from floor_gate import (short_cycle_spectrum, absorbing_set_spectrum,
                        measure_ber_curve, required_ebn0_at_ber, cfg_to_dict)


# --------------------------------------------------------------------------- #
#  cells: a few small, floor-reachable codes
# --------------------------------------------------------------------------- #
def make_cells():
    """(name, Config) list.  BG1, Z=16, w=3, L=24, W=max(6,w+2)=6, max_iter=20.
    mp=24 -> R~0.5 ; mp=9 -> R~0.76."""
    W = max(6, 3 + 2)
    return [
        ("R050_mp24", Config(bg=1, ils=0, Z=16, mp=24, w=3, L=24, W=W,
                             max_iter=20, alpha=0.8)),
        ("R076_mp9",  Config(bg=1, ils=0, Z=16, mp=9,  w=3, L=24, W=W,
                             max_iter=20, alpha=0.8)),
    ]


# =========================================================================== #
#  cheap structural reward: elementary-absorbing-set count (+ 4-cycle tiebreak)
# =========================================================================== #
def as_cost(sc, a_max, beam):
    """Harmful-structure cost: AS-count dominates, #4-cycles breaks ties so the
    surrogate is smooth when AS counts are equal (e.g. both zero)."""
    ab = absorbing_set_spectrum(sc, a_max=a_max, beam=beam)
    n4 = count_4cycles(sc)
    return ab["as_count"] * 1_000_000 + n4, ab


def _as_cost_worker(task):
    """Parallel worker: (cfg_dict, assign, a_max, beam) -> scalar cost."""
    cfg_d, assign, a_max, beam = task
    cfg = Config(**cfg_d)
    sc = cfg.build(assign=np.asarray(assign, np.int64))
    cost, _ = as_cost(sc, a_max, beam)
    return float(cost)


def _batch_costs(cfg, assigns, a_max, beam, pool):
    """Score a batch of assignments by the cheap AS cost, in parallel."""
    tasks = [(cfg_to_dict(cfg), np.asarray(a, np.int64), a_max, beam) for a in assigns]
    if pool is not None:
        return np.array(pool.map(_as_cost_worker, tasks), float)
    return np.array([_as_cost_worker(t) for t in tasks], float)


# =========================================================================== #
#  construction methods (all optimise the CHEAP AS cost, no channel sim)
# =========================================================================== #
def build_round_robin(cfg, **_):
    return round_robin_assign(cfg)


def build_random_best(cfg, n_random, a_max, beam, pool, seed=1000, **_):
    """Best of `n_random` random assignments by AS cost."""
    assigns = [random_assign(cfg, np.random.default_rng(seed + s)) for s in range(n_random)]
    costs = _batch_costs(cfg, assigns, a_max, beam, pool)
    bi = int(np.argmin(costs))
    return np.asarray(assigns[bi]), float(costs[bi])


def build_greedy(cfg, a_max, beam, pool, **_):
    """Classical PEG-like greedy.  Assign systematic edges one at a time (in the
    canonical row-major order); each edge goes to the component that least increases
    the harmful-structure count of the *partially-built* coupled graph.

    The incremental cost is the full AS cost (with #4-cycle tiebreak) of the code
    obtained by fixing the already-placed edges + the trial placement and leaving the
    not-yet-placed edges at component 0 (a fixed continuation).  Each step evaluates
    the C candidate placements in parallel; partial assigns build a valid SC code at
    every step (every systematic entry always lands in exactly one component)."""
    _, _, E = edge_meta(cfg)
    C = n_components(cfg)
    assign = np.zeros(E, dtype=np.int64)          # not-yet-decided edges default to comp 0
    for e in range(E):
        cands = []
        for c in range(C):
            a = assign.copy()
            a[e] = c
            cands.append(a)
        costs = _batch_costs(cfg, cands, a_max, beam, pool)
        assign[e] = int(np.argmin(costs))
    # final cost
    final = _batch_costs(cfg, [assign], a_max, beam, pool)[0]
    return assign, float(final)


def build_cem(cfg, n_steps, batch, a_max, beam, pool, seed=77, log_every=5, **_):
    """CEM over per-edge categoricals minimising the AS cost (cheap structural
    surrogate, no channel sim)."""
    _, _, E = edge_meta(cfg)
    C = n_components(cfg)
    rng = np.random.default_rng(seed)
    p = np.full((E, C), 1.0 / C)
    n_elite = max(2, int(batch * 0.3))
    best = (np.inf, None)
    hist = []
    for step in range(1, n_steps + 1):
        assigns = [np.array([rng.choice(C, p=p[e]) for e in range(E)]) for _ in range(batch)]
        costs = _batch_costs(cfg, assigns, a_max, beam, pool)
        elite = np.argsort(costs)[:n_elite]
        freq = np.zeros((E, C))
        for idx in elite:
            freq[np.arange(E), assigns[idx]] += 1.0
        freq /= n_elite
        p = 0.7 * p + 0.3 * freq
        p = np.clip(p, 1e-3, None); p /= p.sum(axis=1, keepdims=True)
        bi = int(np.argmin(costs))
        if costs[bi] < best[0]:
            best = (float(costs[bi]), np.asarray(assigns[bi]).copy())
        hist.append(float(best[0]))
        if step % log_every == 0 or step == 1:
            print(f"    [CEM] step {step:3d}  batch min cost={costs.min():.0f}  "
                  f"best cost={best[0]:.0f}", flush=True)
    return best[1], best[0], hist


def build_rl(cfg, n_steps, batch, a_max, beam, pool, seed=21, lr=0.15, ent=0.02,
             log_every=5, **_):
    """FeaturePolicy REINFORCE whose terminal reward is -AS cost (cheap structural
    surrogate -> fast, no channel sim).  Uses a within-batch baseline + advantage
    normalisation (the same low-variance estimator used in rl_construct).  Greedy
    champion is re-scored each step and the best-cost greedy rollout is returned."""
    policy = FeaturePolicy(cfg, lr=lr, ent=ent, seed=seed)
    best = (np.inf, None)
    hist = []
    for step in range(1, n_steps + 1):
        traces, assigns = [], []
        for _ in range(batch):
            a, tr = policy.rollout()
            traces.append(tr); assigns.append(a)
        costs = _batch_costs(cfg, assigns, a_max, beam, pool)
        # reward = -cost ; within-batch baseline + advantage normalisation
        rewards = -costs
        adv = rewards - rewards.mean()
        if adv.std() > 1e-9:
            adv = adv / (adv.std() + 1e-9)
        policy.update(list(zip(traces, adv)))
        # champion: the greedy rollout of the current policy, scored on the cheap metric
        g = policy.greedy_assign()
        gcost = _batch_costs(cfg, [g], a_max, beam, pool)[0]
        # also consider the best sampled rollout this step
        bi = int(np.argmin(costs))
        for cand, cc in ((g, gcost), (np.asarray(assigns[bi]), float(costs[bi]))):
            if cc < best[0]:
                best = (float(cc), np.asarray(cand).copy())
        hist.append(float(best[0]))
        if step % log_every == 0 or step == 1:
            print(f"    [RL ] step {step:3d}  batch min cost={costs.min():.0f}  "
                  f"greedy cost={gcost:.0f}  best cost={best[0]:.0f}  "
                  f"theta=[{','.join(f'{t:+.2f}' for t in policy.theta)}]", flush=True)
    return best[1], best[0], hist


# =========================================================================== #
#  floor-BER summarisation
# =========================================================================== #
def floor_ber_metric(fl_ber, fl_nfr, K, si):
    """BER at floor SNR index `si`, with a censoring upper bound (0.5/(nf*K)) used
    when a construction recorded ZERO errors (true BER below that bound)."""
    b = fl_ber[:, si].astype(float).copy()
    nf = fl_nfr[:, si].astype(float)
    cens = 0.5 / np.maximum(nf * K, 1.0)
    return np.where(b > 0, b, cens)


def pick_floor_si(fl_ferr, floor_snrs, na):
    """Deepest floor SNR where at least max(2, na//2) constructions still show an
    error (so the spread is a measured spread, not censoring-limited)."""
    floor_si = len(floor_snrs) - 1
    for si in range(len(floor_snrs) - 1, -1, -1):
        if int((fl_ferr[:, si] > 0).sum()) >= max(2, na // 2):
            floor_si = si
            break
    return floor_si


# =========================================================================== #
#  per-cell driver
# =========================================================================== #
METHOD_ORDER = ["round_robin", "random_best", "greedy", "cem", "rl"]


def run_cell(cell_name, cfg, args, pool):
    t_cell = time.time()
    sc0 = cfg.build(seed=0)
    print(f"\n========== CELL {cell_name} ==========", flush=True)
    print(f"[cfg] BG1 Z={cfg.Z} w={cfg.w} L={cfg.L} mp={cfg.mp} W={cfg.W} "
          f"max_iter={cfg.max_iter} R={sc0.rate:.4f} K={sc0.K} N_tx={sc0.N_tx} "
          f"n_sys_edges={sc0.n_sys_edges} num_var={sc0.num_var} num_chk={sc0.num_chk}",
          flush=True)

    a_max, beam = args.a_max, args.beam

    # ---- build the five constructions (cheap structural optimisation) ------- #
    names, assigns, costs, build_hist = [], [], [], {}

    print("\n[build] round_robin ...", flush=True)
    a_rr = build_round_robin(cfg)
    c_rr = _batch_costs(cfg, [a_rr], a_max, beam, pool)[0]
    names.append("round_robin"); assigns.append(np.asarray(a_rr)); costs.append(float(c_rr))

    print(f"[build] random_best (best of {args.n_random}) ...", flush=True)
    a_rb, c_rb = build_random_best(cfg, args.n_random, a_max, beam, pool)
    names.append("random_best"); assigns.append(a_rb); costs.append(c_rb)

    print("[build] greedy (PEG-like incremental) ...", flush=True)
    t0 = time.time()
    a_gr, c_gr = build_greedy(cfg, a_max, beam, pool)
    print(f"        greedy done in {time.time()-t0:.0f}s  cost={c_gr:.0f}", flush=True)
    names.append("greedy"); assigns.append(a_gr); costs.append(c_gr)

    print(f"[build] cem ({args.opt_steps} steps x batch {args.opt_batch}) ...", flush=True)
    a_cem, c_cem, h_cem = build_cem(cfg, args.opt_steps, args.opt_batch, a_max, beam, pool)
    names.append("cem"); assigns.append(a_cem); costs.append(c_cem); build_hist["cem"] = h_cem

    print(f"[build] rl ({args.opt_steps} steps x batch {args.opt_batch}) ...", flush=True)
    a_rl, c_rl, h_rl = build_rl(cfg, args.opt_steps, args.opt_batch, a_max, beam, pool)
    names.append("rl"); assigns.append(a_rl); costs.append(c_rl); build_hist["rl"] = h_rl

    na = len(assigns)

    # ---- full structure spectra (AS count, 4/6-cycles, girth, min-a) -------- #
    print("\n[measure] structure spectra ...", flush=True)
    t0 = time.time()
    struct = []
    for ai in range(na):
        sc = cfg.build(assign=assigns[ai])
        cyc = short_cycle_spectrum(sc)
        ab = absorbing_set_spectrum(sc, a_max=a_max, beam=beam)
        struct.append({**cyc, **ab})
        print(f"  {names[ai]:12s} n4={cyc['n4']:5d} n6={cyc['n6']:7d} g={cyc['girth']:2d} "
              f"AS={ab['as_count']:4d} minA={ab['as_min_a']} minB={ab['as_min_b']}",
              flush=True)
    print(f"  structure done in {time.time()-t0:.0f}s", flush=True)

    # ---- waterfall sweep (confirm ~invariance) ------------------------------ #
    print("\n[measure] waterfall sweep ...", flush=True)
    t0 = time.time()
    wf_ber, wf_fer, wf_ferr, wf_nfr = measure_ber_curve(
        cfg, assigns, args.wf_snrs, args.wf_frames, frame_seed=900, pool=pool,
        frame_chunks=args.workers)
    req = np.array([required_ebn0_at_ber(args.wf_snrs, wf_ber[ai], 1e-3)
                    for ai in range(na)])
    print(f"  waterfall done in {time.time()-t0:.0f}s", flush=True)

    # ---- floor sweep (the headline) ---------------------------------------- #
    print(f"\n[measure] FLOOR sweep ({args.floor_frames} frames/SNR) ...", flush=True)
    t0 = time.time()
    fl_ber, fl_fer, fl_ferr, fl_nfr = measure_ber_curve(
        cfg, assigns, args.floor_snrs, args.floor_frames, frame_seed=5000, pool=pool,
        frame_chunks=args.workers)
    print(f"  floor done in {time.time()-t0:.0f}s", flush=True)

    # ===================== analysis ===================== #
    floor_si = pick_floor_si(fl_ferr, args.floor_snrs, na)
    floor_ber = floor_ber_metric(fl_ber, fl_nfr, sc0.K, floor_si)

    AS = np.array([s["as_count"] for s in struct], float)
    idx = {n: i for i, n in enumerate(names)}

    def fv(name):
        return float(floor_ber[idx[name]]) if name in idx else float("nan")

    rr, rb, gr, cm, rl = (fv("round_robin"), fv("random_best"), fv("greedy"),
                          fv("cem"), fv("rl"))
    wf_spread = float(np.nanmax(req) - np.nanmin(req))
    floor_log_spread = float(np.log10(np.nanmax(floor_ber)) - np.log10(np.nanmin(floor_ber)))

    def spearman(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return float("nan")
        try:
            from scipy.stats import spearmanr
            return float(spearmanr(x[m], y[m]).statistic)
        except Exception:
            rx = np.argsort(np.argsort(x[m])).astype(float)
            ry = np.argsort(np.argsort(y[m])).astype(float)
            rx -= rx.mean(); ry -= ry.mean()
            den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
            return float((rx * ry).sum() / den) if den > 0 else float("nan")

    sp_as = spearman(AS, np.log10(floor_ber))

    def redux(x):
        return (rr / x) if (x and x > 0 and rr > 0) else float("nan")

    print("\n  ---- CELL SUMMARY ----", flush=True)
    print(f"  {'method':12s} {'AS':>5s} {'n4':>5s} {'floorBER':>11s} {'reqEb/N0@1e-3':>13s}",
          flush=True)
    for ai in range(na):
        print(f"  {names[ai]:12s} {struct[ai]['as_count']:5d} {struct[ai]['n4']:5d} "
              f"{floor_ber[ai]:11.2e} {req[ai]:13.3f}", flush=True)
    print(f"  waterfall spread = {wf_spread:.3f} dB (expect SMALL / ~invariant)", flush=True)
    print(f"  floor spread     = {floor_log_spread:.2f} orders @ {args.floor_snrs[floor_si]} dB",
          flush=True)
    print(f"  Spearman(AS, log10 floorBER) = {sp_as:.3f}", flush=True)
    print(f"  floor-BER REDUCTION vs round_robin:  rl={redux(rl):.1f}x  "
          f"cem={redux(cm):.1f}x  greedy={redux(gr):.1f}x  random_best={redux(rb):.1f}x",
          flush=True)
    print(f"  rl vs cem: {(cm/rl if rl>0 else float('nan')):.2f}x ; "
          f"rl vs greedy: {(gr/rl if rl>0 else float('nan')):.2f}x  "
          f"(>1 => RL wins on the floor)", flush=True)
    print(f"  cell runtime {time.time()-t_cell:.0f}s", flush=True)

    cell_out = {
        "cell": cell_name,
        "cfg": {"bg": cfg.bg, "ils": cfg.ils, "Z": cfg.Z, "w": cfg.w, "L": cfg.L,
                "mp": cfg.mp, "W": cfg.W, "max_iter": cfg.max_iter, "alpha": cfg.alpha,
                "rate": sc0.rate, "K": sc0.K, "N_tx": sc0.N_tx,
                "n_sys_edges": sc0.n_sys_edges, "num_var": sc0.num_var,
                "num_chk": sc0.num_chk},
        "names": names,
        "assigns": [np.asarray(a).tolist() for a in assigns],
        "as_costs": costs,
        "struct": struct,
        "wf_snrs": list(args.wf_snrs), "floor_snrs": list(args.floor_snrs),
        "required_ebn0": req.tolist(),
        "wf_ber": wf_ber.tolist(), "wf_fer": wf_fer.tolist(),
        "wf_ferr": wf_ferr.tolist(), "wf_nfr": wf_nfr.tolist(),
        "floor_ber": fl_ber.tolist(), "floor_fer": fl_fer.tolist(),
        "floor_ferr": fl_ferr.tolist(), "floor_nfr": fl_nfr.tolist(),
        "floor_si": floor_si, "floor_snr": args.floor_snrs[floor_si],
        "floor_ber_metric": floor_ber.tolist(),
        "build_hist": build_hist,
        "summary": {
            "waterfall_spread_db": wf_spread,
            "floor_log_spread": floor_log_spread,
            "spearman_as_logfloor": sp_as,
            "floor_round_robin": rr, "floor_random_best": rb, "floor_greedy": gr,
            "floor_cem": cm, "floor_rl": rl,
            "redux_rl_vs_rr": redux(rl), "redux_cem_vs_rr": redux(cm),
            "redux_greedy_vs_rr": redux(gr), "redux_random_best_vs_rr": redux(rb),
            "rl_vs_cem": (cm / rl if rl > 0 else float("nan")),
            "rl_vs_greedy": (gr / rl if rl > 0 else float("nan")),
        },
        "runtime_s": time.time() - t_cell,
    }

    # per-cell CSV + plots (best-effort)
    try:
        write_cell_csv(args.out_prefix, cell_name, names, struct, req, fl_ber,
                       fl_ferr, floor_ber, args.floor_snrs)
    except Exception as e:
        print(f"  [csv skipped] {e}", flush=True)
    try:
        make_cell_plots(args.out_prefix, cell_name, names, args.wf_snrs,
                        args.floor_snrs, wf_ber, fl_ber, AS, floor_ber, req,
                        sp_as, wf_spread, floor_log_spread, args.floor_snrs[floor_si])
        print(f"  [saved] plots {args.out_prefix}_{cell_name}_*.png", flush=True)
    except Exception as e:
        print(f"  [plots skipped] {e}", flush=True)

    return cell_out


# =========================================================================== #
#  output: CSV + plots
# =========================================================================== #
def write_cell_csv(prefix, cell_name, names, struct, req, fl_ber, fl_ferr,
                   floor_ber, floor_snrs):
    import csv
    fn = f"{prefix}_{cell_name}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        hdr = (["method", "as_count", "as_min_a", "as_min_b", "n4", "n6", "girth",
                "required_ebn0_1e-3"]
               + [f"floor_ber_{s}" for s in floor_snrs]
               + [f"floor_ferr_{s}" for s in floor_snrs]
               + ["floor_ber_metric"])
        w.writerow(hdr)
        for ai in range(len(names)):
            row = ([names[ai], struct[ai]["as_count"], struct[ai]["as_min_a"],
                    struct[ai]["as_min_b"], struct[ai]["n4"], struct[ai]["n6"],
                    struct[ai]["girth"], req[ai]]
                   + [fl_ber[ai, si] for si in range(len(floor_snrs))]
                   + [fl_ferr[ai, si] for si in range(len(floor_snrs))]
                   + [floor_ber[ai]])
            w.writerow(row)


_STYLE = {
    "round_robin": dict(color="k", lw=2.4, marker="o", zorder=5),
    "random_best": dict(color="tab:purple", lw=2.0, marker="v", zorder=5),
    "greedy": dict(color="tab:orange", lw=2.2, marker="D", zorder=6),
    "cem": dict(color="tab:red", lw=2.2, marker="^", zorder=6),
    "rl": dict(color="tab:green", lw=2.6, marker="s", zorder=7),
}


def make_cell_plots(prefix, cell_name, names, wf_snrs, floor_snrs, wf_ber, fl_ber,
                    AS, floor_ber, req, sp_as, wf_spread, floor_log_spread, floor_snr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    na = len(names)
    # ---- plot 1: stitched BER vs SNR (waterfall + floor) ------------------- #
    fig, ax = plt.subplots(figsize=(8, 6))
    for ai in range(na):
        nm = names[ai]
        st = _STYLE.get(nm, dict(color="0.6", lw=1.0))
        snrs = list(wf_snrs) + [s for s in floor_snrs if s not in wf_snrs]
        bers = list(wf_ber[ai]) + [fl_ber[ai, j] for j, s in enumerate(floor_snrs)
                                   if s not in wf_snrs]
        order = np.argsort(snrs)
        snrs = np.array(snrs)[order]; bers = np.array(bers)[order]
        # prefer the (higher-frame) floor measurement on overlapping SNRs
        for j, s in enumerate(floor_snrs):
            if s in wf_snrs and fl_ber[ai, j] > 0:
                bers[list(snrs).index(s)] = fl_ber[ai, j]
        ax.semilogy(snrs, np.where(bers > 0, bers, np.nan), label=nm, **st)
    ax.axhline(1e-3, color="b", ls=":", alpha=0.4)
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("BER")
    ax.set_title(f"{cell_name}: BER vs Eb/N0  (floor spread ~{floor_log_spread:.1f} "
                 f"orders @ {floor_snr} dB)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_{cell_name}_ber.png", dpi=130); plt.close(fig)

    # ---- plot 2: floor-BER bar chart per method ---------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(na)
    cols = [_STYLE.get(n, dict(color="0.6"))["color"] for n in names]
    ax.bar(xs, floor_ber, color=cols, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(f"floor BER @ {floor_snr} dB")
    ax.set_title(f"{cell_name}: floor BER per construction")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_{cell_name}_floorbar.png", dpi=130); plt.close(fig)

    # ---- plot 3: AS-count vs floor-BER scatter ----------------------------- #
    fig, ax = plt.subplots(figsize=(7, 6))
    for ai in range(na):
        nm = names[ai]
        c = _STYLE.get(nm, dict(color="0.6"))["color"]
        ax.scatter(AS[ai], floor_ber[ai], c=c, s=70, edgecolor="w", linewidth=0.6, zorder=5)
        ax.annotate(nm, (AS[ai], floor_ber[ai]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel(f"elementary absorbing-set count (a<={int(AS.size and 6)})")
    ax.set_ylabel(f"floor BER @ {floor_snr} dB")
    ax.set_title(f"{cell_name}: AS count vs floor BER  (Spearman={sp_as:.2f})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_{cell_name}_scatter.png", dpi=130); plt.close(fig)


# =========================================================================== #
#  MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_random", type=int, default=40,
                    help="random pool size for random_best")
    ap.add_argument("--opt_steps", type=int, default=30, help="RL/CEM steps")
    ap.add_argument("--opt_batch", type=int, default=24, help="RL/CEM batch")
    ap.add_argument("--a_max", type=int, default=6, help="max absorbing-set size a")
    ap.add_argument("--beam", type=int, default=1500, help="AS-enumeration beam width")
    ap.add_argument("--wf_frames", type=int, default=800,
                    help="frames/SNR for the waterfall sweep")
    ap.add_argument("--floor_frames", type=int, default=4000,
                    help="frames/SNR for the floor sweep")
    ap.add_argument("--wf_snrs", type=float, nargs="+",
                    default=[2.5, 3.0, 3.5, 4.0])
    ap.add_argument("--floor_snrs", type=float, nargs="+",
                    default=[4.0, 4.5, 5.0, 5.5])
    ap.add_argument("--out_prefix", type=str, default="floor_opt")
    ap.add_argument("--only_cell", type=str, default=None,
                    help="run only the named cell (debug)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    if args.quick:
        args.n_random = 5
        args.opt_steps = 3
        args.opt_batch = 8
        args.beam = 800
        args.wf_frames = 120
        args.floor_frames = 250
        args.wf_snrs = [3.0, 3.5, 4.0]
        args.floor_snrs = [4.0, 4.5, 5.0]

    args.workers = n_workers()
    cells = make_cells()
    if args.only_cell:
        cells = [(n, c) for (n, c) in cells if n == args.only_cell]
        if not cells:
            raise SystemExit(f"no cell named {args.only_cell}")

    t_start = time.time()
    print(f"[floor_opt] {len(cells)} cells, workers={args.workers}, "
          f"n_random={args.n_random}, opt {args.opt_steps}x{args.opt_batch}, "
          f"a_max={args.a_max}, beam={args.beam}, "
          f"wf_frames={args.wf_frames}, floor_frames={args.floor_frames}", flush=True)
    print(f"[floor_opt] wf_snrs={args.wf_snrs}  floor_snrs={args.floor_snrs}", flush=True)
    print(f"[floor_opt] start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    pool = mp.get_context("spawn").Pool(args.workers)
    all_cells = []
    try:
        for ci, (cell_name, cfg) in enumerate(cells):
            cell_out = run_cell(cell_name, cfg, args, pool)
            all_cells.append(cell_out)
            # ---- CHECKPOINT after EVERY cell ---- #
            checkpoint = {
                "meta": {
                    "n_random": args.n_random, "opt_steps": args.opt_steps,
                    "opt_batch": args.opt_batch, "a_max": args.a_max, "beam": args.beam,
                    "wf_frames": args.wf_frames, "floor_frames": args.floor_frames,
                    "wf_snrs": list(args.wf_snrs), "floor_snrs": list(args.floor_snrs),
                    "workers": args.workers,
                    "cells_done": ci + 1, "cells_total": len(cells),
                    "elapsed_s": time.time() - t_start,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "cells": all_cells,
            }
            with open(f"{args.out_prefix}_results.json", "w") as f:
                json.dump(checkpoint, f, indent=1)
            print(f"\n[CHECKPOINT] wrote {args.out_prefix}_results.json "
                  f"({ci+1}/{len(cells)} cells, {time.time()-t_start:.0f}s elapsed)\n",
                  flush=True)
    finally:
        pool.close(); pool.join()

    # ---- cross-cell headline -------------------------------------------- #
    print("\n================ HEADLINE (all cells) ================", flush=True)
    print(f"{'cell':12s} {'R':>5s} {'wf_spread_dB':>12s} {'floor_orders':>12s} "
          f"{'rl/rr':>7s} {'cem/rr':>7s} {'grd/rr':>7s} {'rl/cem':>7s} {'rl/grd':>7s}",
          flush=True)
    for c in all_cells:
        s = c["summary"]
        print(f"{c['cell']:12s} {c['cfg']['rate']:5.2f} "
              f"{s['waterfall_spread_db']:12.3f} {s['floor_log_spread']:12.2f} "
              f"{s['redux_rl_vs_rr']:7.1f} {s['redux_cem_vs_rr']:7.1f} "
              f"{s['redux_greedy_vs_rr']:7.1f} {s['rl_vs_cem']:7.2f} "
              f"{s['rl_vs_greedy']:7.2f}", flush=True)
    print("\n(reduction columns = x-fold floor-BER drop vs round_robin; "
          "rl/cem & rl/grd > 1 => RL wins on the floor)", flush=True)
    print(f"\ntotal runtime {time.time()-t_start:.0f}s "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    print(f"[DONE] results in {args.out_prefix}_results.json", flush=True)


if __name__ == "__main__":
    main()
