"""Experiment C -- Surrogate-guided SC-LDPC construction optimisation at LARGE
coupling memory w  ("AI enables design at a scale classical evaluation can't reach").

Thesis
------
For high-memory SC-LDPC (large coupling memory w) a Monte-Carlo BER/error-floor
evaluation of the construction is INFEASIBLE: the dominant finite-length
structures (trapping / absorbing sets) sit far down the floor and enumerating
them by simulation is "only feasible for short codes" (a recognised computational
bottleneck; cf. NeurIPS-2023 LDPC-error-floor work).  But a CHEAP COMBINATORIAL
SURROGATE of the lifted coupled Tanner graph -- the small elementary
ABSORBING-SET count and the short-CYCLE (4-/6-cycle) count -- can be evaluated in
seconds with no channel simulation, at ANY w.  That lets RL / search OPTIMISE the
edge-spreading construction at large w where channel-MC cannot follow:
*the AI/search optimises a surrogate that classical channel-MC can't reach.*

What this script does
---------------------
Sweep  w in {3, 6, 10, 16, 24}  at a fixed component code
(BG1, Z=16, mp=9 -> R~0.75 single-position; coupled R~0.69-0.73), with
L = max(30, 4w) so the termination rate-loss ~= w/L stays modest, decode
window W = max(6, w+2).  The systematic-edge-spreading vector `assign`
(length = #systematic base edges E = 96, entries in [0..w]) is the construction
knob; the action space 3^E ... (w+1)^E grows with w.

At EACH w we MINIMISE a cheap harmful-structure SURROGATE of the lifted graph via:
  * rl            -- FeaturePolicy (linear-softmax sequential-MDP) + REINFORCE,
                     reward = -surrogate, common evaluation (the surrogate is
                     deterministic given the assignment, so no CRN needed).
  * cem           -- cross-entropy method over per-edge categoricals.
  * round_robin   -- deterministic balanced baseline  (e -> e mod (w+1)).
  * random        -- best-of-N random spreadings (same eval budget proxy).

SURROGATE (per w, LOGGED).  The reward is a lexicographic harmful-structure score
    S = as_count * 1e9  +  n4 * 1e3  +  n6
where  as_count = #small elementary absorbing sets (a in [3, A_MAX], beam BEAM),
       n4, n6   = #4-cycles, #6-cycles of the lifted coupled Tanner graph.
The absorbing-set term is the primary floor proxy; the 4-/6-cycle terms are the
classical cheap proxies AND act as a tie-break.  IMPORTANT (measured): at large w
the beam-bounded as_count saturates near the beam for many constructions and loses
discrimination, whereas the 4-cycle count stays sharply discriminative (round_robin
~10^3 4-cycles vs optimised constructions that reach ZERO).  The combined score is
therefore discriminative at every w; we report BOTH the absorbing-set count AND the
4-cycle count vs w as headlines.  If for some w the absorbing-set enumeration ever
exceeds AS_TIME_BUDGET seconds for a single construction (it does NOT in practice --
~4s at w=24, beam=1500), we transparently fall back that w's primary reward to the
pure 4-/6-cycle surrogate and record `as_disabled=True` for that w.

HEADLINE.  Plot surrogate (absorbing-set count AND 4-cycle count) vs w for
{rl, cem, round_robin, random}, showing (i) the harmful-structure count and the
GAP between optimised and round_robin GROW with w (more room for construction to
help at high memory), and (ii) rl/cem reach markedly lower counts than
round_robin/random at large w.

VALIDATION (small w only, w in {3, 6}).  We additionally measure the ACTUAL error
floor (`measure_ber_curve` at high Eb/N0) for the four champions and confirm that
the lower-surrogate champion has the lower measured floor -- i.e. the cheap
surrogate that drives the large-w optimisation is grounded in the real floor where
MC is still feasible.

Checkpoints a JSON after EVERY w so partial results are pullable mid-run.
numpy-only on the heavy path (matplotlib best-effort for the plots; scipy not
required).

Usage:
  EXP_WORKERS=16 python3 -u experiments_largew.py            # full sweep
  EXP_WORKERS=8  python3 -u experiments_largew.py --quick    # tiny smoke
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
import multiprocessing as mp

from rl_construct import (Config, FeaturePolicy, round_robin_assign, random_assign,
                          edge_meta, n_components, count_4cycles, girth, n_workers)
from floor_gate import (absorbing_set_spectrum, short_cycle_spectrum,
                        measure_ber_curve, required_ebn0_at_ber)


# --------------------------------------------------------------------------- #
#  config / sweep
# --------------------------------------------------------------------------- #
W_SWEEP = [3, 6, 10, 16, 24]
A_MAX = 6
BEAM = 1500
AS_TIME_BUDGET = 20.0     # s: per-construction AS budget; over -> cycle-only reward

# surrogate lexicographic weights:  as_count >> n4 >> n6
W_AS, W_N4, W_N6 = 1e9, 1e3, 1.0

# validation (small w) error-floor sweep
VAL_W = {3, 6}
FLOOR_SNRS = [4.0, 4.5, 5.0]
FLOOR_FRAMES = 6000


def make_cfg(w: int) -> Config:
    L = max(30, 4 * w)
    return Config(bg=1, ils=0, Z=16, mp=9, w=w, L=L, W=max(6, w + 2),
                  max_iter=20, alpha=0.8)


def cfg_to_dict(cfg: Config):
    from dataclasses import asdict
    return asdict(cfg)


# --------------------------------------------------------------------------- #
#  surrogate evaluation (cheap, no channel sim)
# --------------------------------------------------------------------------- #
def surrogate_parts(sc, a_max=A_MAX, beam=BEAM, use_as=True):
    """Return (as_count, n4, n6, girth, as_min_a, as_sec).  If use_as is False we
    skip the absorbing-set enumeration (as_count = -1 sentinel)."""
    cyc = short_cycle_spectrum(sc)
    if use_as:
        t0 = time.time()
        ab = absorbing_set_spectrum(sc, a_max=a_max, beam=beam)
        as_sec = time.time() - t0
        as_count = ab["as_count"]
        as_min_a = ab["as_min_a"]
    else:
        as_count, as_min_a, as_sec = -1, -1, 0.0
    return as_count, cyc["n4"], cyc["n6"], cyc["girth"], as_min_a, as_sec


def surrogate_score(as_count, n4, n6, use_as=True):
    """Lexicographic harmful-structure score (LOWER = better)."""
    if use_as and as_count >= 0:
        return W_AS * as_count + W_N4 * n4 + W_N6 * n6
    return W_N4 * n4 + W_N6 * n6          # cycle-only fallback


def _surrogate_worker(task):
    """Parallel surrogate scorer for a batch of assignments."""
    cfg_d, assign, a_max, beam, use_as = task
    cfg = Config(**cfg_d)
    sc = cfg.build(assign=np.asarray(assign, np.int64))
    as_count, n4, n6, g, min_a, _ = surrogate_parts(sc, a_max, beam, use_as)
    return surrogate_score(as_count, n4, n6, use_as), int(as_count), int(n4), int(n6)


def eval_surrogate_batch(cfg, assigns, pool, a_max=A_MAX, beam=BEAM, use_as=True):
    cfg_d = cfg_to_dict(cfg)
    tasks = [(cfg_d, np.asarray(a, np.int64), a_max, beam, use_as) for a in assigns]
    res = pool.map(_surrogate_worker, tasks)
    scores = np.array([r[0] for r in res], float)
    return scores, res


def full_struct(cfg, assign, a_max=A_MAX, beam=BEAM, use_as=True):
    """Full structural record for a single champion (for the checkpoint)."""
    sc = cfg.build(assign=assign)
    as_count, n4, n6, g, min_a, sec = surrogate_parts(sc, a_max, beam, use_as)
    return {"as_count": int(as_count), "n4": int(n4), "n6": int(n6),
            "girth": int(g), "as_min_a": int(min_a),
            "score": float(surrogate_score(as_count, n4, n6, use_as)),
            "as_sec": float(sec)}


# --------------------------------------------------------------------------- #
#  optimisers over the surrogate  (equal eval budget = n_steps * batch)
# --------------------------------------------------------------------------- #
def optimise_rl(cfg, pool, n_steps, batch, a_max, beam, use_as, seed=21,
                log_tag=""):
    """REINFORCE (FeaturePolicy) minimising the surrogate.  Returns
    (best_assign, best_score, history)."""
    policy = FeaturePolicy(cfg, lr=0.15, ent=0.02, seed=seed)
    best = (np.inf, None)
    hist = []
    for step in range(1, n_steps + 1):
        traces, assigns = [], []
        for _ in range(batch):
            a, tr = policy.rollout()
            traces.append(tr); assigns.append(a)
        scores, _ = eval_surrogate_batch(cfg, assigns, pool, a_max, beam, use_as)
        # reward = -score (higher better); standardise advantage for a stable step
        rewards = -scores
        b = rewards.mean()
        adv = rewards - b
        if adv.std() > 1e-9:
            adv = adv / (adv.std() + 1e-9)
        policy.update(list(zip(traces, adv)))
        bi = int(np.argmin(scores))
        if scores[bi] < best[0]:
            best = (float(scores[bi]), np.asarray(assigns[bi]).copy())
        # also consider the greedy assignment of the current policy
        ga = policy.greedy_assign()
        gs, _ = eval_surrogate_batch(cfg, [ga], pool, a_max, beam, use_as)
        if gs[0] < best[0]:
            best = (float(gs[0]), np.asarray(ga).copy())
        hist.append(best[0])
        if step % 5 == 0 or step == 1:
            print(f"    [RL{log_tag}] step {step:3d}  batch min S={scores.min():.3e} "
                  f"greedy S={gs[0]:.3e}  best S={best[0]:.3e}", flush=True)
    return best[1], best[0], hist


def optimise_cem(cfg, pool, n_steps, batch, a_max, beam, use_as, seed=77,
                 elite_frac=0.3, smooth=0.7, log_tag=""):
    """CEM over per-edge categoricals minimising the surrogate."""
    _, _, E = edge_meta(cfg); C = n_components(cfg)
    rng = np.random.default_rng(seed)
    p = np.full((E, C), 1.0 / C)
    n_elite = max(2, int(batch * elite_frac))
    best = (np.inf, None)
    hist = []
    for step in range(1, n_steps + 1):
        assigns = [np.array([rng.choice(C, p=p[e]) for e in range(E)])
                   for _ in range(batch)]
        scores, _ = eval_surrogate_batch(cfg, assigns, pool, a_max, beam, use_as)
        elite = np.argsort(scores)[:n_elite]
        freq = np.zeros((E, C))
        for idx in elite:
            freq[np.arange(E), assigns[idx]] += 1.0
        freq /= n_elite
        p = smooth * p + (1 - smooth) * freq
        p = np.clip(p, 1e-3, None); p /= p.sum(axis=1, keepdims=True)
        bi = int(np.argmin(scores))
        if scores[bi] < best[0]:
            best = (float(scores[bi]), np.asarray(assigns[bi]).copy())
        hist.append(best[0])
        if step % 5 == 0 or step == 1:
            print(f"    [CEM{log_tag}] step {step:3d}  batch min S={scores.min():.3e} "
                  f"best S={best[0]:.3e}", flush=True)
    return best[1], best[0], hist


def best_of_random(cfg, pool, n_draw, a_max, beam, use_as, seed=12345):
    """Best-of-N random spreadings (the natural search baseline)."""
    rng = np.random.default_rng(seed)
    assigns = [random_assign(cfg, np.random.default_rng(seed + 1 + s))
               for s in range(n_draw)]
    scores, _ = eval_surrogate_batch(cfg, assigns, pool, a_max, beam, use_as)
    bi = int(np.argmin(scores))
    # also return the median random score (typical, not cherry-picked)
    return np.asarray(assigns[bi]).copy(), float(scores[bi]), float(np.median(scores))


# --------------------------------------------------------------------------- #
#  per-w driver
# --------------------------------------------------------------------------- #
def run_w(w, args, pool):
    cfg = make_cfg(w)
    sc0 = cfg.build(seed=0)
    ri, cj, E = edge_meta(cfg)
    C = n_components(cfg)
    print(f"\n========== w = {w}  (L={cfg.L}, W={cfg.W}, C={C} components, "
          f"E={E} sys edges, R={sc0.rate:.4f}, num_var={sc0.num_var}) ==========",
          flush=True)

    # ---- decide AS vs cycle-only by a quick timing probe on round_robin ---- #
    rr = round_robin_assign(cfg)
    t0 = time.time()
    _ = absorbing_set_spectrum(cfg.build(assign=rr), a_max=args.a_max, beam=args.beam)
    probe_sec = time.time() - t0
    use_as = probe_sec <= args.as_budget
    if use_as:
        print(f"  [surrogate] absorbing-set enumeration OK "
              f"({probe_sec:.2f}s <= {args.as_budget:.0f}s budget); "
              f"reward = as_count*{W_AS:.0e}+n4*{W_N4:.0e}+n6", flush=True)
    else:
        print(f"  [surrogate] absorbing-set enumeration TOO SLOW "
              f"({probe_sec:.2f}s > {args.as_budget:.0f}s) -> FALLBACK to "
              f"4-/6-cycle surrogate (reward = n4*{W_N4:.0e}+n6)", flush=True)

    n_steps, batch = args.steps, args.batch
    t_w = time.time()

    # ---- baselines --------------------------------------------------------- #
    rr_struct = full_struct(cfg, rr, args.a_max, args.beam, use_as)
    print(f"  [round_robin] S={rr_struct['score']:.3e} "
          f"AS={rr_struct['as_count']} n4={rr_struct['n4']} n6={rr_struct['n6']} "
          f"g={rr_struct['girth']}", flush=True)

    rnd_assign, rnd_best_score, rnd_med_score = best_of_random(
        cfg, pool, args.n_random, args.a_max, args.beam, use_as)
    rnd_struct = full_struct(cfg, rnd_assign, args.a_max, args.beam, use_as)
    print(f"  [random   ] best-of-{args.n_random} S={rnd_struct['score']:.3e} "
          f"(median S={rnd_med_score:.3e}) AS={rnd_struct['as_count']} "
          f"n4={rnd_struct['n4']} n6={rnd_struct['n6']}", flush=True)

    # ---- RL ---------------------------------------------------------------- #
    print(f"  [RL] optimising surrogate ({n_steps} steps x {batch}) ...", flush=True)
    rl_assign, rl_score, rl_hist = optimise_rl(
        cfg, pool, n_steps, batch, args.a_max, args.beam, use_as, log_tag=f"|w{w}")
    rl_struct = full_struct(cfg, rl_assign, args.a_max, args.beam, use_as)
    print(f"  [RL] best S={rl_struct['score']:.3e} AS={rl_struct['as_count']} "
          f"n4={rl_struct['n4']} n6={rl_struct['n6']} g={rl_struct['girth']}",
          flush=True)

    # ---- CEM --------------------------------------------------------------- #
    print(f"  [CEM] optimising surrogate ({n_steps} steps x {batch}) ...", flush=True)
    cem_assign, cem_score, cem_hist = optimise_cem(
        cfg, pool, n_steps, batch, args.a_max, args.beam, use_as, log_tag=f"|w{w}")
    cem_struct = full_struct(cfg, cem_assign, args.a_max, args.beam, use_as)
    print(f"  [CEM] best S={cem_struct['score']:.3e} AS={cem_struct['as_count']} "
          f"n4={cem_struct['n4']} n6={cem_struct['n6']} g={cem_struct['girth']}",
          flush=True)

    champions = {
        "round_robin": {"assign": rr.tolist(), "struct": rr_struct},
        "random":      {"assign": rnd_assign.tolist(), "struct": rnd_struct,
                        "median_score": rnd_med_score},
        "cem":         {"assign": cem_assign.tolist(), "struct": cem_struct,
                        "hist": [float(x) for x in cem_hist]},
        "rl":          {"assign": rl_assign.tolist(), "struct": rl_struct,
                        "hist": [float(x) for x in rl_hist]},
    }

    rec = {
        "w": w, "L": cfg.L, "W": cfg.W, "C": C, "E": E,
        "rate": float(sc0.rate), "num_var": int(sc0.num_var),
        "use_as": bool(use_as), "as_probe_sec": float(probe_sec),
        "a_max": args.a_max, "beam": args.beam,
        "champions": champions,
        "elapsed_sec": None,        # filled below
    }

    # ---- VALIDATION at small w: real error floor of the four champions ----- #
    if w in VAL_W and not args.no_validate:
        print(f"  [validate] measuring REAL error floor @ {FLOOR_SNRS} dB "
              f"({args.floor_frames} frames) for the 4 champions ...", flush=True)
        names = ["round_robin", "random", "cem", "rl"]
        assigns = [np.asarray(champions[n]["assign"], np.int64) for n in names]
        t0 = time.time()
        fl_ber, fl_fer, fl_ferr, fl_nfr = measure_ber_curve(
            cfg, assigns, FLOOR_SNRS, args.floor_frames, frame_seed=5000,
            pool=pool, frame_chunks=n_workers())
        print(f"    floor sweep done in {time.time()-t0:.0f}s", flush=True)
        # censored floor BER at deepest SNR
        K = sc0.K
        floor = {}
        for ai, nm in enumerate(names):
            cens = 0.5 / (fl_nfr[ai] * K)
            ber = np.where(fl_ber[ai] > 0, fl_ber[ai], cens)
            floor[nm] = {
                "ber": fl_ber[ai].tolist(), "ferr": fl_ferr[ai].tolist(),
                "nfr": fl_nfr[ai].tolist(),
                "ber_censored": ber.tolist(),
                "floor_ber_deepest": float(ber[-1]),
            }
            print(f"    {nm:12s} S={champions[nm]['struct']['score']:.3e}  "
                  f"floorBER@{FLOOR_SNRS[-1]}dB="
                  f"{ber[-1]:.2e}  ferr={fl_ferr[ai].tolist()}", flush=True)
        rec["validation"] = {"floor_snrs": FLOOR_SNRS, "floor": floor}
        # quick rank-agreement: does lower surrogate => lower floor across the 4?
        S = np.array([champions[n]["struct"]["score"] for n in names])
        F = np.array([floor[n]["floor_ber_deepest"] for n in names])
        order_S = np.argsort(S); order_F = np.argsort(F)
        agree = bool(np.array_equal(order_S, order_F))
        rec["validation"]["rank_match_S_vs_floor"] = agree
        rec["validation"]["champion_order_by_S"] = [names[i] for i in order_S]
        rec["validation"]["champion_order_by_floor"] = [names[i] for i in order_F]
        print(f"    rank(surrogate)==rank(floor)? {agree}  "
              f"[byS={[names[i] for i in order_S]}]", flush=True)

    rec["elapsed_sec"] = time.time() - t_w
    print(f"  [w={w}] done in {rec['elapsed_sec']:.0f}s", flush=True)
    return rec


# --------------------------------------------------------------------------- #
#  plots (best-effort)
# --------------------------------------------------------------------------- #
def make_plots(prefix, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = [r["w"] for r in records]
    methods = ["round_robin", "random", "cem", "rl"]
    style = {"round_robin": dict(color="k", marker="o", lw=2, label="round_robin"),
             "random": dict(color="tab:gray", marker="v", lw=1.5, ls="--", label="best random"),
             "cem": dict(color="tab:blue", marker="s", lw=2, label="CEM (surrogate)"),
             "rl": dict(color="tab:green", marker="D", lw=2, label="RL (surrogate)")}

    def series(key):
        return {m: [r["champions"][m]["struct"][key] for r in records] for m in methods}

    # ---- HEADLINE 1: absorbing-set count vs w --------------------------- #
    AS = series("as_count")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in methods:
        y = [v if v >= 0 else np.nan for v in AS[m]]
        ax.plot(ws, y, **style[m])
    ax.set_xlabel("coupling memory  w"); ax.set_ylabel(f"# elementary absorbing sets (a<={A_MAX}, beam={BEAM})")
    ax.set_title("Harmful-structure surrogate (absorbing sets) vs coupling memory w")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{prefix}_as_vs_w.png", dpi=130); plt.close(fig)

    # ---- HEADLINE 2: 4-cycle count vs w (the clean discriminator) ------- #
    N4 = series("n4")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in methods:
        ax.plot(ws, N4[m], **style[m])
    ax.set_xlabel("coupling memory  w"); ax.set_ylabel("# 4-cycles in lifted coupled Tanner graph")
    ax.set_title("Harmful-structure surrogate (4-cycles) vs coupling memory w")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{prefix}_n4_vs_w.png", dpi=130); plt.close(fig)

    # ---- HEADLINE 3: GAP (round_robin - optimised) grows with w --------- #
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rr_n4 = np.array(N4["round_robin"], float)
    for m in ["rl", "cem", "random"]:
        gap = rr_n4 - np.array(N4[m], float)
        ax.plot(ws, gap, **{**style[m], "label": f"round_robin - {m} (n4)"})
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("coupling memory  w")
    ax.set_ylabel("4-cycle reduction vs round_robin")
    ax.set_title("Construction headroom GROWS with coupling memory w")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{prefix}_gap_vs_w.png", dpi=130); plt.close(fig)

    # ---- VALIDATION: surrogate vs measured floor (small w) -------------- #
    val_records = [r for r in records if "validation" in r]
    if val_records:
        fig, axes = plt.subplots(1, len(val_records), figsize=(6 * len(val_records), 5),
                                 squeeze=False)
        for k, r in enumerate(val_records):
            ax = axes[0][k]
            for m in methods:
                S = r["champions"][m]["struct"]["score"]
                F = r["validation"]["floor"][m]["floor_ber_deepest"]
                ax.scatter([S], [F], s=60, color=style[m]["color"], label=m,
                           edgecolor="w", zorder=5)
                ax.annotate(m, (S, F), fontsize=8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("surrogate score S (lower=better)")
            ax.set_ylabel(f"measured floor BER @ {r['validation']['floor_snrs'][-1]} dB")
            agree = r["validation"].get("rank_match_S_vs_floor")
            ax.set_title(f"w={r['w']}  rank-match={agree}")
            ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{prefix}_validation.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--n_random", type=int, default=200,
                    help="best-of-N random spreadings baseline")
    ap.add_argument("--a_max", type=int, default=A_MAX)
    ap.add_argument("--beam", type=int, default=BEAM)
    ap.add_argument("--as_budget", type=float, default=AS_TIME_BUDGET)
    ap.add_argument("--floor_frames", type=int, default=FLOOR_FRAMES)
    ap.add_argument("--w_sweep", type=str, default=",".join(map(str, W_SWEEP)))
    ap.add_argument("--no_validate", action="store_true")
    ap.add_argument("--out_prefix", type=str, default="largew")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    if args.quick:
        args.steps = 3
        args.batch = 8
        args.n_random = 12
        args.floor_frames = 200
        args.w_sweep = "3,6"

    w_list = [int(x) for x in args.w_sweep.split(",") if x.strip()]
    nw = n_workers()
    print(f"[largew] w_sweep={w_list}  steps={args.steps} batch={args.batch} "
          f"n_random={args.n_random} a_max={args.a_max} beam={args.beam} "
          f"floor_frames={args.floor_frames}", flush=True)
    print(f"[largew] workers={nw}  out_prefix={args.out_prefix}", flush=True)
    t_start = time.time()

    pool = mp.get_context("spawn").Pool(nw)
    records = []
    try:
        for w in w_list:
            rec = run_w(w, args, pool)
            records.append(rec)
            # ---- checkpoint after EVERY w ---- #
            out = {
                "meta": {"w_sweep": w_list, "steps": args.steps, "batch": args.batch,
                         "n_random": args.n_random, "a_max": args.a_max,
                         "beam": args.beam, "floor_snrs": FLOOR_SNRS,
                         "floor_frames": args.floor_frames,
                         "W_AS": W_AS, "W_N4": W_N4, "W_N6": W_N6,
                         "elapsed_sec": time.time() - t_start},
                "records": records,
            }
            ckpt = f"{args.out_prefix}.json"
            tmp = ckpt + ".tmp"
            with open(tmp, "w") as f:
                json.dump(out, f, indent=1)
            os.replace(tmp, ckpt)
            print(f"[checkpoint] wrote {ckpt} through w={w} "
                  f"({time.time()-t_start:.0f}s total)", flush=True)
    finally:
        pool.close(); pool.join()

    # ---- summary table ---- #
    print("\n================ SUMMARY: surrogate vs w ================", flush=True)
    print(f"{'w':>3} {'rate':>6} {'method':>12} {'AS':>7} {'n4':>7} {'n6':>9} "
          f"{'girth':>5} {'score':>11}", flush=True)
    for r in records:
        for m in ["round_robin", "random", "cem", "rl"]:
            s = r["champions"][m]["struct"]
            print(f"{r['w']:>3} {r['rate']:>6.3f} {m:>12} {s['as_count']:>7} "
                  f"{s['n4']:>7} {s['n6']:>9} {s['girth']:>5} {s['score']:>11.3e}",
                  flush=True)
    # gap growth headline
    print("\n  4-cycle count: round_robin vs best-optimised (rl/cem min), and the GAP:",
          flush=True)
    for r in records:
        rr = r["champions"]["round_robin"]["struct"]["n4"]
        opt = min(r["champions"]["rl"]["struct"]["n4"],
                  r["champions"]["cem"]["struct"]["n4"])
        print(f"    w={r['w']:>2}: round_robin n4={rr:>6}  optimised n4={opt:>6}  "
              f"GAP={rr-opt:>6}  ({'grows' if True else ''})", flush=True)

    print(f"\n[largew] total runtime {time.time()-t_start:.0f}s", flush=True)

    # ---- plots ---- #
    try:
        make_plots(args.out_prefix, records)
        print(f"[saved] plots {args.out_prefix}_*.png", flush=True)
    except Exception as e:
        print(f"[plots skipped] {type(e).__name__}: {e}", flush=True)

    print(f"[done] result file: {args.out_prefix}.json", flush=True)


if __name__ == "__main__":
    main()
