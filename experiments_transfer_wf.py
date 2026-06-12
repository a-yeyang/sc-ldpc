"""P2 -- WATERFALL-objective zero-shot transfer of the learned edge-spreading
policy, multi-target (T-COM C4 strengthening).

The paper's C4 transfer (results_construct_search.json -> "transfer") covers only
TWO unseen configs and evaluates them at a single fixed SNR (so the high-rate
target sits at FER~0.79, poor discrimination).  A reviewer reads "zero-shot beats
random median on 2 points, but best-of-12 random can edge it out" -> weak claim.

This driver strengthens C4 to a DEFENSIBLE result:
  * train ONE FeaturePolicy (w=2, 8 params) by REINFORCE on the deep base cell
    (the SAME waterfall/FER objective as the rest of the paper -- NOT the floor
    surrogate of experiments_transfer.py);
  * transfer the frozen 8 weights ZERO-SHOT (greedy rollout, no re-search) to
    >=6 UNSEEN w=2 configs varying Z / mp / L / ils / base-graph (incl. a
    cross-BG target -- the hardest test);
  * at each target AUTO-PICK an operating SNR (random-median FER ~0.25, in the
    waterfall) so every target is compared at a discriminating point;
  * compare zero_shot against (a) per-instance RL trained FROM SCRATCH on that
    target [the ceiling], (b) round_robin, (c) best_of_k random search at a fair
    budget, (d) the random-construction median;
  * HEADLINE = fraction of the (round_robin -> per-instance-RL) FER gain that the
    zero-shot policy recovers at ZERO search cost, per target and pooled, with the
    raw frame/bit error counts saved for Clopper-Pearson CIs downstream.

Usage (CPU pod, EXP_WORKERS = pod cores):
  EXP_WORKERS=48 python3 experiments_transfer_wf.py run   # -> results_transfer_wf.json
  python3 experiments_transfer_wf.py agg                  # printed table from the json
Checkpoints after every target so an evicted pod loses at most one target.
"""
from __future__ import annotations
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import rl_construct as R

# --------------------------------------------------------------------------- #
# base (training) cell -- the deep cell the rest of the paper trains on.
# --------------------------------------------------------------------------- #
BASE_CFG = R.Config(bg=2, ils=0, Z=16, mp=8, w=2, L=30, W=6, max_iter=12)

# >=6 UNSEEN transfer targets, ALL w=2 so the 8-param theta transfers verbatim.
# Vary Z (block length), mp (rate), L (chain), ils (lifting set), and base-graph.
TARGETS = {
    "Z24_ils1":     R.Config(bg=2, ils=1, Z=24, mp=8,  w=2, L=30, W=6, max_iter=12),  # longer blocks (paper)
    "mp6_hi_rate":  R.Config(bg=2, ils=0, Z=16, mp=6,  w=2, L=30, W=6, max_iter=12),  # higher rate (paper)
    "Z32_long":     R.Config(bg=2, ils=0, Z=32, mp=8,  w=2, L=30, W=6, max_iter=12),  # 2x block length
    "Z64_long":     R.Config(bg=2, ils=0, Z=64, mp=8,  w=2, L=30, W=6, max_iter=12),  # 4x block length
    "mp10_lo_rate": R.Config(bg=2, ils=0, Z=16, mp=10, w=2, L=30, W=6, max_iter=12),  # lower rate
    "L50_chain":    R.Config(bg=2, ils=0, Z=16, mp=8,  w=2, L=50, W=6, max_iter=12),  # longer chain
    "crossBG1_R07": R.Config(bg=1, ils=0, Z=16, mp=11, w=2, L=30, W=6, max_iter=12),  # cross base-graph (hard)
}

# training budget for the base policy AND each per-instance ceiling (equal budget).
STEPS, BATCH, FRAMES, VAL = 18, 48, 28, 80
PROBE_N, PROBE_FRAMES = 48, 20
RANDOM_K = 12                     # best-of-k random search budget at each target
FINAL = 1500                      # champion re-eval frame bank
FINAL_SEED = 987654
CAND = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]   # SNR auto-pick grid


def _freeze_snr(cfg, pool):
    """Operating SNR where the random-construction median FER ~0.25 (in/before the
    waterfall) -- a discriminating point.  Mirrors experiments_ci._freeze_snr."""
    rng = np.random.default_rng(0)
    assigns = [R.random_assign(cfg, rng) for _ in range(PROBE_N)]
    meds = [float(np.median([m["fer"] for m in R.eval_batch(cfg, assigns, s, 1, PROBE_FRAMES, pool=pool)]))
            for s in CAND]
    usable = [i for i, m in enumerate(meds) if m >= 0.05]
    if not usable:
        return CAND[0]
    return CAND[min(usable, key=lambda i: abs(meds[i] - 0.25))]


def _stats(cfg, assign):
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return {"n4": R.count_4cycles(sc), "girth": R.girth(sc, n_seeds=200)}


def _eval(cfg, assign, snr, pool):
    m = R.eval_assignments(cfg, [np.asarray(assign, dtype=np.int64)], snr, FINAL_SEED,
                           FINAL, pool=pool, frame_chunks=R.n_workers())[0]
    st = _stats(cfg, assign)
    return {"fer": m["fer"], "ber": m["ber"], "bit_err": m["bit_err"], "n_info": m["n_info"],
            "frame_err": m["frame_err"], "n_frames": m["n_frames"],
            "n4": st["n4"], "girth": st["girth"]}


def train_base(pool):
    """Train the transferable FeaturePolicy on the deep base cell; return theta."""
    pol = R.FeaturePolicy(BASE_CFG, seed=0)
    snr = 2.5
    print(f"[base] training FeaturePolicy on deep cell @ {snr}dB "
          f"(E={R.edge_meta(BASE_CFG)[2]}, theta dim={pol.theta.shape[0]}) ...", flush=True)
    _, best = R.train_reinforce(BASE_CFG, pol, snr, STEPS, BATCH, FRAMES, pool=pool,
                                val_frames=VAL, base_seed=1000, verbose=False)
    print(f"[base] trained.  base champion val FER={best.get('fer')}  theta={np.round(pol.theta,3).tolist()}",
          flush=True)
    return pol.theta.copy()


def run():
    fname = "results_transfer_wf.json"
    nw = R.n_workers()
    print(f"workers={nw}", flush=True)
    with Pool(nw) as pool:
        if os.path.exists(fname):
            out = json.load(open(fname))
            theta = np.asarray(out["theta"], dtype=np.float64)
        else:
            theta = train_base(pool)
            out = {"base_cfg": _cfg_d(BASE_CFG), "theta": theta.tolist(),
                   "budget": {"steps": STEPS, "batch": BATCH, "frames": FRAMES,
                              "val": VAL, "final": FINAL, "random_k": RANDOM_K},
                   "targets": {}}
            json.dump(out, open(fname, "w"), indent=1)

        for label, tcfg in TARGETS.items():
            if label in out["targets"]:
                print(f"[{label}] cached, skip", flush=True); continue
            t0 = time.time()
            snr = _freeze_snr(tcfg, pool)
            rate = tcfg.build().rate

            # (a) zero-shot: frozen theta, greedy rollout, NO search on the new code
            pol = R.FeaturePolicy(tcfg)
            assert pol.theta.shape == theta.shape, f"theta must transfer (same w): {label}"
            pol.theta = theta.copy()
            zs_assign = pol.greedy_assign()
            zero_shot = _eval(tcfg, zs_assign, snr, pool)

            # (b) per-instance RL ceiling: fresh policy trained FROM SCRATCH here
            pol2 = R.FeaturePolicy(tcfg, seed=0)
            _, best2 = R.train_reinforce(tcfg, pol2, snr, STEPS, BATCH, FRAMES, pool=pool,
                                         val_frames=VAL, base_seed=4000, verbose=False)
            per_instance = _eval(tcfg, best2["assign"], snr, pool)

            # (c) round-robin heuristic
            rr_assign = R.round_robin_assign(tcfg)
            round_robin = _eval(tcfg, rr_assign, snr, pool)

            # (d) best-of-k random search + random-construction median
            rng = np.random.default_rng(7)
            rand_assigns = [R.random_assign(tcfg, rng) for _ in range(RANDOM_K)]
            mr = R.eval_assignments(tcfg, rand_assigns, snr, FINAL_SEED, 300, pool=pool)
            rfers = sorted(m["fer"] for m in mr)
            best_idx = int(np.argmin([m["fer"] for m in mr]))
            random_best = _eval(tcfg, rand_assigns[best_idx], snr, pool)

            # headline: fraction of (round_robin -> per_instance) gain recovered zero-shot
            gain = round_robin["fer"] - per_instance["fer"]
            recovered = (round_robin["fer"] - zero_shot["fer"]) / gain if gain > 1e-9 else None

            out["targets"][label] = {
                "cfg": _cfg_d(tcfg), "rate": rate, "snr": snr,
                "zero_shot": zero_shot, "per_instance_rl": per_instance,
                "round_robin": round_robin, "random_best_of_k": random_best,
                "random_median_fer": float(np.median(rfers)),
                "random_fers": rfers,
                "gain_recovered_frac": recovered,
                "zs_assign": np.asarray(zs_assign, dtype=np.int64).tolist(),
                "sec": round(time.time() - t0, 1),
            }
            json.dump(out, open(fname, "w"), indent=1)
            print(f"[{label}] rate={rate:.3f} @{snr}dB  zero_shot={zero_shot['fer']:.3f} "
                  f"per_inst={per_instance['fer']:.3f} rr={round_robin['fer']:.3f} "
                  f"rand[med={np.median(rfers):.3f},best={random_best['fer']:.3f}]  "
                  f"recovered={recovered if recovered is None else round(recovered,2)} "
                  f"({out['targets'][label]['sec']}s)", flush=True)
    print(f"--- done -> {fname}", flush=True)
    aggregate()


def aggregate():
    fname = "results_transfer_wf.json"
    if not os.path.exists(fname):
        print("no results_transfer_wf.json"); return
    out = json.load(open(fname))
    print("\n=== P2 waterfall zero-shot transfer (>=6 unseen w=2 targets) ===")
    print(f"{'target':14} {'rate':>5} {'snr':>4} {'zeroShot':>9} {'perInstRL':>9} "
          f"{'roundRob':>8} {'randMed':>8} {'randBest':>8} {'recov%':>7}")
    recs = out.get("targets", {})
    fracs = []
    for label, t in recs.items():
        rec = t.get("gain_recovered_frac")
        if rec is not None:
            fracs.append(rec)
        print(f"{label:14} {t['rate']:>5.3f} {t['snr']:>4} {t['zero_shot']['fer']:>9.3f} "
              f"{t['per_instance_rl']['fer']:>9.3f} {t['round_robin']['fer']:>8.3f} "
              f"{t['random_median_fer']:>8.3f} {t['random_best_of_k']['fer']:>8.3f} "
              f"{'--' if rec is None else round(100*rec):>7}")
    if fracs:
        print(f"\npooled gain recovered zero-shot: mean={100*np.mean(fracs):.0f}%  "
              f"median={100*np.median(fracs):.0f}%  (n={len(fracs)} targets with signal)")
        beats_median = sum(1 for label, t in recs.items()
                           if t['zero_shot']['fer'] < t['random_median_fer'])
        print(f"zero-shot beats random median on {beats_median}/{len(recs)} targets")


def _cfg_d(cfg):
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "agg":
        aggregate()
    elif cmd == "run":
        run()
    else:
        raise SystemExit("usage: experiments_transfer_wf.py [run|agg]")
