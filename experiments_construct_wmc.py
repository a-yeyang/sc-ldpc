"""Monte-Carlo w-sweep: extend the long-code RL-vs-random result (5.7) to larger coupling
memory w in {3,5,8,12} with EXPENSIVE end-to-end evaluation -- the regime where RL's sample
efficiency wins.  (w=30 Monte-Carlo is infeasible at ~18 s/frame; see the proxy sweep 5.8.)

5G NR BG1, Z=32 (~860-1470 bit component codes), wireless rates {1/2,2/3,3/4,5/6,7/8}.
The sliding window must span the coupling, so W = min(w+1, L).  Edge spreading is optimised
at L=50 and the champions are validated across chain length L in {50,100,200}.

Distributed across two 100-core pods by rate slice:
    python3 experiments_construct_wmc.py slice A
    python3 experiments_construct_wmc.py slice B
    python3 experiments_construct_wmc.py plot
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

BG, ILS, Z, MAXIT = 1, 0, 32, 12
OPT_L = 50
RATE_MP = {0.5: 24, 0.667: 13, 0.75: 9, 0.833: 6, 0.875: 5}
WS = [3, 5, 8, 12]
OPT_STEPS, OPT_BATCH, OPT_FRAMES = 10, 100, 8
VAL_FRAMES, FINAL_FRAMES = 80, 300
PROBE_FRAMES, PROBE_N = 16, 100
SNR_CAND = {0.5: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 0.667: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            0.75: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 0.833: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            0.875: [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
BER_OFFSETS = [-0.6, 0.0, 0.6]
BER_FRAMES = [150, 250, 350]
LSWEEP_LS = [50, 100, 200]
LSWEEP_FRAMES = [120, 180]
LSWEEP_OFFSETS = [0.0]
SLICES = {"A": {"rates": [0.5, 0.833], "lsweep": (0.667, 8)},
          "B": {"rates": [0.667, 0.75, 0.875], "lsweep": (0.75, 8)},
          "ALL": {"rates": [0.5, 0.667, 0.75, 0.833, 0.875], "lsweep": (0.667, 8)}}


def cfg(rate, w, L):
    return R.Config(bg=BG, ils=ILS, Z=Z, mp=RATE_MP[rate], w=w, L=L,
                    W=min(w + 1, L), max_iter=MAXIT)


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


def _save(out, fn):
    json.dump(out, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)


def pick_snr(c, candidates, pool):
    rng = np.random.default_rng(0)
    assigns = [R.random_assign(c, rng) for _ in range(PROBE_N)]
    meds = [float(np.median([m["fer"] for m in R.eval_batch(c, assigns, snr, 1, PROBE_FRAMES, pool=pool)]))
            for snr in candidates]
    usable = [i for i, m in enumerate(meds) if m >= 0.05]
    if not usable:
        return round(candidates[0] - 1.0, 2)
    return candidates[min(usable, key=lambda i: abs(meds[i] - 0.25))]


def ber_curve(c, assign, snrs, frames, pool, chunks=None):
    chunks = chunks or R.n_workers()
    xs, bers, fers = [], [], []
    for snr, nf in zip(snrs, frames):
        m = R.eval_assignments(c, [assign], snr, 2025, nf, pool=pool, frame_chunks=chunks)[0]
        xs.append(round(snr, 2)); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def run_cell(rate, w, pool):
    t0 = time.time()
    c = cfg(rate, w, OPT_L)
    rt = c.build().rate
    _, _, E = R.edge_meta(c)
    snr = pick_snr(c, SNR_CAND[rate], pool)
    # header format parsed by the dashboard (pod_status.py)
    print(f"\n=== R={rate} (sc rate {rt:.3f}) w={w}  E={E}  W={min(w+1,OPT_L)} "
          f"L={OPT_L} Z={Z}  train@{snr}dB ===", flush=True)
    pol = R.FeaturePolicy(c, lr=0.15, ent=0.02, seed=0)
    h_f, best_f = R.train_reinforce(c, pol, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                                    val_frames=VAL_FRAMES, val_seed=99, base_seed=1000, log_every=5)
    pol_e = R.PerEdgePolicy(E, w + 1, lr=0.2, ent=0.01, seed=0)
    h_e, best_e = R.train_reinforce(c, pol_e, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                                    val_frames=VAL_FRAMES, val_seed=99, base_seed=4000, log_every=5)
    h_c, best_c = R.train_cem(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                              val_frames=VAL_FRAMES, val_seed=99, base_seed=2000, log_every=5)
    h_r, best_r = R.train_random(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                                 val_frames=VAL_FRAMES, val_seed=99, base_seed=3000, log_every=5)
    champions = {"rl": best_f["assign"], "rl_peredge": best_e["assign"],
                 "cem": best_c["assign"], "random_search": best_r["assign"],
                 "round_robin": R.round_robin_assign(c),
                 "seed0_default": R.random_assign(c, np.random.default_rng(0))}
    snrs = [snr + d for d in BER_OFFSETS]
    finals, stats, curves = {}, {}, {}
    for name, a in champions.items():
        a = np.asarray(a, dtype=np.int64)
        m = R.eval_assignments(c, [a], snr, 12345, FINAL_FRAMES, pool=pool, frame_chunks=R.n_workers())[0]
        finals[name] = {"fer": m["fer"], "ber": m["ber"]}
        stats[name] = {"n4": R.construction_stats(c, a)["n4"]}
        if name not in ("rl_peredge",):
            curves[name] = ber_curve(c, a, snrs, BER_FRAMES, pool)
        print(f"  {name:14s} FER={m['fer']:.4f} BER={m['ber']:.3e} n4={stats[name]['n4']}", flush=True)
    print(f"  [cell done in {time.time()-t0:.0f}s]", flush=True)
    return {"rate": rate, "sc_rate": rt, "w": w, "E": int(E), "train_snr": snr,
            "finals": finals, "stats": stats, "curves": curves,
            "hist": {"rl": h_f, "rl_peredge": h_e, "cem": h_c, "random": h_r},
            "theta_rl": pol.theta.tolist(),
            "champions": {k: _arr(v) for k, v in champions.items()}}


def run_lsweep(rate, w, out, pool):
    key = f"R{rate}_w{w}"
    cell = out["cells"].get(key)
    if cell is None:
        return
    snr0 = cell["train_snr"]; snrs = [snr0 + d for d in LSWEEP_OFFSETS]
    champs = {n: np.asarray(cell["champions"][n], dtype=np.int64)
              for n in ["rl", "random_search", "seed0_default"]}
    lsweep = {}
    print(f"\n=== L-sweep R={rate} w={w} ===", flush=True)
    for L in LSWEEP_LS:
        cL = cfg(rate, w, L); chunks = min(24, R.n_workers())
        lsweep[str(L)] = {"rate": cL.build().rate}
        for n, a in champs.items():
            lsweep[str(L)][n] = ber_curve(cL, a, snrs, LSWEEP_FRAMES, pool, chunks=chunks)
        print(f"  L={L} (sc rate {cL.build().rate:.3f}) done", flush=True)
    out["lsweep"] = {"rate": rate, "w": w, "data": lsweep}


def run_slice(name):
    t0 = time.time()
    sl = SLICES[name]
    fn = f"results_big_{name}.json"          # dashboard reads this name
    out = None
    if os.path.exists(fn):
        try:
            out = json.load(open(fn)); out.setdefault("cells", {})
            print(f"[resume] {fn}: {len(out['cells'])} cells {sorted(out['cells'])}", flush=True)
        except Exception:
            out = None
    if out is None:
        out = {"slice": name, "config": {"bg": BG, "Z": Z, "opt_L": OPT_L, "ws": WS}, "cells": {}}
    with mp.Pool(R.n_workers()) as pool:
        print(f"slice {name}: rates {sl['rates']} x w {WS}  ({R.n_workers()} workers)", flush=True)
        for rate in sl["rates"]:
            for w in WS:
                key = f"R{rate}_w{w}"
                if key in out["cells"]:
                    print(f"  [skip] {key}", flush=True); continue
                try:
                    out["cells"][key] = run_cell(rate, w, pool)
                except Exception as e:
                    print(f"  !! cell {key} failed: {e}", flush=True)
                _save(out, fn)
        if sl.get("lsweep") and "lsweep" not in out:
            try:
                run_lsweep(sl["lsweep"][0], sl["lsweep"][1], out, pool)
            except Exception as e:
                print(f"  !! lsweep failed: {e}", flush=True)
            _save(out, fn)
    print(f"\nslice {name} done in {time.time()-t0:.0f}s -> {fn}", flush=True)


def make_plots():
    cells = {}
    for nm in ["A", "B", "ALL"]:
        try:
            cells.update(json.load(open(f"results_wmc_{nm}.json")).get("cells", {}))
        except FileNotFoundError:
            pass
    if not cells:
        print("no results_wmc_*.json"); return
    rates = sorted(set(c["rate"] for c in cells.values()))
    # headline: RL/random advantage vs w, one line per rate
    series = []
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, rate in enumerate(rates):
        xs, ys = [], []
        for w in WS:
            c = cells.get(f"R{rate}_w{w}")
            if not c:
                continue
            rl = min(c["finals"]["rl"]["fer"], c["finals"]["rl_peredge"]["fer"])
            rnd = c["finals"]["random_search"]["fer"]
            if rl >= 0.999 and rnd >= 0.999:        # skip no-signal cells (FER=1)
                continue
            xs.append(w); ys.append(min(rnd / max(rl, 0.005), 40))
        if xs:
            series.append({"x": xs, "y": ys, "label": f"R={rate}", "color": cols[i % len(cols)]})
    plotting.linear(series, xlabel="coupling memory w", ylabel="random_FER / RL_FER (>1 = RL better)",
                    title="RL vs random search vs w (Monte-Carlo, long Z=32 codes)",
                    path="exp_wmc_rl_vs_random.svg")
    print("wrote exp_wmc_rl_vs_random.svg")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "slice":
        run_slice(sys.argv[2])
    else:
        make_plots()
