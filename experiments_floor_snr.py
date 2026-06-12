"""P3 -- high-precision BER/FER waterfall + floor sweep for the deep-cell champion
constructions (T-COM "eliminates the error floor" substantiation).

The paper claims optimized constructions have NO error floor while default/naive
ones floor at ~3-6e-5.  But Table tab:ber rests on only 5 SNR points at 2000
frames, where "BER=0" at high SNR has no statistical power (95% upper bound at
0/2000 frames is ~1.5e-3 -- nowhere near 1e-6).  A reviewer will reject the
"floor eliminated" wording on that evidence.

This driver re-simulates the EXACT champion constructions the paper plots
(results_construct_search.json -> "champions") over a FINE SNR grid with LARGE,
SNR-growing frame banks (so the high-SNR points actually probe down to ~1e-6/1e-7),
recording raw bit/frame error counts for Clopper-Pearson CIs.  Output either
(a) substantiates "no floor for the optimized codes vs a clear floor for the
default/naive ones", or (b) tells us to soften the wording -- either way honest.

Constructions swept (the six in Fig:ber):
  rl_feature, cem, random_search  (n4=0, claimed floor-free)
  round_robin, seed0_default      (n4~930, claimed to floor)
  uncoupled component code         (reference; built with R.eval if available)

Usage (CPU pod, EXP_WORKERS = pod cores):
  EXP_WORKERS=48 python3 experiments_floor_snr.py run   # -> results_floor_snr.json
  python3 experiments_floor_snr.py agg                  # printed waterfall table
Checkpoints after every (construction, SNR) so an evicted pod loses one point.
"""
from __future__ import annotations
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import rl_construct as R

CFG = R.Config(bg=2, ils=0, Z=16, mp=8, w=2, L=30, W=6, max_iter=12)

# fine SNR grid + frame bank that GROWS with SNR so deep-floor points get enough
# frames to resolve ~1e-6.  Deep cell ~4480 info bits/frame -> 60k frames ~ 2.7e8
# info bits -> resolves BER ~ 1e-7 (tens of bit errors).
SNR_GRID   = [2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0]
FRAME_BANK = {2.2: 4000, 2.5: 4000, 2.8: 12000, 3.1: 25000,
              3.4: 50000, 3.7: 60000, 4.0: 60000}
FRAME_SEED = 24680
CHUNK_FRAMES = 2000        # split a big bank into chunks so each pool job is bounded

# which champions to sweep (keys in results_construct_search.json["champions"]).
WANT = ["rl_feature", "cem", "random_search", "round_robin", "seed0_default"]


def _load_champions():
    """Return {name: assign-array}.  Prefer the paper's plotted champions; fall back
    to reconstructing the naive ones so the driver is self-contained."""
    out = {}
    path = "results/construct/results_construct_search.json"
    if not os.path.exists(path):
        path = "results_construct_search.json"
    if os.path.exists(path):
        d = json.load(open(path))
        champs = d.get("champions", {})
        for name in WANT:
            if name in champs:
                out[name] = np.asarray(champs[name], dtype=np.int64)
    # guaranteed fallbacks for the naive baselines
    if "round_robin" not in out:
        out["round_robin"] = np.asarray(R.round_robin_assign(CFG), dtype=np.int64)
    if "seed0_default" not in out:
        out["seed0_default"] = np.asarray(R.random_assign(CFG, np.random.default_rng(0)),
                                          dtype=np.int64)
    return out


def _eval_big(assign, snr, n_frames, pool):
    """Evaluate one construction at one SNR over n_frames, chunked, pooling raw
    counts so the BER/FER come from the full bank (not an average of averages)."""
    bit_err = n_bits = frame_err = n_frames_done = 0
    chunks = []
    f = 0
    cid = 0
    while f < n_frames:
        nf = min(CHUNK_FRAMES, n_frames - f)
        chunks.append((cid, nf))
        f += nf
        cid += 1
    for cid, nf in chunks:
        # distinct frame seed per chunk so chunks are independent draws
        m = R.eval_assignments(CFG, [assign], snr, FRAME_SEED + cid * 101, nf,
                               pool=pool, frame_chunks=R.n_workers())[0]
        bit_err += m["bit_err"]; n_bits += m["n_info"]
        frame_err += m["frame_err"]; n_frames_done += m["n_frames"]
    ber = bit_err / max(n_bits, 1)
    fer = frame_err / max(n_frames_done, 1)
    return {"ber": ber, "fer": fer, "bit_err": int(bit_err), "n_info": int(n_bits),
            "frame_err": int(frame_err), "n_frames": int(n_frames_done)}


def run():
    fname = "results_floor_snr.json"
    champs = _load_champions()
    nw = R.n_workers()
    print(f"workers={nw}  constructions={list(champs)}", flush=True)
    if os.path.exists(fname):
        out = json.load(open(fname))
    else:
        out = {"cfg": _cfg_d(CFG), "snr_grid": SNR_GRID, "frame_bank": FRAME_BANK,
               "n4": {n: int(R.count_4cycles(CFG.build(assign=a))) for n, a in champs.items()},
               "girth": {n: int(R.girth(CFG.build(assign=a), n_seeds=200)) for n, a in champs.items()},
               "curves": {}}
    with Pool(nw) as pool:
        for name, assign in champs.items():
            out["curves"].setdefault(name, {})
            for snr in SNR_GRID:
                key = f"{snr}"
                if key in out["curves"][name]:
                    continue
                t0 = time.time()
                m = _eval_big(assign, snr, FRAME_BANK[snr], pool)
                m["sec"] = round(time.time() - t0, 1)
                out["curves"][name][key] = m
                json.dump(out, open(fname, "w"), indent=1)
                print(f"  {name:14s} @ {snr}dB: BER={m['ber']:.3e} FER={m['fer']:.3e} "
                      f"({m['bit_err']} bit-err / {m['n_info']} bits, {m['frame_err']}/{m['n_frames']} fr, "
                      f"{m['sec']}s)", flush=True)
    print(f"--- done -> {fname}", flush=True)
    aggregate()


def aggregate():
    fname = "results_floor_snr.json"
    if not os.path.exists(fname):
        print("no results_floor_snr.json"); return
    out = json.load(open(fname))
    grid = out["snr_grid"]
    print("\n=== P3 high-precision BER waterfall (deep cell) ===")
    print(f"{'construction':16} " + " ".join(f"{s:>9}" for s in grid) + "   n4 girth")
    for name, curve in out["curves"].items():
        cells = []
        for s in grid:
            m = curve.get(f"{s}")
            cells.append(f"{m['ber']:>9.2e}" if m else f"{'--':>9}")
        n4 = out["n4"].get(name, "?"); g = out["girth"].get(name, "?")
        print(f"{name:16} " + " ".join(cells) + f"   {n4} {g}")
    # crude floor detector: lowest BER reached + whether the last two points stop decaying
    print("\nfloor check (last two grid points):")
    for name, curve in out["curves"].items():
        pts = [(s, curve.get(f"{s}")) for s in grid if curve.get(f"{s}")]
        if len(pts) >= 2:
            (s1, m1), (s0, m0) = pts[-1], pts[-2]
            ratio = (m0["ber"] / m1["ber"]) if m1["ber"] > 0 else float("inf")
            tag = "FLOOR?" if (m1["ber"] > 0 and ratio < 3) else "decaying"
            print(f"  {name:16} {s0}->{s1}dB BER {m0['ber']:.2e}->{m1['ber']:.2e} "
                  f"(x{ratio:.1f}) {tag}  [{m1['bit_err']} bit-err at {s1}dB]")


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
        raise SystemExit("usage: experiments_floor_snr.py [run|agg]")
