"""SC-LDPC over square-QAM/AWGN: decoding-performance sweep (Part A).

Re-runs the SC-LDPC characterisation under QPSK / 16- / 64- / 256-QAM (channel
still AWGN) across a representative parameter grid and produces BER/FER
waterfalls.  This is the *performance* study (canonical construction); the RL
coded-modulation / probabilistic-shaping study is separate (Part B).

Component code: 5G NR BG1 (Kb=22), rate-matched to R in {1/2,2/3,3/4,5/6,7/8}.
Construction: deterministic balanced "round-robin" edge spreading (the canonical
default); a random spreading is also scored on the main Z=32 grid as a reference.
Modulation: square M-QAM, Gray, exact log-sum-exp soft demap, info-bit Eb/N0
(curves of different orders/rates directly comparable).  See ``qam.py``.

Grid (representative -- see make_cells):
  * main:    4 mod x 5 rate x Z in {16,32,64},  w=3, L=50      (length + order + rate)
  * wsweep:  4 mod x w in {3,4,6,8,12,16},       R=3/4, Z=32, L=100   (coupling depth)
  * lsweep:  4 mod x L in {30,100,200},          R=3/4, Z=32, w=4      (chain length)
The decode window grows with w (W=max(6,w+2)) so the coupling is always captured;
terminated SC-LDPC loses rate ~ w/L, so large w is paired with larger L and the
true transmitted rate (sc_rate) is recorded per cell.

Runs as ONE process driving a single pool over all visible cores; each cell's
frames are split across the pool (frame-level parallelism).  Resumable: completed
cells are reloaded from results_qam.json and skipped.
    EXP_WORKERS=40 python3 experiments_qam_sweep.py run     # the sweep
    python3 experiments_qam_sweep.py plot                   # figures + CSVs
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
import multiprocessing as mp
from dataclasses import asdict

import numpy as np

import plotting
import qam
import rl_construct as R
from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode

# --------------------------------------------------------------------------- #
#  regime
# --------------------------------------------------------------------------- #
BG, ILS = 1, 0
RATE_MP = {0.5: 24, 0.667: 13, 0.75: 9, 0.833: 6, 0.875: 5}     # BG1 (Kb=22) rate matching
RATES = [0.5, 0.667, 0.75, 0.833, 0.875]
MODS = [4, 16, 64, 256]                                          # QPSK, 16/64/256-QAM
MAXIT, ALPHA = 12, 0.8

# operating-point probe + waterfall sampling
SNR_BASE = {4: 3.0, 16: 5.5, 64: 8.5, 256: 13.0}                # ~rate-1/2 cliff (info Eb/N0)
PROBE_ITERS, PROBE_FRAMES, PROBE_TARGET = 5, 12, 1.5e-2
SNR_OFFSETS = [-1.5, -0.75, 0.0, 0.7, 1.4, 2.2]                 # around the BER~1.5e-2 point
# Adaptive sampling: target a fixed number of transmitted INFO BITS per SNR point
# (not a fixed #frames).  Big-Z codes carry more bits/frame, so they need fewer
# frames for the same #errors -- this both equalises statistical confidence across
# code lengths and bounds the cost of the deep, low-BER points.  Deeper points get
# more bits (down to ~1e-5).  Frames are clamped to [MIN,MAX].
TARGET_BITS = [2.0e3, 4.0e3, 1.5e4, 1.2e5, 6.0e5, 2.0e6]
MIN_FRAMES, MAX_FRAMES = 8, 3000
EVAL_SEED = 2025
BER_FLOOR = 1e-7
RESULT_FN = "results_qam.json"


def frames_for(target_bits, K):
    return int(np.clip(math.ceil(target_bits / K), MIN_FRAMES, MAX_FRAMES))


def W_for(w):
    return max(6, w + 2)


def make_cells():
    cells = []
    for Z in (16, 32, 64):                                      # code length (few hundred -> few thousand)
        for M in MODS:
            for r in RATES:
                cells.append(dict(M=M, rate=r, w=3, L=50, Z=Z, mp=RATE_MP[r],
                                  W=W_for(3), group=f"main_Z{Z}"))
    for M in MODS:                                              # coupling-depth sweep
        for w in (3, 4, 6, 8, 12, 16):
            cells.append(dict(M=M, rate=0.75, w=w, L=100, Z=32, mp=RATE_MP[0.75],
                              W=W_for(w), group="wsweep"))
    for M in MODS:                                             # chain-length sweep
        for L in (30, 100, 200):
            cells.append(dict(M=M, rate=0.75, w=4, L=L, Z=32, mp=RATE_MP[0.75],
                              W=W_for(4), group="lsweep"))
    return cells


def cell_key(c):
    return f"M{c['M']}_R{c['rate']}_w{c['w']}_L{c['L']}_Z{c['Z']}_{c['group']}"


# --------------------------------------------------------------------------- #
#  QAM evaluation worker (caches component code + channel per worker)
# --------------------------------------------------------------------------- #
_COMP, _CHAN = {}, {}


def _worker_qam(task):
    cfg_d, assign, M, ebn0, frame_seed, lo, hi = task
    cfg = R.Config(**cfg_d)
    ck = (cfg.bg, cfg.ils, cfg.Z, cfg.mp)
    comp = _COMP.get(ck)
    if comp is None:
        comp = NRLDPCCode(cfg.bg, cfg.ils, cfg.Z, mp=cfg.mp); _COMP[ck] = comp
    sc = SCLDPCCode(comp, w=cfg.w, L=cfg.L, assign=np.asarray(assign, dtype=np.int64))
    chan = _CHAN.get(M)
    if chan is None:
        chan = qam.QAMChannel(M); _CHAN[M] = chan
    sigma = qam.ebn0_to_sigma_qam(ebn0, sc.rate, M)
    be = bits = fe = 0
    for idx in range(lo, hi):
        rng = np.random.default_rng([int(frame_seed), int(idx)])     # CRN per (seed,idx)
        info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
        cw, _ = sc.encode(info)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = chan.transmit(cw[sc.tx_mask], sigma, rng)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=cfg.W, max_iter=cfg.max_iter, alpha=cfg.alpha)
        err = int((sc.extract_info(hard) != info).sum())
        be += err; bits += info.size; fe += int(err > 0)
    return be, bits, fe, hi - lo


def eval_point(cfg, assign, M, ebn0, n_frames, pool, chunks):
    chunks = max(1, min(chunks, n_frames))
    bounds = np.linspace(0, n_frames, chunks + 1).astype(int)
    cfg_d = asdict(cfg)
    aa = np.asarray(assign, dtype=np.int64)
    tasks = []
    for ci in range(chunks):
        lo, hi = int(bounds[ci]), int(bounds[ci + 1])
        if hi > lo:
            tasks.append((cfg_d, aa, M, ebn0, EVAL_SEED, lo, hi))
    raw = pool.map(_worker_qam, tasks)
    be = bits = fe = nf = 0
    for a, b, c, d in raw:
        be += a; bits += b; fe += c; nf += d
    return {"ber": be / max(bits, 1), "fer": fe / max(nf, 1),
            "bit_err": be, "n_info": bits, "frame_err": fe, "n_frames": nf}


def probe_snr(cfg, assign, M, rate, pool):
    """Bisect SNR to where the canonical construction sits at BER~PROBE_TARGET."""
    center = SNR_BASE[M] + 10.0 * (rate - 0.5)
    lo, hi = center - 3.5, center + 6.0
    nw = R.n_workers()
    for _ in range(PROBE_ITERS):
        mid = round((lo + hi) / 2.0, 2)
        m = eval_point(cfg, assign, M, mid, PROBE_FRAMES, pool, nw)
        if m["ber"] < PROBE_TARGET:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2.0, 2)


def run_cell(cell, pool):
    cfg = R.Config(bg=BG, ils=ILS, Z=cell["Z"], mp=cell["mp"], w=cell["w"], L=cell["L"],
                   W=cell["W"], max_iter=MAXIT, alpha=ALPHA)
    sc0 = cfg.build()
    sc_rate, K = sc0.rate, sc0.K
    rr = R.round_robin_assign(cfg)
    snr0 = probe_snr(cfg, rr, cell["M"], cell["rate"], pool)
    snrs = [round(snr0 + d, 2) for d in SNR_OFFSETS]
    nfr = [frames_for(tb, K) for tb in TARGET_BITS]
    constructions = [("round_robin", rr)]
    if cell["group"] == "main_Z32":                            # reference only on the main grid
        constructions.append(("random_seed0", R.random_assign(cfg, np.random.default_rng(0))))
    nw = R.n_workers()
    curves = {}
    for name, a in constructions:
        xs, bers, fers = [], [], []
        for snr, nf in zip(snrs, nfr):
            m = eval_point(cfg, a, cell["M"], snr, nf, pool, nw)
            xs.append(snr); bers.append(m["ber"]); fers.append(m["fer"])
        curves[name] = {"x": xs, "y": bers, "fer": fers}
    return {"key": cell_key(cell), "M": cell["M"], "rate": cell["rate"], "sc_rate": sc_rate,
            "w": cell["w"], "L": cell["L"], "Z": cell["Z"], "mp": cell["mp"], "W": cell["W"],
            "group": cell["group"], "se": qam.bits_per_symbol(cell["M"]) * sc_rate, "K": K,
            "frames": nfr, "snr0": snr0, "curves": curves}


def run_all():
    cells = make_cells()
    done = {}
    if os.path.exists(RESULT_FN):                              # resume
        try:
            prev = json.load(open(RESULT_FN))
            done = {c["key"]: c for c in prev.get("cells", [])}
        except Exception:
            done = {}
    out = {"config": {"bg": BG, "ils": ILS, "rate_mp": RATE_MP, "mods": MODS,
                      "snr_offsets": SNR_OFFSETS, "maxit": MAXIT, "alpha": ALPHA},
           "cells": list(done.values())}
    t_all = time.time()
    with mp.Pool(R.n_workers()) as pool:
        print(f"QAM sweep: {len(cells)} cells, {R.n_workers()} workers "
              f"({len(done)} already done)", flush=True)
        for i, cell in enumerate(cells):
            k = cell_key(cell)
            if k in done:
                print(f"[{i+1}/{len(cells)}] skip (done) {k}", flush=True)
                continue
            t0 = time.time()
            try:
                res = run_cell(cell, pool)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  !! cell {k} failed: {e}", flush=True)
                continue
            out["cells"].append(res); done[k] = res
            json.dump(out, open(RESULT_FN, "w"))
            yend = res["curves"]["round_robin"]["y"][-1]
            print(f"[{i+1}/{len(cells)}] {cell['group']} M{cell['M']} R{cell['rate']} "
                  f"w{cell['w']} L{cell['L']} Z{cell['Z']}  scR={res['sc_rate']:.3f} "
                  f"snr0={res['snr0']}  rrBER@+2.5={yend:.2e}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"DONE all cells in {time.time()-t_all:.0f}s -> {RESULT_FN}", flush=True)


# --------------------------------------------------------------------------- #
#  plotting
# --------------------------------------------------------------------------- #
MOD_LAB = {4: "QPSK", 16: "16-QAM", 64: "64-QAM", 256: "256-QAM"}
MOD_COL = {4: "#1f77b4", 16: "#2ca02c", 64: "#ff7f0e", 256: "#d62728"}
RATE_COL = {0.5: "#1f77b4", 0.667: "#2ca02c", 0.75: "#ff7f0e", 0.833: "#9467bd", 0.875: "#d62728"}


def _ebn0_at_ber(curve, target):
    xs, ys = curve["x"], [max(y, BER_FLOOR) for y in curve["y"]]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 >= target >= y1:                                 # descending crossing
            t = (math.log10(target) - math.log10(y0)) / (math.log10(y1) - math.log10(y0) - 1e-30)
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def make_plots():
    if not os.path.exists(RESULT_FN):
        print(f"no {RESULT_FN}"); return
    cells = json.load(open(RESULT_FN))["cells"]
    by = {c["key"]: c for c in cells}
    g = lambda pred: [c for c in cells if pred(c)]

    def waterfall(series, title, path):
        plotting.semilogy(series, xlabel="info Eb/N0 [dB]", ylabel="info BER",
                          title=title, path=path)

    # 1) modulation comparison: per (Z, rate) overlay the 4 modulations (round_robin)
    for Z in (16, 32, 64):
        for r in RATES:
            ser = []
            for M in MODS:
                cs = g(lambda c: c["group"] == f"main_Z{Z}" and c["M"] == M and c["rate"] == r)
                if cs:
                    cu = cs[0]["curves"]["round_robin"]
                    ser.append({"x": cu["x"], "y": cu["y"], "label": MOD_LAB[M], "color": MOD_COL[M]})
            if ser:
                waterfall(ser, f"SC-LDPC over QAM/AWGN  R={r}  Z={Z}  (w=3,L=50)",
                          f"exp_qam_mod_Z{Z}_R{int(r*1000)}.svg")

    # 2) rate comparison: at Z=32 per modulation overlay the 5 rates
    for M in MODS:
        ser = []
        for r in RATES:
            cs = g(lambda c: c["group"] == "main_Z32" and c["M"] == M and c["rate"] == r)
            if cs:
                cu = cs[0]["curves"]["round_robin"]
                ser.append({"x": cu["x"], "y": cu["y"], "label": f"R={r}", "color": RATE_COL[r]})
        if ser:
            waterfall(ser, f"SC-LDPC {MOD_LAB[M]}/AWGN  rate sweep  (Z=32,w=3,L=50)",
                      f"exp_qam_rate_M{M}.svg")

    # 3) code-length effect: per modulation at R=1/2 overlay Z in {16,32,64}
    for M in MODS:
        for r in (0.5, 0.75):
            ser = []
            for Z, col in zip((16, 32, 64), ("#1f77b4", "#2ca02c", "#d62728")):
                cs = g(lambda c: c["group"] == f"main_Z{Z}" and c["M"] == M and c["rate"] == r)
                if cs:
                    cu = cs[0]["curves"]["round_robin"]
                    ser.append({"x": cu["x"], "y": cu["y"],
                                "label": f"Z={Z} (N~{cs[0]['mp']+22}*Z)", "color": col})
            if ser:
                waterfall(ser, f"{MOD_LAB[M]}  R={r}  code-length (Z) sweep (w=3,L=50)",
                          f"exp_qam_len_M{M}_R{int(r*1000)}.svg")

    # 4) coupling-depth (w) sweep: per modulation overlay w
    wcols = {3: "#1f77b4", 4: "#2ca02c", 6: "#ff7f0e", 8: "#9467bd", 12: "#8c564b", 16: "#d62728"}
    for M in MODS:
        ser = []
        for w in (3, 4, 6, 8, 12, 16):
            cs = g(lambda c: c["group"] == "wsweep" and c["M"] == M and c["w"] == w)
            if cs:
                cu = cs[0]["curves"]["round_robin"]
                ser.append({"x": cu["x"], "y": cu["y"],
                            "label": f"w={w} (scR={cs[0]['sc_rate']:.2f})", "color": wcols[w]})
        if ser:
            waterfall(ser, f"{MOD_LAB[M]}  coupling-depth w sweep  (R=3/4,Z=32,L=100)",
                      f"exp_qam_wsweep_M{M}.svg")

    # 5) chain-length (L) sweep: per modulation overlay L
    lcols = {30: "#1f77b4", 100: "#2ca02c", 200: "#d62728"}
    for M in MODS:
        ser = []
        for L in (30, 100, 200):
            cs = g(lambda c: c["group"] == "lsweep" and c["M"] == M and c["L"] == L)
            if cs:
                cu = cs[0]["curves"]["round_robin"]
                ser.append({"x": cu["x"], "y": cu["y"],
                            "label": f"L={L} (scR={cs[0]['sc_rate']:.2f})", "color": lcols[L]})
        if ser:
            waterfall(ser, f"{MOD_LAB[M]}  chain-length L sweep  (R=3/4,Z=32,w=4)",
                      f"exp_qam_lsweep_M{M}.svg")

    # 6) summary: required Eb/N0 @ BER=1e-4 vs spectral efficiency (Z=32 main grid)
    tgt = 1e-4
    ser = []
    for M in MODS:
        xs, ys = [], []
        for r in RATES:
            cs = g(lambda c: c["group"] == "main_Z32" and c["M"] == M and c["rate"] == r)
            if cs:
                e = _ebn0_at_ber(cs[0]["curves"]["round_robin"], tgt)
                if e is not None:
                    xs.append(cs[0]["se"]); ys.append(e)
        if xs:
            ser.append({"x": xs, "y": ys, "label": MOD_LAB[M], "color": MOD_COL[M]})
    if ser:
        plotting.linear(ser, xlabel="spectral efficiency  m*R [bits/symbol]",
                        ylabel="info Eb/N0 @ BER=1e-4 [dB]",
                        title="SC-LDPC over QAM/AWGN: SNR cost of order x rate (Z=32)",
                        path="exp_qam_summary_se.svg")

    _export_csv(cells)
    print("wrote exp_qam_*.svg + results_qam_finals.csv")


def _export_csv(cells):
    import csv
    with open("results_qam_finals.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["key", "group", "M", "mod", "rate", "sc_rate", "se", "w", "L", "Z", "W",
                     "snr0", "ebn0@1e-4", "construction", "ebn0_grid", "ber_grid", "fer_grid"])
        for c in sorted(cells, key=lambda c: c["key"]):
            for name, cu in c["curves"].items():
                e4 = _ebn0_at_ber(cu, 1e-4)
                wr.writerow([c["key"], c["group"], c["M"], MOD_LAB[c["M"]], c["rate"],
                             f"{c['sc_rate']:.4f}", f"{c['se']:.3f}", c["w"], c["L"], c["Z"], c["W"],
                             c["snr0"], f"{e4:.2f}" if e4 is not None else "",
                             name, "|".join(f"{x}" for x in cu["x"]),
                             "|".join(f"{y:.3e}" for y in cu["y"]),
                             "|".join(f"{x:.3f}" for x in cu["fer"])])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run_all()
    elif cmd == "plot":
        make_plots()
    elif cmd == "cells":
        cs = make_cells()
        print(f"{len(cs)} cells")
        for grp in sorted(set(c["group"] for c in cs)):
            print(f"  {grp}: {sum(1 for c in cs if c['group']==grp)}")
    else:
        print(__doc__)
