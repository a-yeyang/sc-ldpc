"""Large-scale RL construction optimisation for LONG, HIGH-RATE SC-LDPC over a
PAM4 + RRC pulse-shaped waveform channel (the harder, lower-SNR-margin sibling of
``experiments_construct_big.py``, which optimises over plain BPSK/AWGN).

Why PAM4 is harder
------------------
The component code, edge spreading and windowed decoder are identical; only the
*channel* changes: coded bits -> Gray PAM4 -> upsample sps -> RRC(beta) shaping ->
AWGN on the waveform -> matched RRC + downsample -> exact (log-sum-exp) soft LLR
demap -> windowed BP.  4-ary signalling packs 2 bits/symbol, so at a fixed info
Eb/N0 the per-bit reliability is lower and the waterfall sits ~3-4 dB higher than
BPSK -- the construction (which systematic edge goes to which coupling component)
therefore has *more* room to help, which is exactly what we test with RL.

Reuse of the RL machinery
-------------------------
All policy-gradient / CEM / random-search training and the end-to-end evaluation
plumbing live in ``rl_construct`` and are channel-agnostic *except* for the single
per-construction evaluation worker ``rl_construct._worker`` (hard-wired to BPSK).
We swap in a PAM4 worker (``R._worker = _worker_pam4``) so every ``R.eval_*`` /
``R.train_*`` call routes frames through the PAM4+RRC chain with no other change.
Common-random-numbers (every code in a batch scored on identical frames+noise) is
preserved by seeding a per-frame RNG deterministically from (frame_seed, idx): the
transmitted-bit count is constant across constructions of one config, so the same
waveform noise vector is drawn for every code on a given frame.

Regime (3GPP TS 38.212, BG1, Kb=22):
  * code length > 1000 for ALL rates  ->  Z=64  (component len (Kb+mp)*Z = 1728..2944)
  * wireless code rates: 1/2, 2/3, 3/4, 5/6, 7/8  ->  BG1 mp in {24,13,9,6,5}
  * coupling memory w in {1,2,3}      ->  edge-spreading space (w+1)^E
  * coupling chain length L: tens-to-hundreds (windowed decode; validated by L-sweep)

Distributed across PARALLEL slices on ONE big pod (each slice = its own process +
pool of EXP_WORKERS cores); the rate x w grid is cost-balanced (LPT) across slices
so all cores stay busy:
    python3 experiments_construct_pam4.py slice 0      # ... one cost-balanced cell group
    ...                                                 #     (run all N_SLICES in parallel)
    python3 experiments_construct_pam4.py slice 5
    python3 experiments_construct_pam4.py plot          # merge results_pam4big_*.json -> figures
"""
from __future__ import annotations
import json
import sys
import time
import multiprocessing as mp
import numpy as np

import plotting
import rl_construct as R
import pam4_rrc as p4
from sc_ldpc import SCLDPCCode

# ----------------------------------------------------------------------------- #
#  regime
# ----------------------------------------------------------------------------- #
BG, ILS, Z, W_DEC, MAXIT, ALPHA = 1, 0, 64, 6, 12, 0.8     # Z=64 -> all rates >1000 bits
OPT_L = 30                                                  # chain length while optimising
RATE_MP = {0.5: 24, 0.667: 13, 0.75: 9, 0.833: 6, 0.875: 5}    # BG1 (Kb=22) rate matching
RATES = [0.5, 0.667, 0.75, 0.833, 0.875]
WS = [1, 2, 3]                                              # coupling memory values
BETA, SPAN, SPS = 0.1, 10, 4                                # RRC roll-off / span / samples-per-sym

# budgets: PAM4 frames are expensive -> modest; batch is a few worker-waves per step
OPT_STEPS, OPT_BATCH, OPT_FRAMES = 12, 40, 8
VAL_FRAMES, FINAL_FRAMES = 40, 300
PROBE_FRAMES, PROBE_N = 10, 40
# PAM4 waterfalls sit ~3-4 dB above BPSK -> per-rate info-Eb/N0 candidate grids
SNR_CAND = {0.5: [3.0, 3.5, 4.0, 4.5, 5.0], 0.667: [3.5, 4.0, 4.5, 5.0, 5.5],
            0.75: [4.0, 4.5, 5.0, 5.5, 6.0], 0.833: [5.0, 5.5, 6.0, 6.5, 7.0],
            0.875: [6.0, 6.5, 7.0, 7.5, 8.0]}
BER_OFFSETS = [-0.6, 0.0, 0.6]
BER_FRAMES = [150, 250, 350]
LSWEEP_LS = [30, 50, 100, 200]
LSWEEP_FRAMES = [120, 160]
LSWEEP_OFFSETS = [0.0]
LSWEEP_CELLS = [(0.5, 2), (0.875, 2)]      # (rate,w) whose champions get an L=30..200 sweep

N_SLICES = 6                               # parallel slice-processes (sum of pools == pod cores)


# ----------------------------------------------------------------------------- #
#  PAM4 evaluation worker  (the ONLY channel-specific code; swapped into R)
# ----------------------------------------------------------------------------- #
def _worker_pam4(task):
    """Evaluate one assignment over a CRN frame range [lo,hi) through PAM4+RRC.
    Same 7-field task tuple as rl_construct._worker, so R.eval_assignments is reused
    verbatim; channel params come from this module's globals (inherited via fork)."""
    cfg_d, assign, ebn0, frame_seed, n_frames, lo, hi = task
    cfg = R.Config(**cfg_d)
    ckey = (cfg.bg, cfg.ils, cfg.Z, cfg.mp)
    comp = R._CACHE.get(ckey)
    if comp is None:
        comp = cfg.component(); R._CACHE[ckey] = comp
    sc = SCLDPCCode(comp, w=cfg.w, L=cfg.L, assign=np.asarray(assign, dtype=np.int64))
    chkey = ("pam4", BETA, SPAN, SPS)
    chan = R._CACHE.get(chkey)
    if chan is None:
        chan = p4.PAM4RRCChannel(BETA, SPAN, SPS); R._CACHE[chkey] = chan
    sigma = p4.ebn0_to_sigma_pam4(ebn0, sc.rate)
    be = bits = fe = 0
    for idx in range(lo, hi):
        rng = np.random.default_rng([int(frame_seed), int(idx)])    # CRN: identical per (seed,idx)
        info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
        cw, _ = sc.encode(info)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = chan.transmit(cw[sc.tx_mask], sigma, rng)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=cfg.W, max_iter=cfg.max_iter, alpha=cfg.alpha)
        err = int((sc.extract_info(hard) != info).sum())
        be += err; bits += info.size; fe += int(err > 0)
    return be, bits, fe, hi - lo


R._worker = _worker_pam4        # route ALL R.eval_*/R.train_* through the PAM4 channel


# ----------------------------------------------------------------------------- #
#  cost-balanced (LPT) partition of the rate x w grid across parallel slices
# ----------------------------------------------------------------------------- #
def _cells():
    return [(r, w) for r in RATES for w in WS]


def _weight(cell):
    r, w = cell
    return (22 + RATE_MP[r]) * Z * (1.0 + 0.12 * (w - 1))    # ~ component size x coupling


def _partition(cells, n):
    bins = [[] for _ in range(n)]
    load = [0.0] * n
    for c in sorted(cells, key=_weight, reverse=True):       # longest-processing-time first
        k = min(range(n), key=lambda i: load[i])
        bins[k].append(c); load[k] += _weight(c)
    return bins


SLICES = _partition(_cells(), N_SLICES)


def cfg(rate, w, L):
    return R.Config(bg=BG, ils=ILS, Z=Z, mp=RATE_MP[rate], w=w, L=L,
                    W=W_DEC, max_iter=MAXIT, alpha=ALPHA)


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


# ----------------------------------------------------------------------------- #
#  per-construction helpers (mirror experiments_construct_big.py)
# ----------------------------------------------------------------------------- #
def pick_snr(c, candidates, pool):
    """Pick the training Eb/N0 where the median random construction sits in the
    waterfall (FER ~ 0.3) so RL gets a gradient; step off-grid if the grid misses."""
    rng = np.random.default_rng(0)
    assigns = [R.random_assign(c, rng) for _ in range(PROBE_N)]
    meds = []
    for snr in candidates:
        ms = R.eval_batch(c, assigns, snr, 1, PROBE_FRAMES, pool=pool)
        meds.append(float(np.median([m["fer"] for m in ms])))
    if meds[0] < 0.05:
        return round(candidates[0] - 1.0, 2)
    if meds[-1] > 0.6:
        return round(candidates[-1] + 0.5, 2)
    return candidates[int(np.argmin([abs(m - 0.3) for m in meds]))]


def ber_curve(c, assign, snrs, frames, pool, chunks=None):
    chunks = chunks or R.n_workers()
    xs, bers, fers = [], [], []
    for snr, nf in zip(snrs, frames):
        m = R.eval_assignments(c, [assign], snr, 2025, nf, pool=pool, frame_chunks=chunks)[0]
        xs.append(round(snr, 2)); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def run_cell(rate, w, pool):
    """One (rate,w) construction-optimisation cell over the PAM4 channel."""
    t0 = time.time()
    c = cfg(rate, w, OPT_L)
    rt = c.build().rate
    _, _, E = R.edge_meta(c)
    snr = pick_snr(c, SNR_CAND[rate], pool)
    print(f"\n=== R={rate} (sc rate {rt:.3f}) w={w} Z={Z}  E={E}  space {w+1}^{E}  "
          f"L={OPT_L}  PAM4 b{BETA} sps{SPS}  train@{snr}dB ===", flush=True)

    pol = R.FeaturePolicy(c, lr=0.15, ent=0.02, seed=0)
    h_rl, best_rl = R.train_reinforce(c, pol, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES,
                                      pool=pool, val_frames=VAL_FRAMES, val_seed=99,
                                      base_seed=1000, log_every=8)
    pol_e = R.PerEdgePolicy(E, w + 1, lr=0.2, ent=0.01, seed=0)
    h_e, best_e = R.train_reinforce(c, pol_e, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES,
                                    pool=pool, val_frames=VAL_FRAMES, val_seed=99,
                                    base_seed=4000, log_every=8)
    h_c, best_c = R.train_cem(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                              val_frames=VAL_FRAMES, val_seed=99, base_seed=2000, log_every=8)
    h_r, best_r = R.train_random(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool=pool,
                                 val_frames=VAL_FRAMES, val_seed=99, base_seed=3000, log_every=8)
    champions = {"rl": best_rl["assign"], "rl_peredge": best_e["assign"],
                 "cem": best_c["assign"], "random_search": best_r["assign"],
                 "round_robin": R.round_robin_assign(c),
                 "seed0_default": R.random_assign(c, np.random.default_rng(0))}
    snrs = [snr + d for d in BER_OFFSETS]
    finals, stats, curves = {}, {}, {}
    for name, a in champions.items():
        a = np.asarray(a, dtype=np.int64)
        m = R.eval_assignments(c, [a], snr, 12345, FINAL_FRAMES, pool=pool,
                               frame_chunks=R.n_workers())[0]
        finals[name] = {"fer": m["fer"], "ber": m["ber"]}
        st = R.construction_stats(c, a)
        stats[name] = {"n4": st["n4"], "comp_load": st["comp_load"]}
        if name not in ("rl_peredge",):
            curves[name] = ber_curve(c, a, snrs, BER_FRAMES, pool)
        print(f"  {name:14s} FER={m['fer']:.4f} BER={m['ber']:.3e} n4={st['n4']}", flush=True)
    print(f"  [cell done in {time.time()-t0:.0f}s]", flush=True)
    return {"rate": rate, "sc_rate": rt, "w": w, "Z": Z, "mp": RATE_MP[rate], "E": int(E),
            "train_snr": snr, "finals": finals, "stats": stats, "curves": curves,
            "hist": {"rl": h_rl, "rl_peredge": h_e, "cem": h_c, "random": h_r},
            "theta_rl": pol.theta.tolist(),
            "champions": {k: _arr(v) for k, v in champions.items()}}


def run_lsweep(rate, w, out, pool):
    """Validate champions across chain lengths L=30..200 (threshold saturation)."""
    key = f"R{rate}_w{w}"
    cell = out["cells"].get(key)
    if cell is None:
        print(f"  (lsweep: no optimised cell {key}; skipping)", flush=True)
        return
    snr0 = cell["train_snr"]
    snrs = [snr0 + d for d in LSWEEP_OFFSETS]
    champs = {n: np.asarray(cell["champions"][n], dtype=np.int64)
              for n in ["rl", "random_search", "seed0_default"]}
    lsweep = {}
    print(f"\n=== L-sweep @ R={rate} w={w} Z={Z} (champions vs L) ===", flush=True)
    for L in LSWEEP_LS:
        cL = cfg(rate, w, L)
        chunks = min(R.n_workers(), 20)        # bound concurrent heavy (large-L) codes
        lsweep[str(L)] = {"rate": cL.build().rate}
        for n, a in champs.items():
            lsweep[str(L)][n] = ber_curve(cL, a, snrs, LSWEEP_FRAMES, pool, chunks=chunks)
        print(f"  L={L:3d} (sc rate {cL.build().rate:.3f}) done", flush=True)
    out.setdefault("lsweeps", []).append({"rate": rate, "w": w, "Z": Z, "data": lsweep})


def run_slice(k):
    t0 = time.time()
    cells = SLICES[k]
    out = {"slice": k, "config": {"bg": BG, "Z": Z, "opt_L": OPT_L, "W": W_DEC,
                                  "beta": BETA, "span": SPAN, "sps": SPS,
                                  "rate_mp": RATE_MP, "ws": WS}, "cells": {}}
    fn = f"results_pam4big_{k}.json"
    with mp.Pool(R.n_workers()) as pool:
        print(f"slice {k}: cells {cells}  ({R.n_workers()} workers, PAM4)", flush=True)
        for rate, w in cells:
            try:
                out["cells"][f"R{rate}_w{w}"] = run_cell(rate, w, pool)
            except Exception as e:
                print(f"  !! cell R{rate} w{w} failed: {e}", flush=True)
            json.dump(out, open(fn, "w"))
        for (lr, lw) in LSWEEP_CELLS:
            if (lr, lw) in cells:
                try:
                    run_lsweep(lr, lw, out, pool)
                except Exception as e:
                    print(f"  !! lsweep R{lr} w{lw} failed: {e}", flush=True)
                json.dump(out, open(fn, "w"))
    print(f"\nslice {k} done in {time.time()-t0:.0f}s -> {fn}", flush=True)


# --------------------------------------------------------------------------- #
#  plotting (run locally after pulling results_pam4big_*.json)
# --------------------------------------------------------------------------- #
def _merge():
    out = {"cells": {}, "lsweeps": []}
    for k in range(N_SLICES):
        try:
            d = json.load(open(f"results_pam4big_{k}.json"))
        except FileNotFoundError:
            continue
        out["cells"].update(d.get("cells", {}))
        out["lsweeps"].extend(d.get("lsweeps", []))
    return out


LAB = {"rl": "RL feature [ours]", "rl_peredge": "RL per-edge [ours]", "cem": "CEM",
       "random_search": "random search", "round_robin": "round-robin",
       "seed0_default": "random default (seed0)"}
COL = {"rl": "#d62728", "rl_peredge": "#ff7f0e", "cem": "#9467bd",
       "random_search": "#1f77b4", "round_robin": "#8c564b", "seed0_default": "#7f7f7f"}


def make_plots():
    out = _merge()
    cells = out["cells"]
    if not cells:
        print("no results_pam4big_*.json found"); return
    rates = sorted(set(c["rate"] for c in cells.values()))
    # Fig 1: RL-vs-equal-budget-random advantage vs rate, one line per w
    series = []
    for w in WS:
        xs, ys = [], []
        for rate in rates:
            c = cells.get(f"R{rate}_w{w}")
            if not c:
                continue
            rl = c["finals"]["rl"]["fer"]; rnd = c["finals"]["random_search"]["fer"]
            xs.append(rate); ys.append(rnd / max(rl, 1e-4))
        if xs:
            series.append({"x": xs, "y": ys, "label": f"w={w}",
                           "color": ["#1f77b4", "#d62728", "#2ca02c"][w - 1]})
    plotting.linear(series, xlabel="code rate R", ylabel="random_FER / RL_FER  (>1 = RL better)",
                    title="RL vs equal-budget random search over PAM4+RRC (Z=64 long codes)",
                    path="exp_pam4big_rl_vs_random.svg")
    # Fig 2: per-cell BER curves + learning curves
    for rate in rates:
        for w in WS:
            c = cells.get(f"R{rate}_w{w}")
            if not c:
                continue
            ser = [{"x": c["curves"][n]["x"], "y": c["curves"][n]["y"], "label": LAB[n], "color": COL[n]}
                   for n in ["rl", "cem", "random_search", "round_robin", "seed0_default"]
                   if n in c["curves"]]
            plotting.semilogy(ser, xlabel="info Eb/N0 [dB]", ylabel="info BER",
                              title=f"Long SC-LDPC over PAM4 R={rate} w={w} (BG1 Z=64, L={OPT_L})",
                              path=f"exp_pam4big_ber_R{int(rate*1000)}_w{w}.svg")
            h = c.get("hist")
            if h:
                floor = 0.5 / VAL_FRAMES
                ls = [{"x": h[m]["evals"], "y": [max(f, floor) for f in h[m]["best_val_fer"]],
                       "label": LAB.get(m, m), "color": COL.get(m, "#333")}
                      for m in ["rl", "rl_peredge", "cem", "random"] if m in h]
                plotting.semilogy(ls, xlabel="# construction evaluations", ylabel="best val FER",
                                  title=f"PAM4 learning curves R={rate} w={w} (sample efficiency)",
                                  path=f"exp_pam4big_learn_R{int(rate*1000)}_w{w}.svg")
    # Fig 3: L-sweep BER (threshold saturation) per available lsweep
    for ls in out["lsweeps"]:
        data = ls["data"]; rate = ls["rate"]; w = ls["w"]
        for champ, tag in [("rl", "RL"), ("seed0_default", "default")]:
            ser = []
            for i, L in enumerate(LSWEEP_LS):
                if str(L) in data and champ in data[str(L)]:
                    cc = data[str(L)][champ]
                    ser.append({"x": cc["x"], "y": cc["y"], "label": f"L={L}",
                                "color": ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"][i]})
            if ser:
                plotting.semilogy(ser, xlabel="info Eb/N0 [dB]", ylabel="info BER",
                                  title=f"PAM4 chain-length L sweep ({tag}, R={rate} w={w})",
                                  path=f"exp_pam4big_lsweep_R{int(rate*1000)}_w{w}_{champ}.svg")
    _export_tables(out)
    json.dump(out, open("results_pam4big_merged.json", "w"))
    print("wrote exp_pam4big_*.svg, results_pam4big_merged.json, results_pam4big_{finals,hist}.csv")


def _export_tables(out):
    import csv
    cells = out["cells"]
    with open("results_pam4big_finals.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["cell", "rate", "w", "Z", "E", "method", "fer", "ber", "n4"])
        for k, c in sorted(cells.items()):
            for m, fv in c["finals"].items():
                wr.writerow([k, c["rate"], c["w"], c.get("Z", Z), c["E"], m, f"{fv['fer']:.5f}",
                             f"{fv['ber']:.3e}", c["stats"].get(m, {}).get("n4", "")])
    with open("results_pam4big_hist.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cell", "rate", "w", "method", "eval", "mean_reward_loss", "best_val_fer"])
        for k, c in sorted(cells.items()):
            for m, h in c.get("hist", {}).items():
                ev = h.get("evals", []); mr = h.get("mean_reward", []); bf = h.get("best_val_fer", [])
                for i in range(len(ev)):
                    wr.writerow([k, c["rate"], c["w"], m, ev[i],
                                 f"{mr[i]:.4f}" if i < len(mr) else "",
                                 f"{bf[i]:.4f}" if i < len(bf) else ""])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "slice":
        run_slice(int(sys.argv[2]))
    elif cmd == "plot":
        make_plots()
    elif cmd == "slices":          # print the cost-balanced partition (for launch scripting)
        for i, s in enumerate(SLICES):
            print(i, s, round(sum(_weight(c) for c in s), 1))
    else:
        print(__doc__)
