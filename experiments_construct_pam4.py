"""Large-scale RL construction optimisation for LONG, HIGH-RATE SC-LDPC over a
PAM4 + RRC pulse-shaped waveform channel -- the harder, lower-margin sibling of
``experiments_construct_big.py`` (which optimises over plain BPSK/AWGN).

Why PAM4 is harder
------------------
Component code, edge spreading and windowed decoder are identical; only the
*channel* changes: coded bits -> Gray PAM4 -> upsample sps -> RRC(beta) shaping ->
AWGN on the waveform -> matched RRC + downsample -> exact (log-sum-exp) soft LLR
demap -> windowed BP.  4-ary signalling packs 2 bits/symbol, so at a fixed info
Eb/N0 per-bit reliability is lower and the waterfall sits ~3-4 dB above BPSK -- the
construction (which systematic edge -> which coupling component) has *more* room to
help, which is what we test with RL.

BER-centric optimisation (the key difference from the BPSK big run)
-------------------------------------------------------------------
These are LONG codes (Z=64 -> K ~ 1e4 info bits): the FRAME error rate is a near
step function of SNR (good codes FER~0, bad codes FER~1, almost nothing between),
so FER gives RL essentially no gradient.  We therefore drive the whole search by
BIT error rate: reward = -log10(BER); CEM elites / champions ranked by BER; and the
training SNR is chosen (bisection) where the median random construction sits at
BER ~ 3e-3 -- deep enough in the waterfall that constructions are clearly separable.

Reuse of the RL machinery
-------------------------
Policies (FeaturePolicy / PerEdgePolicy) and the parallel end-to-end evaluation
plumbing come from ``rl_construct`` unchanged; we (a) swap in a PAM4 evaluation
worker (``R._worker = _worker_pam4``) so every ``R.eval_*`` routes frames through
the PAM4+RRC chain, and (b) re-implement the 3 training loops here BER-centrically.
Common random numbers (every code in a batch scored on identical frames+noise) is
preserved by seeding a per-frame RNG from (frame_seed, idx): the transmitted-bit
count is constant across constructions of a config, so the same waveform noise is
drawn for every code on a given frame.

Regime (3GPP TS 38.212, BG1, Kb=22):
  * code length > 1000 for ALL rates -> Z=64 (component len (Kb+mp)*Z = 1728..2944)
  * code rates 1/2, 2/3, 3/4, 5/6, 7/8 -> BG1 mp in {24,13,9,6,5}
  * coupling memory w in {1,2,3};  chain length L tens-to-hundreds (validated by L-sweep)

Runs as PARALLEL slices on one big pod (each slice = its own process + pool of
EXP_WORKERS cores); the rate x w grid is cost-balanced (LPT) across slices so all
cores stay busy:
    python3 experiments_construct_pam4.py slice 0 ... slice 5     # run all in parallel
    python3 experiments_construct_pam4.py plot                    # merge -> figures/CSVs
"""
from __future__ import annotations
import json
import sys
import time
import multiprocessing as mp
import numpy as np

import outpaths as OP
import plotting
import rl_construct as R
import pam4_rrc as p4
from sc_ldpc import SCLDPCCode

# ----------------------------------------------------------------------------- #
#  regime
# ----------------------------------------------------------------------------- #
BG, ILS, Z, W_DEC, MAXIT, ALPHA = 1, 0, 64, 6, 12, 0.8     # Z=64 -> all rates >1000 bits
OPT_L = 12                                                  # chain length while optimising (edge spreading is L-independent)
RATE_MP = {0.5: 24, 0.667: 13, 0.75: 9, 0.833: 6, 0.875: 5}    # BG1 (Kb=22) rate matching
RATES = [0.5, 0.667, 0.75, 0.833, 0.875]
WS = [1, 2, 3]                                              # coupling memory values
BETA, SPAN, SPS = 0.1, 10, 4                                # RRC roll-off / span / samples-per-symbol

# budgets (tuned so each slice saturates its pool and the whole grid finishes in ~1h;
# BER at the 3e-3 operating point is well-estimated even with few frames, so frames stay modest)
OPT_STEPS, OPT_BATCH, OPT_FRAMES = 12, 40, 8
VAL_FRAMES, FINAL_FRAMES = 40, 300
PROBE_N, PROBE_FRAMES = 36, 10
BER_FLOOR = 2e-6
TARGET_BER = 3e-3                                           # operating point: median random-construction BER
BISECT_ITERS = 5
SNR_BRACKET = {0.5: (1.0, 5.0), 0.667: (2.0, 6.0), 0.75: (3.0, 7.0),
               0.833: (4.0, 8.0), 0.875: (5.0, 9.5)}        # Eb/N0 search bracket per rate
BER_OFFSETS = [-0.6, 0.0, 0.6, 1.2]                         # final BER curve span around op. SNR
BER_FRAMES = [150, 220, 320, 450]
LSWEEP_LS = [30, 100, 200]
LSWEEP_OFFSETS = [0.0, 0.8]
LSWEEP_FRAMES = [120, 200]
LSWEEP_CHAMPS = ["rl", "seed0_default"]
LSWEEP_CELLS = [(0.75, 2), (0.875, 2)]     # (rate,w) whose champions get an L=30..200 sweep
                                           # (kept off the very slow R=1/2 large-L codes)

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


R._worker = _worker_pam4        # route ALL R.eval_* through the PAM4 channel


# ----------------------------------------------------------------------------- #
#  cost-balanced (LPT) partition of the rate x w grid across parallel slices
# ----------------------------------------------------------------------------- #
def _cells():
    return [(r, w) for r in RATES for w in WS]


def _weight(cell):
    r, w = cell
    return (22 + RATE_MP[r]) * Z * (1.0 + 0.12 * (w - 1))


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


def _logber(m):
    return -np.log10(max(m["ber"], BER_FLOOR))


def _validate(c, assign, snr, nframes, seed, pool):
    """Re-score a champion on a big frame bank, split across all cores."""
    return R.eval_assignments(c, [assign], snr, seed, nframes, pool=pool,
                              frame_chunks=R.n_workers())[0]


# ----------------------------------------------------------------------------- #
#  operating-point selection: bisection on SNR to median random-construction BER
# ----------------------------------------------------------------------------- #
def pick_snr(c, rate, pool):
    lo, hi = SNR_BRACKET[rate]
    rng = np.random.default_rng(0)
    assigns = [R.random_assign(c, rng) for _ in range(PROBE_N)]

    def med_ber(snr):
        ms = R.eval_batch(c, assigns, snr, 1, PROBE_FRAMES, pool=pool)
        return float(np.median([m["ber"] for m in ms]))

    for _ in range(BISECT_ITERS):
        mid = round((lo + hi) / 2.0, 2)
        mb = med_ber(mid)
        if mb < TARGET_BER:        # too clean -> need a lower SNR
            hi = mid
        else:                      # too noisy -> need a higher SNR
            lo = mid
    snr = round((lo + hi) / 2.0, 2)
    return snr, med_ber(snr)


# ----------------------------------------------------------------------------- #
#  BER-centric training loops (policies + eval reused from rl_construct)
# ----------------------------------------------------------------------------- #
def train_pg(c, policy, snr, steps, batch, frames, pool, val_frames, val_seed,
             base_seed, tag, log_every=5):
    feature = isinstance(policy, R.FeaturePolicy)
    hist = {"evals": [], "mean_reward": [], "best_val_ber": [], "best_val_fer": []}
    best = {"ber": np.inf, "fer": np.inf, "assign": None}
    seen = 0
    for step in range(1, steps + 1):
        fs = base_seed + step
        traces, assigns, ps = [], [], []
        for _ in range(batch):
            if feature:
                a, tr = policy.rollout(); traces.append(tr)
            else:
                a, p = policy.sample(); ps.append(p)
            assigns.append(a)
        ms = R.eval_batch(c, assigns, snr, fs, frames, pool=pool)
        seen += batch
        rew = np.array([_logber(m) for m in ms])
        adv = rew - rew.mean()
        if adv.std() > 1e-9:
            adv = adv / (adv.std() + 1e-9)
        if feature:
            policy.update(list(zip(traces, adv)))
        else:
            policy.update([(assigns[i], ps[i], adv[i]) for i in range(batch)])
        bi = int(np.argmin([m["ber"] for m in ms]))
        vm = _validate(c, assigns[bi], snr, val_frames, val_seed, pool)
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"], "assign": np.asarray(assigns[bi]).copy()}
        hist["evals"].append(seen); hist["mean_reward"].append(float(rew.mean()))
        hist["best_val_ber"].append(best["ber"]); hist["best_val_fer"].append(best["fer"])
        if step % log_every == 0 or step == 1:
            print(f"  [{tag}] step {step:3d} meanR={rew.mean():+.2f} "
                  f"batchBER[min={min(m['ber'] for m in ms):.2e}] "
                  f"bestVAL_BER={best['ber']:.2e} (FER={best['fer']:.2f})", flush=True)
    return hist, best


def train_cem(c, snr, steps, batch, frames, pool, val_frames, val_seed, base_seed,
              elite_frac=0.3, smooth=0.7, seed=7, log_every=5):
    _, _, E = R.edge_meta(c); C = R.n_components(c)
    rng = np.random.default_rng(seed)
    p = np.full((E, C), 1.0 / C); n_elite = max(2, int(batch * elite_frac))
    hist = {"evals": [], "best_val_ber": [], "best_val_fer": []}
    best = {"ber": np.inf, "fer": np.inf, "assign": None}; seen = 0
    for step in range(1, steps + 1):
        fs = base_seed + step
        assigns = [np.array([rng.choice(C, p=p[e]) for e in range(E)]) for _ in range(batch)]
        ms = R.eval_batch(c, assigns, snr, fs, frames, pool=pool); seen += batch
        bers = np.array([m["ber"] for m in ms])
        elite = np.argsort(bers)[:n_elite]
        freq = np.zeros((E, C))
        for idx in elite:
            freq[np.arange(E), assigns[idx]] += 1.0
        freq /= n_elite
        p = smooth * p + (1 - smooth) * freq
        p = np.clip(p, 1e-3, None); p /= p.sum(axis=1, keepdims=True)
        cand = assigns[int(np.argmin(bers))]
        vm = _validate(c, cand, snr, val_frames, val_seed, pool)
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"], "assign": np.asarray(cand).copy()}
        hist["evals"].append(seen); hist["best_val_ber"].append(best["ber"])
        hist["best_val_fer"].append(best["fer"])
        if step % log_every == 0 or step == 1:
            print(f"  [CEM] step {step:3d} batchBER[min={bers.min():.2e}] "
                  f"bestVAL_BER={best['ber']:.2e}", flush=True)
    return hist, best


def train_random(c, snr, steps, batch, frames, pool, val_frames, val_seed, base_seed,
                 seed=11, log_every=5):
    _, _, E = R.edge_meta(c); C = R.n_components(c)
    rng = np.random.default_rng(seed)
    hist = {"evals": [], "best_val_ber": [], "best_val_fer": []}
    best = {"ber": np.inf, "fer": np.inf, "assign": None}; seen = 0
    for step in range(1, steps + 1):
        fs = base_seed + step
        assigns = [rng.integers(0, C, size=E) for _ in range(batch)]
        ms = R.eval_batch(c, assigns, snr, fs, frames, pool=pool); seen += batch
        bers = np.array([m["ber"] for m in ms])
        cand = assigns[int(np.argmin(bers))]
        vm = _validate(c, cand, snr, val_frames, val_seed, pool)
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"], "assign": np.asarray(cand).copy()}
        hist["evals"].append(seen); hist["best_val_ber"].append(best["ber"])
        hist["best_val_fer"].append(best["fer"])
        if step % log_every == 0 or step == 1:
            print(f"  [RND] step {step:3d} batchBER[min={bers.min():.2e}] "
                  f"bestVAL_BER={best['ber']:.2e}", flush=True)
    return hist, best


# ----------------------------------------------------------------------------- #
#  BER curve + one construction-optimisation cell
# ----------------------------------------------------------------------------- #
def ber_curve(c, assign, snrs, frames, pool, chunks=None):
    chunks = chunks or R.n_workers()
    xs, bers, fers = [], [], []
    for snr, nf in zip(snrs, frames):
        m = R.eval_assignments(c, [assign], snr, 2025, nf, pool=pool, frame_chunks=chunks)[0]
        xs.append(round(snr, 2)); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def run_cell(rate, w, pool):
    """One (rate,w) BER-centric construction-optimisation cell over PAM4."""
    t0 = time.time()
    c = cfg(rate, w, OPT_L)
    rt = c.build().rate
    _, _, E = R.edge_meta(c)
    snr, med = pick_snr(c, rate, pool)
    print(f"\n=== R={rate} (sc rate {rt:.3f}) w={w} Z={Z}  E={E}  space {w+1}^{E}  "
          f"L={OPT_L}  PAM4 b{BETA} sps{SPS}  train@{snr}dB (med random BER {med:.2e}) ===", flush=True)

    pol = R.FeaturePolicy(c, lr=0.15, ent=0.02, seed=0)
    h_rl, best_rl = train_pg(c, pol, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool,
                             VAL_FRAMES, 99, 1000, "PG")
    pol_e = R.PerEdgePolicy(E, w + 1, lr=0.2, ent=0.01, seed=0)
    h_e, best_e = train_pg(c, pol_e, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool,
                           VAL_FRAMES, 99, 4000, "PGe")
    h_c, best_c = train_cem(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool,
                            VAL_FRAMES, 99, 2000)
    h_r, best_r = train_random(c, snr, OPT_STEPS, OPT_BATCH, OPT_FRAMES, pool,
                               VAL_FRAMES, 99, 3000)
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
        if name != "rl_peredge":
            curves[name] = ber_curve(c, a, snrs, BER_FRAMES, pool)
        print(f"  {name:14s} BER={m['ber']:.3e} FER={m['fer']:.3f} n4={st['n4']}", flush=True)
    print(f"  [cell done in {time.time()-t0:.0f}s]", flush=True)
    return {"rate": rate, "sc_rate": rt, "w": w, "Z": Z, "mp": RATE_MP[rate], "E": int(E),
            "train_snr": snr, "finals": finals, "stats": stats, "curves": curves,
            "hist": {"rl": h_rl, "rl_peredge": h_e, "cem": h_c, "random": h_r},
            "theta_rl": pol.theta.tolist(),
            "champions": {k: _arr(v) for k, v in champions.items()}}


def run_lsweep(rate, w, out, pool):
    key = f"R{rate}_w{w}"
    cell = out["cells"].get(key)
    if cell is None:
        print(f"  (lsweep: no optimised cell {key}; skipping)", flush=True)
        return
    snr0 = cell["train_snr"]
    snrs = [snr0 + d for d in LSWEEP_OFFSETS]
    champs = {n: np.asarray(cell["champions"][n], dtype=np.int64) for n in LSWEEP_CHAMPS}
    lsweep = {}
    print(f"\n=== L-sweep @ R={rate} w={w} Z={Z} (champions vs L) ===", flush=True)
    for L in LSWEEP_LS:
        cL = cfg(rate, w, L)
        chunks = min(R.n_workers(), 20)
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
                                  "target_ber": TARGET_BER, "rate_mp": RATE_MP, "ws": WS},
           "cells": {}}
    fn = f"results_pam4big_{k}.json"
    with mp.Pool(R.n_workers()) as pool:
        print(f"slice {k}: cells {cells}  ({R.n_workers()} workers, PAM4, BER-centric)", flush=True)
        for rate, w in cells:
            try:
                out["cells"][f"R{rate}_w{w}"] = run_cell(rate, w, pool)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  !! cell R{rate} w{w} failed: {e}", flush=True)
            json.dump(out, open(OP.route(fn), "w"))
        for (lr, lw) in LSWEEP_CELLS:
            if (lr, lw) in cells:
                try:
                    run_lsweep(lr, lw, out, pool)
                except Exception as e:
                    print(f"  !! lsweep R{lr} w{lw} failed: {e}", flush=True)
                json.dump(out, open(OP.route(fn), "w"))
    print(f"\nslice {k} done in {time.time()-t0:.0f}s -> {fn}", flush=True)


# --------------------------------------------------------------------------- #
#  plotting (run after pulling results_pam4big_*.json)
# --------------------------------------------------------------------------- #
def _merge():
    out = {"cells": {}, "lsweeps": []}
    for k in range(N_SLICES):
        try:
            d = json.load(open(OP.route(f"results_pam4big_{k}.json")))
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
    # Fig 1: RL-vs-equal-budget-random BER advantage vs rate, one line per w
    series = []
    for w in WS:
        xs, ys = [], []
        for rate in rates:
            c = cells.get(f"R{rate}_w{w}")
            if not c:
                continue
            rl = c["finals"]["rl"]["ber"]; rnd = c["finals"]["random_search"]["ber"]
            xs.append(rate); ys.append(rnd / max(rl, BER_FLOOR))
        if xs:
            series.append({"x": xs, "y": ys, "label": f"w={w}",
                           "color": ["#1f77b4", "#d62728", "#2ca02c"][w - 1]})
    plotting.linear(series, xlabel="code rate R", ylabel="random_BER / RL_BER  (>1 = RL better)",
                    title="RL vs equal-budget random search over PAM4+RRC (Z=64 long codes)",
                    path="exp_pam4big_rl_vs_random.svg")
    # Fig 2: per-cell BER curves + BER learning curves
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
                ls = [{"x": h[m]["evals"], "y": [max(b, BER_FLOOR) for b in h[m]["best_val_ber"]],
                       "label": LAB.get(m, m), "color": COL.get(m, "#333")}
                      for m in ["rl", "rl_peredge", "cem", "random"] if m in h]
                plotting.semilogy(ls, xlabel="# construction evaluations", ylabel="best val BER",
                                  title=f"PAM4 learning curves R={rate} w={w} (sample efficiency)",
                                  path=f"exp_pam4big_learn_R{int(rate*1000)}_w{w}.svg")
    # Fig 3: L-sweep BER (threshold saturation)
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
    json.dump(out, open(OP.route("results_pam4big_merged.json"), "w"))
    print("wrote exp_pam4big_*.svg, results_pam4big_merged.json, results_pam4big_{finals,hist}.csv")


def _export_tables(out):
    import csv
    cells = out["cells"]
    with open(OP.route("results_pam4big_finals.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["cell", "rate", "w", "Z", "E", "train_snr", "method", "ber", "fer", "n4"])
        for k, c in sorted(cells.items()):
            for m, fv in c["finals"].items():
                wr.writerow([k, c["rate"], c["w"], c.get("Z", Z), c["E"], c["train_snr"], m,
                             f"{fv['ber']:.4e}", f"{fv['fer']:.4f}", c["stats"].get(m, {}).get("n4", "")])
    with open(OP.route("results_pam4big_hist.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cell", "rate", "w", "method", "eval", "mean_reward", "best_val_ber"])
        for k, c in sorted(cells.items()):
            for m, h in c.get("hist", {}).items():
                ev = h.get("evals", []); mr = h.get("mean_reward", []); bb = h.get("best_val_ber", [])
                for i in range(len(ev)):
                    wr.writerow([k, c["rate"], c["w"], m, ev[i],
                                 f"{mr[i]:.4f}" if i < len(mr) else "",
                                 f"{bb[i]:.4e}" if i < len(bb) else ""])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "slice":
        run_slice(int(sys.argv[2]))
    elif cmd == "plot":
        make_plots()
    elif cmd == "slices":
        for i, s in enumerate(SLICES):
            print(i, s, round(sum(_weight(c) for c in s), 1))
    else:
        print(__doc__)
