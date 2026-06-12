"""CPU#1 -- does the RL edge-spreading construction gain hold over HIGHER-ORDER
MODULATION (16/64/256-QAM)?  All prior construction experiments are BPSK/AWGN; a
reviewer will ask whether the win is BPSK-specific.  This driver answers it.

For ONE QAM order M (so the work parallelises one pod per order) it compares six
constructions of the deep-dive SC-LDPC cell over the full BER/FER waterfall on a
complex-AWGN M-QAM channel (qam.py, exact log-sum-exp soft demap, info-bit Eb/N0):
  rl_qam        : FeaturePolicy trained NATIVELY on this M-QAM channel (the hero)
  rl_bpsk       : the BPSK-trained champion, evaluated on M-QAM (cross-channel transfer)
  cutvec        : the named literature cutting vector (best by n4), evaluated on M-QAM
  random        : best-of-budget random search scored on M-QAM
  round_robin   : naive balanced default
  seed0         : repository default single random spreading

COMPLETE DATA IS SAVED (not just plots): per-construction, per-SNR raw bit/frame
error counts (-> Clopper-Pearson CIs and BER/FER waterfalls), the assignment
vector, and n4/girth; PLUS constellation data for drawing later -- the ideal M-QAM
points with their Gray bit labels and a sample of received noisy symbols at the
mid-SNR.  Everything goes to results_qam_construct_M<M>.json.

Usage (CPU pod, EXP_WORKERS = pod cores), ONE order per pod:
  EXP_WORKERS=24 python3 experiments_qam_construct.py run 16
  EXP_WORKERS=24 python3 experiments_qam_construct.py run 64
  EXP_WORKERS=24 python3 experiments_qam_construct.py run 256
  python3 experiments_qam_construct.py agg          # printed comparison once all 3 done
Checkpoints after every (construction) so an evicted pod loses at most one.
"""
from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import asdict
from itertools import combinations_with_replacement
from multiprocessing import Pool

import numpy as np

import rl_construct as R
import qam
from sc_ldpc import SCLDPCCode

# deep-dive cell (we have its BPSK champions; small + fast over QAM too)
CFG = R.Config(bg=2, ils=0, Z=16, mp=8, w=2, L=30, W=6, max_iter=12)
BPSK_TRAIN_SNR = 2.5

# candidate info-bit Eb/N0 grids per order (rate ~0.60); the full grid IS the
# waterfall.  Higher order needs more SNR for the same info-bit error rate.
SNR_GRID = {
    16:  [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    64:  [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
    256: [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
}
# training budgets (equal for rl_qam / rl_bpsk / random)
STEPS, BATCH, FRAMES, VAL = 16, 24, 28, 200
RANDOM_BUDGET = 64
FINAL_SEED = 987654

# ----------------------------- QAM evaluator -------------------------------- #
_CACHE, _CHAN = {}, {}


def _worker_qam(task):
    cfg_d, assign, M, ebn0, frame_seed, lo, hi = task
    cfg = R.Config(**cfg_d)
    key = (cfg.bg, cfg.ils, cfg.Z, cfg.mp)
    comp = _CACHE.get(key)
    if comp is None:
        comp = cfg.component(); _CACHE[key] = comp
    sc = SCLDPCCode(comp, w=cfg.w, L=cfg.L, assign=np.asarray(assign, dtype=np.int64))
    chan = _CHAN.get(M)
    if chan is None:
        chan = qam.QAMChannel(M); _CHAN[M] = chan
    sigma = qam.ebn0_to_sigma_qam(ebn0, sc.rate, M)
    be = bits = fe = 0
    for idx in range(lo, hi):
        rng = np.random.default_rng([int(frame_seed), int(idx)])   # CRN: same info+noise across constructions
        info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
        cw, _ = sc.encode(info)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = chan.transmit(cw[sc.tx_mask], sigma, rng)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=cfg.W, max_iter=cfg.max_iter, alpha=cfg.alpha)
        err = int((sc.extract_info(hard) != info).sum())
        be += err; bits += info.size; fe += int(err > 0)
    return be, bits, fe, hi - lo


def eval_qam(cfg, assigns, M, ebn0, frame_seed, n_frames, pool, chunks=1):
    cfg_d = asdict(cfg)
    bounds = np.linspace(0, n_frames, chunks + 1).astype(int)
    tasks, owner = [], []
    for ai, a in enumerate(assigns):
        aa = np.asarray(a, dtype=np.int64)
        for ci in range(chunks):
            lo, hi = int(bounds[ci]), int(bounds[ci + 1])
            if hi > lo:
                tasks.append((cfg_d, aa, M, ebn0, frame_seed, lo, hi)); owner.append(ai)
    raw = pool.map(_worker_qam, tasks)
    agg = [[0, 0, 0, 0] for _ in assigns]
    for o, (be, bits, fe, nf) in zip(owner, raw):
        agg[o][0] += be; agg[o][1] += bits; agg[o][2] += fe; agg[o][3] += nf
    return [{"ber": be / max(bits, 1), "fer": fe / max(nf, 1), "bit_err": int(be),
             "n_info": int(bits), "frame_err": int(fe), "n_frames": int(nf)}
            for be, bits, fe, nf in agg]


# ----------------------------- constructions -------------------------------- #
def _n4_of(cfg, assign):
    return R.count_4cycles(cfg.build(assign=np.asarray(assign, dtype=np.int64)))


def _stats(cfg, assign):
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return {"n4": int(R.count_4cycles(sc)), "girth": int(R.girth(sc, n_seeds=200))}


def cutvec_best_n4(cfg, pool):
    rows, cols, E = R.edge_meta(cfg); mb = int(rows.max()) + 1; nc = int(cols.max()) + 1; w = cfg.w
    grid = sorted(set(int(round(x)) for x in np.linspace(0, mb, min(mb + 1, 9))))
    cands = [np.repeat(np.array(t, dtype=np.int64)[:, None], nc, axis=1)
             for t in combinations_with_replacement(grid, w)]
    rng = np.random.default_rng(0)
    cands += [np.sort(rng.integers(0, mb + 1, size=(w, nc)), axis=0) for _ in range(200)]

    def cv(cuts):
        a = np.zeros(E, dtype=np.int64)
        for k in range(w):
            a += (cuts[k, cols] <= rows).astype(np.int64)
        return a
    seen, uniq = set(), []
    for c in cands:
        a = cv(c); b = a.tobytes()
        if b not in seen:
            seen.add(b); uniq.append(a)
    n4s = pool.starmap(_n4_of, [(cfg, a) for a in uniq])
    return uniq[int(np.argmin(n4s))]


def train_rl_qam(cfg, M, ebn0, pool, seed=0):
    pol = R.FeaturePolicy(cfg, seed=seed)
    best = {"fer": 2.0, "assign": None}
    for step in range(STEPS):
        rolls = [pol.rollout() for _ in range(BATCH)]
        assigns = [a for a, _ in rolls]
        mets = eval_qam(cfg, assigns, M, ebn0, frame_seed=10000 + step, n_frames=FRAMES, pool=pool)
        rew = np.array([-m["fer"] for m in mets])
        adv = rew - rew.mean(); sd = adv.std()
        if sd > 1e-9:
            adv = adv / sd
        pol.update([(tr, float(adv[i])) for i, (a, tr) in enumerate(rolls)])
        gi = int(np.argmin([m["fer"] for m in mets]))
        if mets[gi]["fer"] < best["fer"]:
            best = {"fer": mets[gi]["fer"], "assign": assigns[gi]}
    greedy = pol.greedy_assign()
    cand = [greedy, best["assign"] if best["assign"] is not None else greedy]
    vm = eval_qam(cfg, cand, M, ebn0, frame_seed=99999, n_frames=VAL, pool=pool)
    return np.asarray(cand[int(np.argmin([m["fer"] for m in vm]))], dtype=np.int64), pol.theta.tolist()


# bigger native-QAM training budget (the first pass under-trained: 16x24x28 at a
# poorly-chosen SNR where the QAM FER had no gradient signal).
NATIVE_STEPS, NATIVE_BATCH, NATIVE_FRAMES, NATIVE_VAL = 48, 32, 48, 400


def pick_train_snr(out):
    """Train at the SNR where the round-robin FER is in the discriminating waterfall
    region (closest to 0.30); training where FER~1 or ~0 gives no gradient signal."""
    rr = out["constructions"]["round_robin"]["waterfall"]
    return min(rr, key=lambda p: abs(p["fer"] - 0.30))["snr"]


def train_rl_qam_big(cfg, M, ebn0, pool, seed=0):
    pol = R.FeaturePolicy(cfg, seed=seed)
    best = {"fer": 2.0, "assign": None}
    for step in range(NATIVE_STEPS):
        rolls = [pol.rollout() for _ in range(NATIVE_BATCH)]
        assigns = [a for a, _ in rolls]
        mets = eval_qam(cfg, assigns, M, ebn0, frame_seed=20000 + step, n_frames=NATIVE_FRAMES, pool=pool)
        rew = np.array([-m["fer"] for m in mets])
        adv = rew - rew.mean(); sd = adv.std()
        if sd > 1e-9:
            adv = adv / sd
        pol.update([(tr, float(adv[i])) for i, (a, tr) in enumerate(rolls)])
        gi = int(np.argmin([m["fer"] for m in mets]))
        if mets[gi]["fer"] < best["fer"]:
            best = {"fer": mets[gi]["fer"], "assign": assigns[gi]}
    greedy = pol.greedy_assign()
    cand = [greedy, best["assign"] if best["assign"] is not None else greedy]
    vm = eval_qam(cfg, cand, M, ebn0, frame_seed=88888, n_frames=NATIVE_VAL, pool=pool)
    return np.asarray(cand[int(np.argmin([m["fer"] for m in vm]))], dtype=np.int64), pol.theta.tolist()


def rerun_rlqam(M):
    """Retrain ONLY rl_qam (bigger budget, discriminating training SNR), recompute its
    waterfall, overwrite its entry -- the other five constructions are kept as-is."""
    fname = f"results_qam_construct_M{M}.json"
    out = json.load(open(fname))
    grid, fg = out["snr_grid"], out["frames_grid"]
    nw = R.n_workers()
    print(f"M={M} rl_qam RERUN  workers={nw}", flush=True)
    with Pool(nw) as pool:
        tsnr = pick_train_snr(out)
        print(f"  train @ {tsnr}dB (round-robin FER~0.3), budget "
              f"{NATIVE_STEPS}x{NATIVE_BATCH}x{NATIVE_FRAMES}", flush=True)
        a, theta = train_rl_qam_big(CFG, M, tsnr, pool)
        st = _stats(CFG, a)
        wf = waterfall(CFG, a, M, grid, fg, pool)
        out["constructions"]["rl_qam"] = {"assign": a.tolist(), **st, "waterfall": wf,
                                          "train_snr": tsnr,
                                          "budget": [NATIVE_STEPS, NATIVE_BATCH, NATIVE_FRAMES]}
        out["theta_rl_qam"] = theta
        out.setdefault("_assigns", {})["rl_qam"] = a.tolist()
        json.dump(out, open(fname, "w"))
        best = min(wf, key=lambda p: p["fer"])
        print(f"  rl_qam NEW: n4={st['n4']} girth={st['girth']} "
              f"minFER={best['fer']:.3e}@{best['snr']}dB", flush=True)
    print(f"--- M={M} rl_qam rerun done -> {fname}", flush=True)


def best_random_qam(cfg, M, ebn0, pool):
    rng = np.random.default_rng(123)
    cand = [R.random_assign(cfg, rng) for _ in range(RANDOM_BUDGET)]
    vm = eval_qam(cfg, cand, M, ebn0, frame_seed=55555, n_frames=VAL, pool=pool)
    return np.asarray(cand[int(np.argmin([m["fer"] for m in vm]))], dtype=np.int64)


def constellation_data(M, sigma, n_sym=4000, seed=2024):
    """Ideal M-QAM points (+ Gray labels) and a sample of received noisy symbols."""
    rng = np.random.default_rng(seed)
    T = qam._qam_tables(M)
    a = T["amps_scaled"]; bt = T["bit_table"]; k = T["k"]
    ideal_I, ideal_Q, labels = [], [], []
    for ii in range(len(a)):
        for qi in range(len(a)):
            ideal_I.append(float(a[ii])); ideal_Q.append(float(a[qi]))
            labels.append("".join(map(str, bt[ii].tolist())) + "".join(map(str, bt[qi].tolist())))
    bits = rng.integers(0, 2, size=n_sym * T["m"])
    syms, _ = qam.bits_to_symbols(bits, M)
    y = qam.awgn_complex(syms, sigma, rng)
    return {"M": M, "sigma": sigma, "ideal_I": ideal_I, "ideal_Q": ideal_Q, "labels": labels,
            "rx_I": np.round(np.real(y), 4).tolist(), "rx_Q": np.round(np.imag(y), 4).tolist()}


# ------------------------------ driver -------------------------------------- #
def _frames_grid(grid):
    return np.geomspace(600, 8000, len(grid)).astype(int).tolist()


def waterfall(cfg, assign, M, grid, frames_grid, pool):
    pts = []
    for snr, nf in zip(grid, frames_grid):
        m = eval_qam(cfg, [assign], M, snr, FINAL_SEED, int(nf), pool=pool, chunks=R.n_workers())[0]
        pts.append({"snr": snr, **m})
    return pts


def run(M):
    fname = f"results_qam_construct_M{M}.json"
    grid = SNR_GRID[M]; fg = _frames_grid(grid)
    nw = R.n_workers()
    print(f"M={M}  workers={nw}  SNR grid={grid}", flush=True)
    with Pool(nw) as pool:
        if os.path.exists(fname):
            out = json.load(open(fname))
        else:
            out = {"M": M, "cfg": asdict(CFG), "rate": CFG.build().rate,
                   "snr_grid": grid, "frames_grid": fg,
                   "bits_per_symbol": qam.bits_per_symbol(M),
                   "spectral_eff": qam.bits_per_symbol(M) * CFG.build().rate,
                   "constructions": {}, "constellation": None}

        # ---- build the six constructions (cached in JSON once computed) ----
        if "round_robin" not in out["constructions"]:
            assigns = {}
            t0 = time.time()
            assigns["round_robin"] = np.asarray(R.round_robin_assign(CFG), dtype=np.int64)
            assigns["seed0"] = np.asarray(R.random_assign(CFG, np.random.default_rng(0)), dtype=np.int64)
            print("  cutvec search ...", flush=True)
            assigns["cutvec"] = cutvec_best_n4(CFG, pool)
            print("  rl_bpsk training ...", flush=True)
            _, bpsk_best = R.train_reinforce(CFG, R.FeaturePolicy(CFG, seed=0), BPSK_TRAIN_SNR,
                                             STEPS, BATCH, FRAMES, pool=pool, val_frames=VAL,
                                             base_seed=2000, verbose=False)
            assigns["rl_bpsk"] = np.asarray(bpsk_best["assign"], dtype=np.int64)
            print("  random search on QAM ...", flush=True)
            assigns["random"] = best_random_qam(CFG, M, grid[1], pool)
            print("  rl_qam native training ...", flush=True)
            assigns["rl_qam"], theta = train_rl_qam(CFG, M, grid[1], pool)
            out["theta_rl_qam"] = theta
            out["_assigns"] = {k: v.tolist() for k, v in assigns.items()}
            print(f"  constructions built ({time.time()-t0:.0f}s)", flush=True)
            json.dump(out, open(fname, "w"))

        assigns = {k: np.asarray(v, dtype=np.int64) for k, v in out["_assigns"].items()}
        order = ["rl_qam", "rl_bpsk", "cutvec", "random", "round_robin", "seed0"]
        for name in order:
            if name in out["constructions"]:
                continue
            t0 = time.time()
            wf = waterfall(CFG, assigns[name], M, grid, fg, pool)
            st = _stats(CFG, assigns[name])
            out["constructions"][name] = {"assign": assigns[name].tolist(), **st, "waterfall": wf}
            json.dump(out, open(fname, "w"))
            best = min(wf, key=lambda p: p["fer"])
            print(f"  {name:12s} n4={st['n4']} girth={st['girth']} "
                  f"minFER={best['fer']:.3e}@{best['snr']}dB ({time.time()-t0:.0f}s)", flush=True)

        # ---- constellation sample at the mid-SNR (for drawing) ----
        if out.get("constellation") is None:
            mid = grid[len(grid) // 2]
            sigma = qam.ebn0_to_sigma_qam(mid, out["rate"], M)
            out["constellation"] = {"snr_db": mid, **constellation_data(M, sigma)}
            json.dump(out, open(fname, "w"))
            print(f"  constellation saved @ {mid}dB (sigma={sigma:.4f})", flush=True)
    print(f"--- M={M} done -> {fname}", flush=True)


def agg():
    print("\n=== QAM construction comparison (min FER over the waterfall) ===")
    for M in (16, 64, 256):
        f = f"results_qam_construct_M{M}.json"
        if not os.path.exists(f):
            print(f"M={M}: not run"); continue
        d = json.load(open(f))
        print(f"\n[{M}-QAM] rate={d['rate']:.3f}  spectral eff={d['spectral_eff']:.2f} bit/s/Hz")
        for name, c in d["constructions"].items():
            best = min(c["waterfall"], key=lambda p: p["fer"])
            print(f"  {name:12s} n4={c['n4']:>4} girth={c['girth']} "
                  f"minFER={best['fer']:.3e}@{best['snr']}dB")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "agg"
    if cmd == "agg":
        agg()
    elif cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "rlqam":
        rerun_rlqam(int(sys.argv[2]))
    else:
        raise SystemExit("usage: experiments_qam_construct.py [run <M>|rlqam <M>|agg]")
