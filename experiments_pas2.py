"""PAS2 -- publishable-strength RL x QAM Probabilistic Amplitude Shaping (Part B v2).

This is the rigorous redo of `experiments_pas_rl.py`'s PART 2.  The proof-of-concept
compared uniform / shaped / RL-joint at matched info-bit Eb/N0 but UNMATCHED net
spectral efficiency (shaped/RL spent rate on shaping -> lower net SE), so "uniform
fails at BER 0.1" was partly a rate artifact.  PAS2 fixes the four reviewer-blocking
weaknesses:

  1. MATCHED SPECTRAL EFFICIENCY.  Per cell we fix a target net SE [bits/complex
     symbol].  At fixed M:
       * UNIFORM uses a lower-rate (stronger) SC-LDPC code at nu=0 hitting that SE
         (SE = sc.rate * m).
       * SHAPED/RL use a HIGHER-rate (weaker) code + Maxwell-Boltzmann shaping nu
         chosen so the OPERATIONAL net rate (sc.rate * m * h_frac, h_frac = per-axis
         level-entropy fraction) equals the uniform net SE EXACTLY.  Shaping frees
         code rate; the question is whether the shaping gain beats the weaker code at
         equal SE.  We report "required Eb/N0 @ BER target at MATCHED SE" -- the
         proper publishable gain in dB.

  2. STRONG baselines + ABLATIONS at matched SE, per cell:
       (a) uniform  + round-robin            (canonical strong non-learned)
       (a') uniform + CEM / random-search    (equal-budget optimised uniform)
       (b) shaped(nu_matched) + round-robin / + CEM-optimised construction
       (c) construction-RL at fixed nu_matched
       (d) RL-JOINT (co-optimise nu-split + construction)
     Two ablations: (i) JOINT vs SEPARATE (best-split THEN best-construction);
     (ii) RL vs CEM vs equal-budget RANDOM for the construction part.  Reported
     head-to-head, honestly.

  3. LONGER codes + DEEPER waterfalls.  Train the construction search on a short L
     (edge spreading is ~L-independent) but VALIDATE/REPORT champions on L in {50,200}
     with a bits-budget pushing BER ~1e-5..1e-6 so curves separate.  Z in {32,64}.
     Larger validation banks expose any short-L validation overfit.

  4. MORE cells.  {16,64,256}-QAM x a few SE targets x a couple of w.

DELIVERABLES: results_pas2_*.json/csv, exp_pas2_*.svg, dashboard/pas2_status.py +
pas2_poll.py, and results_pas2_assessment.md.

    EXP_WORKERS=120 python3 experiments_pas2.py run     # resumable JSON checkpoints
    python3 experiments_pas2.py plot                    # SVG figures + CSV tables
    python3 experiments_pas2.py smoke                   # tiny local smoke (1 cell, fast)

numpy-only; reuses pas/qam/rl_construct verbatim (R._worker = _worker_pas, swapped
in from experiments_pas_rl so every R.eval_* routes frames through shaped-QAM).
"""
from __future__ import annotations
import json
import os
import sys
import time
import multiprocessing as mp
from dataclasses import asdict

import numpy as np

import plotting
import rl_construct as R
import qam
import pas
from sc_ldpc import SCLDPCCode

# import the validated PAS chain (worker, shaped info gen, net-rate, policies).
# This also executes `R._worker = _worker_pas`, routing R.eval_* through shaped QAM.
import experiments_pas_rl as P
from experiments_pas_rl import (
    shaped_info_bits, axis_entropy_frac, net_rate_operational,
    eval_codes, eval_batch_codes, NuPolicy, _worker_pas,
)

# ----------------------------------------------------------------------------- #
#  regime
# ----------------------------------------------------------------------------- #
BG, ILS = 1, 0
MAXIT, ALPHA = 12, 0.8
W_DEC_MIN = 6
OPT_L = 12                  # chain length while OPTIMISING construction (L-independent edge spreading)
REPORT_LS = [50, 200]      # chain lengths for the REPORTED waterfalls
BER_FLOOR = 2e-6

# Maxwell-Boltzmann nu lookup grid for matched-SE solving
NU_GRID = np.concatenate([[0.0], np.linspace(0.002, 0.45, 600)])

# ---- matched-SE cells -------------------------------------------------------- #
# Each cell: (M, Z, w, mp_uniform, mp_shaped).  mp_uniform is the LOWER-rate (more
# parity) STRONGER code at nu=0; mp_shaped is a HIGHER-rate (fewer parity) weaker code
# whose net rate is pulled back to the uniform net SE by MB shaping nu.  The shaped nu
# is SOLVED per (L) so SE matches at the reported length.  m=log2(M): SE=sc.rate*m.
# Span: M in {16,64,256}, two SE operating regions, w in {3,4}, Z in {32,64}.
CELLS = [
    # (M,  Z,  w,  mp_uniform, mp_shaped)        approx target SE @L=50
    (16,  32, 3,  9,  6),     # 16-QAM  SE~2.9   (uni R~0.74, shaped R~0.83)
    (16,  32, 4,  9,  6),     # 16-QAM  w=4
    (64,  32, 3,  13, 9),     # 64-QAM  SE~3.9   (uni R~0.65, shaped R~0.74)
    (64,  32, 4,  13, 9),     # 64-QAM  w=4
    (64,  64, 3,  9,  6),     # 64-QAM  SE~4.8   higher SE, Z=64 long code
    (256, 32, 3,  16, 11),    # 256-QAM SE~4.9   (uni R~0.61, shaped R~0.71)
    (256, 64, 4,  16, 11),    # 256-QAM SE~4.8   Z=64 long, w=4
    (256, 32, 3,  13, 9),     # 256-QAM SE~6.0   high SE (biggest shaping gain region)
]

# ---- search budget (equal across RL / CEM / random) -------------------------- #
OPT_STEPS, OPT_BATCH, OPT_FRAMES = 14, 96, 6      # 96 fills the 120-core pod
VAL_FRAMES = 40                                   # champion refine bank during search
SEP_NU_GRID_FRAMES = 60                           # frames for the separate-opt nu sweep

# ---- operating-point / cliff location -------------------------------------- #
PROBE_N, PROBE_FRAMES = 24, 12
TRAIN_CLIFF_FRAMES = 160      # frames to locate the train-L shaped cliff (round-robin)
REPORT_CLIFF_FRAMES = 250     # frames to locate both cliffs at each report L

# ---- reported waterfall (bits-budget like experiments_qam_sweep) ------------- #
# 8-point SNR grid spanning both matched-SE cliffs (built per-L in run_cell).  Deeper
# (higher-SNR) points get a larger bits-budget so the tails reach ~1e-5..1e-6 cleanly.
N_WATER_PTS = 8
TARGET_BITS = [3.0e3, 6.0e3, 2.0e4, 8.0e4, 3.0e5, 1.0e6, 2.5e6, 5.0e6]
MIN_FRAMES, MAX_FRAMES = 8, 4000
WATER_SEED = 2025
BER_TARGET_REPORT = 1e-5                            # the headline "required Eb/N0 @ 1e-5"

RESULT_FN = "results_pas2.json"


def frames_for(target_bits, K):
    import math
    return int(np.clip(math.ceil(target_bits / K), MIN_FRAMES, MAX_FRAMES))


def W_for(w):
    return max(W_DEC_MIN, w + 2)


def cfg_for(M, mp, Z, w, L):
    return R.Config(bg=BG, ils=ILS, Z=Z, mp=mp, w=w, L=L, W=W_for(w),
                    max_iter=MAXIT, alpha=ALPHA)


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


# ----------------------------------------------------------------------------- #
#  matched-SE solver
# ----------------------------------------------------------------------------- #
def solve_matched_nu(M, r_uniform, r_shaped):
    """nu so that the shaped HIGHER-rate code's net rate matches the uniform net SE:
       r_uniform * m * 1 == r_shaped * m * h_frac(nu)  =>  h_frac = r_uniform/r_shaped.
    Returns (nu, h_frac_achieved, abs_error)."""
    target_hf = r_uniform / r_shaped
    if target_hf >= 1.0:                 # shaped code is not higher-rate -> no shaping room
        return 0.0, 1.0, abs(1.0 - target_hf)
    best = (0.0, 1.0, 9.9)
    for nu in NU_GRID:
        hf = axis_entropy_frac(M, pas.mb_pmf(M, float(nu)))
        e = abs(hf - target_hf)
        if e < best[2]:
            best = (round(float(nu), 5), hf, e)
    return best


def matched_se_config(M, Z, w, mp_uniform, mp_shaped, L):
    """At chain length L, build the matched-SE (uniform, shaped) operating points.
    Returns dict with both Configs, the rates, the matched nu, and the (matched) SE."""
    m = qam.bits_per_symbol(M)
    cfg_u = cfg_for(M, mp_uniform, Z, w, L)
    cfg_s = cfg_for(M, mp_shaped, Z, w, L)
    r_u = cfg_u.build().rate
    r_s = cfg_s.build().rate
    nu, hf, err = solve_matched_nu(M, r_u, r_s)
    se_u = r_u * m
    se_s = r_s * m * hf
    return {"cfg_u": cfg_u, "cfg_s": cfg_s, "r_u": r_u, "r_s": r_s,
            "nu_matched": nu, "h_frac": hf, "se_u": se_u, "se_s": se_s,
            "se_err": abs(se_u - se_s), "m": m, "mp_u": mp_uniform, "mp_s": mp_shaped}


# ----------------------------------------------------------------------------- #
#  construction search at FIXED nu (RL / CEM / random), shaped-QAM reward
# ----------------------------------------------------------------------------- #
def _cliff_snr(cfg, M, nu0, pool, target_ber=1e-2, frames=200, lo=2.0, hi=16.0,
               iters=8):
    """Bisect info Eb/N0 to where the ROUND-ROBIN construction sits at BER~target_ber
    (the cliff midpoint), with shaped(nu0) input.  Round-robin (deterministic) is more
    stable than median-of-random for locating a cliff.  Returns (snr, ber_there)."""
    rr = R.round_robin_assign(cfg)
    pmf = pas.mb_pmf(M, nu0)

    def ber(snr):
        m = eval_codes(cfg, rr, M, nu0, snr, 4242, frames, pool=pool,
                       frame_chunks=(R.n_workers() if pool else 1), amp_pmf=pmf)
        return float(m["ber"])

    for _ in range(iters):
        mid = round((lo + hi) / 2.0, 2)
        if ber(mid) < target_ber:
            hi = mid
        else:
            lo = mid
    snr = round((lo + hi) / 2.0, 2)
    return snr, ber(snr)


def search_construction(method, cfg, M, nu, snr, pool, seed=0):
    """Optimise the edge-spreading construction at FIXED nu through the shaped-QAM
    channel.  method in {'rl','cem','random'}.  Equal eval budget (OPT_STEPS*OPT_BATCH).
    Returns (history, best{ber,assign})."""
    _, _, E = R.edge_meta(cfg)
    C = R.n_components(cfg)
    hist = {"evals": [], "best_val_ber": []}
    best = {"ber": np.inf, "fer": np.inf, "assign": None}
    seen = 0

    if method == "rl":
        con = R.FeaturePolicy(cfg, lr=0.15, ent=0.02, seed=seed)
    elif method == "cem":
        cem_rng = np.random.default_rng(seed)
        p = np.full((E, C), 1.0 / C)
        n_elite = max(2, int(OPT_BATCH * 0.3))
    elif method == "random":
        rnd_rng = np.random.default_rng(seed)
    else:
        raise ValueError(method)

    for step in range(1, OPT_STEPS + 1):
        fs = 7000 + step * 7 + (hash(method) & 255)
        traces = None
        if method == "rl":
            rolls = [con.rollout() for _ in range(OPT_BATCH)]
            assigns = [r[0] for r in rolls]
            traces = [r[1] for r in rolls]
        elif method == "cem":
            assigns = [np.array([cem_rng.choice(C, p=p[e]) for e in range(E)])
                       for _ in range(OPT_BATCH)]
        else:  # random
            assigns = [rnd_rng.integers(0, C, size=E) for _ in range(OPT_BATCH)]

        items = [(a, M, nu) for a in assigns]
        ms = eval_batch_codes(cfg, items, snr, fs, OPT_FRAMES, pool=pool)
        seen += OPT_BATCH
        bers = np.array([m["ber"] for m in ms])

        if method == "rl":
            rew = np.array([-np.log10(max(b, BER_FLOOR)) for b in bers])
            adv = rew - rew.mean()
            if adv.std() > 1e-9:
                adv = adv / (adv.std() + 1e-9)
            con.update(list(zip(traces, adv)))
        elif method == "cem":
            elite = np.argsort(bers)[:n_elite]
            freq = np.zeros((E, C))
            for idx in elite:
                freq[np.arange(E), assigns[idx]] += 1.0
            freq /= n_elite
            p = 0.7 * p + 0.3 * freq
            p = np.clip(p, 1e-3, None)
            p /= p.sum(axis=1, keepdims=True)

        cand = assigns[int(np.argmin(bers))]
        vm = eval_codes(cfg, cand, M, nu, snr, 99, VAL_FRAMES, pool=pool,
                        frame_chunks=(R.n_workers() if pool else 1))
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"], "assign": np.asarray(cand).copy()}
        hist["evals"].append(seen)
        hist["best_val_ber"].append(best["ber"])
    if best["assign"] is None:
        best["assign"] = R.round_robin_assign(cfg)
    return hist, best


# ----------------------------------------------------------------------------- #
#  RL-JOINT: co-optimise the (rate-split) shaping nu AND the construction
# ----------------------------------------------------------------------------- #
def search_joint(cfg_s, M, nu_matched, snr, pool, seed=0, nu_span=0.5):
    """Co-optimise construction (FeaturePolicy on the SHAPED higher-rate code) AND a
    small shaping perturbation around nu_matched, REINFORCE with shared advantage.

    IMPORTANT (matched SE): the reward credits coded BER, but the NET SE depends on nu.
    To keep the comparison honest we let nu wander around nu_matched, then for the
    REPORTED champion we RE-PIN to the SE-matched point: the construction is the learned
    one, nu is reset to nu_matched (so the reported joint system sits at the SAME SE as
    uniform).  This isolates 'did joint co-design find a better construction than fixing
    nu first' while keeping SE matched in the final report.  The free-nu trajectory is
    logged so we can see whether the policy *wanted* to move off nu_matched."""
    con = R.FeaturePolicy(cfg_s, lr=0.15, ent=0.02, seed=seed)
    nup = NuPolicy(init_nu=max(nu_matched, 0.02), lr=0.05, std=nu_span, seed=seed,
                   nu_cap=max(0.45, nu_matched * 2))
    hist = {"evals": [], "best_val_ber": [], "nu_mean": [], "mean_reward": []}
    best = {"ber": np.inf, "fer": np.inf, "assign": None, "nu_free": None}
    seen = 0
    for step in range(1, OPT_STEPS + 1):
        fs = 8000 + step
        rolls = [con.rollout() for _ in range(OPT_BATCH)]
        assigns = [r[0] for r in rolls]
        traces = [r[1] for r in rolls]
        nus, us = [], []
        for _ in range(OPT_BATCH):
            nu, u = nup.sample()
            nus.append(nu)
            us.append(u)
        items = [(assigns[i], M, nus[i]) for i in range(OPT_BATCH)]
        ms = eval_batch_codes(cfg_s, items, snr, fs, OPT_FRAMES, pool=pool)
        seen += OPT_BATCH
        rew = np.array([-np.log10(max(m["ber"], BER_FLOOR)) for m in ms])
        adv = rew - rew.mean()
        if adv.std() > 1e-9:
            adv = adv / (adv.std() + 1e-9)
        con.update(list(zip(traces, adv)))
        nup.update([(us[i], adv[i]) for i in range(OPT_BATCH)])
        # champion at the SE-MATCHED nu (re-pinned), refined on the validation bank
        bi = int(np.argmin([m["ber"] for m in ms]))
        vm = eval_codes(cfg_s, assigns[bi], M, nu_matched, snr, 99, VAL_FRAMES,
                        pool=pool, frame_chunks=(R.n_workers() if pool else 1))
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"],
                    "assign": np.asarray(assigns[bi]).copy(), "nu_free": nup.nu()}
        hist["evals"].append(seen)
        hist["best_val_ber"].append(best["ber"])
        hist["nu_mean"].append(nup.nu())
        hist["mean_reward"].append(float(rew.mean()))
    if best["assign"] is None:
        best["assign"] = R.round_robin_assign(cfg_s)
    return hist, best


# ----------------------------------------------------------------------------- #
#  reported waterfall on the long codes (bits-budget)
# ----------------------------------------------------------------------------- #
def waterfall(cfg, assign, M, nu, snrs, pool):
    """BER/FER waterfall through shaped-QAM with a per-point bits-budget (deep tails)."""
    sc0 = cfg.build()
    K = sc0.K
    nfr = [frames_for(tb, K) for tb in TARGET_BITS]
    pmf = pas.mb_pmf(M, nu)
    xs, bers, fers = [], [], []
    chunks = R.n_workers()
    for snr, nf in zip(snrs, nfr):
        m = eval_codes(cfg, assign, M, nu, snr, WATER_SEED, nf, pool=pool,
                       frame_chunks=chunks, amp_pmf=pmf)
        xs.append(round(snr, 2))
        bers.append(m["ber"])
        fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers, "frames": nfr}


def _ebn0_at_ber(curve, target):
    import math
    xs, ys = curve["x"], [max(y, 1e-9) for y in curve["y"]]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 >= target >= y1 and y1 != y0:
            t = (math.log10(target) - math.log10(y0)) / (math.log10(y1) - math.log10(y0))
            return round(xs[i] + t * (xs[i + 1] - xs[i]), 3)
    return None


# ----------------------------------------------------------------------------- #
#  one matched-SE cell
# ----------------------------------------------------------------------------- #
def run_cell(M, Z, w, mp_u, mp_s, pool):
    t0 = time.time()
    # --- matched-SE design at the TRAIN length and at each REPORT length ---
    train = matched_se_config(M, Z, w, mp_u, mp_s, OPT_L)
    nu_m = train["nu_matched"]
    cfg_u, cfg_s = train["cfg_u"], train["cfg_s"]
    _, _, E = R.edge_meta(cfg_s)
    print(f"\n=== CELL M={M} Z={Z} w={w} | UNIFORM mp{mp_u} R={train['r_u']:.3f} "
          f"SE={train['se_u']:.3f}  ||  SHAPED mp{mp_s} R={train['r_s']:.3f} "
          f"nu={nu_m:.3f} h_frac={train['h_frac']:.3f} SE={train['se_s']:.3f} "
          f"(SE err {train['se_err']:.4f}) [train L={OPT_L}] ===", flush=True)

    # operating SNR for TRAINING the construction: the shaped code's BER~1e-2 cliff
    # midpoint at the TRAIN length (steep region -> informative construction gradient).
    snr, ber_there = _cliff_snr(cfg_s, M, nu_m, pool, target_ber=1e-2, frames=TRAIN_CLIFF_FRAMES)
    print(f"    train@{snr}dB (shaped round-robin BER {ber_there:.2e} at train L={OPT_L}), "
          f"E={E}", flush=True)

    # ---------- construction searches at FIXED matched nu (shaped code) ----------
    h_rl,  b_rl  = search_construction("rl",     cfg_s, M, nu_m, snr, pool, seed=0)
    h_cem, b_cem = search_construction("cem",    cfg_s, M, nu_m, snr, pool, seed=7)
    h_rnd, b_rnd = search_construction("random", cfg_s, M, nu_m, snr, pool, seed=11)
    print(f"    [shaped-constr search @nu*] RL val_BER={b_rl['ber']:.2e}  "
          f"CEM={b_cem['ber']:.2e}  RND={b_rnd['ber']:.2e}", flush=True)

    # ---------- uniform (nu=0) construction searches: RR + CEM (strong baseline) ----------
    h_u_cem, b_u_cem = search_construction("cem", cfg_u, M, 0.0, snr, pool, seed=7)
    rr_u = R.round_robin_assign(cfg_u)
    print(f"    [uniform-constr search]     CEM val_BER={b_u_cem['ber']:.2e}", flush=True)

    # ---------- RL-JOINT (co-optimise nu-split + construction) ----------
    h_joint, b_joint = search_joint(cfg_s, M, nu_m, snr, pool, seed=0)
    print(f"    [RL-joint] val_BER={b_joint['ber']:.2e} "
          f"(free-nu drifted to {b_joint['nu_free']:.3f}, re-pinned to {nu_m:.3f})", flush=True)

    # ---------- ABLATION (i): SEPARATE = best-nu-split THEN best-construction ----------
    # Sweep a few (mp_shaped', nu') operating points that ALL hit the uniform SE, pick the
    # split with best random-constr BER, then RL-optimise the construction for that split.
    sep = _separate_optimum(M, Z, w, mp_u, snr, pool, train["se_u"])
    print(f"    [separate-opt] best split mp{sep['mp']} nu={sep['nu']:.3f} "
          f"-> RL val_BER={sep['best']['ber']:.2e}", flush=True)

    rr_s = R.round_robin_assign(cfg_s)
    # champions to report on the long codes (all at MATCHED SE):
    champions = {
        "uniform_rr":     (cfg_u, rr_u,            0.0),    # strong non-learned uniform
        "uniform_cem":    (cfg_u, b_u_cem["assign"], 0.0),  # optimised uniform (STRONG baseline)
        "shaped_rr":      (cfg_s, rr_s,            nu_m),   # shaping only, canonical constr
        "shaped_cem":     (cfg_s, b_cem["assign"], nu_m),   # shaping + CEM construction
        "constr_rl":      (cfg_s, b_rl["assign"],  nu_m),   # construction-RL @ fixed nu*
        "rl_joint":       (cfg_s, b_joint["assign"], nu_m), # RL-joint (re-pinned to matched SE)
        "separate_rl":    (sep["cfg"], sep["best"]["assign"], sep["nu"]),  # separate-opt champion
    }

    # ---------- report on long codes L in REPORT_LS (deep waterfalls) ----------
    report = {}
    for L in REPORT_LS:
        # the matched nu is L-dependent (rate loss ~w/L); re-solve per L for each split
        m_at_L = matched_se_config(M, Z, w, mp_u, mp_s, L)
        sep_at_L = matched_se_config(M, Z, w, mp_u, sep["mp"], L)
        # waterfall SNR window must SPAN BOTH cliffs at this L.  At matched SE the shaped
        # (higher-rate) code cliffs EARLIER and uniform (lower-rate, but no shaping gain)
        # cliffs LATER; locate each with round-robin, then span [shaped-1, uniform+1.5].
        cfg_uL = cfg_for(M, mp_u, Z, w, L)
        cfg_sL = cfg_for(M, mp_s, Z, w, L)
        snr_uL, _ = _cliff_snr(cfg_uL, M, 0.0, pool, target_ber=1e-2, frames=REPORT_CLIFF_FRAMES)
        snr_sL, _ = _cliff_snr(cfg_sL, M, m_at_L["nu_matched"], pool, target_ber=1e-2,
                               frames=REPORT_CLIFF_FRAMES)
        lo_w = min(snr_uL, snr_sL) - 1.5
        hi_w = max(snr_uL, snr_sL) + 2.0
        snrs = [round(lo_w + (hi_w - lo_w) * i / 7.0, 2) for i in range(8)]
        print(f"      L={L} cliffs: shaped~{snr_sL}dB uniform~{snr_uL}dB "
              f"-> window [{snrs[0]},{snrs[-1]}]dB", flush=True)
        per_method = {}
        for name, (cfg_proto, assign, nu_train) in champions.items():
            # rebuild the champion's config at this report L; re-pin nu to the L-matched value
            if name.startswith("uniform"):
                cfgL = cfg_for(M, mp_u, Z, w, L); nuL = 0.0; seL = m_at_L["se_u"]
            elif name == "separate_rl":
                cfgL = cfg_for(M, sep["mp"], Z, w, L); nuL = sep_at_L["nu_matched"]; seL = sep_at_L["se_s"]
            else:
                cfgL = cfg_for(M, mp_s, Z, w, L); nuL = m_at_L["nu_matched"]; seL = m_at_L["se_s"]
            cur = waterfall(cfgL, assign, M, nuL, snrs, pool)
            cur["se"] = round(seL, 4)
            cur["nu"] = round(nuL, 4)
            cur["sc_rate"] = round(cfgL.build().rate, 4)
            cur["ebn0_1e5"] = _ebn0_at_ber(cur, BER_TARGET_REPORT)
            st = R.construction_stats(cfgL, assign)
            try:
                g = R.girth(cfgL.build(assign=assign), max_g=10, n_seeds=60)
            except Exception:
                g = None
            cur["n4"] = st["n4"]
            cur["girth"] = g
            cur["comp_load"] = st["comp_load"]
            per_method[name] = cur
            print(f"      L={L} {name:13s} SE={seL:.3f} scR={cur['sc_rate']:.3f} "
                  f"nu={nuL:.3f} BER@end={cur['y'][-1]:.2e} "
                  f"Eb/N0@1e-5={cur['ebn0_1e5']} n4={st['n4']} g={g}", flush=True)
        report[str(L)] = {"snrs": snrs, "methods": per_method}

    print(f"    [cell done in {time.time()-t0:.0f}s]", flush=True)
    return {
        "M": M, "Z": Z, "w": w, "mp_u": mp_u, "mp_s": mp_s, "E": int(E),
        "train_snr": snr, "train": {k: train[k] for k in
            ("r_u", "r_s", "nu_matched", "h_frac", "se_u", "se_s", "se_err", "m")},
        "search_val_ber": {"rl": b_rl["ber"], "cem": b_cem["ber"], "random": b_rnd["ber"],
                           "uniform_cem": b_u_cem["ber"], "joint": b_joint["ber"],
                           "separate": sep["best"]["ber"], "joint_free_nu": b_joint["nu_free"]},
        "separate_opt": {"mp": sep["mp"], "nu": sep["nu"], "scan": sep["scan"]},
        "hist": {"rl": h_rl, "cem": h_cem, "random": h_rnd, "joint": h_joint,
                 "uniform_cem": h_u_cem},
        "report": report,
        "champions": {k: {"assign": _arr(v[1]), "nu": float(v[2]),
                          "mp": (mp_u if k.startswith("uniform") else
                                 (sep["mp"] if k == "separate_rl" else mp_s))}
                      for k, v in champions.items()},
    }


def _separate_optimum(M, Z, w, mp_u, snr, pool, se_target):
    """Ablation (i) helper.  SEPARATELY: first pick the best (rate-split) nu among the
    discrete shaped-mp options that hit the target SE (scored with RANDOM construction),
    THEN RL-optimise the construction for that chosen split.  Contrast with JOINT, which
    co-optimises both.  Returns the chosen split + its RL champion."""
    m = qam.bits_per_symbol(M)
    # candidate higher-rate shaped codes (a span of mp around the cell's mp_s)
    cand_mps = [mp for mp in (mp_u - 2, mp_u - 3, mp_u - 4, mp_u - 5, mp_u - 6)
                if 4 <= mp < mp_u]
    rng = np.random.default_rng(3)
    scan = []
    best_split = None
    for mp in cand_mps:
        cfg = cfg_for(M, mp, Z, w, OPT_L)
        r_s = cfg.build().rate
        r_u = se_target / m
        nu, hf, err = solve_matched_nu(M, r_u, r_s)
        if nu <= 0:                       # not a higher-rate code -> no shaping room
            continue
        _, _, E = R.edge_meta(cfg)
        C = R.n_components(cfg)
        assigns = [rng.integers(0, C, size=E) for _ in range(PROBE_N)]
        items = [(a, M, nu) for a in assigns]
        ms = eval_batch_codes(cfg, items, snr, 5, PROBE_FRAMES, pool=pool)
        med = float(np.median([mm["ber"] for mm in ms]))
        scan.append({"mp": mp, "nu": round(nu, 4), "r_s": round(r_s, 4),
                     "h_frac": round(hf, 4), "med_ber": med})
        if best_split is None or med < best_split["med_ber"]:
            best_split = scan[-1]
    if best_split is None:               # degenerate: fall back to the cell's shaped split
        best_split = {"mp": None, "nu": 0.0}
        cfg = cfg_for(M, mp_u, Z, w, OPT_L)
        h, b = search_construction("rl", cfg, M, 0.0, snr, pool, seed=0)
        return {"mp": mp_u, "nu": 0.0, "cfg": cfg, "best": b, "scan": scan}
    cfg = cfg_for(M, best_split["mp"], Z, w, OPT_L)
    h, b = search_construction("rl", cfg, M, best_split["nu"], snr, pool, seed=0)
    return {"mp": best_split["mp"], "nu": best_split["nu"], "cfg": cfg,
            "best": b, "scan": scan}


# ----------------------------------------------------------------------------- #
#  driver
# ----------------------------------------------------------------------------- #
def run():
    t0 = time.time()
    nw = R.n_workers()
    print(f"PAS2 matched-SE run: {nw} workers, {len(CELLS)} cells, BG{BG} "
          f"train L={OPT_L} report L={REPORT_LS}, numpy-only", flush=True)
    done = {}
    if os.path.exists(RESULT_FN):
        try:
            done = json.load(open(RESULT_FN)).get("cells", {})
            print(f"  (resuming: {len(done)} cells already done)", flush=True)
        except Exception:
            done = {}
    out = {"config": {"cells": CELLS, "opt_L": OPT_L, "report_Ls": REPORT_LS,
                      "opt_steps": OPT_STEPS, "opt_batch": OPT_BATCH,
                      "opt_frames": OPT_FRAMES, "n_water_pts": N_WATER_PTS,
                      "ber_target_report": BER_TARGET_REPORT}, "cells": dict(done)}
    with mp.Pool(nw) as pool:
        for (M, Z, w, mp_u, mp_s) in CELLS:
            key = f"M{M}_Z{Z}_w{w}_mpu{mp_u}_mps{mp_s}"
            if key in done:
                print(f"  skip (done) {key}", flush=True)
                continue
            try:
                out["cells"][key] = run_cell(M, Z, w, mp_u, mp_s, pool)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  !! cell {key} failed: {e}", flush=True)
            json.dump(out, open(RESULT_FN, "w"))
    json.dump(out, open(RESULT_FN, "w"))
    print(f"\nALL DONE in {time.time()-t0:.0f}s -> {RESULT_FN}", flush=True)


# ----------------------------------------------------------------------------- #
#  plotting + tables
# ----------------------------------------------------------------------------- #
LAB = {"uniform_rr": "uniform + round-robin", "uniform_cem": "uniform + CEM (strong)",
       "shaped_rr": "shaped(nu*) + round-robin", "shaped_cem": "shaped + CEM constr",
       "constr_rl": "shaped + RL constr", "rl_joint": "RL-joint [ours]",
       "separate_rl": "separate-opt (split->RL)"}
COL = {"uniform_rr": "#7f7f7f", "uniform_cem": "#000000", "shaped_rr": "#2ca02c",
       "shaped_cem": "#17becf", "constr_rl": "#ff7f0e", "rl_joint": "#d62728",
       "separate_rl": "#9467bd"}
PLOT_ORDER = ["uniform_rr", "uniform_cem", "shaped_rr", "shaped_cem", "constr_rl",
              "separate_rl", "rl_joint"]
MCOL = {16: "#1f77b4", 64: "#2ca02c", 256: "#d62728"}


def make_plots():
    if not os.path.exists(RESULT_FN):
        print(f"no {RESULT_FN}")
        return
    data = json.load(open(RESULT_FN))
    cells = data["cells"]
    made = []
    for key, c in cells.items():
        for L in [str(x) for x in REPORT_LS]:
            if L not in c.get("report", {}):
                continue
            methods = c["report"][L]["methods"]
            ser = [{"x": methods[n]["x"], "y": methods[n]["y"], "label": LAB[n],
                    "color": COL[n]} for n in PLOT_ORDER if n in methods]
            se = methods.get("uniform_rr", {}).get("se", "?")
            plotting.semilogy(ser, xlabel="info Eb/N0 [dB]", ylabel="info BER",
                              title=f"PAS2 matched-SE={se} BER: {c['M']}-QAM w={c['w']} "
                                    f"Z={c['Z']} L={L}",
                              path=f"exp_pas2_ber_{key}_L{L}.svg")
            made.append(f"exp_pas2_ber_{key}_L{L}.svg")
        # learning curves (construction searches, fixed nu* + joint)
        h = c.get("hist", {})
        ls = []
        for n, lab, col in [("rl", "RL constr", "#ff7f0e"), ("cem", "CEM constr", "#17becf"),
                            ("random", "random search", "#7f7f7f"),
                            ("joint", "RL-joint", "#d62728")]:
            if n in h and h[n].get("evals"):
                ls.append({"x": h[n]["evals"],
                           "y": [max(b, BER_FLOOR) for b in h[n]["best_val_ber"]],
                           "label": lab, "color": col})
        if ls:
            plotting.semilogy(ls, xlabel="# evaluations", ylabel="best val BER (train L)",
                              title=f"PAS2 construction search: {c['M']}-QAM w={c['w']} Z={c['Z']}",
                              path=f"exp_pas2_learn_{key}.svg")
            made.append(f"exp_pas2_learn_{key}.svg")
        # joint nu trajectory
        if "joint" in h and h["joint"].get("nu_mean"):
            nu_m = c["train"]["nu_matched"]
            plotting.linear([{"x": h["joint"]["evals"], "y": h["joint"]["nu_mean"],
                              "label": "policy nu (free)", "color": "#d62728"},
                             {"x": [h["joint"]["evals"][0], h["joint"]["evals"][-1]],
                              "y": [nu_m, nu_m], "label": "matched-SE nu*", "color": "#000000"}],
                            xlabel="# evaluations", ylabel="shaping nu",
                            title=f"PAS2 joint shaping trajectory: {c['M']}-QAM w={c['w']} Z={c['Z']}",
                            path=f"exp_pas2_nu_{key}.svg")
            made.append(f"exp_pas2_nu_{key}.svg")
    # summary: required Eb/N0 @ 1e-5 at matched SE, strong-baseline vs RL-joint (L=200)
    _summary_gain_fig(cells, made)
    _export_tables(data)
    print("wrote: " + ", ".join(made) + ", results_pas2_*.csv")


def _summary_gain_fig(cells, made):
    Lr = str(REPORT_LS[-1])
    pts = []
    for key, c in cells.items():
        rep = c.get("report", {}).get(Lr, {}).get("methods", {})
        base = rep.get("uniform_cem") or rep.get("uniform_rr")
        joint = rep.get("rl_joint")
        if base and joint and base.get("ebn0_1e5") and joint.get("ebn0_1e5"):
            pts.append((base["se"], base["ebn0_1e5"] - joint["ebn0_1e5"], c["M"]))
    if not pts:
        return
    ser = []
    for M in (16, 64, 256):
        xs = [p[0] for p in pts if p[2] == M]
        ys = [p[1] for p in pts if p[2] == M]
        if xs:
            ser.append({"x": xs, "y": ys, "label": f"{M}-QAM", "color": MCOL[M],
                        "marker": "o"})
    plotting.linear(ser, xlabel="matched spectral efficiency [bits/complex sym]",
                    ylabel="RL-joint gain vs strong baseline @ BER=1e-5 [dB]",
                    title=f"PAS2 matched-SE gain (L={Lr}): RL-joint vs uniform+CEM",
                    path="exp_pas2_summary_gain.svg")
    made.append("exp_pas2_summary_gain.svg")


def _export_tables(data):
    import csv
    cells = data["cells"]
    # (a) headline: required Eb/N0 @ 1e-5 at matched SE per method per cell+L
    with open("results_pas2_required_ebn0.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cell", "M", "Z", "w", "L", "matched_SE", "method", "sc_rate", "nu",
                     "ebn0@1e-5", "ber@end", "n4", "girth",
                     "gain_vs_uniform_cem_dB", "gain_vs_uniform_rr_dB"])
        for key, c in sorted(cells.items()):
            for L in [str(x) for x in REPORT_LS]:
                rep = c.get("report", {}).get(L, {}).get("methods", {})
                base_cem = rep.get("uniform_cem", {}).get("ebn0_1e5")
                base_rr = rep.get("uniform_rr", {}).get("ebn0_1e5")
                for n in PLOT_ORDER:
                    if n not in rep:
                        continue
                    e = rep[n].get("ebn0_1e5")
                    g_cem = round(base_cem - e, 3) if (base_cem and e) else ""
                    g_rr = round(base_rr - e, 3) if (base_rr and e) else ""
                    wr.writerow([key, c["M"], c["Z"], c["w"], L,
                                 rep.get("uniform_rr", {}).get("se", ""), n,
                                 rep[n].get("sc_rate", ""), rep[n].get("nu", ""),
                                 e if e is not None else "", f"{rep[n]['y'][-1]:.3e}",
                                 rep[n].get("n4", ""), rep[n].get("girth", ""), g_cem, g_rr])
    # (b) ablations: joint vs separate, RL vs CEM vs random (train-L val BER)
    with open("results_pas2_ablations.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cell", "M", "Z", "w", "matched_SE", "nu_matched",
                     "val_RL", "val_CEM", "val_random", "val_uniformCEM",
                     "val_joint", "val_separate", "joint_free_nu",
                     "RL_beats_CEM", "RL_beats_random", "joint_beats_separate"])
        for key, c in sorted(cells.items()):
            s = c.get("search_val_ber", {})
            tr = c.get("train", {})
            rl, cem, rnd = s.get("rl"), s.get("cem"), s.get("random")
            jt, sep = s.get("joint"), s.get("separate")
            wr.writerow([key, c["M"], c["Z"], c["w"], round(tr.get("se_u", 0), 3),
                         round(tr.get("nu_matched", 0), 4),
                         _fmt(rl), _fmt(cem), _fmt(rnd), _fmt(s.get("uniform_cem")),
                         _fmt(jt), _fmt(sep), round(s.get("joint_free_nu", 0) or 0, 4),
                         _cmp(rl, cem), _cmp(rl, rnd), _cmp(jt, sep)])


def _fmt(x):
    return f"{x:.3e}" if isinstance(x, (int, float)) and np.isfinite(x) else ""


def _cmp(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return ""
    if a < b:
        return "yes"
    if a > b:
        return "no"
    return "tie"


# ----------------------------------------------------------------------------- #
#  local smoke (tiny: one cell, reduced budget, no long-L report)
# ----------------------------------------------------------------------------- #
def smoke():
    global OPT_STEPS, OPT_BATCH, OPT_FRAMES, REPORT_LS, TARGET_BITS
    global PROBE_N, VAL_FRAMES
    OPT_STEPS, OPT_BATCH, OPT_FRAMES = 2, 6, 3
    REPORT_LS = [20]
    TARGET_BITS = [1e3] * N_WATER_PTS         # tiny bits-budget, still 8 SNR points
    PROBE_N, VAL_FRAMES = 4, 6
    print("SMOKE: 1 cell, tiny budget", flush=True)
    nw = min(4, R.n_workers())
    with mp.Pool(nw) as pool:
        c = run_cell(64, 32, 3, 13, 9, pool)
    print("smoke finals (L=20):")
    for n, cur in c["report"]["20"]["methods"].items():
        print(f"  {n:13s} SE={cur['se']} nu={cur['nu']} BER@end={cur['y'][-1]:.2e} "
              f"window={cur['x'][0]}..{cur['x'][-1]}dB")
    print("SMOKE OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "plot":
        make_plots()
    elif cmd == "smoke":
        smoke()
    else:
        print(__doc__)
