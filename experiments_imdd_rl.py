"""STEP 3 (decisive): RL-optimized SC-LDPC *construction* (edge spreading) over a
bandwidth-limited IM/DD optical link.

Question
--------
Step 1+2 (``imdd_channel.py`` / ``imdd_findings.md``) established that the
post-equalization effective channel of a *sub-Nyquist* IM/DD link (PAM4, PD
B=30 GHz, DFE(6), O-band SSMF L=2 km) carries exploitable structure that ordinary
density-evolution (memoryless-AWGN) code design cannot see:
  * COLORED residual noise   (lag-1 autocorr -0.4..-0.8, SFM 0.4..0.8),
  * 2x UNEQUAL bit reliability (PAM4 LSB ~ 2x worse than MSB),
  * residual ISI the equalizer cannot flatten (corr ~0.13..0.41).
Does an RL-optimized edge-spreading construction give a real, defensible gain over
a STRONG classical reliability-matched baseline -- and if so, does the gain come
from the *non-classical* (colored / ISI) structure beyond the MSB/LSB split?

What is held fixed vs. searched
-------------------------------
* component code, windowed decoder, PAM4 Gray map, bit-INTERLEAVING: all FIXED, so
  the four constructions differ *only* in the edge-spreading assignment (the RL knob).
  With Z even and the default sequential interleave, every base column contributes a
  50/50 MSB/LSB mix, so the construction -- not a lucky interleave -- is the variable.
* edge-spreading assignment (length = #systematic base edges, entries in [0..w]):
  the only thing that changes between round_robin / reliability_matched / CEM / RL.

The four constructions (equal evaluation budget)
------------------------------------------------
  round_robin          edge e -> e mod (w+1).  naive balanced baseline.
  reliability_matched  the classical baseline to beat.  A genuine reliability-aware
                       heuristic: it protects the WEAKEST variable nodes (lowest
                       LDPC column degree) by giving their edges maximal coupling
                       DIVERSITY (each placed on a distinct, least-loaded component
                       so the few edges of a weak node never share a component),
                       while high-degree (already well-protected) nodes absorb the
                       residual load imbalance.  Uses ONLY the LDPC degree profile +
                       the known 2x LSB penalty -- i.e. exactly what classical UEP /
                       PEG-style design exploits -- and is BLIND to the colored/ISI
                       temporal structure.  (A second flavor, *_interleave, instead
                       reliability-matches via the BIT MAP -- high-degree columns ->
                       weak LSB -- to separate "match via construction" from
                       "match via mapping".)
  cem                  cross-entropy method over per-edge categoricals (strong
                       non-learned search).
  rl                   FeaturePolicy (linear-softmax sequential MDP) + PerEdgePolicy,
                       REINFORCE, reward = -log10(BER), common random numbers.

Everything routes through the REAL IM/DD chain per frame (``imdd_channel``); the
chain is ~0.1-0.25 s/frame so it runs IN THE LOOP (no surrogate) -- this captures
the colored / ISI structure exactly.  CRN: a per-frame rng seeded by (frame_seed,
idx) draws identical info bits AND seeds the PD noise identically, so every
construction in a batch is scored on the same frames+noise.

Usage:
  EXP_WORKERS=16 python3 experiments_imdd_rl.py run          # main op. point (B=30/DFE6)
  EXP_WORKERS=16 python3 experiments_imdd_rl.py run_b40      # milder-structure contrast
  python3 experiments_imdd_rl.py plot                        # figures + CSVs from JSON
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
from sc_ldpc import SCLDPCCode
from imdd_channel import (LinkConfig, forward_channel, equalize,
                          soft_metrics_pam4, pam_levels, PAM4_BITS_BY_LEVEL)

# --------------------------------------------------------------------------- #
#  regime / operating point
# --------------------------------------------------------------------------- #
BG, ILS, Z = 1, 0, 32
MP = 13                 # BG1 rate-match -> SC rate ~ 0.646 (mid rate, codable)
W_COUP, L_OPT, W_DEC = 3, 30, 6
MAXIT, ALPHA = 12, 0.8

# IM/DD channel (the richest-structure, still-codable point from Step 1+2)
PD_B = 30e9             # photodiode bandwidth [Hz] (sub-Nyquist 0.375x Nyquist)
DFE_TAPS = 6            # decision-directed DFE (FFE is catastrophic at B=30)
FFE_TAPS = 31
L_KM = 2.0             # O-band SSMF (D~0 -> isolates bandwidth damage)
ROP_OP = -9.0          # operating ROP [dBm]: median round-robin coded BER ~5e-3 (in waterfall)
ROP_WATERFALL = [-12.0, -11.0, -10.0, -9.0, -8.0, -7.0]
WF_FRAMES = [120, 160, 220, 320, 400, 400]   # more frames as BER drops

# search / eval budget (tuned for ~1-2h on 16 cores; ~1.15 s/frame in-loop)
OPT_STEPS, OPT_BATCH, OPT_FRAMES = 14, 32, 8
VAL_FRAMES = 48
FINAL_FRAMES = 400
BER_FLOOR = 5e-6

PD_B_LABEL = {30e9: "b30", 40e9: "b40"}


def cfg_imdd(L=L_OPT):
    return R.Config(bg=BG, ils=ILS, Z=Z, mp=MP, w=W_COUP, L=L,
                    W=W_DEC, max_iter=MAXIT, alpha=ALPHA)


# ----------------------------------------------------------------------------- #
#  channel-knob globals (inherited by forked workers); set by run()/run_b40()
# ----------------------------------------------------------------------------- #
_PD_B = PD_B
_DFE = DFE_TAPS
_FFE = FFE_TAPS
_LKM = L_KM

# Gray (b_hi,b_lo) -> ascending PAM4 level index (inverse of PAM4_BITS_BY_LEVEL).
# level idx 0..3 -> levels [-3,-1,+1,+3]/sqrt5 ; bits 00->0,01->1,11->2,10->3
_BITS2LVL = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
_LVL_LUT = np.array([_BITS2LVL[(0, 0)], _BITS2LVL[(0, 1)],
                     _BITS2LVL[(1, 1)], _BITS2LVL[(1, 0)]])  # placeholder; built below
_LV = pam_levels(4)


def _bits_to_levelidx(b_hi, b_lo):
    """Vectorized Gray (hi,lo)->ascending level index."""
    # 00->0, 01->1, 11->2, 10->3
    return np.where(b_hi == 0,
                    np.where(b_lo == 0, 0, 1),
                    np.where(b_lo == 0, 3, 2)).astype(np.int64)


def imdd_llr(bits, rop, frame_seed, idx, perm=None):
    """coded transmitted bits -> PAM4 (default or permuted interleave) -> IM/DD
    chain (B, DFE) -> exact per-bit Gray LLRs (same length as `bits`).

    `perm` (optional) permutes the bit stream BEFORE PAM4 mapping (a fixed
    interleaver); the inverse is applied to the LLRs so the decoder sees bits in
    their original order.  CRN: PD noise seed is a deterministic function of
    (frame_seed, idx) so every construction sees identical noise on a given frame."""
    bits = np.asarray(bits, dtype=np.int64)
    n0 = bits.size
    if perm is not None:
        bits = bits[perm]
    pad = bits.size % 2
    if pad:
        bits = np.concatenate([bits, [0]])
    b_hi = bits[0::2]
    b_lo = bits[1::2]
    lvl = _bits_to_levelidx(b_hi, b_lo)
    syms = _LV[lvl]
    # deterministic, well-mixed PD seed from (frame_seed, idx) -- pure-Python ints
    # (no numpy uint64 overflow); the PD noise is thus identical across constructions
    # for a given (frame_seed, idx) but varies frame to frame.
    pd_seed = ((2862933555777941757 * (int(frame_seed) & 0xFFFFFFFF)
                + int(idx) * 3037000493 + 12345) % 2_000_000_000)
    lc = LinkConfig(n_levels=4, nsym=syms.size, Rs=80e9, SpS=3, rolloff=0.1,
                    pd_B=_PD_B, dfe_taps=_DFE, ffe_taps=_FFE, ROP_dBm=float(rop),
                    L_km=_LKM, eq_sps=2, seed=pd_seed)
    rx, _ = forward_channel(lc, syms)
    eq = equalize(lc, rx, syms, _LV)
    yhat = eq["yhat"]
    e = yhat - syms
    sigma = float(np.sqrt(np.mean(e ** 2)) + 1e-9)
    llr, _, _ = soft_metrics_pam4(yhat, _LV, sigma)
    llr = llr[:bits.size - pad] if pad else llr
    if perm is not None:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        llr = llr[inv]
    return llr[:n0]


# ----------------------------------------------------------------------------- #
#  IM/DD evaluation worker  (the ONLY channel-specific code; swapped into R)
# ----------------------------------------------------------------------------- #
#  We extend the 7-field task tuple with an optional 8th element: a fixed bit
#  permutation (interleaver) for the reliability-matched-via-mapping variant.
def _worker_imdd(task):
    cfg_d, assign, rop, frame_seed, n_frames, lo, hi = task[:7]
    perm = task[7] if len(task) > 7 else None
    cfg = R.Config(**cfg_d)
    ckey = (cfg.bg, cfg.ils, cfg.Z, cfg.mp)
    comp = R._CACHE.get(ckey)
    if comp is None:
        comp = cfg.component(); R._CACHE[ckey] = comp
    sc = SCLDPCCode(comp, w=cfg.w, L=cfg.L, assign=np.asarray(assign, dtype=np.int64))
    be = bits = fe = 0
    for idx in range(lo, hi):
        rng = np.random.default_rng([int(frame_seed), int(idx)])     # CRN info bits
        info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
        cw, _ = sc.encode(info)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = imdd_llr(cw[sc.tx_mask], rop, frame_seed, idx, perm=perm)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=cfg.W, max_iter=cfg.max_iter, alpha=cfg.alpha)
        err = int((sc.extract_info(hard) != info).sum())
        be += err; bits += info.size; fe += int(err > 0)
    return be, bits, fe, hi - lo


R._worker = _worker_imdd        # route ALL R.eval_* through the IM/DD channel


# A perm-aware evaluation (rl_construct's eval_assignments builds 7-tuples; we add
# the 8th perm field by post-processing).  Only used for the *_interleave baseline.
def eval_with_perm(c, assign, rop, frame_seed, n_frames, pool, perm, chunks):
    bounds = np.linspace(0, n_frames, chunks + 1).astype(int)
    cfg_d = R.asdict(c) if hasattr(R, "asdict") else __import__("dataclasses").asdict(c)
    aa = np.asarray(assign, dtype=np.int64)
    tasks = []
    for ci in range(chunks):
        lo, hi = int(bounds[ci]), int(bounds[ci + 1])
        if hi > lo:
            tasks.append((cfg_d, aa, rop, frame_seed, n_frames, lo, hi, perm))
    raw = pool.map(_worker_imdd, tasks) if pool is not None else [_worker_imdd(t) for t in tasks]
    be = bits = fe = nf = 0
    for a, b, f, n in raw:
        be += a; bits += b; fe += f; nf += n
    return {"ber": be / max(bits, 1), "fer": fe / max(nf, 1), "bit_err": be,
            "n_info": bits, "frame_err": fe, "n_frames": nf}


# ----------------------------------------------------------------------------- #
#  reliability-matched classical construction (the baseline to beat)
# ----------------------------------------------------------------------------- #
def reliability_matched_assign(cfg):
    """Degree-aware edge spreading: protect the WEAKEST variable nodes (lowest
    LDPC base column degree) by giving each of their edges a DISTINCT, least-loaded
    coupling component (maximal coupling diversity), processing columns weakest-
    first.  Strong (high-degree) columns are placed last and absorb the residual
    load imbalance.  Uses only the degree profile + the known LSB penalty; blind to
    colored/ISI temporal structure.  This is a genuine classical reliability-aware
    heuristic (PEG/UEP spirit), strictly better-motivated than round-robin."""
    rows, cols, E = R.edge_meta(cfg)
    C = R.n_components(cfg)
    comp = cfg.component()
    coldeg = (comp.B >= 0).sum(axis=0)              # base VN degree per column
    assign = np.full(E, -1, dtype=np.int64)
    load = np.zeros(C, dtype=np.int64)
    # canonical systematic-edge order is row-major over the systematic part; group
    # edges by their column and order columns by increasing degree (weak first).
    edge_cols = cols
    order_cols = np.argsort(coldeg[:int(edge_cols.max()) + 1], kind="stable")
    for col in order_cols:
        es = np.where(edge_cols == col)[0]
        if es.size == 0:
            continue
        # weakest columns: each edge to a distinct least-loaded component (diversity).
        used = set()
        for e in es:
            cand = np.argsort(load + np.array([1e6 if k in used else 0 for k in range(C)]))
            a = int(cand[0])
            assign[e] = a; load[a] += 1; used.add(a)
            if len(used) == C:
                used = set()                         # reset once all comps used by this col
    # any edge not covered (shouldn't happen) -> round-robin fallback
    miss = np.where(assign < 0)[0]
    assign[miss] = miss % C
    return assign


def uep_interleave(cfg):
    """A fixed bit-INTERLEAVER that reliability-matches via the MAP: it routes the
    bits of HIGH-degree systematic columns (strong LDPC protection) onto the WEAK
    LSB PAM4 position and LOW-degree columns onto the strong MSB position -- textbook
    UEP channel matching.  Returns a permutation of range(N_tx) applied before PAM4
    mapping.  (Used only for the reliability_matched_interleave contrast.)"""
    sc = cfg.build()
    Z, nb = sc.Z, sc.nb
    tx_idx = np.nonzero(sc.tx_mask)[0]
    col_of = (tx_idx % (nb * Z)) // Z
    comp = cfg.component()
    coldeg = (comp.B >= 0).sum(axis=0).astype(float)
    # protection score per transmitted bit = its column's LDPC degree (higher=stronger)
    score = coldeg[col_of]
    Nt = tx_idx.size
    Nt2 = Nt - (Nt % 2)
    # MSB positions are even indices, LSB positions are odd (default pairing).  We
    # want strong-column (high score) bits on LSB (odd), weak on MSB (even).
    order = np.argsort(score, kind="stable")        # weak..strong
    perm = np.empty(Nt, dtype=np.int64)
    half = Nt2 // 2
    # weakest half -> MSB slots (even), strongest half -> LSB slots (odd)
    msb_slots = np.arange(0, Nt2, 2)
    lsb_slots = np.arange(1, Nt2, 2)
    perm[msb_slots] = order[:half]
    perm[lsb_slots] = order[half:2 * half][::-1]
    if Nt % 2:                                       # trailing pad bit stays put
        perm[Nt - 1] = order[-1]
    # perm maps NEW position -> OLD bit index; imdd_llr expects bits[perm], so we
    # return the inverse (OLD->NEW) consumed as a gather.  Build OLD->NEW.
    inv = np.empty(Nt, dtype=np.int64)
    inv[perm] = np.arange(Nt)
    return inv


# ----------------------------------------------------------------------------- #
#  BER-centric training loops (policies + parallel eval reused from rl_construct)
# ----------------------------------------------------------------------------- #
def _logber(m):
    return -np.log10(max(m["ber"], BER_FLOOR))


def _validate(c, assign, rop, nframes, seed, pool):
    return R.eval_assignments(c, [assign], rop, seed, nframes, pool=pool,
                              frame_chunks=R.n_workers())[0]


def train_pg(c, policy, rop, steps, batch, frames, pool, val_frames, val_seed,
             base_seed, tag, log_every=2):
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
        ms = R.eval_batch(c, assigns, rop, fs, frames, pool=pool)
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
        vm = _validate(c, assigns[bi], rop, val_frames, val_seed, pool)
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"], "assign": np.asarray(assigns[bi]).copy()}
        hist["evals"].append(seen); hist["mean_reward"].append(float(rew.mean()))
        hist["best_val_ber"].append(best["ber"]); hist["best_val_fer"].append(best["fer"])
        if step % log_every == 0 or step == 1:
            extra = ""
            if feature:
                extra = "  theta=[" + ",".join(f"{t:+.2f}" for t in policy.theta) + "]"
            print(f"  [{tag}] step {step:3d} meanR={rew.mean():+.2f} "
                  f"batchBER[min={min(m['ber'] for m in ms):.2e}] "
                  f"bestVAL_BER={best['ber']:.2e} (FER={best['fer']:.2f}){extra}", flush=True)
    return hist, best


def train_cem(c, rop, steps, batch, frames, pool, val_frames, val_seed, base_seed,
              elite_frac=0.3, smooth=0.7, seed=7, log_every=2):
    _, _, E = R.edge_meta(c); C = R.n_components(c)
    rng = np.random.default_rng(seed)
    p = np.full((E, C), 1.0 / C); n_elite = max(2, int(batch * elite_frac))
    hist = {"evals": [], "best_val_ber": [], "best_val_fer": []}
    best = {"ber": np.inf, "fer": np.inf, "assign": None}; seen = 0
    for step in range(1, steps + 1):
        fs = base_seed + step
        assigns = [np.array([rng.choice(C, p=p[e]) for e in range(E)]) for _ in range(batch)]
        ms = R.eval_batch(c, assigns, rop, fs, frames, pool=pool); seen += batch
        bers = np.array([m["ber"] for m in ms])
        elite = np.argsort(bers)[:n_elite]
        freq = np.zeros((E, C))
        for idx in elite:
            freq[np.arange(E), assigns[idx]] += 1.0
        freq /= n_elite
        p = smooth * p + (1 - smooth) * freq
        p = np.clip(p, 1e-3, None); p /= p.sum(axis=1, keepdims=True)
        cand = assigns[int(np.argmin(bers))]
        vm = _validate(c, cand, rop, val_frames, val_seed, pool)
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"], "assign": np.asarray(cand).copy()}
        hist["evals"].append(seen); hist["best_val_ber"].append(best["ber"])
        hist["best_val_fer"].append(best["fer"])
        if step % log_every == 0 or step == 1:
            print(f"  [CEM] step {step:3d} batchBER[min={bers.min():.2e}] "
                  f"bestVAL_BER={best['ber']:.2e}", flush=True)
    return hist, best


# ----------------------------------------------------------------------------- #
#  BER waterfall vs ROP for one construction
# ----------------------------------------------------------------------------- #
def ber_waterfall(c, assign, rops, frames, pool, perm=None, chunks=None):
    chunks = chunks or R.n_workers()
    xs, bers, fers = [], [], []
    for rop, nf in zip(rops, frames):
        if perm is None:
            m = R.eval_assignments(c, [assign], rop, 2025, nf, pool=pool, frame_chunks=chunks)[0]
        else:
            m = eval_with_perm(c, assign, rop, 2025, nf, pool, perm, chunks)
        xs.append(round(rop, 2)); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def required_rop(curve, target):
    """Linear interpolation (in dB vs log10 BER) of the ROP achieving `target` BER.
    Returns None if the curve never reaches it within the swept range."""
    x = np.array(curve["x"], dtype=float)
    y = np.array(curve["y"], dtype=float)
    ok = y > 0
    x, y = x[ok], y[ok]
    if x.size < 2:
        return None
    ly = np.log10(y); lt = np.log10(target)
    # find a bracketing pair (BER crosses target as ROP increases)
    for i in range(len(x) - 1):
        a, b = ly[i], ly[i + 1]
        if (a - lt) * (b - lt) <= 0 and a != b:
            f = (lt - a) / (b - a)
            return float(x[i] + f * (x[i + 1] - x[i]))
    return None


# ----------------------------------------------------------------------------- #
#  construction analysis: classical (n4/girth/load) + ISI/temporal-locality metric
# ----------------------------------------------------------------------------- #
def temporal_locality_metric(cfg, assign, n_probe_frames=0):
    """How well a construction DECORRELATES temporally-adjacent coded bits across
    checks -- the lever the colored/ISI structure exposes but classical DE cannot
    model.  Residual ISI couples PAM4 symbol k with k+-1, i.e. *consecutive* coded
    bit pairs.  If two coded bits that map to adjacent symbols share a check, their
    (correlated) residual errors hit the same parity equation together, which hurts
    BP.  We measure, over the lifted coupled Tanner graph, the mean number of checks
    shared by pairs of variable nodes whose transmitted-bit positions are within a
    small window D (proxy for ISI span).  LOWER = better decorrelation.

    Returns dict with shared-check counts at lag 1/2 and the 4-cycle/girth stats."""
    sc = SCLDPCCode(cfg.component(), w=cfg.w, L=cfg.L, assign=np.asarray(assign, dtype=np.int64))
    tan = sc.full_tanner()
    nb, Z = sc.nb, sc.Z
    # transmitted-variable -> linear transmitted position (the PAM4 symbol stream order)
    txpos = -np.ones(sc.num_var, dtype=np.int64)
    txpos[sc.tx_mask] = np.arange(int(sc.tx_mask.sum()))
    # symbol index of each transmitted bit (2 bits/symbol, sequential interleave)
    symidx = np.where(txpos >= 0, txpos // 2, -1)
    # per check, the set of variable nodes; accumulate co-membership by symbol lag
    chk_vars = [[] for _ in range(tan.num_chk)]
    for cc, vv in zip(tan.e_chk.tolist(), tan.e_var.tolist()):
        chk_vars[cc].append(vv)
    lag_shared = {1: 0, 2: 0, 3: 0}
    lag_pairs = {1: 0, 2: 0, 3: 0}
    for vs in chk_vars:
        sv = sorted(set(vs))
        s_syms = [symidx[v] for v in sv if symidx[v] >= 0]
        s_syms.sort()
        # count pairs within this check at small symbol lag
        for i in range(len(s_syms)):
            for j in range(i + 1, len(s_syms)):
                d = s_syms[j] - s_syms[i]
                if d > 3:
                    break
                if d in lag_shared:
                    lag_shared[d] += 1
    # normalize by total #adjacent-symbol pairs that exist anywhere (rough density)
    out = {
        "shared_checks_lag1": int(lag_shared[1]),
        "shared_checks_lag2": int(lag_shared[2]),
        "shared_checks_lag3": int(lag_shared[3]),
    }
    st = R.construction_stats(cfg, assign)
    out.update(n4=st["n4"], comp_load=st["comp_load"], comp_load_std=st["comp_load_std"],
               chk_deg_mean=st["chk_deg_mean"], chk_deg_max=st["chk_deg_max"])
    try:
        out["girth"] = R.girth(sc, max_g=12, n_seeds=200, seed=0)
    except Exception:
        out["girth"] = None
    return out


# ----------------------------------------------------------------------------- #
#  main run (one operating point, done thoroughly)
# ----------------------------------------------------------------------------- #
def run(pd_b=PD_B, dfe=DFE_TAPS, ffe=FFE_TAPS, rop_op=ROP_OP, tag=None):
    global _PD_B, _DFE, _FFE
    _PD_B, _DFE, _FFE = pd_b, dfe, ffe
    tag = tag or PD_B_LABEL.get(pd_b, "bX")
    t0 = time.time()
    c = cfg_imdd(L_OPT)
    rows, cols, E = R.edge_meta(c)
    sc = c.build()
    print(f"=== IM/DD RL construction :: PD B={pd_b/1e9:.0f}GHz DFE{dfe} L={L_KM}km PAM4 ===", flush=True)
    print(f"  code BG{BG} Z={Z} mp={MP} w={W_COUP} L={L_OPT} -> SC rate {sc.rate:.3f} "
          f"K={sc.K} N_tx={sc.N_tx} E={E} space {W_COUP+1}^{E}", flush=True)
    print(f"  operating ROP={rop_op}dBm  workers={R.n_workers()}", flush=True)

    with mp.Pool(R.n_workers()) as pool:
        # warmup one frame to JIT the optic chain in every worker process
        _ = R.eval_batch(c, [R.round_robin_assign(c)], rop_op, 1, R.n_workers(), pool=pool)

        # ---- baselines ----
        rr = R.round_robin_assign(c)
        relm = reliability_matched_assign(c)
        perm_uep = uep_interleave(c)

        # ---- RL (FeaturePolicy) ----
        print("\n-- RL FeaturePolicy (REINFORCE, reward -log10 BER) --", flush=True)
        pol = R.FeaturePolicy(c, lr=0.15, ent=0.02, seed=0)
        h_rl, best_rl = train_pg(c, pol, rop_op, OPT_STEPS, OPT_BATCH, OPT_FRAMES,
                                 pool, VAL_FRAMES, 99, 1000, "PG")
        # ---- RL (PerEdgePolicy) ----
        print("\n-- RL PerEdgePolicy --", flush=True)
        pol_e = R.PerEdgePolicy(E, W_COUP + 1, lr=0.2, ent=0.01, seed=0)
        h_pe, best_pe = train_pg(c, pol_e, rop_op, OPT_STEPS, OPT_BATCH, OPT_FRAMES,
                                 pool, VAL_FRAMES, 99, 4000, "PGe")
        # ---- CEM ----
        print("\n-- CEM --", flush=True)
        h_cem, best_cem = train_cem(c, rop_op, OPT_STEPS, OPT_BATCH, OPT_FRAMES,
                                    pool, VAL_FRAMES, 99, 2000)

        champions = {
            "round_robin": np.asarray(rr, dtype=np.int64),
            "reliability_matched": np.asarray(relm, dtype=np.int64),
            "cem": np.asarray(best_cem["assign"], dtype=np.int64),
            "rl": np.asarray(best_rl["assign"], dtype=np.int64),
            "rl_peredge": np.asarray(best_pe["assign"], dtype=np.int64),
        }

        # ---- score all at the operating point (large bank) + waterfalls ----
        print("\n-- scoring champions @ operating point + waterfalls --", flush=True)
        finals, curves, analysis = {}, {}, {}
        for name, a in champions.items():
            m = R.eval_assignments(c, [a], rop_op, 12345, FINAL_FRAMES, pool=pool,
                                   frame_chunks=R.n_workers())[0]
            finals[name] = {"ber": m["ber"], "fer": m["fer"], "n_frames": m["n_frames"]}
            curves[name] = ber_waterfall(c, a, ROP_WATERFALL, WF_FRAMES, pool)
            analysis[name] = temporal_locality_metric(c, a)
            print(f"  {name:22s} BER={m['ber']:.3e} FER={m['fer']:.2f} "
                  f"n4={analysis[name]['n4']} girth={analysis[name]['girth']} "
                  f"load_std={analysis[name]['comp_load_std']:.2f} "
                  f"sharedISI(1,2)=({analysis[name]['shared_checks_lag1']},"
                  f"{analysis[name]['shared_checks_lag2']})", flush=True)

        # ---- reliability_matched_interleave (UEP via bit map, round-robin edges) ----
        print("\n-- reliability_matched_interleave (UEP bit-map, round-robin edges) --", flush=True)
        m_int_op = eval_with_perm(c, rr, rop_op, 12345, FINAL_FRAMES, pool, perm_uep, R.n_workers())
        curve_int = ber_waterfall(c, rr, ROP_WATERFALL, WF_FRAMES, pool, perm=perm_uep)
        finals["reliability_matched_interleave"] = {"ber": m_int_op["ber"], "fer": m_int_op["fer"],
                                                    "n_frames": m_int_op["n_frames"]}
        curves["reliability_matched_interleave"] = curve_int
        print(f"  reliability_matched_interleave BER={m_int_op['ber']:.3e} FER={m_int_op['fer']:.2f}",
              flush=True)

    # ---- required-ROP table @ BER=1e-4 and 1e-5 ----
    req = {}
    for name, cv in curves.items():
        req[name] = {"1e-3": required_rop(cv, 1e-3),
                     "1e-4": required_rop(cv, 1e-4),
                     "1e-5": required_rop(cv, 1e-5)}

    out = {
        "tag": tag, "channel": {"pd_B_GHz": pd_b / 1e9, "dfe": dfe, "ffe": ffe,
                                "L_km": L_KM, "eq": "DFE" if dfe > 0 else "FFE"},
        "code": {"bg": BG, "Z": Z, "mp": MP, "w": W_COUP, "L": L_OPT, "W": W_DEC,
                 "rate": sc.rate, "K": sc.K, "N_tx": sc.N_tx, "E": int(E)},
        "rop_op": rop_op, "rop_waterfall": ROP_WATERFALL, "wf_frames": WF_FRAMES,
        "budget": {"opt_steps": OPT_STEPS, "opt_batch": OPT_BATCH, "opt_frames": OPT_FRAMES,
                   "val_frames": VAL_FRAMES, "final_frames": FINAL_FRAMES},
        "finals": finals, "curves": curves, "required_rop": req, "analysis": analysis,
        "hist": {"rl": h_rl, "rl_peredge": h_pe, "cem": h_cem},
        "theta_rl": pol.theta.tolist(),
        "champions": {k: v.tolist() for k, v in champions.items()},
    }
    fn = f"results_imdd_rl_{tag}.json"
    OP.save(out, fn)
    print(f"\n[run {tag} done in {time.time()-t0:.0f}s] -> {fn}", flush=True)
    _print_verdict(out)
    return out


def _print_verdict(out):
    f = out["finals"]; req = out["required_rop"]
    print("\n================ DB TABLE (operating point + required ROP) ================")
    print(f"  operating ROP={out['rop_op']}dBm   (lower required ROP = better)")
    order = ["round_robin", "reliability_matched", "reliability_matched_interleave",
             "cem", "rl_peredge", "rl"]
    print(f"  {'construction':30s} {'BER@op':>10s}  {'reqROP@1e-4':>11s} {'reqROP@1e-5':>11s}")
    for n in order:
        if n not in f:
            continue
        r4 = req.get(n, {}).get("1e-4"); r5 = req.get(n, {}).get("1e-5")
        print(f"  {n:30s} {f[n]['ber']:10.3e}  "
              f"{('%.2f' % r4) if r4 is not None else '   n/a':>11s} "
              f"{('%.2f' % r5) if r5 is not None else '   n/a':>11s}")

    def gain(a, b, lvl):
        ra = req.get(a, {}).get(lvl); rb = req.get(b, {}).get(lvl)
        if ra is None or rb is None:
            return None
        return rb - ra            # dB the better code saves (b worse - a better)
    print("\n  HEADLINE dB GAINS (required ROP, + = RL better):")
    for lvl in ["1e-4", "1e-5"]:
        g_rm = gain("rl", "reliability_matched", lvl)
        g_cem = gain("rl", "cem", lvl)
        g_rr = gain("rl", "round_robin", lvl)
        def s(x):
            return f"{x:+.2f}dB" if x is not None else "n/a"
        print(f"    @BER={lvl}:  RL-vs-relMatched={s(g_rm)}   RL-vs-CEM={s(g_cem)}   "
              f"RL-vs-roundRobin={s(g_rr)}")
    # BER-ratio at operating point (a second, floor-robust comparison)
    if "rl" in f and "reliability_matched" in f:
        rr_ber = f["reliability_matched"]["ber"]; rl_ber = f["rl"]["ber"]
        print(f"  BER@op ratio relMatched/RL = {rr_ber/max(rl_ber,1e-9):.2f}x "
              f"(>1 => RL better at the operating point)")
    print("===========================================================================\n")


# ----------------------------------------------------------------------------- #
#  B=40 milder-structure contrast (FFE works there)
# ----------------------------------------------------------------------------- #
def run_b40():
    # at B=40/FFE the channel is much cleaner; pick a lower ROP so it sits in the
    # waterfall.  Reuse the same code; sweep ROP downward.
    global ROP_WATERFALL, WF_FRAMES
    ROP_WATERFALL = [-16.0, -15.0, -14.0, -13.0, -12.0, -11.0]
    WF_FRAMES = [120, 160, 220, 320, 400, 400]
    return run(pd_b=40e9, dfe=0, ffe=41, rop_op=-13.0, tag="b40")


# ----------------------------------------------------------------------------- #
#  plotting
# ----------------------------------------------------------------------------- #
LAB = {"rl": "RL feature [ours]", "rl_peredge": "RL per-edge [ours]", "cem": "CEM",
       "reliability_matched": "reliability-matched (classical)",
       "reliability_matched_interleave": "reliability-matched (UEP bit-map)",
       "round_robin": "round-robin"}
COL = {"rl": "#d62728", "rl_peredge": "#ff7f0e", "cem": "#9467bd",
       "reliability_matched": "#2ca02c", "reliability_matched_interleave": "#17becf",
       "round_robin": "#8c564b"}


def make_plots():
    for tag in ["b30", "b40"]:
        fn = f"results_imdd_rl_{tag}.json"
        try:
            out = OP.load(fn)
        except FileNotFoundError:
            print(f"(no {fn}; skipping)")
            continue
        rate = out["code"]["rate"]; ch = out["channel"]
        ttl = f"SC-LDPC over IM/DD PAM4 B={ch['pd_B_GHz']:.0f}GHz {ch['eq']} (R={rate:.2f}, L={out['code']['L']})"
        # Fig 1: BER waterfalls
        order = ["rl", "cem", "reliability_matched", "reliability_matched_interleave",
                 "round_robin"]
        ser = [{"x": out["curves"][n]["x"], "y": out["curves"][n]["y"],
                "label": LAB[n], "color": COL[n]} for n in order if n in out["curves"]]
        plotting.semilogy(ser, xlabel="received optical power ROP [dBm]", ylabel="coded info BER",
                          title=ttl, path=f"exp_imdd_rl_waterfall_{tag}.svg")
        # Fig 2: learning curves (sample efficiency)
        h = out["hist"]
        ls = [{"x": h[m]["evals"], "y": [max(b, BER_FLOOR) for b in h[m]["best_val_ber"]],
               "label": LAB.get(m, m), "color": COL.get(m, "#333")}
              for m in ["rl", "rl_peredge", "cem"] if m in h]
        plotting.semilogy(ls, xlabel="# construction evaluations", ylabel="best validation BER",
                          title=f"IM/DD {tag} learning curves (sample efficiency)",
                          path=f"exp_imdd_rl_learn_{tag}.svg")
        # Fig 3: construction analysis bar-style (n4 + shared-ISI-checks) as linear plot
        an = out["analysis"]
        names = [n for n in order if n in an]
        x = list(range(len(names)))
        s_n4 = {"x": x, "y": [an[n]["n4"] for n in names], "label": "4-cycles (n4)", "color": "#1f77b4"}
        s_isi = {"x": x, "y": [an[n]["shared_checks_lag1"] for n in names],
                 "label": "ISI-adjacent shared checks (lag1)", "color": "#d62728"}
        plotting.linear([s_n4, s_isi], xlabel="construction (0=RL..)", ylabel="count",
                        title=f"IM/DD {tag} construction structure: "
                              f"[{', '.join(names)}]",
                        path=f"exp_imdd_rl_analysis_{tag}.svg")
        _export_csv(out, tag)
        print(f"wrote exp_imdd_rl_*_{tag}.svg + results_imdd_rl_{tag}.csv")


def _export_csv(out, tag):
    import csv
    with open(OP.route(f"results_imdd_rl_{tag}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["construction", "ber_op", "fer_op", "reqROP_1e-3", "reqROP_1e-4",
                     "reqROP_1e-5", "n4", "girth", "comp_load_std", "shared_isi_lag1",
                     "shared_isi_lag2", "shared_isi_lag3"])
        req = out["required_rop"]; an = out["analysis"]; fin = out["finals"]
        for n in ["round_robin", "reliability_matched", "reliability_matched_interleave",
                  "cem", "rl_peredge", "rl"]:
            if n not in fin:
                continue
            a = an.get(n, {})
            wr.writerow([n, f"{fin[n]['ber']:.4e}", f"{fin[n]['fer']:.3f}",
                         req.get(n, {}).get("1e-3"), req.get(n, {}).get("1e-4"),
                         req.get(n, {}).get("1e-5"),
                         a.get("n4", ""), a.get("girth", ""),
                         f"{a.get('comp_load_std', 0):.3f}" if a else "",
                         a.get("shared_checks_lag1", ""), a.get("shared_checks_lag2", ""),
                         a.get("shared_checks_lag3", "")])
    # waterfall CSV
    with open(OP.route(f"results_imdd_rl_{tag}_waterfall.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["construction", "rop_dBm", "ber", "fer"])
        for n, cv in out["curves"].items():
            for i in range(len(cv["x"])):
                wr.writerow([n, cv["x"][i], f"{cv['y'][i]:.4e}", f"{cv['fer'][i]:.3f}"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "run":
        run()
    elif cmd == "run_b40":
        run_b40()
    elif cmd == "plot":
        make_plots()
    else:
        print(__doc__)
