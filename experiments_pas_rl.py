"""RL for INTEGRATED coding + modulation: jointly optimise SC-LDPC construction
(edge spreading) AND QAM Probabilistic Amplitude Shaping (PAS) over complex AWGN.

Two parts (run together by `run`; each writes its own resumable JSON):

PART 1 -- shaping gain (fast, information-theoretic).  For M in {4,16,64,256} we
grid-search the Maxwell-Boltzmann shaping nu* maximising the BMD (bit-metric
decoding) rate R_BMD at each Es/N0, and report the dB SNR saving at matched R_BMD
targets.  QPSK is the null sanity (~0 gain); the gain grows with M towards the
1.53 dB shaping-gain asymptote.  Pure pas.r_bmd, cheap, embarrassingly parallel.

PART 2 -- JOINT RL (the headline).  At a target net spectral efficiency and an
operating SNR (chosen BER-centrically by a probe, like the PAM4 run), REINFORCE
jointly optimises (i) the shaping nu (a small Gaussian NuPolicy) and (ii) the
SC-LDPC edge-spreading `assign` (FeaturePolicy / PerEdgePolicy reused verbatim
from rl_construct) to MINIMISE coded BER (reward = -log10(BER), common random
numbers).  Baselines at EQUAL eval budget: uniform+round_robin, shaped(nu*)+
round_robin, shaped+random-construction, and RL-joint.  We then draw matched-net-
rate BER waterfalls (uniform vs shaped(nu*) vs RL-joint), learning curves, and a
table of learned nu*/construction stats (n4/girth) and the achieved shaping gain.

PAS chain (idealised i.i.d. distribution matcher -- the standard "i.i.d. shaping"
evaluation, no CCDM):
  * the SC-LDPC SYSTEMATIC INFO carries SHAPED AMPLITUDE bits: per QAM axis we draw
    the k-bit Gray *level label* from the shaped per-axis signed-level PMF (signs
    uniform, amplitudes ~ MB), so the transmitted info-derived symbols are shaped;
  * PARITY (and the punctured/known structure) is the ordinary 5G NR encoding;
  * QAM-modulate at the shaped Es=1 geometry, complex AWGN, A-PRIORI-AWARE soft
    demap (pas.soft_demap_apriori -> the decoder sees the shaping prior), windowed
    BP, measure info BER/FER;
  * the operational NET RATE is measured, not assumed: net bits/cu = (shaped info
    entropy that the receiver recovers) / (complex channel uses).  Matched-rate
    comparisons adjust nu / code rate so shaped and uniform sit at the same net SE.

All numpy-only (pas/qam reuse qam._logsumexp); parallel over rl_construct.n_workers()
with EXP_WORKERS honoured.  We SWAP IN a PAS evaluation worker (R._worker =
_worker_pas) so every R.eval_* routes frames through the shaped-QAM chain, exactly
as experiments_construct_pam4.py swaps in its PAM4 worker.

    python3 experiments_pas_rl.py run     # PART 1 + PART 2, resumable JSON checkpoints
    python3 experiments_pas_rl.py plot    # merge -> SVG figures + CSV tables
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
import qam
import pas
from sc_ldpc import SCLDPCCode

# ----------------------------------------------------------------------------- #
#  regime
# ----------------------------------------------------------------------------- #
BG, ILS = 1, 0
W_DEC_MIN, MAXIT, ALPHA = 6, 12, 0.8
OPT_L = 12                       # chain length while optimising (edge spreading is L-independent)
RATE_MP = {0.5: 24, 0.667: 13, 0.75: 9, 0.833: 6, 0.875: 5}   # BG1 (Kb=22) rate matching
BER_FLOOR = 2e-6

# ---- PART 1 (shaping gain) ----
P1_ORDERS = [4, 16, 64, 256]
P1_ESN0 = list(np.round(np.arange(2.0, 28.01, 2.0), 2))       # Es/N0 grid for R_BMD curves
P1_NUS = list(np.round(np.concatenate([[0.0], np.linspace(0.005, 0.35, 28)]), 4))
P1_NSYM = 60000                                               # MC symbols per (M,nu,snr)
P1_TARGETS = {16: [2.0, 3.0], 64: [3.0, 4.5], 256: [4.0, 6.0, 7.0]}   # matched-R_BMD dB-gain targets

# ---- PART 2 (joint RL) ----
# representative grid: a few rates x w, M chosen to expose shaping (>=16-QAM), Z in {32,64}.
# (rate, w, M, Z) cells; kept small so the pod finishes in a few hours.
P2_CELLS = [
    (0.75, 3, 64, 32),     # 64-QAM, Z=32 -> K~few hundred*... mid length
    (0.833, 4, 64, 32),    # higher rate, larger memory
    (0.75, 3, 256, 32),    # 256-QAM (biggest shaping gain) at Z=32
    (0.833, 4, 256, 64),   # 256-QAM, Z=64 long code, high rate
]
P2_L = 50                  # chain length for the coded sims
# OPT_BATCH=96 fills the 120-core pod (one construction-eval per core) at ~the same
# wall-time as batch 24 on 24 cores, but a 4x larger REINFORCE batch = lower-variance
# joint gradient.  (raised from 24 when the run moved from a 40c to the 120c pod.)
OPT_STEPS, OPT_BATCH, OPT_FRAMES = 14, 96, 6
VAL_FRAMES, FINAL_FRAMES = 30, 200
PROBE_N, PROBE_FRAMES = 24, 8
TARGET_BER = 3e-3
BISECT_ITERS = 7
# info Eb/N0 brackets spanning the (steep) waterfall for these short-L=12 PAS cells.
# Verified locally: M=64/256 R=0.75 L=12 cliff ~6-8 dB (BER 8.5e-2@6 -> 3e-5@8), so the
# BER~3e-3 operating point is ~7 dB.  The earlier (8,22) floor sat ABOVE the cliff ->
# the probe pinned at BER~0 (no RL gradient).  Wide low floors let the bisection land it.
SNR_BRACKET = {0.75: (2.0, 14.0), 0.833: (3.0, 16.0)}
BER_OFFSETS = [-1.0, 0.0, 1.0, 2.0]
BER_FRAMES = [120, 180, 260, 360]

N_PROC = 1                 # PART2 runs as ONE process (parts are cheap relative to a pool);
                           # the pool of n_workers() cores does the heavy lifting per cell.


# ----------------------------------------------------------------------------- #
#  config helpers
# ----------------------------------------------------------------------------- #
def cfg_for(M, rate, Z, w, L):
    """Build an rl_construct.Config; W = max(6, w+2) (>= w+1, the SC requirement)."""
    return R.Config(bg=BG, ils=ILS, Z=Z, mp=RATE_MP[rate], w=w, L=L,
                    W=max(W_DEC_MIN, w + 2), max_iter=MAXIT, alpha=ALPHA)


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


# ----------------------------------------------------------------------------- #
#  shaped systematic-info generation (idealised i.i.d. amplitude shaping)
# ----------------------------------------------------------------------------- #
def _level_bits_table(M):
    """[2^k, k] Gray bit labels indexed by ascending level index (== qam bit_table)."""
    return qam._qam_tables(M)["bit_table"]


def shaped_info_bits(M, amp_pmf, K, rng):
    """K systematic info bits whose per-axis k-bit groups are drawn from the shaped
    signed-level PMF (amplitudes ~ MB, signs uniform).  Any tail bits (K not a
    multiple of k) are uniform.  This is the idealised distribution matcher."""
    T = qam._qam_tables(M)
    k = T["k"]
    bt = T["bit_table"]                       # [2^k, k] Gray labels by level idx
    plv = pas.axis_level_pmf(M, amp_pmf)      # PMF over level indices (ascending)
    n_groups = K // k
    rem = K - n_groups * k
    out = np.empty(K, dtype=np.uint8)
    if n_groups:
        lv = rng.choice(plv.size, size=n_groups, p=plv)
        out[: n_groups * k] = bt[lv].reshape(-1)
    if rem:
        out[n_groups * k:] = rng.integers(0, 2, size=rem).astype(np.uint8)
    return out


def axis_entropy_frac(M, amp_pmf):
    """Per-axis level entropy / k  (fraction of raw bits carried after shaping)."""
    k = qam._qam_tables(M)["k"]
    return pas.entropy_bits(pas.axis_level_pmf(M, amp_pmf)) / k


def net_rate_operational(sc, M, amp_pmf):
    """Operational net spectral efficiency [info bits / complex channel use].

    Channel uses = (#transmitted bits)/m.  The recovered info is the shaped level
    sequence; its entropy per systematic info bit is `axis_entropy_frac`.  We count
    the SHAPED info content (entropy) the receiver recovers, divided by channel uses,
    so shaped and uniform systems are compared at matched recovered-information rate."""
    n_tx_bits = int(sc.tx_mask.sum())
    n_cu = n_tx_bits / qam.bits_per_symbol(M)
    h_frac = axis_entropy_frac(M, amp_pmf)            # 1.0 for uniform
    return (sc.K * h_frac) / n_cu


# ----------------------------------------------------------------------------- #
#  PAS evaluation worker (the ONLY channel-specific code; swapped into R)
# ----------------------------------------------------------------------------- #
def _worker_pas(task):
    """Evaluate one (assign, nu) over a CRN frame range [lo,hi) through shaped QAM.

    The 7-field task tuple is (cfg_dict, payload, snr, frame_seed, n_frames, lo, hi)
    where `payload` packs the construction assign AND the shaping nu/pmf and the QAM
    order M, so R.eval_assignments is reused verbatim (it just forwards `payload`).
    snr is the INFO-bit Eb/N0; sigma uses the operational net rate so shaped/uniform
    compare at matched info-bit energy.  CRN: per-frame rng seeded by (frame_seed,idx)
    -> identical info+noise for every code in a batch (low-variance advantage)."""
    cfg_d, payload, ebn0, frame_seed, n_frames, lo, hi = task
    assign, M, nu, pmf_list = payload
    cfg = R.Config(**cfg_d)
    ckey = (cfg.bg, cfg.ils, cfg.Z, cfg.mp)
    comp = R._CACHE.get(ckey)
    if comp is None:
        comp = cfg.component(); R._CACHE[ckey] = comp
    sc = SCLDPCCode(comp, w=cfg.w, L=cfg.L, assign=np.asarray(assign, dtype=np.int64))
    amp_pmf = np.asarray(pmf_list) if pmf_list is not None else pas.mb_pmf(M, nu)
    chkey = ("pas", M, round(float(nu), 6) if pmf_list is None else id(pmf_list))
    chan = R._CACHE.get(chkey)
    if chan is None:
        chan = pas.ShapedQAMChannel(M, nu=nu, amp_pmf=amp_pmf); R._CACHE[chkey] = chan
    net_rate = net_rate_operational(sc, M, amp_pmf)
    sigma = pas.ebn0_to_sigma(ebn0, net_rate)
    be = bits = fe = 0
    for idx in range(lo, hi):
        rng = np.random.default_rng([int(frame_seed), int(idx)])    # CRN
        info = shaped_info_bits(M, amp_pmf, sc.K, rng)
        cw, _ = sc.encode(info)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = chan.transmit(cw[sc.tx_mask], sigma, rng)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=cfg.W, max_iter=cfg.max_iter, alpha=cfg.alpha)
        err = int((sc.extract_info(hard) != info).sum())
        be += err; bits += info.size; fe += int(err > 0)
    return be, bits, fe, hi - lo


R._worker = _worker_pas         # route ALL R.eval_* through the shaped-QAM channel


def eval_codes(cfg, assign, M, nu, snr, frame_seed, n_frames, pool=None,
               frame_chunks=1, amp_pmf=None):
    """Evaluate a single (assign, nu) construction over CRN frames; returns metric.

    Mirrors R.eval_assignments but carries the PAS payload (assign,M,nu,pmf)."""
    from dataclasses import asdict
    cfg_d = asdict(cfg)
    pmf_list = amp_pmf.tolist() if amp_pmf is not None else None
    payload = (np.asarray(assign, dtype=np.int64), M, float(nu), pmf_list)
    bounds = np.linspace(0, n_frames, frame_chunks + 1).astype(int)
    tasks = []
    for ci in range(frame_chunks):
        a, b = int(bounds[ci]), int(bounds[ci + 1])
        if b > a:
            tasks.append((cfg_d, payload, snr, frame_seed, n_frames, a, b))
    raw = pool.map(_worker_pas, tasks) if pool is not None else [_worker_pas(t) for t in tasks]
    be = bits = fe = nf = 0
    for (b_, bi_, f_, n_) in raw:
        be += b_; bits += bi_; fe += f_; nf += n_
    return {"ber": be / max(bits, 1), "fer": fe / max(nf, 1), "bit_err": be,
            "n_info": bits, "frame_err": fe, "n_frames": nf}


def eval_batch_codes(cfg, items, snr, frame_seed, n_frames, pool=None):
    """Batch of (assign, nu) pairs (or (assign,nu,pmf)) on the SAME CRN frames:
    one task per item.  `items` = list of (assign, M, nu) or (assign, M, nu, pmf)."""
    from dataclasses import asdict
    cfg_d = asdict(cfg)
    tasks = []
    for it in items:
        if len(it) == 4:
            assign, M, nu, pmf = it
            pmf_list = None if pmf is None else np.asarray(pmf).tolist()
        else:
            assign, M, nu = it; pmf_list = None
        payload = (np.asarray(assign, dtype=np.int64), M, float(nu), pmf_list)
        tasks.append((cfg_d, payload, snr, frame_seed, n_frames, 0, n_frames))
    raw = pool.map(_worker_pas, tasks) if pool is not None else [_worker_pas(t) for t in tasks]
    return [{"ber": be / max(bits, 1), "fer": fe / max(nf, 1), "bit_err": be,
             "frame_err": fe, "n_frames": nf} for (be, bits, fe, nf) in raw]


# ----------------------------------------------------------------------------- #
#  PART 1 -- shaping gain (information-theoretic)
# ----------------------------------------------------------------------------- #
def _p1_cell(args):
    """R_BMD over the nu grid at one (M, snr); returns (M, snr, best_nu, best_rbmd,
    uniform_rbmd, mi_uniform)."""
    M, snr = args
    uni = pas.r_bmd(M, 0.0, snr, n_sym=P1_NSYM, seed=11)
    best_nu, best = 0.0, uni
    for nu in P1_NUS:
        d = pas.r_bmd(M, nu, snr, n_sym=P1_NSYM, seed=11)
        if d["r_bmd"] > best["r_bmd"]:
            best, best_nu = d, nu
    return (M, snr, best_nu, best["r_bmd"], uni["r_bmd"], uni["mi"], best["mi"])


def run_part1(pool):
    print("\n########## PART 1: shaping gain (R_BMD, MB nu*) ##########", flush=True)
    t0 = time.time()
    jobs = [(M, snr) for M in P1_ORDERS for snr in P1_ESN0]
    res = pool.map(_p1_cell, jobs) if pool is not None else [_p1_cell(j) for j in jobs]
    out = {"orders": P1_ORDERS, "esn0": P1_ESN0, "curves": {}, "dbgain": {}}
    for M in P1_ORDERS:
        rows = [r for r in res if r[0] == M]
        rows.sort(key=lambda r: r[1])
        out["curves"][str(M)] = {
            "esn0": [r[1] for r in rows],
            "nu_star": [round(r[2], 4) for r in rows],
            "rbmd_shaped": [round(r[3], 4) for r in rows],
            "rbmd_uniform": [round(r[4], 4) for r in rows],
            "mi_uniform": [round(r[5], 4) for r in rows],
            "mi_shaped": [round(r[6], 4) for r in rows],
        }
        print(f"  M={M}: nu*(by snr)={out['curves'][str(M)]['nu_star']}", flush=True)
    # dB gain at matched R_BMD targets (interp on the uniform & shaped curves)
    for M, targets in P1_TARGETS.items():
        cu = out["curves"][str(M)]
        out["dbgain"][str(M)] = []
        for tgt in targets:
            su = _interp_snr(cu["esn0"], cu["rbmd_uniform"], tgt)
            ss = _interp_snr(cu["esn0"], cu["rbmd_shaped"], tgt)
            gain = (su - ss) if (su is not None and ss is not None) else None
            out["dbgain"][str(M)].append({"target": tgt, "snr_uniform": su,
                                          "snr_shaped": ss, "gain_db": gain})
            if gain is not None:
                print(f"  M={M} R_BMD={tgt}b: uniform {su:.2f}dB -> shaped {ss:.2f}dB "
                      f"= {gain:+.2f} dB", flush=True)
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(OP.route("results_pas_part1.json"), "w"))
    print(f"  [PART 1 done in {out['secs']:.0f}s -> results_pas_part1.json]", flush=True)
    return out


def _interp_snr(xs, ys, target):
    """Smallest Es/N0 where the (monotone-ish) rate curve crosses `target` (linear interp)."""
    xs = list(xs); ys = list(ys)
    for i in range(1, len(ys)):
        if (ys[i - 1] - target) * (ys[i] - target) <= 0 and ys[i] != ys[i - 1]:
            t = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return round(xs[i - 1] + t * (xs[i] - xs[i - 1]), 3)
    return None


# ----------------------------------------------------------------------------- #
#  PART 2 -- shaping policy (Gaussian on nu, reparam through softplus to nu>=0)
# ----------------------------------------------------------------------------- #
def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class NuPolicy:
    """Gaussian policy over a latent u; nu = softplus(u) >= 0.  REINFORCE on u.

    Keeps a fixed exploration std on u so the joint search keeps probing shaping.
    The mean latent mu is the learned parameter (Adam ascent on -log10 BER)."""

    def __init__(self, init_nu=0.08, lr=0.05, std=0.5, seed=0, nu_cap=0.4):
        # invert softplus to set the initial latent mean
        self.mu = float(np.log(np.expm1(max(init_nu, 1e-3))))
        self.std = std
        self.nu_cap = nu_cap
        self.opt = R.Adam((1,), lr=lr)
        self.rng = np.random.default_rng(seed)
        self._mu_arr = np.array([self.mu])

    def nu(self):
        return float(min(self.nu_cap, _softplus(self.mu)))

    def sample(self):
        u = self.mu + self.std * self.rng.standard_normal()
        nu = float(min(self.nu_cap, _softplus(u)))
        return nu, u

    def grad_logp(self, u):
        """d log N(u; mu, std^2)/d mu = (u-mu)/std^2."""
        return (u - self.mu) / (self.std ** 2)

    def update(self, samples_adv):
        """samples_adv: list of (u, advantage).  Ascent on E[adv]."""
        g = np.array([np.mean([adv * self.grad_logp(u) for (u, adv) in samples_adv])])
        self._mu_arr = self.opt.step(self._mu_arr, g)
        self.mu = float(self._mu_arr[0])


# ----------------------------------------------------------------------------- #
#  joint-RL training state + one step (also used by the smoke test)
# ----------------------------------------------------------------------------- #
class JointTrainState:
    def __init__(self, cfg, M, rate, snr):
        self.cfg = cfg
        self.M = M
        self.rate = rate
        self.snr = snr
        self.best = {"ber": np.inf, "fer": np.inf, "assign": None, "nu": None}
        self.hist = {"evals": [], "mean_reward": [], "best_val_ber": [], "nu_mean": []}
        self.seen = 0


def joint_step(state, con_policy, nu_policy, batch, frames, pool, step,
               val_frames=VAL_FRAMES, val_seed=99, base_seed=7000):
    """One REINFORCE step jointly updating construction + shaping.  Shared advantage
    (a single coded BER per (assign,nu) sample) drives BOTH policies (the standard
    way to credit a joint action by its joint reward)."""
    feature = isinstance(con_policy, R.FeaturePolicy)
    fs = base_seed + step
    traces, assigns, ps, us, items = [], [], [], [], []
    for _ in range(batch):
        if feature:
            a, tr = con_policy.rollout(); traces.append(tr)
        else:
            a, p = con_policy.sample(); ps.append(p)
        nu, u = nu_policy.sample()
        assigns.append(a); us.append(u)
        items.append((a, state.M, nu))
    ms = eval_batch_codes(state.cfg, items, state.snr, fs, frames, pool=pool)
    state.seen += batch
    rew = np.array([-np.log10(max(m["ber"], BER_FLOOR)) for m in ms])
    adv = rew - rew.mean()
    if adv.std() > 1e-9:
        adv = adv / (adv.std() + 1e-9)
    if feature:
        con_policy.update(list(zip(traces, adv)))
    else:
        con_policy.update([(assigns[i], ps[i], adv[i]) for i in range(batch)])
    nu_policy.update([(us[i], adv[i]) for i in range(batch)])
    # champion = best batch BER, refined on a validation bank at the policy-mean nu
    bi = int(np.argmin([m["ber"] for m in ms]))
    cand_nu = nu_policy.nu()
    vm = eval_codes(state.cfg, assigns[bi], state.M, cand_nu, state.snr, val_seed,
                    val_frames, pool=pool, frame_chunks=(R.n_workers() if pool else 1))
    if vm["ber"] < state.best["ber"]:
        state.best = {"ber": vm["ber"], "fer": vm["fer"],
                      "assign": np.asarray(assigns[bi]).copy(), "nu": cand_nu}
    state.hist["evals"].append(state.seen)
    state.hist["mean_reward"].append(float(rew.mean()))
    state.hist["best_val_ber"].append(state.best["ber"])
    state.hist["nu_mean"].append(cand_nu)
    return ms


# ----------------------------------------------------------------------------- #
#  operating-point selection (BER-centric, like the PAM4 run)
# ----------------------------------------------------------------------------- #
def pick_snr(cfg, M, rate, nu0, pool):
    """Bisection on info Eb/N0 to the median random-construction BER ~ TARGET_BER,
    with shaped(nu0) input (so the operating point is set in the shaped regime)."""
    lo, hi = SNR_BRACKET[rate]
    rng = np.random.default_rng(0)
    _, _, E = R.edge_meta(cfg); C = R.n_components(cfg)
    assigns = [rng.integers(0, C, size=E) for _ in range(PROBE_N)]

    def med_ber(snr):
        items = [(a, M, nu0) for a in assigns]
        ms = eval_batch_codes(cfg, items, snr, 1, PROBE_FRAMES, pool=pool)
        return float(np.median([m["ber"] for m in ms]))

    for _ in range(BISECT_ITERS):
        mid = round((lo + hi) / 2.0, 2)
        if med_ber(mid) < TARGET_BER:
            hi = mid
        else:
            lo = mid
    snr = round((lo + hi) / 2.0, 2)
    return snr, med_ber(snr)


# ----------------------------------------------------------------------------- #
#  one joint-RL cell
# ----------------------------------------------------------------------------- #
def ber_curve_codes(cfg, assign, M, nu, snrs, frames, pool, chunks=None, amp_pmf=None):
    chunks = chunks or R.n_workers()
    xs, bers, fers = [], [], []
    for snr, nf in zip(snrs, frames):
        m = eval_codes(cfg, assign, M, nu, snr, 2025, nf, pool=pool,
                       frame_chunks=chunks, amp_pmf=amp_pmf)
        xs.append(round(snr, 2)); bers.append(m["ber"]); fers.append(m["fer"])
    return {"x": xs, "y": bers, "fer": fers}


def run_cell(rate, w, M, Z, pool, p1):
    t0 = time.time()
    cfg = cfg_for(M, rate, Z, w, OPT_L)
    sc0 = cfg.build()
    _, _, E = R.edge_meta(cfg)
    # nu* from PART 1 at a representative Es/N0 for this M (use the matched-rate one)
    nu_star = _nu_star_for_cell(M, p1)
    snr, med = pick_snr(cfg, M, rate, nu_star, pool)
    print(f"\n=== JOINT R={rate} w={w} M={M} Z={Z} E={E} L={OPT_L} "
          f"train@{snr}dB (med random BER {med:.2e}) nu*_P1={nu_star:.3f} ===", flush=True)

    # --- RL-joint (FeaturePolicy construction + NuPolicy shaping) ---
    con = R.FeaturePolicy(cfg, lr=0.15, ent=0.02, seed=0)
    nup = NuPolicy(init_nu=max(nu_star, 0.04), lr=0.06, std=0.5, seed=0)
    state = JointTrainState(cfg, M, rate, snr)
    for step in range(1, OPT_STEPS + 1):
        joint_step(state, con, nup, OPT_BATCH, OPT_FRAMES, pool, step)
        if step % 4 == 0 or step == 1:
            print(f"  [JOINT] step {step:3d} meanR={state.hist['mean_reward'][-1]:+.2f} "
                  f"bestVAL_BER={state.best['ber']:.2e} nu={state.best['nu']:.3f} "
                  f"(policy nu={nup.nu():.3f})", flush=True)

    # --- baselines at equal eval budget ---
    rng = np.random.default_rng(1)
    h_rnd, best_rnd = train_shaped_random(cfg, M, nu_star, snr, OPT_STEPS, OPT_BATCH,
                                          OPT_FRAMES, pool, seed=11)

    rr = R.round_robin_assign(cfg)
    champions = {
        "rl_joint": (state.best["assign"], state.best["nu"]),
        "shaped_random": (best_rnd["assign"], nu_star),
        "shaped_rr": (rr, nu_star),
        "uniform_rr": (rr, 0.0),
    }
    snrs = [snr + d for d in BER_OFFSETS]
    finals, stats, curves = {}, {}, {}
    for name, (a, nu) in champions.items():
        a = np.asarray(a, dtype=np.int64)
        pmf = pas.mb_pmf(M, nu)
        m = eval_codes(cfg, a, M, nu, snr, 12345, FINAL_FRAMES, pool=pool,
                       frame_chunks=R.n_workers())
        net = net_rate_operational(sc0, M, pmf)
        finals[name] = {"ber": m["ber"], "fer": m["fer"], "nu": float(nu),
                        "net_rate": round(net, 4),
                        "h_frac": round(axis_entropy_frac(M, pmf), 4)}
        st = R.construction_stats(cfg, a)
        try:
            g = R.girth(cfg.build(assign=a), max_g=10, n_seeds=40)
        except Exception:
            g = None
        stats[name] = {"n4": st["n4"], "comp_load": st["comp_load"], "girth": g}
        curves[name] = ber_curve_codes(cfg, a, M, nu, snrs, BER_FRAMES, pool, amp_pmf=pmf)
        print(f"  {name:14s} BER={m['ber']:.3e} FER={m['fer']:.3f} nu={nu:.3f} "
              f"net={net:.3f}b/cu n4={st['n4']} girth={g}", flush=True)
    print(f"  [cell done in {time.time()-t0:.0f}s]", flush=True)
    return {"rate": rate, "w": w, "M": M, "Z": Z, "mp": RATE_MP[rate], "E": int(E),
            "train_snr": snr, "nu_star_p1": nu_star, "finals": finals, "stats": stats,
            "curves": curves,
            "hist": {"rl_joint": state.hist, "shaped_random": h_rnd},
            "theta_rl": con.theta.tolist(),
            "champions": {k: {"assign": _arr(v[0]), "nu": float(v[1])}
                          for k, v in champions.items()}}


def train_shaped_random(cfg, M, nu, snr, steps, batch, frames, pool, seed=11):
    """Random-construction baseline at fixed shaped nu (equal eval budget)."""
    _, _, E = R.edge_meta(cfg); C = R.n_components(cfg)
    rng = np.random.default_rng(seed)
    hist = {"evals": [], "best_val_ber": []}
    best = {"ber": np.inf, "fer": np.inf, "assign": None}; seen = 0
    for step in range(1, steps + 1):
        fs = 9000 + step
        assigns = [rng.integers(0, C, size=E) for _ in range(batch)]
        items = [(a, M, nu) for a in assigns]
        ms = eval_batch_codes(cfg, items, snr, fs, frames, pool=pool); seen += batch
        bers = np.array([m["ber"] for m in ms])
        cand = assigns[int(np.argmin(bers))]
        vm = eval_codes(cfg, cand, M, nu, snr, 99, VAL_FRAMES, pool=pool,
                        frame_chunks=(R.n_workers() if pool else 1))
        if vm["ber"] < best["ber"]:
            best = {"ber": vm["ber"], "fer": vm["fer"], "assign": np.asarray(cand).copy()}
        hist["evals"].append(seen); hist["best_val_ber"].append(best["ber"])
    return hist, best


def _nu_star_for_cell(M, p1):
    """Pick a representative nu* for this M from PART 1 (median of nu* over the grid,
    excluding the trivial high-SNR tail where nu*->0)."""
    if p1 is None or str(M) not in p1.get("curves", {}):
        return {16: 0.06, 64: 0.04, 256: 0.03, 4: 0.0}.get(M, 0.0)
    nus = [n for n in p1["curves"][str(M)]["nu_star"] if n > 0]
    return float(round(np.median(nus), 4)) if nus else 0.0


def run_part2(pool, p1):
    print("\n########## PART 2: JOINT RL (construction + shaping) ##########", flush=True)
    out = {"cells": {}, "config": {"L": OPT_L, "ber_offsets": BER_OFFSETS,
                                   "opt_steps": OPT_STEPS, "opt_batch": OPT_BATCH,
                                   "opt_frames": OPT_FRAMES, "cells": P2_CELLS}}
    fn = "results_pas_part2.json"
    for (rate, w, M, Z) in P2_CELLS:
        key = f"R{rate}_w{w}_M{M}_Z{Z}"
        try:
            out["cells"][key] = run_cell(rate, w, M, Z, pool, p1)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! cell {key} failed: {e}", flush=True)
        json.dump(out, open(OP.route(fn), "w"))
    print(f"\n[PART 2 done -> {fn}]", flush=True)
    return out


# ----------------------------------------------------------------------------- #
#  driver
# ----------------------------------------------------------------------------- #
def run():
    t0 = time.time()
    print(f"PAS RL run: {R.n_workers()} workers, BG{BG} Z grid, numpy-only", flush=True)
    with mp.Pool(R.n_workers()) as pool:
        p1 = run_part1(pool)
        run_part2(pool, p1)
    print(f"\nALL DONE in {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
#  plotting
# --------------------------------------------------------------------------- #
LAB = {"rl_joint": "RL-joint (constr+shaping) [ours]", "shaped_random": "shaped + random constr",
       "shaped_rr": "shaped(nu*) + round-robin", "uniform_rr": "uniform + round-robin"}
COL = {"rl_joint": "#d62728", "shaped_random": "#1f77b4", "shaped_rr": "#2ca02c",
       "uniform_rr": "#7f7f7f"}
MCOL = {4: "#7f7f7f", 16: "#1f77b4", 64: "#2ca02c", 256: "#d62728"}


def make_plots():
    made = []
    # PART 1 figures
    try:
        p1 = json.load(open(OP.route("results_pas_part1.json")))
    except FileNotFoundError:
        p1 = None
    if p1:
        # Fig: R_BMD vs Es/N0, uniform vs shaped, one panel per M (all on one fig per M)
        for M in p1["orders"]:
            cu = p1["curves"][str(M)]
            ser = [{"x": cu["esn0"], "y": cu["rbmd_uniform"], "label": f"uniform (m={qam.bits_per_symbol(M)})",
                    "color": "#7f7f7f"},
                   {"x": cu["esn0"], "y": cu["rbmd_shaped"], "label": "shaped (MB nu*)",
                    "color": MCOL.get(M, "#d62728")}]
            plotting.linear(ser, xlabel="Es/N0 [dB]", ylabel="R_BMD [bits/complex sym]",
                            title=f"PAS shaping gain: R_BMD uniform vs shaped, {M}-QAM",
                            path=f"exp_pas_rbmd_M{M}.svg")
            made.append(f"exp_pas_rbmd_M{M}.svg")
        # Fig: shaping gain (dB) vs M at the headline targets, as a combined curve
        gain_ser = []
        for M in [16, 64, 256]:
            dg = p1["dbgain"].get(str(M), [])
            xs = [d["target"] for d in dg if d.get("gain_db") is not None]
            ys = [d["gain_db"] for d in dg if d.get("gain_db") is not None]
            if xs:
                gain_ser.append({"x": xs, "y": ys, "label": f"{M}-QAM", "color": MCOL[M]})
        if gain_ser:
            plotting.linear(gain_ser, xlabel="R_BMD target [bits/complex sym]",
                            ylabel="shaping gain [dB]",
                            title="PAS shaping gain vs rate (SNR saving at matched R_BMD)",
                            path="exp_pas_shaping_gain_db.svg")
            made.append("exp_pas_shaping_gain_db.svg")
        # nu* vs Es/N0
        nu_ser = [{"x": p1["curves"][str(M)]["esn0"], "y": p1["curves"][str(M)]["nu_star"],
                   "label": f"{M}-QAM", "color": MCOL.get(M, "#333")}
                  for M in p1["orders"] if M != 4]
        plotting.linear(nu_ser, xlabel="Es/N0 [dB]", ylabel="optimal MB nu*",
                        title="PAS: optimal Maxwell-Boltzmann nu* vs SNR",
                        path="exp_pas_nustar.svg")
        made.append("exp_pas_nustar.svg")

    # PART 2 figures
    try:
        p2 = json.load(open(OP.route("results_pas_part2.json")))
    except FileNotFoundError:
        p2 = None
    if p2:
        for key, c in p2["cells"].items():
            ser = [{"x": c["curves"][n]["x"], "y": c["curves"][n]["y"], "label": LAB[n], "color": COL[n]}
                   for n in ["rl_joint", "shaped_rr", "shaped_random", "uniform_rr"]
                   if n in c["curves"]]
            plotting.semilogy(ser, xlabel="info Eb/N0 [dB]", ylabel="info BER",
                              title=f"Coded PAS BER: {c['M']}-QAM R={c['rate']} w={c['w']} Z={c['Z']}",
                              path=f"exp_pas_ber_{key}.svg")
            made.append(f"exp_pas_ber_{key}.svg")
            h = c.get("hist", {})
            ls = []
            if "rl_joint" in h:
                ls.append({"x": h["rl_joint"]["evals"],
                           "y": [max(b, BER_FLOOR) for b in h["rl_joint"]["best_val_ber"]],
                           "label": LAB["rl_joint"], "color": COL["rl_joint"]})
            if "shaped_random" in h:
                ls.append({"x": h["shaped_random"]["evals"],
                           "y": [max(b, BER_FLOOR) for b in h["shaped_random"]["best_val_ber"]],
                           "label": LAB["shaped_random"], "color": COL["shaped_random"]})
            if ls:
                plotting.semilogy(ls, xlabel="# joint evaluations", ylabel="best val BER",
                                  title=f"Joint-RL learning curve: {c['M']}-QAM R={c['rate']} w={c['w']}",
                                  path=f"exp_pas_learn_{key}.svg")
                made.append(f"exp_pas_learn_{key}.svg")
            # nu trajectory
            if "rl_joint" in h and h["rl_joint"].get("nu_mean"):
                plotting.linear([{"x": h["rl_joint"]["evals"], "y": h["rl_joint"]["nu_mean"],
                                  "label": "policy nu", "color": COL["rl_joint"]}],
                                xlabel="# joint evaluations", ylabel="shaping nu",
                                title=f"Joint-RL shaping trajectory: {c['M']}-QAM R={c['rate']} w={c['w']}",
                                path=f"exp_pas_nu_{key}.svg")
                made.append(f"exp_pas_nu_{key}.svg")
    _export_tables(p1, p2)
    print("wrote: " + ", ".join(made) + ", results_pas_*.csv")


def _export_tables(p1, p2):
    import csv
    if p1:
        with open(OP.route("results_pas_part1.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["M", "esn0_db", "nu_star", "rbmd_uniform", "rbmd_shaped",
                         "mi_uniform", "mi_shaped"])
            for M in p1["orders"]:
                cu = p1["curves"][str(M)]
                for i in range(len(cu["esn0"])):
                    wr.writerow([M, cu["esn0"][i], cu["nu_star"][i], cu["rbmd_uniform"][i],
                                 cu["rbmd_shaped"][i], cu["mi_uniform"][i], cu["mi_shaped"][i]])
        with open(OP.route("results_pas_gain_db.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["M", "rbmd_target", "snr_uniform_db", "snr_shaped_db", "gain_db"])
            for M, rows in p1.get("dbgain", {}).items():
                for d in rows:
                    wr.writerow([M, d["target"], d["snr_uniform"], d["snr_shaped"], d["gain_db"]])
    if p2:
        with open(OP.route("results_pas_part2_finals.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["cell", "rate", "w", "M", "Z", "E", "train_snr", "method",
                         "ber", "fer", "nu", "net_rate", "n4", "girth"])
            for key, c in sorted(p2["cells"].items()):
                for m, fv in c["finals"].items():
                    st = c["stats"].get(m, {})
                    wr.writerow([key, c["rate"], c["w"], c["M"], c["Z"], c["E"], c["train_snr"],
                                 m, f"{fv['ber']:.4e}", f"{fv['fer']:.4f}", fv["nu"],
                                 fv.get("net_rate", ""), st.get("n4", ""), st.get("girth", "")])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "part1":
        with mp.Pool(R.n_workers()) as pool:
            run_part1(pool)
    elif cmd == "part2":
        with mp.Pool(R.n_workers()) as pool:
            p1 = json.load(open(OP.route("results_pas_part1.json"))) if __import__("os").path.exists(OP.route("results_pas_part1.json")) else None
            run_part2(pool, p1)
    elif cmd == "plot":
        make_plots()
    else:
        print(__doc__)
