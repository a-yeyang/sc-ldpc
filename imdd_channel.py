"""Bandwidth-limited IM/DD (intensity-modulation / direct-detection) optical link.

A parameterized OptiCommPy chain for SHORT-REACH high-baud IM/DD, built to
*isolate the bandwidth-limitation damage* of a sub-Nyquist photodiode and ask
whether the post-equalizer residual still carries exploitable structure
(colored noise / unequal level-or-bit reliability / residual ISI) that an
SC-LDPC construction could later exploit.  PAM4 (M=4) and PAM6 (6 levels) are
treated as two groups for side-by-side comparison.

Physical chain (single real drive -> intensity)
-----------------------------------------------
  bits / level indices
    -> PAM amplitudes (PAM4 Gray 2-bit; PAM6 = 6 equally-spaced levels)   [this file]
    -> upsample x SpS + RRC pulse shaping                 dsp.core.{upsample,pulseShape,firFilter}
    -> AWG model: resample drive to 120 GSa/s and back    dsp.core.resample
       (DAC rate / analog-bandwidth limit on the drive; SECONDARY impairment)
    -> bias to unipolar, scale, drive MZM @ quadrature    models.devices.mzm  (Vb=-Vpi/2)
    -> optical intensity field  E = sqrt(Pin) * cos(.)
    -> SSMF split-step Fourier, O-band (D~=0) short L      models.channels.ssfm
    -> set received optical power ROP (scale field power)  [this file]
    -> DIRECT-DETECTION photodiode, square-law i ~ |E|^2,  models.devices.photodiode
       BANDWIDTH B (30/40 GHz) + thermal + shot noise      (ideal=False, bandwidthLimitation=True)
       *** B < Rs/2 = 40 GHz  =>  SEVERELY sub-Nyquist, strong ISI ***
    -> resample photocurrent to 2 SpS                      dsp.core.resample
    -> remove DC, normalize                                [this file]
    -> data-aided MMSE-FFE (and optional DFE)              [this file]
    -> equalized symbols yhat (1 SpS), residual e=yhat-x

The DOMINANT impairment is the PD bandwidth limit (sub-Nyquist).  CD is made
negligible by O-band D~=0 so the study isolates the bandwidth damage; one C-band
point (D~=16) can be requested to show CD fading appears.

The module exposes one function -- run_link(cfg) -> dict -- returning the
TX symbols, equalized symbols, residual, per-level / per-bit soft metrics
(LLRs ready for the SC-LDPC bridge), and effective-SNR / EVM / SER numbers.
imdd_structure.py drives it across (ROP, PD-B, L) x {PAM4, PAM6}.
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optic.utils import parameters
from optic.models.devices import mzm, photodiode, dBm2W
from optic.models.channels import ssfm
from optic.dsp.core import pulseShape, firFilter, upsample, resample, signalPower, pnorm

# --------------------------------------------------------------------------- #
#  PAM level sets and labelings
# --------------------------------------------------------------------------- #
# PAM4: standard 2-bit Gray, value v = 2*b_hi + b_lo  (same convention as pam4_rrc.py)
#   v=00 -> -3 ; v=01 -> -1 ; v=11 -> +1 ; v=10 -> +3      (ascending levels are Gray-adjacent)
PAM4_LEVELS_BIPOLAR = np.array([-3.0, -1.0, 1.0, 3.0])      # ASCENDING physical levels
# Gray bit labels for each ASCENDING level index 0..3 (MSB,LSB):
#   level -3 -> 00, -1 -> 01, +1 -> 11, +3 -> 10
PAM4_BITS_BY_LEVEL = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int64)

# PAM6: 6 equally-spaced real levels.  Non-power-of-2 (log2 6 ~= 2.585 b/sym), so
# OptiCommPy's square M-PAM Gray map does not apply.  We build the levels here and
# work at the SYMBOL (per-level) granularity for the structure study; a "natural"
# index labeling (0..5 -> ascending level) is attached for bookkeeping only.
PAM6_LEVELS_BIPOLAR = np.linspace(-5.0, 5.0, 6)             # [-5,-3,-1,1,3,5]


def pam_levels(n_levels: int):
    """Unit-mean-square (Es=1) ascending PAM levels for n_levels in {4,6}."""
    if n_levels == 4:
        lv = PAM4_LEVELS_BIPOLAR.copy()
    elif n_levels == 6:
        lv = PAM6_LEVELS_BIPOLAR.copy()
    else:
        # generic equally spaced
        lv = (2 * np.arange(n_levels) - (n_levels - 1)).astype(float)
    lv = lv / np.sqrt(np.mean(lv ** 2))                     # normalize to Es=1
    return lv


# --------------------------------------------------------------------------- #
#  config
# --------------------------------------------------------------------------- #
@dataclass
class LinkConfig:
    n_levels: int = 4          # 4 (PAM4) or 6 (PAM6)
    nsym: int = 4000           # transmitted PAM symbols
    Rs: float = 80e9           # symbol rate [Baud]
    SpS: int = 3               # samples/symbol for the optical sim (Fs = SpS*Rs)
    rolloff: float = 0.1       # RRC rolloff
    ntaps_rrc: int = 401       # RRC FIR taps
    awg_fs: float = 120e9      # AWG DAC rate [Sa/s] (drive resampled to this & back)
    awg_model: bool = True     # apply AWG DAC-rate/bandwidth limit on the drive
    # MZM / optical
    Vpi: float = 2.0
    mzm_swing_frac: float = 0.9    # peak drive swing as fraction of Vpi about quadrature
    Pin_dBm: float = 6.0       # CW laser power into MZM [dBm] (sets pre-fiber optical power)
    # fiber (O-band, near-transparent to isolate the bandwidth limit)
    L_km: float = 2.0
    alpha: float = 0.3         # O-band loss [dB/km]
    D: float = 0.0             # dispersion [ps/nm/km]  (~0 in O-band near ZDW)
    gamma: float = 1.3         # nonlinearity [1/W/km]  (negligible at IM/DD short reach)
    Fc: float = 229e12         # O-band optical carrier ~1310 nm
    hz: float = 0.5            # SSFM step [km]
    # photodiode (direct detection)
    pd_B: float = 30e9         # PD electrical bandwidth [Hz]  (30 or 40 GHz)
    ROP_dBm: float = -8.0      # received optical power into the PD [dBm] -> sets SNR
    pd_R: float = 1.0          # responsivity [A/W]
    pd_ideal: bool = False     # False -> thermal+shot noise + bandwidth limit
    # equalizer
    eq_sps: int = 2            # equalizer operates at this SpS (resample PD output to it)
    ffe_taps: int = 31         # FFE tap count (fractionally spaced if eq_sps>1)
    dfe_taps: int = 0          # DFE feedback taps (0 = FFE only); decision-directed
    train_frac: float = 0.5    # fraction of symbols used as the data-aided training set
    # misc
    seed: int = 0


# --------------------------------------------------------------------------- #
#  TX: bits/levels -> PAM symbols
# --------------------------------------------------------------------------- #
def make_tx_symbols(cfg: LinkConfig, rng):
    """Random PAM symbols.  Returns (levels[Es=1], lvl_idx[0..n-1], bits_or_None).

    PAM4 returns the 2-bit Gray labels (interleaved [b_hi,b_lo,...]); PAM6 returns
    None for bits (symbol-level study)."""
    lv = pam_levels(cfg.n_levels)
    idx = rng.integers(0, cfg.n_levels, size=cfg.nsym)
    syms = lv[idx]
    if cfg.n_levels == 4:
        bits = PAM4_BITS_BY_LEVEL[idx].reshape(-1)         # [2*nsym], interleaved hi,lo
    else:
        bits = None
    return syms, idx, bits, lv


# --------------------------------------------------------------------------- #
#  forward optical chain: PAM symbols -> photocurrent samples (eq_sps per symbol)
# --------------------------------------------------------------------------- #
def forward_channel(cfg: LinkConfig, syms: np.ndarray):
    """Push real PAM symbols through the IM/DD optical chain; return the
    photocurrent resampled to cfg.eq_sps samples/symbol, plus diagnostics."""
    SpS, Rs = cfg.SpS, cfg.Rs
    Fs = SpS * Rs

    # ---- RRC pulse shaping (real baseband drive) ----
    pp = parameters()
    pp.pulseType = "rrc"; pp.SpS = SpS
    pp.nFilterTaps = cfg.ntaps_rrc; pp.rollOff = cfg.rolloff
    pulse = pulseShape(pp)
    pulse = pulse / np.sqrt(np.sum(pulse ** 2))            # unit-energy RRC
    drive = firFilter(pulse, upsample(syms, SpS)).real     # real shaped drive, ~zero mean

    # ---- AWG model: limit drive to DAC rate / analog bandwidth ----
    # Resample the shaped drive down to the AWG DAC rate (120 GSa/s) and back up to
    # Fs.  resample() applies an anti-alias lowpass at each stage, so this imposes
    # the AWG's ~ (awg_fs/2) analog bandwidth on the drive (a real, mild impairment
    # well above the dominant PD limit).  Secondary effect as specified.
    if cfg.awg_model and cfg.awg_fs < Fs:
        rp = parameters(); rp.inFs = Fs; rp.outFs = cfg.awg_fs
        d2 = resample(drive, rp)
        rp2 = parameters(); rp2.inFs = cfg.awg_fs; rp2.outFs = Fs
        drive = resample(d2, rp2)
        # resample can change length slightly; trim/pad to SpS*nsym
        want = SpS * syms.size
        if drive.size >= want:
            drive = drive[:want]
        else:
            drive = np.concatenate([drive, np.zeros(want - drive.size)])

    # ---- normalize drive to unit peak, then map to a unipolar MZM swing ----
    # Bias MZM at quadrature (Vb = -Vpi/2): Ao = Ai*cos(0.5/Vpi*(u+Vb)*pi).
    # At u=0 -> cos(-pi/4); we swing u in +/- (swing_frac*Vpi/2) about quadrature so
    # the cos() output (optical field amplitude) tracks the PAM drive ~ linearly.
    pk = np.max(np.abs(drive)) + 1e-12
    u = (drive / pk) * (cfg.mzm_swing_frac * cfg.Vpi / 2.0)
    mp = parameters(); mp.Vpi = cfg.Vpi; mp.Vb = -cfg.Vpi / 2.0
    Ai = np.sqrt(dBm2W(cfg.Pin_dBm))
    Eopt = mzm(Ai, u, mp)                                  # optical field (real here)
    # pin pre-fiber optical power to Pin exactly
    Eopt = Eopt * np.sqrt(dBm2W(cfg.Pin_dBm) / signalPower(Eopt))

    # ---- SSMF: split-step Fourier, O-band short fiber ----
    pc = parameters()
    pc.Ltotal = cfg.L_km; pc.Lspan = cfg.L_km; pc.hz = cfg.hz
    pc.alpha = cfg.alpha; pc.D = cfg.D; pc.gamma = cfg.gamma
    pc.Fc = cfg.Fc; pc.Fs = Fs; pc.amp = "ideal"; pc.prgsBar = False
    Ech = ssfm(Eopt.astype(np.complex128), pc)

    # ---- set received optical power ROP into the PD (unamplified IM/DD) ----
    Ech = Ech * np.sqrt(dBm2W(cfg.ROP_dBm) / signalPower(Ech))

    # ---- DIRECT DETECTION: square-law PD with bandwidth limit + noise ----
    pd = parameters()
    pd.R = cfg.pd_R; pd.B = cfg.pd_B; pd.Fs = Fs
    pd.ideal = cfg.pd_ideal
    pd.shotNoise = True; pd.thermalNoise = True
    pd.bandwidthLimitation = True
    pd.seed = cfg.seed + 7
    ipd = photodiode(Ech, pd)                              # real photocurrent [A]

    # ---- resample photocurrent to the equalizer rate (eq_sps per symbol) ----
    if cfg.eq_sps != SpS:
        rp = parameters(); rp.inFs = Fs; rp.outFs = cfg.eq_sps * Rs
        irx = resample(ipd, rp)
    else:
        irx = ipd
    want = cfg.eq_sps * syms.size
    if irx.size >= want:
        irx = irx[:want]
    else:
        irx = np.concatenate([irx, np.full(want - irx.size, irx.mean())])

    diag = dict(Fs=Fs, drive_pk=float(pk), ipd_mean=float(ipd.mean()),
                ipd_std=float(ipd.std()))
    return irx, diag


# --------------------------------------------------------------------------- #
#  data-aided MMSE-FFE (+ optional DFE)
# --------------------------------------------------------------------------- #
def _build_ffe_matrix(rx: np.ndarray, eq_sps: int, n_ff: int, nsym: int):
    """Stack length-n_ff fractionally-spaced input windows, one row per symbol.
    Window for symbol k is centred at sample k*eq_sps."""
    half = n_ff // 2
    base = np.arange(nsym) * eq_sps                        # centre sample per symbol
    # pad rx so windowed indexing never goes out of range
    pad = half + eq_sps
    rxp = np.concatenate([np.full(pad, rx[0]), rx, np.full(pad, rx[-1])])
    cols = base[:, None] + pad + (np.arange(n_ff) - half)[None, :]
    return rxp[cols]                                       # [nsym, n_ff]


def equalize(cfg: LinkConfig, rx: np.ndarray, syms: np.ndarray, lv: np.ndarray):
    """Data-aided MMSE linear FFE (and optional decision-directed DFE).

    Trains the equalizer on the first `train_frac` symbols (known), applies to all.
    Returns dict with yhat (equalized, 1 SpS, scaled to Es=1 level grid), the FFE
    taps, and the train/test split mask."""
    eq_sps, n_ff = cfg.eq_sps, cfg.ffe_taps
    nsym = syms.size
    # DC-remove & roughly normalize the photocurrent before equalization
    rx = rx - np.mean(rx)
    rx = rx / (np.std(rx) + 1e-12)

    X = _build_ffe_matrix(rx, eq_sps, n_ff, nsym)          # [nsym, n_ff]
    ntr = max(n_ff * 4, int(cfg.train_frac * nsym))
    ntr = min(ntr, nsym)
    tr = slice(0, ntr)

    if cfg.dfe_taps <= 0:
        # ---- pure linear MMSE-FFE: solve least squares w = argmin ||X w - d|| ----
        d = syms                                           # desired = ideal Es=1 symbols
        Xtr, dtr = X[tr], d[tr]
        # ridge for conditioning
        A = Xtr.T @ Xtr + 1e-6 * np.eye(n_ff)
        w = np.linalg.solve(A, Xtr.T @ dtr)
        yhat = X @ w
        fb = None
    else:
        # ---- MMSE DFE: feedforward on rx + feedback on past *decided* symbols ----
        n_fb = cfg.dfe_taps
        # data-aided training: feedback uses the known past TX symbols.
        D = np.zeros((nsym, n_fb))
        for j in range(1, n_fb + 1):
            D[j:, j - 1] = syms[:-j]
        Xtot = np.concatenate([X, -D], axis=1)             # feedback subtracts past
        d = syms
        Xtr, dtr = Xtot[tr], d[tr]
        A = Xtr.T @ Xtr + 1e-6 * np.eye(Xtot.shape[1])
        wtot = np.linalg.solve(A, Xtr.T @ dtr)
        w, fb = wtot[:n_ff], wtot[n_ff:]
        # decision-directed apply on the test region (train uses known feedback)
        yhat = np.empty(nsym)
        dec = syms.copy()                                  # known in train region
        ff = X @ w
        for k in range(nsym):
            acc = ff[k]
            for j in range(1, n_fb + 1):
                if k - j >= 0:
                    acc -= fb[j - 1] * dec[k - j]
            yhat[k] = acc
            if k >= ntr:                                   # decide in test region
                dec[k] = lv[np.argmin(np.abs(lv - acc))]

    # final scale align to the Es=1 level grid (LS gain so slicer/LLR use lv directly)
    g = np.dot(yhat[tr], syms[tr]) / (np.dot(yhat[tr], yhat[tr]) + 1e-12)
    yhat = yhat * g
    return dict(yhat=yhat, w=w, fb=fb, ntr=ntr,
                test=slice(ntr, nsym), train=tr)


# --------------------------------------------------------------------------- #
#  soft metrics (LLR bridge) -- per-bit for PAM4, per-level for PAM6
# --------------------------------------------------------------------------- #
def soft_metrics_pam4(yhat: np.ndarray, lv: np.ndarray, sigma: float):
    """Exact per-bit LLRs for Gray PAM4 (2 per symbol, interleaved [hi,lo,...]).
    lv must be the ASCENDING Es=1 PAM4 levels; LLR>0 favours bit 0."""
    from scipy.special import logsumexp
    d = (yhat[:, None] - lv[None, :]) ** 2                 # [nsym,4]
    m = -d / (2.0 * sigma * sigma)
    bhi = PAM4_BITS_BY_LEVEL[:, 0]                          # MSB per ascending level
    blo = PAM4_BITS_BY_LEVEL[:, 1]
    llr_hi = logsumexp(m[:, bhi == 0], axis=1) - logsumexp(m[:, bhi == 1], axis=1)
    llr_lo = logsumexp(m[:, blo == 0], axis=1) - logsumexp(m[:, blo == 1], axis=1)
    out = np.empty(yhat.size * 2)
    out[0::2] = llr_hi
    out[1::2] = llr_lo
    return out, llr_hi, llr_lo


def soft_metrics_pam6(yhat: np.ndarray, lv: np.ndarray, sigma: float):
    """Per-level Gaussian log-likelihoods for PAM6 (6 columns/symbol).  A later
    SC-LDPC bridge can derive bit metrics from any chosen 6->bits labeling; here
    we return the raw per-level soft metric (log p(y|level), unnormalized)."""
    d = (yhat[:, None] - lv[None, :]) ** 2                 # [nsym,6]
    return -d / (2.0 * sigma * sigma)


# --------------------------------------------------------------------------- #
#  top-level driver
# --------------------------------------------------------------------------- #
def run_link(cfg: LinkConfig):
    """Run the full IM/DD link once; return TX/RX symbols, residual, soft metrics
    and uncoded performance numbers."""
    rng = np.random.default_rng(cfg.seed)
    syms, idx, bits, lv = make_tx_symbols(cfg, rng)
    rx, fdiag = forward_channel(cfg, syms)
    eq = equalize(cfg, rx, syms, lv)
    yhat = eq["yhat"]

    # --- evaluate on the TEST region only (equalizer trained on train region) ---
    te = eq["test"]
    x_te = syms[te]; y_te = yhat[te]; idx_te = idx[te]
    e = y_te - x_te                                         # residual error
    sigma = float(np.sqrt(np.mean(e ** 2)))                # residual std (per real dim)

    # effective SNR (signal power / residual power) and EVM
    snr_lin = np.mean(x_te ** 2) / (np.mean(e ** 2) + 1e-30)
    snr_dB = 10 * np.log10(snr_lin)
    evm = float(np.sqrt(np.mean(e ** 2) / np.mean(x_te ** 2)))

    # hard-decision SER (nearest level)
    dist = (y_te[:, None] - lv[None, :]) ** 2
    dec_idx = np.argmin(dist, axis=1)
    ser = float(np.mean(dec_idx != idx_te))

    # soft metrics / LLR bridge + uncoded BER for PAM4
    out = dict(cfg=cfg, syms=syms, idx=idx, yhat=yhat, lv=lv,
               test=te, residual=e, sigma=sigma, snr_dB=snr_dB, evm=evm,
               ser=ser, fdiag=fdiag, w=eq["w"], fb=eq["fb"], ntr=eq["ntr"])
    if cfg.n_levels == 4:
        llr, llr_hi, llr_lo = soft_metrics_pam4(y_te, lv, sigma)
        bits_te = PAM4_BITS_BY_LEVEL[idx_te].reshape(-1)
        hard_bits = (llr < 0).astype(np.uint8)
        ber = float(np.mean(hard_bits != bits_te))
        # per-bit hard BER (MSB vs LSB)
        ber_hi = float(np.mean((llr_hi < 0).astype(np.uint8) != PAM4_BITS_BY_LEVEL[idx_te, 0]))
        ber_lo = float(np.mean((llr_lo < 0).astype(np.uint8) != PAM4_BITS_BY_LEVEL[idx_te, 1]))
        out.update(llr=llr, llr_hi=llr_hi, llr_lo=llr_lo,
                   bits_te=bits_te, ber=ber, ber_hi=ber_hi, ber_lo=ber_lo)
    else:
        soft = soft_metrics_pam6(y_te, lv, sigma)
        out.update(soft=soft)
    return out


if __name__ == "__main__":
    # quick self-test: one PAM4 and one PAM6 point
    for nl in (4, 6):
        cfg = LinkConfig(n_levels=nl, nsym=3000, L_km=2.0, pd_B=30e9, ROP_dBm=-6.0)
        r = run_link(cfg)
        tag = f"PAM{nl}"
        extra = f" BER={r['ber']:.2e} (hi={r['ber_hi']:.2e} lo={r['ber_lo']:.2e})" if nl == 4 else ""
        print(f"{tag}: SNR_eff={r['snr_dB']:.2f}dB EVM={r['evm']*100:.1f}% "
              f"SER={r['ser']:.2e}{extra}  sigma={r['sigma']:.3f}")
