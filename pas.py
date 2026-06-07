"""Probabilistic Amplitude Shaping (PAS) for square-QAM over complex AWGN.

This extends ``qam.py`` (Gray square M-QAM, two independent sqrt(M)-PAM axes) with
the Boecherer/Steiner/Schulte PAS architecture (IEEE TCOM 2015): the constellation
input distribution is shaped so that *amplitudes* follow a Maxwell-Boltzmann (MB)
PMF while *signs* stay uniform.  Shaping moves the input PMF towards a Gaussian and
buys back, at a fixed SNR, up to ~1.53 dB of the gap to capacity (the "shaping
gain"); the gain grows with M and ~vanishes for QPSK.

What lives here (qam.py is imported, never edited):
  * MB amplitude PMF  P_A(a) propto exp(-nu a^2),  nu >= 0   (nu=0 -> uniform);
    plus the induced per-axis *signed-level* PMF  P(x) = P_A(|x|)/2.
  * Energy-normalised shaped-QAM mapping: the shaped constellation has a different
    average energy, so we renormalise to unit symbol energy Es=1 before the channel
    so SNR comparisons stay fair.
  * A-PRIORI-AWARE exact soft demap.  With a non-uniform input prior the bit LLR
    must fold in the prior:
        L_i = lse_{x:b_i=0}[ logP(x) - |y-x|^2/2s^2 ]
            - lse_{x:b_i=1}[ logP(x) - |y-x|^2/2s^2 ] .
    Per axis this is just "add log P_A(|level|) to each level's metric"; with nu=0
    it reduces *exactly* to qam.soft_demap (verified in test_pas.py).
  * Monte-Carlo achievable rates over the noise: symbol-wise MI I(X;Y) and the
    bit-metric-decoding (BMD) rate  R_BMD = H(X) - sum_i H(B_i|Y)  [bits/complex
    symbol] -- the rate a BICM/LDPC (PAS) system achieves with input P_X.  This is
    the fast, rigorous shaping-gain signal.
  * An idealised shaped-symbol sampler (i.i.d. amplitude shaping -- the standard
    "i.i.d. DM" evaluation, no CCDM needed) and a ``ShapedQAMChannel`` exposing a
    ``transmit(bits, sigma, rng)`` that returns a-priori-aware per-bit LLRs, a
    drop-in for the SC-LDPC worker.

Conventions match qam.py exactly: unit symbol energy Es=1; LLR>0 favours bit 0;
Eb/N0 is the information-bit Eb/N0.  numpy-only (no scipy); reuses qam._logsumexp
and qam._qam_tables.
"""
from __future__ import annotations
import numpy as np

import qam
from qam import _logsumexp


# --------------------------------------------------------------------------- #
#  per-axis shaping tables (cached by (M, nu) and by (M, logits) is recomputed)
# --------------------------------------------------------------------------- #
def axis_amps(M: int) -> np.ndarray:
    """Unscaled per-axis PAM amplitudes {-(2^k-1),...,-1,1,...,2^k-1} (ascending)."""
    return qam._qam_tables(M)["amps_axis"].copy()


def amplitude_set(M: int) -> np.ndarray:
    """The 2^(k-1) positive amplitudes A = {1,3,...,2^k-1}."""
    a = axis_amps(M)
    return np.unique(np.abs(a))


def mb_pmf(M: int, nu: float) -> np.ndarray:
    """Maxwell-Boltzmann amplitude PMF over A={1,3,...,2^k-1}: P_A(a) ~ exp(-nu a^2).

    nu=0 gives the uniform amplitude PMF.  Returns an array aligned with
    amplitude_set(M)."""
    A = amplitude_set(M).astype(np.float64)
    z = -float(nu) * A * A
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def pmf_from_logits(M: int, logits) -> np.ndarray:
    """Free per-amplitude PMF from softmax logits (the richer shaping action space).

    `logits` has length 2^(k-1) (one per positive amplitude)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def axis_level_pmf(M: int, amp_pmf: np.ndarray) -> np.ndarray:
    """Per-axis signed-level PMF P(x), aligned with axis_amps(M) (ascending).

    Signs are uniform, so P(level x) = P_A(|x|) / 2."""
    A = amplitude_set(M)
    amp_of = {int(a): i for i, a in enumerate(A)}
    amps = axis_amps(M)
    out = np.empty(amps.size)
    for i, x in enumerate(amps):
        out[i] = amp_pmf[amp_of[int(abs(x))]] * 0.5
    return out


def shaping_scale(M: int, amp_pmf: np.ndarray) -> float:
    """Renormaliser s so the *complex* symbol has Es=1 under the shaped PMF.

    Per axis E[x^2] = sum_x P(x) x^2 (x = unscaled amps); a complex symbol is two
    independent axes so E[|s X|^2] = 2 s^2 E[x^2] = 1  ->  s = 1/sqrt(2 E[x^2])."""
    amps = axis_amps(M).astype(np.float64)
    plv = axis_level_pmf(M, amp_pmf)
    ex2 = float(np.sum(plv * amps * amps))
    return float(np.sqrt(1.0 / (2.0 * ex2)))


def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def symbol_entropy(M: int, amp_pmf: np.ndarray) -> float:
    """H(X) of the complex shaped symbol [bits/complex symbol] = 2 * H(per-axis level)."""
    return 2.0 * entropy_bits(axis_level_pmf(M, amp_pmf))


# --------------------------------------------------------------------------- #
#  SNR mapping for the shaped constellation
# --------------------------------------------------------------------------- #
def esn0_to_sigma(esn0_db: float) -> float:
    """Per-real-dim noise std at a given Es/N0 [dB] with Es=1 (shaped or not).

    N0 = 2 sigma^2, Es/N0 = 1/(2 sigma^2)  ->  sigma = sqrt(1/(2*Es/N0))."""
    esn0 = 10.0 ** (esn0_db / 10.0)
    return float(np.sqrt(1.0 / (2.0 * esn0)))


def ebn0_to_sigma(ebn0_db: float, rate_bits_per_cu: float) -> float:
    """Per-real-dim noise std for an info-bit Eb/N0 [dB].

    `rate_bits_per_cu` = information bits per complex channel use (the operational
    net spectral efficiency).  Eb/N0 = Es/(R N0) with Es=1 -> sigma^2 = 1/(2 R Eb/N0).
    Use the *operational* net rate (uniform info bits in / complex channel uses) so
    shaped and uniform systems are compared at matched info-bit energy."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return float(np.sqrt(1.0 / (2.0 * rate_bits_per_cu * ebn0)))


# --------------------------------------------------------------------------- #
#  a-priori-aware per-axis soft demap
# --------------------------------------------------------------------------- #
def _axis_llr_apriori(y_axis: np.ndarray, sigma: float, M: int,
                      log_plv: np.ndarray) -> np.ndarray:
    """Exact a-priori-aware per-bit LLRs (k per axis sample), shape [nsym, k].

    `log_plv` = log P(level) per scaled axis level (aligned with amps_scaled).
    LLR>0 favours bit 0.  With log_plv constant (uniform) this is qam._axis_llr."""
    T = qam._qam_tables(M)
    amps, bt, k = T["amps_scaled"], T["bit_table"], T["k"]
    d = (y_axis[:, None] - amps[None, :]) ** 2                  # [nsym, 2^k]
    metric = -d / (2.0 * sigma * sigma) + log_plv[None, :]      # + log prior
    out = np.empty((y_axis.size, k))
    for p in range(k):
        out[:, p] = (_logsumexp(metric[:, bt[:, p] == 0], axis=1)
                     - _logsumexp(metric[:, bt[:, p] == 1], axis=1))
    return out


def soft_demap_apriori(y: np.ndarray, sigma: float, M: int,
                       amp_pmf: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Shaped complex symbols -> a-priori-aware per-bit LLRs (m=log2(M) per symbol),
    interleaved [I bits (k), Q bits (k), ...] (inverts qam.bits_to_symbols).

    `amp_pmf` aligns with amplitude_set(M); `scale` is the shaping renormaliser
    (defaults to shaping_scale).  With amp_pmf uniform this matches qam.soft_demap."""
    T = qam._qam_tables(M)
    k, m = T["k"], T["m"]
    if scale is None:
        scale = shaping_scale(M, amp_pmf)
    # log prior per scaled axis level: P(level) under the shaped PMF
    plv = axis_level_pmf(M, amp_pmf)
    log_plv = np.log(plv + 1e-300)
    # note: amps_scaled in qam are at the *uniform* Es=1 scale; for shaped Es=1 the
    # geometry rescales by (scale / qam_scale).  We evaluate the demap against the
    # actually-transmitted scaled amplitudes (see ShapedQAMChannel) -- here y is in
    # the shaped scale, so rescale the reference amps accordingly.
    rescale = scale / T["scale"]
    yk = y / rescale     # bring y back to qam's amps_scaled grid
    llr_i = _axis_llr_apriori(np.real(yk), sigma / rescale, M, log_plv)
    llr_q = _axis_llr_apriori(np.imag(yk), sigma / rescale, M, log_plv)
    out = np.empty((y.size, m))
    out[:, :k] = llr_i
    out[:, k:] = llr_q
    return out.reshape(-1)


# --------------------------------------------------------------------------- #
#  idealised shaped-symbol sampler (i.i.d. amplitude shaping)
# --------------------------------------------------------------------------- #
def sample_axis_levels(M: int, amp_pmf: np.ndarray, n: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Draw n per-axis *unscaled* levels i.i.d. from the shaped signed-level PMF."""
    amps = axis_amps(M)
    plv = axis_level_pmf(M, amp_pmf)
    idx = rng.choice(amps.size, size=n, p=plv)
    return amps[idx].astype(np.float64)


def sample_shaped_symbols(M: int, amp_pmf: np.ndarray, n: int,
                          rng: np.random.Generator, scale: float | None = None):
    """n complex shaped QAM symbols (Es=1), i.i.d. amplitude shaping.

    Returns (symbols, scale)."""
    if scale is None:
        scale = shaping_scale(M, amp_pmf)
    xi = sample_axis_levels(M, amp_pmf, n, rng)
    xq = sample_axis_levels(M, amp_pmf, n, rng)
    return (xi + 1j * xq) * scale, scale


# --------------------------------------------------------------------------- #
#  achievable rates (Monte-Carlo over the noise)
# --------------------------------------------------------------------------- #
def _axis_posteriors(y_axis, sigma, amps_scaled, log_plv):
    """log P(x | y) per axis level, shape [nsym, nlev] (normalised over levels)."""
    d = (y_axis[:, None] - amps_scaled[None, :]) ** 2
    logj = -d / (2.0 * sigma * sigma) + log_plv[None, :]        # log joint up to const
    logpy = _logsumexp(logj, axis=1)                             # log p(y)
    return logj - logpy[:, None]                                 # log p(x|y)


def r_bmd(M: int, nu_or_pmf, esn0_db: float, n_sym: int = 20000, seed: int = 0,
          pmf: bool = False):
    """Monte-Carlo BMD (bit-metric decoding) rate and symbol-wise MI [bits/complex sym].

    Samples X~P_X (shaped), Y=X+N at Es/N0=esn0_db, evaluates per-bit posteriors:
        R_BMD = H(X) - sum_i H(B_i | Y),   I(X;Y) = H(X) - H(X|Y).
    `nu_or_pmf` is the MB nu (default) or, with pmf=True, an amplitude PMF/logits.
    Returns dict(hx, r_bmd, mi, h_b_given_y per axis-bit list)."""
    amp_pmf = pmf_from_logits(M, nu_or_pmf) if pmf else mb_pmf(M, float(nu_or_pmf))
    T = qam._qam_tables(M)
    k, m = T["k"], T["m"]
    amps = T["amps_scaled"]                       # uniform-Es grid
    bt = T["bit_table"]
    rng = np.random.default_rng([seed, M, int(round(esn0_db * 100))])
    sigma = esn0_to_sigma(esn0_db)

    # sample shaped symbols *on the shaped Es=1 grid*, then map back to the uniform grid
    scale = shaping_scale(M, amp_pmf)
    rescale = scale / T["scale"]
    xi = sample_axis_levels(M, amp_pmf, n_sym, rng)
    xq = sample_axis_levels(M, amp_pmf, n_sym, rng)
    # transmit at shaped scale, add noise, then express on the uniform grid:
    # y_shaped = scale*x_unscaled + n;  y_uni = y_shaped/rescale = qam_scale*x_unscaled + n/rescale
    s_unscaled = T["scale"]
    yi = s_unscaled * xi + (sigma / rescale) * rng.standard_normal(n_sym)
    yq = s_unscaled * xq + (sigma / rescale) * rng.standard_normal(n_sym)
    sig_eff = sigma / rescale

    plv = axis_level_pmf(M, amp_pmf)
    log_plv = np.log(plv + 1e-300)

    hx_axis = entropy_bits(plv)                   # H(level) per axis
    hx = 2.0 * hx_axis                            # H(X) complex symbol

    # symbol-wise H(X|Y) per axis = E[ -sum_x P(x|y) log2 P(x|y) ]
    def axis_cond_entropy(y):
        logpost = _axis_posteriors(y, sig_eff, amps, log_plv)   # natural log
        post = np.exp(logpost)
        h = -np.sum(post * (logpost / np.log(2.0)), axis=1)     # bits, per sample
        return float(h.mean())

    hxy_axis = 0.5 * (axis_cond_entropy(yi) + axis_cond_entropy(yq))
    mi = 2.0 * (hx_axis - hxy_axis)               # I(X;Y) complex symbol

    # per-bit conditional entropy H(B_i|Y) (BMD): average over I and Q axes/bits
    def axis_bit_cond_entropy(y):
        logpost = _axis_posteriors(y, sig_eff, amps, log_plv)   # [n, nlev], nat log
        post = np.exp(logpost)
        hbits = []
        for p in range(k):
            q1 = post[:, bt[:, p] == 1].sum(axis=1)             # P(B_p=1 | y)
            q1 = np.clip(q1, 1e-12, 1 - 1e-12)
            hb = -(q1 * np.log2(q1) + (1 - q1) * np.log2(1 - q1))
            hbits.append(float(hb.mean()))
        return hbits

    hb_i = axis_bit_cond_entropy(yi)
    hb_q = axis_bit_cond_entropy(yq)
    sum_hb = sum(hb_i) + sum(hb_q)                # sum_i H(B_i|Y) over m bits
    r_bmd_val = hx - sum_hb
    return {"hx": hx, "mi": mi, "r_bmd": float(r_bmd_val),
            "hb": [hb_i, hb_q], "nu": (None if pmf else float(nu_or_pmf))}


def mi_symbol(M: int, nu_or_pmf, esn0_db: float, n_sym: int = 20000, seed: int = 0,
              pmf: bool = False) -> float:
    """Symbol-wise mutual information I(X;Y) [bits/complex symbol] (convenience)."""
    return r_bmd(M, nu_or_pmf, esn0_db, n_sym=n_sym, seed=seed, pmf=pmf)["mi"]


def best_nu(M: int, esn0_db: float, nus=None, n_sym: int = 20000, seed: int = 0):
    """Grid-search the MB nu maximising R_BMD at a given Es/N0.  Returns (nu*, dict)."""
    if nus is None:
        nus = np.concatenate([[0.0], np.linspace(0.005, 0.35, 28)])
    best = (0.0, r_bmd(M, 0.0, esn0_db, n_sym=n_sym, seed=seed))
    for nu in nus:
        d = r_bmd(M, float(nu), esn0_db, n_sym=n_sym, seed=seed)
        if d["r_bmd"] > best[1]["r_bmd"]:
            best = (float(nu), d)
    return best


# --------------------------------------------------------------------------- #
#  ShapedQAMChannel (drop-in for the SC-LDPC worker; mirrors qam.QAMChannel)
# --------------------------------------------------------------------------- #
class ShapedQAMChannel:
    """Shaped square-M-QAM over complex AWGN with a-priori-aware demap.

    The *amplitude* bits feeding bits_to_symbols are assumed already drawn from the
    shaping distribution (the SC-LDPC systematic input carries shaped amplitude bits;
    signs are uniform info).  We modulate with the standard Gray map, renormalise to
    the shaped Es=1, add complex AWGN, and demap with the a-priori-aware metric so
    the decoder sees the shaping prior."""

    def __init__(self, M: int, nu: float = 0.0, amp_pmf=None):
        assert M in qam.ORDERS
        self.M = M
        self.m = qam.bits_per_symbol(M)
        self.amp_pmf = mb_pmf(M, nu) if amp_pmf is None else np.asarray(amp_pmf)
        self.nu = nu
        self.scale = shaping_scale(M, self.amp_pmf)
        self._qam_scale = qam._qam_tables(M)["scale"]
        self.rescale = self.scale / self._qam_scale

    def transmit(self, bits: np.ndarray, sigma: float, rng) -> np.ndarray:
        """coded (shaped-amplitude + sign) bits -> shaped QAM/AWGN -> a-priori LLRs.

        Returns per-bit LLRs of the same length as `bits`.  `sigma` is the per-real-
        dim noise std for the shaped Es=1 constellation (use ebn0_to_sigma here)."""
        syms_uni, n_pad = qam.bits_to_symbols(bits, self.M)   # uniform-Es geometry
        syms = syms_uni * self.rescale                        # shaped Es=1 geometry
        y = qam.awgn_complex(syms, sigma, rng)
        llr = soft_demap_apriori(y, sigma, self.M, self.amp_pmf, scale=self.scale)
        return llr[: bits.size] if n_pad else llr
