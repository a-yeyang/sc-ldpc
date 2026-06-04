"""Waveform-level PAM4 + RRC pulse-shaping channel for (SC-)LDPC over AWGN.

Chain:  coded bits -> PAM4 (Gray) -> upsample (sps) -> RRC shaping
        -> AWGN on the waveform -> RRC matched filter -> downsample
        -> soft (LLR) demapping -> BP decoder

Gray PAM4 mapping (2 bits -> 1 of 4 levels), value v = 2*b_hi + b_lo:
    v=00 -> -3   v=01 -> -1   v=11 -> +1   v=10 -> +3      (levels /sqrt(5), Es=1)
so adjacent amplitude levels differ in exactly one bit.

With a unit-energy RRC and ideal timing the matched-filter output at the symbol
instants is  y_k = a_k + n_k ,  n_k ~ N(0, sigma_w^2)  (Nyquist => ISI-free,
noise samples i.i.d.), i.e. the waveform chain reduces to the symbol-level AWGN
model -- the realism it adds is finite-span filter truncation (slight ISI).
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

SQRT5 = np.sqrt(5.0)
# levels indexed by 2-bit value v = 2*b_hi + b_lo  (Gray), normalised to Es=1
LEVELS = np.array([-3.0, -1.0, 3.0, 1.0]) / SQRT5
# bit tables for the 4 values
_B_HI = np.array([0, 0, 1, 1])     # MSB per value v
_B_LO = np.array([0, 1, 0, 1])     # LSB per value v


def rrc_filter(beta: float, span: int, sps: int) -> np.ndarray:
    """Root-raised-cosine impulse response, length span*sps+1, unit energy."""
    N = span * sps
    t = (np.arange(N + 1) - N / 2) / sps          # time in symbol periods
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1 - beta)) +
                   4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))                   # unit energy
    return h


def ebn0_to_sigma_pam4(ebn0_db: float, rate: float) -> float:
    """Waveform noise std for PAM4 (Es=1, 2 bits/sym) at info-bit Eb/N0.
    Decision model y=a+n, Var(n)=sigma^2 ; Eb/N0 = Es/(4 R sigma^2)."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return np.sqrt(1.0 / (4.0 * rate * ebn0))


def bits_to_symbols(bits: np.ndarray):
    """Map a bit vector to PAM4 amplitudes (pads to even length, returns pad flag)."""
    bits = np.asarray(bits, dtype=np.int64)
    pad = bits.size % 2
    if pad:
        bits = np.concatenate([bits, [0]])
    b_hi = bits[0::2]
    b_lo = bits[1::2]
    v = 2 * b_hi + b_lo
    return LEVELS[v], pad


def pulse_shape(symbols: np.ndarray, sps: int, h: np.ndarray) -> np.ndarray:
    up = np.zeros(symbols.size * sps)
    up[::sps] = symbols
    return np.convolve(up, h)                       # 'full'


def matched_filter_downsample(rx: np.ndarray, sps: int, h: np.ndarray,
                              nsyms: int) -> np.ndarray:
    """RRC matched filter + symbol-rate downsampling, aligned to the TX symbols."""
    mf = np.convolve(rx, h)                          # second RRC
    delay = h.size - 1                               # total delay of the two 'full' convs
    idx = delay + sps * np.arange(nsyms)
    return mf[idx]


def soft_demap(y: np.ndarray, sigma: float) -> np.ndarray:
    """Exact (log-sum-exp) LLRs, 2 per PAM4 symbol, interleaved as [b_hi,b_lo,...].
    LLR>0 favours bit 0 (decoder convention)."""
    # metric_v = -(y-level_v)^2 / (2 sigma^2)  for each of 4 levels, per symbol
    d = (y[:, None] - LEVELS[None, :]) ** 2
    m = -d / (2.0 * sigma * sigma)                  # [nsym, 4]
    hi0 = logsumexp(m[:, _B_HI == 0], axis=1)
    hi1 = logsumexp(m[:, _B_HI == 1], axis=1)
    lo0 = logsumexp(m[:, _B_LO == 0], axis=1)
    lo1 = logsumexp(m[:, _B_LO == 1], axis=1)
    llr_hi = hi0 - hi1
    llr_lo = lo0 - lo1
    out = np.empty(y.size * 2)
    out[0::2] = llr_hi
    out[1::2] = llr_lo
    return out


class PAM4RRCChannel:
    """Bundles the waveform chain for one (beta, span, sps)."""

    def __init__(self, beta=0.1, span=8, sps=4):
        self.beta, self.span, self.sps = beta, span, sps
        self.h = rrc_filter(beta, span, sps)

    def transmit(self, bits: np.ndarray, sigma: float, rng):
        """coded bits -> waveform chain -> per-bit LLRs (same length as bits)."""
        syms, pad = bits_to_symbols(bits)
        tx = pulse_shape(syms, self.sps, self.h)
        rx = tx + sigma * rng.standard_normal(tx.size)
        y = matched_filter_downsample(rx, self.sps, self.h, syms.size)
        llr = soft_demap(y, sigma)
        return llr[: bits.size]                      # drop the pad bit's LLR
