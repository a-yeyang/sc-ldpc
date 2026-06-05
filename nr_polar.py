"""5G NR polar component code (3GPP TS 38.212 Sections 5.3.1 + 5.4.1).

This module builds a standards-compliant polar code and provides the encoder
and the SC / CRC-aided SCL decoders.  It is the polar counterpart of
`nr_ldpc.NRLDPCCode`, and is reused per spatial position by the
spatially-coupled (PIC) construction in `sc_polar.py`.

Polar transform
---------------
The codeword is x = u G_N over GF(2) with the Arikan kernel

        F = [[1, 0],
             [1, 1]]          G_N = F^{(x)n}  (n-fold Kronecker power, N = 2^n)

TS 38.212 uses the *non* bit-reversed transform, so the input index i of u is
decoded by successive cancellation in natural order 0..N-1.

Construction (given a target (A, E))
------------------------------------
  * K = A + crc_len           information + CRC bits placed on the code
  * N = 2^n  from `compute_N`  (TS 38.212 5.3.1: n_max = 9 for PBCH/PDCCH,
                                10 for PUCCH/PUSCH)
  * rate matching (5.4.1): a 32-block sub-block interleaver followed by
    puncturing / shortening / repetition to reach exactly E transmitted bits
  * frozen / information set (5.3.1.2): the punctured/shortened coordinates are
    pre-frozen, then the K most reliable of the remaining coordinates (per the
    Table 5.3.1.2-1 reliability sequence Q) carry the information+CRC bits.

The reliability sequence Q (length 1024) lives in `data/polar_Q_Nmax.txt`;
Q[0] is the least and Q[1023] the most reliable synthetic channel.
"""
from __future__ import annotations
import os
import numpy as np

import polar_decoder as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LARGE_LLR = 30.0   # LLR magnitude for a perfectly known (shortened) coordinate

# 3GPP TS 38.212 Table 5.4.1.1-1 : sub-block interleaver pattern (32 blocks)
RM_SUBBLOCK_P = np.array(
    [0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
     12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31],
    dtype=np.int64)

# 3GPP TS 38.212 Section 5.1 CRC generator polynomials, as coefficient bits
# from the highest degree (D^L) down to D^0.
CRC_POLY = {
    6:  [1, 1, 0, 0, 0, 0, 1],                                  # D^6+D^5+1
    11: [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],                   # D^11+D^10+D^9+D^5+1
    24: [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,                 # CRC24C:
         0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],                   # D^24+D^23+D^21+D^20+D^17
}                                                               # +D^15+D^13+D^12+D^8+D^4+D^2+D+1


# --------------------------------------------------------------------------- #
#  reliability sequence and block length
# --------------------------------------------------------------------------- #
def _load_q_nmax() -> np.ndarray:
    q = np.loadtxt(os.path.join(DATA_DIR, "polar_Q_Nmax.txt"), dtype=np.int64)
    assert q.size == 1024
    return q


Q_NMAX = _load_q_nmax()


def reliability_sequence(N: int) -> np.ndarray:
    """The length-N reliability sequence (ascending: least -> most reliable).

    Obtained by keeping the entries of Q_Nmax that are < N (nested property)."""
    return Q_NMAX[Q_NMAX < N]


def compute_N(K: int, E: int, n_max: int = 9) -> int:
    """N = 2^n from (K, E, n_max), per TS 38.212 Section 5.3.1 (K includes CRC)."""
    if E <= (9 / 8) * 2 ** (int(np.ceil(np.log2(E))) - 1) and K / E < 9 / 16:
        n1 = int(np.ceil(np.log2(E))) - 1
    else:
        n1 = int(np.ceil(np.log2(E)))
    n2 = int(np.ceil(np.log2(K / (1 / 8))))     # R_min = 1/8
    n = max(5, min(n1, n2, n_max))
    return 1 << n


# --------------------------------------------------------------------------- #
#  rate matching (TS 38.212 5.4.1)
# --------------------------------------------------------------------------- #
def rate_matching_pattern(K: int, N: int, E: int):
    """Return (rm_idx, mode).

    rm_idx[k] in [0, N) is the polar-transform output coordinate that supplies
    the k-th transmitted bit (e = x[rm_idx]).  `mode` is one of
    'repetition' / 'puncturing' / 'shortening'."""
    assert N >= 32 and (N & (N - 1)) == 0
    step = N // 32
    n_idx = np.arange(N)
    i = (32 * n_idx) // N
    J = RM_SUBBLOCK_P[i] * step + (n_idx % step)   # interleaver: y[n] = x[J[n]]
    if E >= N:
        rm = J[np.arange(E) % N]
        mode = "repetition"
    elif K / E <= 7 / 16:
        rm = J[np.arange(E) + (N - E)]             # drop the first N-E (puncturing)
        mode = "puncturing"
    else:
        rm = J[np.arange(E)]                        # drop the last N-E (shortening)
        mode = "shortening"
    return rm.astype(np.int64), mode


def build_info_set(N: int, rm_idx: np.ndarray, mode: str, K: int):
    """Return (info_positions sorted ascending, frozen_mask) per TS 38.212 5.3.1.2."""
    transmitted = np.zeros(N, dtype=bool)
    transmitted[rm_idx] = True
    prefrozen = set(np.nonzero(~transmitted)[0].tolist())   # untransmitted -> pre-frozen
    if mode == "puncturing":
        E = rm_idx.size
        lim = int(np.ceil(3 * N / 4 - E / 2)) if E >= 3 * N / 4 else int(np.ceil(9 * N / 16 - E / 4))
        prefrozen |= set(range(lim))
    cand = [q for q in reliability_sequence(N) if q not in prefrozen]   # ascending reliability
    if len(cand) < K:
        raise ValueError(f"too many pre-frozen coordinates: {len(cand)} usable < K={K}")
    info = np.array(sorted(cand[-K:]), dtype=np.int64)      # K most reliable usable coords
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info] = False
    return info, frozen_mask


# --------------------------------------------------------------------------- #
#  polar transform and CRC
# --------------------------------------------------------------------------- #
def polar_transform(u: np.ndarray) -> np.ndarray:
    """x = u G_N over GF(2), via the in-place butterfly (N = 2^n)."""
    x = np.asarray(u, dtype=np.uint8).copy()
    N = x.size
    m = 1
    while m < N:
        for i in range(0, N, 2 * m):
            x[i:i + m] ^= x[i + m:i + 2 * m]
        m <<= 1
    return x


def crc_bits(payload: np.ndarray, crc_len: int) -> np.ndarray:
    """The crc_len CRC parity bits of `payload` (binary polynomial remainder)."""
    gen = np.array(CRC_POLY[crc_len], dtype=np.uint8)
    reg = np.concatenate([payload.astype(np.uint8), np.zeros(crc_len, dtype=np.uint8)])
    for i in range(payload.size):
        if reg[i]:
            reg[i:i + crc_len + 1] ^= gen
    return reg[payload.size:]


def crc_ok(info_with_crc: np.ndarray, crc_len: int) -> bool:
    """True iff the trailing crc_len bits are a valid CRC of the leading bits."""
    a = info_with_crc[:-crc_len]
    return bool(np.array_equal(crc_bits(a, crc_len), info_with_crc[-crc_len:]))


# --------------------------------------------------------------------------- #
#  5G NR polar code
# --------------------------------------------------------------------------- #
class NRPolarCode:
    """A standards-compliant 5G NR polar code for a target (A info bits, E coded bits)."""

    def __init__(self, A: int, E: int, n_max: int = 9, crc_len: int = 11):
        assert crc_len in CRC_POLY, f"crc_len must be one of {sorted(CRC_POLY)}"
        self.A = int(A)                 # payload bits
        self.crc_len = int(crc_len)
        self.K = self.A + self.crc_len  # bits placed on information coordinates
        self.E = int(E)                 # transmitted bits
        self.n_max = int(n_max)
        self.N = compute_N(self.K, self.E, self.n_max)
        self.n = int(np.log2(self.N))
        self.rm_idx, self.mode = rate_matching_pattern(self.K, self.N, self.E)
        self.info_positions, self.frozen_mask = build_info_set(self.N, self.rm_idx, self.mode, self.K)
        self._untransmitted = np.setdiff1d(np.arange(self.N), self.rm_idx, assume_unique=False)

    # ---- encoding --------------------------------------------------------- #
    def encode_u(self, info_with_crc: np.ndarray) -> np.ndarray:
        """Place K info+CRC bits on the information coordinates and transform."""
        u = np.zeros(self.N, dtype=np.uint8)
        u[self.info_positions] = np.asarray(info_with_crc, dtype=np.uint8)
        return polar_transform(u)

    def rate_match(self, x: np.ndarray) -> np.ndarray:
        return x[self.rm_idx]

    def encode(self, a: np.ndarray) -> np.ndarray:
        """a: A payload bits -> e: E transmitted bits (CRC attached automatically)."""
        a = np.asarray(a, dtype=np.uint8)
        assert a.size == self.A
        info = np.concatenate([a, crc_bits(a, self.crc_len)])
        return self.rate_match(self.encode_u(info))

    # ---- LLR de-rate-matching -------------------------------------------- #
    def rate_dematch(self, llr_E: np.ndarray) -> np.ndarray:
        """Spread E channel LLRs back onto the N transform coordinates.

        Shortened coords are known 0 (LLR +LARGE); punctured coords are
        unknown (LLR 0); repeated coords accumulate their LLRs."""
        llr_N = np.zeros(self.N, dtype=np.float64)
        if self.mode == "shortening":
            llr_N[self._untransmitted] = LARGE_LLR
        np.add.at(llr_N, self.rm_idx, np.asarray(llr_E, dtype=np.float64))
        return llr_N

    # ---- decoding --------------------------------------------------------- #
    def _frozen_values(self):
        return np.zeros(self.N, dtype=np.uint8)

    def decode_sc(self, llr_E: np.ndarray) -> np.ndarray:
        """SC decoding.  Returns A payload bits."""
        u_hat = pd.sc_decode(self.rate_dematch(llr_E), self.frozen_mask, self._frozen_values())
        return u_hat[self.info_positions][:self.A]

    def decode_scl(self, llr_E: np.ndarray, L: int = 8, use_crc: bool = True):
        """CA-SCL decoding.  Returns (A payload bits, crc_passed)."""
        check = (lambda info: crc_ok(info, self.crc_len)) if (use_crc and self.crc_len) else None
        u_hat = pd.scl_decode(self.rate_dematch(llr_E), self.frozen_mask, self._frozen_values(),
                              L=L, info_positions=self.info_positions, crc_check=check)
        info = u_hat[self.info_positions]
        passed = crc_ok(info, self.crc_len) if self.crc_len else True
        return info[:self.A], passed

    @property
    def rate(self) -> float:
        """Transmitted information rate A / E."""
        return self.A / self.E

    def __repr__(self):
        return (f"NRPolarCode(A={self.A}, E={self.E}, N={self.N}, K={self.K}, "
                f"crc={self.crc_len}, mode={self.mode}, R={self.rate:.3f})")
