"""Square M-QAM modulation over a complex AWGN channel for (SC-)LDPC.

Generalises ``pam4_rrc`` from 4-ary real PAM to square QAM of any order
M in {4, 16, 64, 256} (QPSK / 16/64/256-QAM).  A square M-QAM symbol is the
product of two independent Gray-coded sqrt(M)-PAM signals on the I and Q axes,
so soft demapping factorises into two per-axis PAM demaps -- exact and cheap
even for 256-QAM (16 levels/axis instead of 256 2-D points).

Chain (symbol level, Nyquist / ISI-free, i.e. the channel IS AWGN):
    coded bits -> Gray QAM symbol (Es=1) -> complex AWGN -> exact (log-sum-exp)
    per-bit LLR -> BP decoder.

Conventions (match the rest of the repo):
  * unit average symbol energy  Es = 1;
  * LLR > 0 favours bit 0  (decoder convention, same as channel.llr_awgn);
  * Eb/N0 is the INFORMATION-bit Eb/N0, so curves of different orders/rates are
    directly comparable.  With m = log2(M) coded bits/symbol and code rate R,
    per-real-dimension noise variance  sigma^2 = 1/(2 m R Eb/N0).

Bit order within a symbol: the first m/2 bits select the I level (MSB first),
the next m/2 the Q level -- the same interleaving pam4_rrc uses for its 2 bits.
For M=16 each axis is exactly pam4_rrc's verified Gray PAM4, which anchors the
construction.
"""
from __future__ import annotations
import numpy as np

# orders we support and their bits/symbol
ORDERS = (4, 16, 64, 256)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically-stable log(sum(exp(a))) along `axis` (numpy-only, no scipy)."""
    amax = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - amax), axis=axis))
    return s + np.squeeze(amax, axis=axis)


# --------------------------------------------------------------------------- #
#  Gray-coded sqrt(M)-PAM building block (one I or Q axis)
# --------------------------------------------------------------------------- #
def _inv_gray(g: np.ndarray, k: int) -> np.ndarray:
    """Inverse Gray code: recover the level index j from its Gray codeword g."""
    x = np.asarray(g, dtype=np.int64).copy()
    s = 1
    while s < k:
        x ^= x >> s
        s <<= 1
    return x


def _gray_pam(k: int):
    """k bits/axis -> (amps[2^k], bit_table[2^k, k]).

    amps[j] is the j-th PAM amplitude (ascending), one of {-(2^k-1),...,-1,1,...,
    2^k-1}; bit_table[j] is the k-bit (MSB-first) Gray label of level j, so
    adjacent amplitudes differ in exactly one bit.
    """
    n = 1 << k
    j = np.arange(n, dtype=np.int64)
    g = j ^ (j >> 1)                                        # Gray code of level j
    shifts = np.arange(k - 1, -1, -1)                       # MSB first
    bit_table = ((g[:, None] >> shifts[None, :]) & 1).astype(np.int64)
    amps = (2 * j - (n - 1)).astype(np.float64)             # ..., -3, -1, 1, 3, ...
    return amps, bit_table


def _qam_tables(M: int):
    """Per-axis QAM tables, cached by order.

    Returns dict with: k (bits/axis), m (bits/symbol), scale (Es=1 normaliser),
    amps_axis[2^k] (UN-scaled), amps_scaled[2^k], bit_table[2^k,k],
    amp_by_bits[2^k] (scaled amplitude indexed by the k-bit integer label)."""
    if M not in _qam_tables._cache:
        assert M in ORDERS, f"unsupported QAM order {M}"
        m = int(round(np.log2(M)))
        assert m % 2 == 0, "square QAM needs an even number of bits/symbol"
        k = m // 2
        amps, bit_table = _gray_pam(k)
        scale = float(np.sqrt(3.0 / (2.0 * (M - 1))))      # E[|s|^2] = 1
        bvals = np.arange(1 << k, dtype=np.int64)
        amp_by_bits = amps[_inv_gray(bvals, k)] * scale     # label-integer -> amplitude
        _qam_tables._cache[M] = dict(
            k=k, m=m, scale=scale, amps_axis=amps, amps_scaled=amps * scale,
            bit_table=bit_table, amp_by_bits=amp_by_bits,
            pow2=(1 << np.arange(k - 1, -1, -1)).astype(np.int64))   # MSB-first weights
    return _qam_tables._cache[M]


_qam_tables._cache = {}


# --------------------------------------------------------------------------- #
#  SNR mapping
# --------------------------------------------------------------------------- #
def ebn0_to_sigma_qam(ebn0_db: float, rate: float, M: int) -> float:
    """Per-real-dimension noise std for square M-QAM (Es=1) at info-bit Eb/N0.

    m = log2(M) coded bits/symbol; complex AWGN n=n_I+jn_Q, Var(n_I)=Var(n_Q)=sigma^2.
    Eb/N0 = Es/(m R N0), N0 = 2 sigma^2  =>  sigma^2 = 1/(2 m R Eb/N0)."""
    m = int(round(np.log2(M)))
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return float(np.sqrt(1.0 / (2.0 * m * rate * ebn0)))


def bits_per_symbol(M: int) -> int:
    return int(round(np.log2(M)))


# --------------------------------------------------------------------------- #
#  map / demap
# --------------------------------------------------------------------------- #
def bits_to_symbols(bits: np.ndarray, M: int):
    """Map a bit vector to complex QAM symbols (Es=1); pads to a multiple of
    m=log2(M) with zeros, returning (symbols, n_pad)."""
    T = _qam_tables(M)
    m, k, pow2 = T["m"], T["k"], T["pow2"]
    bits = np.asarray(bits, dtype=np.int64)
    n_pad = (-bits.size) % m
    if n_pad:
        bits = np.concatenate([bits, np.zeros(n_pad, dtype=np.int64)])
    g = bits.reshape(-1, m)                                  # [nsym, m]
    bi = g[:, :k] @ pow2                                     # I label integer (MSB first)
    bq = g[:, k:] @ pow2                                     # Q label integer
    syms = T["amp_by_bits"][bi] + 1j * T["amp_by_bits"][bq]
    return syms, n_pad


def awgn_complex(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Complex AWGN: independent N(0, sigma^2) on real and imaginary parts."""
    n = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    return x + sigma * n


def _axis_llr(y_axis: np.ndarray, sigma: float, M: int) -> np.ndarray:
    """Exact per-bit LLRs (k per axis sample), shape [nsym, k]. LLR>0 favours 0."""
    T = _qam_tables(M)
    amps, bt, k = T["amps_scaled"], T["bit_table"], T["k"]
    d = (y_axis[:, None] - amps[None, :]) ** 2              # [nsym, 2^k]
    metric = -d / (2.0 * sigma * sigma)
    out = np.empty((y_axis.size, k))
    for p in range(k):
        out[:, p] = (_logsumexp(metric[:, bt[:, p] == 0], axis=1)
                     - _logsumexp(metric[:, bt[:, p] == 1], axis=1))
    return out


def soft_demap(y: np.ndarray, sigma: float, M: int) -> np.ndarray:
    """Complex symbols -> per-bit LLRs (m=log2(M) per symbol), interleaved
    [I bits (k), Q bits (k), ...] to invert bits_to_symbols. LLR>0 favours 0."""
    T = _qam_tables(M)
    k, m = T["k"], T["m"]
    llr_i = _axis_llr(np.real(y), sigma, M)                 # [nsym, k]
    llr_q = _axis_llr(np.imag(y), sigma, M)                 # [nsym, k]
    out = np.empty((y.size, m))
    out[:, :k] = llr_i
    out[:, k:] = llr_q
    return out.reshape(-1)


# --------------------------------------------------------------------------- #
#  bundle (drop-in for the (SC-)LDPC worker, mirrors PAM4RRCChannel.transmit)
# --------------------------------------------------------------------------- #
class QAMChannel:
    """Symbol-level square-M-QAM over complex AWGN."""

    def __init__(self, M: int):
        assert M in ORDERS, f"unsupported QAM order {M}"
        self.M = M
        self.m = bits_per_symbol(M)

    def transmit(self, bits: np.ndarray, sigma: float, rng) -> np.ndarray:
        """coded bits -> QAM/AWGN -> per-bit LLRs (same length as `bits`)."""
        syms, n_pad = bits_to_symbols(bits, self.M)
        y = awgn_complex(syms, sigma, rng)
        llr = soft_demap(y, sigma, self.M)
        return llr[: bits.size] if n_pad else llr
