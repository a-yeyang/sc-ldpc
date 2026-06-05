"""Belief-propagation (BP) decoder for polar codes on the Arikan factor graph.

Unlike SC/SCL this decoder is *soft-in soft-out*: it returns an extrinsic LLR
for every input (u) coordinate, which is exactly what the spatially-coupled
(PIC) windowed decoder needs in order to exchange information about the shared
bits between neighbouring blocks (the polar analogue of SC-LDPC windowed BP).

Factor graph (N = 2^n, non bit-reversed):
  * n+1 columns of N nodes; column n carries the channel LLRs, column 0 the u
    priors (frozen -> +-LARGE, information -> 0, plus any coupling prior).
  * each stage s pairs row i (bit s == 0) with row i + 2^s through the kernel
    [[1,1],[0,1]].  R messages flow towards the channel, L messages towards u.

Message updates (box-plus  f, ordinary sum  +):
    R[s+1, t] = f(R[s,t], R[s,b] + L[s+1,b])      L[s, t] = f(L[s+1,t], L[s+1,b] + R[s,b])
    R[s+1, b] = f(R[s,t], L[s+1,t]) + R[s,b]      L[s, b] = f(R[s,t], L[s+1,t]) + L[s+1,b]
with t the top (bit s == 0) and b = t + 2^s the bottom rows of every PE.
"""
from __future__ import annotations
import numpy as np

LARGE = 30.0   # LLR magnitude representing a (hard) known u coordinate


def _bp(a: np.ndarray, b: np.ndarray, scale: float = 0.9375) -> np.ndarray:
    """Scaled min-sum box-plus (the normalised-min-sum used by 5G receivers)."""
    return scale * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _stage_tops(N: int, n: int):
    """Top-row indices (bit s == 0) of every processing element, per stage."""
    idx = np.arange(N)
    return [idx[((idx >> s) & 1) == 0] for s in range(n)]


def bp_decode(channel_llr, frozen_mask, frozen_values, max_iter=40,
              prior=None, return_soft=False):
    """Flooding BP on the polar factor graph.

    channel_llr : length-N LLRs at the code (x) side (>0 favours bit 0)
    frozen_mask/frozen_values : the frozen specification (supports dynamic
        non-zero freezing for coupling)
    prior : optional length-N a-priori LLRs added at the u side (used by the
        coupled decoder to inject a neighbour's belief on shared bits)
    return_soft : also return the extrinsic LLRs on the u coordinates.
    """
    channel_llr = np.asarray(channel_llr, dtype=np.float64)
    N = channel_llr.size
    n = int(round(np.log2(N)))
    R = np.zeros((n + 1, N), dtype=np.float64)
    L = np.zeros((n + 1, N), dtype=np.float64)
    L[n] = channel_llr
    R[0][frozen_mask] = np.where(frozen_values[frozen_mask] == 0, LARGE, -LARGE)
    if prior is not None:
        R[0] = R[0] + prior
    tops = _stage_tops(N, n)
    for _ in range(max_iter):
        for s in range(n):                     # right sweep (towards channel)
            t = tops[s]; b = t + (1 << s)
            a, bb, c, d = R[s, t], R[s, b], L[s + 1, t], L[s + 1, b]
            R[s + 1, t] = _bp(a, bb + d)
            R[s + 1, b] = _bp(a, c) + bb
        for s in range(n - 1, -1, -1):         # left sweep (towards u)
            t = tops[s]; b = t + (1 << s)
            a, bb, c, d = R[s, t], R[s, b], L[s + 1, t], L[s + 1, b]
            L[s, t] = _bp(c, d + bb)
            L[s, b] = _bp(a, c) + d
    soft_u = L[0]                              # extrinsic on the u coordinates
    total = soft_u + R[0]
    u_hat = (total < 0).astype(np.uint8)
    u_hat[frozen_mask] = frozen_values[frozen_mask]
    if return_soft:
        return u_hat, soft_u
    return u_hat
