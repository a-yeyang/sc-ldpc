"""Successive-cancellation (SC) and CRC-aided SC-list (CA-SCL) polar decoders.

Both operate on a length-N channel LLR vector (>0 favours bit 0) and a frozen
specification (`frozen_mask`, `frozen_values`).  A frozen position is forced to
its given value during decoding; ordinary positions are decided from the LLRs.
Allowing *non-zero* frozen values is what lets the spatially-coupled (PIC)
decoder in `sc_polar.py` clamp a block's shared coordinates to a neighbour's
already-decoded bits ("dynamic freezing").

The decoders use the standard log-domain message passing on the polar factor
graph (Arikan butterfly, natural / non bit-reversed order):

    f(a, b) = 2 atanh(tanh(a/2) tanh(b/2))   (numerically stable box-plus)
    g(a, b, u) = b + (1 - 2u) a

SCL keeps the L lowest path metrics under the LLR-domain metric
    PM <- PM + softplus(-(1 - 2 u_hat) * LLR)
(Balatsoukas-Stimming, Parizi, Burg 2015), and the CRC selects the surviving
path.  Stage memory is computed once per node (O(N log N) per path).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def _f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stable exact box-plus LLR combine."""
    return (np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
            + np.log1p(np.exp(-np.abs(a + b))) - np.log1p(np.exp(-np.abs(a - b))))


def _g(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    return b + (1.0 - 2.0 * u.astype(np.float64)) * a


def _softplus(z: float) -> float:
    return max(z, 0.0) + np.log1p(np.exp(-abs(z)))


class _Path:
    """LLR/partial-sum stage memory for one SC(L) path (stage n = channel)."""
    __slots__ = ("Lm", "Bm", "u", "pm", "n", "N")

    def __init__(self, n: int, N: int):
        self.n, self.N = n, N
        self.Lm = np.zeros((n + 1, N), dtype=np.float64)   # row s: stage-s LLRs
        self.Bm = np.zeros((n + 1, N), dtype=np.uint8)     # row s: stage-s partial sums
        self.u = np.zeros(N, dtype=np.uint8)
        self.pm = 0.0

    def clone(self) -> "_Path":
        p = _Path.__new__(_Path)
        p.n, p.N, p.pm = self.n, self.N, self.pm
        p.Lm = self.Lm.copy(); p.Bm = self.Bm.copy(); p.u = self.u.copy()
        return p

    def calc_llr(self, s: int, phi: int):
        """Compute the stage-s LLRs of the node on the path to leaf phi."""
        if s == self.n:
            return
        b = phi >> s
        half = 1 << s
        pbase = (phi >> (s + 1)) << (s + 1)
        if b % 2 == 0:
            self.calc_llr(s + 1, phi)
        a = self.Lm[s + 1, pbase:pbase + half]
        c = self.Lm[s + 1, pbase + half:pbase + 2 * half]
        dbase = b << s
        if b % 2 == 0:
            self.Lm[s, dbase:dbase + half] = _f(a, c)
        else:
            lb = self.Bm[s, (b - 1) << s:((b - 1) << s) + half]
            self.Lm[s, dbase:dbase + half] = _g(a, c, lb)

    def set_bits(self, phi: int):
        """Propagate the just-decided leaf bit up the completed right children."""
        self.Bm[0, phi] = self.u[phi]
        b = phi
        for s in range(self.n):
            if b % 2 == 0:
                break
            half = 1 << s
            l = self.Bm[s, (b - 1) << s:((b - 1) << s) + half]
            r = self.Bm[s, b << s:(b << s) + half]
            ppar = (b >> 1) << (s + 1)
            self.Bm[s + 1, ppar:ppar + half] = l ^ r
            self.Bm[s + 1, ppar + half:ppar + 2 * half] = r
            b >>= 1


def sc_decode(llr, frozen_mask, frozen_values) -> np.ndarray:
    """Successive-cancellation decoding.  Returns the length-N estimate u_hat."""
    llr = np.asarray(llr, dtype=np.float64)
    N = llr.size
    n = int(round(np.log2(N)))
    p = _Path(n, N)
    p.Lm[n] = llr
    for phi in range(N):
        p.calc_llr(0, phi)
        p.u[phi] = frozen_values[phi] if frozen_mask[phi] else np.uint8(p.Lm[0, phi] < 0)
        p.set_bits(phi)
    return p.u


def scl_decode(llr, frozen_mask, frozen_values, L=8, info_positions=None,
               crc_check=None) -> np.ndarray:
    """CRC-aided SC-list decoding.  Returns the chosen path's length-N u_hat.

    If `crc_check` is given it is called on u_hat[info_positions] for each
    surviving path (lowest metric first); the first path that passes is
    returned, else the lowest-metric path."""
    llr = np.asarray(llr, dtype=np.float64)
    N = llr.size
    n = int(round(np.log2(N)))
    p0 = _Path(n, N)
    p0.Lm[n] = llr
    paths = [p0]
    for phi in range(N):
        for p in paths:
            p.calc_llr(0, phi)
        if frozen_mask[phi]:
            v = int(frozen_values[phi])
            for p in paths:
                p.pm += _softplus(-(1.0 - 2.0 * v) * p.Lm[0, phi])
                p.u[phi] = v
                p.set_bits(phi)
        else:
            cand = []   # (metric, parent_index, bit)
            for idx, p in enumerate(paths):
                val = p.Lm[0, phi]
                cand.append((p.pm + _softplus(-val), idx, 0))
                cand.append((p.pm + _softplus(val), idx, 1))
            cand.sort(key=lambda t: t[0])
            keep = cand[:min(L, len(cand))]
            groups = defaultdict(list)
            for pm, idx, bit in keep:
                groups[idx].append((pm, bit))
            new = []
            for idx, lst in groups.items():
                parent = paths[idx]
                objs = [parent] + [parent.clone() for _ in range(len(lst) - 1)]
                for (pm, bit), q in zip(lst, objs):
                    q.pm = pm
                    q.u[phi] = bit
                    q.set_bits(phi)
                    new.append(q)
            paths = new
    paths.sort(key=lambda p: p.pm)
    if crc_check is not None and info_positions is not None:
        for p in paths:
            if crc_check(p.u[info_positions]):
                return p.u
    return paths[0].u
