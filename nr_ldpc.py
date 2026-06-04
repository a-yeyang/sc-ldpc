"""5G NR QC-LDPC component code (3GPP TS 38.212 base graphs).

This module loads a real 5G NR base graph (BG1 / BG2) and provides an efficient
QC-LDPC encoder that exploits the standardised parity structure:

    H_parity = [ C  0 ]      C : 4x4 block "core"      (first 4 rows / cols)
               [ E  I ]      I : identity accumulator  (rows >= 4, dual-diagonal)

Encoding a row-syndrome s = [s_core ; s_rest] (the contribution of every
*non-parity* variable to each check row) is then:

    p_core = C^{-1} s_core          (one small GF(2) solve, precomputed inverse)
    p_i    = s_i  XOR  E_i p_core    for i >= 4   (pure accumulation)

The same `solve_parity` routine is reused by the spatially-coupled encoder,
where `s` additionally carries the coupling contribution from neighbouring
spatial positions.
"""
from __future__ import annotations
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# (base graph) -> number of systematic (information) base columns Kb
KB = {1: 22, 2: 10}


def mp_for_rate(bg: int, rate: float) -> int:
    """Parity rows mp giving (punctured) rate ~= Kb/(Kb+mp-2)  ->  mp = Kb/rate - Kb + 2."""
    Kb = KB[bg]
    return max(4, int(round(Kb / rate - Kb + 2)))


# --------------------------------------------------------------------------- #
#  low level helpers
# --------------------------------------------------------------------------- #
def load_base_matrix(bg: int, ils: int, Z: int) -> np.ndarray:
    """Load base matrix NR_{bg}_{ils}_{Z}.txt -> int array, -1 == no edge."""
    path = os.path.join(DATA_DIR, f"NR_{bg}_{ils}_{Z}.txt")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([int(x) for x in line.split()])
    return np.array(rows, dtype=np.int64)


def shift_vec(x: np.ndarray, v: int) -> np.ndarray:
    """Apply the QC circulant P^v to a length-Z vector: (P^v x)[r] = x[(r+v)%Z].

    Consistent with lifting an entry `v` to np.roll(I, v, axis=1)."""
    return np.roll(x, -v)


def gf2_inv(M: np.ndarray) -> np.ndarray | None:
    """Inverse of a square binary matrix over GF(2) (None if singular)."""
    n = M.shape[0]
    A = np.concatenate([M.astype(np.uint8) & 1, np.eye(n, dtype=np.uint8)], axis=1)
    r = 0
    for c in range(n):
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        p = r + piv[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        sel = A[:, c].astype(bool).copy()
        sel[r] = False
        A[sel] ^= A[r]
        r += 1
        if r == n:
            break
    return A[:, n:] if r == n else None


def base_entries(B: np.ndarray):
    """Return (rows, cols, shifts) of the non-(-1) entries of a base matrix."""
    r, c = np.nonzero(B >= 0)
    return r, c, B[r, c]


def edges_from_entries(rows, cols, shifts, Z, n_block_rows, n_block_cols):
    """Expand QC base entries into a flat edge list of the lifted graph.

    Entry (i, j, v) -> Z edges: check (i*Z + r) <-> var (j*Z + (r+v)%Z).
    Returns (e_chk, e_var) int arrays of length (#entries * Z).
    """
    rows = np.asarray(rows); cols = np.asarray(cols); shifts = np.asarray(shifts)
    r = np.arange(Z)
    # broadcast: for each entry e, Z edges
    chk = (rows[:, None] * Z + r[None, :]).ravel()
    var = (cols[:, None] * Z + ((r[None, :] + shifts[:, None]) % Z)).ravel()
    return chk.astype(np.int64), var.astype(np.int64)


# --------------------------------------------------------------------------- #
#  5G NR component code
# --------------------------------------------------------------------------- #
class NRLDPCCode:
    """A lifted 5G NR QC-LDPC code (single spatial position / ordinary block code)."""

    def __init__(self, bg: int = 2, ils: int = 1, Z: int = 24, mp: int | None = None):
        """mp : 5G rate matching -- keep only the first `mp` parity base rows
        (and the first Kb+mp base columns).  mp in [4, m_b_full]; default = full
        base graph (lowest rate).  Higher rate <=> smaller mp."""
        self.bg, self.ils, self.Z = bg, ils, Z
        B = load_base_matrix(bg, ils, Z)
        self.Kb = KB[bg]
        self.mb_full = B.shape[0]
        if mp is None:
            mp = self.mb_full
        assert 4 <= mp <= self.mb_full, f"mp must be in [4, {self.mb_full}]"
        # keep first mp rows and first Kb+mp columns (info + first mp parity)
        self.B = B[:mp, :self.Kb + mp].copy()
        self.mb, self.nb = self.B.shape
        self.mp = mp
        assert self.nb - self.Kb == self.mb
        self.K = self.Kb * Z          # info bits
        self.N = self.nb * Z          # (rate-matched) codeword bits
        self.M = self.mb * Z          # parity bits / checks
        self.n_punct = 2 * Z          # first two systematic columns are punctured
        self._build_parity_structure()

    # ---- precompute the encoder ------------------------------------------- #
    def _build_parity_structure(self):
        B, Z, Kb, mb = self.B, self.Z, self.Kb, self.mb
        # 4Z x 4Z lifted core (rows 0..3, parity cols Kb..Kb+3)
        core = np.zeros((4 * Z, 4 * Z), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                v = B[i, Kb + j]
                if v >= 0:
                    core[i*Z:(i+1)*Z, j*Z:(j+1)*Z] = np.roll(np.eye(Z, dtype=np.uint8), v, axis=1)
        self.core_inv = gf2_inv(core)
        assert self.core_inv is not None, "5G NR core block must be invertible"
        # E block : for rows >= 4, which core parity cols (0..3) they touch
        self.E = []  # list over rows 4..mb-1 of [(core_idx, shift), ...]
        for i in range(4, mb):
            lst = [(c, int(B[i, Kb + c])) for c in range(4) if B[i, Kb + c] >= 0]
            self.E.append(lst)

    # ---- parity solver reused by the SC encoder --------------------------- #
    def solve_parity(self, s: np.ndarray) -> np.ndarray:
        """Given row syndromes s (mb x Z, contribution of all non-parity vars),
        return parity blocks p (mb x Z) with H_parity @ p = s over GF(2)."""
        Z, mb = self.Z, self.mb
        p = np.zeros((mb, Z), dtype=np.uint8)
        # core: p_core = C^{-1} s_core
        s_core = s[0:4].reshape(4 * Z).astype(np.uint8)
        p_core = (self.core_inv @ s_core) & 1
        p[0:4] = p_core.reshape(4, Z)
        # accumulator rows
        for idx, i in enumerate(range(4, mb)):
            acc = s[i].copy()
            for (c, v) in self.E[idx]:
                acc ^= shift_vec(p[c], v)
            p[i] = acc & 1
        return p

    def message_syndrome(self, msg_blocks: np.ndarray) -> np.ndarray:
        """Row syndromes from the systematic bits only (mb x Z)."""
        Z, mb, Kb, B = self.Z, self.mb, self.Kb, self.B
        s = np.zeros((mb, Z), dtype=np.uint8)
        for i in range(mb):
            for j in range(Kb):
                v = B[i, j]
                if v >= 0:
                    s[i] ^= shift_vec(msg_blocks[j], v)
        return s

    # ---- public encode ---------------------------------------------------- #
    def encode(self, msg: np.ndarray) -> np.ndarray:
        """msg: K bits -> codeword: N bits (systematic, first K are msg)."""
        Z, Kb, mb = self.Z, self.Kb, self.mb
        msg_blocks = msg.reshape(Kb, Z).astype(np.uint8)
        s = self.message_syndrome(msg_blocks)
        p = self.solve_parity(s)
        return np.concatenate([msg_blocks.reshape(-1), p.reshape(-1)]).astype(np.uint8)

    # ---- parity-check graph ---------------------------------------------- #
    def edges(self):
        r, c, v = base_entries(self.B)
        return edges_from_entries(r, c, v, self.Z, self.mb, self.nb)

    @property
    def rate(self):
        """Transmitted code rate accounting for the 2 punctured columns."""
        return self.K / (self.N - self.n_punct)
