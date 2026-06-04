"""Validate structural assumptions of the 5G NR base graphs (BG1/BG2).

Before building the encoder we verify, on the *real* downloaded base matrices,
the structure that the efficient 5G NR QC-LDPC encoder relies on:

  * dimensions (BG1: 46x68, Kb=22 ; BG2: 42x52, Kb=10)
  * the first 4 rows only touch the first 4 parity columns (the "core")
  * rows >= 4 have an identity diagonal in parity cols (Kb+4 .. nb-1), shift 0
  * the 4Z x 4Z lifted core block is invertible over GF(2)

If all of this holds, the encoder in nr_ldpc.py is correct by construction.
"""
import numpy as np


def load_base_matrix(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([int(x) for x in line.split()])
    B = np.array(rows, dtype=int)
    return B


def gf2_inv(M):
    """Inverse of a square binary matrix over GF(2). Returns None if singular."""
    n = M.shape[0]
    A = np.concatenate([M.astype(np.uint8) & 1, np.eye(n, dtype=np.uint8)], axis=1)
    r = 0
    for c in range(n):
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        p = r + piv[0]
        A[[r, p]] = A[[p, r]]
        sel = A[:, c].copy().astype(bool)
        sel[r] = False
        A[sel] ^= A[r]
        r += 1
        if r == n:
            break
    if r != n:
        return None
    return A[:, n:]


def lift_block(shift, Z):
    """Z x Z circulant: identity cyclically shifted; shift==-1 -> zero block."""
    if shift < 0:
        return np.zeros((Z, Z), dtype=np.uint8)
    I = np.eye(Z, dtype=np.uint8)
    return np.roll(I, shift % Z, axis=1)


def analyze(path, Z, name, Kb):
    B = load_base_matrix(path)
    mb, nb = B.shape
    print(f"\n=== {name}  ({path}) ===")
    print(f"shape = {mb} x {nb}   Kb(info cols) = {Kb}   parity cols = {nb-Kb}")
    nnz = int((B >= 0).sum())
    print(f"non-zero base entries = {nnz}")
    assert nb - Kb == mb, "parity cols must equal #rows"

    # shifts in range
    vals = B[B >= 0]
    print(f"shift range = [{vals.min()}, {vals.max()}]  (Z={Z})")
    assert vals.max() < Z, "shift >= Z!"

    # (1) first 4 rows touch only parity cols Kb..Kb+3
    top = B[0:4, Kb+4:nb]
    print(f"first 4 rows, parity cols >= Kb+4 : nnz = {int((top>=0).sum())} (expect 0)")
    assert (top < 0).all()

    # (2) rows >=4 : identity diagonal at col (Kb + i), shift 0; nothing else in
    #     parity cols >= Kb+4 except own diagonal
    ok_diag = True
    for i in range(4, mb):
        diag_col = Kb + i
        if B[i, diag_col] != 0:
            ok_diag = False
            print(f"  row {i}: diagonal col {diag_col} shift = {B[i, diag_col]} (expect 0)")
        # any other entry in parity cols >= Kb+4 ?
        others = [c for c in range(Kb+4, nb) if c != diag_col and B[i, c] >= 0]
        if others:
            ok_diag = False
            print(f"  row {i}: unexpected parity entries at {others}")
    print(f"accumulator identity-diagonal structure (rows>=4): {'OK' if ok_diag else 'FAIL'}")
    assert ok_diag

    # core column degrees within first 4 rows
    print("core 4x4 (rows 0-3, parity cols Kb..Kb+3) shift submatrix:")
    print(B[0:4, Kb:Kb+4])

    # (3) lifted 4Z x 4Z core invertible over GF(2)
    core = np.zeros((4*Z, 4*Z), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            core[i*Z:(i+1)*Z, j*Z:(j+1)*Z] = lift_block(B[i, Kb+j], Z)
    inv = gf2_inv(core)
    print(f"lifted core block 4Z x 4Z ({4*Z}x{4*Z}) invertible over GF(2): {inv is not None}")
    assert inv is not None

    # which of p0..p3 (parity cols Kb..Kb+3) connect to rows >=4 (the E block)
    e_cols = {j: int((B[4:, Kb+j] >= 0).sum()) for j in range(4)}
    print(f"E-block: #rows>=4 connecting to core parity p0..p3 = {e_cols}")
    print("STRUCTURE OK ✔")
    return B


if __name__ == "__main__":
    analyze("data/NR_2_1_24.txt", 24, "BG2 iLS=1 Z=24", Kb=10)
    analyze("data/NR_1_1_24.txt", 24, "BG1 iLS=1 Z=24", Kb=22)
