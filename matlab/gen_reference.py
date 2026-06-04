"""Regenerate ref_component.json — deterministic reference vectors for the MATLAB
cross-validation (crossValidate.m).

The 5G NR component code has NO randomness, so MATLAB must reproduce these
bit-for-bit.  Run from this folder:

    python3 gen_reference.py

It imports the original Python modules from the parent directory and writes
ref_component.json next to crossValidate.m.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nr_ldpc import NRLDPCCode
from decoder import Tanner


def det_msg(K):
    k = np.arange(K)
    return ((k * 1103515245 + 12345) >> 3 & 1).astype(np.uint8)


def det_llr_base(cw, scale=1.5):
    """Deterministic channel-like LLR (no zeroing)."""
    N = cw.size
    idx = np.arange(N)
    llr = scale * (1.0 - 2.0 * cw.astype(float)) + 0.8 * np.cos(idx * 0.7)
    flip = (idx % 17 == 0)
    llr[flip] = -llr[flip]
    return llr


def det_llr_from_cw(cw, tx_mask, scale=1.5):
    """Realistic LLR with punctured columns set to 0 (used for min-sum)."""
    llr = det_llr_base(cw, scale)
    llr[~tx_mask] = 0.0
    return llr


def main():
    ref = {"component_cases": []}
    configs = [(2, 1, 24, None), (1, 1, 24, None), (2, 0, 16, None),
               (2, 6, 52, None), (2, 1, 24, 8), (1, 1, 24, 10)]
    for (bg, ils, Z, mp) in configs:
        code = NRLDPCCode(bg, ils, Z, mp=mp)
        msg = det_msg(code.K)
        cw = code.encode(msg)
        chk, var = code.edges()
        tan = Tanner(chk, var, code.M, code.N)
        punct = np.zeros(code.N, dtype=bool); punct[:code.n_punct] = True
        tx = ~punct
        llr = det_llr_from_cw(cw, tx)                 # punctured -> 0 (realistic, min-sum)
        llr_nz = det_llr_base(cw)                      # no zero edges (sum-product, no NaN corner)
        hard_ms = tan.decode(llr.copy(), max_iter=50, method="minsum", alpha=0.8)
        hard_sp_nz = tan.decode(llr_nz.copy(), max_iter=50, method="sumproduct", alpha=0.8)
        ref["component_cases"].append(dict(
            bg=bg, ils=ils, Z=int(Z), mp=(mp if mp else code.mb_full),
            Kb=code.Kb, mb=int(code.mb), nb=int(code.nb),
            K=int(code.K), N=int(code.N), M=int(code.M),
            rate=float(code.rate), n_punct=int(code.n_punct),
            syndrome_cw=int(tan.syndrome_weight(cw)),
            msg=msg.tolist(), cw=cw.tolist(),
            hard_minsum=hard_ms.tolist(), hard_sumprod_nz=hard_sp_nz.tolist()))

    code = NRLDPCCode(2, 1, 24)
    ci = code.core_inv.astype(int)
    ref["core_inv_bg2_z24"] = dict(shape=list(ci.shape),
                                   rowsum=ci.sum(axis=1).tolist(),
                                   total=int(ci.sum()), diag=int(np.trace(ci)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_component.json")
    json.dump(ref, open(out, "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
