"""End-to-end demonstration of the 5G-NR-based spatially-coupled LDPC system.

Run:  python demo.py
It builds the code, prints the coupled band structure, encodes a random
message, transmits over BPSK/AWGN, decodes with both the full-graph and the
sliding-window decoder, and visualises the SC "decoding wave".
"""
from __future__ import annotations
import numpy as np

from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
import channel as ch


def show_band(sc):
    """ASCII picture of the block band-diagonal coupled parity-check matrix."""
    print("Coupled base matrix  H_SC[i,t] = B_(i-t)   "
          "(rows = check positions i, cols = variable positions t)")
    print("    " + "".join(f"{t%10}" for t in range(sc.L)) + "   <- variable position t")
    for i in range(sc.L + sc.w):
        row = ""
        for t in range(sc.L):
            d = i - t
            row += "B" if 0 <= d <= sc.w else "."
        tag = "  <- termination checks" if i >= sc.L else ""
        print(f"i={i:2d} {row}{tag}")


def per_position_errors(sc, cw_hat, cw):
    """Bit errors per spatial position (info+parity), to watch the wave."""
    blk = sc.nb * sc.Z
    errs = []
    for t in range(sc.L):
        e = int((cw_hat[t*blk:(t+1)*blk] != cw[t*blk:(t+1)*blk]).sum())
        errs.append(e)
    return errs


def main():
    print("=" * 70)
    print("5G NR  ->  Spatially-Coupled LDPC   (BPSK / AWGN)")
    print("=" * 70)

    comp = NRLDPCCode(bg=2, ils=1, Z=24)
    print(f"\nComponent code: 5G NR BG2, Z={comp.Z}  "
          f"(K={comp.K}, N={comp.N}, rate={comp.rate:.3f}, "
          f"{comp.n_punct} punctured bits)")

    sc = SCLDPCCode(comp, w=2, L=12, seed=0)
    print(f"SC-LDPC: coupling memory w={sc.w}, coupling length L={sc.L}, "
          f"seed={sc.seed}")
    print(f"  info bits K={sc.K}, transmitted N={sc.N_tx}, rate={sc.rate:.3f} "
          f"(component rate {comp.rate:.3f} - small termination loss)\n")
    show_band(sc)

    # ---- encode --------------------------------------------------------- #
    rng = np.random.default_rng(2024)
    cw, info = sc.encode(rng)
    tan = sc.full_tanner()
    syn = tan.syndrome_weight(cw)
    print(f"\nEncoded one codeword. Parity-check syndrome weight = {syn} "
          f"(0 => valid codeword)")

    # ---- transmit + decode at a working SNR ----------------------------- #
    ebn0 = 1.0
    sigma = ch.ebn0_to_sigma(ebn0, sc.rate)
    llr = sc.make_llr(cw, sigma, rng)

    hard_full = tan.decode(llr, max_iter=60, alpha=0.8)
    ber_full = (sc.extract_info(hard_full) != info).mean()
    print(f"\n[Full-graph BP]   Eb/N0={ebn0} dB   info-BER={ber_full:.2e}   "
          f"{'recovered ✔' if ber_full == 0 else 'errors'}")

    hard_win = sc.decode_windowed(llr, W=6, max_iter=30, alpha=0.8)
    ber_win = (sc.extract_info(hard_win) != info).mean()
    print(f"[Windowed   BP]   W=6                 info-BER={ber_win:.2e}   "
          f"{'recovered ✔' if ber_win == 0 else 'errors'}")

    # ---- visualise the decoding wave near threshold --------------------- #
    print("\nDecoding wave (per-position bit errors) just below threshold:")
    ebn0_low = 0.3
    sigma = ch.ebn0_to_sigma(ebn0_low, sc.rate)
    llr = sc.make_llr(cw, sigma, rng)
    print(f"  channel only (hard on LLR), Eb/N0={ebn0_low} dB:")
    hard0 = (llr < 0).astype(np.uint8)
    print("   " + " ".join(f"{e:3d}" for e in per_position_errors(sc, hard0, cw)))
    for it in [2, 5, 10, 30]:
        hard = tan.decode(llr, max_iter=it, alpha=0.8)
        e = per_position_errors(sc, hard, cw)
        print(f"  after {it:2d} BP iters: " + " ".join(f"{x:3d}" for x in e)
              + ("   (wave eats inward from both terminated ends)" if it == 2 else ""))

    print("\nDone.  Run `python simulate.py {component|sc|windowed}` then "
          "`python simulate.py plot` for BER curves.")


if __name__ == "__main__":
    main()
