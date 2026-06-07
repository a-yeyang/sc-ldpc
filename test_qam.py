"""Validate qam.py against closed-form theory (run locally; numpy+scipy only)."""
import math
import numpy as np
import qam


def Q(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def qam_ber_approx(M, ebn0_db):
    """Nearest-neighbour Gray square-QAM uncoded BER (dominant term)."""
    m = np.log2(M)
    ebn0 = 10 ** (ebn0_db / 10)
    return (4.0 / m) * (1 - 1 / np.sqrt(M)) * Q(np.sqrt(3 * m * ebn0 / (M - 1)))


def uncoded_ber(M, ebn0_db, nbits, rng):
    sigma = qam.ebn0_to_sigma_qam(ebn0_db, 1.0, M)          # uncoded -> R=1
    bits = rng.integers(0, 2, size=nbits).astype(np.int64)
    syms, npad = qam.bits_to_symbols(bits, M)
    y = qam.awgn_complex(syms, sigma, rng)
    llr = qam.soft_demap(y, sigma, M)[: bits.size]
    hard = (llr < 0).astype(np.int64)                       # LLR>0 favours bit 0
    return float((hard != bits).mean()), syms


def main():
    rng = np.random.default_rng(0)
    print("=== energy normalisation  E[|s|^2] (target 1.0) ===")
    for M in qam.ORDERS:
        syms, _ = qam.bits_to_symbols(rng.integers(0, 2, size=20000 * qam.bits_per_symbol(M)), M)
        print(f"  M={M:3d}  E|s|^2={np.mean(np.abs(syms)**2):.4f}  "
              f"#points={len(np.unique(np.round(syms, 6)))}")

    print("\n=== uncoded BER  (sim vs nearest-neighbour theory) ===")
    ok = True
    for M in qam.ORDERS:
        # operate each order near BER~1e-2..1e-3 (shift up with order)
        ebn0 = {4: 6.0, 16: 10.0, 64: 14.0, 256: 18.0}[M]
        sim, _ = uncoded_ber(M, ebn0, 4_000_000, rng)
        th = qam_ber_approx(M, ebn0)
        rel = abs(sim - th) / th
        flag = "OK" if rel < 0.20 else "CHECK"
        ok &= rel < 0.25
        print(f"  M={M:3d}  Eb/N0={ebn0:4.1f}dB  sim={sim:.3e}  theory={th:.3e}  rel={rel:5.1%}  {flag}")

    print("\n=== QPSK must equal BPSK:  BER = Q(sqrt(2 Eb/N0)) ===")
    for ebn0 in (4.0, 7.0):
        sim, _ = uncoded_ber(4, ebn0, 4_000_000, rng)
        th = Q(np.sqrt(2 * 10 ** (ebn0 / 10)))
        print(f"  Eb/N0={ebn0}dB  QPSK sim={sim:.3e}  BPSK theory={th:.3e}  rel={abs(sim-th)/th:.1%}")

    print("\n=== 16-QAM I-axis reproduces pam4_rrc's Gray PAM4 (same labels) ===")
    # pam4_rrc.LEVELS (inlined to avoid its scipy import) indexed by v=2*b_hi+b_lo
    pam4_levels = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(5.0)
    T = qam._qam_tables(16)
    # 16-QAM splits Es over I+Q, so each axis is pam4 scaled by 1/sqrt(2)
    same = all(abs(T["amp_by_bits"][b] * np.sqrt(2) - pam4_levels[b]) < 1e-9 for b in range(4))
    print(f"  axis labels match pam4 (up to Es-split sqrt2 scale): {same}")
    print(f"  ours/axis = {np.round(T['amp_by_bits'],4)}   pam4/sqrt2 = {np.round(pam4_levels/np.sqrt(2),4)}")

    print("\nALL OK" if ok else "\nSOME CHECKS OFF -- inspect above")


if __name__ == "__main__":
    main()
