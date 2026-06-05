"""Fast self-checks for the SC-polar implementation.  Run: python tests_polar.py"""
import numpy as np

import nr_polar as npc
import polar_decoder as pd
import polar_bp as bp
from sc_polar import SCPolarCode
import channel as ch


def test_polar_transform():
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5, 7):
        N = 1 << n
        G = np.array([[1]], dtype=np.uint8)
        for _ in range(n):
            G = np.kron(G, F)
        for _ in range(20):
            u = rng.integers(0, 2, N).astype(np.uint8)
            assert np.array_equal(npc.polar_transform(u), (u @ G) % 2), "x != u G_N"
    print("PASS  polar transform equals u G_N (G_N = F^{(x)n}), n up to 7")


def test_reliability_sequence():
    assert npc.Q_NMAX.size == 1024
    assert sorted(npc.Q_NMAX.tolist()) == list(range(1024)), "Q not a permutation"
    for N in (32, 64, 128, 256, 512):
        sub = npc.reliability_sequence(N)
        assert sub.size == N and sorted(sub.tolist()) == list(range(N)), "nested property"
    print("PASS  reliability sequence: permutation of 0..1023, nested for N=32..512")


def test_rate_matching():
    cases = {(40, 120): "puncturing", (64, 128): "repetition", (80, 100): "shortening"}
    for (A, E), mode in cases.items():
        code = npc.NRPolarCode(A, E)
        assert code.mode == mode, f"A={A} E={E}: expected {mode}, got {code.mode}"
        assert code.rm_idx.size == E and code.rm_idx.max() < code.N
        # info positions and frozen mask are complementary and sized K
        assert code.frozen_mask.sum() + code.K == code.N
    print("PASS  rate matching: puncturing/repetition/shortening modes + frozen set sizes")


def test_crc():
    rng = np.random.default_rng(3)
    for L in (6, 11, 24):
        a = rng.integers(0, 2, 32).astype(np.uint8)
        b = np.concatenate([a, npc.crc_bits(a, L)])
        assert npc.crc_ok(b, L), "clean CRC must pass"
        bad = b.copy(); bad[rng.integers(0, b.size)] ^= 1
        assert not npc.crc_ok(bad, L), "single bit flip must fail CRC"
    print("PASS  CRC6 / CRC11 / CRC24C: clean passes, single-bit error caught")


def test_component_decode_highsnr():
    rng = np.random.default_rng(4)
    code = npc.NRPolarCode(A=40, E=128, crc_len=11)
    sigma = ch.ebn0_to_sigma(7.0, code.rate)
    sc_ok = scl_ok = bp_ok = eqL1 = True
    for _ in range(60):
        a = rng.integers(0, 2, code.A).astype(np.uint8)
        e = code.encode(a)
        llr = ch.llr_awgn(ch.awgn(ch.bpsk(e), sigma, rng), sigma)
        sc_ok &= np.array_equal(code.decode_sc(llr), a)
        scl_ok &= np.array_equal(code.decode_scl(llr, L=8)[0], a)
        u = bp.bp_decode(code.rate_dematch(llr), code.frozen_mask,
                         np.zeros(code.N, np.uint8), max_iter=40)
        bp_ok &= np.array_equal(u[code.info_positions][:code.A], a)
        # SCL with L=1 equals SC on the same LLRs (u-domain)
        u_sc = pd.sc_decode(code.rate_dematch(llr), code.frozen_mask, np.zeros(code.N, np.uint8))
        u_l1 = pd.scl_decode(code.rate_dematch(llr), code.frozen_mask,
                             np.zeros(code.N, np.uint8), L=1)
        eqL1 &= np.array_equal(u_sc, u_l1)
    assert sc_ok and scl_ok and bp_ok, "component SC/SCL/BP failed at 7 dB"
    assert eqL1, "SCL(L=1) != SC"
    print("PASS  component SC / CA-SCL / BP recover at 7 dB, and SCL(L=1)==SC")


def test_pic_consistency():
    rng = np.random.default_rng(5)
    comp = npc.NRPolarCode(A=40, E=128, crc_len=11)
    sc = SCPolarCode(comp, L=12, J=1, c=8, seed=0)
    sigma = ch.ebn0_to_sigma(8.0, sc.rate)
    scl_ok = bp_ok = crc_ok_all = True
    for _ in range(20):
        cw, info = sc.encode(rng)
        llr = sc.make_llr(cw, sigma, rng)
        ih_scl, cp = sc.decode(llr, list_size=8)
        ih_bp, _ = sc.decode_bp(llr, bp_iter=30)
        scl_ok &= np.array_equal(ih_scl, info)
        bp_ok &= np.array_equal(ih_bp, info)
        crc_ok_all &= bool(cp.all())
    assert scl_ok and bp_ok, "PIC windowed SCL/BP failed at 8 dB"
    assert crc_ok_all, "not all per-block CRC passed at high SNR"
    print("PASS  PIC encode/decode consistent: SCL & BP recover all info at 8 dB (shared bits agree)")


def test_pic_rate():
    comp = npc.NRPolarCode(A=40, E=128, crc_len=11)
    sc = SCPolarCode(comp, L=20, J=1, c=8, seed=0)
    # free info = L*Kfresh + (#valid out groups)*c ; coupled rate below component rate
    assert sc.Kfresh == sc.A - 2 * sc.J * sc.c
    assert sc.K == sc.L * sc.Kfresh + (sc.L - 1) * sc.c        # J=1: L-1 valid out groups
    assert sc.rate < comp.rate, "coupling must cost some rate (overhead)"
    print(f"PASS  PIC rate={sc.rate:.4f} < component rate={comp.rate:.4f} (coupling overhead)")


if __name__ == "__main__":
    test_polar_transform()
    test_reliability_sequence()
    test_rate_matching()
    test_crc()
    test_component_decode_highsnr()
    test_pic_consistency()
    test_pic_rate()
    print("\nAll tests passed.")
