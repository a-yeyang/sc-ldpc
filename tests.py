"""Fast self-checks for the SC-LDPC implementation.  Run: python tests.py"""
import numpy as np
import scipy.sparse as sp

from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
from decoder import Tanner
import channel as ch


def test_component_encode():
    rng = np.random.default_rng(0)
    for (bg, ils, Z) in [(2, 1, 24), (1, 1, 24), (2, 0, 16), (2, 6, 52)]:
        code = NRLDPCCode(bg, ils, Z)
        chk, var = code.edges()
        H = sp.csr_matrix((np.ones(chk.size, np.uint8), (chk, var)), shape=(code.M, code.N))
        for _ in range(10):
            msg = rng.integers(0, 2, code.K).astype(np.uint8)
            c = code.encode(msg)
            assert not (np.asarray(H.dot(c)).ravel() & 1).any(), "H·c != 0"
            assert np.array_equal(c[:code.K], msg), "not systematic"
    print("PASS  component encoder: H·c=0 and systematic (BG1/BG2, several Z)")


def test_sc_encode_valid():
    rng = np.random.default_rng(1)
    sc = SCLDPCCode(NRLDPCCode(2, 1, 24), w=2, L=20, seed=0)
    tan = sc.full_tanner()
    for _ in range(5):
        cw, info = sc.encode(rng)
        assert tan.syndrome_weight(cw) == 0, "H_SC·c != 0"
    print("PASS  SC encoder: H_SC·c=0  (terminated, systematic edge spreading)")


def test_sc_rate():
    sc = SCLDPCCode(NRLDPCCode(2, 1, 24), w=2, L=30, seed=0)
    assert sc.K == (sc.L - sc.w) * sc.Kb * sc.Z
    assert 0.18 < sc.rate < 0.20 < (sc.comp.rate + 1e-9)
    print(f"PASS  SC rate={sc.rate:.4f} < component rate={sc.comp.rate:.4f} (termination loss)")


def test_decode_highsnr():
    rng = np.random.default_rng(2)
    sc = SCLDPCCode(NRLDPCCode(2, 1, 24), w=2, L=20, seed=0)
    tan = sc.full_tanner()
    sigma = ch.ebn0_to_sigma(2.0, sc.rate)
    full_ok = win_ok = True
    for _ in range(4):
        cw, info = sc.encode(rng)
        llr = sc.make_llr(cw, sigma, rng)
        full_ok &= (sc.extract_info(tan.decode(llr, max_iter=60)) == info).all()
        win_ok &= (sc.extract_info(sc.decode_windowed(llr, W=6, max_iter=30)) == info).all()
    assert full_ok and win_ok, "decoding failed at 2 dB"
    print("PASS  full-graph & windowed decoders recover the message at 2 dB")


if __name__ == "__main__":
    test_component_encode()
    test_sc_encode_valid()
    test_sc_rate()
    test_decode_highsnr()
    print("\nAll tests passed.")
