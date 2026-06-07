"""End-to-end smoke: SC-LDPC over square-QAM/AWGN decodes (run locally)."""
import numpy as np
import qam
from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode


def run(M, ebn0, w=3, L=20, Z=16, mp=24, W=6, frames=12, seed=0):
    comp = NRLDPCCode(1, 0, Z, mp=mp)
    sc = SCLDPCCode(comp, w=w, L=L, assign=np.arange(comp_nsys(comp)) % (w + 1))
    chan = qam.QAMChannel(M)
    sigma = qam.ebn0_to_sigma_qam(ebn0, sc.rate, M)
    be = bits = fe = 0
    for idx in range(frames):
        rng = np.random.default_rng([seed, idx])
        info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
        cw, _ = sc.encode(info)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = chan.transmit(cw[sc.tx_mask], sigma, rng)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=W, max_iter=12, alpha=0.8)
        err = int((sc.extract_info(hard) != info).sum())
        be += err; bits += info.size; fe += int(err > 0)
    return sc.rate, be / bits, fe / frames


def comp_nsys(comp):
    return int((comp.B[:, :comp.Kb] >= 0).sum())


if __name__ == "__main__":
    for M in (4, 16, 64):
        print(f"--- M={M} (rate-1/2 SC-LDPC, Z=16, w=3, L=20) ---")
        for ebn0 in (2, 4, 6, 8, 10, 12):
            rate, ber, fer = run(M, ebn0)
            print(f"  Eb/N0={ebn0:4.1f}dB  rate={rate:.3f}  BER={ber:.3e}  FER={fer:.2f}")
