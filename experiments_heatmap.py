"""Heatmap hyper-parameter search for SC-LDPC over PAM4+RRC.

For each code rate in {0.7,0.8,0.9} we sweep all combinations of
    code length   Z (5G NR / 3GPP lifting sizes, BG1 iLS0)  in {16,32,64}
    chain length  L                                          in {30,100,300}
    coupling depth w (shallow)                               in {1,2,3}
and record the BER at a fixed per-rate operating Eb/N0 (near the SC waterfall).
A plain (uncoupled) 5G NR QC-LDPC of the same rate & Z is the CONTROL column.

Each (rate,Z,L,w) cell is one process so subagents can run them in parallel:
    python experiments_heatmap.py cell 0.8 32 100 2
    python experiments_heatmap.py ctrl 0.8 32
    python experiments_heatmap.py render
"""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np

from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
from decoder import Tanner
from experiments import best_mp_for_rate
from experiments_pam4 import BETA, SPAN, SPS, W
import pam4_rrc as p4
import plotting

HM_RATES = [0.7, 0.8, 0.9]
HM_Z = [16, 32, 64]
HM_L = [30, 100, 300]
HM_W = [1, 2, 3]
HM_EBN0 = {0.7: 5.7, 0.8: 6.4, 0.9: 8.2}      # operating point per rate (dB)


def _ber_at(decode_one, ebn0, rate, n_frames=16, target_berr=200, min_frames=6):
    rng = np.random.default_rng(2025)
    sigma = p4.ebn0_to_sigma_pam4(ebn0, rate)
    be = bits = nf = 0
    while nf < n_frames:
        e, b = decode_one(sigma, rng)
        be += e; bits += b; nf += 1
        if nf >= min_frames and be >= target_berr:
            break
    return be / bits, nf


def cell(rate, Z, L, w):
    comp = NRLDPCCode(1, 0, Z, mp=best_mp_for_rate(1, rate))
    sc = SCLDPCCode(comp, w=w, L=L, seed=0)
    sc._window_layout(W)
    ch = p4.PAM4RRCChannel(BETA, SPAN, SPS)

    def one(sigma, rng):
        cw, info = sc.encode(rng)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = ch.transmit(cw[sc.tx_mask], sigma, rng)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=W, max_iter=12, alpha=0.8)
        return int((sc.extract_info(hard) != info).sum()), info.size

    t0 = time.time()
    ber, nf = _ber_at(one, HM_EBN0[rate], sc.rate,
                      n_frames=24, target_berr=300, min_frames=10)
    out = {"rate": sc.rate, "ebn0": HM_EBN0[rate], "ber": ber, "Z": Z, "L": L, "w": w}
    json.dump(out, open(f"results_hm_{int(rate*100)}_{Z}_{L}_{w}.json", "w"))
    print(f"cell R{rate} Z{Z} L{L} w{w}: SC_rate={sc.rate:.3f} "
          f"BER@{HM_EBN0[rate]}dB={ber:.3e} ({nf} fr, {time.time()-t0:.0f}s)", flush=True)


def ctrl(rate, Z):
    code = NRLDPCCode(1, 0, Z, mp=best_mp_for_rate(1, rate))
    tan = Tanner(*code.edges(), code.M, code.N)
    punct = np.zeros(code.N, dtype=bool); punct[:code.n_punct] = True
    tx = ~punct
    ch = p4.PAM4RRCChannel(BETA, SPAN, SPS)

    def one(sigma, rng):
        msg = rng.integers(0, 2, code.K).astype(np.uint8)
        cw = code.encode(msg)
        llr = np.zeros(code.N); llr[tx] = ch.transmit(cw[tx], sigma, rng)
        hard = tan.decode(llr, max_iter=30, alpha=0.8)
        return int((hard[:code.K] != msg).sum()), code.K

    ber, nf = _ber_at(one, HM_EBN0[rate], code.rate, n_frames=40, target_berr=300, min_frames=12)
    out = {"rate": code.rate, "ebn0": HM_EBN0[rate], "ber": ber, "Z": Z, "L": 0, "w": 0}
    json.dump(out, open(f"results_hm_ctrl_{int(rate*100)}_{Z}.json", "w"))
    print(f"ctrl R{rate} Z{Z}: rate={code.rate:.3f} BER@{HM_EBN0[rate]}dB={ber:.3e} ({nf} fr)", flush=True)


def render():
    for rate in HM_RATES:
        R2 = int(rate * 100)
        for w in HM_W:
            M, rows = [], []
            for Z in HM_Z:
                row = []
                for L in HM_L:
                    fn = f"results_hm_{R2}_{Z}_{L}_{w}.json"
                    row.append(json.load(open(fn))["ber"] if os.path.exists(fn) else None)
                cf = f"results_hm_ctrl_{R2}_{Z}.json"
                row.append(json.load(open(cf))["ber"] if os.path.exists(cf) else None)
                M.append(row); rows.append(f"Z={Z}")
            cols = [f"L={L}" for L in HM_L] + ["plain"]
            plotting.heatmap(M, rows, cols,
                             title=f"BER @ {HM_EBN0[rate]}dB  (PAM4, R={rate}, w={w})",
                             path=f"exp_hm_{R2}_w{w}.svg",
                             note="green=low BER (better). cols: SC chain lengths + plain-5G control")
            print(f"wrote exp_hm_{R2}_w{w}.svg")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "cell":
        cell(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif c == "ctrl":
        ctrl(float(sys.argv[2]), int(sys.argv[3]))
    elif c == "render":
        render()
