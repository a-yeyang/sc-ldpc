"""Monte-Carlo BER/FER simulation over AWGN: 5G NR component code vs SC-LDPC.

Sub-commands (so independent sweeps can run in parallel) :
    python simulate.py component   -> results_component.json
    python simulate.py sc          -> results_sc.json
    python simulate.py windowed    -> results_windowed.json
    python simulate.py plot        -> ber_curves.svg + results.csv  (combines the json)

Each sweep uses an adaptive stop: it stops at a point once `target_ferr` frame
errors have been collected (with a minimum frame count), which spends Monte-Carlo
effort where it matters and skips lower SNRs once a point is error-free.
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np

from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
from decoder import Tanner
import channel as ch
import plotting

Z = 24   # lifting size used throughout


def _run(encode_decode, ebn0_list, rate, n_frames, target_ferr, min_frames, label):
    rng = np.random.default_rng(12345)
    xs, bers, fers = [], [], []
    print(f"\n# {label}  (rate={rate:.4f})", flush=True)
    for ebn0 in ebn0_list:
        sigma = ch.ebn0_to_sigma(ebn0, rate)
        be = bits = fe = nf = 0
        t0 = time.time()
        while nf < n_frames:
            e, b, f = encode_decode(sigma, rng)
            be += e; bits += b; fe += f; nf += 1
            if nf >= min_frames and fe >= target_ferr:
                break
        ber = be / bits; fer = fe / nf
        xs.append(float(ebn0)); bers.append(ber); fers.append(fer)
        print(f"  Eb/N0={ebn0:+.2f}dB  BER={ber:.3e}  FER={fer:.3e}  "
              f"({nf} frames, {time.time()-t0:.1f}s)", flush=True)
        if fe == 0:
            break
    return {"x": xs, "y": bers, "fer": fers, "label": label, "rate": rate}


def run_component():
    code = NRLDPCCode(2, 1, Z)
    chk, var = code.edges()
    tan = Tanner(chk, var, code.M, code.N)
    punct = np.zeros(code.N, dtype=bool); punct[:code.n_punct] = True
    tx_mask = ~punct                       # the 2 punctured columns are not transmitted

    def ed(sigma, rng):
        msg = rng.integers(0, 2, size=code.K).astype(np.uint8)
        cw = code.encode(msg)
        y = ch.awgn(ch.bpsk(cw[tx_mask]), sigma, rng)
        llr = np.zeros(code.N)
        llr[tx_mask] = ch.llr_awgn(y, sigma)   # punctured bits keep llr = 0
        hard = tan.decode(llr, max_iter=50, alpha=0.8)
        err = int((hard[:code.K] != msg).sum())
        return err, code.K, int(err > 0)

    cur = _run(ed, [0.6, 0.9, 1.2, 1.5, 1.8, 2.1], code.rate,
               n_frames=8000, target_ferr=150, min_frames=300,
               label=f"5G NR BG2 component (Z={Z}, R={code.rate:.3f})")
    json.dump(cur, open("results_component.json", "w"))


def run_sc():
    sc = SCLDPCCode(NRLDPCCode(2, 1, Z), w=2, L=30, seed=0)
    tan = sc.full_tanner()

    def ed(sigma, rng):
        cw, info = sc.encode(rng)
        llr = sc.make_llr(cw, sigma, rng)
        hard = tan.decode(llr, max_iter=40, alpha=0.8)
        err = int((sc.extract_info(hard) != info).sum())
        return err, info.size, int(err > 0)

    cur = _run(ed, [0.0, 0.2, 0.35, 0.5, 0.65, 0.8], sc.rate,
               n_frames=700, target_ferr=50, min_frames=80,
               label=f"SC-LDPC full-BP (L={sc.L}, w={sc.w}, R={sc.rate:.3f})")
    json.dump(cur, open("results_sc.json", "w"))


def run_windowed():
    sc = SCLDPCCode(NRLDPCCode(2, 1, Z), w=2, L=30, seed=0)
    W = 8
    sc._window_layout(W)

    def ed(sigma, rng):
        cw, info = sc.encode(rng)
        llr = sc.make_llr(cw, sigma, rng)
        hard = sc.decode_windowed(llr, W=W, max_iter=15, alpha=0.8)
        err = int((sc.extract_info(hard) != info).sum())
        return err, info.size, int(err > 0)

    cur = _run(ed, [0.35, 0.5, 0.65, 0.8], sc.rate,
               n_frames=120, target_ferr=30, min_frames=40,
               label=f"SC-LDPC windowed (W={W})")
    json.dump(cur, open("results_windowed.json", "w"))


def run_plot():
    curves = []
    for fn in ["results_component.json", "results_sc.json", "results_windowed.json"]:
        try:
            curves.append(json.load(open(fn)))
        except FileNotFoundError:
            print(f"(missing {fn}, skipping)")
    plotting.write_csv(curves, "results.csv")
    plotting.semilogy(curves, ylabel="BER",
                      title="5G NR LDPC vs Spatially-Coupled LDPC (BPSK / AWGN)",
                      path="ber_curves.svg")
    print("wrote ber_curves.svg and results.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    {"component": run_component, "sc": run_sc,
     "windowed": run_windowed, "plot": run_plot}[cmd]()
