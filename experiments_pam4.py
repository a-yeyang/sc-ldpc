"""SC-LDPC over a PAM4 + RRC pulse-shaped AWGN waveform channel.

Chain per frame:  (SC-)LDPC encode -> take transmitted bits -> PAM4(Gray)
   -> upsample sps=4 -> RRC(beta=0.1, span=10) -> AWGN -> matched RRC + downsample
   -> soft LLR demap -> sliding-window BP decode.

Sub-commands (one (rate,L) per process so subagents can run them in parallel):
    python experiments_pam4.py run 0.7 100     -> results_pam4_70_100.json
    python experiments_pam4.py plot            -> exp_pam4_<R>.svg  (per rate)
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np

import outpaths as OP
from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
from decoder import Tanner
from experiments import best_mp_for_rate
import pam4_rrc as p4
import plotting

BETA, SPAN, SPS, W = 0.1, 10, 4, 6

# PAM4 window-decoder waterfalls are higher than BPSK; per-rate Eb/N0 grids
PAM4_GRID = {0.5: [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
             0.6: [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
             0.7: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
             0.8: [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
             0.9: [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]}


def make_llr_pam4(sc, cw, channel, sigma, rng, large=30.0):
    llr = np.zeros(sc.num_var)
    llr[sc.tx_mask] = channel.transmit(cw[sc.tx_mask], sigma, rng)
    llr[sc.known_mask] = large
    return llr


def pam4_curve(rate, L, ebn0_list, label, **kw):
    comp = NRLDPCCode(1, 1, 24, mp=best_mp_for_rate(1, rate))
    sc = SCLDPCCode(comp, w=2, L=L, seed=0)
    sc._window_layout(W)
    ch = p4.PAM4RRCChannel(BETA, SPAN, SPS)

    # custom loop (PAM4 noise std differs from the BPSK helper)
    rng = np.random.default_rng(2025)
    xs, ys = [], []
    print(f"\n# {label} (R={sc.rate:.3f}, PAM4 b{BETA} span{SPAN} sps{SPS}, W={W})", flush=True)
    for ebn0 in ebn0_list:
        sigma = p4.ebn0_to_sigma_pam4(ebn0, sc.rate)
        be = bits = nf = 0
        t0 = time.time()
        while nf < kw.get("n_frames", 40):
            cw, info = sc.encode(rng)
            llr = make_llr_pam4(sc, cw, ch, sigma, rng)
            hard = sc.decode_windowed(llr, W=W, max_iter=12, alpha=0.8)
            be += int((sc.extract_info(hard) != info).sum()); bits += info.size; nf += 1
            if nf >= kw.get("min_frames", 12) and be >= kw.get("target_berr", 400):
                break
        ber = be / bits
        xs.append(float(ebn0)); ys.append(ber)
        print(f"  Eb/N0={ebn0:+.2f}dB  BER={ber:.3e}  ({nf} fr, {time.time()-t0:.1f}s)", flush=True)
        if be == 0:
            break
    return {"x": xs, "y": ys, "label": label, "rate": sc.rate}


def pam4_component_curve(ils, Z, rate, ebn0_list, label, n_frames=200,
                         target_berr=400, min_frames=30):
    """CONTROL group: plain (uncoupled) 5G NR QC-LDPC, same rate & lifting size,
    full-graph BP over the same PAM4+RRC channel."""
    code = NRLDPCCode(1, ils, Z, mp=best_mp_for_rate(1, rate))
    tan = Tanner(*code.edges(), code.M, code.N)
    punct = np.zeros(code.N, dtype=bool); punct[:code.n_punct] = True
    tx = ~punct
    ch = p4.PAM4RRCChannel(BETA, SPAN, SPS)
    rng = np.random.default_rng(2025)
    xs, ys = [], []
    print(f"\n# {label} (R={code.rate:.3f}, N={code.N}, PAM4 full-BP)", flush=True)
    for ebn0 in ebn0_list:
        sigma = p4.ebn0_to_sigma_pam4(ebn0, code.rate)
        be = bits = nf = 0
        t0 = time.time()
        while nf < n_frames:
            msg = rng.integers(0, 2, code.K).astype(np.uint8)
            cw = code.encode(msg)
            llr = np.zeros(code.N)
            llr[tx] = ch.transmit(cw[tx], sigma, rng)
            hard = tan.decode(llr, max_iter=30, alpha=0.8)
            be += int((hard[:code.K] != msg).sum()); bits += code.K; nf += 1
            if nf >= min_frames and be >= target_berr:
                break
        ys.append(be / bits); xs.append(float(ebn0))
        print(f"  Eb/N0={ebn0:+.2f}dB  BER={be/bits:.3e}  ({nf} fr, {time.time()-t0:.1f}s)", flush=True)
        if be == 0:
            break
    return {"x": xs, "y": ys, "label": label, "rate": code.rate}


def run_control(rate):
    grid = list(PAM4_GRID[rate]) + [PAM4_GRID[rate][-1] + d for d in (0.7, 1.4, 2.1, 2.8)]
    cur = pam4_component_curve(1, 24, rate, grid, "plain 5G LDPC (uncoupled, full-BP)")
    json.dump(cur, open(OP.route(f"results_pam4_ctrl_{int(rate*100)}.json"), "w"))
    print(f"\nsaved results_pam4_ctrl_{int(rate*100)}.json", flush=True)


def run(rate, L):
    grid = PAM4_GRID[rate]
    # larger L => many more info bits per frame => fewer frames needed; also keeps
    # each (rate,L) job under the 10-min shell limit so it can run foreground.
    nf, tb = {30: (40, 400), 100: (30, 350), 200: (20, 250), 300: (14, 180)}.get(L, (30, 350))
    cur = pam4_curve(rate, L, grid, f"L={L}", n_frames=nf, target_berr=tb, min_frames=8)
    json.dump(cur, open(OP.route(f"results_pam4_{int(rate*100)}_{L}.json"), "w"))
    print(f"\nsaved results_pam4_{int(rate*100)}_{L}.json", flush=True)


def plot():
    import os
    for rate in [0.5, 0.6, 0.7, 0.8, 0.9]:
        R2 = int(rate * 100)
        curves = []
        cf = f"results_pam4_ctrl_{R2}.json"          # control first (plain 5G LDPC)
        if os.path.exists(OP.route(cf)):
            curves.append(json.load(open(OP.route(cf))))
        for L in [30, 100, 200, 300]:
            fn = f"results_pam4_{R2}_{L}.json"
            if os.path.exists(OP.route(fn)):
                curves.append(json.load(open(OP.route(fn))))
        if curves:
            Rc = curves[0]["rate"]
            plotting.semilogy(curves, ylabel="BER",
                              title=f"SC-LDPC over PAM4+RRC, R={rate:.1f} (BG1, windowed)",
                              path=f"exp_pam4_{R2}.svg")
            print(f"wrote exp_pam4_{R2}.svg ({len(curves)} curves)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(float(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "ctrl":
        run_control(float(sys.argv[2]))
    elif cmd == "plot":
        plot()
