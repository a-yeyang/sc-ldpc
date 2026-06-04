"""Parametric SC-LDPC experiments over AWGN (BPSK).

Sweeps the four design knobs, one experiment each (run them in parallel):

    python experiments.py L       # coupling length  L   (chain length)
    python experiments.py w       # coupling depth   w   (memory)
    python experiments.py rate    # 5G NR code rate  R   (base-graph truncation)
    python experiments.py len     # code length      N   (lifting size Z)
    python experiments.py plot    # combine -> exp_summary.svg

Each writes results_exp_<name>.json and exp_<name>.svg.
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np

from nr_ldpc import NRLDPCCode, mp_for_rate, KB
from sc_ldpc import SCLDPCCode
from decoder import Tanner
import channel as ch
import plotting

MB_FULL = {1: 46, 2: 42}


def best_mp_for_rate(bg, target):
    """Integer mp whose (punctured) rate Kb/(Kb+mp-2) is closest to `target`."""
    Kb = KB[bg]
    return min(range(4, MB_FULL[bg] + 1),
              key=lambda mp: abs(Kb / (Kb + mp - 2) - target))


def _run(ed, ebn0_list, rate, label, n_frames=130, target_ferr=40,
         min_frames=50, target_berr=None):
    rng = np.random.default_rng(2025)
    xs, bers, fers = [], [], []
    print(f"\n# {label}  (rate={rate:.4f})", flush=True)
    for ebn0 in ebn0_list:
        sigma = ch.ebn0_to_sigma(ebn0, rate)
        be = bits = fe = nf = 0
        t0 = time.time()
        while nf < n_frames:
            e, b, f = ed(sigma, rng)
            be += e; bits += b; fe += f; nf += 1
            if nf >= min_frames and (fe >= target_ferr or
                                     (target_berr and be >= target_berr)):
                break
        ber, fer = be / bits, fe / nf
        xs.append(float(ebn0)); bers.append(ber); fers.append(fer)
        print(f"  Eb/N0={ebn0:+.2f}dB  BER={ber:.3e}  FER={fer:.3e}  "
              f"({nf} fr, {time.time()-t0:.1f}s)", flush=True)
        if fe == 0:
            break
    return {"x": xs, "y": bers, "fer": fers, "label": label, "rate": rate}


def sc_curve(bg, ils, Z, mp, w, L, ebn0_list, label, max_iter=40, **kw):
    comp = NRLDPCCode(bg, ils, Z, mp=mp)
    sc = SCLDPCCode(comp, w=w, L=L, seed=0)
    tan = sc.full_tanner()

    def ed(sigma, rng):
        cw, info = sc.encode(rng)
        llr = sc.make_llr(cw, sigma, rng)
        hard = tan.decode(llr, max_iter=max_iter, alpha=0.8)
        err = int((sc.extract_info(hard) != info).sum())
        return err, info.size, int(err > 0)

    return _run(ed, ebn0_list, sc.rate, label, **kw)


def comp_curve(bg, ils, Z, mp, ebn0_list, label, max_iter=50, **kw):
    code = NRLDPCCode(bg, ils, Z, mp=mp)
    tan = Tanner(*code.edges(), code.M, code.N)
    punct = np.zeros(code.N, dtype=bool); punct[:code.n_punct] = True
    tx = ~punct

    def ed(sigma, rng):
        msg = rng.integers(0, 2, code.K).astype(np.uint8)
        cw = code.encode(msg)
        y = ch.awgn(ch.bpsk(cw[tx]), sigma, rng)
        llr = np.zeros(code.N); llr[tx] = ch.llr_awgn(y, sigma)
        hard = tan.decode(llr, max_iter=max_iter, alpha=0.8)
        err = int((hard[:code.K] != msg).sum())
        return err, code.K, int(err > 0)

    return _run(ed, ebn0_list, code.rate, label, **kw)


def sc_window_curve(bg, ils, Z, mp, w, L, W, ebn0_list, label, max_iter=12, **kw):
    """SC curve decoded with the SLIDING-WINDOW decoder (latency ~ W, not L).
    This is the only scalable decoder for very long chains (L = 100..300):
    flooding BP would need O(L) iterations for the wave to cross the chain."""
    comp = NRLDPCCode(bg, ils, Z, mp=mp)
    sc = SCLDPCCode(comp, w=w, L=L, seed=0)
    sc._window_layout(W)

    def ed(sigma, rng):
        cw, info = sc.encode(rng)
        llr = sc.make_llr(cw, sigma, rng)
        hard = sc.decode_windowed(llr, W=W, max_iter=max_iter, alpha=0.8)
        err = int((sc.extract_info(hard) != info).sum())
        return err, info.size, int(err > 0)

    return _run(ed, ebn0_list, sc.rate, label, **kw)


# --------------------------------------------------------------------------- #
def exp_L():
    """Coupling length L : larger L -> smaller rate loss, threshold -> saturation."""
    Z = 16
    curves = [comp_curve(2, 0, Z, None, [0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
                         f"component BG2 Z={Z} (R=0.20)")]
    for L in [8, 16, 40]:
        curves.append(sc_curve(2, 0, Z, None, 2, L, [0.0, 0.3, 0.5, 0.7, 0.9, 1.1],
                               f"SC w=2 L={L}"))
    _save("L", curves, "Effect of coupling length L (BG2, Z=16, w=2)")


def exp_w():
    """Coupling depth w : stronger coupling vs more rate loss."""
    Z, L = 16, 30
    curves = [comp_curve(2, 0, Z, None, [0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
                         f"component BG2 Z={Z} (R=0.20)")]
    for w in [1, 2, 4]:
        curves.append(sc_curve(2, 0, Z, None, w, L, [0.0, 0.3, 0.5, 0.7, 0.9, 1.1],
                               f"SC L={L} w={w}"))
    _save("w", curves, "Effect of coupling depth w (BG2, Z=16, L=30)")


def exp_rate():
    """5G NR code rate (base-graph truncation): component vs SC at each rate."""
    Z, w, L = 24, 2, 30
    plan = {0.20: [0.4, 0.7, 1.0, 1.3, 1.6, 1.9],
            0.33: [1.0, 1.3, 1.6, 1.9, 2.2, 2.5],
            0.50: [1.6, 1.9, 2.2, 2.5, 2.8, 3.1]}
    curves = []
    for R, ebn0 in plan.items():
        mp = mp_for_rate(2, R)
        curves.append(comp_curve(2, 1, Z, mp, ebn0, f"component R={R:.2f}"))
        curves.append(sc_curve(2, 1, Z, mp, w, L, [e - 0.6 for e in ebn0],
                               f"SC R={R:.2f}"))
    _save("rate", curves, "Effect of code rate (5G NR rate matching, Z=24, w=2, L=30)")


def exp_rate_high():
    """High 5G NR code rates via BG1 rate matching: component vs SC at each rate.
    (BG2 maxes out near R=0.83, so BG1 -- K_b=22 -- is used for R>=0.6.)"""
    bg, Z, w, L = 1, 24, 2, 30
    # target rate -> (SC Eb/N0 grid, component Eb/N0 grid)  [from waterfall probe]
    plan = {0.6: ([0.8, 1.2, 1.6, 2.0, 2.4],       [1.4, 1.8, 2.2, 2.6, 3.0]),
            0.7: ([1.4, 1.8, 2.2, 2.6, 3.0],       [2.0, 2.4, 2.8, 3.2, 3.6]),
            0.8: ([2.0, 2.4, 2.8, 3.2, 3.6],       [2.6, 3.0, 3.4, 3.8, 4.2]),
            0.9: ([3.0, 3.4, 3.8, 4.2, 4.6, 5.0],  [3.8, 4.2, 4.6, 5.0, 5.4, 5.8])}
    curves = []
    for target, (sc_e, comp_e) in plan.items():
        mp = best_mp_for_rate(bg, target)
        Rc = NRLDPCCode(bg, 1, Z, mp=mp).rate
        curves.append(comp_curve(bg, 1, Z, mp, comp_e, f"component R={Rc:.2f}",
                                 n_frames=4000, target_ferr=120, min_frames=200))
        curves.append(sc_curve(bg, 1, Z, mp, w, L, sc_e, f"SC R={Rc:.2f}",
                               n_frames=150, target_ferr=40, min_frames=40))
    _save("rate_high", curves,
          "High code rates (5G NR BG1 rate matching, Z=24, w=2, L=30)")


def exp_len():
    """Code length via lifting size Z : longer blocks -> steeper waterfall."""
    w, L = 2, 24
    curves = [comp_curve(2, 0, 32, None, [0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
                         "component BG2 Z=32 (R=0.20)")]
    for Z in [16, 32, 64]:
        curves.append(sc_curve(2, 0, Z, None, w, L, [0.2, 0.4, 0.6, 0.8, 1.0],
                               f"SC Z={Z} (N/pos={52*Z})"))
    _save("len", curves, "Effect of code length / lifting size Z (BG2, w=2, L=24)")


def exp_chain(L):
    """Very long coupling chains L in {30,100,200,300}, sliding-window decoded.
    Run one L per process (parallel):  python experiments.py chain 300
    Demonstrates threshold saturation: the waterfall converges as L grows, while
    the windowed decoder's latency/complexity stays fixed (independent of L)."""
    Z, w, W = 16, 2, 6
    ebn0 = [0.4, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
    cur = sc_window_curve(2, 0, Z, None, w, L, W, ebn0,
                          f"SC L={L} (window W={W})", max_iter=12,
                          n_frames=40, target_ferr=10**9, target_berr=400,
                          min_frames=12)
    json.dump(cur, open(f"results_exp_chain_{L}.json", "w"))
    print(f"\nsaved results_exp_chain_{L}.json", flush=True)


def plot_chain():
    import os
    curves = []
    for L in [30, 100, 200, 300]:
        fn = f"results_exp_chain_{L}.json"
        if os.path.exists(fn):
            curves.append(json.load(open(fn)))
    if curves:
        plotting.semilogy(curves, ylabel="BER",
                          title="Threshold saturation: coupling length L (BG2 Z=16, w=2, windowed)",
                          path="exp_chain.svg")
        print(f"wrote exp_chain.svg ({len(curves)} curves)")


# Eb/N0 grids for the high-rate long-chain runs (window-decoder waterfalls, BG1 Z=24)
CHAIN_HIGH_GRID = {0.7: [1.8, 2.2, 2.5, 2.7, 2.9, 3.1, 3.4],
                   0.8: [2.2, 2.6, 2.9, 3.1, 3.3, 3.5, 3.8],
                   0.9: [3.2, 3.6, 3.9, 4.1, 4.3, 4.6, 4.9]}


def exp_chain_high(target, Ls=(30, 100, 200, 300)):
    """Long chains at HIGH code rate (BG1 rate matching), sliding-window decoded.
    Run one rate (optionally one L) per process for parallelism:
        python experiments.py chain_high 0.8            # all L
        python experiments.py chain_high 0.8 300        # just L=300
    """
    Z, w, W = 24, 2, 6
    mp = best_mp_for_rate(1, target)
    grid = CHAIN_HIGH_GRID[target]
    Rc = NRLDPCCode(1, 1, Z, mp=mp).rate
    for L in Ls:
        cur = sc_window_curve(1, 1, Z, mp, w, L, W, grid,
                              f"L={L} (R={Rc:.2f})", max_iter=12,
                              n_frames=40, target_ferr=10**9, target_berr=400,
                              min_frames=12)
        json.dump(cur, open(f"results_exp_chain_high_{int(target*100)}_{L}.json", "w"))
        print(f"\nsaved chain_high R~{target} L={L}", flush=True)


def plot_chain_high():
    import os
    for target in [0.7, 0.8, 0.9]:
        R2 = int(target * 100)
        curves = []
        for L in [30, 100, 200, 300]:
            fn = f"results_exp_chain_high_{R2}_{L}.json"
            if os.path.exists(fn):
                curves.append(json.load(open(fn)))
        if curves:
            Rc = NRLDPCCode(1, 1, 24, mp=best_mp_for_rate(1, target)).rate
            plotting.semilogy(curves, ylabel="BER",
                              title=f"Long chains at high rate R={Rc:.2f} (BG1 Z=24, w=2, windowed)",
                              path=f"exp_chain_high_{R2}.svg")
            print(f"wrote exp_chain_high_{R2}.svg ({len(curves)} curves)")


def _save(name, curves, title):
    json.dump(curves, open(f"results_exp_{name}.json", "w"))
    plotting.semilogy(curves, ylabel="BER", title=title, path=f"exp_{name}.svg")
    print(f"\nwrote results_exp_{name}.json and exp_{name}.svg", flush=True)


def run_plot():
    import os
    for name in ["L", "w", "rate", "len"]:
        fn = f"results_exp_{name}.json"
        if os.path.exists(fn):
            curves = json.load(open(fn))
            plotting.semilogy(curves, ylabel="BER", title=f"experiment: {name}",
                              path=f"exp_{name}.svg")
            print(f"re-plotted exp_{name}.svg ({len(curves)} curves)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "chain":
        exp_chain(int(sys.argv[2]))
    elif cmd == "chain_plot":
        plot_chain()
    elif cmd == "chain_high":
        Ls = [int(x) for x in sys.argv[3:]] or (30, 100, 200, 300)
        exp_chain_high(float(sys.argv[2]), Ls)
    elif cmd == "chain_high_plot":
        plot_chain_high()
    else:
        {"L": exp_L, "w": exp_w, "rate": exp_rate, "rate_high": exp_rate_high,
         "len": exp_len, "plot": run_plot}[cmd]()
