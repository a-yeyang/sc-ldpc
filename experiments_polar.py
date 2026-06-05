"""Spatially-coupled (PIC) polar experiments over AWGN (BPSK).

Run one experiment per process (they are independent):

    python experiments_polar.py main    # coupling gain: PIC vs uncoupled, matched rate
    python experiments_polar.py algos    # decoding algorithms on the SAME coupled code
    python experiments_polar.py J        # effect of coupling depth J
    python experiments_polar.py wave     # per-block decoding-wave profile
    python experiments_polar.py plot     # re-plot every results_polar_*.json

Each writes results_polar_<name>.json and exp_polar_<name>.svg.

The headline (main) compares, at matched information rate:
  * uncoupled NR polar under BP        -- the fair same-decoder baseline
  * PIC spatially-coupled polar, windowed BP   -- the coupling gain
  * uncoupled NR polar under CA-SCL    -- the strong single-block reference
The coupling gain is realised by *soft* windowed decoding: a shared bit is
protected by two blocks, and BP combines both blocks' channel evidence.  Hard
SC/SCL coupling cannot gain at equal rate (see `algos`).
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np

from nr_polar import NRPolarCode
from sc_polar import SCPolarCode
import polar_bp as bp
import channel as ch
import plotting


def _run(ed, ebn0_list, rate, label, n_frames=120, target_ferr=40,
         min_frames=20, target_berr=600):
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
        if be == 0:
            break
    return {"x": xs, "y": bers, "fer": fers, "label": label, "rate": rate}


# --------------------------------------------------------------------------- #
#  curve builders
# --------------------------------------------------------------------------- #
def comp_curve(A, E, ebn0_list, label, decoder="scl", L=8, bp_iter=30,
               crc_len=11, n_max=9, **kw):
    """Uncoupled 5G NR polar component (one block per frame)."""
    code = NRPolarCode(A, E, n_max=n_max, crc_len=crc_len)

    def ed(sigma, rng):
        a = rng.integers(0, 2, code.A).astype(np.uint8)
        e = code.encode(a)
        llr = ch.llr_awgn(ch.awgn(ch.bpsk(e), sigma, rng), sigma)
        if decoder == "sc":
            ah = code.decode_sc(llr)
        elif decoder == "bp":
            u = bp.bp_decode(code.rate_dematch(llr), code.frozen_mask,
                             np.zeros(code.N, np.uint8), max_iter=bp_iter)
            ah = u[code.info_positions][:code.A]
        else:
            ah, _ = code.decode_scl(llr, L=L)
        err = int((ah != a).sum())
        return err, code.A, int(err > 0)

    return _run(ed, ebn0_list, code.rate, label, **kw)


def pic_curve(A, E, L, J, c, ebn0_list, label, decoder="bp", list_size=8,
              bp_iter=30, **kw):
    """PIC spatially-coupled polar (one chain of L blocks per frame)."""
    comp = NRPolarCode(A, E, n_max=9, crc_len=11)
    sc = SCPolarCode(comp, L=L, J=J, c=c, seed=0)

    def ed(sigma, rng):
        cw, info = sc.encode(rng)
        llr = sc.make_llr(cw, sigma, rng)
        if decoder == "bp":
            ih, _ = sc.decode_bp(llr, bp_iter=bp_iter)
        else:                                   # 'sc' (list_size=1) or 'scl'
            ih, _ = sc.decode(llr, list_size=list_size)
        err = int((ih != info).sum())
        return err, info.size, int(err > 0)

    return _run(ed, ebn0_list, sc.rate, label, **kw)


# --------------------------------------------------------------------------- #
#  experiments
# --------------------------------------------------------------------------- #
def exp_main():
    """Coupling gain at matched rate (~0.25): PIC windowed-BP vs uncoupled."""
    grid = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    ref = SCPolarCode(NRPolarCode(40, 128), L=40, J=1, c=8)
    A_unc = int(round(ref.rate * 128))
    curves = [
        comp_curve(A_unc, 128, grid, f"uncoupled BP (R={A_unc/128:.2f})", decoder="bp"),
        pic_curve(40, 128, 40, 1, 8, grid, f"PIC windowed-BP (R={ref.rate:.2f})", decoder="bp"),
        comp_curve(A_unc, 128, grid, f"uncoupled CA-SCL L=8 (R={A_unc/128:.2f})", decoder="scl"),
    ]
    _save("main", curves, "Spatially-coupled polar (PIC) coupling gain over AWGN")


def exp_algos():
    """Decoding algorithms on the SAME coupled code (and uncoupled references).
    Shows that only SOFT (windowed BP) coupling beats the uncoupled baseline at
    equal rate; hard feed-forward/bidirectional SC(L) coupling does not."""
    grid = [2.5, 3.0, 3.5, 4.0, 4.5]
    A, E, L, J, c = 40, 128, 30, 1, 8
    ref = SCPolarCode(NRPolarCode(A, E), L=L, J=J, c=c)
    A_unc = int(round(ref.rate * E))
    curves = [
        comp_curve(A_unc, E, grid, f"uncoupled BP (R={A_unc/E:.2f})", decoder="bp"),
        pic_curve(A, E, L, J, c, grid, "PIC SC (list=1, hard)", decoder="scl", list_size=1),
        pic_curve(A, E, L, J, c, grid, "PIC CA-SCL L=8 (hard)", decoder="scl", list_size=8),
        pic_curve(A, E, L, J, c, grid, "PIC windowed-BP (soft)", decoder="bp"),
    ]
    _save("algos", curves, "Coupled-polar decoding algorithms (L=30, J=1, c=8)")


def exp_J():
    """Effect of coupling depth J at MATCHED rate (windowed BP).
    Keeping the per-side coupling J*c fixed holds the rate ~ (A - J*c)/E
    constant, so the curves isolate the effect of J (more gain for J>=2)."""
    grid = [2.5, 3.0, 3.5, 4.0, 4.5]
    A, E, L = 40, 128, 30                  # J*c = 8 fixed -> rate ~ (40-8)/128 = 0.25
    curves = [comp_curve(32, E, grid, "uncoupled BP (R=0.25)", decoder="bp")]
    for (J, c) in [(1, 8), (2, 4), (4, 2)]:
        sc = SCPolarCode(NRPolarCode(A, E), L=L, J=J, c=c)
        curves.append(pic_curve(A, E, L, J, c, grid,
                                f"PIC J={J} c={c} (R={sc.rate:.2f})", decoder="bp"))
    _save("J", curves, "PIC coupling depth J at matched rate (windowed BP, L=30)")


def exp_wave():
    """Per-block decoding-wave profile: block error rate vs position at three
    Eb/N0.  The chain is terminated at both ends, so reliability is highest at
    the boundaries and the wave propagates inward (the spatial-coupling
    signature)."""
    comp = NRPolarCode(40, 128)
    sc = SCPolarCode(comp, L=40, J=1, c=8)
    curves = []
    for ebn0 in [2.5, 3.0, 3.5]:
        rng = np.random.default_rng(0)
        sigma = ch.ebn0_to_sigma(ebn0, sc.rate)
        fail = np.zeros(sc.L)
        nf = 60
        t0 = time.time()
        for _ in range(nf):
            cw, _info = sc.encode(rng)
            _, cp = sc.decode_bp(sc.make_llr(cw, sigma, rng), bp_iter=30)
            fail += (~cp)
        curves.append({"x": list(range(sc.L)), "y": (fail / nf).tolist(),
                       "label": f"Eb/N0={ebn0}dB", "rate": sc.rate})
        print(f"  wave Eb/N0={ebn0}dB done ({time.time()-t0:.1f}s)", flush=True)
    json.dump(curves, open("results_polar_wave.json", "w"))
    plotting.semilogy(curves, xlabel="block position t", ylabel="block error rate",
                      title="PIC decoding wave (chain terminated at both ends)",
                      path="exp_polar_wave.svg")
    print("wrote results_polar_wave.json and exp_polar_wave.svg", flush=True)


def _save(name, curves, title):
    json.dump(curves, open(f"results_polar_{name}.json", "w"))
    plotting.semilogy(curves, ylabel="BER", title=title, path=f"exp_polar_{name}.svg")
    print(f"\nwrote results_polar_{name}.json and exp_polar_{name}.svg", flush=True)


def run_plot():
    import os
    for name in ["main", "algos", "J"]:
        fn = f"results_polar_{name}.json"
        if os.path.exists(fn):
            curves = json.load(open(fn))
            plotting.semilogy(curves, ylabel="BER", title=f"SC-polar experiment: {name}",
                              path=f"exp_polar_{name}.svg")
            print(f"re-plotted exp_polar_{name}.svg ({len(curves)} curves)")
    if os.path.exists("results_polar_wave.json"):
        curves = json.load(open("results_polar_wave.json"))
        plotting.semilogy(curves, xlabel="block position t", ylabel="block error rate",
                          title="PIC decoding wave (chain terminated at both ends)",
                          path="exp_polar_wave.svg")
        print("re-plotted exp_polar_wave.svg")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    {"main": exp_main, "algos": exp_algos, "J": exp_J, "wave": exp_wave,
     "plot": run_plot}[cmd]()
