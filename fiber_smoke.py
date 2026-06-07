"""SC-LDPC over a fiber-optic link via OptiCommPy -- integration SMOKE TEST.

Wires this repo's spatially-coupled 5G-NR LDPC codec (numpy-only: ``nr_ldpc``,
``sc_ldpc``, ``qam``) to the open-source fiber simulator **OptiCommPy** and asks
the only question that matters for code-design research: does the whole chain
(encode -> QAM -> pulse-shape -> optical modulator -> SSMF nonlinear propagation
-> ASE -> coherent receiver -> DSP -> LLR -> windowed BP decode) close the loop,
behave like real fiber, and run fast enough for Monte-Carlo?

Physical chain (single polarization)
------------------------------------
  bits (our SC-LDPC encoder)
    -> Gray square-QAM symbols, Es=1                      qam.bits_to_symbols
    -> upsample x SpS + RRC pulse shaping                 dsp.core.{upsample,pulseShape,firFilter}
    -> launch-power scaling to Pin [dBm]                  dsp.core.pnorm + devices.dBm2W
    -> IQ Mach-Zehnder modulator (Vpi/bias exposed)       models.devices.iqm
    -> SSMF, split-step Fourier (CD + Kerr + EDFA ASE)    models.channels.ssfm
    -> (optional) extra ASE to hit a target OSNR          models.devices.gaussianComplexNoise
    -> coherent receiver, CW LO, zero linewidth           models.devices.coherentReceiver
    -> matched RRC filter                                 dsp.core.firFilter
    -> electronic chromatic-dispersion compensation       dsp.equalization.edc
    -> decimate to 1 sample/symbol (max-variance timing)  dsp.core.decimate

LLR bridge (the crux)
---------------------
After DSP we GENIE-ALIGN the recovered symbols to the known TX constellation:
estimate the single complex gain  g = <y, x*> / <x, x*>  from the transmitted
symbols x and divide it out (a smoke-test stand-in for blind equalisation +
carrier-phase recovery; legitimate here because linewidth = 0 so there is no
real phase noise -- only a static rotation/scale from the modulator + LO).  We
then estimate the effective per-dimension noise std  sigma_eff = sqrt(E|y-x|^2/2)
and feed  qam.soft_demap(y, sigma_eff, M)  straight into the windowed SC-LDPC
decoder.  Because our QAM uses Es=1 and the genie gain restores that scale, no
extra power bookkeeping is needed for the demapper.

Experiments (smoke success criteria)
------------------------------------
  1. BER vs OSNR waterfall   (low power, short fiber, ideal amp = ASE-only):
     coded BER must fall off a cliff -> the SC-LDPC waterfall + LLR bridge work.
  2. BER vs launch power     (multi-span, real EDFA ASE): the classic nonlinear
     "optimal launch power" hump -- effective noise (and BER at high power)
     improves then WORSENS as Kerr dominates -> SSFM nonlinearity is real.
  3. coded vs uncoded        at one OSNR -> the decoder closes a clear gap.
  4. length sweep L in {2,20} km at fixed power/OSNR -> dispersion/length effect.

Run on the pod (OptiCommPy + /work codec):  python3 /work/fiber_smoke.py
Outputs: fiber_smoke_*.csv and fiber_smoke.png next to the script.
"""
from __future__ import annotations
import sys, os, time, csv
import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qam
from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode

from optic.utils import parameters
from optic.models.devices import iqm, coherentReceiver, dBm2W, gaussianComplexNoise
from optic.models.channels import ssfm
from optic.dsp.core import pulseShape, firFilter, upsample, decimate, signalPower, pnorm
from optic.dsp.equalization import edc

# --------------------------------------------------------------------------- #
#  fixed simulation / optical front-end parameters
# --------------------------------------------------------------------------- #
SpS  = 4                  # samples per symbol (>=2 so RRC + Kerr are resolved)
Rs   = 32e9              # symbol rate [baud]
Fs   = SpS * Rs          # simulation sampling rate [Sa/s]
Fc   = 193.1e12          # optical carrier [Hz]  (C-band)
Bref = 12.5e9            # OSNR reference bandwidth (0.1 nm) [Hz]
ALPHA = 0.2              # fiber loss   [dB/km]
D     = 16.0             # dispersion   [ps/nm/km]
GAMMA = 1.3              # nonlinearity [1/W/km]
NF    = 4.5             # EDFA noise figure [dB]
HZ    = 0.5             # SSFM step [km]
ROLLOFF = 0.1
NTAPS   = 1024
PLO_DBM = 10.0          # local-oscillator power [dBm]

# RRC pulse + IQM parameters are stateless -> build once.
_pp = parameters(); _pp.pulseType = "rrc"; _pp.SpS = SpS
_pp.nFilterTaps = NTAPS; _pp.rollOff = ROLLOFF
PULSE = pulseShape(_pp); PULSE = PULSE / np.max(np.abs(PULSE))

_iqm = parameters(); _iqm.Vpi = 2.0; _iqm.VbI = -2.0; _iqm.VbQ = -2.0; _iqm.Vphi = 1.0
IQM_DRIVE_FRAC = 0.25    # drive amplitude = frac * Vpi  (small-signal ~ linear IQM)


def build_code(Z=16, MP=24, w=1, L=12):
    """Our SC-LDPC code: BG1 rate-1/2 component, round-robin edge spreading."""
    comp = NRLDPCCode(1, 0, Z, mp=MP)
    assign = np.arange(int((comp.B[:, :comp.Kb] >= 0).sum())) % (w + 1)
    sc = SCLDPCCode(comp, w=w, L=L, assign=assign)
    return sc, w


def transmit_frame(sc, w, info, M, Pin_dBm, Ltotal, Lspan, OSNR_dB, amp, seed_noise):
    """Push one info-bit frame through the full optical chain; return
    (coded info-bit errors, K, sigma_eff, uncoded tx-bit errors, n_tx_bits)."""
    # ---- our encoder -> transmitted coded bits -> QAM symbols (Es=1) ----
    cw, _ = sc.encode(info)
    txbits = cw[sc.tx_mask]
    syms, _ = qam.bits_to_symbols(txbits, M)
    nsym = syms.size

    # ---- RRC pulse shaping + upsample, then set launch power Pin ----
    sigTx = pnorm(firFilter(PULSE, upsample(syms, SpS)))      # unit-mean-power baseband
    Pin_W = dBm2W(Pin_dBm)
    u = IQM_DRIVE_FRAC * _iqm.Vpi * sigTx                     # complex IQM drive
    opt = iqm(np.sqrt(Pin_W), u, _iqm)                       # CW laser field sqrt(Pin)
    opt = opt * np.sqrt(Pin_W / signalPower(opt))            # pin optical power to Pin exactly

    # ---- SSMF: split-step Fourier (CD + Kerr + per-span EDFA ASE if amp='edfa') ----
    pc = parameters()
    pc.Ltotal = Ltotal; pc.Lspan = Lspan; pc.hz = HZ
    pc.alpha = ALPHA; pc.D = D; pc.gamma = GAMMA
    pc.Fc = Fc; pc.Fs = Fs; pc.amp = amp; pc.NF = NF
    pc.seed = seed_noise; pc.prgsBar = False
    sigCh = ssfm(opt, pc)

    # ---- optional explicit ASE to hit a TARGET OSNR (over Bref) ----
    if OSNR_dB is not None:
        Psig = signalPower(sigCh)
        Pase = (Psig / 10 ** (OSNR_dB / 10)) * (Fs / Bref)   # spread over full Fs
        sigCh = sigCh + gaussianComplexNoise(sigCh.shape, Pase, seed_noise)

    # ---- coherent receiver (CW LO, zero linewidth) ----
    Elo = np.sqrt(dBm2W(PLO_DBM)) * np.ones_like(sigCh)
    pPD = parameters(); pPD.ideal = True; pPD.Fs = Fs
    ybb = coherentReceiver(sigCh, Elo, pPD)

    # ---- DSP: matched filter -> EDC -> decimate to 1 SpS ----
    ymf = firFilter(PULSE, ybb)
    # EDC sizes its filter from the accumulated dispersion |beta2|*L*Rs^2.  For a
    # very short fiber that tap-count is tiny and its auto FFT size collapses to
    # NFFT == Ntaps, which makes blockwiseFFTConv's overlap-save block length 1
    # and returns an empty array (crash).  In that regime accumulated CD is
    # negligible, so we skip EDC entirely (nothing meaningful to compensate).
    c_kms = 299792458.0 / 1e3
    lam = c_kms / Fc
    beta2 = -(D * lam ** 2) / (2 * np.pi * c_kms)
    ntaps_edc = int(2 * np.ceil(6.67 * abs(beta2) * Ltotal * Rs ** 2 * (Fs / Rs)))
    nfft_edc = 2 ** int(np.ceil(np.log2(ntaps_edc))) if ntaps_edc > 0 else 0
    if nfft_edc > ntaps_edc:        # valid overlap-save block length (>=2)
        pE = parameters(); pE.L = Ltotal; pE.D = D; pE.Fc = Fc; pE.Fs = Fs; pE.Rs = Rs
        yedc = edc(ymf, pE)
    else:
        yedc = ymf       # accumulated CD negligible on this short span -> skip EDC
    pD = parameters(); pD.SpSin = SpS; pD.SpSout = 1
    yd = decimate(yedc, pD)

    # ---- genie-align + LLR bridge ----
    n = min(yd.size, nsym)
    yd = yd[:n]; x = syms[:n]
    g = np.vdot(x, yd) / np.vdot(x, x)                       # complex gain estimate
    yd = yd / g                                              # restore Es=1 scale/rotation
    sigma_eff = float(np.sqrt(np.mean(np.abs(yd - x) ** 2) / 2.0))
    llr_sym = qam.soft_demap(yd, sigma_eff, M)

    # ---- coded: windowed SC-LDPC BP decode ----
    llr = np.zeros(sc.num_var)
    llr[sc.tx_mask] = llr_sym[:txbits.size]
    llr[sc.known_mask] = 30.0
    Wdec = max(6, w + 2)
    hard = sc.decode_windowed(llr, W=Wdec, max_iter=12, alpha=0.8)
    coded_err = int((sc.extract_info(hard) != info).sum())

    # ---- uncoded reference: hard-slice the SAME symbols, count tx-bit errors ----
    uncoded_err = int(((llr_sym[:txbits.size] < 0).astype(np.uint8) != txbits).sum())
    return coded_err, sc.K, sigma_eff, uncoded_err, txbits.size


def sweep_point(sc, w, M, frames, seed=0, **kw):
    """Average a sweep point over `frames` independent frames (common random numbers)."""
    cerr = uerr = K = NB = 0
    ses = []
    t0 = time.time()
    for idx in range(frames):
        rng = np.random.default_rng([seed, idx])
        info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
        ce, k, se, ue, nb = transmit_frame(sc, w, info, M, seed_noise=10_000 + idx, **kw)
        cerr += ce; uerr += ue; K += k; NB += nb; ses.append(se)
    dt = (time.time() - t0) / frames
    return dict(coded_ber=cerr / K, uncoded_ber=uerr / NB,
                sigma_eff=float(np.mean(ses)), s_per_frame=dt,
                coded_err=cerr, K=K)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in fields})
    print(f"  wrote {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sc, w = build_code(Z=16, MP=24, w=1, L=12)
    print(f"SC-LDPC: K={sc.K} num_var={sc.num_var} N_tx={sc.N_tx} "
          f"rate={sc.rate:.4f}  w={w} L={sc.L}  (Z={sc.Z})")
    print(f"optical: Rs={Rs/1e9:.0f}GBd SpS={SpS} Fs={Fs/1e9:.0f}GSa/s "
          f"alpha={ALPHA} D={D} gamma={GAMMA} NF={NF}\n")
    results = {}

    # ----- Exp 1: BER vs OSNR waterfall (QPSK + 16-QAM), low power, short fiber, ASE-only
    print("=== Exp 1: BER vs OSNR waterfall  (Pin=-2 dBm, L=20 km, amp=ideal) ===")
    e1 = []
    for M, osnrs, fr in [(4,  [4, 5, 6, 7, 8, 9],          24),
                         (16, [11, 12, 12.5, 13, 14, 15],  24)]:
        for OSNR in osnrs:
            r = sweep_point(sc, w, M, fr, Pin_dBm=-2.0, Ltotal=20, Lspan=20,
                            OSNR_dB=OSNR, amp="ideal")
            r.update(M=M, OSNR_dB=OSNR)
            e1.append(r)
            print(f"  M={M:3d} OSNR={OSNR:5.1f}dB  codedBER={r['coded_ber']:.2e}  "
                  f"uncodedBER={r['uncoded_ber']:.2e}  {r['s_per_frame']*1000:.0f}ms/fr")
    results["osnr"] = e1
    write_csv(os.path.join(here, "fiber_smoke_osnr.csv"), e1,
              ["M", "OSNR_dB", "coded_ber", "uncoded_ber", "sigma_eff", "s_per_frame"])

    # ----- Exp 2: BER vs launch power, multi-span real EDFA ASE -> nonlinear hump
    print("\n=== Exp 2: BER vs launch power  (16-QAM, 5x80 km, real EDFA ASE) ===")
    e2 = []
    for Pin in [-6, -4, -2, 0, 2, 4, 5, 6, 7, 8, 9]:
        r = sweep_point(sc, w, 16, 16, Pin_dBm=Pin, Ltotal=400, Lspan=80,
                        OSNR_dB=None, amp="edfa")
        r.update(M=16, Pin_dBm=Pin)
        e2.append(r)
        print(f"  Pin={Pin:5.1f}dBm  sigma_eff={r['sigma_eff']:.4f}  "
              f"codedBER={r['coded_ber']:.2e}  {r['s_per_frame']:.2f}s/fr")
    results["power"] = e2
    write_csv(os.path.join(here, "fiber_smoke_power.csv"), e2,
              ["M", "Pin_dBm", "sigma_eff", "coded_ber", "uncoded_ber", "s_per_frame"])
    # report optimum
    Pin_opt = min(e2, key=lambda r: r["sigma_eff"])["Pin_dBm"]
    print(f"  --> optimal launch power (min sigma_eff): {Pin_opt:+.1f} dBm")

    # ----- Exp 3: coded vs uncoded at one operating point (printed from Exp 1 cliff)
    print("\n=== Exp 3: coded vs uncoded  (16-QAM, L=20 km, OSNR at the cliff) ===")
    for r in e1:
        if r["M"] == 16 and r["OSNR_dB"] in (12, 12.5, 13):
            gain = (r["uncoded_ber"] / r["coded_ber"]) if r["coded_ber"] > 0 else float("inf")
            print(f"  OSNR={r['OSNR_dB']:4.1f}dB  coded={r['coded_ber']:.2e}  "
                  f"uncoded={r['uncoded_ber']:.2e}  (factor {gain:.0f}x)" if r["coded_ber"] > 0
                  else f"  OSNR={r['OSNR_dB']:4.1f}dB  coded=0 (no errors)  "
                       f"uncoded={r['uncoded_ber']:.2e}")

    # ----- Exp 4: fiber length sweep at fixed power/OSNR
    print("\n=== Exp 4: length sweep L in {2,20} km  (16-QAM, Pin=-2 dBm, OSNR=12 dB) ===")
    e4 = []
    for Lkm in [2, 20]:
        r = sweep_point(sc, w, 16, 24, Pin_dBm=-2.0, Ltotal=Lkm, Lspan=Lkm,
                        OSNR_dB=12.0, amp="ideal")
        r.update(M=16, L_km=Lkm)
        e4.append(r)
        print(f"  L={Lkm:3d}km  codedBER={r['coded_ber']:.2e}  "
              f"uncodedBER={r['uncoded_ber']:.2e}  sigma_eff={r['sigma_eff']:.4f}  "
              f"{r['s_per_frame']*1000:.0f}ms/fr")
    results["length"] = e4
    write_csv(os.path.join(here, "fiber_smoke_length.csv"), e4,
              ["M", "L_km", "coded_ber", "uncoded_ber", "sigma_eff", "s_per_frame"])

    # ----- plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

        # (a) OSNR waterfall, coded vs uncoded
        for M in (4, 16):
            pts = [r for r in e1 if r["M"] == M]
            o = [r["OSNR_dB"] for r in pts]
            cb = [max(r["coded_ber"], 1e-6) for r in pts]
            ub = [max(r["uncoded_ber"], 1e-6) for r in pts]
            ax[0].semilogy(o, cb, "o-", label=f"coded {M}-QAM")
            ax[0].semilogy(o, ub, "x--", alpha=0.5, label=f"uncoded {M}-QAM")
        ax[0].set_xlabel("OSNR [dB]"); ax[0].set_ylabel("BER")
        ax[0].set_title("(1) OSNR waterfall  (L=20km, ASE-only)")
        ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8)

        # (b) launch-power: sigma_eff hump + coded BER
        p = [r["Pin_dBm"] for r in e2]
        se = [r["sigma_eff"] for r in e2]
        ax[1].plot(p, se, "s-", color="C2", label="sigma_eff")
        ax[1].axvline(Pin_opt, color="k", ls=":", alpha=0.6,
                      label=f"opt {Pin_opt:+.0f} dBm")
        ax[1].set_xlabel("launch power [dBm]"); ax[1].set_ylabel("effective noise std")
        ax[1].set_title("(2) nonlinear hump  (16-QAM, 5x80km)")
        ax[1].grid(True, alpha=0.3)
        ax1b = ax[1].twinx()
        ax1b.semilogy(p, [max(r["coded_ber"], 1e-6) for r in e2], "^--",
                      color="C3", alpha=0.7, label="coded BER")
        ax1b.set_ylabel("coded BER", color="C3")
        l1, lab1 = ax[1].get_legend_handles_labels()
        l2, lab2 = ax1b.get_legend_handles_labels()
        ax[1].legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper center")

        # (c) length sweep
        Ls = [r["L_km"] for r in e4]
        ax[2].semilogy(Ls, [max(r["coded_ber"], 1e-6) for r in e4], "o-", label="coded")
        ax[2].semilogy(Ls, [max(r["uncoded_ber"], 1e-6) for r in e4], "x--", label="uncoded")
        ax[2].set_xlabel("fiber length [km]"); ax[2].set_ylabel("BER")
        ax[2].set_title("(4) length sweep  (OSNR=12dB)")
        ax[2].grid(True, which="both", alpha=0.3); ax[2].legend(fontsize=8)

        fig.suptitle("SC-LDPC over OptiCommPy fiber link -- smoke test", fontsize=12)
        fig.tight_layout()
        out = os.path.join(here, "fiber_smoke.png")
        fig.savefig(out, dpi=120)
        print(f"\n  wrote {out}")
    except Exception as e:
        print("  [plot skipped]", repr(e))

    print("\nSMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
