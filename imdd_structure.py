"""EFFECTIVE-CHANNEL STRUCTURE measurement for the bandwidth-limited IM/DD link.

The go/no-go deliverable.  After equalization (imdd_channel.run_link), form the
residual error  e[k] = yhat[k] - x[k]  and ask, for PAM4 and PAM6 over a grid of
(ROP, PD-B, L) points, whether the post-EQ channel still carries STRUCTURE an
SC-LDPC construction could exploit:

  (1) NOISE COLOR     -- PSD / autocorrelation of e.  Colored (off-diagonal acf,
                         non-flat PSD) = FFE noise-enhancement on a band-limited
                         channel = exploitable.  Reports acf at lags +/-1,2,3 and
                         a spectral-flatness measure (SFM in [0,1], 1=white).
  (2) RESIDUAL ISI    -- cross-correlation of e[k] with neighbour symbols x[k+/-1,2].
                         Non-zero = leftover ISI after the FFE.
  (3) UNEQUAL RELIAB. -- per-transmitted-LEVEL error variance (does reliability
                         depend on the PAM level?); for PAM4 also per-BIT (MSB vs
                         LSB) error variance / hard-BER -- the key structure for
                         LDPC variable-node degree matching.
  (4) EFF-SNR / EVM / SER / BER vs ROP, and the BANDWIDTH PENALTY: the SNR gap of
                         PD B=30/40 GHz vs a wide-open IDEAL PD at the same ROP.
  (5) CLEAN LLR OUTPUT (per-bit PAM4, per-level PAM6) + one quick SC-LDPC sanity
                         decode to confirm the LLR bridge works on IM/DD.

Outputs (next to this file, routed through outpaths if present):
  imdd_structure.csv         -- one row per (group, B, L, ROP, eq) point, all metrics
  imdd_acf.csv               -- residual autocorrelation vs lag for the headline points
  imdd_per_level.csv         -- per-level error variance for the headline points
  imdd_structure.png         -- acf / PSD / per-level-reliability / SNR-vs-ROP panels
  imdd_llr_pam4_*.npy        -- saved per-bit LLRs (MSB/LSB) for a headline PAM4 point

Run on the pod (OptiCommPy + /work codec):  python3 /work/imdd_structure.py
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imdd_channel import LinkConfig, run_link, PAM4_BITS_BY_LEVEL

try:
    import outpaths as OP
    def route(name): return OP.route(name)
except Exception:
    HERE = os.path.dirname(os.path.abspath(__file__))
    def route(name): return os.path.join(HERE, name)


# --------------------------------------------------------------------------- #
#  structure metrics on a residual e[k] (+ neighbour symbols for ISI)
# --------------------------------------------------------------------------- #
def autocorr(e: np.ndarray, maxlag: int = 6):
    """Normalized autocorrelation r[l] = E[e_k e_{k+l}] / E[e^2], l=0..maxlag."""
    e = e - e.mean()
    v = np.mean(e ** 2) + 1e-30
    r = np.empty(maxlag + 1)
    for l in range(maxlag + 1):
        r[l] = np.mean(e[: e.size - l] * e[l:]) / v if l < e.size else 0.0
    return r


def spectral_flatness(e: np.ndarray, nfft: int = 256):
    """Spectral flatness measure (geometric mean / arithmetic mean of PSD), in
    (0,1]; 1 == perfectly white, < 1 == colored.  Welch-style averaged."""
    e = e - e.mean()
    seg = nfft
    if e.size < seg:
        seg = 1 << int(np.floor(np.log2(e.size)))
    nseg = e.size // seg
    if nseg < 1:
        return 1.0, np.ones(seg // 2)
    win = np.hanning(seg)
    P = np.zeros(seg)
    for i in range(nseg):
        s = e[i * seg:(i + 1) * seg] * win
        P += np.abs(np.fft.fft(s)) ** 2
    P /= nseg
    P = P[: seg // 2] + 1e-30
    gm = np.exp(np.mean(np.log(P)))
    am = np.mean(P)
    return float(gm / am), P


def residual_isi(e: np.ndarray, x: np.ndarray, maxlag: int = 3):
    """Cross-correlation rho[l] = corr(e[k], x[k-l]) for l=-maxlag..maxlag (signed).
    Leftover ISI leaks neighbouring symbols into the residual."""
    e = e - e.mean()
    xs = (x - x.mean())
    out = {}
    se = np.std(e) + 1e-30
    sx = np.std(xs) + 1e-30
    for l in range(-maxlag, maxlag + 1):
        if l == 0:
            continue
        if l > 0:        # e[k] vs x[k-l]
            a = e[l:]; b = xs[:xs.size - l]
        else:            # e[k] vs x[k-l] = x[k+|l|]
            a = e[:e.size + l]; b = xs[-l:]
        n = min(a.size, b.size)
        out[l] = float(np.mean(a[:n] * b[:n]) / (se * sx))
    return out


def per_level_var(e: np.ndarray, idx: np.ndarray, n_levels: int):
    """Per-transmitted-level residual variance and count."""
    var = np.full(n_levels, np.nan)
    cnt = np.zeros(n_levels, dtype=int)
    for j in range(n_levels):
        m = idx == j
        cnt[j] = int(m.sum())
        if cnt[j] > 1:
            var[j] = float(np.var(e[m]))
    return var, cnt


# --------------------------------------------------------------------------- #
#  one structure point
# --------------------------------------------------------------------------- #
def measure_point(cfg: LinkConfig, label_extra: dict):
    r = run_link(cfg)
    e = r["residual"]; x = r["syms"][r["test"]]; idx = r["idx"][r["test"]]
    acf = autocorr(e, maxlag=6)
    sfm, _ = spectral_flatness(e)
    isi = residual_isi(e, x, maxlag=3)
    plv, pcnt = per_level_var(e, idx, cfg.n_levels)
    row = dict(group=f"PAM{cfg.n_levels}", n_levels=cfg.n_levels,
               pd_B_GHz=cfg.pd_B / 1e9, L_km=cfg.L_km, ROP_dBm=cfg.ROP_dBm,
               eq="DFE%d" % cfg.dfe_taps if cfg.dfe_taps > 0 else "FFE",
               ffe_taps=cfg.ffe_taps, dfe_taps=cfg.dfe_taps,
               snr_dB=r["snr_dB"], evm_pct=r["evm"] * 100, ser=r["ser"],
               sigma=r["sigma"],
               acf1=acf[1], acf2=acf[2], acf3=acf[3], sfm=sfm,
               isi_m1=isi[-1], isi_p1=isi[1], isi_m2=isi[-2], isi_p2=isi[2],
               isi_m3=isi[-3], isi_p3=isi[3],
               # per-level reliability spread (max/min variance ratio)
               lvl_var_ratio=float(np.nanmax(plv) / (np.nanmin(plv) + 1e-30)))
    if cfg.n_levels == 4:
        row.update(ber=r["ber"], ber_msb=r["ber_hi"], ber_lsb=r["ber_lo"])
        # per-bit residual reliability: variance of the soft LLR magnitude proxy ->
        # report the MSB/LSB hard-BER ratio (the LDPC-relevant unequal-bit metric)
        row["bit_ber_ratio"] = float(r["ber_lo"] / (r["ber_hi"] + 1e-30))
    else:
        row.update(ber=np.nan, ber_msb=np.nan, ber_lsb=np.nan, bit_ber_ratio=np.nan)
    row.update(label_extra)
    return row, r, acf, (plv, pcnt)


# --------------------------------------------------------------------------- #
#  quick SC-LDPC sanity decode on the IM/DD LLRs (PAM4 headline point)
# --------------------------------------------------------------------------- #
def sanity_decode_pam4(r):
    """Plug the PAM4 per-bit LLRs into a small windowed SC-LDPC decode to confirm
    the LLR bridge works on the IM/DD channel.  Uses the codec's own encoder to
    define the bit pattern, then *re-derives* LLR sign/scale from the measured
    residual statistics so the soft inputs are honest IM/DD reliabilities.

    Simpler robust check: drive the codec with random info, BPSK->our PAM4 levels,
    push through the SAME run_link statistics (sigma), and decode.  Here we do the
    lightweight version: take the measured per-bit BER -> equivalent AWGN sigma,
    build a frame, corrupt at that reliability, decode, report coded vs uncoded."""
    from nr_ldpc import NRLDPCCode
    from sc_ldpc import SCLDPCCode
    # small code
    Z, MP, w, L = 16, 24, 1, 12
    comp = NRLDPCCode(1, 0, Z, mp=MP)
    assign = np.arange(int((comp.B[:, :comp.Kb] >= 0).sum())) % (w + 1)
    sc = SCLDPCCode(comp, w=w, L=L, assign=assign)
    rng = np.random.default_rng(1)
    info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
    cw, _ = sc.encode(info)
    txbits = cw[sc.tx_mask]
    # map coded bits -> PAM4 (Gray) levels, run through one IM/DD frame at the SAME cfg
    cfg = r["cfg"]
    from imdd_channel import pam_levels, PAM4_BITS_BY_LEVEL, forward_channel, equalize
    from imdd_channel import soft_metrics_pam4
    lv = pam_levels(4)
    # bits -> level idx (invert the per-level bit table)
    b = txbits[: (txbits.size // 2) * 2].reshape(-1, 2)
    # match (hi,lo) to a level index
    lut = {(int(PAM4_BITS_BY_LEVEL[j, 0]), int(PAM4_BITS_BY_LEVEL[j, 1])): j for j in range(4)}
    lidx = np.array([lut[(int(bb[0]), int(bb[1]))] for bb in b])
    syms = lv[lidx]
    cfg2 = LinkConfig(**{**cfg.__dict__})
    cfg2.nsym = syms.size
    rx, _ = forward_channel(cfg2, syms)
    eq = equalize(cfg2, rx, syms, lv)
    yhat = eq["yhat"]
    te = eq["test"]
    sigma = float(np.sqrt(np.mean((yhat[te] - syms[te]) ** 2)))
    llr2, _, _ = soft_metrics_pam4(yhat, lv, sigma)        # full-length per-bit LLRs
    # build full LLR vector for the codec (only the trained/test bits are honest;
    # use the whole stream -- training region is data-aided so still valid bits)
    llr_bits = llr2[: txbits.size]
    llr = np.zeros(sc.num_var)
    llr[sc.tx_mask] = llr_bits
    llr[sc.known_mask] = 30.0
    hard = sc.decode_windowed(llr, W=6, max_iter=20, alpha=0.8)
    coded_err = int((sc.extract_info(hard) != info).sum())
    uncoded_err = int(((llr_bits < 0).astype(np.uint8) != txbits).sum())
    return dict(K=sc.K, coded_err=coded_err, coded_ber=coded_err / sc.K,
                uncoded_err=uncoded_err, uncoded_ber=uncoded_err / txbits.size,
                sigma=sigma, rate=sc.rate)


# --------------------------------------------------------------------------- #
#  the sweep grid + main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    NSYM = 8000
    rows = []
    acf_rows = []
    plv_rows = []
    headline = {}            # keep run dicts for plotting / LLR dump

    # Operating points chosen so each cell sits in the STRUCTURE-RICH regime
    # (non-trivial SER, not error-free, not a dead channel):
    #   B=40 GHz (main, O-band): FFE-only, ROP in the waterfall.
    #   B=30 GHz (severe, sub-0.4-Nyquist): PAM4 needs a DFE; PAM6 reported as the
    #     failing comparison (FFE-only to show the catastrophe + DFE attempt).
    # L in {2,10} km (O-band, D~=0 so CD negligible -> isolates bandwidth damage).

    grid = []
    for L in (2.0, 10.0):
        # --- B=40 GHz: FFE-only, both groups, ROP sweep ---
        for nl in (4, 6):
            for rop in (-12.0, -10.0, -8.0, -6.0):
                grid.append(dict(n_levels=nl, pd_B=40e9, L_km=L, ROP_dBm=rop,
                                 ffe_taps=41, dfe_taps=0))
        # --- B=30 GHz: PAM4 with DFE(6) (rescue) + FFE-only (catastrophe); PAM6 both ---
        for nl in (4, 6):
            for rop in (-2.0, 2.0, 6.0):
                grid.append(dict(n_levels=nl, pd_B=30e9, L_km=L, ROP_dBm=rop,
                                 ffe_taps=41, dfe_taps=6))
                grid.append(dict(n_levels=nl, pd_B=30e9, L_km=L, ROP_dBm=rop,
                                 ffe_taps=41, dfe_taps=0))

    print(f"=== IM/DD structure sweep: {len(grid)} points x {NSYM} sym ===")
    for gi, g in enumerate(grid):
        cfg = LinkConfig(nsym=NSYM, seed=0, **g)
        row, r, acf, (plv, pcnt) = measure_point(cfg, {})
        rows.append(row)
        print(f"  [{gi+1:2d}/{len(grid)}] {row['group']} B={row['pd_B_GHz']:.0f} "
              f"L={row['L_km']:.0f} ROP={row['ROP_dBm']:+.0f} {row['eq']:5s}: "
              f"SNR={row['snr_dB']:5.2f}dB SER={row['ser']:.2e} "
              f"acf1={row['acf1']:+.3f} SFM={row['sfm']:.3f} "
              f"isi+1={row['isi_p1']:+.3f} lvlVarRatio={row['lvl_var_ratio']:.2f}")
        # keep a few headline points for acf/per-level CSV + plots
        key = (row['group'], int(row['pd_B_GHz']), int(row['L_km']), row['eq'])
        # headline = the most structure-rich ROP we keep per (group,B,L,eq): pick mid SER
        if key not in headline or abs(row['ser'] - 1e-2) < abs(headline[key][0]['ser'] - 1e-2):
            headline[key] = (row, r, acf, plv, pcnt)

    # --- acf + per-level CSVs for headline points ---
    for key, (row, r, acf, plv, pcnt) in headline.items():
        for l in range(acf.size):
            acf_rows.append(dict(group=row['group'], pd_B_GHz=row['pd_B_GHz'],
                                 L_km=row['L_km'], eq=row['eq'], ROP_dBm=row['ROP_dBm'],
                                 lag=l, acf=acf[l]))
        for j in range(len(plv)):
            plv_rows.append(dict(group=row['group'], pd_B_GHz=row['pd_B_GHz'],
                                 L_km=row['L_km'], eq=row['eq'], ROP_dBm=row['ROP_dBm'],
                                 level=j, err_var=plv[j], count=int(pcnt[j])))

    # --- bandwidth penalty: SNR(B) vs SNR(ideal PD) at matched ROP ---
    print("\n=== bandwidth penalty (SNR gap vs wide-open IDEAL PD, L=2km) ===")
    bw_rows = []
    for nl in (4, 6):
        for rop, eqd in [(-10.0, 0), (2.0, 6)]:    # a B40-style and a B30-style ROP
            ref = run_link(LinkConfig(n_levels=nl, nsym=NSYM, pd_B=40e9, L_km=2.0,
                                      ROP_dBm=rop, ffe_taps=41, dfe_taps=eqd,
                                      pd_ideal=True))
            for B in (30e9, 40e9):
                r = run_link(LinkConfig(n_levels=nl, nsym=NSYM, pd_B=B, L_km=2.0,
                                        ROP_dBm=rop, ffe_taps=41, dfe_taps=eqd))
                pen = ref["snr_dB"] - r["snr_dB"]
                bw_rows.append(dict(group=f"PAM{nl}", ROP_dBm=rop,
                                    eq="DFE%d" % eqd if eqd else "FFE",
                                    pd_B_GHz=B / 1e9, snr_ideal_dB=ref["snr_dB"],
                                    snr_dB=r["snr_dB"], penalty_dB=pen, ser=r["ser"]))
                print(f"  PAM{nl} ROP={rop:+.0f} {('DFE%d'%eqd) if eqd else 'FFE':5s} "
                      f"B={B/1e9:.0f}GHz: SNR={r['snr_dB']:5.2f} vs ideal "
                      f"{ref['snr_dB']:5.2f} -> penalty {pen:5.2f} dB")

    # --- LLR bridge sanity decode on a headline PAM4 B=40 point ---
    print("\n=== SC-LDPC LLR-bridge sanity decode (PAM4, IM/DD LLRs) ===")
    sanity = None
    try:
        # pick a PAM4 B=40 L=2 FFE point in the waterfall (ROP=-10)
        rc = run_link(LinkConfig(n_levels=4, nsym=NSYM, pd_B=40e9, L_km=2.0,
                                 ROP_dBm=-10.0, ffe_taps=41, dfe_taps=0))
        sanity = sanity_decode_pam4(rc)
        print(f"  PAM4 B=40 ROP=-10 L=2  rate={sanity['rate']:.3f}  "
              f"uncoded BER={sanity['uncoded_ber']:.2e} -> coded BER={sanity['coded_ber']:.2e} "
              f"(K={sanity['K']})")
        # dump the per-bit LLRs (MSB/LSB) for this point so RL/Step-3 can plug in
        np.save(route("imdd_llr_pam4_b40_msb.npy"), rc["llr_hi"])
        np.save(route("imdd_llr_pam4_b40_lsb.npy"), rc["llr_lo"])
        print(f"  saved per-bit LLRs -> imdd_llr_pam4_b40_msb.npy / _lsb.npy")
    except Exception as ex:
        print("  [sanity decode skipped]", repr(ex))

    # --- write CSVs ---
    def write_csv(path, rws, fields):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
            for rw in rws:
                wr.writerow({k: rw.get(k, "") for k in fields})
        print(f"  wrote {path}")

    print()
    write_csv(route("imdd_structure.csv"), rows,
              ["group", "n_levels", "pd_B_GHz", "L_km", "ROP_dBm", "eq",
               "ffe_taps", "dfe_taps", "snr_dB", "evm_pct", "ser", "ber",
               "ber_msb", "ber_lsb", "bit_ber_ratio", "sigma",
               "acf1", "acf2", "acf3", "sfm",
               "isi_m1", "isi_p1", "isi_m2", "isi_p2", "isi_m3", "isi_p3",
               "lvl_var_ratio"])
    write_csv(route("imdd_acf.csv"), acf_rows,
              ["group", "pd_B_GHz", "L_km", "eq", "ROP_dBm", "lag", "acf"])
    write_csv(route("imdd_per_level.csv"), plv_rows,
              ["group", "pd_B_GHz", "L_km", "eq", "ROP_dBm", "level", "err_var", "count"])
    write_csv(route("imdd_bw_penalty.csv"), bw_rows,
              ["group", "ROP_dBm", "eq", "pd_B_GHz", "snr_ideal_dB", "snr_dB",
               "penalty_dB", "ser"])

    # --- plots ---
    make_plots(headline, rows, bw_rows)

    print(f"\nDONE in {time.time()-t0:.1f}s  ({len(rows)} sweep points)")
    return rows, headline, bw_rows, sanity


def make_plots(headline, rows, bw_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print("  [plots skipped]", repr(ex)); return

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    # (a) residual autocorrelation for a few headline points
    for key, (row, r, acf, plv, pcnt) in headline.items():
        if int(row['L_km']) != 2:
            continue
        lab = f"{row['group']} B{int(row['pd_B_GHz'])} {row['eq']} (SNR{row['snr_dB']:.0f})"
        ax[0, 0].plot(np.arange(acf.size), acf, "o-", ms=4, label=lab)
    ax[0, 0].axhline(0, color="k", lw=0.6)
    ax[0, 0].set_xlabel("lag"); ax[0, 0].set_ylabel("residual autocorrelation")
    ax[0, 0].set_title("(1) noise COLOR: residual ACF\n(off-zero = colored = exploitable)")
    ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=7)

    # (b) residual PSD for the same points
    for key, (row, r, acf, plv, pcnt) in headline.items():
        if int(row['L_km']) != 2:
            continue
        e = r["residual"]
        sfm, P = spectral_flatness(e)
        f = np.linspace(0, 0.5, P.size)
        ax[0, 1].plot(f, 10 * np.log10(P / P.mean()),
                      label=f"{row['group']} B{int(row['pd_B_GHz'])} {row['eq']} SFM={sfm:.2f}")
    ax[0, 1].set_xlabel("normalized freq (1=symbol rate)")
    ax[0, 1].set_ylabel("residual PSD [dB, norm]")
    ax[0, 1].set_title("(1) residual PSD\n(non-flat = colored noise enhancement)")
    ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize=7)

    # (c) per-level error variance
    for key, (row, r, acf, plv, pcnt) in headline.items():
        if int(row['L_km']) != 2:
            continue
        ax[0, 2].plot(np.arange(len(plv)), plv, "s-", ms=5,
                      label=f"{row['group']} B{int(row['pd_B_GHz'])} {row['eq']}")
    ax[0, 2].set_xlabel("transmitted PAM level index")
    ax[0, 2].set_ylabel("residual error variance")
    ax[0, 2].set_title("(3) UNEQUAL reliability: per-level error var\n(flat = equal AWGN; sloped = exploitable)")
    ax[0, 2].grid(alpha=0.3); ax[0, 2].legend(fontsize=7)

    # (d) residual ISI (cross-corr e[k] vs x[k+l]) for headline points
    for key, (row, r, acf, plv, pcnt) in headline.items():
        if int(row['L_km']) != 2:
            continue
        lags = [-3, -2, -1, 1, 2, 3]
        vals = [row[f"isi_m{abs(l)}"] if l < 0 else row[f"isi_p{l}"] for l in lags]
        ax[1, 0].plot(lags, vals, "o-", ms=4,
                      label=f"{row['group']} B{int(row['pd_B_GHz'])} {row['eq']}")
    ax[1, 0].axhline(0, color="k", lw=0.6)
    ax[1, 0].set_xlabel("symbol lag l  (corr e[k] vs x[k-l])")
    ax[1, 0].set_ylabel("cross-correlation")
    ax[1, 0].set_title("(2) residual ISI\n(non-zero = leftover ISI after FFE)")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)

    # (e) eff-SNR vs ROP for B=40 FFE, both groups, L=2
    for nl, mk in [(4, "o-"), (6, "s--")]:
        pts = [r for r in rows if r["n_levels"] == nl and r["pd_B_GHz"] == 40
               and r["L_km"] == 2 and r["eq"] == "FFE"]
        pts.sort(key=lambda r: r["ROP_dBm"])
        if pts:
            ax[1, 1].plot([p["ROP_dBm"] for p in pts], [p["snr_dB"] for p in pts],
                          mk, label=f"PAM{nl} B40 FFE")
    # B=30 DFE points
    for nl, mk in [(4, "^-"), (6, "v--")]:
        pts = [r for r in rows if r["n_levels"] == nl and r["pd_B_GHz"] == 30
               and r["L_km"] == 2 and r["eq"] == "DFE6"]
        pts.sort(key=lambda r: r["ROP_dBm"])
        if pts:
            ax[1, 1].plot([p["ROP_dBm"] for p in pts], [p["snr_dB"] for p in pts],
                          mk, label=f"PAM{nl} B30 DFE")
    ax[1, 1].set_xlabel("ROP [dBm]"); ax[1, 1].set_ylabel("effective SNR [dB]")
    ax[1, 1].set_title("(4) eff-SNR vs ROP\n(PAM4 vs PAM6, B=40 FFE / B=30 DFE)")
    ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=7)

    # (f) bandwidth penalty bars
    labels = [f"{r['group']}\nB{int(r['pd_B_GHz'])}\n{r['eq']}\nROP{r['ROP_dBm']:+.0f}"
              for r in bw_rows]
    pens = [r["penalty_dB"] for r in bw_rows]
    cols = ["C3" if r["pd_B_GHz"] == 30 else "C0" for r in bw_rows]
    ax[1, 2].bar(range(len(pens)), pens, color=cols)
    ax[1, 2].set_xticks(range(len(pens)))
    ax[1, 2].set_xticklabels(labels, fontsize=6)
    ax[1, 2].set_ylabel("bandwidth penalty [dB]\n(SNR vs ideal wide-open PD)")
    ax[1, 2].set_title("(4) BANDWIDTH PENALTY\n(red=30GHz, blue=40GHz)")
    ax[1, 2].grid(alpha=0.3, axis="y")

    fig.suptitle("IM/DD bandwidth-limited link: post-EQ effective-channel STRUCTURE "
                 "(PAM4 vs PAM6)", fontsize=13)
    fig.tight_layout()
    out = route("imdd_structure.png")
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
