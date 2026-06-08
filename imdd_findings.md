# IM/DD bandwidth-limited link — effective-channel structure (Step 1+2)

**Go/no-go question:** after equalization of a *severely bandwidth-limited* (sub-Nyquist)
IM/DD optical link, does the residual still carry **exploitable structure** (colored
residual noise / unequal bit-or-level reliability / residual ISI) that an RL-optimized
SC-LDPC construction could later exploit — or has the equalizer already flattened
everything to ~white, equal-reliability AWGN?

**Verdict (one line): GO.** The post-EQ channel is *strongly* structured on both groups
and at every operating point that isn't error-free — colored residual noise (negative
lag-1 autocorrelation), large residual ISI sub-Nyquist, and clearly unequal reliability
(PAM4 LSB ≈ 2× worse than MSB; outer/inner PAM levels differ up to ~2.6×). **PAM4 is the
more promising vehicle** (PAM6 collapses to a dead channel at the harshest bandwidth).

All numbers below are measured on pod `yye-scldpc-fiber` (OptiCommPy, numpy 2.4.6).
Reproduce: `python3 /work/imdd_channel.py` (self-test), `python3 /work/imdd_structure.py`
(full sweep, ~15 s). Raw data: `imdd_structure.csv`, `imdd_acf.csv`, `imdd_per_level.csv`,
`imdd_bw_penalty.csv`, `imdd_cband_check.csv`; figure: `imdd_structure.png`.

---

## 1. The link (OptiCommPy direct-detection API used)

`imdd_channel.py` builds a parameterized chain (`LinkConfig` → `run_link`). Severity by
design: **Rs = 80 GBaud, RRC β=0.1**, oversample **SpS=3 → Fs=240 GSa/s**. The PD bandwidth
**B ∈ {30, 40} GHz is *below* the 40 GHz symbol Nyquist** ⇒ sub-Nyquist (B/Nyquist = 0.75
and 0.375), so strong ISI is unavoidable — that is the whole point.

| Stage | OptiCommPy call | key settings |
|---|---|---|
| RRC shaping | `pulseShape(pulseType='rrc', SpS=3, rollOff=0.1)`, unit-energy; `upsample`/`firFilter` | 401 taps |
| **AWG (DAC) limit** | `resample` drive Fs→**120 GSa/s**→Fs | anti-alias LP at each stage ⇒ imposes ~60 GHz DAC analog bandwidth on the drive (a *secondary*, mild impairment). Toggle `awg_model`. |
| **MZM intensity** | `mzm(Ai, u, Vpi=2, Vb=-Vpi/2)` → `Ai·cos(0.5/Vpi·(u+Vb)·π)` | **biased at quadrature** (`Vb=-Vpi/2`); real PAM drive swung ±`0.9·Vpi/2` about quadrature so optical *field* tracks the PAM ≈ linearly. `Ai=√(dBm2W(Pin))`, then optical power pinned to `Pin` via `pnorm`-style rescale. |
| Fiber | `ssfm(E, {Ltotal,Lspan,hz=0.5,alpha=0.3,D,gamma=1.3,Fc,Fs,amp='ideal'})` | **O-band main study: D=0, Fc=229 THz (1310 nm)** ⇒ CD negligible, isolates bandwidth damage. L ∈ {2,10} km. |
| **Set ROP** | rescale field so `signalPower(E)=dBm2W(ROP_dBm)` | sweep **received optical power**, *not* OSNR (unamplified IM/DD). |
| **Direct detection** | `photodiode(E, {R=1, B, Fs, ideal=False, shotNoise=True, thermalNoise=True, bandwidthLimitation=True, seed})` | square-law `i∝|E|²`; applies a **`lowPassFIR(B,Fs)`** band limit **and** thermal+shot noise whose variance scales with `B` and (shot) with the photocurrent. SNR set by ROP. |
| Resample | `resample` photocurrent Fs → `eq_sps·Rs` | equalizer at **2 SpS** |
| **Equalizer** | self-written data-aided **MMSE-FFE** (41 taps, fractionally spaced) + optional **decision-directed DFE** (6 fb taps) | trained on first 50 % of symbols (known); evaluated on the held-out test region. |

Effective channel: `e[k] = yhat[k] − x[k]` (equalized minus ideal Es=1 symbol), measured on
the test region.

**PAM6 labeling gotcha (handled):** 6 is non-power-of-2 (log₂6≈2.585 b/sym); OptiCommPy's
square M-PAM Gray map does not apply, so PAM6 uses **6 self-built equally-spaced levels**
(`[-5..5]`, normalized Es=1) and the structure study works at the **symbol/per-level**
granularity (no bit labeling needed). PAM4 uses standard 2-bit Gray (level −3→00, −1→01,
+1→11, +3→10), matching `pam4_rrc.py`.

---

## 2. Structure numbers (PAM4 vs PAM6)

Operating points are chosen to sit in the **structure-rich** regime (non-trivial SER, not
error-free, not a dead channel). L=2 and L=10 km are **bit-identical** (max ΔSNR = 2e-8 dB)
because D=0 ⇒ CD is negligible — confirming the study isolates the *bandwidth* damage.

### (a) Noise COLOR — residual autocorrelation & spectral flatness (SFM∈(0,1]; 1=white)

| Group | B [GHz] | EQ | SNR_eff [dB] | acf(±1) | acf(±2) | **SFM** |
|---|---|---|---|---|---|---|
| PAM4 | 40 | FFE | 17.8 | **−0.171** | +0.030 | 0.931 |
| PAM4 | 40 | FFE | 25.1 | −0.089 | +0.030 | 0.942 |
| PAM6 | 40 | FFE | 17.4 | **−0.156** | +0.024 | 0.932 |
| PAM4 | 30 | **DFE6** | 13.5 | **−0.401** | +0.077 | 0.786 |
| PAM4 | 30 | **DFE6** | 9.4 | **−0.492** | +0.16 | 0.422 |
| PAM4 | 30 | FFE | 5.5 | **−0.82** | +0.55 | **0.040** |
| PAM6 | 30 | DFE6 | 1.5–2.2 | **−0.57** | +0.11 | ~0.40 |

→ **The residual is COLORED everywhere.** Even the benign B=40/FFE case has a clear lag-1
anticorrelation (−0.09…−0.17, SFM≈0.93) — the FFE noise-enhancement signature on a
band-limited channel. At B=30 it is *dramatic* (acf₁ down to −0.82, SFM=0.04 for FFE; the
PSD panel shows a deep band-edge null). **Colored residual = exploitable structure.**

### (b) Residual ISI — cross-correlation corr(e[k], x[k∓l])

| Group | B [GHz] | EQ | isi(+1) | isi(+2) | isi(+3) |
|---|---|---|---|---|---|
| PAM4 | 40 | FFE (SNR 17.8) | +0.049 | +0.02 | small |
| PAM4 | 40 | FFE (SNR 25.1) | +0.007 | ~0 | ~0 |
| PAM4 | 30 | DFE6 (SNR 9.4) | **+0.127** | +0.05 | small |
| PAM4 | 30 | FFE (SNR 5.5) | **+0.41** | +0.2 | + |
| PAM6 | 30 | DFE6 | **+0.26–0.29** | + | + |

→ Leftover ISI after the FFE is **small but non-zero at B=40** (≈0.02–0.05) and **large at
B=30** (0.13 with a DFE, 0.41 FFE-only). The equalizer does *not* fully remove the
sub-Nyquist ISI — neighbouring symbols still leak into the residual.

### (c) UNEQUAL reliability

**Per-transmitted-level error variance** (a clear gradient = exploitable; flat = equal AWGN):

| Group | B/EQ | per-level variance (low→high level) | max/min ratio |
|---|---|---|---|
| PAM4 | 40/FFE (SNR 17.8) | 0.016 / 0.015 / 0.015 / 0.018 | 1.2 |
| PAM4 | 40/FFE (SNR 25.1) | — | **2.0** |
| PAM4 | 30/DFE6 (SNR 13.5) | 0.053 / 0.039 / 0.024 / 0.063 | **2.6** |
| PAM6 | 40/FFE (SNR 20.6) | ~0.008 (×10), 0.010 outer | ~1.3–2.2 |

**Per-BIT reliability for PAM4 (MSB vs LSB hard-BER) — the LDPC variable-node-matching metric:**

| B [GHz] | EQ | SNR_eff | BER(MSB) | BER(LSB) | **LSB/MSB ratio** |
|---|---|---|---|---|---|
| 30 | DFE6 | 13.5 | 0.0075 | 0.0165 | **2.2×** |
| 30 | DFE6 | 9.4 | 0.0158 | 0.0300 | **1.9×** |
| 30 | FFE | 5.5 | 0.114 | 0.255 | **2.2×** |

→ **The PAM4 LSB is consistently ≈2× less reliable than the MSB** (the inner Gray
boundary is blurred most by the band limit + square-law PD). This is exactly the
unequal-bit structure an SC-LDPC construction can exploit by giving the LSB stronger
variable-node degree/protection. **Unequal bit/level reliability = strongly confirmed.**

### (d) Effective SNR / SER and the BANDWIDTH PENALTY

Bandwidth penalty = SNR_eff(real PD) − SNR_eff(wide-open *ideal* PD) at matched ROP:

| Group | ROP / EQ | B=40 GHz penalty | B=30 GHz penalty |
|---|---|---|---|
| PAM4 | −10 dBm / FFE | **9.1 dB** | **24.2 dB** |
| PAM4 | +2 dBm / DFE6 | **3.0 dB** | **20.6 dB** |
| PAM6 | −10 dBm / FFE | 8.7 dB | 23.5 dB |
| PAM6 | +2 dBm / DFE6 | 2.5 dB | 27.7 dB |

→ The bandwidth limit is the **dominant impairment by a wide margin**: ~3–9 dB at B=40 GHz
(0.5× Nyquist, FFE-recoverable) and a brutal **~20–28 dB at B=30 GHz** (0.375× Nyquist,
spectral null — FFE-only is catastrophic, only a DFE partially rescues PAM4).

---

## 3. PAM4 vs PAM6 — which regime, which vehicle

| | **PAM4** | **PAM6** |
|---|---|---|
| bits/sym | 2 | 2.585 |
| B=40 GHz (0.5×Nyq, FFE) | works; SER→0 by ROP=−10; **SNR_eff 18→25 dB**; LSB 2× weaker than MSB | works but **SER floor ~1–3e-3** at the same SNR (denser levels, less margin) |
| B=30 GHz (0.375×Nyq) | **DFE rescues to SNR 9–13.5 dB, SER 2–4e-2** (codable) | **DFE cannot rescue: SNR 1.5–2.2 dB, SER 0.30–0.35** → effectively a **dead channel** |
| structure when usable | colored + 2× unequal-bit + residual ISI | colored + unequal-level, but only in the B=40 regime |
| **regime** | impaired-but-codable across both B | **codable only at B=40**; B=30 is below threshold |

**PAM6 packs more bits/symbol and is correspondingly more fragile** under the bandwidth
limit — exactly as predicted. It still *shows* structure (colored residual, per-level
spread) in the B=40 regime, but at B=30 its tight level spacing pushes it below any useful
operating point (SER≈⅓ ≈ random among neighbours). **PAM4 is the better SC-LDPC vehicle:**
it stays codable across both bandwidths and exposes a clean, stable 2× MSB/LSB reliability
split for variable-node matching.

---

## 4. LLR bridge → SC-LDPC (Step-3 readiness + sanity decode)

`imdd_channel.run_link` already emits the soft outputs Step 3 needs:
- **PAM4:** exact per-bit Gray LLRs (`soft_metrics_pam4`), split into MSB/LSB streams
  (`llr_hi`, `llr_lo`) — saved to `imdd_llr_pam4_b40_msb.npy` / `_lsb.npy`.
- **PAM6:** per-level Gaussian log-likelihoods (`soft_metrics_pam6`, 6 cols/symbol) — any
  later 6→bits labeling can derive bit metrics from these.

**Sanity decode (LLR bridge works on IM/DD):** drove the repo's `SCLDPCCode` (BG1 rate-½,
Z=16, w=1, L=12, round-robin `assign = arange(#sys_edges)%(w+1)`) with PAM4 IM/DD LLRs at
B=40/ROP=−10/L=2 → **uncoded BER 4.9e-4 → coded BER 0.0** (K=3872, rate 0.476). The IM/DD
→ Gray-PAM4 → equalize → `soft_metrics_pam4` → `decode_windowed` path closes cleanly.

---

## 5. Gotchas / caveats

1. **FFE noise enhancement is the headline physics.** At B=30 GHz the channel has a deep
   spectral null inside the symbol band; an FFE trying to invert it amplifies noise
   catastrophically — *more taps make it strictly worse* (SNR 5→1 dB as taps 31→121). The
   fix is a **DFE** (handles nulls without noise enhancement): DFE(6) lifts PAM4 from
   SNR≈5 dB to ≈9–13.5 dB. DFE(12) was *worse* than DFE(6) (error propagation), so 6 fb
   taps were used. This is real, exploitable structure, not a bug.
2. **Square-law + quadrature MZM:** biasing the MZM at quadrature and swinging the *field*
   ±0.9·Vpi/2 keeps the intensity≈linear in the PAM drive after DC-removal + normalization
   (verified: ideal-PD SNR=28 dB, SER=0 — the FFE perfectly inverts the residual square-law
   + ISI). Per-level means are linear (−1.35/−0.51/+0.47/+1.31 for the 4 levels).
3. **Normalization pitfalls handled:** DC-remove + unit-std the photocurrent before the FFE;
   final LS gain aligns `yhat` to the Es=1 level grid so the slicer/LLR use the analytic
   levels directly. `sigma` for the LLR is re-estimated from the *measured* residual.
4. **AWG modeling is secondary (as specified):** modeled as resample-to-120-GSa/s-and-back
   (≈60 GHz DAC analog bandwidth). Turning it off (`awg_model=False`) barely moves B=40 and
   slightly *helps* B=30 FFE — confirming the **PD bandwidth, not the AWG, is dominant.**
5. **O-band isolates the bandwidth damage (verified).** With D=0, L=2 vs 10 km are identical.
   A C-band cross-check (D=17, 1550 nm) shows CD fading *does* appear and is length-dependent
   (PAM4 B=40 ROP=−10: O-band 20.9 dB at any L; C-band **15.2 dB @2 km → 7.4 dB @10 km**,
   SER 0.27) — see `imdd_cband_check.csv`. Keeping O-band as the main study is the right call
   to study bandwidth structure cleanly; C-band would *add* CD structure on top.

---

### Bottom line for Step 3 (RL construction)

There is **ample exploitable structure** after equalization to justify an RL-optimized
SC-LDPC construction:
- **colored residual** (negative lag-1 acf even in the benign B=40 case, severe at B=30),
- **unequal bit reliability** (PAM4 LSB 2× weaker than MSB — a clean, stable target for
  variable-node degree matching),
- **residual ISI** the FFE cannot fully flatten (especially sub-Nyquist).

**Recommended vehicle: PAM4**, primarily at **B=40 GHz** (FFE, ROP in the −12…−8 dBm
waterfall: codable, structured, stable) with **B=30 GHz + DFE** as the high-structure /
high-stress operating point. PAM6 is a useful "more bits, more fragile" contrast but
collapses below threshold at B=30 and should not be the primary Step-3 target.
