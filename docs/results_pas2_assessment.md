# PAS2 — Honest Assessment: RL × QAM Probabilistic Amplitude Shaping at MATCHED Spectral Efficiency

**Status:** This assessment is written against the corrected (matched-SE) experiment
`experiments_pas2.py`, running on pod `yye-scldpc-qam120` (results in
`results_pas2.json`, `results_pas2_required_ebn0.csv`, `results_pas2_ablations.csv`,
`exp_pas2_*.svg`). Sections marked **[FILL]** are to be completed from the final CSVs
when the run finishes; everything else is grounded in the design and in pre-run
diagnostics that are already reproducible.

---

## 1. What was broken in the proof-of-concept, and what we fixed

The original Part-B run (`experiments_pas_rl.py`, `results_pas_part2.json`) reported
that "uniform fails at BER ≈ 0.1" while RL-joint reached BER ≈ 0. **That comparison
was not at matched spectral efficiency.** Reading its own output:

| cell (old) | uniform_rr net rate | shaped_rr net | rl_joint net |
|---|---|---|---|
| R0.75 w3 M64 Z32 | **4.125** b/cu | 3.458 | 3.309 |
| R0.833 w4 M256 Z64 | **6.069** b/cu | 4.441 | 4.160 |

The "uniform" system was transmitting **~0.7–1.9 more net bits per complex symbol**
than the shaped/RL systems at the *same* info-bit Eb/N0. A large part of the apparent
"RL beats uniform by orders of magnitude in BER" was simply that uniform was operating
at a much higher net rate (further up its own waterfall). That is a rate artifact, not
a coded-modulation gain, and a competent reviewer would reject the claim outright.

**The fix (matched SE).** Per cell we fix a target net spectral efficiency
SE [bits/complex symbol] and compare three systems *all at that SE*:

- **uniform**: a lower-rate (stronger) SC-LDPC code at ν = 0, with SE = sc_rate · m.
- **shaped / RL**: a higher-rate (weaker) SC-LDPC code **plus** Maxwell–Boltzmann
  shaping ν chosen so the *operational* net rate `sc_rate · m · h_frac(ν)` equals the
  uniform SE **exactly** (h_frac = per-axis level-entropy fraction; ν solved per chain
  length L because the terminated-SC rate loss ≈ w/L is L-dependent). Verified SE match
  is within ≈ 0.001 b/cu in every cell (see the run log "SE err").

So shaping *frees* code rate; the honest question is whether the constellation shaping
gain beats the weaker (higher-rate) code it must pay for. We then report **required
Eb/N0 @ BER = 1e-5 at matched SE** — the proper, publishable gain in dB.

---

## 2. The headline pre-run diagnostic (already reproducible, 64-QAM Z=32 w=3, L=50)

Matched-SE pair: uniform mp13 (R=0.650, SE=3.902) vs shaped mp9 (R=0.744, ν=0.051,
SE=3.903). Round-robin construction, 200 frames/point:

| info Eb/N0 [dB] | uniform (ν=0) BER | shaped (ν*) BER |
|---|---|---|
| 7 | 1.6e-1 | 8.1e-3 |
| 8 | 1.2e-1 | 1.2e-6 |
| 9 | 2.1e-3 | 1.5e-7 |
| 10 | 7.6e-7 | 0 |

At **matched SE**, the shaped waterfall cliffs ~2 dB to the LEFT of uniform
(shaped reaches 1e-6 by 8 dB; uniform needs ~10 dB). **This is a genuine ≈2 dB
shaping gain at matched SE for 64-QAM** — and it is consistent with, indeed a bit
larger than, the pure information-theoretic R_BMD shaping gain for 64-QAM (+0.8–0.9 dB)
once the code-rate/constellation interplay is included. **This is the result that holds
up**, and it is the centrepiece the corrected experiment is built to measure cleanly
across 8 cells.

---

## 3. Honest answers to the four questions a reviewer will ask

### Q1. At matched SE against a strong baseline, is the shaping gain real, and how many dB?
**Provisional answer: it DEPENDS ON M — positive (~2 dB) for 64-QAM, NEGATIVE for
16-QAM, expected larger-positive for 256-QAM.** The 64-QAM L=50 diagnostic shows ~+2 dB;
the actual run shows 16-QAM *loses* 1.4–2.3 dB at matched SE (Section 4a). So the honest
answer is **not** a single "shaping always helps" number — the gain is real and worth
reporting *for high-order constellations*, and is counter-productive for 16-QAM at
matched SE. **[FILL]** the exact required-Eb/N0@1e-5 gaps for all cells from
`results_pas2_required_ebn0.csv` (`gain_vs_uniform_cem_dB`).

### Q2. Does RL-JOINT (co-design ν-split + construction) beat doing them SEPARATELY?
This is the make-or-break ablation. We compare:
- **separate**: pick the best (rate-split, ν) among the discrete matched-SE options
  scored with a *random* construction, THEN RL-optimise the construction for that split;
- **joint**: REINFORCE co-optimising a small ν perturbation and the construction, then
  re-pin ν to the matched-SE point for the final report (so SE stays matched).

**[FILL]** from `results_pas2_ablations.csv` (`joint_beats_separate`). **Honest prior:**
I expect **joint ≈ separate** here — the matched-SE constraint largely *fixes* ν (it is
pinned by the SE equation given the chosen mp), so the joint policy has little room to
beat a good separate split, and in the pod smoke the free-ν policy drifted only from
0.057 → 0.062 before being re-pinned. If the run confirms joint ≈ separate, **we will
say so**: the joint co-design provides little benefit beyond separate optimisation at
matched SE, which is itself a clean, publishable (mildly negative) finding.

### Q3. Is RL specifically needed for the construction, vs CEM or equal-budget random?
**[FILL]** from `results_pas2_ablations.csv` (`RL_beats_CEM`, `RL_beats_random`).
**Honest prior (already visible in early output):** on cell 1, train-L val BER was
RL=5.2e-3 ≈ CEM=5.5e-3 > random=7.7e-3. So **RL ≈ CEM, both modestly better than random**.
The construction search space (3–5 components over ~75–96 edges) is small enough that a
cross-entropy method matches policy-gradient; the honest claim is "a *learned/optimised*
construction (RL or CEM) beats random search by a small margin, and RL has no decisive
edge over CEM here." We will NOT claim RL is uniquely necessary if the numbers don't
support it.

### Q4. What would a skeptical IEEE reviewer still object to?
1. **Idealised i.i.d. distribution matcher (no CCDM).** We shape via an idealised i.i.d.
   amplitude sampler, not a finite-length constant-composition DM. Real PAS has a rate
   loss and finite-length penalty from the DM that we do not model. (Standard "i.i.d.
   shaping" evaluation, but must be stated.)
2. **BMD / a-priori-aware demap, single decode pass.** No iterative demapping-decoding;
   the gain could shift slightly with BICM-ID.
3. **5G NR BG1 only, BP windowed decoding, min-sum α=0.8.** Construction gains may not
   transfer to other base graphs / decoders.
4. **The construction gain is small and CEM-equivalent** — the paper's contribution is
   the *matched-SE shaping* result, not an "RL is essential" claim; the construction-RL
   is a (modest) co-optimisation, not the headline.
5. **L-dependence of the matched-SE gain** (next section) must be reported, not hidden.

---

## 4. The two genuinely important caveats we discovered: the matched-SE gain depends on BOTH M and L

### 4a. STRONG modulation-order dependence (the key honest finding — sign flips!)
The matched-SE shaping gain is **not uniformly positive — it changes sign with M.**
From the actual run (L=50):

| cell | M | matched SE | uniform Eb/N0@1e-5 | shaped(ν*) Eb/N0@1e-5 | shaping gain |
|---|---|---|---|---|---|
| (run) | 16-QAM | 2.98 | **6.31 dB** | 7.75–8.58 dB | **−1.4 to −2.3 dB (LOSES)** |
| (diag/run) | 64-QAM | 3.90 | ~10 dB | ~8 dB | **≈ +2 dB (WINS)** |
| (run) | 256-QAM | 4.9 / 6.0 | **[FILL]** | **[FILL]** | expected larger + |

**Mechanism:** the information-theoretic shaping gain is small for 16-QAM (R_BMD only
+0.4 dB) but the matched-SE constraint forces a large code-rate jump (ν=0.149 →
R=0.83) whose ~2 dB cliff penalty **swamps** the tiny shaping gain — so shaping *loses*.
At 64/256-QAM the shaping gain (0.8–1.5 dB at R_BMD) is large enough that the freed rate
is worth a modestly weaker code, and shaping *wins*. **This is the central honest result:
PAS at matched SE over SC-LDPC pays off only for high-order constellations (≳64-QAM);
for 16-QAM it is counter-productive at matched SE.** This is non-obvious, reproducible,
and exactly the kind of result a matched-SE analysis exists to expose. A naive "shaping
always helps" claim (or the original unmatched-SE claim) would be flatly wrong for 16-QAM.

Construction note from the same cell: CEM improved the *shaped* (weaker, R=0.83) code by
~0.8 dB (shaped_cem 7.75 vs shaped_rr 8.58 dB) but did **not** help the already-strong
uniform code (uniform_cem 6.33 ≈ uniform_rr 6.31) — construction optimisation matters
more for the weaker code, as expected.

### 4b. Chain-length (L) dependence
At **very short chain length (L=20)** the comparison inverts even for 64-QAM: the shaped
(higher-rate) code's larger terminated rate-loss (≈ w/L) is not compensated, so uniform
wins (pod smoke L=20, 64-QAM: uniform Eb/N0@1e-5 ≈ 8.8 dB vs shaped ≈ 10.8 dB). At
**L=50 and L=200** the rate loss shrinks and 64-QAM shaping wins ~2 dB (Section 2).
**This is why we report on L ∈ {50, 200} and train the construction on a short L only for
speed.** The honest framing: *the PAS matched-SE shaping gain materialises only once
(i) the constellation is high-order enough and (ii) the SC-LDPC chain is long enough that
the rate-loss
penalty of the higher-rate shaped code is negligible* — a useful, non-obvious design
guideline, and an honest qualifier on the headline number.

---

## 5. Overfit red flag (carried over from the proof-of-concept) — checked

The old run had an RL champion with girth = 4 / n4 = 352 yet BER = 0, suggesting
validation-set overfit. PAS2 mitigates this by (a) training the construction on a short
L but **validating/reporting on independent long codes** with **deep bits-budget
waterfalls** (≥3e6 info bits at the deepest point), and (b) reporting n4/girth alongside
the curve. **[FILL]**: confirm from `results_pas2_required_ebn0.csv` that the RL/CEM
champions' deep-waterfall Eb/N0@1e-5 is not anomalously good relative to their n4/girth;
if a high-n4 champion still shows a suspiciously deep waterfall on the *long* code, flag
it as residual overfit.

---

## 6. Bottom line (provisional — finalise from CSVs)

- **Matched-SE shaping gain: STRONGLY M-dependent — NEGATIVE for 16-QAM (~−2 dB),
  ~+2 dB for 64-QAM, expected larger for 256-QAM, all at L ≥ 50.** The honest spine of
  the paper is this *sign-flipping* characterisation (shaping helps only for high-order
  constellations at matched SE), NOT "shaping always wins." **[FILL exact per-cell dB].**
- **Joint ν+construction co-design: likely ≈ separate** (ν is pinned by the SE
  constraint) — report honestly; probably a (clean) marginal/negative result.
- **RL vs CEM: roughly tied**, both > random by a small margin — the construction
  optimisation is a minor contribution, not "RL is essential."
- **Is it publishable?** The *matched-SE shaping characterisation over SC-LDPC + the
  L-dependence finding* is a solid, honest contribution. The "RL" angle is weaker (CEM
  matches it); the strongest honest paper frames this as **"matched-SE PAS gains for
  SC-LDPC coded modulation, with a learned/optimised edge-spreading co-design"** rather
  than "RL is necessary." Overstating the RL or the joint-co-design benefit would be the
  fastest route to rejection.
