# IEEE T-COM paper draft

**Title:** *Learning to Spread Edges: Reinforcement-Learning Construction of
Spatially-Coupled LDPC Codes*

First full draft for IEEE Transactions on Communications (journal, two-column
IEEEtran). Built from the blueprint in `../docs/TCOM_PLAN.md` and the verified
result tables in `../docs/RL_CONSTRUCTION.md` / `../docs/RESEARCH_ANALYSIS.md`.
All experimental numbers were cross-checked against the raw result files in
`../results/construct/` — none are fabricated.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | The paper. |
| `refs.bib` | Bibliography (BibTeX). |
| `IEEEtran.cls`, `IEEEtran.bst` | Vendored IEEE journal class + BibTeX style (self-contained). |
| `figs/*.pdf` | The 8 key figures, converted from `../figures/construct/*.svg`. |

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Last build: **9 pages**, clean (no undefined references, no overfull boxes).

## Figures

The 8 figures referenced (TCOM_PLAN §7) were converted from the repository SVGs
to PDF with `svglib`/`reportlab` (pure-Python; no `rsvg-convert`/`inkscape`/
native `cairo` were available on this machine). To regenerate:

```python
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
for f in ["exp_construct_ber","exp_construct_learn","exp_construct_rate_threshold",
          "exp_construct_rate_gain","exp_big_rl_vs_random","exp_wmc_rl_vs_random",
          "exp_construct_transfer","exp_wl_n4_vs_w"]:
    renderPDF.drawToFile(svg2rlg(f"../figures/construct/{f}.svg"), f"figs/{f}.pdf")
```

## Outstanding TODOs (experiments still needed)

These appear in `main.tex` both as `% TODO:` comments and as visible
`\textcolor{red}{[TODO: ...]}` markers (grep for `TODO`). They map to the two
"gap-filler" sections of TCOM_PLAN §5, which are **written as method + theory but
not yet experimentally completed**:

1. **TODO-1 / PEXIT-1** (§IV-D): Spearman correlation of the finite-length PEXIT
   threshold ordering vs. the Monte-Carlo waterfall-threshold ordering.
2. **TODO-2 / PEXIT-2** (§IV-D): train RL with the PEXIT reward, re-test the
   champion by Monte-Carlo, compare to the MC-reward champion.
3. **TODO-3 / PEXIT-3** (§IV-D): measure PEXIT evaluation cost vs. Monte-Carlo
   cost (sharpens the C3 criterion). Stub table: `tab:pexit-stub`.
4. **TODO-4 / LEMMA-num** (§V-C): numerical verification of the same-column ↔
   4-cycle lemma via `count_4cycles` across `Z ∈ {16,32,64}` (the measured
   930 → 0 cross-check). The lemma **statement + proof sketch are written**;
   only the dedicated numerical-check script is pending.
5. **TODO-5 / author** (title block): real author names, affiliation, e-mail,
   funding/acknowledgement.

Everything else (C1–C4 results: Tables `tab:c1`, `tab:ber`, `tab:learn`,
`tab:theta`, `tab:rate`, `tab:big`, `tab:wmc`, `tab:wl`, `tab:transfer`) is
backed by real data already in the repository.
