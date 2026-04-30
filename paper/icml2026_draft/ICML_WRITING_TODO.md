# ICML Writing TODO

Target: ICML 2026 Mechanistic Interpretability Workshop long paper.

Deadline: May 8, 2026 AOE (May 9, 2026 20:59 KST).

Working draft: `paper/icml2026_draft/main.tex`

## Today: Skeleton Paper

- [ ] Open `main.tex` and confirm the ICML style loads from `icml2026.sty`.
- [ ] Replace every section TODO with 1-3 sentence rough prose.
- [ ] Port only the strongest claims from `docs/paper_outline.md`.
- [ ] Add placeholder figure slots for the 4-6 main figures.
- [ ] Add real bibliography entries to `references.bib`.
- [ ] Keep the main body within the ICML long-paper target: 8 pages excluding references and appendix.
- [ ] Add appendix headings for prompt templates, hyperparameters, ablations, full tables, and source paths.

## Main Claim Order

Use this order unless a later review finds a better narrative:

1. Behavioral evidence: logit differences track context-normalized `z = (x - mu) / sigma`.
2. Representation evidence: residual-stream geometry contains z-like directions across pairs.
3. Transfer evidence: a partly shared direction supports cross-pair steering/transfer.
4. Temporal evidence: z is encoded early and becomes causally potent later.
5. Decomposition evidence: lexical/output-facing components explain much of the diagonal effect, while residualized directions transfer better off-diagonal.
6. SAE evidence: z-correlated sparse features exist, but the population is mixed rather than pure.

## Main Figure Budget

Pick 4-6 for the main body:

- [ ] Figure 1: task schematic and x/mu/z grid.
- [ ] Figure 2: behavioral z-signal across adjective pairs and models.
- [ ] Figure 3: representation geometry or PCA/probe summary.
- [ ] Figure 4: shared direction and cross-pair transfer matrix.
- [ ] Figure 5: encode-early/use-late layer sweep.
- [ ] Figure 6: lexical projection vs residual transfer decomposition.

Everything else goes to appendix unless it is needed to defend a headline claim.

## Claim Ledger

Before a number enters the paper, record:

- [ ] Exact claim text.
- [ ] Source file or script that produced the number.
- [ ] Result artifact path.
- [ ] Figure/table destination.
- [ ] Caveat or failure mode.
- [ ] Whether the claim is main-body or appendix-only.

## Hard No List

- Do not claim a pure universal non-lexical z-code.
- Do not claim a localized causal circuit.
- Do not hide lexical confounding.
- Do not imply BMI/absolute controls cleanly separate from relative adjectives unless the current evidence supports it.
- Do not include author names, affiliations, GitHub usernames, HuggingFace usernames, acknowledgments, or identifying paths in the blind submission.

## Pre-Submission Checklist

- [ ] PDF uses ICML 2026 style.
- [ ] Main body is within 8 pages.
- [ ] References and appendix are in the same PDF.
- [ ] All figures are readable in two-column format.
- [ ] Figure captions state one claim each.
- [ ] Every strong claim has a limitation nearby.
- [ ] Related work distinguishes behavioral pragmatics/numerical representation from this mechanistic study.
- [ ] Anonymous code link is ready through anonymous.4open.science.
- [ ] At least one reciprocal reviewer is ready for OpenReview.

