# Review Rubric

Use this before asking a mentor, Claude, Codex, or yourself to review the paper.

## First Rule

This draft is scaffold prose. Do not judge it as final writing until a human has
rewritten it. Reviewers should focus first on structure, claim discipline, and
missing evidence.

## ICML Format Review

- [ ] Uses `\usepackage{icml2026}` for anonymous submission, not `[accepted]`.
- [ ] Does not edit `icml2026.sty` or compress margins/spacing.
- [ ] Main body is at most 8 pages for the workshop long-paper submission.
- [ ] References and appendix are included in the same PDF.
- [ ] Includes an unnumbered `Impact Statement` before references.
- [ ] No acknowledgments in anonymous submission.
- [ ] Submission is blind: no author names, affiliations, grants, GitHub/HF usernames, public repo URLs, or identifying paths.
- [ ] Abstract is one paragraph, roughly 4-6 sentences.
- [ ] Figures have captions below; tables have captions above.
- [ ] Figures are readable in two-column format.
- [ ] Plot titles are not duplicated inside figures if captions already state the claim.
- [ ] PDF size follows current ICML author instructions; prefer author instructions over example-paper discrepancy.

## Structure Review

- [ ] Paper starts with a concrete phenomenon, not abstract theory.
- [ ] Problem setup defines `x`, `mu`, `sigma`, `z`, and logit difference before using them.
- [ ] Evidence types are separated: behavior, readout, geometry, causality, decomposition.
- [ ] Results are organized as subsections under a main evidence/results section.
- [ ] Related work positions the paper against closest prior work, not as a literature dump.
- [ ] Limitations are visible in the main text, not buried only in appendix.
- [ ] Appendix has prompts, hyperparameters, full tables, extra controls, failed/mixed results, and APPENDIX.md definitions.

## Mechanistic Interpretability Review

- [ ] The work is not only behavioral: it makes a representation-level claim.
- [ ] Geometry/PCA is treated as descriptive evidence, not proof.
- [ ] Probe results are framed as availability, not mechanism.
- [ ] Steering/interventions are separated from correlational evidence.
- [ ] Lexical confounds are directly addressed.
- [ ] SAE evidence is framed as support/texture unless pure features are actually shown.
- [ ] Attention analysis is marked TODO/future unless a real mechanism is established.
- [ ] The paper is explicit about what is not reverse-engineered.

## Novelty Review

- [ ] Related work covers gradable adjectives and pragmatic thresholding.
- [ ] Related work covers numerical representations and in-context statistical inference.
- [ ] Related work covers LRH, steering, manifolds, and sparse features.
- [ ] The novelty claim is specific and narrow.
- [ ] Similar work is not caricatured.
- [ ] The final claim is chosen only after literature review and active TODO experiments.

## Evidence Review

- [ ] Each headline number has a source path.
- [ ] Each headline claim has a caveat.
- [ ] Each main figure has one clear job.
- [ ] Controls include raw-x baselines or steering where relevant.
- [ ] OOD/extreme values are either completed or clearly marked future work.
- [ ] Zero/few/many-shot comparison is either completed or clearly marked future work.
- [ ] Negative/mixed results improve the framing rather than being hidden.

## Reviewer Questions

1. Why is this not just behavioral pragmatics?
2. Why is this not just a probe finding?
3. How do we know the effect is context-normalized rather than raw magnitude or lexical bias?
4. What evidence is descriptive geometry, and what evidence is causal?
5. Does any shared direction transfer across adjective concepts?
6. How much of the effect is lexical/output-facing?
7. What remains unknown about the end-to-end mechanism?
8. Why is this a good fit for the ICML Mechanistic Interpretability Workshop?

