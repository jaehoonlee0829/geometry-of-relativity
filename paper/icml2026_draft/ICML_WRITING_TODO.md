# ICML Writing TODO

Target: ICML 2026 Mechanistic Interpretability Workshop long paper.

Deadline: May 8, 2026 AOE (May 9, 2026 20:59 KST).

Working draft: `paper/icml2026_draft/main.tex`

## Current Principle

This is a **skeleton**, not submission prose. Every visible paragraph in `main.tex`
must be rewritten by a human before submission. The skeleton exists to preserve
paper structure, claim hygiene, ICML syntax, and research TODOs.

## Today: Skeleton Paper

- [ ] Keep the ICML template compiling.
- [ ] Use `main.tex` as the Overleaf main file.
- [ ] Preserve the high-level skeleton unless a human changes the story.
- [ ] Fill sections with light prose only after claim decisions are made.
- [ ] Keep unresolved claims as comments/TODOs, not confident prose.
- [ ] Add placeholder figure slots for 4-6 main figures.
- [ ] Keep the main body within 8 pages excluding references and appendix.
- [ ] Add appendix hooks for prompt templates, hyperparameters, ablations, full tables, and source paths.

## Recommended Paper Shape

This follows recent ICML/NeurIPS empirical papers and recent mechanistic interpretability papers:

1. **Introduction**
   - concrete phenomenon first
   - mechanistic question
   - provisional thesis
   - contribution map
2. **Background and Problem Setup**
   - gradable adjectives
   - variables `x`, `mu`, `sigma`, `z`
   - evidence standards
3. **Experimental Design**
   - models
   - prompts / shot conditions
   - adjective concepts
   - measurements
4. **Evidence for Context-Normalized Relativity**
   - behavioral grids
   - representation geometry
   - layer evolution
   - shared direction
   - lexical composition
   - SAE / attention TODOs
5. **Related Work and Positioning**
   - gradable adjectives / pragmatics
   - numerical representations / in-context statistics
   - LRH / steering / manifold geometry
6. **Discussion and Limitations**
   - what can be claimed
   - what cannot be claimed
   - remaining TODOs
7. **Conclusion**
8. **Impact Statement**
9. **Appendix**

## Active Research TODOs From Handcrafted Notion Page

- [ ] Robust scoring beyond naive single-token logit difference.
- [ ] OOD / extreme value experiments, e.g. implausible heights.
- [ ] Zero-shot vs 2-shot vs 15-shot comparison: does the model act like a data scientist over context?
- [ ] Raw-`x` steering controls.
- [ ] More objective or semantically distant control concepts.
- [ ] Full attention-pattern analysis for the end-to-end relativity computation.
- [ ] Cross-model / cross-family generalization if feasible.
- [ ] Multimodal/Gemma 4 generalization only if it becomes feasible without hurting the main paper.

## Candidate Claim Space

Do not lock the key claim until literature review and TODO experiments settle novelty.
Candidate framings:

- Models map relative scalar concepts in context using a variable related to `z = (x - mu) / sigma`.
- Residual-stream geometry contains a context-normalized relativity component.
- A partly shared direction transfers across adjective concepts, but the mechanism is mixed.
- Lexical/output-facing components explain part of the effect and must be separated from any domain-general claim.
- Shot-condition experiments may reveal whether the model infers comparison-class statistics from examples.

## Figure Budget

Pick 4-6 for the main body:

- [ ] Figure 1: task schematic and x/mu/sigma/z grid.
- [ ] Figure 2: behavioral z-signal across adjective pairs and models.
- [ ] Figure 3: representation geometry / PCA / probe summary.
- [ ] Figure 4: layer evolution: readout early vs steering later.
- [ ] Figure 5: shared direction and cross-pair transfer.
- [ ] Figure 6: lexical projection vs residual transfer.

Everything else goes to appendix unless needed to defend the abstract.

## Claim Ledger

Before a number enters the paper, record:

- [ ] exact claim text
- [ ] source file or script
- [ ] result artifact path
- [ ] figure/table destination
- [ ] caveat or failure mode
- [ ] main-body or appendix-only

## Hard No List

- Do not claim a pure universal non-lexical z-code.
- Do not claim a localized causal circuit.
- Do not hide lexical confounding.
- Do not treat probe results as mechanisms.
- Do not treat PCA plots as proof.
- Do not imply BMI or absolute controls cleanly separate unless the final evidence supports it.
- Do not include author names, affiliations, GitHub usernames, HuggingFace usernames, acknowledgments, or identifying paths in the blind submission.

## Overleaf To GitHub Sync

Use this after serious Overleaf work:

1. Overleaf: **Menu -> Download -> Source**.
2. Save the downloaded zip.
3. Unzip into a temporary folder.
4. Copy source into `paper/icml2026_draft/`.
5. Do not commit generated files: `.aux`, `.log`, `.out`, `.bbl`, `.blg`, `.synctex.gz`, compiled PDFs.
6. Review diff before commit:

```bash
cd <repo-root>
git checkout paper/icml-workshop-scaffold
git status
git diff -- paper/icml2026_draft
```

Commit only after human approval.
