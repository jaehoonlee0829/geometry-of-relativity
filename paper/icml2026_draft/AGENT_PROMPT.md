# Agent Prompt For Continuing The ICML Paper

You are working in the local `geometry-of-relativity` repository.

Goal: improve the ICML 2026 Mechanistic Interpretability Workshop skeleton for
"The Geometry of Relativity: Context-Normalized Encoding of Gradable Adjectives
in LLMs".

Primary file: `paper/icml2026_draft/main.tex`

Supporting files:

- `docs/paper_outline.md`
- `APPENDIX.md`
- `docs/V12_RESULTS_SUMMARY.md`
- `docs/V12_1_RESULTS_SUMMARY.md`
- `docs/V12_2_RESULTS_SUMMARY.md`
- `paper/icml2026_draft/ICML_WRITING_TODO.md`
- `paper/icml2026_draft/REVIEW_RUBRIC.md`

Important: the current `main.tex` is a skeleton. Do not turn TODOs into confident
claims unless the supporting source file is identified. Keep prose light and mark
uncertainty explicitly.

Instructions:

1. Preserve ICML 2026 format. Do not modify `icml2026.sty`.
2. Keep `\usepackage{icml2026}` for blind submission.
3. Keep the main paper under 8 pages excluding references and appendix.
4. Keep the submission anonymous.
5. Use the conservative working thesis: models map relative concepts in
   in-context examples using a variable related to `z = (x - mu) / sigma`, but
   the mechanism may be mixed with lexical/output-facing components.
6. Do not claim a pure universal z-code, localized causal circuit, fully solved
   mechanism, or that a probe result equals mechanism.
7. Follow the structure in `main.tex`: phenomenon, setup, experimental design,
   evidence subsections, related work, limitations, conclusion, impact statement,
   appendix.
8. Active TODOs from the handcrafted Notion page must remain visible:
   robust scoring beyond naive logit diff, OOD/extreme values, shot-condition
   comparison, attention-pattern analysis, raw-x steering controls, objective
   controls, and cross-model/multimodal generalization.
9. Use `APPENDIX.md` for definitions and reproducibility details.
10. Run the paper against `REVIEW_RUBRIC.md` before declaring it improved.

Today's concrete target:

- Keep the skeleton coherent and compile-safe.
- Add or refine TODO comments, source-path hooks, and figure placeholders.
- Do not invent final claims or citations.
- Do not push to GitHub unless explicitly asked.
