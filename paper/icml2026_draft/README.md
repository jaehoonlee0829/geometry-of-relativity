# ICML 2026 Draft

This folder is a working ICML 2026-format draft for the Geometry of Relativity paper.

This is an authoring workspace. The submitted PDF should be generated from
`main.tex`; helper markdown files are for coordination and should not be treated
as submission prose.

Official files were downloaded from:

- ICML author instructions: https://icml.cc/Conferences/2026/AuthorInstructions
- Style zip: https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip
- Example paper: https://media.icml.cc/Conferences/ICML2026/Styles/example_paper.pdf
- Workshop CFP: https://mechinterpworkshop.com/cfp/
- Workshop OpenReview: https://openreview.net/group?id=ICML.cc/2026/Workshop/Mech_Interp

Compile from this folder with:

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Submission target: ICML Mechanistic Interpretability Workshop long paper, max 8 pages excluding references and appendix.

Companion docs:

- `ICML_WRITING_TODO.md` - writing checklist and figure/claim plan.
- `REVIEW_RUBRIC.md` - format and semantic review checklist.
- `AGENT_PROMPT.md` - prompt for a future Codex/Claude pass.

Blind-submission hygiene:

- Remove the draft note in `main.tex` before uploading to OpenReview.
- Do not submit helper markdown files as supplementary material unless they have
  been separately anonymized.
- Check for names, usernames, repository URLs, personal paths, acknowledgments,
  and public code links before submission.
