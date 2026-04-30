# Review Rubric

Use this before asking a mentor, Claude, Codex, or yourself to review the paper.

## Format Review

- [ ] ICML 2026 template is used without manual spacing hacks.
- [ ] Main body is at most 8 pages for the workshop long-paper submission.
- [ ] References and appendix are included in the same PDF.
- [ ] The submission is blind: no author names, affiliations, acknowledgments, project usernames, or identifying repo paths.
- [ ] Tables have captions above; figures have captions below.
- [ ] All plots are readable when printed at paper size.

## Semantic Review

- [ ] The abstract says what was studied, what was found, and why it matters in 4-6 sentences.
- [ ] The introduction gives a one-sentence pitch a reviewer could repeat.
- [ ] Each section answers one clear reviewer question.
- [ ] Every figure has exactly one job.
- [ ] Every paragraph has a purpose; remove paragraphs whose job is not obvious.
- [ ] The paper states positive claims directly but does not inflate them.

## Mechanistic Interpretability Review

- [ ] The work is not only behavioral: it makes a representation-level claim.
- [ ] Causal evidence is separated from correlational evidence.
- [ ] Probe results are not overstated as mechanisms.
- [ ] Steering results include controls or caveats.
- [ ] SAE evidence is framed as support/texture, not as proof of a pure feature unless demonstrated.
- [ ] The paper is clear about what is not reverse-engineered.

## Novelty Review

- [ ] Related work covers gradable adjectives and pragmatic thresholding.
- [ ] Related work covers numerical representation in LLMs.
- [ ] Related work covers linear representations, activation steering, and SAE decomposition.
- [ ] The novelty claim is specific: mechanistic evidence for context-normalized scalar standing in residual streams.
- [ ] Similar prior work is not caricatured; the distinction is fair and narrow.

## Evidence Review

- [ ] Each headline number has a source path.
- [ ] Each headline claim has a failure mode or caveat.
- [ ] Main-body evidence is enough to support the abstract.
- [ ] Appendix evidence is discoverable but not required for the main narrative.
- [ ] Negative or mixed results are used to improve the framing instead of being hidden.

## Reviewer Questions To Answer In The Draft

1. Why is this not just behavioral pragmatics?
2. Why is this not just a probe finding?
3. How do we know the effect is context-normalized rather than raw magnitude or lexical bias?
4. How much of the effect is lexical/output-facing?
5. Does the shared direction transfer across adjective pairs?
6. Where does the claim fail or weaken?
7. Why should a mechanistic interpretability workshop care?

