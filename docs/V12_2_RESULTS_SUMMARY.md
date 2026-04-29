# V12.2 Residual vs Lexical Transfer Summary

V12.2 compares cross-pair transfer from full `primal_z`, its lexical
projection, and its lexical residual at Gemma 2 9B L33. This is a
single-seed steering follow-up using seed0 unique target cells.

## Aggregate Transfer

- full: diag=+0.067, offdiag=+0.026, ratio=2.529947298681014, offdiag_pos=1.00
- lexical_projection: diag=+0.087, offdiag=+0.011, ratio=8.15863643995028, offdiag_pos=0.73
- lexical_residual: diag=+0.044, offdiag=+0.024, ratio=1.826720261826565, offdiag_pos=1.00
- random_null: diag=+0.000, offdiag=-0.001, ratio=-0.46153843579207693, offdiag_pos=0.39

## Paired Off-diagonal Comparisons

- residual_minus_lexical: mean=+0.014, CI95=[+0.010, +0.018], positive_fraction=0.82
- residual_minus_full: mean=-0.002, CI95=[-0.003, -0.001], positive_fraction=0.32
- lexical_minus_full: mean=-0.016, CI95=[-0.019, -0.012], positive_fraction=0.16

## Target Lexical-subspace Leakage

- corr(full offdiag transfer, target lexical overlap) = +0.855
- corr(lexical_projection offdiag transfer, target lexical overlap) = +0.659
- corr(lexical_residual offdiag transfer, target lexical overlap) = +0.788

## Interpretation Guardrails

- These are single-seed transfer matrices, not BH-FDR significance claims.
- Residual transfer should not be called non-lexical unless target-side lexical overlap is low.
- Projection/residual directions are normalized before steering, so matrix values compare intervention potency, not vector-energy fractions.

## Interpretation

V12.2 supports the mixed-mechanism Case D from the session plan.

The residualized directions transfer substantially better across pairs than the
normalized lexical projection component: residual off-diagonal mean is +0.024
versus lexical-projection off-diagonal mean +0.011, and residual is higher in
46/56 directed off-diagonal cells. The unreduced full `primal_z` directions
remain slightly stronger on mean off-diagonal transfer (+0.026), so residual
recovers most, but not all, of the full cross-pair effect.

The lexical projection is high-gain locally rather than broadly shared: it has
the largest diagonal mean (+0.087) but much weaker off-diagonal transfer. This
matches the interpretation that lexical/output-facing components are potent for
the same pair and related pairs, while the residual carries broader transfer.

However, the target lexical-subspace leakage check prevents a clean
"non-lexical residual" claim. Residual off-diagonal transfer is still strongly
correlated with target lexical overlap (r=+0.788). Residual transfer therefore
persists after the source lexical split, but its off-diagonal structure is still
organized by target-side lexical/domain similarity.

Safe wording:

```text
In this single-seed Gemma 2 9B L33 follow-up, residualized directions retain
broad cross-pair transfer and outperform the lexical projection off-diagonal,
but they remain slightly below full primal_z and still track target lexical
overlap. This is evidence for a residual shared component, not proof of a clean
non-lexical shared code.
```
