# V12.1 Lexical Disentanglement Summary

V12.1 tests whether V12 lexical steering came from literal adjective-token
semantics, sentence-final/template state, or a non-lexical residual of
`primal_z` after removing a lexical subspace.

## Aggregate Results

- mean cos(primal_z, word-token lexical direction): +0.104
- mean cos(primal_z, sentence-token lexical direction): +0.102
- mean cos(primal_z, sentence-final lexical direction): +0.260
- mean fraction of primal_z norm^2 in lexical subspace: 0.080
- mean lexical-projection/primal steering ratio: +1.246
- mean lexical-residual/primal steering ratio: +0.688

## Interpretation Guardrails

- If residual steering is strong, V12 lexical effects do not exhaust the causal direction.
- If lexical projection dominates and residual steering is weak, the causal direction is mostly lexical/adjective geometry.
- If both are nonzero, use the mixed-mechanism framing.

## Per-pair Steering Ratios

- height: lexical_projection/primal=+0.603, residual/primal=+0.864, lexical_norm2=0.067
- age: lexical_projection/primal=+0.815, residual/primal=+0.849, lexical_norm2=0.065
- weight: lexical_projection/primal=+1.515, residual/primal=+0.661, lexical_norm2=0.065
- size: lexical_projection/primal=+1.121, residual/primal=+0.604, lexical_norm2=0.161
- speed: lexical_projection/primal=+1.655, residual/primal=+0.575, lexical_norm2=0.063
- wealth: lexical_projection/primal=+1.013, residual/primal=+0.732, lexical_norm2=0.095
- experience: lexical_projection/primal=+1.835, residual/primal=+0.566, lexical_norm2=0.066
- bmi_abs: lexical_projection/primal=+1.412, residual/primal=+0.656, lexical_norm2=0.060
