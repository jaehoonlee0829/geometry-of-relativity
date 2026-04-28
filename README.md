# geometry-of-relativity

Mechanistic interpretability study of how Gemma represents **context-relative
gradable adjectives**: when the prompt says `Person 16: 170 cm. This person is
___`, does the model judge 170 cm as tall in absolute terms, or relative to the
15-person context it just read?

**Target venue:** ICML 2026 Mechanistic Interpretability Workshop.

## Core Setup

For each prompt, we separate three quantities:

- `$x$`: the raw target value, e.g. `170 cm`.
- `$\mu$`: the context mean, e.g. the group average height.
- `$z = (x - \mu) / \sigma$`: the target's context-normalized standing.

A model using `$x$` says "170 cm is tall" regardless of the group. A model using
`$z$` says "170 cm is tall in a short group, average in an average group, and
short in a very tall group."

The main v11 evaluation uses a dense grid:

- 8 adjective pairs: height, age, weight, size, speed, wealth, experience, BMI.
- 20 raw values `$x$` x 20 relative standings `$z$` x 10 context seeds.
- 4,000 prompts per pair per model.
- Gemma 2 2B and Gemma 2 9B, with Gemma Scope SAE analyses.

The grid is designed so `$x$` and `$z$` are not accidentally the same variable:
we choose `$x$` and `$z$`, then derive `$\mu = x - z\sigma$`.

![v11.5 9B story](figures/v11_5/readme_9b_story.png)

## Main Findings

### 1. Gemma's adjective behavior tracks context-normalized z

On the dense v11 grid, cell-mean `corr(logit_diff, z)` is at least 0.92 for
all 8 pairs on both Gemma 2 2B and 9B. Here
`logit_diff = logit(high adjective) - logit(low adjective)`, e.g.
`logit(" tall") - logit(" short")`.

This is the behavioral anchor: the model's high-vs-low adjective preference is
much better explained by relative standing than by raw magnitude alone.

### 2. z is available early, then carried forward

Fold-aware increment analyses show that most new linear `z` information appears
in the first few layers, especially L1-L3 in Gemma 2 9B. After that, cumulative
decodability stays high while new per-layer information is near zero.

Interpretation: the model computes a context-normalized feature early and then
carries it forward. This is different from saying late layers recompute `z`.

The older v9 layer-sweep figure below is still useful because it includes
steering strength by layer: `z` is decodable early, but causal steering becomes
strong only later. It was run on a smaller Gemma 2 2B setup, so the exact figure
should be regenerated on v11 9B as a GPU follow-up.

![v9 layer sweep](figures/v9/layer_sweep_combined.png)

### 3. A shared z direction transfers across adjective pairs

For each pair we compute a simple mean-difference direction:

`primal_z = mean(h | z > 0) - mean(h | z < 0)`

In v11.5, a single shared direction built from the 8 per-pair `primal_z`
directions steers 7/8 pairs in Gemma 2 9B at at least 50% of the within-pair
direction's effect. Pairwise `primal_z` alignment averages about 0.52 in 9B.

Cross-pair steering is also real: with 5 seeds and BH-FDR correction over all
56 off-diagonal source-target pairs, 56/56 transfer effects are significant on
both 2B and 9B. In plain terms, a direction learned from one adjective pair
often moves another pair in the expected high-vs-low direction.

Important caveat: speed and experience are the weakest / most pair-specific
cases. The shared feature is broad, not universal in the strict 8/8 sense.

### 4. SAE features track z, not just numerals

The SAE control asks whether the apparent `z` features are merely tracking raw
number size or token frequency. For each pair, v11.5 takes the top SAE features
by `R²(z)` and also measures:

- `R²(x)`: does the feature track raw target value?
- `R²(token)`: does it track the numeric token / magnitude proxy?

The top features have high `R²(z)` but near-zero `R²(x)` and `R²(token)`.
For Gemma 2 9B, the top feature averages about 0.68 `R²(z)` across pairs while
the raw-value and token controls are near zero.

This supports the interpretation that the features are responding to "above vs
below the local norm", not just "large-looking number."

### 5. SAE sharing increases with scale

Cross-pair SAE overlap is measured with top-50 Jaccard similarity: for each
pair, take the 50 most `z`-correlated SAE features and ask how much the sets
overlap across pairs.

- Gemma 2 2B mean off-diagonal Jaccard: 0.109.
- Gemma 2 9B mean off-diagonal Jaccard: 0.223.

9B has roughly 2x the cross-pair SAE feature overlap. This is one of the more
interesting scaling results: the larger model appears to use a more shared
feature basis for context-normalized scalar judgments.

![SAE overlap 9B](figures/v11/sae/cross_pair_feature_overlap_gemma2-9b.png)

## What We Do Not Claim

### The attention-head taxonomy is not causal

v10 produced a descriptive taxonomy of attention heads: "mu aggregators",
"comparators", and "z writers." v11/v11.5 refuted the causal version of that
story:

- single-head ablations were null;
- joint ablation of all tagged heads was null on 2B;
- joint ablation of all tagged heads slightly improved `corr(logit_diff, z)` on
  9B;
- permutation-null checks showed much of the taxonomy structure is chance-level.

So the taxonomy is a descriptive DLA pattern, not a mechanism we can claim is
load-bearing.

### W_U orthogonality is only a supporting control

`primal_z` is nearly orthogonal to `W_U[high] - W_U[low]`, the final lexical
readout direction. This is useful only as a weak control: `primal_z` is not
trivially the final "say tall instead of short" vector.

It is not strong evidence by itself, because earlier and middle layers can
rotate representations before the final unembedding. The stronger evidence is
behavioral tracking, steering, cross-pair transfer, and SAE controls.

### Fisher / Park-style metric fixes did not rescue the original hypothesis

Several earlier experiments tested whether Fisher pullback or Park's causal
inner product would rotate statistical probes into causal steering directions.
In the tested regimes, they did not. These are useful negative results but are
not part of the current main claim.

## Repository Layout

```
geometry-of-relativity/
  README.md
  FINDINGS.md              # full experimental log: v4-v11.5
  STATUS.md                # current status and retractions
  TODO.md                  # paper queue and follow-ups
  docs/
    paper_outline.md       # current paper plan
  scripts/
    vast_remote/           # GPU extraction and steering scripts
    analyze_v9_*.py        # Gemma 2 / SAE / layer sweep analyses
    analyze_v10_*.py       # dense height deep dive
    analyze_v11_*.py       # dense cross-model analyses
    analyze_v11_5_*.py     # shared-z, transfer, SAE controls, bootstrap CIs
    plot_v11_5_*.py        # README / paper figures from JSON summaries
  results/
    v11/                   # dense per-model/per-pair JSON summaries
    v11_5/                 # shared-z, transfer, SAE, ablation, CI summaries
  figures/
    v9/                    # layer sweep and SAE figures
    v10/                   # dense height figures
    v11/                   # PCA, SAE overlap, transfer figures
    v11_5/                 # summary figures
```

Large activation `.npz` files are intentionally not committed; they live on the
private Hugging Face dataset `xrong1729/mech-interp-relativity-activations`.

## Reproduce

```bash
cp .env.example .env       # add HF_TOKEN if fetching private activations
pip install -e ".[dev]"
pytest tests/ -v -m "not gpu"

# Fetch activation artifacts when needed.
python scripts/fetch_from_hf.py --only v11

# Rebuild the README story figure from committed JSON files.
python scripts/plot_v11_5_readme_story.py

# Re-run v11.5 JSON analyses.
bash scripts/run_v11_5_all.sh
```

## License

CC-BY-4.0 for the paper, MIT for the code.
