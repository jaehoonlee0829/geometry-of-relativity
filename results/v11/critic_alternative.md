# alternative critic

# Alternative-Explanation Critique of v11

## 1. Cross-pair transfer is mostly a generic "make-the-number-bigger" signal, not a shared z-code
**File:** `results/v11/gemma2-2b/cross_pair_transfer_dense.json`, `results/v11/gemma2-9b/cross_pair_transfer_dense.json`

Within-pair slopes (2B): height 0.040, weight 0.058, age 0.067, wealth 0.079, speed 0.073, size 0.100, experience 0.108. Off-diagonals are routinely 40–80% of diagonal — but look at the structure: `size→height` = 0.078 vs within-height 0.040 (transfer *exceeds* within!), and `size→weight` = 0.080 vs within-weight 0.058. That isn't "40% transfer of a domain-general z-code"; it's that the `size` direction is just a stronger generic magnitude pusher. The cheap explanation: shared **number-token-magnitude** direction in residual stream (numerals 150/170/200 etc. share embedding geometry across all numeric pairs). No z-normalization needed. Test the team didn't run: transfer with μ held constant (pure x-shift) should give similar slopes if true.

## 2. PC1≈z is just PC1≈x in disguise for high-σ/μ-ratio pairs
**File:** `results/v11/gemma2-2b/height/pca_summary.json` (PC1_vs_z=0.969, PC1_vs_x=1e-10) vs `size/pca_summary.json` (PC1_vs_z=0.075, PC1_vs_x=0.651) and `age/pca_summary.json` (PC1_vs_z=0.21, PC1_vs_x=0.42).

The pairs where PC1 "tracks z" (height 0.97, weight 0.95, bmi_abs 0.92, wealth 0.86) are exactly the pairs where x and z were most successfully **decorrelated by grid design** — so PC1 picks up whichever has more variance after grid balancing. For `size` and `age` PC1 picks x. The cheap explanation: PC1 tracks whichever of {x, z} the prompt grid happened to load more variance onto; this isn't evidence of a "z-representation."

## 3. cos(PC1, primal_z) sign-flipping across adjacent layers = arbitrary PCA sign convention
**File:** `cos_pc1_primal_summary.json`, height 2B layers 19→25: −0.999, +0.999, −0.999, +0.999, −0.999, +0.999, +0.999.

These flips are not phase transitions — they're sign ambiguity of PCA eigenvectors. Reporting `|cos|` would collapse the "story." The cheap reading: cos magnitude is ~1 from L13 onward, period; no per-layer dynamics to interpret.

## 4. Head-ablation "causal" results are null
**File:** `results/v11/gemma2-2b/head_ablation_causal.json`: baseline corr_z=0.976; after ablating the comparator (L13h2) it's 0.973 (Δ=−0.003); early-writer (L3h0) Δ=−0.008; mu_aggregator Δ=+0.0002. **Same in 9B**: `head_ablation_causal.json` z_writer Δ=−0.001, comparator Δ=+0.002.

These are within noise. Cheap explanation: the v10/v11 "head taxonomy" is observational DLA correlation; ablation shows redundancy/no causal role. The team's 9B taxonomy "alignment" with 2B is then meaningless — both taxonomies tag heads that don't matter, so any overlap is consistent with chance given threshold gates (top-quartile per metric).

## 5. SAE z-features track output-token frequency / number-magnitude, not z
**File:** `results/v11/gemma2-2b/sae_L7_L20_overlap.json`: top L20 features have R²(z) of 0.83, 0.77... but the team never reports R²(x|feature) or R²(token-frequency|feature) as a control. Given §1 above and the v10 finding that "most z-features are monotonic," the cheap explanation is these are **monotonic-in-numeral-magnitude** features (firing on "180" > "150" regardless of context μ). Cross-pair Jaccard being low (figures only) is then trivially explained by per-pair tokenizer-level numeral subsets, not "lack of shared z-structure."
