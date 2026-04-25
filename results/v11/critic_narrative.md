# narrative critic

# Narrative consistency critique: v11 vs v9/v10

**1. Head taxonomy: v11 9B silently disagrees with v10 2B's "canonical z-writers."**
FINDINGS §14.6 names L13h2 the standout comparator in 2B and L0h6 the μ-aggregator. v11's 2B `head_ablation_causal.json` ablates exactly these (L13h2, L0h6, L3h0) and finds **Δcorr_z ∈ [−0.0075, +0.0002]** — i.e. the "canonical" heads are causally inert (<1% of baseline 0.976). The 9B run picks entirely new candidates (L21h3 z_writer, L16h3 comparator) with similarly null effects (Δ ≤ 0.0024). This either (a) refutes the v10 §14.6 taxonomy as load-bearing, or (b) shows the taxonomy doesn't transfer 2B→9B. v11 reports the numbers but FINDINGS §14.7's claim that "L13h2 comparator, L10h0/L17h7 z-writers" are mechanistically meaningful is left standing without acknowledgment that ablating them does ~nothing. This is an unacknowledged self-refutation, not just a "structural difference in detail."

**2. Cross-pair transfer contradicts v8's 97% cross-template / 0.19 PC1 cosine framing.**
`cross_pair_transfer_dense.json` (2b, L20) shows within-pair slopes 0.040–0.108 vs off-diagonal medians ~0.025–0.05 — i.e. cross-pair transfer is **~30–60% of within-pair**, not the near-zero implied by v8's "cross-pair PC1 cosine 0.19" (STATUS bullet). 9B is similar (within 0.044–0.093 vs off-diagonal ~0.02–0.05). Whether this *strengthens* or *weakens* the "shared z-direction" story is never reconciled with v8.

**3. SAE story: v11 higher-N replication actually undercuts v9 §12 / §14.5.**
v9 §12 said primal_z is "distributed across thousands of features"; §14.5 revised down to "~10² genuinely z-modulated features (138/65k)." v11 `sae_L7_L20_overlap.json` for height L20 reports **n_linear=59, n_bump=43, n_dead=16,049** out of 16,384 — so only ~100 active z-features in the 16k SAE, top R²(z) ≈ 0.83. v9's "Jaccard 0.06 cross-pair" claim is neither replicated nor refuted in the supplied JSON; the file ends at top-50 feature indices per pair without a Jaccard number. **Tension:** the figure `cross_pair_feature_overlap_*.png` exists but the headline metric is absent from the JSON, so v11 cannot be said to "replicate" or "change" the 0.06 finding — yet STATUS implies v11 is a higher-N replication.

**4. Park causal inner product — no direct re-introduction, but PC1≈primal flirts with it.**
v9 §10.3 refuted (W_U^T W_U)⁻¹·probe_z. v11's `cos_pc1_primal_per_layer` shows |cos|>0.99 at late layers for height/weight/bmi_abs (e.g. height L20=−0.999, L25=0.999). This is *unembedding-free*, so it doesn't re-litigate Park. But the PCA framing ("PC1 ≈ primal_z") is presented without noting that v9 already established primal ≫ probe by 8–18× — i.e. PC1 alignment with primal is *not* evidence the probe direction is causal. Mild tension, easily fixed with one sentence.

**5. Orthogonalized R² goes strongly negative (e.g. height orth_r2 = −5.7 at L25; bmi_abs −8.5).**
This is new in v11 and not contextualized against v10 §14's "encode-vs-use peak L14." Strongly negative orth_r2 at deep layers suggests x and z are entangled in a way that the v10 narrative ("z is encoded cleanly by L7, used L11+") does not predict. Unacknowledged.
