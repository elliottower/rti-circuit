# Heatmap Exploration Lab Notebook

Chronological record of experiments investigating what visual patterns in W_OV heatmaps distinguish RTI circuit heads from non-circuit heads in GPT-2 Small.

---

## 2026-07-30: Session 1 — Can we quantify the visual patterns?

### Motivation

The heatmap discovery paper identifies 15 circuit heads by visual inspection of W_E[:k] @ W_V @ W_O @ W_U[:, :k] logit matrices. The question: can we turn these visual patterns into computable metrics that automatically separate circuit from non-circuit heads?

### E4 — 108-feature weight classifier (moved to weight-feature paper)

Ran 108 engineered features (SVD, token interaction, composition, direction alignment) through bootstrap greedy classifier. Results: AUROC 0.997, perfect recall at threshold 0.7. **These belong to the weight-feature paper, not this paper.** Results moved to `paper_numbers/weight_feature_paper_E4_20260730/`.

- Script: `paper/E4_weight_feature_classifier.py`
- Pre-reg: `paper/prereg/E4_weight_feature_classifier.md` (SHA 75e03a4)
- Results: `paper/paper_numbers/weight_feature_paper_E4_20260730/`

### E4b — Simple heatmap metrics (diagonal score, vertical band score)

Computed 3 simple metrics for all 144 heads:
1. **Diagonal score**: mean(diag) - mean(offdiag). Positive = copy head.
2. **Vertical band score**: var(column means) / var(all entries). High = vertical stripes.
3. **Effective rank**: from SVD of the logit matrix.

- Script: `paper/E4b_heatmap_metrics.py`
- Pre-reg: `paper/prereg/E4b_heatmap_metrics.md` (SHA 578b5f0)
- Results: `paper/paper_numbers/E4b_heatmap_metrics/`

**Key finding:** Copier heads have POSITIVE diagonal scores (mean +0.65), not negative. They are copy heads — they boost the logit for whatever token they attend to. The pre-reg prediction that copier heads would cluster at the negative extreme was wrong.

Zero copier heads appear in the bottom 15 on diagonal score. They're at the positive extreme instead. Simple scalar metrics (diagonal score, vertical band) don't cleanly separate copier heatmaps from non-circuit heatmaps.

**Open question:** If copier heads copy (positive diagonal), how does the circuit suppress repetition? The answer must be in WHAT they attend to, not what they do when attending.

### E4c — Copier attention patterns on repeated-token prompts

Investigation: run GPT-2 on 10 prompts with repeated tokens. For each circuit head, measure how much attention goes to repeated vs non-repeated source positions.

- Script: `paper/E4c_copier_attention_patterns.py`
- Results: `paper/paper_numbers/E4c_attention_patterns/`
- Status: COMPLETE

**Key finding: Copier heads attend to NON-repeated tokens.** Every single copier head preferentially attends to unique token positions. The detector (L4H11) is the only circuit head that preferentially attends to repeated tokens.

| Tier | Head | Attn to repeated | Attn to unique | Preference |
|------|------|------------------|----------------|------------|
| Detector | L4H11 | 0.700 | 0.300 | +0.40 (REPEATED) |
| Copier | L5H7 | 0.108 | 0.892 | -0.78 (UNIQUE) |
| Copier | L8H4 | 0.210 | 0.790 | -0.58 (UNIQUE) |
| Copier | L9H10 | 0.216 | 0.784 | -0.57 (UNIQUE) |
| Copier | L7H0 | 0.219 | 0.781 | -0.56 (UNIQUE) |
| Copier | L5H6 | 0.237 | 0.763 | -0.53 (UNIQUE) |
| Copier | L8H7 | 0.255 | 0.745 | -0.49 (UNIQUE) |
| Copier | L4H0 | 0.283 | 0.717 | -0.43 (UNIQUE) |
| Copier | L9H3 | 0.355 | 0.645 | -0.29 (UNIQUE) |
| Readout | L11H9 | 0.029 | 0.971 | -0.94 (UNIQUE) |
| Readout | L10H11 | 0.119 | 0.881 | -0.76 (UNIQUE) |
| Readout | L11H11 | 0.136 | 0.864 | -0.73 (UNIQUE) |

**Mechanism:** The circuit suppresses repetition by boosting unique tokens, not by directly suppressing repeated ones:
1. Detector (L4H11) attends to repeated tokens — detects repetition
2. Copier heads attend to non-repeated tokens and copy their logits (positive W_OV diagonal)
3. This boosts logits for unique continuations relative to repeated ones
4. Readout heads attend even more strongly to unique tokens (L11H9: 97% to unique)

### E4d — All 144 heads heatmap grid

Generated W_OV heatmaps for all 144 heads (12x12 grid, rows = layers, columns = heads). Circuit heads have colored borders by tier.

- Script: `paper/generate_all_heads_heatmap_grid.py`
- Results: `paper/generated_figures/all_144_heads_wov_grid_k50.png`, `.pdf`
- Status: COMPLETE

**Visual observations (pending detailed comparison):** Most non-circuit heads show near-uniform grey (low magnitude). Circuit copier heads show more structured patterns with visible diagonal and banding. Need to look at which non-circuit heads also show structure to understand false-positive landscape.

---

## Emerging picture of the RTI circuit mechanism

The W_OV heatmap (what a head does when attending) and the QK/attention pattern (what a head attends to) tell complementary stories:

- **W_OV positive diagonal** = "copy the attended token's logit" (boost it in output)
- **Attention to unique tokens** = "I attend to tokens that haven't appeared before"
- **Combined** = "boost logits for tokens that haven't appeared yet"

This is anti-repetition by promotion, not suppression. The paper's language about "negative diagonals" needs correction — the diagonal is positive (copy), and the anti-repetition comes from the attention pattern selecting unique tokens.

---

## E4e — Tier detection from weight-space metrics (AUROC/F1)

Computed OV diagonal score + QK same-token score + Frobenius norms for all 144 heads. Tested whether single metrics or combinations can detect each circuit tier.

- Script: `paper/E4e_tier_detection_from_weights.py`
- Results: `paper/paper_numbers/E4e_tier_detection/`
- Status: COMPLETE

Key results:
- **Detector (L4H11)**: perfectly detectable from QK Frobenius norm alone (AUROC 1.000, F1 1.000). QK norm = 58.2, next highest = 28.0.
- **Backbone**: QK Frobenius norm low (AUROC 0.917). Layer 0 + weak QK = backbone.
- **Copier heads**: hardest to detect. Best single metric = OV diagonal (AUROC 0.772, F1 0.323). Many non-circuit heads also have high OV diagonal (L11H8 = 4.03).
- **Readout**: OV Frobenius norm (AUROC 0.868).
- **Full circuit**: no single metric cleanly separates all 15 from the other 129.

## Abstract correction

Fixed the abstract in `weight_heatmap_discovery_v6a.tex`. Backup at `weight_heatmap_discovery_v6a_backup_pre_abstract_fix.tex`.

Changes:
- "six heads whose negative diagonals" → "eight heads whose positive diagonals indicate token-copying capacity"
- "write negative logits for repeated tokens — token suppression" → "despite near-zero activation-based copy scores"
- "suppresses the logits of tokens that have already appeared" → "attend preferentially to non-repeated tokens and boost their logits"
- "we identify" → "we discover a 15-head circuit from weights alone that is invisible to activation-based methods"
- "nine additional" → "seven additional" (8+7=15)
- Reordered P1: discovery claim first, then interaction stats as explanation

Discovery was weight-based throughout: W_OV heatmaps found 8 copier heads, composition scores (W_OV @ W_QK, also weight-based) traced backbone/readout/detector. Activation-based work (ablation, DAS, path patching) was all validation after circuit assembly.

The starting point was IOI: visual comparison of weight signatures against known IOI heads. L9H3 appeared as mirror of S-inhibition. RTI task was defined after the circuit was found. Automated methods (ACDC/EAP) were then tested on RTI — they miss it even given the correct task.

---

## v7 reviewer-response edits (2026-07-30)

Systematic review pass identifying NEMI reviewer objections. Created v7 with fixes:

**Text fixes applied:**
- Abstract: reframed as circuit contribution (not method); qualified "invisible" → "unrecoverable"; added Jaccard=0.00 complementarity finding; added K-comp formula; reframed degeneration as fragility
- Title: changed to "A Distributed Circuit in GPT-2 Found from Weights and Missed by Activation-Based Discovery"
- Intro contribution P: explicit "0/8 copier heads across five methods"; mentions companion weight-feature paper
- Section 2: added explicit k=50 and W_E[:k] @ W_V @ W_O @ W_U[:, :k] formula
- Section 3.4: added K-comp background statistics (mean=10.4, P95=21.0); explicit excluded-head scores (7-15, within P95)
- Section 3.4: added transparency note owning post-hoc task design
- Section 4.3: added paragraph explaining copier naming (weight capacity vs activation behavior)
- Section 4.3: trimmed inline numbers, moved to appendix references
- Section 4.2: defined 1% edge threshold explicitly
- Section 6: added comparison-asymmetry acknowledgment paragraph
- Section 7: reframed title ("Repetition Degeneration and Circuit Fragility"); lead with fragility finding; closing paragraph frames safety implications
- Discussion: strengthened post-hoc limitation with 3 mitigating factors
- Discussion: added threshold-sensitivity limitation paragraph
- Conclusion: rewritten with method details, fragility framing, complementarity as "broader finding"
- All "invisible" → "unrecoverable" except title and one qualified conclusion use

**Experiments completed (2026-07-31):**

### E4f — k sensitivity (COMPLETE)
- Pre-reg SHA: cfdd1b6
- Script: `paper/E4f_k_sensitivity.py`
- Results: `paper/paper_numbers/E4f_k_sensitivity/`
- **Rank correlation**: 0.987 (k=50 vs 100), 0.934 (k=50 vs 1000). k=50 is not cherry-picked.
- **Sign stability**: 8/8 copier heads positive at ALL k values (50, 100, 500, 1000).
- **Score trend**: Scores INCREASE with k (mean 0.65 → 1.21). Opposite to pre-reg prediction — larger token sets amplify the diagonal signal.
- **Rank stability**: Pre-reg "all in top 30" FALSIFIED. L4H0 ranks 95th, L5H6 ranks 72nd. Early copier heads have weak diagonals. Consistent with progressive amplification narrative (later copiers = stronger copying capacity).
- **Takeaway**: k=50 is robust. The paper can state "results are stable across k ∈ {50, 100, 500, 1000} (Spearman ρ > 0.93)."

### E4g — K-comp threshold sensitivity (COMPLETE)
- Pre-reg SHA: 58bca45
- Script: `paper/E4g_kcomp_threshold.py`
- Results: `paper/paper_numbers/E4g_kcomp_threshold/`
- **Background**: mean=10.4, P95=21.0, P99=40.9 (9504 edges total)
- **Backbone→detector gap**: +19.4 (circuit min 76.0 vs non-circuit max 56.7). Ratio 1.34×. Clean separation.
- **Backbone→copier gap**: -4.7 (NEGATIVE). Some non-circuit L0 edges to copier heads beat some circuit edges. No clean separation.
- **Detector→copier, copier→readout**: Near background. Confirmed these communicate via residual stream.
- **Threshold sensitivity**: P90–P95: all 15/15 circuit heads involved. P97: 14/15. P99: 10/15.
- **Takeaway**: Backbone→detector is the structurally unambiguous pathway (gap 19.4). Other pathways rely on residual-stream communication, not direct K-composition. The P95 threshold preserves all 15 circuit heads.

### E4h — Attention patterns at scale (COMPLETE)
- Pre-reg SHA: 9f114fa
- Script: `paper/E4h_attention_scaled.py`
- Results: `paper/paper_numbers/E4h_attention_scaled/`
- 204 prompts (12 templates × 17 name pairs), 189 valid (15 had no repeated tokens)
- **Copier tier**: All 8 heads have negative repeat-preference with 95% CIs excluding zero. Mean preference -0.924. Pre-reg confirmed.
- **L4H11 caveat**: pref=-0.999 (vs +0.40 in E4c). L4H11 is a PTH — it attends to t-1. In these templates, t-1 is a unique function word, not the repeated name. The metric doesn't capture PTH behavior; it measures correlation with repetition position, not PTH's actual mechanism.
- **BOS confound**: Copier heads attend 56-82% to BOS (always unique). This inflates the "unique preference" metric. The signal is real (copiers read BOS for backbone context) but the magnitude is partly driven by BOS attention, not selective avoidance of repeated content tokens.
- **Takeaway**: Copier heads reliably attend to non-repeated positions (n=189, all CIs exclude zero). The effect is robust. L4H11's behavior requires task-specific prompts where the repeated token IS at t-1 to show its detector role.

### E4i — Proper path patching (Wang et al. 2022 methodology) (COMPLETE)
- Script: `paper/E4i_path_patching_proper.py`
- Modal wrapper: `paper/modal_E4i_path_patching.py`
- Results: `paper/paper_numbers/E4i_path_patching_proper/`
- 302 prompts, 97 intra-circuit edges, 6 tier-to-tier pathways
- Ran on Modal A10G GPU, 2026-07-31

**Methodology**: Edge-level path patching following Wang et al. (2022) IOI paper:
1. Patch sender head output from corrupted input
2. Freeze all intermediate attention heads to clean (blocks attention-mediated propagation)
3. At receiver layer, freeze all non-receiver heads to clean
4. Post-receiver layers recompute normally (IOI paper: "layers after R recompute as in a normal forward pass")
5. MLPs recompute at all layers (isolates attention-to-attention paths)

**Bug found and fixed**: Previous implementation (`rerun_all_statistics.py` lines 239-310) accepted `receiver_heads` as a parameter but never used it — all non-sender heads saw unrestricted corruption propagation. This made L0→L4H11 identical to L0→downstream_direct. The new script isolates each edge properly.

**Additional fix**: Prompts have variable token lengths (6-22 tokens). Clean/corrupt pairs are right-padded to matching length with EOS token. Logit diff measured at original (unpadded) last position.

**Pathway recovery (ratio-of-means, bootstrap 95% CI):**

| Pathway | Effect (LD) | 95% CI | Recovery | 95% CI |
|---------|-------------|--------|----------|--------|
| backbone→detector | 0.231 | [0.161, 0.301] | 98.3% | [68.7%, 128.3%] |
| copier→readout | 0.110 | [0.055, 0.169] | 47.1% | [23.6%, 71.7%] |
| backbone→copier | 0.039 | [0.001, 0.078] | 16.6% | [0.9%, 32.6%] |
| detector→copier | 0.026 | [0.008, 0.044] | 11.1% | [3.5%, 19.0%] |
| backbone→readout | 0.008 | [-0.009, 0.024] | 3.2% | [-3.6%, 10.1%] |
| detector→readout | 0.000 | [-0.012, 0.012] | 0.1% | [-4.8%, 5.0%] |

Full circuit effect: 0.235 LD (baseline 1.526, corrupt 1.292).

**Top 5 individual edges:**

| Edge | Effect | 95% CI |
|------|--------|--------|
| L0H9→L4H0 | +0.170 | [0.108, 0.234] |
| L0H9→L4H11 | +0.167 | [0.105, 0.230] |
| L0H9→L5H6 | +0.162 | [0.105, 0.219] |
| L0H9→L5H7 | +0.158 | [0.102, 0.215] |
| L4H11→L5H6 | +0.074 | [0.035, 0.115] |

One inhibitory edge discovered: L5H7→L7H0 = -0.054 [-0.081, -0.029].

**Takeaway**: backbone→detector is the dominant attention-mediated pathway (98% of full circuit effect), confirming the K-comp gap analysis (E4g). L0H9 is the critical hub — top 4 edges all originate from it. The copier→readout pathway carries 47% of the effect. Other pathways are weak, consistent with residual-stream (not direct composition) communication.

**Why per-prompt recovery CIs were originally wide**: Mean-of-ratios blows up when the per-prompt full circuit effect (denominator) is near zero. Ratio-of-means (used above) gives stable, publishable CIs.

### E4i — Interaction / epistasis analysis (COMPLETE)
- Results: `paper/paper_numbers/E4i_interaction_analysis/interaction_results.json`
- Data source: 302-prompt ablation rerun (`paper/paper_numbers/bootstrap_reruns/`)

**The central finding**: Individual heads explain only 36% of the circuit's total causal effect. The remaining 64% comes from interactions between heads.

| Measure | Interaction % | 95% CI |
|---------|--------------|--------|
| LOO-based (15 individual heads vs full circuit) | **63.2%** | [48.5%, 75.5%] |
| Tier-based (4 tier ablations vs full circuit) | **22.3%** | [11.9%, 33.6%] |

The LOO-based measure is the relevant one for the paper's thesis: activation-based methods rank heads by marginal effect, but individual marginals capture only 36% of what this circuit does. The remaining 64% is interaction — precisely what marginal-effect methods cannot see.

**Decomposition**: within-tier interaction = 63% - 22% = ~41%. Cross-tier interaction = 22%. Most interaction is within-tier (copier heads cooperating), with additional cross-tier synergy.

**Per-head marginals (% of full effect)**:
- L4H11 (detector): 13.6% — largest individual contributor
- L0H9 (backbone): 6.0%
- L9H3 (copier): 5.9%
- L5H6 (copier): 5.0%
- L0H8 (backbone): 4.8%
- L8H7 (copier): 4.0%
- L7H0 (copier): 2.4%
- L10H11, L11H9 (readout): 1.8% each
- L5H7 (copier): **-6.4%** (removing it helps — antagonistic in isolation)
- L4H0, L9H10, L8H4 (copier): negative marginals (removing helps individually)

**Abstract correction**: Paper said "interactions account for 30%" — the correct number is **63%** (LOO-based) or **22%** (tier-based). The 30% appears to have been an unsourced approximation. Using the LOO-based number strengthens the paper's thesis.

**Experiments still needed:**
5. **Circuit diagram figure**: node-and-edge diagram showing 4-tier hierarchy with K-comp edge widths

## Open questions

1. Which non-circuit heads also show structured W_OV heatmaps? (look at 144-head grid carefully)
2. Does the detector's role change the residual stream in a way that causes downstream copier heads to attend differently?
3. Can the W_OV diagonal score + QK same/diff ratio together separate circuit heads?
4. Is this "boost unique" mechanism documented in existing repetition suppression literature?
5. How does this interact with the unembedding? The copier heads copy in logit space — do they specifically boost tokens that are contextually plausible continuations?
6. ~~The paper body (Section 3.3, line 222) says "negative diagonal" for L9H3 but the figure caption says "positive diagonal (diag/off = 2.3)" — needs fixing beyond abstract~~ FIXED in v6a
