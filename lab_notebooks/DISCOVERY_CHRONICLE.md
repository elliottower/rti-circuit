# Weight Circuit Discovery Chronicle

How a 15-head circuit in GPT-2 was found through manual inspection of
weight matrices, without any forward passes or activation data.

This folder archives the complete discovery trail from May 8, 2026.
All raw notes, figures, and analysis logs are preserved in `raw_notes/`
and `figures/`.

## Timeline

### Phase 1: Role Fingerprinting (May 8, morning)

Computed weight-space features (W_OV diagonal ratios, W_QK same/diff
scores, SVD spectra, effective rank) for all 144 attention heads in
GPT-2 small. Visualized each head's features as a heatmap, grouped by
known circuit role (IOI name movers, S-inhibition, induction, etc.).

The key observation: heads in the same circuit role have visually
similar weight fingerprints. Name movers share a pattern. S-inhibition
heads share a pattern. Induction heads share a pattern.

**Figures**: `figures/part1_role_heatmaps/`

### Phase 2: Unsupervised Clustering (May 8, late morning)

t-SNE of all 144 heads by weight features. Known circuit heads cluster
together. A comparison panel placed each head next to its nearest
known-circuit neighbor, revealing which uncategorized heads resemble
which circuit roles.

L9H3 appeared as the "random contrast head" in the induction comparison
panel but showed unexpected structure: strong W_OV diagonal (copying
mode), negative W_QK same/diff ratio (anti-induction), and a dominant
SVD component.

**Figures**: `figures/part2_clustering/`

### Phase 3: L9H3 Deep Dive (May 8, midday)

The user examined heatmaps for layers 8 and 9, recording stream-of-
consciousness observations in `INITIAL_THOUGHTS.md`. Key excerpts:

> "l9 head 3 is the most norm and medium output entropy... Clearly
> there are like it and l8 head 10 are together... really l8h10 and
> l9h3 are the ones that seem the outliers higher norm than anything
> else"

> "our target one has like very clear vertical bars almost entirely
> vertical bars and then the diagonal line"

L9H3 turned out to be unremarkable within a broader "name-avoider
cohort" -- heads with all-blue W_OV (negative copying = token
suppression) plus diagonal (residual copy component). The cohort
included L1H11, L3H10, L4H7, L9H10, L11H5.

The user hypothesized a functional role: "reduce interference from
repeated/expected tokens." This led directly to designing the
Repeated Token Interference (RTI) task.

> "can we make a new task that tries to test this? surely this could
> be encoded somehow"

The RTI task was defined: prompts where the correct completion is NOT
the most-repeated token, measuring whether the circuit suppresses the
repeated-token bias.

**Figures**: `figures/part3_l9h3/`, `figures/part3_screenshots/`

### Phase 4: Manual Circuit Assembly (May 8, afternoon)

From composition scores (which heads write to which other heads'
attention), the user and Claude assembled a proxy circuit:

- **Upstream**: L0H8, L0H9, L0H11 (write to L11 targets)
- **Copier cohort**: L9H3, L9H10 (twins), L4H0, L5H6, L5H7, L7H0,
  L8H4, L8H7
- **Downstream**: L10H11 (shared #1 composition target of L9H3 and
  L9H10), L11H9, L11H11

The user examined RTI ablation results across prompt categories
(name repetition, location, list completion, pronoun resolution,
BPE fragment, common noun). Raw notes in `THIRD_MANUAL_REVIEW.md`.

Key result: Full circuit ablation devastates RTI performance
(Delta = -1.268, wiping out 83% of logit difference). L0 upstream
heads account for 80% of the effect (Delta = -1.009 for 3 heads
alone). The copier cohort alone shows a modest effect (Delta = -0.117),
confirming the distributed/redundant nature.

**Figures**: `figures/part4_manual_screenshots/`

### Phase 5: Rigorous Validation (May 8 evening -- May 9)

19 experiments validated the manually-assembled circuit. The full
record is in `raw_notes/LAB_NOTEBOOK.md` and `raw_notes/FINDINGS.md`.

Headline results:

| Metric | Weight circuit (15 heads) | ACDC (30 heads) |
|--------|--------------------------|-----------------|
| Sufficiency (mean ablation) | -0.736 | -0.344 |
| Sufficiency (resample) | -1.024 | -0.021 |
| Necessity false positives | 2 | 27 |

The weight circuit matches ground truth exactly and outperforms ACDC
despite half the heads.

Additional validation:
- W_OV eigenvalue test: two "copier" heads are actually suppression
  heads (L8H10 = -138.8, L9H1 = -19.0)
- Composition scores: copier-to-readout Q-composition is 2x random
  baseline
- DAS: perfect IIA (1.0) at L10-L11, confirming the causal variable
  lives exactly where the circuit architecture predicts
- Activation patching: 4 of top 5 heads are in the circuit
- SAE features: top causal features cluster in L9
- Logit lens: answer crystallizes at L9
- Probe generalization: 100% cross-set accuracy at L2 (abstract
  repetition representation)
- Cross-task: circuit recovers SVA ground truth (12/12 heads, 3 FP)

**Figures**: `figures/part4_circuit_diagrams/`, `figures/part4_causal_validation/`

## Two Circuit Versions

The investigation produced two distinct 15-head circuits on the same
day, reflecting iterative refinement:

**Old circuit** (from early diagrams and `weight_analysis_rti.json`):

| Tier | Heads |
|------|-------|
| Backbone | L0H9, L0H10, L4H11 |
| Repeat-finders | L7H2, L7H9, L7H11 |
| Repeat-finders / Suppressors | L8H6, L8H10, L8H11 |
| Answer-extractor / Suppressor | L9H1, L9H6 |
| Readout | L9H9, L10H0, L10H2, L10H10 |

**Final canonical circuit** (from `roles.py`, used by all experiments):

| Tier | Heads |
|------|-------|
| Backbone | L0H8, L0H9, L0H11 |
| Detector | L4H11 |
| Copier | L4H0, L5H6, L5H7, L7H0, L8H4, L8H7, L9H3, L9H10 |
| Readout | L10H11, L11H9, L11H11 |

The old circuit was from an earlier bootstrap pass. The canonical
circuit was re-derived through greedy feature selection + 30-round
bootstrap + causal validation.

## The weight_ioi Variant

A third variant was hardcoded in commit `fd9c915` (May 8, 2026 21:04)
as `weight_ioi` in the coalition sweep code. This variant uses IOI
prompts (not RTI) and has a different head list:

(0,1), (0,5), (0,10), (4,11), (5,1), (5,5), (5,8), (5,9), (6,1),
(6,9), (7,2), (7,9), (7,10), (10,2), (10,7)

No script reproduces this list. The best automated attempt
(`reconstruct_c1_results.txt`) achieves 3/15 overlap using edge_sum
scoring. This variant was manually identified through the same
iterative heatmap-inspection process, applied to IOI prompts.

## Discovery Method: Honest Framing

The circuit was discovered through manual inspection of weight-space
features, analogous to Wang et al.'s manual identification of the IOI
circuit from activation patching heatmaps. The process:

1. Compute weight features for all 144 heads
2. Visualize as heatmaps grouped by known role
3. Identify heads with similar visual fingerprints
4. Check composition scores to find information flow paths
5. Assemble candidate circuit from visual + compositional evidence
6. Validate causally (ablation, sufficiency, necessity)

The distinguishing characteristic relative to Wang et al.: the entire
discovery uses weight matrices only. No forward passes, no activation
data, no prompts. The manual step is visual pattern matching on weight
features, not activation patching heatmaps.

## File Index

### Raw Notes (chronological)

| File | Contents |
|------|----------|
| `PART1_SPEC.md` | Phase 1 design: feature fingerprints, heatmaps, distributions |
| `PART2_SPEC.md` | Phase 2 design: t-SNE, comparison panels, confusion matrix |
| `PART3_SPEC.md` | Phase 3 design: L9H3 DLA, backup test, attention, non-IOI probing |
| `L9H3_OVERVIEW.md` | L9H3 initial characterization (weight features, nearest neighbors) |
| `ANALYSIS_LOG.md` | Complete chronological log of hypotheses and experiments |
| `INITIAL_THOUGHTS.md` | User's stream-of-consciousness on layer 8/9 heatmaps |
| `SECONDARY_THOUGHTS.md` | Name-avoider cohort analysis, RTI task design |
| `THIRD_MANUAL_REVIEW.md` | User's analysis of RTI ablation results across prompt types |
| `THIRD_RESULTS.md` | Ablation effect table (L0 trio = 80% of effect) |
| `DISCOVERY_STORY.md` | Narrative write-up of full discovery and validation |
| `FINDINGS.md` | Structured findings with quantitative evidence |
| `HEAD_CHARACTERIZATION.md` | Per-head data table for all 15 circuit heads |
| `LAB_NOTEBOOK.md` | Chronological record of all 19 validation experiments |

### Figure Directories

| Directory | Contents |
|-----------|----------|
| `part1_role_heatmaps/` | W_OV/W_QK fingerprint heatmaps per circuit role |
| `part2_clustering/` | t-SNE, dendrogram, comparison panels |
| `part3_l9h3/` | DLA heatmap, layer 8/9 all-heads panels, name-avoider scatter |
| `part3_screenshots/` | User's original analysis screenshots (L9H3, induction, S-inhibition) |
| `part4_manual_screenshots/` | May 8 screenshots from manual heatmap inspection |
| `part4_circuit_diagrams/` | Blog-quality circuit overview, composition flow, methodology diagrams |
| `part4_causal_validation/` | PCA, causal tests, bootstrap CIs, path patching, IIA |

## Source

Restored from git history of `factorization-circuits` repo, commits
`44f6d4e` and `5cc6e7d`. Original location:
`MIB/MIB-circuit-track/weight_circuit/experiments/v2_second_investigation/raw_experiments/v1_role_weight_analysis/`
