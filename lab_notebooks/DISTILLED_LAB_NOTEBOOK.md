# Distilled Lab Notebook: Weight-Space Circuit Discovery

Complete synthesis of all source materials from the May 8–10 2026
investigation. Every number is traced to its source file.

---

## 1. What Happened (Chronological)

### May 8 Morning: Role Fingerprinting

Computed weight features for all 144 GPT-2 Small attention heads:
W_OV diagonal ratios, W_QK same/diff scores, SVD spectra, effective
rank. Plotted as heatmaps grouped by known circuit role (IOI name
movers, S-inhibition, induction, etc.).

**Key observation**: Heads in the same circuit role look visually
similar. Name movers share one W_OV pattern. S-inhibition heads share
another. Induction heads share a third.

Source: `raw_notes/PART1_SPEC.md`, `figures/part1_role_heatmaps/`

### May 8 Late Morning: t-SNE Clustering

t-SNE of 144 heads by weight features. Known circuit heads cluster
together. L9H3 appeared in the induction comparison panel as a
"random contrast" but showed unexpected structure: strong W_OV diagonal,
negative W_QK same/diff ratio, dominant SVD component.

Source: `raw_notes/PART2_SPEC.md`, `figures/part2_clustering/`

### May 8 Midday: The L9H3 Deep Dive

Elliot examined Layer 8 and Layer 9 heatmaps, recording stream-of-
consciousness observations. Key quotes from `INITIAL_THOUGHTS.md`:

> "l9 head 3 is the most norm and medium output entropy... really
> l8h10 and l9h3 are the ones that seem the outliers higher norm than
> anything else"

> "our target one has like very clear vertical bars almost entirely
> vertical bars and then the diagonal line"

> "l9 head ten is similar to our layer 9 head 3 i think visually its
> like blue lines vertically and then a red line diagonally"

L9H3 features from weight analysis:
- OV effective rank: 59.72 (near typical)
- OV Frobenius norm: 22.91
- QK Frobenius norm: 12.55
- Behavioral copying: 0.856 (highest in copier tier)
- OV copy_frac: 0.562 (highest by eigenvalue test)
- BOS attention: 36.5% (lowest among copiers)
- RTI attention: 6.2%
- GMM cluster: 2

Source: `raw_notes/INITIAL_THOUGHTS.md`, `HEAD_CHARACTERIZATION.md`

### May 8 Midday (continued): Name-Avoider Cohort

Elliot examined the "name avoider" cohort — heads with all-blue W_OV
(negative copying) plus diagonal:

> "L1 head 11 is entirely blue W_OV and blue diagonal and then red
> diagonal and all red for the W_QK. thats cool"

> "L3 H 10 is the same all blue left with blue diagonal"

> "l9 head 10 and l11 h5 look like copies of our l9 head 3 one visually"

The cohort: L1H11, L3H10, L4H7, L9H10, L11H5.

L9H3 is unremarkable within this cohort — sits dead center on the
scatter. Not an outlier.

**The hypothesis that emerged**: These heads "reduce interference from
repeated/expected tokens." Helpful for tasks needing to look past the
obvious (IOI: suppress repeated name; SVA: suppress distractors).
Harmful for factual recall and gendered pronoun (where you need to
retrieve the name).

Source: `raw_notes/SECONDARY_THOUGHTS.md`

### May 8 Midday: RTI Task Design

> "can we make a new task that tries to test this? surely this could
> be encoded somehow"

Repeated Token Interference (RTI) task designed:
- Clean: "Then Alice and Bob went to the store. Alice gave a drink to"
  → correct: Bob
- Corrupt: "Then Bob and Alice went to the store. Bob gave a drink to"
  → correct: Alice
- Measure logit(correct) - logit(incorrect)

Source: `raw_notes/SECONDARY_THOUGHTS.md`

### May 8 Afternoon: Circuit Assembly

From composition scores (which heads write to which other heads' key
channels):

**Proxy circuit assembled**:
- Upstream: L0H8, L0H9, L0H11
- L9H3's downstream targets: L10H11 (#1 shared with L9H10), L11H9,
  L11H11
- Copier cohort: L4H0, L5H6, L5H7, L7H0, L8H4, L8H7, L9H3, L9H10

Key composition scores (L9H3/L9H10 → downstream):
- L9H3 → L10H11: 29.16
- L9H10 → L10H11: 21.71
- Both → L11H11 and L11H9 in top-6

Source: `raw_notes/SECONDARY_THOUGHTS.md`

### May 8 Afternoon: First Ablation Results

Elliot examined RTI ablation results across prompt categories. From
`THIRD_MANUAL_REVIEW.md`:

> "full circuit seems to always push them down"
> "The huge effect of full circuit ablation suggests the early
> positional heads are the real drivers, not just the name-avoider
> cohort"

| Condition | Mean LD | Δ from baseline |
|-----------|---------|-----------------|
| Baseline | +1.526 | — |
| Cohort ablation (9 heads) | +1.410 | -0.117 |
| Downstream ablation (3 heads) | +1.566 | +0.039 |
| Full circuit (15 heads) | +0.259 | -1.268 |

**80% of effect from L0 trio alone** (Δ = -1.009 for 3 heads).

Source: `raw_notes/THIRD_MANUAL_REVIEW.md`, `raw_notes/THIRD_RESULTS.md`

### May 8 Evening – May 9: Rigorous Validation

19 experiments run. See Section 3 below for all results.

---

## 2. The Two Circuit Versions

**Old circuit** (early diagrams, `weight_analysis_rti.json`):

| Tier | Heads |
|------|-------|
| Backbone | L0H9, L0H10, L4H11 |
| Repeat-finders | L7H2, L7H9, L7H11 |
| Repeat-finders/Suppressors | L8H6, L8H10, L8H11 |
| Answer-extractor/Suppressor | L9H1, L9H6 |
| Readout | L9H9, L10H0, L10H2, L10H10 |

**Canonical circuit** (`roles.py`, used by all experiments):

| Tier | Heads |
|------|-------|
| Backbone | L0H8, L0H9, L0H11 |
| Detector | L4H11 |
| Copier | L4H0, L5H6, L5H7, L7H0, L8H4, L8H7, L9H3, L9H10 |
| Readout | L10H11, L11H9, L11H11 |

**weight_ioi variant** (hardcoded in commit `fd9c915`, May 8 21:04):
(0,1), (0,5), (0,10), (4,11), (5,1), (5,5), (5,8), (5,9), (6,1),
(6,9), (7,2), (7,9), (7,10), (10,2), (10,7)

Best automated reproduction of weight_ioi: 3/15 overlap using
edge_sum scoring (`reconstruct_c1_results.txt`). Not reproducible
algorithmically.

---

## 3. All Validation Results (Hard Numbers)

### 3.1 Ablation

| Test | Type | Δ | Significance |
|------|------|---|-------------|
| Full circuit (15 heads) | Zero | -1.267 | p=0.032 (permutation, N=30) |
| L0 trio only | Zero | -1.009 | p=0.032 |
| Copier cohort only | Zero | -0.117 | p=0.42 (not significant) |
| Full circuit | Resample | -0.511 | 95% CI [-0.655, -0.370] |
| Faithfulness ratio | Resample | -0.535 | Negative = circuit reverses LD |
| Control (other L0 heads) | Zero | -0.282 | — |
| Control (random 3 heads) | Zero | -0.133 | — |

Source: FINDINGS.md Phase 13, causal_tests.json

### 3.2 Path Patching

| Path | Recovery | Std |
|------|----------|-----|
| Full circuit | 94.7% | 0.225 |
| Copier → downstream | 105.9% | 0.923 |
| L4H11 → downstream | 94.8% | — |
| L0 → downstream (direct) | 97.0% | 6.79 |
| L0 → L4H11 | 53.4% | 2.670 |

Source: FINDINGS.md Phase 13

### 3.3 Minimality (Resample LOO)

| Rank | Head | Tier | LOO |
|------|------|------|-----|
| 1 | L9H10 | Copier | -0.693 |
| 2 | L11H9 | Readout | -0.656 |
| 3 | L8H4 | Copier | -0.586 |
| 4 | L7H0 | Copier | -0.570 |
| 5 | L4H11 | Detector | -0.562 |
| 6 | L8H7 | Copier | -0.515 |
| 7 | L5H7 | Copier | -0.444 |
| 8 | L0H9 | Backbone | -0.439 |
| 9 | L9H3 | Copier | -0.404 |
| 10 | L5H6 | Copier | -0.402 |
| 11 | L0H11 | Backbone | -0.402 |
| 12 | L4H0 | Copier | -0.400 |
| 13 | L0H8 | Backbone | -0.368 |
| 14 | L11H11 | Readout | -0.272 |
| 15 | L10H11 | Readout | +0.022 |

Source: FINDINGS.md Phase 13, causal_tests.json

### 3.4 DAS IIA

| Layer | k=1 | k=4 | k=8 | k=16 | k=32 | k=64 |
|-------|-----|-----|-----|------|------|------|
| L0-L4 | 0 | 0 | 0 | 0 | 0 | 0 |
| L6 | 0.01 | 0.02 | 0.01 | 0.01 | 0.04 | 0.04 |
| L8 | 0.12 | 0.48 | 0.67 | 0.74 | 0.78 | 0.79 |
| L10 | 0.21 | 0.76 | 0.96 | **1.00** | **1.00** | **1.00** |
| L11 | 0.23 | 0.80 | **1.00** | **1.00** | **1.00** | **1.00** |

Random IIA = 0 everywhere. Source: W&B run das-rti-20260509T001001

### 3.5 SAE Features

| Rank | Feature | Layer | Effect |
|------|---------|-------|--------|
| 1 | f19512 | L9 | 0.711 |
| 2 | f1721 | L9 | 0.701 |
| 3 | f3081 | L9 | 0.691 |
| 4 | f11865 | L7 | 0.655 |
| 5 | f16001 | L9 | 0.629 |

Per-layer max: L0 (0.12) → L4 (0.33) → L7 (0.65) → L9 (0.71)

Source: W&B run sae-rti-20260508T221921

### 3.6 Logit Lens Phase Transition

- L0-L8: LD near zero
- **L9: LD jumps to +5.712**
- L10-L11: stable

Per-layer attribution:
- L9: +34.6 (dominant)
- L11: -13.1 (active suppression)

Source: FINDINGS.md Phase 14

### 3.7 Sufficiency Comparison

| Method | Heads | Sufficiency (mean) | Sufficiency (resample) | Necessity FP |
|--------|-------|--------------------|------------------------|-------------|
| Weight circuit | 15 | -0.736 | -1.024 | 2 |
| Ground truth | 15 | -0.736 | -1.023 | 0 |
| ACDC | 30 | -0.344 | -0.021 | 27 |

Source: W&B runs causal-val-rti-*

### 3.8 Cross-Task Transfer

**SVA**: 12/12 GT heads found, 3 FP. Progressive AUC: weight ≈ GT.
L6H0 is SVA detector analogue (23.2% drop when ablated).

**IOI**: 13/26 GT heads found, 4 FP. Sufficiency: weight -0.022 vs
GT -0.018. Progressive AUC: weight 0.600 vs random 0.214 (3× better).

**Gendered Pronoun**: 5/5 GT heads found.

Source: FINDINGS.md Phases 11, 16, 18

### 3.9 Copier Subset Exhaustive Test (256 subsets)

Baseline (all 8 copiers): LD +2.124. No copiers: +1.811.

| Rank | Subset | LD | Δ vs baseline |
|------|--------|----|---------------|
| 1 | {L5H6, L5H7, L7H0, L9H3} | +2.251 | +0.127 |
| 2 | {L5H6, L7H0, L9H3} | +2.251 | +0.126 |
| 9 | {L5H6, L9H3} | +2.195 | +0.071 |
| 41 | all 8 | +2.124 | 0.000 |
| 256 | none | +1.811 | -0.313 |

**Core copier pair: L5H6 + L9H3.** Present in 94.4% and 65.4% of
>50%-effect subsets.

Best 4-copier subset OUTPERFORMS full 8 copiers — extra copiers add
mild interference (dampening, not amplification).

Source: LAB_NOTEBOOK.md Phase 21

### 3.10 Cross-Method Ablation

Baseline LD: +2.243. Zero and resample ablation of each method's heads:

| Method | Heads | Zero Δ | Resample Δ | Flips LD? | Zero/Resample |
|--------|-------|--------|------------|-----------|---------------|
| Weight | 15 | -2.464 | -0.355 | YES (-0.221) | 6.94 |
| EAP | 15 | -1.566 | -1.711 | No (+0.678) | 0.91 |
| EAP-IG | 15 | -1.731 | -1.932 | No (+0.513) | 0.90 |
| ActPatch | 15 | -1.067 | -1.372 | No (+1.176) | 0.78 |
| ACDC | 30 | -1.942 | -1.029 | No (+0.301) | 1.89 |

Weight circuit is the ONLY set whose ablation flips LD sign.

Source: LAB_NOTEBOOK.md Phase 23

### 3.11 EAP-Exact Verdict

5.5 hours, 32,491 edges, exact gradients. Top-15 recall: 0/15.
Same as approximate. Weight method: 14/15 in 2 minutes on CPU.

Source: LAB_NOTEBOOK.md Phase 30b

### 3.12 Per-Tier Method Recovery

| Tier | Weight | EAP/EAP-IG | ActPatch | ACDC | Wang ABA |
|------|--------|------------|----------|------|---------|
| Backbone (3) | 3/3 | 0/3 | 0/3 | 2/3 | 0/3 |
| Detector (1) | 1/1 | 0/1 | 1/1 | 1/1 | 0/1 |
| Copier (8) | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| Readout (3) | 3/3 | 0/3 | 0/3 | 0/3 | 2/3 |
| **Total** | **15/15** | **0/15** | **1/15** | **3/15** | **2/15** |

**Zero copier heads found by ANY non-weight method.**

Source: HEAD_CHARACTERIZATION.md Cross-Method section

### 3.13 Anti-Repetition

| Condition | Degeneration rate | Δ from baseline |
|-----------|-------------------|-----------------|
| Baseline | 47.5% | — |
| Backbone ablation (3 heads) | 85.8% | +38.3 pp |
| Full circuit ablation (15 heads) | 87.3% | +39.8 pp |
| Matched-layer random control | 60.4% | +12.9 pp |

Effect is 2.9× matched-layer random, 8.2σ above matched-count random.

Source: paper_c_anti_repetition.tex

### 3.14 Cross-Model Transfer

| Model | Tier | Top head | Stability |
|-------|------|----------|-----------|
| Medium | Backbone | L2H0 | 0.57 |
| Medium | Copier | L18H3 | 0.97 |
| Medium | Readout | L21H10 | 0.77 |
| Large | Backbone | L2H9 | 0.97 |
| Large | Copier | L35H0 (6 tied) | 0.77 |
| Large | Readout | L29H10 | 0.67 |

Backbone transfer improves with scale (0.57 → 0.97).

Source: LAB_NOTEBOOK.md Phases 26-27

---

## 4. Key Mechanistic Findings

### 4.1 L4H11 is NOT an Induction Head

Prefix matching: 7.37×10⁻²³ (rank 144/144, dead last).
QK Frobenius norm: 58.16 (3-4× any other head).
RTI attention: 0.234 (rank 4/144).
Attention entropy: 0.018 (nearly a delta function).

It detects repetition via backbone-written subspace directions, NOT
via [A][B]...[A] → predict [B] pattern matching.

BUT: Phase 18 revealed L4H11 is actually GPT-2's previous token head.
22 controlled prompts where position t-1 and the repeated token sit
at different positions: L4H11 attended to t-1 in 15/18 valid tests.
"Detection" was coincidence — in standard RTI prompts the repeated
token happens to be at t-1.

### 4.2 The Minimality-Composition Inversion

L0H11: Highest fan-out (K-comp 25-130), smallest LOO effect (-0.402)
L0H9: Narrower fan-out, largest LOO effect (-0.439)

Explanation: L0H11's signal is replicated across redundant paths
(removing it triggers backups). L0H9 carries a unique non-redundant
channel.

Extreme case: L0H9 has K-comp 111.2 to L4H11 (second highest in
circuit) and the largest backbone LOO effect (-0.439), yet ALL 12 of
its outgoing edges are INACTIVE in QKV path patching. Zero causal
flow despite maximum structural capacity.

### 4.3 Weight Capacity ≠ Causal Flow

Weight analysis measures structural ability (what the network is wired
to do). Activation analysis measures runtime function (what it does on
a given distribution). These are different things:

- Backbone→L4H11: K-comp strongest edge class (76-130), but actual
  causal flow through L4H11→downstream is the active pathway
- Copier→readout: K-comp near baseline (8.9 mean), but path patching
  shows 106% recovery through this pathway (via residual stream)

### 4.4 Readout Antagonism

Removing readout heads IMPROVES RTI performance:
- Remove L10H11: +0.022
- Remove L11H9: +0.048
- Remove L11H11: +0.067
- Remove all readout: +0.115

Layer 11 per-layer attribution = -13.1 (active suppression). The
readout tier calibrates/dampens predictions rather than boosting them.

### 4.5 Two Information Transfer Mechanisms

1. **K-composition pathway** (backbone → {detector, copier}):
   K-comp scores 27-130, far above background 10.4. Information flows
   through direct weight composition.

2. **Residual stream pathway** (detector → copier, copier → readout):
   K-comp 6-9, near background. Information flows through the shared
   residual stream.

### 4.6 OV Negativity Hypothesis Was Wrong

The Perplexity review predicted cohort heads would have negative OV
scores ("token suppressors"). Actually: cohort heads have the MOST
POSITIVE OV scores (mean +2.17). They are copy amplifiers, not
suppressors.

The actual suppression heads are L6H6 (-3.89), L5H11 (-3.48),
L3H10 (-3.21) — separate from the circuit.

### 4.7 The Distributed Copier Tier

8 copier heads, no single one essential. Mean activation patching
effect: +0.015 (individually below any threshold). But:
- Full copier tier path patching: 106% recovery
- Best 2-copier pair {L5H6, L9H3}: preserves 123% of copier effect
- Best 4-copier subset: OUTPERFORMS full 8 copiers

This is why every automated method (EAP, ACDC, ActPatch, Wang ABA)
misses every copier: each individual copier is below marginal-effect
threshold.

---

## 5. What Distinguishes This from the 173-Feature Classifier

The 173-feature bootstrap classifier (paper_a_weight_method.tex) is
SEPARATE from the manual discovery described here. Timeline:

1. Manual heatmap inspection → circuit hypothesis (this paper)
2. Circuit validated by 19+ causal experiments (this paper)
3. THEN: 173 features extracted, bootstrap classifier built, F1=1.00
   achieved (paper_a — a different paper)

The classifier validates that the circuit's weight geometry is
systematically distinctive. But the circuit was found BEFORE the
classifier existed, through visual pattern recognition.

Key differences:
- Paper A: Automated pipeline, bootstrap, feature selection, greedy
  algorithm. The contribution is the METHOD.
- This paper: Manual discovery through visual inspection. The
  contribution is the CIRCUIT and the DISCOVERY PROCESS.

The 173-feature classifier confirmed:
- Backbone identified by K-alignment with ICA directions (spectral)
- Copiers identified by QK singular value gap (NOT by OV — the
  greedy algorithm completely ignored OV features for copiers)
- Detector identified by cluster-pair alignment
- Readout identified by OV cluster alignment + K-alignment

---

## 6. The Reproducibility Question

Can the manual discovery be reproduced algorithmically?
`reconstruct_c1_results.txt` tested 20+ automated scoring methods:

| Method | Best overlap with weight_ioi (15 heads) |
|--------|----------------------------------------|
| edge_sum | 3/15 |
| composition_strength | 1/15 |
| full_combined | 1/15 |

**No.** The manual circuit is NOT reproducible from any automated
weight procedure. The human pattern-matching step is essential. This
makes the discovery more like Wang et al.'s IOI investigation (manual
with activation patching) than like an algorithm.

---

## 7. Source File Index

### Raw Notes (in c1_manual_investigation/raw_notes/)
- INITIAL_THOUGHTS.md — Elliot's stream-of-consciousness on L8/L9 heatmaps
- SECONDARY_THOUGHTS.md — Name-avoider cohort, RTI task design
- THIRD_MANUAL_REVIEW.md — RTI ablation results manual analysis
- THIRD_RESULTS.md — Ablation effect table
- DISCOVERY_STORY.md — Narrative write-up
- FINDINGS.md — Structured findings with numbers (THE key reference)
- HEAD_CHARACTERIZATION.md — Per-head data table for all 15 heads
- LAB_NOTEBOOK.md — Chronological record of all experiments (30+ phases)

### Blog Posts (restored from git)
- 01_weight_circuit_discovery.md — Method and comparison with ACDC
- 02_rti_circuit.md — The 4-tier architecture in detail
- 03_feature_prediction.md — What weight features predict circuit heads

### Existing Paper Drafts
- paper_a_weight_method.tex — 173-feature classifier (SEPARATE paper)
- paper_c_anti_repetition.tex — Anti-repetition finding
- weight_heatmap_discovery_v1.tex — THIS standalone paper (just created)

### Key Data Files
- data/head_census.json — Weight features for all 144 heads
- data/behavioral_census.json — Behavioral features (500 sequences)
- data/causal_tests.json — All causal validation results
- data/weight_analysis_rti.json — Eigenvalues, composition, inhibition
- data/layer_deep_dives.json — Per-tier SVD, attention, logit lens
- data/acdc_circuit_rti.json — ACDC's 30-head circuit
- data/circuit_synthesis.json — Bootstrap classification

### Figure Directories
- figures/part1_role_heatmaps/ — W_OV/W_QK per circuit role
- figures/part2_clustering/ — t-SNE, dendrograms
- figures/part3_l9h3/ — DLA, all-heads panels
- figures/part3_screenshots/ — Original analysis screenshots
- figures/part4_manual_screenshots/ — May 8 manual inspection
- figures/part4_circuit_diagrams/ — Circuit topology diagrams
- figures/part4_causal_validation/ — PCA, causal tests, bootstrap CIs

---

## 8. Open Questions

1. **L4H11 is a previous token head, not a repeat detector** (Phase 18).
   The paper should be honest about this — the "detection" mechanism is
   positional coincidence, not semantic repeat detection. The circuit
   still validates causally, but the mechanistic interpretation of L4H11
   needs nuance.

2. **Cross-model causal validation is missing.** Weight features transfer
   to GPT-2 medium/large (stability 0.53-0.97), but the transferred
   heads haven't been causally validated. We don't know if they form
   functional circuits in the larger models.

3. **The weight_ioi variant is not reproducible.** 3/15 overlap with best
   automated method. This is either a strength (manual discovery finds
   things algorithms can't) or a weakness (irreproducibility).

4. **Transfer controls running but incomplete.** Depth-matched random and
   shuffled-label controls were running on pods (Phase 28) — unclear if
   they completed.
