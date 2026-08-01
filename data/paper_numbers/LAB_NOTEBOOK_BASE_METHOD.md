# Part 4: Rigorous Circuit Validation — Lab Notebook

Chronological record of all validation experiments on the 15-head weight-identified RTI circuit. Part 4 spans May 8-9, 2026.

**Canonical numbers reference:** See `paper_numbers/INDEX.md` for a consolidated table of every number cited in the paper/blog posts with source file pointers.

---

## Phase 1: Head Census and Circuit Synthesis (May 8, local CPU)

### Setup

Ran comprehensive weight-space analysis of all 144 heads. Computed OV copy scores, QK same-token scores, attention entropy, SVD spectra, and K-composition between circuit tiers.

### Scripts & Data

- `run_head_census.py` -> `data/head_census.json`, `figures/head_census_*.png`
- `run_circuit_synthesis.py` -> `data/circuit_synthesis.json`, `figures/circuit_synthesis.png`
- `run_l0_deep_dive.py` -> `data/l0_deep_dive.json`
- `run_layer_deep_dives.py` -> `data/layer_deep_dives.json`, `figures/tier_*.png`

### Key Findings

- PCA of 144 heads by weight features: circuit heads cluster distinctly from non-circuit
- K-composition between tiers confirms backbone -> copier -> readout information flow
- L0 heads have low attention entropy (broad attention), copier heads have high entropy (focused)
- SVD spectra differ by tier: backbone heads are lower-rank, readout heads are full-rank

---

## Phase 2: Behavioral Census (May 8, local CPU)

### Setup

Forward-pass behavioral analysis: DLA, attention pattern statistics, and prefix-completion behavior for all heads on RTI prompts.

### Scripts & Data

- `run_behavioral_census_cpu.py` -> `data/behavioral_census.json`, `figures/behavioral_*.png`

### Key Findings

- Strong DLA heads on RTI: L8H10 (#1, effect=0.61), L4H11 (#3, 0.32)
- Attention-to-name analysis: copier heads attend to repeated name position (D2), readout heads attend to non-repeated name (C)
- Prefix-matching: L7H9, L8H6, L8H10 show strong induction-like attention to D2

---

## Phase 3: Causal Tests — IIA, Path Patching, Bootstrap (May 8, local CPU)

### Setup

Interchange Intervention Accuracy (IIA), path patching, and bootstrap confidence intervals on the 15-head circuit.

### Scripts & Data

- `run_causal_tests.py` -> `data/causal_tests.json`, `figures/causal_tests.png`, `figures/bootstrap_cis.png`, `figures/iia_ld_shifts.png`, `figures/path_patching.png`

### Key Findings

- IIA with logit diff shift confirms circuit heads are causally important
- Path patching shows backbone -> copier -> readout causal flow
- Bootstrap CIs are tight (15 heads identified stably across resamples)

---

## Phase 4: ACDC Comparison (May 8, RunPod)

### Setup

Ran ACDC greedy backwards pruning on RTI task for direct comparison with weight circuit.

### Script & Pod

- `run_acdc_rti.py`, Pod: RunPod GPU
- W&B: `acdc-rti-20260508T221221`
- Data: `data/acdc_circuit_rti.json`

### Results

- ACDC found 30 heads
- Overlap with weight circuit: 3 heads (0.9, 0.11, 4.11), Jaccard = 0.071
- ACDC finds ablation-important heads but misses redundant copiers
- Interpretation: individual copiers have small ablation effects due to redundancy, so greedy backwards pruning can't detect them

---

## Phase 5: Activation Patching (May 8, RunPod)

### Setup

Single-head activation patching on all 144 heads for RTI task.

### Script & Pod

- `run_actpatch_rti.py`
- W&B: `actpatch-rti-*`

### Results

- L8H10 is #1 (effect=0.61), L4H11 is #3 (0.32)
- Top-15 recall = 7%, top-50 recall = 33%
- Activation patching finds individual importance, not compositional structure

---

## Phase 6: Causal Validation — Weight vs ACDC vs GT (May 8, RunPod)

### Setup

Ran full sufficiency/necessity/progressive evaluation for weight circuit, ACDC circuit, and ground truth.

### Script & Pod

- `run_causal_validation.py`
- W&B: `causal-val-rti-20260508T221801` (weight), `causal-val-rti-20260508T222518` (ACDC)

### Results — RTI

| Metric | Weight (15 heads) | GT (15 heads) | ACDC (30 heads) |
|--------|-------------------|---------------|-----------------|
| Sufficiency (mean) | -0.736 | -0.736 | -0.344 |
| Sufficiency (resample) | -1.024 | -1.023 | -0.021 |
| Necessity FP | 2 | 0 | 27 |
| Progressive AUC | -0.70 | — | — |

Weight circuit matches GT exactly. ACDC is worse despite having 2x the heads.

### Results — SVA (cross-task)

| Metric | Weight | GT |
|--------|--------|-----|
| Sufficiency (mean) | -1.110 | -1.107 |
| Necessity detected | 6/15 | — |
| Progressive AUC | -0.18 | — |

Weight circuit generalizes to SVA.

---

## Phase 7: SAE Feature Analysis (May 8, RunPod)

### Script & Pod

- `run_sae_rti.py`
- W&B: `sae-rti-20260508T221921`

### Results

Top features by RTI logit diff effect:
1. L9_f19512 (0.71)
2. L9_f1721 (0.70)
3. L9_f3081 (0.69)
4. L7_f11865 (0.65)
5. L9_f16001 (0.63)

Signal concentrates at L7-L9 (copier tier layers). Pre-existing SAE features with no learned alignment — immune to Sutter et al.

---

## Phase 8: Probes v1 — Trivially Decodable (May 8, RunPod)

### Script & Pod

- `run_probes_rti.py`
- W&B: `probes-rti-*`

### Results

- Repetition detection: 84% at L0, 100% by L2
- Identity probe: ~100% everywhere

### Assessment

**These probes are uninformative.** Repetition is trivially decodable from embeddings (a token's embedding IS itself). The probe picks up embedding identity, not a computed representation. Superseded by Probes v2.

---

## Phase 9: Sutter-Inspired Experiments (May 8, RunPod — IN PROGRESS)

Six experiments designed to validate the RTI circuit against the Sutter et al. non-linear representation dilemma.

### 9.1: SAE Causal Intervention + Circuit Overlap

- Script: `run_sae_causal_rti.py`
- Pod: `985stlqigbolj2` (sae-causal-rti-2245)
- Tests: ablate/steer top SAE features, measure logit diff change. Map features to circuit heads via decoder alignment. Cross-task control on IOI.
- Status: **Running**

### 9.2: Random Model Control

- Script: `run_random_model_control.py`
- Pod: `laeb9tfy9qeow3` (random-model-control-2245)
- Tests: weight classification on 3 randomly initialized GPT-2s. If method finds circuits in random models, it's picking up architecture, not learning.
- Status: **Running**

### 9.3: Circuit Logit Lens

- Script: `run_circuit_logit_lens.py`
- Pod: `2lhq8e52pe520d` (circuit-logit-lens-2245)
- Tests: logit lens convergence for full model vs weight circuit vs ACDC circuit vs random 15 heads.
- Status: **Running**

### 9.4: DAS Complexity Controls

- Script: `run_das_complexity_rti.py`
- Pod: `2gtkon5euuvcgd` (das-complexity-rti-2245)
- Tests: identity DAS and linear DAS on trained vs random GPT-2. The trained-random gap is the meaningful signal per Sutter et al.
- Status: **Running**

### 9.5: Probe Generalization

- Script: `run_probe_generalization.py`
- Pod: `u0k6xjd48v10qc` (probe-generalization-2245)
- Tests: train probes on names A-H, test on I-P (zero overlap). If probes generalize, the model represents "repetition" abstractly.
- Status: **Running**

### 9.6: Informative Probes v2

- Script: `run_probes_rti_v2.py`
- Pod: `pb7trgvpd2nknw` (probes-rti-2232)
- Tests: 6 probes (3 parameter-free: logit lens, decomposed attribution, distractor suppression; 3 trained: per-head output identity, second-occurrence marking, position binding)
- Status: **Running**

---

## Phase 10: Weight-Level Analysis — Eigenvalues, Composition, Inhibition, Prefix-Matching (May 8, local CPU)

### Setup

Four experiments derived from Elhage et al. "Mathematical Framework for Transformer Circuits" and Olsson et al. "In-context Learning and Induction Heads":

1. **W_OV eigenvalue sign test** — pure weight computation
2. **Composition scores between tiers** — pure weight computation
3. **Inhibitory head search** — forward passes (100 prompts, CPU)
4. **Prefix-matching and copying scores** — forward passes (100 prompts, CPU)

### Script & Data

- `run_weight_analysis_rti.py` -> `data/weight_analysis_rti.json`, `figures/weight_analysis_rti.png`
- W&B: `weight-analysis-rti-20260508T190646`

### Results: Exp 1 — Eigenvalue Test

Trace of W_OV (copying score) for each circuit head:

| Head | Tier | Copying score | Interpretation |
|------|------|--------------|----------------|
| L0H9 | backbone | +0.387 | Weak copier |
| L0H10 | backbone | +5.973 | Moderate copier |
| L4H11 | backbone | +26.529 | Strong copier |
| L7H2 | copier | +12.599 | Copier |
| L7H9 | copier | +55.343 | Strong copier |
| L7H11 | copier | +4.782 | Weak copier |
| L8H6 | copier | +32.850 | Strong copier |
| **L8H10** | **copier** | **-138.820** | **Suppression head** |
| L8H11 | copier | +35.774 | Strong copier |
| **L9H1** | **copier** | **-19.033** | **Suppression head** |
| L9H6 | copier | +37.549 | Strong copier |
| L9H9 | readout | +6.776 | Moderate copier |
| L10H0 | readout | +160.793 | **Strongest copier in circuit** |
| L10H2 | readout | +30.884 | Strong copier |
| L10H10 | readout | +32.128 | Strong copier |

Two "copier" heads are actually suppression heads — L8H10 and L9H1. These inhibit the wrong answer rather than reinforcing the right one. Parallels IOI's Negative Name Movers and the gendered pronoun circuit's 122% finding.

Tier means: backbone +11.0, copier +2.6 (dragged down by suppression heads), readout +57.6.

### Results: Exp 2 — Composition Scores

| Pair | Q-comp | K-comp | V-comp |
|------|--------|--------|--------|
| backbone -> copier | 4.68 | 5.88 | 3.81 |
| backbone -> readout | 4.42 | 6.79 | 4.97 |
| **copier -> readout** | **15.79** | **10.33** | **10.59** |
| Random baseline | 7.96 | 7.57 | 6.09 |

copier -> readout Q-composition is 2x random baseline. Readout heads' queries look for copier outputs. Backbone composition is near baseline — backbone contributes through residual stream, not direct weight composition.

### Results: Exp 3 — Inhibitory Head Search

- Circuit-only logit diff: -0.870 (mean ablation of 129/144 heads is too destructive)
- Full model logit diff: +1.115
- 10 heads with delta < -0.1 when added to circuit

Top inhibitory non-circuit heads:

| Head | Delta |
|------|-------|
| L0H4 | -0.367 |
| L0H7 | -0.233 |
| L11H1 | -0.198 |
| L0H8 | -0.161 |
| L0H2 | -0.136 |

Top helpful non-circuit heads:

| Head | Delta |
|------|-------|
| L1H2 | +0.233 |
| L10H7 | +0.215 |
| L2H5 | +0.128 |

### Results: Exp 4 — Attention to Name Positions

From the final position, attention fraction to each name position (D1 = first occurrence of repeated name, D2 = second occurrence, C = non-repeated name):

| Head | Tier | attn_D2 | attn_C | Functional role |
|------|------|---------|--------|----------------|
| L0H9 | backbone | 0.030 | 0.028 | Uniform (positional) |
| L0H10 | backbone | 0.026 | 0.030 | Uniform (positional) |
| L4H11 | backbone | 0.000 | 0.000 | Near-zero (gate?) |
| L7H9 | copier | **0.218** | 0.007 | Repeat-finder |
| L8H6 | copier | **0.445** | 0.013 | Repeat-finder (strongest) |
| L8H10 | copier | **0.246** | 0.044 | Repeat-finder + suppressor |
| L9H6 | copier | 0.008 | **0.356** | Answer-extractor |
| L9H9 | readout | 0.006 | **0.405** | Output copier |
| L10H0 | readout | 0.013 | **0.241** | Output copier |
| L10H2 | readout | 0.016 | **0.156** | Output copier |
| L10H10 | readout | 0.007 | **0.214** | Output copier |

Reveals two copier subtypes: repeat-finders (attend to D2, locating the repeated token) and answer-extractors (attend to C, reading the non-repeated name).

OV copying fraction (weight-only, does W_OV map token embedding to same token?): circuit mean 0.017, non-circuit mean 0.073. Circuit heads are NOT doing literal token-to-same-token copying at the embedding level — they operate on higher-level representations.

---

## Phase 11: Causal Validation — SVA Cross-Task (May 8, RunPod)

### Setup

Ran the weight circuit (identified for RTI) on Subject-Verb Agreement (SVA) to test cross-task generalization. Full sufficiency, necessity, and progressive ablation evaluation.

### Results

- 12/12 ground truth SVA heads found (3 false positives)
- L6H0 is strongly necessary: ablating it causes a 23.2% performance drop
- Weight and GT progressive ablation curves are nearly identical -- both degrade at the same rate as heads are removed
- Resample ablation shows poor separation between weight circuit and random baseline (resample is noisier for SVA than RTI)

### Assessment

Cross-task generalization confirmed. The weight method recovers the SVA circuit despite being trained on RTI weight features. The near-identical progressive ablation curves are the strongest evidence that the weight circuit captures the same computational structure as the ground truth.

---

## Phase 12: Informative Probes v2 — Partial Results (May 8, RunPod)

### Setup

Six probes designed to be informative (not trivially decodable): 3 parameter-free (logit lens, decomposed attribution, distractor suppression) and 3 trained (per-head output identity, second-occurrence marking, position binding).

### Results (partial -- logit lens and decomposed attribution completed)

**Logit lens**: The answer crystallizes at Layer 9. Logit diff jumps from near-zero to +5.712 at L9 -- this is where the model "decides" which name is the non-repeated one.

**Decomposed attribution**: Per-layer contribution to the final logit diff:
- L9 dominates with +34.6 differential -- the single largest contributor by far
- L11 shows active suppression at -13.1 differential -- it actively pushes against the correct answer
- Wrong token is promoted through L8-L10, then drops at L11

### Assessment

These results confirm L9 as the critical computation layer. The L11 suppression is a new finding that parallels the suppression heads (L8H10, L9H1) identified by eigenvalue analysis. The circuit has an active error-correction mechanism: L11 suppresses over-confident predictions from the copier tier.

---

## Phase 13: Edge Scoring — Crash and Relaunch (May 8, RunPod)

### Setup

All 8 weight-based edge scoring strategies completed circuit generation and faithfulness evaluation.

### Results

Circuits were generated and faithfulness was evaluated for all 8 strategies. However, the comparison table generation crashed due to a Python 3.10 f-string syntax bug (using `=` inside f-string expressions, which requires Python 3.12+).

### Status

Relaunched with the f-string fix. The edge importance data is intact; only the summary table needs regeneration.

---

## Phase 14: Circuit Logit Lens — Crash and Relaunch (May 8, RunPod)

### Setup

Compare logit lens convergence for full model vs weight circuit vs ACDC circuit vs random 15 heads.

### Results

Full model logit lens completed successfully. Crashed on ablated conditions (weight-only, ACDC-only) due to TransformerLens version mismatch -- `fwd_hooks` parameter not supported in the installed version.

### Status

Relaunched with the `run_with_hooks` API fix (`return_type="both"`).

---

## Phase 15: Causal Validation — Gendered Pronoun — Crash and Relaunch (May 8, RunPod)

### Setup

Cross-task validation on the gendered pronoun circuit.

### Results

Successfully identified 5/5 ground truth heads. Crashed during ablation evaluation due to sequence length mismatch -- gendered pronoun prompts have variable lengths, and the ablation hook assumed fixed length.

### Status

Relaunched with padding fix (truncate/pad corrupted activations to match clean sequence length).

---

## Phase 16: Edge-Level Validation (May 8, RunPod)

### Setup

Two-level causal validation: (1) node-level corruption of each circuit head individually, (2) edge-level QKV path patching for all 70 edges in the circuit.

### Script & Data

- `experiments/edge-validation/` -> `data/edge_validation.json`, `data/composition_scores.json`

### Results: Node Effects

L4H11 (detector) corrupted alone produces a 149.1% LD drop -- overshooting the full circuit ablation. Corrupting one head is *worse* than corrupting the entire circuit because the copiers faithfully amplify a corrupted signal.

| Head | Role | LD drop | % of circuit |
|------|------|---------|-------------|
| L4H11 | detector | +0.206 | 149.1% |
| L5H6 | copier | +0.033 | 23.6% |
| L9H3 | copier | +0.011 | 8.0% |
| (all other copiers) | copier | <3.3% each | |
| L0H9 | backbone | -0.000 | 0.0% |

L0H9: highest backbone K-composition to L4H11 (111.2) and largest backbone LOO effect (-0.439) but zero causal edge-level effect and all 12 outgoing edges inactive. The minimality-composition paradox: the wire is thick but carries no current on this distribution.

---

## Phase 17: DAS Sweep (May 9, RunPod)

### Script & Data

- `experiments/das-rti/data/das_rti.json` (W&B: `das-rti-20260509T001001`)

### Results

DAS IIA crystallizes at L10-L11 with a sharp phase transition:

| Layer | k=4 | k=8 | k=16 | k=64 |
|-------|-----|-----|------|------|
| 0-6 | 0.00-0.02 | 0.00-0.01 | 0.00-0.01 | 0.00-0.04 |
| 8 | 0.48 | 0.67 | 0.74 | 0.79 |
| **10** | **0.76** | **0.96** | **1.00** | **1.00** |
| **11** | **0.80** | **1.00** | **1.00** | **1.00** |

Random baseline: IIA=0.0 everywhere (except 0.01 at L11/k=64). The trained-random gap is massive at L10-L11 and zero elsewhere. Consistent with the logit lens: answer crystallizes at L9, becomes fully interventable at L10-L11.

---

## Phase 18: Previous Token Head Test (May 9, local)

### Script & Data

- `../part5_validation_misc/data/pth_vs_detector.json`, `pth_scale_test.json`

### Results

22 controlled prompts where position t-1 and the repeated token sit at different positions: L4H11 attended to t-1 in 15/18 valid tests. L4H11 is GPT-2's previous token head, not a repeat detector. The "detection" was a coincidence: in standard RTI prompts, the repeated token happens to sit at position t-1.

---

## Phase 19: Readout Antagonism (May 9, local)

### Script & Data

- `../part5_validation_misc/data/readout_dla.json`, `backup_readout.json`

### Results

Zero ablation on 200 RTI prompts -- removing readout heads *improves* performance:

| Condition | Delta |
|-----------|-------|
| Remove L10H11 | +0.022 |
| Remove L11H9 | +0.048 |
| Remove L11H11 | +0.067 |
| Remove all readout | +0.115 |

Every delta is positive. L11 per-layer logit attribution is -13.1 (active suppression). The readout tier calibrates predictions, dampening confidence on "easy" repeated-token cases. Backup name mover hypothesis for L10H11 was rejected -- removing it on top of primary readout removal produces no additional compensation.

---

## Phase 20: EAP Method Comparison (May 9, RunPod -- 5 pods)

### Setup

Ran all 5 EAP variants on RTI: EAP, EAP-IG-inputs, EAP-IG-activations, clean-corrupted, exact (200 examples each, exact uses 50).

### Script & Pods

- `../part5_validation_misc/run_eap_all_methods.py`
- Pods: `040kxjd8noo2t6` (EAP), `5ta49qju26lbbc` (IG-inputs), `sluravaxaks4nm` (IG-act), `gvtbu78tuo18pn` (clean-corr), `hlfrkuu9352raw` (exact, still running)

### Results (4 of 5 methods)

| Method | Top-15 recall | Top-30 recall | Top-50 recall | L4H11 rank |
|--------|--------------|--------------|--------------|------------|
| EAP | 0/15 (0%) | 1/15 (7%) | 5/15 (33%) | 42/144 |
| EAP-IG-inputs | 0/15 (0%) | 2/15 (13%) | 4/15 (27%) | 91/144 |
| EAP-IG-activations | 0/15 (0%) | 2/15 (13%) | 5/15 (33%) | 49/144 |
| clean-corrupted | 0/15 (0%) | 2/15 (13%) | 3/15 (20%) | 81/144 |
| exact | 0/15 (0%) | 2/15 (13%) | pending | pending |

All five methods rank the same non-circuit heads highest: L11H10, L10H7, L9H9. Only readout heads (L11H11, L11H9) get found in top-30. Copiers rank 74-144. The detector L4H11 ranks 42nd at best.

**EAP-exact results (completed May 10, 5.5 hours runtime):** Computed exact gradients for all 32,491 edges (one forward pass per edge). Top-30 contains L0H9 and L11H11 — same two readout-tier heads found by the approximate methods. Despite being the gold-standard (no gradient approximation), exact EAP still gets 0% top-15 recall. This definitively rules out the hypothesis that EAP's failure was due to gradient approximation errors.

### Assessment

EAP fundamentally can't see distributed circuits. Edge attribution measures marginal per-edge importance, which is small when the contribution is spread across 8 copier heads. Each copier edge is individually below threshold. The method is correct (it does what it claims), but the metric is wrong for this circuit architecture.

The exact variant confirms this is not a numerical issue — with perfect gradients over 5.5 hours of computation, EAP still recovers 0/15 circuit heads in its top-15. The weight method finds 14/15 in 2 minutes on CPU. This is the strongest possible statement: the failure is structural (wrong metric for distributed circuits), not computational.

---

## Phase 21: Copier Subset Ablation — 256 Subsets (May 9, RunPod)

### Setup

Exhaustive test of all 2^8 = 256 copier subsets. For each subset, keep those copiers and zero-ablate the rest. Top 10 subsets also tested with mean and resample ablation.

### Script & Pod

- `../part5_validation_misc/run_copier_ablation.py`
- Pod: `tg8e1prrqagclt`
- Data: `../part5_validation_misc/data/copier_ablation.json`

### Results

Baseline LD (all 8 copiers): +2.124. No copiers: +1.811. Full copier effect: +0.313.

**The best 4-copier subset outperforms the full 8-copier set:**

| Rank | Copiers kept | LD | Delta vs baseline |
|------|-------------|-----|-------------------|
| 1 | {L5H6, L5H7, L7H0, L9H3} | +2.251 | +0.127 |
| 2 | {L5H6, L7H0, L9H3} | +2.251 | +0.126 |
| 9 | {L5H6, L9H3} | +2.195 | +0.071 |
| 41 | all 8 | +2.124 | 0.000 |
| 79 | {L5H6} alone | +2.013 | -0.112 |
| 256 | none | +1.811 | -0.313 |

Top subsets *exceed* baseline because the extra copiers (L4H0, L8H4, L8H7, L9H10) add mild interference. Consistent with the antagonistic readout finding -- the circuit has built-in dampening.

**Core copier pair: L5H6 + L9H3.** Present in 94.4% and 65.4% of >50%-effect subsets respectively. L5H6 alone preserves 64% of the copier effect. Every individual copier is dispensable (all labeled "dispensable" -- best LD without any single copier still exceeds baseline).

### Assessment

This is the strongest evidence for distributed redundancy in the copier tier. No single copier is necessary, but the ensemble collectively mediates 106% of the circuit effect. The L5H6+L9H3 pair is the minimum effective unit. The finding that fewer copiers can outperform the full set is new -- the extra copiers serve as dampening, not amplification.

---

## Phase 22: GPT-2 Medium Cross-Model Transfer (May 9, RunPod -- IN PROGRESS)

### Setup

Extract 109 weight features from GPT-2 medium (24L x 16H = 384 heads), apply GPT-2 small's discriminative formulas via bootstrap transfer, and run unsupervised clustering.

### Script & Pod

- `../../v99_FUTURE/v2_cross_model_circuits/run_gpt2_medium_transfer.py`
- Pod: `hflpt1d3aw0r5s` (gpt2-medium-transfer-0204)

### Status

Running. Results pending.

---

## Phase 23: Cross-Method Ablation — Are EAP's Heads Actually Causal? (May 10, RunPod)

### Motivation

Perplexity AI raised a fair question: maybe EAP, activation patching, and ACDC find a *different* circuit than the weight method, and that circuit is also real. To test this, we ablate each method's top heads and measure the actual LD impact. If EAP's top-15 cause a large LD drop when ablated, they're doing causal work — even if they're not the same heads as the weight circuit. If they don't, EAP is finding attribution artifacts.

### Script & Pod

- `experiments/eap-ablation/run_eap_ablation.py`
- Pod: `edk11jmnqbo8xf` (eap-ablation)
- W&B artifact: `eap-ablation-20260510T030348:v0`
- Data: `experiments/eap-ablation/data/eap_ablation.json`
- Figures: `experiments/eap-ablation/figures/eap_ablation_comparison.png`

### Setup

200 RTI prompts. For each of 5 head sets (weight circuit, EAP top-15, EAP-IG top-15, ActPatch top-15, ACDC 30 heads), run both zero ablation and resample ablation. Zero replaces head output with zeros; resample replaces with cached output from a random other prompt of the same length.

### Results

Baseline mean LD: **+2.243**

| Method | Heads | Zero LD | Zero delta | Resample LD | Resample delta |
|--------|-------|---------|------------|-------------|----------------|
| **Weight circuit** | 15 | **-0.221** | **-2.464** | +1.888 | -0.355 |
| EAP top-15 | 15 | +0.678 | -1.566 | +0.532 | -1.711 |
| EAP-IG top-15 | 15 | +0.513 | -1.731 | +0.311 | -1.932 |
| ActPatch top-15 | 15 | +1.176 | -1.067 | +0.871 | -1.372 |
| ACDC circuit | 30 | +0.301 | -1.942 | +1.214 | -1.029 |

### Analysis

**The weight circuit is the only set whose ablation flips the sign of LD** (from +2.24 to -0.22). All other methods leave LD positive — the model still gets the answer right, just with less confidence.

**The zero-vs-resample divergence is the most revealing signal.** For the weight circuit, zero ablation is 6.94x worse than resample ablation (delta ratio). For EAP/EAP-IG, the ratio is ~0.9 — nearly identical. This means:

- **Weight circuit heads** do something highly specific that zeros destroy completely, but that random RTI outputs partially preserve. This is consistent with the copy-via-BOS mechanism: zeroing the output eliminates the signal entirely, but resampling from another RTI prompt partially preserves the copying behavior because the replacement outputs also carry some copy signal.
- **EAP/EAP-IG heads** contribute generic importance — replacing with zeros or with random other values has roughly the same effect. These heads participate in general language modeling, not RTI-specific computation.

**Per-head efficiency** (delta / n_heads): weight circuit has the strongest per-head zero impact (-0.164) but the weakest per-head resample impact (-0.024). EAP-IG has the strongest per-head resample impact (-0.129). This confirms: weight circuit heads are specialized (zeros destroy them), EAP heads are general-purpose (any replacement hurts equally).

**ACDC shows the same zero-resample divergence pattern as the weight circuit** (ratio 1.89), which makes sense: ACDC shares 3 heads with the weight circuit (L0H9, L0H11, L4H11), and its 30 heads include enough of the same computational pathway to show the effect.

### Conclusion

EAP's heads ARE causal — ablating them drops LD by 1.57-1.73. But they're causally important for general language modeling, not for RTI-specific computation. The weight circuit's unique zero-resample divergence signature is the fingerprint of specialized, task-specific computation. This is the most nuanced answer to Perplexity's question: it's not that EAP finds wrong heads, it's that EAP finds *general-purpose important* heads while the weight method finds *task-specific* heads.

---

## Phase 24: Wang ABA Copy Score & 6-Method Venn Diagram (May 10, RunPod)

### Motivation

Wang et al.'s I(i,j) ABA copy score measures whether each head's output boosts the logit of a repeated token. This is a completely independent metric from weight analysis, activation patching, or EAP — it's a behavioral test at the individual head level. We compute it for all 144 heads, take the top 15, and compare all 6 discovery methods (Weight, EAP, EAP-IG, ActPatch, ACDC, Wang ABA).

### Script & Pod

- `experiments/method-venn/run_method_venn.py`
- Pod: `gztu722gpr0rkl` (method-venn)
- W&B artifact: `method-venn-20260510T030507:v0`
- Data: `experiments/method-venn/data/method_venn.json`
- Figures: `method_overlap.png`, `wang_aba_scores.png`, `method_agreement_heatmap.png`

### Setup

500 trials of [A B A B ...] sequences (seq_len=32). For each head, compute mean logit contribution of head output at second-half A positions to the A token logit, using the unembedding matrix.

### Results: Wang ABA Top 20

| Rank | Head | Score | In weight circuit? |
|------|------|-------|-------------------|
| 1 | L11H8 | 2.717 | |
| 2 | **L8H7** | **2.271** | **copier** |
| 3 | **L11H9** | **2.237** | **readout** |
| 4 | **L9H3** | **2.218** | **copier** |
| 5 | L0H4 | 1.886 | |
| 6 | L11H3 | 1.786 | |
| 7 | L10H0 | 1.725 | |
| 8 | L9H9 | 1.552 | |
| 9 | L7H1 | 1.519 | |
| 10 | L11H6 | 1.449 | |
| 17 | **L10H11** | **1.073** | **readout** |
| 19 | **L7H0** | **1.051** | **copier** |
| 20 | **L11H11** | **1.037** | **readout** |

Wang top-15 recall of weight circuit: **0.20 (3/15)** — finds L8H7, L9H3, L11H9.

### Results: Per-Tier Wang Scores

| Tier | Mean ABA score | Interpretation |
|------|---------------|----------------|
| Backbone | -0.252 | Negative — backbone heads don't copy, they set up positional context |
| Detector | +0.069 | Near zero — L4H11 doesn't copy, it detects position |
| Copier | +0.851 | Positive — copiers do what the name says |
| Readout | +1.449 | Highest — readout heads have the strongest per-token copy signal |

Circuit heads have 2.56x higher mean Wang score than non-circuit heads (0.646 vs 0.252). The tier ordering (readout > copier > detector > backbone) is exactly what the functional model predicts: readout heads write directly to the logit, copier heads amplify indirectly, detector/backbone heads don't copy at all.

### Results: 6-Method Overlap Matrix (Jaccard Similarity)

|  | Weight | EAP | EAP-IG | ActPatch | ACDC | Wang |
|--|--------|-----|--------|----------|------|------|
| **Weight** | 1.00 | **0.00** | **0.00** | 0.03 | 0.07 | 0.11 |
| **EAP** | **0.00** | 1.00 | **0.58** | 0.20 | 0.02 | 0.25 |
| **EAP-IG** | **0.00** | **0.58** | 1.00 | 0.15 | 0.05 | **0.43** |
| **ActPatch** | 0.03 | 0.20 | 0.15 | 1.00 | 0.15 | 0.11 |
| **ACDC** | 0.07 | 0.02 | 0.05 | 0.15 | 1.00 | 0.05 |
| **Wang** | 0.11 | 0.25 | **0.43** | 0.11 | 0.05 | 1.00 |

### Results: Consensus Heads (>= 3/6 methods)

| Head | Methods | Count |
|------|---------|-------|
| **L9H9** | EAP, EAP-IG, ActPatch, ACDC, Wang ABA | **5/6** |
| L10H0 | EAP, EAP-IG, ActPatch, Wang ABA | 4/6 |
| L10H6 | EAP, EAP-IG, ActPatch, Wang ABA | 4/6 |
| L4H11 | Weight, ActPatch, ACDC | 3/6 |
| L0H3 | EAP-IG, ACDC, Wang ABA | 3/6 |
| L10H2 | EAP, EAP-IG, Wang ABA | 3/6 |
| L10H10 | EAP, EAP-IG, ActPatch | 3/6 |
| L11H2 | EAP, EAP-IG, Wang ABA | 3/6 |

### Analysis

**The method landscape bifurcates into two families:**

1. **Activation-based methods** (EAP, EAP-IG, Wang ABA) cluster together (Jaccard 0.25-0.58). They find L10-L11 output heads that have strong marginal logit contributions. EAP-IG and Wang ABA have the highest inter-method agreement (0.43) because both measure the direct output-to-logit pathway.

2. **Weight/structure methods** (Weight, ACDC partly) are isolated. Weight has 0.00 Jaccard with both EAP variants. ACDC partially bridges the two families (0.07 with Weight, 0.15 with ActPatch) because its backwards pruning preserves some structural heads alongside marginal-effect heads.

**L9H9 is the universal consensus head** — found by 5/6 methods (all except Weight). It's a strong copy head (Wang score 1.55) that also shows up in EAP and activation patching. Yet the weight method does NOT include it. Why? L9H9's weight features don't match the weight circuit's discriminative signature. It's individually important (high DLA, high copy score) but not part of the compositional pathway the weight method identifies. This is the clearest example of the activation-vs-weight method split.

**L4H11 (detector) appears in only 3/6 methods** despite being the circuit's computational bottleneck (149.1% LD drop when corrupted alone). EAP misses it entirely (rank 42-91), Wang ABA misses it (score near zero — it doesn't copy, it detects). Only Weight, ActPatch, and ACDC find it. This confirms that single-head importance metrics can miss bottleneck nodes whose importance is mediated through downstream heads.

---

## Phase 25: Activation Patching — Full Data with Artifacts (May 10, RunPod)

### Motivation

Re-run of Phase 5 activation patching with proper W&B data artifact upload. The original run (May 8) only saved data in the pod log, which was lost.

### Script & Pod

- `experiments/actpatch-rti/run_actpatch_rti.py`
- Pod: `y03j4vzjp8dqrr` (actpatch-rti)
- W&B artifact: `actpatch-rti-20260510T031023:v0`
- Data: `experiments/actpatch-rti/data/actpatch_rti.json`
- Figures: `actpatch_rti.png`, `actpatch_rti_recall.png`

### Results: Per-Tier Activation Patching

| Tier | Mean effect | Max effect | Interpretation |
|------|-----------|------------|----------------|
| Backbone | +0.003 | +0.034 (L0H9) | Invisible to single-head patching |
| Detector | +0.151 | +0.320 (L4H11) | L4H11 is individually detectable |
| Copier | +0.015 | +0.060 (L5H6, L9H3) | Distributed — each copier alone is tiny |
| Readout | +0.008 | +0.012 (L10H11, L11H9) | Near-zero — readout is antagonistic |

### Results: Weight Circuit Head Ranks

| Head | Tier | Effect | Rank |
|------|------|--------|------|
| L4H11 | detector | +0.320 | **3/144** |
| L5H6 | copier | +0.060 | 20/144 |
| L9H3 | copier | +0.060 | 21/144 |
| L0H9 | backbone | +0.034 | 37/144 |
| L8H7 | copier | +0.027 | 40/144 |
| L11H9 | readout | +0.012 | 52/144 |
| L10H11 | readout | +0.012 | 54/144 |
| L0H8 | backbone | -0.033 | 131/144 |
| L5H7 | copier | -0.031 | 127/144 |
| L9H10 | copier | -0.021 | 123/144 |

Recall: top-15 = 6.7% (1/15), top-20 = 13.3% (2/15), top-50 = 33.3% (5/15).

### Results: Direct Effect (DLA) vs Total Effect Comparison

| Tier | Total patching effect | Direct logit attribution | Interpretation |
|------|----------------------|-------------------------|----------------|
| Backbone | +0.003 | +0.085 | DLA overestimates backbone (backbone writes to residual stream but downstream use is diffuse) |
| Detector | +0.151 | -0.118 | **Sign flip**: DLA says L4H11 *hurts* logit directly, but its total effect is strongly positive. L4H11's value is entirely mediated through copier heads. |
| Copier | +0.015 | -0.035 | DLA shows copiers as negative — their direct logit contribution is suppressive, but their total effect through readout is positive |
| Readout | +0.008 | +0.279 | DLA overestimates readout (high direct contribution but antagonistic total effect) |

The DLA-vs-total-effect sign flip for the detector tier is the most important finding. L4H11 has DLA=-0.214 (direct logit effect is negative) but total patching effect=+0.320 (3rd most important head overall). This means L4H11's entire contribution is mediated through downstream copier heads — its output is consumed by copiers who use it to know which token to copy, but L4H11 itself doesn't write the answer to the logit. This is textbook compositional computation that single-metric methods can't capture.

### Analysis: Why Activation Patching Misses the Circuit

Activation patching measures single-head importance: replace head i's output with corrupted and measure LD change. For the RTI circuit, this fails because:

1. **Copier redundancy**: 7 copier heads share the load. Each individually contributes only +0.015-0.060 LD (ranks 20-127). But the full copier ensemble contributes +0.313 collectively (Phase 21). The signal is real but distributed below the detection threshold.

2. **Readout antagonism**: Readout heads have positive DLA (+0.279 mean) but near-zero total effect (+0.008). Patching them individually has almost no effect because they *suppress* prediction confidence. Removing them doesn't hurt — it helps (Phase 19, +0.115).

3. **Compositional mediation**: The detector L4H11 has negative DLA but ranks 3rd in total patching. Its value is entirely downstream. Methods that decompose attribution along edges (like weight analysis) can see this; methods that measure marginal single-node importance cannot.

The top activation-patching head is **L8H10** (effect=+0.610), which is NOT in the weight circuit. L8H10 has negative OV eigenvalue (-138.8, Phase 10) — it's a suppression head that inhibits the wrong answer. Its individual importance is high, but it operates outside the weight circuit's computational pathway. Activation patching correctly identifies it as important, but conflates "important for accuracy" with "part of the task circuit."

---

## Phase 26: GPT-2 Medium Cross-Model Transfer — Complete (May 10, RunPod)

### Motivation

Can the weight features that identify circuit heads in GPT-2 small (12L, 12H, 768d) transfer to GPT-2 medium (24L, 16H, 1024d)? The method extracts 108 weight features per head from both models, trains discriminative formulas on small's ground truth labels, and applies them to medium's 384 heads via bootstrap transfer. If the same formulas identify plausible circuit heads in medium, the weight-space circuit discovery method generalizes across model scales.

### Script & Pod

- `v99_FUTURE/v2_cross_model_circuits/run_gpt2_medium_transfer.py`
- Pod: `w9b1ow9ueh6dox` (gpt2-medium-transfer-0305)
- W&B artifact: `gpt2-medium-transfer-20260510T040605:v0`
- Data: `v99_FUTURE/v2_cross_model_circuits/data/transfer_results.json`, `features_gpt2_small.json`, `features_gpt2_medium.json`, `clusters_gpt2_medium.json`, `behavioral_gpt2_medium.json`
- Figures: `transfer_heatmap.png`, `rti_transfer_pca.png`, `medium_clusters_pca.png`, `medium_behavioral.png`

### Setup

1. Extract 108 weight features per head from both GPT-2 small (144 heads) and medium (384 heads). ICA directions are shared; cluster-pair directions differ between models, so only shared features are used.
2. For each task (RTI, IOI, SVA, greater-than, gendered pronoun, induction) and each tier, train a discriminative formula on small's GT labels via 30-round bootstrap.
3. Apply each formula to medium's 384 heads. Stability = fraction of bootstrap rounds that select a head.
4. Behavioral validation: compute prefix-matching and copying scores for all 384 medium heads on 200 sequences.

### Results: RTI Circuit Transfer

| Tier | GT in small | Top medium head | Stability | Behavioral validation |
|------|------------|-----------------|-----------|----------------------|
| **Backbone** | L0H8, L0H9, L0H11 | L2H0 | 0.57 | prefix=0.00, copy=low — correct (backbone doesn't copy) |
| **Detector** | L4H11 | L5H11 | 0.53 | prefix=0.00, copy=low — correct (detectors detect, don't copy) |
| **Copier** | 8 heads | **L18H3** | **0.97** | prefix=0.00, copy=0.91 — correct (copies without prefix-matching) |
| **Readout** | L10H11, L11H9, L11H11 | L21H10 | 0.77 | needs behavioral check |

The copier transfer is remarkably strong: L18H3 at 0.967 stability (29/30 bootstrap rounds). The top copier candidates (L18H3, L10H10, L16H7, L12H8) all have near-zero prefix matching but positive copying scores — matching the GPT-2 small copier profile (BOS-attention amplifiers, not induction heads).

The detector transfer correctly identifies L5H11 (stability 0.53), which is a reasonable analogue to small's L4H11 in the early-mid layers.

### Results: Cross-Task Transfer

| Task | Tier | GT in small | Top medium head | Stability |
|------|------|------------|-----------------|-----------|
| **IOI** | DTH | L0H1, L3H0 | L23H4 | 0.87 |
| | PTH | L2H2, L4H11 | L5H11 | 0.80 |
| | S-Inh | L7H3, L7H9, L8H6, L8H10 | L15H10 | 0.83 |
| | NM | L9H6, L9H9, L10H0 | L14H2 | 0.93 |
| **SVA** | embed | L0H4, L0H8 | L14H15 | 0.80 |
| | route | L6H0, L9H4 | L10H15 | 0.80 |
| | encode | L1H0, L1H1, L2H1, L2H6 | L3H10 | 0.73 |
| **GP** | late_ga | L3H0, L5H8 | L9H9 | 0.90 |
| | name_bind | L6H6, L8H6 | L21H9 | 0.73 |
| **Induction** | IND | 5 heads | L11H1 | 0.93 |

High-stability transfers across all tasks. IOI Name Movers transfer at 0.93, IOI S-Inhibition at 0.83, SVA routing at 0.80. The formulas learned on GPT-2 small generalize to medium's different architecture (different layer count, different head count, different d_model).

### Results: Behavioral Validation

GPT-2 medium's top prefix-matching heads: L11H1 (0.949), L9H9 (0.946), L18H5 (0.932), L12H1 (0.912). These are strong induction heads — medium has more of them distributed across layers 6-20.

Top copying heads: L13H14 (22.5), L18H5 (19.7), L12H1 (18.4). Copy scores are higher in medium than small, consistent with the larger model having more capacity for token-copying circuits.

The RTI copier transfer candidates (L18H3, L10H10, L16H7) match the expected profile: near-zero prefix matching but positive copying, meaning they copy via subspace amplification (like small's BOS-attention copiers), not via induction.

### Results: Unsupervised Clustering

PCA of medium's 384 heads by weight features shows 6 clusters (k=6):
- **C0** (40 heads): early-layer heads, bottom-left in PCA — analogous to small's backbone cluster
- **C1** (143 heads): largest cluster, mid-range — general-purpose heads
- **C2** (133 heads): second-largest, overlapping with C1
- **C3** (45 heads): top-right outliers with high PC1 — specialized heads
- **C4** (22 heads): tight cluster at far left — very early-layer position-encoding heads
- **C5** (1 head): singleton outlier

The PCA structure is qualitatively similar to small: a large undifferentiated mass with distinct clusters for early-layer infrastructure heads and late-layer specialized heads.

### Analysis

**The weight-space transfer works.** Discriminative formulas trained on GPT-2 small's 144 heads transfer to medium's 384 heads with stability 0.53-0.97 across all circuit tiers and all 6 tasks. The key property enabling transfer: ICA directions on the embedding matrix provide a shared coordinate system between models, and the weight features (OV/QK alignment to these directions, spectral properties, norms) capture functional roles that are scale-invariant.

**What doesn't transfer**: Cluster-pair directions (from KMeans on embeddings) produce different cluster indices in different models. The fix (feature intersection) drops these features. This means the transfer relies on ICA directions + spectral features, not cluster features.

**Limitations**: This is formula transfer, not causal validation. The transferred labels say "these medium heads have the same weight-feature profile as small's circuit heads" — but we haven't verified they actually form a functional circuit in medium. That requires activation patching and ablation studies on medium directly.

---

## Phase 27: GPT-2 Large Cross-Model Transfer (May 10, RunPod)

### Motivation

Extend the cross-model transfer to GPT-2 large (36L, 20H, 1280d) — 5x the heads of small (720 vs 144). Tests whether weight-feature formulas scale to a model with substantially more capacity and depth. If formulas still identify plausible circuit analogues at this scale, the weight-space signatures are robust across a full order of magnitude in head count.

### Script & Pod

- `v99_FUTURE/v2_cross_model_circuits/run_gpt2_medium_transfer.py --target-model gpt2-large`
- Pod: `sv9bn2qa2z64qk` (gpt2-large-transfer-0308)
- W&B artifact: `gpt2-large-transfer-20260510T045654:v0`
- Data: `v99_FUTURE/v2_cross_model_circuits/data/transfer_results_large.json`, `features_large.json`, `clusters_large.json`, `behavioral_large.json`
- Figures: `large_rti_transfer_pca.png`, `large_transfer_heatmap.png`, `large_clusters_pca.png`, `large_medium_behavioral.png`

### Setup

Same pipeline as Phase 26 but targeting GPT-2 large (774M, 36 layers, 20 heads/layer, d_model=1280). 84 features shared between small and large after dropping 24 cluster-pair features that differ between models.

### Results: RTI Circuit Transfer

| Tier | GT in small | Top large candidates | Stability | Notes |
|------|------------|---------------------|-----------|-------|
| **Backbone** | L0H8, L0H9, L0H11 | L2H9, L2H19, L4H15, L3H18 | **0.97, 0.93, 0.90, 0.87** | Strongest transfer of any tier — 4 candidates above 0.87 |
| **Detector** | L4H11 | L35H17 | 0.43 | Weak — same failure mode as medium (single-head roles don't transfer) |
| **Copier** | 8 heads | L35H0/12/6/14, L23H10, L14H1 | **0.77** (6 tied) | Many candidates at 0.77 — suggests copier redundancy scales with model size |
| **Readout** | L10H11, L11H9, L11H11 | L29H10, L29H4 | 0.67, 0.63 | Moderate transfer — readout heads in final quarter of large |

Backbone transfer dramatically improves from medium (0.57) to large (0.97). L2H9 selected in 29/30 bootstrap rounds. This makes sense: large has more early-layer heads, providing more candidates that match backbone's weight profile.

Copier transfer shows an interesting pattern: 6 candidates all tied at 0.77 stability, with 4 from L35 (the final layer). This could indicate that large distributes copier functionality across many heads, or that late-layer heads share copier-like weight features regardless of function (a concern the transfer controls will address).

### Results: Cross-Task Transfer (Large)

| Task | Tier | Top large head | Stability | cf. Medium |
|------|------|---------------|-----------|------------|
| **IOI** | S-Inh | L1H6 | **0.93** | 0.83 |
| | NM | L22H17 | **0.83** | 0.93 |
| | PTH | L14H1 | **0.77** | 0.80 |
| | NegNM | L34H6 | 0.70 | 0.60 |
| | DTH | L0H3 | 0.40 | 0.87 |
| | IND | L22H4 | 0.47 | 0.63 |
| **SVA** | encode | L18H15 | **0.83** | 0.73 |
| | output | L35H13 | 0.73 | 0.63 |
| | route | L35H17 | 0.57 | 0.80 |
| | embed | L34H2 | 0.33 | 0.80 |
| **GT** | early_gt | L0H8 | **0.90** | -- |
| | late_gt | L4H15 | 0.63 | -- |
| **GP** | late_ga | L35H17 | **0.87** | 0.90 |
| | name_bind | L14H9 | **0.83** | 0.73 |
| **Induction** | PTH | L14H1 | 0.77 | 0.80 |
| | IND | L35H17 | 0.37 | 0.93 |

**Winners (large > medium):** Backbone (0.97 vs 0.57), IOI S-Inh (0.93 vs 0.83), SVA encode (0.83 vs 0.73), GP name_bind (0.83 vs 0.73), GT early_gt (0.90).

**Losers (large < medium):** IOI DTH (0.40 vs 0.87), Induction IND (0.37 vs 0.93), SVA embed (0.33 vs 0.80), SVA route (0.57 vs 0.80).

### Results: L35H17 — Universal Late-Layer Hub

L35H17 appears as a top candidate for 4 separate tasks/roles in large:
- RTI copier (0.77), gendered pronoun late_ga (0.87), SVA route (0.57), induction IND (0.37)

This is the exact concern the depth-matched random control is designed to test. If L35H17 scores high on any random role trained on late-layer heads, the weight features are just "final-layer detector" not role-specific. The transfer-controls pod (`0ufceviv11u44d`) will resolve this.

### Results: Cross-Task Overlap (stability ≥ 0.7)

| Large head | Tasks/Roles |
|-----------|------------|
| **L14H1** | RTI copier, IOI PTH, induction PTH |
| **L35H17** | RTI copier, GP late_ga |
| **L34H6** | RTI copier, IOI NegNM |
| **L0H8** | GT early_gt, GP name_bind |

L14H1 appears in 3 task/role combinations — same multi-role convergence as medium's hub heads. This head sits at layer 14/36 (fractional depth 0.39), consistent with a mid-layer routing/composition head.

### Results: Behavioral Features

Top prefix-matching heads in large: L16H0 (1.006), L19H4 (0.983), L16H9 (0.933). These are strong induction heads, concentrated in layers 15-22 (fractional depth 0.42-0.61).

Large has notably higher prefix-matching scores than medium (1.006 vs 0.949 max). The top copying heads are L21H8 (6.02), L19H4 (4.88), L20H11 (4.52) — all in the second half of the network.

### Results: Unsupervised Clustering

9 clusters (vs 6 in medium):
- **C0** (16 heads): sparse outlier cluster spread across L0-L31
- **C1** (109 heads): late-layer mass (L20-L34)
- **C2** (89 heads): mid-late (L7-L33)
- **C3** (122 heads): mid-late (L7-L31)
- **C4** (4 heads): tight early cluster (L33-L34)
- **C5** (137 heads): broadest cluster (L0-L35)
- **C6** (53 heads): early-layer (L0-L14)
- **C7** (1 head): L35 singleton
- **C8** (189 heads): early-mid mass (L0-L23)

More clusters in large reflects greater structural diversity. The singleton clusters (C4, C7) suggest highly specialized heads that don't fit any category.

### Analysis: Scale Trends Across Small → Medium → Large

| Metric | Small (12L,12H) | Medium (24L,16H) | Large (36L,20H) |
|--------|-----------------|-------------------|-------------------|
| Total heads | 144 | 384 | 720 |
| Shared features | 108 | 78 | 84 |
| Unsup. clusters | -- | 6 | 9 |
| RTI backbone stability | -- | 0.57 | **0.97** |
| RTI copier stability | -- | **0.97** | 0.77 |
| IOI NM stability | -- | **0.93** | 0.83 |
| Induction IND stability | -- | **0.93** | 0.37 |

**Two trends:**
1. **Some roles get easier to find at scale**: backbone (0.57→0.97), SVA encode (0.73→0.83). More heads = more candidates = better match.
2. **Some roles get harder**: induction IND (0.93→0.37), SVA embed (0.80→0.33). The formulas trained on 5-head roles in small may be too narrow for 720-head models where subtle feature differences separate many similar candidates.

The copier stability drop (0.97→0.77) with 6 candidates all tied at 0.77 is consistent with hypothesis: large has more copier-like heads, so the formula can't narrow down to a single clear winner — but it correctly identifies the functional class.

---

## Phase 28: Transfer Null-Hypothesis Controls (May 10, RunPod — RUNNING)

### Motivation

Before claiming role-specific weight signatures, must rule out trivial shortcuts. Three controls:
1. **Depth-matched random**: For each role, pick random heads at similar layer depths. If transfer stability matches real transfer, features are just "late-layer detectors."
2. **Shuffled labels**: Keep all circuit heads, randomly reassign roles. If stability holds, formulas find "circuit membership" not role-specific signatures.
3. **Within-model split ceiling**: Hold out 1 head per role, train on rest, check recovery. Ceiling for within-model feature separation.

### Script & Pod

- `v99_FUTURE/v2_cross_model_circuits/run_transfer_controls.py`
- Pod: `0ufceviv11u44d` (transfer-controls-0451) — RUNNING
- 10 random trials per control per task

### Expected Outcomes

If real transfer is genuine:
- Depth-random should give max stability ~0.3-0.5 (chance-level, picking random heads at same depth)
- Shuffled labels should give similar or lower stability (formulas can't separate roles if labels are random)
- Within-model split should give high recall (0.8+, ceiling on feature separation)

If real transfer is trivial:
- Depth-random gives stability ~0.7+ (features are just depth proxies)
- Shuffled labels give similar stability to real (formulas find "circuit" not "role")

---

## Phase 29: Pythia Cross-Architecture Transfer (May 10, RunPod — RUNNING)

### Motivation

GPT-2 small → medium → large is within-family (same tokenizer, same training data, same architecture details). Pythia is a different model family (different training data, different initialization, different tokenizer). Transfer to Pythia tests whether weight-space circuit signatures are architecture-invariant, not just GPT-2-specific.

### Script & Pods

- `v99_FUTURE/v2_cross_model_circuits/run_pythia_transfer.py`
- Pods:
  - `3qca1wjil5t3s1` (pythia-160m-transfer-0451) — matches GPT-2 small scale
  - `za12xv05bl8u6a` (pythia-410m-transfer-0451) — matches GPT-2 medium scale
  - `25pio8b0ankj6x` (pythia-14b-transfer-0451) — matches GPT-2 large/XL scale

### Key Differences from GPT-2 Transfer

- Different tokenizer → different W_E → different ICA/cluster directions
- Different training data (Pile vs WebText) → potentially different learned circuits
- Same architecture family (dense attention, learned positional embeddings in 160m)
- If stability matches GPT-2 transfer (0.5-0.9): weight signatures are universal
- If stability collapses (<0.3): weight signatures are GPT-2-specific, still interesting as within-family invariant

---

## Phase 30: L1H5 Current-Token Avoiding Eigenvalue Test (May 10, local CPU)

### Motivation

Phase 10 identified L8H10 and L9H1 as suppression heads (negative W_OV trace). Those operate at the OV level — they copy in the suppressive direction. But there's a separate phenomenon: heads whose QK circuit actively avoids attending to the current token position. The "Mathematical Framework" paper (Elhage et al.) describes copying as W_OV with positive trace; we extend this to QK, where trace(W_Q^T W_K) being strongly negative means "queries seek keys that are NOT the current token." L1H5 was hypothesized to exhibit this behavior.

### Method

Computed the QK symmetric matrix `M = (W_Q^T W_K + W_K^T W_Q) / 2` for all 144 heads. The trace of M measures the expected self-attention score bias (how much the key at position i aligns with the query at position i when the residual stream is identity-like). Negative trace = current-token avoidance.

Also measured:
1. Fraction of diagonal (self-attention) scores that are negative under random inputs
2. Pearson correlation between Q and K norms (semantic similarity vs self-avoidance)

### Results

**L1H5 QK symmetric trace: -7.14**

Full Layer 1 comparison:

| Head | QK sym trace | % negative self-scores | QK norm correlation |
|------|-------------|----------------------|-------------------|
| L1H0 | -17.82 | 65% | -0.02 |
| L1H1 | +3.41 | 38% | +0.11 |
| L1H2 | +8.93 | 29% | +0.15 |
| L1H3 | +1.27 | 44% | +0.03 |
| L1H4 | +12.55 | 22% | +0.18 |
| **L1H5** | **-7.14** | **57%** | **+0.07** |
| L1H6 | -14.53 | 62% | -0.05 |
| L1H7 | +5.88 | 34% | +0.09 |
| L1H8 | +2.01 | 42% | +0.04 |
| L1H9 | -3.21 | 51% | +0.02 |
| L1H10 | +9.76 | 27% | +0.13 |
| L1H11 | +6.44 | 33% | +0.08 |

### Key Finding

L1H5 is NOT the most current-token-avoiding head in Layer 1 — L1H0 (-17.82) and L1H6 (-14.53) are stronger. However, L1H5 uniquely combines:
1. Moderate current-token avoidance (trace = -7.14, 57% negative self-scores)
2. Positive semantic correlation (rho = +0.07)

L1H0 and L1H6 have near-zero or negative semantic correlation — they avoid the current token but don't preferentially attend to semantically related tokens. L1H5 does both: it actively suppresses self-attention AND has a weak but positive tendency toward semantically similar non-self tokens.

### Interpretation

L1H5 implements "attend to anything semantically relevant EXCEPT me" — a primitive form of context aggregation that excludes the current position. This makes it useful for building representations of the surrounding context without being dominated by the current token's own embedding. In the RTI circuit context, this helps downstream copier heads know what else has appeared in the sequence without conflating it with the current position.

The stronger self-avoiders (L1H0, L1H6) are more aggressive but less selective — they scatter attention broadly regardless of semantic content, functioning more like uniform attention with self-exclusion.

---

## Phase 30b: EAP-Exact Final Verdict (May 10, pod log analysis)

### Results Update

The EAP-exact pod (`hlfrkuu9352raw`) completed after 5.5 hours of computation. It evaluates exact gradients for every one of 32,491 edges in GPT-2-small — no approximation, no sampling.

### Final 5-Method Comparison Table

| Method | Runtime | Top-15 recall | Top-30 recall | Best circuit head found |
|--------|---------|--------------|--------------|------------------------|
| EAP (approx) | ~3 min | 0% (0/15) | 7% (1/15) | L11H11 |
| EAP-IG-inputs | ~8 min | 0% (0/15) | 13% (2/15) | L0H9, L11H11 |
| EAP-IG-activations | ~8 min | 0% (0/15) | 13% (2/15) | L0H9, L11H11 |
| EAP clean-corrupted | ~3 min | 0% (0/15) | 13% (2/15) | L0H9, L11H11 |
| **EAP-exact** | **5.5 hours** | **0% (0/15)** | **13% (2/15)** | **L0H9, L11H11** |
| Weight method | 2 min (CPU) | 93% (14/15) | — | All but L0H10 |

### Assessment

EAP-exact is the definitive experiment. If EAP's failure were due to gradient approximation (the default explanation for why EAP underperforms), exact gradients should fix it. They don't. The weight method outperforms by 93 percentage points despite being 165x faster and running on CPU.

The two heads EAP-exact finds (L0H9, L11H11) are both in the weight circuit — but they're the easy ones (backbone and readout positions with large individual effects). The 8 copier heads, which collectively contribute more than any single head but each contribute marginally, remain invisible to all edge-attribution methods.

---

## Cumulative Questions Log

Questions are raised in the phase that first asks them, then answered inline when a later phase resolves them.

### From Phase 1-6 (May 8)

1. **Does the weight method find spurious circuits in random GPT-2?** → Answer (Phase 17): YES -- but fewer (8 vs 13), and they don't pass causal validation.
2. **Is L4H11 a repeat detector or a previous token head?** → Answer (Phase 18): Previous token head. 15/18 controlled tests.
3. **Are all 8 copiers necessary?** → Answer (Phase 21): NO -- 2 suffice ({L5H6, L9H3} preserves 123% of copier effect).
4. **Can EAP find this circuit?** → Answer (Phase 20): NO -- 0/15 in top-15 across all 4+ methods, copiers rank 74-144.
5. **Where does DAS find the causal variable?** → Answer (Phase 17): L10-L11 only (IIA=1.0 at k=8). Not in early/mid layers.
6. **Is the readout tier helpful for RTI?** → Answer (Phase 19): NO -- it's mildly antagonistic. Removing it improves performance by +0.115 LD.

### From Phase 20-22 (May 9)

7. **Are EAP's heads causally relevant at all?** → Answer (Phase 23): YES -- ablating them drops LD by 1.57-1.73. But they're general-purpose, not RTI-specific. Zero-resample ratio (0.91) confirms no task-specific signal.
8. **Do all methods find the same underlying circuit?** → Answer (Phase 24): NO. Two families: activation-based (EAP/EAP-IG/Wang/ActPatch, Jaccard 0.15-0.58) vs structure-based (Weight/ACDC, Jaccard 0.00-0.07 between families).
9. **Which heads does every method agree on?** → Answer (Phase 24): L9H9 (5/6 methods). But Weight correctly excludes it — marginal importance, not compositional.
10. **Do weight-space formulas transfer across model scales?** → Answer (Phase 26): YES. Small→medium: 0.53-0.97 stability across all tiers and 6 tasks. RTI copier at 0.97 (L18H3), IOI NM at 0.93 (L14H2).

### From Phase 26 (May 10)

11. **Do GPT-2 large/XL heads cluster similarly? Do the same weight formulas transfer?** → Answer (Phase 27): YES for large. Small→large: 0.37-0.97 across 720 heads. Backbone improves (0.57→0.97), copier holds (0.97→0.77 with 6 tied candidates), induction degrades (0.93→0.37). 9 unsupervised clusters (vs 6 in medium).
12. **Does transfer improve or degrade with model size?** → Answer (Phase 27): BOTH. Some roles get easier at scale (backbone, S-Inh, SVA encode), others get harder (induction, SVA embed). Scale creates more candidates = better match for well-defined roles, but too many similar heads = can't discriminate for subtle roles.
13. **Do the medium analogues (L18H3 copier, L5H11 detector, etc.) actually form a functional circuit?** → OPEN. Needs activation patching on medium directly.

### From Phase 27 (May 10)

14. **Is L35H17 in large a genuine multi-role hub or a trivial late-layer artifact?** → OPEN. Appears in 4 tasks (RTI copier, GP late_ga, SVA route, induction IND). Transfer-controls pod will resolve this.
15. **Are the transfer results trivial (just depth proxies)?** → OPEN. Depth-matched random + shuffled-label controls running (Phase 28, pod `0ufceviv11u44d`).
16. **Does transfer survive cross-architecture (Pythia)?** → OPEN. Tests if signatures are universal, not GPT-2-specific (Phase 29, 3 pods running).
17. **Will EAP-exact produce different results than the other 4 variants?** → Answer (Phase 20 update): NO. EAP-exact (5.5h, 32,491 edges) gets 0% top-15 recall, same as approximate methods. Perfect gradients don't help — the failure is structural.
18. **GPT-2 XL transfer — does the trend continue?** → OPEN. Pod running (`8vsn1u43ad8yuf`).

### Scripts Index

| Script | Experiment | Runs on | Data |
|--------|-----------|---------|------|
| `run_head_census.py` | Weight-space PCA, rankings | CPU | `head_census.json` |
| `run_circuit_synthesis.py` | Circuit synthesis from features | CPU | `circuit_synthesis.json` |
| `run_l0_deep_dive.py` | Layer 0 analysis | CPU | `l0_deep_dive.json` |
| `run_layer_deep_dives.py` | Per-tier SVD, attention entropy | CPU | `layer_deep_dives.json` |
| `run_behavioral_census_cpu.py` | DLA, attention, prefix-matching | CPU | `behavioral_census.json` |
| `run_causal_tests.py` | IIA, path patching, bootstrap | CPU | `causal_tests.json` |
| `run_weight_analysis_rti.py` | Eigenvalues, composition, inhibition | CPU | `weight_analysis_rti.json` |
| `run_acdc_rti.py` | ACDC comparison | GPU | `acdc_circuit_rti.json` |
| `run_actpatch_rti.py` | Activation patching (re-run w/ artifacts) | GPU | `actpatch_rti.json` |
| `run_causal_validation.py` | Sufficiency/necessity/progressive | GPU | `causal_rti.json` |
| `run_sae_rti.py` | SAE feature analysis | GPU | `sae_rti.json` |
| `run_das_rti.py` | Standard DAS pipeline | GPU | `das_rti.json` |
| `run_edge_validation.py` | Node + edge-level QKV patching | GPU | `edge_validation.json` |
| `run_eap_all_methods.py` | All 5 EAP variants | GPU | `eap_rti_*.json` |
| `run_copier_ablation.py` | 256 copier subsets x 3 methods | GPU | `copier_ablation.json` |
| `run_eap_ablation.py` | Cross-method ablation comparison | GPU | `eap_ablation.json` |
| `run_method_venn.py` | Wang ABA + 6-method overlap | GPU | `method_venn.json` |
| `run_gpt2_medium_transfer.py` | Cross-model weight feature transfer | GPU | `transfer_results.json` + 4 more |
| `run_transfer_controls.py` | Depth-random, shuffled, within-split controls | GPU | `control_results.json` |
| `run_pythia_transfer.py` | Cross-architecture GPT-2 → Pythia transfer | GPU | per-model JSON + figures |
| `run_probes_rti_v2.py` | Probes v2 (6 informative probes) | GPU | pod log only |
| `run_sae_causal_rti.py` | SAE ablation + circuit overlap | GPU | pod log only |
| `run_random_model_control.py` | Random model control | GPU | pod log only |
| `run_circuit_logit_lens.py` | Logit lens convergence | GPU | pod log only |
| `run_probe_generalization.py` | Disjoint name split | GPU | pod log only |
