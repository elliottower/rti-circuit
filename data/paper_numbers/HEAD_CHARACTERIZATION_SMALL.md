# RTI Circuit: Complete Head Characterization

15 attention heads in GPT-2 small that implement Repeated Token Identification.
All data from weight-space analysis + causal validation — no activation-based discovery.

## Important: Two Circuit Versions Exist

**There are two different circuits in this repo.** The diagrams and the code disagree.

The **old circuit** (from `diagram_residual_stream.png`, `diagram_circuit_overview.png`, `diagram_ioi_style_circuit.png`, and the `in_circuit` flags in `weight_analysis_rti.json`) uses:

| Tier | Heads |
|------|-------|
| Backbone | L0H9, L0H10, L4H11 |
| Repeat-finders | L7H2, L7H9, L7H11 |
| Repeat-finders / Suppressors | L8H6, L8H10, L8H11 |
| Answer-extractor / Suppressor | L9H1, L9H6 |
| Readout | L9H9, L10H0, L10H2, L10H10 |

The **current canonical circuit** (from `roles.py` lines 148-173, used by all experiment scripts) uses:

| Tier | Heads |
|------|-------|
| Backbone | L0H8, L0H9, L0H11 |
| Detector | L4H11 |
| Copier | L4H0, L5H6, L5H7, L7H0, L8H4, L8H7, L9H3, L9H10 |
| Readout | L10H11, L11H9, L11H11 |

**This document describes the current canonical circuit.** The old circuit was from an earlier bootstrap pass; the current one was re-derived with the greedy feature selection + 30-round bootstrap + causal validation pipeline described in `FINDINGS.md`.

The diagrams (`diagram_*.png`) are stale and should be regenerated.

---

## Data Sources

All paths relative to repo root.

| File | What it contains |
|------|-----------------|
| [`roles.py`](../../../../../../roles.py) | Canonical circuit definition (lines 148-173) |
| [`FINDINGS.md`](../FINDINGS.md) | Full experimental narrative |
| [`data/head_census.json`](../data/head_census.json) | Weight-space features for all 144 heads (OV/QK norms, effective rank, GMM cluster, copy scores) |
| [`data/behavioral_census.json`](../data/behavioral_census.json) | Behavioral features on 500 sequences (prefix matching, copying score, RTI attention fractions) |
| [`data/causal_tests.json`](../data/causal_tests.json) | Faithfulness, IIA, path patching, bootstrap CIs, minimality (resample LOO for all 15 heads) |
| [`data/weight_analysis_rti.json`](../data/weight_analysis_rti.json) | Eigenvalue test, composition score summaries, inhibitory deltas, prefix-copying OV/attention |
| [`data/l0_deep_dive.json`](../data/l0_deep_dive.json) | L0 head SVD, logit lens, K-composition scores from backbone to all downstream |
| [`data/layer_deep_dives.json`](../data/layer_deep_dives.json) | Per-head attention pattern statistics, OV SVD, logit lens for all circuit heads |
| [`data/head_logit_lens.json`](../data/head_logit_lens.json) | W_OV @ W_U top promoted/suppressed tokens |
| [`data/circuit_synthesis.json`](../data/circuit_synthesis.json) | Bootstrap classification, feature importances |
| [`data/wandb_method_comparison.json`](../data/wandb_method_comparison.json) | DAS, activation patching, ACDC, SAE comparison results |
| [`data/acdc_circuit_rti.json`](../data/acdc_circuit_rti.json) | ACDC's 30-head circuit (Jaccard = 0.071 with weight circuit) |
| [`experiments/edge-validation/data/edge_validation.json`](experiments/edge-validation/data/edge_validation.json) | Level 1 node effects + Level 2 edge-level QKV path patching for all 70 edges |

---

## Circuit-Level Causal Validation

Before the per-head breakdown. Source: `data/causal_tests.json`.

| Metric | Value | Notes |
|--------|-------|-------|
| Full model mean LD | +1.526 | Baseline logit diff (correct - incorrect) |
| Circuit-only mean LD | -0.480 | 15-head circuit reverses behavior (RTI = suppression circuit) |
| Circuit-ablated mean LD | +1.015 | Ablating the circuit reduces but doesn't eliminate LD |
| Faithfulness ratio (median) | -0.535 | Negative = circuit opposes the model's output direction |
| Completeness drop | -0.511 | How much performance drops when circuit is ablated |
| Bootstrap ablation effect | -0.511, CI [-0.655, -0.370] | 95% bootstrap CI |
| Path patching: full circuit | 94.7% recovery (std 0.225) | Near-complete |
| Path patching: L0->L4H11 | 53.4% recovery | Backbone-to-detector path = half the signal |
| Path patching: L4H11->downstream | 94.8% recovery | Detector-to-downstream path = nearly all |
| Path patching: cohort->downstream | 105.9% recovery | Overshoots (copiers carry more than needed) |
| IIA: upstream (backbone) | 0.0% (0/52 flips) | Swapping backbone activations doesn't flip behavior |
| IIA: cohort (copiers) | 3.8% (2/52), LD shift -0.614 | Small signal through copiers |
| IIA: detector L4H11 | 1.9% (1/52), LD shift -0.424 | Detector alone has some causal power |
| IIA: downstream (readout) | 0.0%, LD shift +0.087 | Readout is not independently causal |
| IIA: full circuit | 1.9% (1/52), LD shift -0.524 | Full circuit swapping |

**Composition score summaries** (source: `data/weight_analysis_rti.json`):

| Tier Pair | Q-comp mean | Q-comp std | K-comp mean | K-comp std |
|-----------|-------------|------------|-------------|------------|
| backbone->copier | 4.68 | 1.66 | 5.88 | 2.11 |
| backbone->readout | 4.42 | 1.58 | 6.79 | 2.27 |
| copier->readout | 15.79 | 4.60 | 10.33 | 2.68 |
| random baseline | 7.96 | 5.69 | 7.57 | 4.77 |

Copier->readout Q-composition is 2x the random baseline. Backbone->copier K-composition is below random mean but backbone->L4H11 K-composition is 10-20x random (see per-head section).

---

## Tier 1: Backbone

Three heads in Layer 0. They don't do anything "interesting" by themselves — they write fixed low-rank directions into the residual stream that downstream heads read via K-composition. They attend broadly to context and BOS. None of them are induction heads (zero prefix matching). None copy tokens through their OV matrix.

Their importance is entirely relational: without the directions they write, L4H11's giant QK matrix has nothing to read.

### L0H8

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Backbone | `roles.py` |
| **What it does** | Writes fixed directions via low-rank OV. Context reader (64.5% to earlier tokens, 25.8% to last, 7.3% to BOS). | behavioral_census, layer_deep_dives |
| **OV effective rank** | 37.4 | head_census |
| **OV Frobenius norm** | 11.21 | head_census |
| **OV top singular value** | 6.446 | l0_deep_dive |
| **OV SV ratio (top/sum)** | 0.094 | l0_deep_dive |
| **QK effective rank** | 57.68 | head_census |
| **QK Frobenius norm** | 9.69 | head_census |
| **QK same-token score** | +0.700 | head_census |
| **OV neg score** | 0.543 | head_census |
| **OV diag/off ratio** | 0.603 | head_census |
| **Prefix matching** | 0.016 (rank ~140/144) | behavioral_census |
| **Behavioral copying** | 0.008 (near zero) | behavioral_census |
| **RTI attn to repeated** | 0.037 | behavioral_census |
| **Attention entropy** | 1.893 (std 0.243) | layer_deep_dives |
| **Attn to BOS** | 0.071 (std 0.108) | layer_deep_dives |
| **Attn to repeated** | 0.023 (std 0.051) | layer_deep_dives |
| **Attn to correct** | 0.020 (std 0.038) | layer_deep_dives |
| **Attn to other context** | 0.628 (std 0.144) | layer_deep_dives |
| **Attn to last** | 0.259 (std 0.107) | layer_deep_dives |
| **GMM cluster** | 0 | head_census |
| **OV copy_frac** | 0.000 | weight_analysis_rti |
| **Prefix attn (d1/d2/c)** | 0.006 / 0.007 / 0.007 | weight_analysis_rti |
| **Eigenvalue: pos_frac** | 0.479 | weight_analysis_rti |
| **Eigenvalue: top** | +6.883 | weight_analysis_rti |
| **Eigenvalue: bottom** | -6.054 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 2.82 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.368 | causal_tests |
| **Edge validation L1** | drop=+0.0036 (2.6% of baseline) — all 12 outgoing edges ACTIVE | edge_validation.json |
| **Inhibitory delta** | -0.161 (adding hurts circuit LD) | weight_analysis_rti |
| **Logit lens top promoted** | "dayName" (1.59), "iHUD" (1.49), "ゼウス" (1.46) — junk | l0_deep_dive |
| **Logit lens top suppressed** | "milo" (-2.04), "ramid" (-2.00) — junk | l0_deep_dive |
| **Top SV direction promoted** | " the" (1.24), "," (1.17), " a" (1.16) — function words | layer_deep_dives |

**K-composition (outgoing to circuit heads):**

| Target | K-comp | Q-comp |
|--------|--------|--------|
| L4H11 (detector) | 76.0 | 27.5 |
| L9H3 (copier) | 33.0 | 23.3 |
| L4H0 (copier) | 31.1 | 16.9 |
| L8H7 (copier) | 30.8 | 21.3 |
| L9H10 (copier) | 30.6 | 22.8 |
| L11H9 (readout) | 30.3 | 15.6 |
| L7H0 (copier) | 27.5 | 20.2 |
| L11H11 (readout) | 26.8 | 21.1 |
| L8H4 (copier) | 26.8 | 19.7 |
| L5H7 (copier) | 21.3 | 15.9 |
| L5H6 (copier) | 20.4 | 15.9 |
| L10H11 (readout) | 19.6 | 14.1 |

**Interpretation**: Lowest BOS attention among backbone (7%). Highest same-token QK score (+0.70), meaning its Q/K circuit specifically looks for positions with similar tokens. Its OV output promotes function words (" the", ",", " a") via its top singular vector — it may be writing a "generic context" direction. Moderate LOO effect (-0.368) and moderate inhibitory delta (-0.161).

---

### L0H9

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Backbone | `roles.py` |
| **What it does** | Writes fixed directions via low-rank OV. BOS + context reader (29.8% BOS, 55.9% context, 8.5% last). | behavioral_census, layer_deep_dives |
| **OV effective rank** | 37.08 | head_census |
| **OV Frobenius norm** | 13.67 | head_census |
| **OV top singular value** | 6.838 | l0_deep_dive |
| **OV SV ratio** | 0.082 | l0_deep_dive |
| **QK effective rank** | 58.75 | head_census |
| **QK Frobenius norm** | 4.19 (smallest in circuit) | head_census |
| **QK same-token score** | **-2.524** (most negative in circuit) | head_census |
| **OV neg score** | -0.451 | head_census |
| **OV diag/off ratio** | -0.730 | head_census |
| **Prefix matching** | 0.017 | behavioral_census |
| **Behavioral copying** | **-0.320** (negative — writes composition channels, not tokens) | behavioral_census |
| **RTI attn to repeated** | 0.072 | behavioral_census |
| **Attention entropy** | 2.278 (std 0.327) | layer_deep_dives |
| **Attn to BOS** | 0.298 (std 0.089) | layer_deep_dives |
| **Attn to repeated** | 0.054 (std 0.033) | layer_deep_dives |
| **Attn to correct** | 0.049 (std 0.030) | layer_deep_dives |
| **Attn to other context** | 0.515 (std 0.124) | layer_deep_dives |
| **Attn to last** | 0.085 (std 0.022) | layer_deep_dives |
| **GMM cluster** | 0 | head_census |
| **OV copy_frac** | 0.000 | weight_analysis_rti |
| **Prefix attn (d1/d2/c)** | 0.032 / 0.030 / 0.028 | weight_analysis_rti |
| **Eigenvalue: top / bottom** | +6.479 / -7.455 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 0.39 (lowest in circuit) | weight_analysis_rti |
| **Minimality (resample LOO)** | **-0.439** (largest LOO effect among backbone) | causal_tests |
| **Edge validation L1** | drop=-0.0000 (-0.0% of baseline) — **all 12 outgoing edges INACTIVE** | edge_validation.json |
| **Inhibitory delta** | not in top inhibitory list — near zero | weight_analysis_rti |
| **Logit lens top promoted** | " Devi" (0.73), " Kart" (0.72) — junk | l0_deep_dive |
| **Top SV direction promoted** | "ogether" (0.95), "Without" (0.85) | layer_deep_dives |

**K-composition (outgoing):**

| Target | K-comp | Q-comp |
|--------|--------|--------|
| L4H11 (detector) | **111.2** | 37.0 |
| L9H3 (copier) | 47.7 | 33.2 |
| L9H10 (copier) | 44.7 | 32.4 |
| L4H0 (copier) | 44.3 | 22.6 |
| L8H7 (copier) | 44.2 | 30.6 |
| L7H0 (copier) | 39.0 | 28.9 |
| L8H4 (copier) | 38.3 | 28.0 |
| L11H11 (readout) | 36.7 | 29.6 |
| L11H9 (readout) | 32.3 | 19.2 |
| L5H7 (copier) | 28.8 | 21.0 |
| L5H6 (copier) | 26.4 | 21.6 |
| L10H11 (readout) | 24.4 | 16.2 |

**Interpretation**: The most causally important backbone head (LOO = -0.439). Has the most negative QK same-token score in the entire circuit (-2.524), meaning it actively anti-correlates with same-token matching — it's doing positional or structural attention, not content attention. Smallest QK norm (4.19) means its attention pattern is nearly uniform, yet it writes the most distinctive OV directions. Copying score 0.39 is the lowest in the circuit — its OV matrix neither copies nor suppresses, it just writes a fixed subspace. **Carries a unique non-redundant causal signal** that no other backbone head replicates (narrower fan-out than L0H11 but largest LOO, the "minimality-composition inversion" from FINDINGS.md).

---

### L0H11

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Backbone | `roles.py` |
| **What it does** | Infrastructure hub. Writes fixed directions via the LOWEST OV effective rank in the entire model (33.5). BOS + context reader (30.9% BOS, 55.2% context). | layer_deep_dives |
| **OV effective rank** | **33.51** (lowest in entire model) | head_census |
| **OV Frobenius norm** | 11.04 | head_census |
| **OV top singular value** | 4.873 | l0_deep_dive |
| **OV SV ratio** | 0.076 | l0_deep_dive |
| **QK effective rank** | 42.07 | head_census |
| **QK Frobenius norm** | 4.97 | head_census |
| **QK same-token score** | -0.702 | head_census |
| **OV neg score** | 0.932 | head_census |
| **OV diag/off ratio** | 1.002 | head_census |
| **Prefix matching** | 0.022 | behavioral_census |
| **Behavioral copying** | **-0.854** (most negative in backbone — strongest composition channel writer) | behavioral_census |
| **RTI attn to repeated** | 0.078 | behavioral_census |
| **Attention entropy** | 2.260 (std 0.279) | layer_deep_dives |
| **Attn to BOS** | 0.309 (std 0.066) | layer_deep_dives |
| **Attn to repeated** | 0.059 (std 0.050) | layer_deep_dives |
| **Attn to other context** | 0.501 (std 0.127) | layer_deep_dives |
| **Attn to last** | 0.076 (std 0.018) | layer_deep_dives |
| **GMM cluster** | 0 | head_census |
| **OV copy_frac** | 0.000 | weight_analysis_rti |
| **Prefix attn (d1/d2/c)** | 0.034 / 0.034 / 0.037 | weight_analysis_rti |
| **Eigenvalue: top / bottom** | +5.856 / -5.018 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 1.85 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.402 | causal_tests |
| **Edge validation L1** | drop=-0.0014 (-1.0% of baseline) — all 12 outgoing edges ACTIVE (negative drops) | edge_validation.json |
| **Inhibitory delta** | **+0.025** (adding it slightly HELPS — redundant) | weight_analysis_rti |
| **Logit lens top promoted** | "ridge" (1.14), "crow" (1.11) — junk | l0_deep_dive |
| **Top SV direction promoted** | "iverse" (0.71), "bare" (0.70) | layer_deep_dives |

**K-composition (outgoing, highest fan-out in entire circuit):**

| Target | K-comp | Q-comp |
|--------|--------|--------|
| L4H11 (detector) | **130.4** (highest in entire circuit) | 40.8 |
| L9H3 (copier) | 55.5 | 38.2 |
| L4H0 (copier) | 52.2 | 25.4 |
| L9H10 (copier) | 51.3 | 37.1 |
| L8H7 (copier) | 51.2 | 34.9 |
| L7H0 (copier) | 45.0 | 33.1 |
| L8H4 (copier) | 44.0 | 31.7 |
| L11H11 (readout) | 39.9 | 31.4 |
| L5H7 (copier) | 33.1 | 23.3 |
| L11H9 (readout) | 31.6 | 18.5 |
| L5H6 (copier) | 29.7 | 24.2 |
| L10H11 (readout) | 24.8 | 15.7 |

**Control comparison**: L0H3 (non-circuit) -> L4H11 has K-comp = 50.6. All three backbone heads exceed this (76, 111, 130), confirming they're specifically wired into L4H11 beyond what random L0 heads achieve.

**Interpretation**: Highest fan-out in the circuit — feeds every downstream target with K-comp 25-130. But paradoxically has the **smallest individual LOO effect** among backbone (-0.402 vs -0.439 for L0H9). This is the IOI "Backup Name Mover" pattern: its signal is replicated across many redundant downstream paths, so removing it barely matters — some other path carries the information. Lowest OV rank in the entire model (33.5) means it concentrates its output into very few directions, making it an efficient "broadcast antenna." Adding it to the circuit slightly helps (+0.025 inhibitory delta), confirming it's constructive but redundant.

---

### Backbone Summary

The three backbone heads share a common profile:
- Layer 0, GMM cluster 0
- Low OV effective rank (33-37 vs model median ~60)
- Zero prefix matching, zero OV copying
- Attend broadly to context + BOS (not to specific tokens)
- Their OV top-promoted tokens are non-interpretable junk — they're not writing token-level information, they're writing **subspace directions** that downstream heads key into

Their K-composition to L4H11 (detector) is massive: 76, 111, 130. The next-highest non-circuit head (L0H3) only hits 50.6. This 2-3x gap is the weight-space signature of the backbone-detector information channel.

Together, zero-ablating all three produces LD shift = -1.009 (80% of the full circuit's -1.267 effect), p=0.032.

---

## Tier 2: Detector

### L4H11

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Detector | `roles.py` |
| **What it does** | The "have I seen this token before?" detector. Reads backbone directions via massive QK matrix to attend sharply to repeated token positions. NOT an induction head. | behavioral_census, FINDINGS.md |
| **OV effective rank** | 60.49 | head_census |
| **OV Frobenius norm** | 7.25 | head_census |
| **OV top singular value** | 1.655 | layer_deep_dives |
| **OV SV ratio** | 0.029 | layer_deep_dives |
| **QK effective rank** | 45.02 | head_census |
| **QK Frobenius norm** | **58.16** (3-4x larger than any other head in the model) | head_census |
| **QK top singular value** | **18.46** | layer_deep_dives |
| **QK same-token score** | -0.239 | head_census |
| **OV neg score** | 2.655 | head_census |
| **OV diag/off ratio** | 3.478 | head_census |
| **Prefix matching** | **7.37e-23** (rank 144/144 — dead last, NOT an induction head) | behavioral_census |
| **Behavioral copying** | 0.156 | behavioral_census |
| **RTI attn to repeated** | **0.303** (rank 4/144 model-wide) | behavioral_census |
| **Attention entropy** | **0.018** (std 0.080) — extremely sharp/peaked | layer_deep_dives |
| **Attn to BOS** | **5.85e-08** (essentially ZERO) | layer_deep_dives |
| **Attn to repeated** | 0.220 (std 0.409) | layer_deep_dives |
| **Attn to correct** | 0.030 (std 0.159) | layer_deep_dives |
| **Attn to other context** | 0.745 (std 0.443) | layer_deep_dives |
| **Attn to last** | 0.005 (std 0.033) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | 0.093 | weight_analysis_rti |
| **Prefix attn (d1/d2/c)** | 0.000 / 0.000 / 0.000 | weight_analysis_rti |
| **Eigenvalue: top / bottom** | +2.074 / -1.499 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 26.53 | weight_analysis_rti |
| **Minimality (resample LOO)** | **-0.562** | causal_tests |
| **Inhibitory delta** | not in top inhibitory list | weight_analysis_rti |
| **Edge validation L1** | drop=+0.2056 (**149.1%** of baseline) — all 10 outgoing edges ACTIVE (~0.20 drop each) | edge_validation.json |
| **Activation patching rank** | **3/144** (effect = 0.320) | wandb_method_comparison |
| **Logit lens top promoted** | "ccording" (0.37), " corrid" (0.36) — low magnitude, non-interpretable | layer_deep_dives |
| **Top SV direction promoted** | "anwhile" (0.93), "crow" (0.67) | layer_deep_dives |

**K-composition (incoming from backbone):**

| Source | K-comp | Q-comp | In circuit? |
|--------|--------|--------|-------------|
| L0H11 (backbone) | **130.4** | 40.8 | Yes |
| L0H9 (backbone) | **111.2** | 37.0 | Yes |
| L0H8 (backbone) | **76.0** | 27.5 | Yes |
| L0H3 (control) | 50.6 | 21.2 | No |

**K-composition (outgoing to copiers — LOW, operates through residual stream):**

| Target | K-comp | Q-comp |
|--------|--------|--------|
| L5H6 | 9.8 | 13.4 |
| L11H9 | 9.0 | 5.2 |
| L10H11 | 6.2 | 5.8 |
| L11H11 | 5.1 | 6.6 |
| L7H0 | 4.6 | 8.1 |
| L5H7 | 4.4 | 6.6 |
| L8H7 | 4.1 | 7.0 |
| L8H4 | 4.0 | 5.5 |
| L9H10 | 3.8 | 5.5 |
| L9H3 | 3.6 | 6.6 |

**Path patching:**

| Path | Recovery |
|------|----------|
| L0 -> L4H11 | 53.4% |
| L4H11 -> downstream | 94.8% |

**Interpretation**: This is the most mechanistically interesting head in the circuit. Its QK Frobenius norm (58.16) is 3-4x any other head in the entire model — it has an absurdly amplified attention circuit that reads the specific subspace directions L0 heads write. This lets it attend sharply to repeated tokens (attention entropy 0.018, nearly a delta function). But it is NOT an induction head — its prefix matching is literally 7e-23, dead last in the model. It detects repetition through a completely different mechanism than classical induction: via backbone-written subspace directions, not [A][B]...[A] -> predict [B] pattern matching.

Its outgoing K-composition to copiers is LOW (3-10), suggesting it communicates downstream through the residual stream directly, not via key-query composition. Its OV matrix has low norm (7.25) and tiny top SV (1.655), so it writes a small but targeted signal.

L4H11 is in the **old circuit as "backbone"** and in the **new circuit as "detector"**. Both agree it's critical — it's the bridge between early-layer position encoding and mid-layer repeat processing.

---

## Tier 3: Copiers

Eight heads across layers 4-9. They amplify the non-repeated signal and copy token identities downstream. Most attend heavily to BOS (37-82%). They have high OV copying scores (eigenvalue test copying_score 9-171) and positive OV diagonal scores. The backbone heads feed them via K-composition, but L4H11's outgoing K-comp to them is small — they read the backbone directly.

### L4H0

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Copier/amplifier. Heavy BOS attention (37.6%). Near-zero RTI attention. | behavioral_census |
| **OV effective rank** | 60.90 | head_census |
| **OV Frobenius norm** | 10.73 | head_census |
| **QK Frobenius norm** | 18.35 | head_census |
| **QK same-token score** | -0.477 | head_census |
| **OV diag/off ratio** | **47.70** (anomalously high) | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | 0.128 | behavioral_census |
| **RTI attn to repeated** | 0.029 | behavioral_census |
| **Attn to BOS** | 0.376 (std 0.223) | layer_deep_dives |
| **Attn entropy** | 1.315 (std 0.325) | layer_deep_dives |
| **GMM cluster** | 0 | head_census |
| **OV copy_frac** | 0.003 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 9.40 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.400 | causal_tests |
| **Inhibitory delta** | -0.037 | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 52.2, L0H9: 44.3, L0H8: 31.1 | l0_deep_dive |

---

### L5H6

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Copier. Heaviest BOS attention among early copiers (69.4%). Moderate RTI attn (0.123). | behavioral_census |
| **OV effective rank** | 60.50 | head_census |
| **OV Frobenius norm** | 12.82 | head_census |
| **QK Frobenius norm** | 17.30 | head_census |
| **OV neg score** | 1.223 | head_census |
| **OV diag/off ratio** | 2.203 | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | 0.376 | behavioral_census |
| **RTI attn to repeated** | **0.123** (second highest in copier tier) | behavioral_census |
| **Attn to BOS** | 0.694 (std 0.224) | layer_deep_dives |
| **Attn entropy** | 0.793 (std 0.402) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | 0.000 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 10.59 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.402 | causal_tests |
| **Edge validation L1** | drop=+0.0325 (**23.6%** of baseline — **second strongest node**) | edge_validation.json |
| **Inhibitory delta** | **+0.045** (adding helps — constructive) | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 29.7, L0H9: 26.4, L0H8: 20.4, L4H11: 9.8 | l0_deep_dive |

---

### L5H7

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Copier. BOS-heavy (56.7%). Lowest RTI attention in copier tier (0.014). Synergistic pair member with L9H10. | behavioral_census, FINDINGS.md |
| **OV effective rank** | 62.33 | head_census |
| **OV Frobenius norm** | 17.78 | head_census |
| **QK Frobenius norm** | 16.85 | head_census |
| **OV neg score** | 2.606 | head_census |
| **OV diag/off ratio** | 3.886 | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | 0.128 | behavioral_census |
| **RTI attn to repeated** | **0.014** (lowest in copier tier) | behavioral_census |
| **Attn to BOS** | 0.567 (std 0.315) | layer_deep_dives |
| **Attn entropy** | 1.005 (std 0.587) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | 0.045 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 54.84 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.444 | causal_tests |
| **Inhibitory delta** | -0.051 | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 33.1, L0H9: 28.8, L0H8: 21.3, L4H11: 4.4 | l0_deep_dive |

**Note**: L5H7 + L9H10 form a synergistic pair — their joint 7-head delta is the most negative (-0.198) of any pair tested (FINDINGS.md Part 3).

---

### L7H0

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Copier with moderate OV copying. BOS 56.7%, moderate RTI attn (0.076). | behavioral_census |
| **OV effective rank** | 59.33 | head_census |
| **OV Frobenius norm** | 16.69 | head_census |
| **QK Frobenius norm** | 12.95 | head_census |
| **OV neg score** | 2.098 | head_census |
| **Copy score** | 0.255 | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | 0.285 | behavioral_census |
| **RTI attn to repeated** | 0.076 | behavioral_census |
| **Attn to BOS** | 0.567 (std 0.231) | layer_deep_dives |
| **Attn entropy** | 1.166 (std 0.390) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | **0.312** | weight_analysis_rti |
| **Eigenvalue: copying_score** | 73.16 | weight_analysis_rti |
| **Eigenvalue: top** | +6.13 | weight_analysis_rti |
| **Minimality (resample LOO)** | **-0.570** | causal_tests |
| **Inhibitory delta** | +0.004 (neutral) | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 45.0, L0H9: 39.0, L0H8: 27.5, L4H11: 4.6 | l0_deep_dive |

**Note**: This head is in the **old circuit diagrams** as "Repeat-finders: L7H9, L8H6, L8H10" — but L7H0 is NOT in the old circuit (L7H9 was). L7H0 is only in the new circuit. The old diagrams show a different set of copier-tier heads.

---

### L8H4

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Copier. Heaviest BOS in copier tier (75.1%). Very low RTI attn (0.014). Highest OV neg_score in entire circuit (2.995). | behavioral_census, head_census |
| **OV effective rank** | 62.19 | head_census |
| **OV Frobenius norm** | 25.52 | head_census |
| **QK Frobenius norm** | 12.72 | head_census |
| **OV neg score** | **2.995** (highest in circuit) | head_census |
| **OV diag/off ratio** | 3.891 | head_census |
| **Copy score** | 0.165 | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | 0.159 | behavioral_census |
| **RTI attn to repeated** | 0.014 | behavioral_census |
| **Attn to BOS** | **0.751** (std 0.169) | layer_deep_dives |
| **Attn entropy** | 0.892 (std 0.424) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | 0.226 | weight_analysis_rti |
| **Eigenvalue: copying_score** | 113.25 | weight_analysis_rti |
| **Eigenvalue: top** | +7.28 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.586 | causal_tests |
| **Inhibitory delta** | -0.019 | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 44.0, L0H9: 38.3, L0H8: 26.8, L4H11: 4.0 | l0_deep_dive |

---

### L8H7

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Copier. BOS 59.2%. Highest OV copy_frac among copiers (0.460). | behavioral_census, weight_analysis_rti |
| **OV effective rank** | 60.41 | head_census |
| **OV Frobenius norm** | 20.38 | head_census |
| **QK Frobenius norm** | 12.60 | head_census |
| **OV neg score** | 2.479 | head_census |
| **Copy score** | 0.215 | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | 0.404 | behavioral_census |
| **RTI attn to repeated** | 0.030 | behavioral_census |
| **Attn to BOS** | 0.592 (std 0.228) | layer_deep_dives |
| **Attn entropy** | 1.096 (std 0.329) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | **0.460** (highest among copiers) | weight_analysis_rti |
| **Eigenvalue: copying_score** | 104.71 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.515 | causal_tests |
| **Inhibitory delta** | -0.007 (near zero) | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 51.2, L0H9: 44.2, L0H8: 30.8, L4H11: 4.1 | l0_deep_dive |

---

### L9H3

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Strongest copier by behavioral score (0.856) and OV copy_frac (0.562). Moderate BOS (36.5% — lowest among copiers). Promotes plural nouns through logit lens. | behavioral_census, layer_deep_dives |
| **OV effective rank** | 59.72 | head_census |
| **OV Frobenius norm** | 22.91 | head_census |
| **QK Frobenius norm** | 12.55 | head_census |
| **OV neg score** | 2.019 | head_census |
| **Copy score** | 0.335 | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | **0.856** (highest in copier tier) | behavioral_census |
| **RTI attn to repeated** | 0.062 | behavioral_census |
| **Attn to BOS** | 0.365 (std 0.175) — lowest BOS among copiers | layer_deep_dives |
| **Attn entropy** | 1.743 (std 0.304) — highest entropy among copiers | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | **0.562** (highest among copiers by OV test) | weight_analysis_rti |
| **Eigenvalue: copying_score** | 119.86 | weight_analysis_rti |
| **Eigenvalue: top** | +8.95 | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.404 | causal_tests |
| **Edge validation L1** | drop=+0.0111 (**8.0%** of baseline — **third strongest node**) | edge_validation.json |
| **Inhibitory delta** | -0.008 (near zero) | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 55.5, L0H9: 47.7, L0H8: 33.0, L4H11: 3.6 | l0_deep_dive |
| **Logit lens top promoted** | " Languages" (1.55), " Diseases" (1.52) — plural nouns | layer_deep_dives |

---

### L9H10

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Copier | `roles.py` |
| **What it does** | Copier with heaviest BOS in entire circuit (81.8%). Very low RTI attn (0.010). Highest OV Frobenius norm in copier tier (31.12). Synergistic pair with L5H7. | behavioral_census, head_census |
| **OV effective rank** | 61.94 | head_census |
| **OV Frobenius norm** | **31.12** (highest in copier tier) | head_census |
| **QK Frobenius norm** | 12.44 | head_census |
| **OV neg score** | 2.879 | head_census |
| **Copy score** | 0.275 | head_census |
| **Prefix matching** | ~0.000 | behavioral_census |
| **Behavioral copying** | -0.152 (negative) | behavioral_census |
| **RTI attn to repeated** | **0.010** (lowest in copier tier) | behavioral_census |
| **Attn to BOS** | **0.818** (std 0.116) — highest in entire circuit | layer_deep_dives |
| **Attn entropy** | 0.750 (std 0.334) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | 0.301 | weight_analysis_rti |
| **Eigenvalue: copying_score** | **171.43** (highest among copiers) | weight_analysis_rti |
| **Eigenvalue: top** | +9.43 | weight_analysis_rti |
| **Minimality (resample LOO)** | **-0.693** (MOST IMPORTANT HEAD IN ENTIRE CIRCUIT) | causal_tests |
| **Edge validation L1** | drop=+0.0019 (1.3% of baseline) — all 3 outgoing edges to readout ACTIVE | edge_validation.json |
| **Inhibitory delta** | -0.004 (near zero) | weight_analysis_rti |
| **Incoming K-comp** | L0H11: 51.3, L0H9: 44.7, L0H8: 30.6, L4H11: 3.8 | l0_deep_dive |

**Interpretation**: Despite attending almost exclusively to BOS (81.8%) and barely attending to repeated tokens (1.0%), this head has the **largest resample LOO effect in the entire circuit** (-0.693). How? Its OV matrix has the highest eigenvalue copying score (171.43) and highest OV norm (31.12) among copiers — it amplifies whatever direction it reads from BOS with enormous gain. BOS accumulates a running average of the residual stream, so L9H10 reads the "global context summary" from BOS and amplifies it through a high-gain OV channel. Removing it collapses the circuit more than removing any other single head.

---

### Copier Summary

| Head | BOS attn | RTI attn | Copy score (eig) | OV copy_frac | Minimality LOO |
|------|----------|----------|-------------------|-------------|----------------|
| L4H0 | 37.6% | 2.9% | 9.4 | 0.003 | -0.400 |
| L5H6 | 69.4% | 12.3% | 10.6 | 0.000 | -0.402 |
| L5H7 | 56.7% | 1.4% | 54.8 | 0.045 | -0.444 |
| L7H0 | 56.7% | 7.6% | 73.2 | 0.312 | -0.570 |
| L8H4 | 75.1% | 1.4% | 113.3 | 0.226 | -0.586 |
| L8H7 | 59.2% | 3.0% | 104.7 | 0.460 | -0.515 |
| L9H3 | 36.5% | 6.2% | 119.9 | 0.562 | -0.404 |
| L9H10 | 81.8% | 1.0% | 171.4 | 0.301 | **-0.693** |

**Pattern**: Copying score increases with layer depth (9 -> 171). BOS attention also generally increases. The copiers that attend most to BOS (L8H4, L9H10) are NOT the ones with the highest behavioral copying — they amplify via subspace rather than direct token copying.

---

## Tier 4: Readout

Three heads in layers 10-11. Two are classic induction heads (L10H11, L11H9) and one is an anomalous specialized channel (L11H11).

### L10H11

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Readout | `roles.py` |
| **What it does** | Classic induction head. Prefix matching (0.279), token copying (8.2). Translates upstream signal into predictions. | behavioral_census |
| **OV effective rank** | 61.53 | head_census |
| **OV Frobenius norm** | 30.83 | head_census |
| **OV top singular value** | 7.553 | layer_deep_dives |
| **QK Frobenius norm** | 15.36 | head_census |
| **QK same-token score** | 0.673 | head_census |
| **Copy score** | 0.135 | head_census |
| **Prefix matching** | **0.279** | behavioral_census |
| **Behavioral copying** | **8.19** | behavioral_census |
| **RTI attn to repeated** | 0.077 | behavioral_census |
| **Attn to BOS** | 0.562 (std 0.265) | layer_deep_dives |
| **Attn to correct** | **0.164** (std 0.150) — attends to correct next token | layer_deep_dives |
| **Attn entropy** | 1.219 (std 0.439) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | 0.349 | weight_analysis_rti |
| **Prefix attn (d1/d2/c)** | 0.039 / 0.012 / **0.047** | weight_analysis_rti |
| **Eigenvalue: copying_score** | 144.36 | weight_analysis_rti |
| **Eigenvalue: top** | +9.20 | weight_analysis_rti |
| **Minimality (resample LOO)** | **+0.022** (fully redundant) | causal_tests |
| **Inhibitory delta** | -0.009 | weight_analysis_rti |

**K-composition (incoming from copiers):**

| Source | K-comp | Q-comp |
|--------|--------|--------|
| L8H7 (copier) | 12.5 | — |
| L9H3 (copier) | 12.4 | — |
| L7H0 (copier) | 10.5 | — |
| L9H10 (copier) | 9.9 | — |
| L0H11 (backbone, direct) | 24.8 | — |
| L0H9 (backbone, direct) | 24.4 | — |

**Interpretation**: A real induction head — does [A][B]...[A] -> predict [B] pattern matching (prefix matching 0.279, top ~3% of all heads). Its minimality LOO is **+0.022**, meaning removing it has essentially zero effect — its signal is fully backed up by L11H9 and L11H11. Added late in Part 4 analysis (not in the original Part 3 cohort).

---

### L11H9

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Readout | `roles.py` |
| **What it does** | Strongest induction head in the circuit. Highest prefix matching (0.325) and highest behavioral copying (12.69). Promotes abstract nouns through logit lens. | behavioral_census, layer_deep_dives |
| **OV effective rank** | 61.96 | head_census |
| **OV Frobenius norm** | **90.40** (anomalously high) | head_census |
| **OV top singular value** | **18.63** | layer_deep_dives |
| **QK Frobenius norm** | 18.24 | head_census |
| **QK same-token score** | -0.945 | head_census |
| **Copy score** | 0.005 | head_census |
| **Prefix matching** | **0.325** (highest in circuit) | behavioral_census |
| **Behavioral copying** | **12.69** (highest in circuit) | behavioral_census |
| **RTI attn to repeated** | 0.057 | behavioral_census |
| **Attn to BOS** | 0.757 (std 0.130) | layer_deep_dives |
| **Attn to correct** | 0.098 (std 0.087) | layer_deep_dives |
| **Attn entropy** | 0.888 (std 0.327) | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | 0.012 | weight_analysis_rti |
| **Prefix attn (d1/d2/c)** | 0.040 / 0.009 / **0.061** | weight_analysis_rti |
| **Eigenvalue: top / bottom** | **+21.12 / -17.55** | weight_analysis_rti |
| **Eigenvalue: copying_score** | 163.67 | weight_analysis_rti |
| **Minimality (resample LOO)** | **-0.656** (second most important head in circuit) | causal_tests |
| **Inhibitory delta** | -0.029 | weight_analysis_rti |
| **Logit lens top promoted** | " aversion" (5.79), " perception" (5.56), " habit" (5.44) — abstract nouns | layer_deep_dives |
| **Top SV direction** | "esta" (0.91), "ase" (0.76), "ook" (0.75) | layer_deep_dives |

**Interpretation**: The circuit's primary output head. OV norm 90.4 is 3x any copier — it writes with huge magnitude. Its eigenvalues (+21.1 / -17.5) are the most extreme in the circuit after L11H11, meaning its OV matrix has strong both-direction structure. Interestingly its logit lens promotes abstract nouns (" aversion", " perception", " habit") — actual semantic content, unlike the backbone junk. Second most important head by LOO (-0.656), confirming it does critical work.

---

### L11H11

| Property | Value | Source |
|----------|-------|--------|
| **Role** | Readout | `roles.py` |
| **What it does** | Anomalous specialized readout. Low prefix matching (0.023) but extreme weight properties: lowest OV rank among readout (40.3), GIANT top singular value (62.3), highest OV Frobenius norm in entire circuit (95.0). Writes in one highly concentrated direction. | head_census, layer_deep_dives |
| **OV effective rank** | **40.28** (anomalously low — specialized) | head_census |
| **OV Frobenius norm** | **95.02** (highest in entire circuit) | head_census |
| **OV top singular value** | **62.31** (3.3x L11H9, 8.2x L10H11) | layer_deep_dives |
| **OV SV ratio** | **0.103** (highest in circuit — single direction dominates) | layer_deep_dives |
| **QK Frobenius norm** | 18.79 | head_census |
| **QK same-token score** | -0.880 | head_census |
| **Copy score** | **0.420** (highest in circuit) | head_census |
| **OV neg score** | 0.144 | head_census |
| **OV diag/off ratio** | 0.223 | head_census |
| **Prefix matching** | 0.023 (low for readout) | behavioral_census |
| **Behavioral copying** | 2.40 | behavioral_census |
| **RTI attn to repeated** | 0.037 | behavioral_census |
| **Attn to BOS** | 0.487 (std 0.201) | layer_deep_dives |
| **Attn to correct** | 0.053 (std 0.045) | layer_deep_dives |
| **Attn to other context** | 0.383 (std 0.172) | layer_deep_dives |
| **Attn entropy** | 1.556 (std 0.399) — highest among readout | layer_deep_dives |
| **GMM cluster** | 2 | head_census |
| **OV copy_frac** | **0.686** (highest in entire circuit) | weight_analysis_rti |
| **Prefix attn (d1/d2/c)** | 0.029 / 0.007 / 0.015 | weight_analysis_rti |
| **Eigenvalue: top / bottom** | **+60.01 / -64.69** (by far most extreme) | weight_analysis_rti |
| **Eigenvalue: copying_score** | **403.90** (2.5x next highest) | weight_analysis_rti |
| **Minimality (resample LOO)** | -0.272 (moderate) | causal_tests |
| **Inhibitory delta** | -0.032 | weight_analysis_rti |
| **Logit lens top promoted** | " eyebrow" (2.77), " roofs" (2.44) — body/structure parts | layer_deep_dives |
| **Top SV direction** | " eleph" (1.71), " metic" (1.32) | layer_deep_dives |

**K-composition (incoming):**

| Source | K-comp |
|--------|--------|
| L0H11 (backbone, direct) | 39.9 |
| L0H9 (backbone, direct) | 36.7 |
| L0H8 (backbone, direct) | 26.8 |
| L9H3 (copier) | 11.1 |
| L10H11 (readout) | 10.6 |
| L8H7 (copier) | 9.6 |
| L9H10 (copier) | 9.0 |

**Interpretation**: This head is weird. It's NOT a standard induction head — low prefix matching (0.023). But its OV matrix is the most extreme in the entire circuit by every measure: eigenvalue copying score 403.9 (2.5x next), top SV 62.3, OV norm 95.0, OV copy_frac 0.686. It concentrates its output into a single direction (SV ratio 0.103, OV effective rank 40.3) with enormous magnitude. It's a specialized amplifier that reads a specific direction from the residual stream and blasts one token identity into the logits. Its low prefix matching but high copy_frac means: it doesn't find the right position by induction, but once it attends to something, it copies that token's identity with 68.6% fidelity — the highest in the circuit.

---

### Readout Summary

| Head | Prefix matching | Behavioral copy | OV top SV | OV norm | OV copy_frac | Eigenvalue copy_score | Minimality LOO |
|------|-----------------|-----------------|-----------|---------|-------------|----------------------|----------------|
| L10H11 | 0.279 | 8.19 | 7.6 | 30.8 | 0.349 | 144.4 | +0.022 (redundant) |
| L11H9 | **0.325** | **12.69** | 18.6 | 90.4 | 0.012 | 163.7 | **-0.656** |
| L11H11 | 0.023 | 2.40 | **62.3** | **95.0** | **0.686** | **403.9** | -0.272 |

Two different strategies:
- L10H11 and L11H9 are **induction heads** (find position by prefix matching, copy token)
- L11H11 is a **specialized amplifier** (attend broadly, but copy with extreme gain in one direction)

---

## Overall Minimality Ranking

All 15 heads sorted by how much removing them hurts the circuit (resample LOO, source: `causal_tests.json`):

| Rank | Head | Tier | LOO Effect | Interpretation |
|------|------|------|-----------|----------------|
| 1 | **L9H10** | Copier | **-0.693** | Most critical — high-gain BOS amplifier |
| 2 | **L11H9** | Readout | **-0.656** | Primary induction output |
| 3 | L8H4 | Copier | -0.586 | BOS-heavy copier |
| 4 | L7H0 | Copier | -0.570 | Moderate copier with OV copy 0.312 |
| 5 | L4H11 | Detector | -0.562 | The repeat-detection relay |
| 6 | L8H7 | Copier | -0.515 | Highest OV copy_frac copier (0.460) |
| 7 | L5H7 | Copier | -0.444 | Low RTI attn, synergistic with L9H10 |
| 8 | L0H9 | Backbone | -0.439 | Unique non-redundant backbone signal |
| 9 | L9H3 | Copier | -0.404 | Strongest behavioral copier (0.856) |
| 10 | L5H6 | Copier | -0.402 | BOS-heavy early copier |
| 11 | L0H11 | Backbone | -0.402 | Highest fan-out but redundant |
| 12 | L4H0 | Copier | -0.400 | Anomalous OV diag ratio (47.7) |
| 13 | L0H8 | Backbone | -0.368 | Context reader, moderate effect |
| 14 | L11H11 | Readout | -0.272 | Specialized single-direction amplifier |
| 15 | L10H11 | Readout | **+0.022** | Fully redundant induction head |

---

## Edge-Level Causal Validation

Source: [`experiments/edge-validation/data/edge_validation.json`](experiments/edge-validation/data/edge_validation.json).

Level 1: Corrupt each sender's output with activation patching, measure logit diff drop. Level 2 (Wang et al. 2022 path patching): Corrupt sender, let only one receiver QKV channel recompute, freeze other two at clean values. An edge is "active" if max(|Q_drop|, |K_drop|, |V_drop|) > 1% of baseline LD. Baseline LD = 0.1379. n=200 IOI-style examples.

### Level 1: Node Effects

| Rank | Head | Role | Drop | % of baseline |
|------|------|------|------|---------------|
| 1 | **L4H11** | detector | +0.2056 | **149.1%** |
| 2 | L5H6 | copier | +0.0325 | 23.6% |
| 3 | L9H3 | copier | +0.0111 | 8.0% |
| 4 | L0H8 | backbone | +0.0036 | 2.6% |
| 5 | L9H10 | copier | +0.0019 | 1.3% |
| 6 | L8H7 | copier | +0.0018 | 1.3% |
| 7 | L8H4 | copier | +0.0008 | 0.6% |
| 8 | L7H0 | copier | -0.0003 | -0.2% |
| 9 | L0H9 | backbone | -0.0000 | -0.0% |
| 10 | L0H11 | backbone | -0.0014 | -1.0% |
| 11 | L4H0 | copier | -0.0013 | -0.9% |
| 12 | L5H7 | copier | -0.0039 | -2.8% |

L4H11 dominates — corrupting it alone overshoots the full baseline (149.1%). L5H6 is a distant second (23.6%). L0H9 has zero effect despite being the most causally important backbone head by LOO minimality (-0.439). This is the "minimality-composition inversion": L0H9's contribution is unique but small in magnitude; removing it hurts because nothing else replicates its signal, but corrupting it barely changes the output because the signal is tiny.

### Level 2: Pathway Summary

| Pathway | Active / Total | Rate |
|---------|----------------|------|
| detector → copier | **7 / 7** | **100%** |
| detector → readout | **3 / 3** | **100%** |
| backbone → copier | 16 / 24 | 67% |
| copier → readout | 16 / 24 | 67% |
| backbone → readout | 6 / 9 | 67% |
| backbone → detector | 2 / 3 | 67% |
| **Total** | **50 / 70** | **71%** |

All L4H11 outgoing edges are active (10/10, 100%). The 20 inactive edges come from: L0H9 (all 12 edges inactive), L4H0→L11H9/L11H11, L7H0→all readout, L8H4→all readout.

### Level 2: L4H11 Outgoing Edges

All 10 edges from L4H11 carry ~0.20 LD drop uniformly through every QKV channel:

| Edge | Q_drop | K_drop | V_drop |
|------|--------|--------|--------|
| L4H11→L5H6 | 0.2067 | 0.2015 | 0.2005 |
| L4H11→L5H7 | 0.2057 | 0.2056 | 0.2057 |
| L4H11→L7H0 | 0.2056 | 0.2056 | 0.2056 |
| L4H11→L8H4 | 0.2054 | 0.2056 | 0.2058 |
| L4H11→L8H7 | 0.2050 | 0.2057 | 0.2063 |
| L4H11→L9H3 | 0.1993 | 0.2001 | 0.2055 |
| L4H11→L9H10 | 0.2053 | 0.2057 | 0.2057 |
| L4H11→L10H11 | 0.2052 | 0.2045 | 0.2043 |
| L4H11→L11H9 | 0.2048 | 0.2051 | 0.2056 |
| L4H11→L11H11 | 0.2042 | 0.2054 | 0.2072 |

The nearly identical Q/K/V drops mean L4H11's signal propagates uniformly through all receiver input channels — it writes a direction that affects receivers regardless of whether they read it through Q, K, or V.

### Inactive Edges and the Minimality-Composition Paradox

All 12 L0H9 outgoing edges are inactive (max |drop| < 0.0003). This is despite L0H9 having the highest backbone K-composition to L4H11 (111.2) and the largest backbone LOO effect (-0.439). Weight-space composition score measures the **capacity** of an edge (‖W_O · W_K‖_F), while activation patching measures the **actual causal flow** through it. L0H9 writes directions that L4H11 *can* read, but under the IOI distribution the actual signal is negligible.

The copier→readout edges from L7H0 (all 3), L8H4 (all 3), and L4H0→L11H9/L11H11 are also inactive — these copiers affect the logit through the residual stream directly, not through direct head-to-head composition with readout.

### Edge Validation vs Weight-Space Composition

The edge validation reveals a key discrepancy between weight-space structure and causal flow:
- **Weight-space**: Backbone→L4H11 K-comp is the strongest edge class (76-130)
- **Causal flow**: L4H11→downstream is the strongest pathway (all 10 edges at ~0.20 drop)
- **Weight-space**: L0H9 has highest backbone K-comp to L4H11 (111.2)
- **Causal flow**: L0H9 has zero causal effect through any edge

This supports the interpretation that weight-space analysis identifies circuit **structure** (what the network is wired to do) while activation-level analysis identifies circuit **function** (what it actually does on a given distribution).

---

## What the Diagrams Get Wrong

The old diagrams (`diagram_residual_stream.png`, `diagram_circuit_overview.png`, `diagram_ioi_style_circuit.png`) were generated from an earlier analysis pass and show a different circuit. Key discrepancies:

| Old diagram | Current roles.py | Issue |
|-------------|-----------------|-------|
| Backbone: L0H9, L0H10, L4H11 | Backbone: L0H8, L0H9, L0H11 | L0H10 was replaced by L0H8; L4H11 moved to detector tier |
| "Repeat-finders": L7H9, L8H6, L8H10 | Not in current circuit | These heads were dropped; replaced by L7H0, L8H4, L8H7 |
| "Suppressors": L8H10, L9H1 | Not in current circuit | Suppressor concept folded into copier tier |
| "Answer-extractor": L9H6 | Not in current circuit | Replaced by L9H3, L9H10 |
| Readout: L9H9, L10H0, L10H2, L10H10 | Readout: L10H11, L11H9, L11H11 | Completely different readout heads |

The `weight_analysis_rti.json` `ov_copying` and `prefix_copying` `in_circuit` flags also use the old circuit definition.

**The old diagrams should be regenerated to match `roles.py`.**

---

## Logit Lens Phase Transition

Source: `wandb_method_comparison.json`, DAS localization results.

| Layer | DAS IIA (k=16) | Notes |
|-------|----------------|-------|
| L0-L4 | 0.0 | No causal variable present |
| L8 | 0.79 (k=64) | Partially formed |
| L9 | LD jumps to +5.712 | Phase transition |
| L10 | **1.0** (k=16) | Fully isolable |
| L11 | **1.0** (k=8) | Fully isolable |

L9 attribution = +34.6 (dominant layer). L11 attribution = -13.1 (active suppression / error correction).

---

## SAE Feature Localization

Source: `wandb_method_comparison.json`.

Top causal SAE features concentrate in L9:
- f19512: effect 0.711
- f1721: effect 0.701
- f3081: effect 0.691

Per-layer max effects follow the circuit topology: L0 (0.12) -> L4 (0.33) -> L7 (0.65) -> L9 (0.71).

---

## ACDC Comparison

Source: `data/acdc_circuit_rti.json`.

ACDC found 30 heads. Weight circuit found 15 heads. Jaccard overlap = **0.071** (3 shared heads out of 42 union).

Shared: L0H9, L0H11, L4H11.

ACDC missed: All 8 copiers and all 3 readout heads from the weight circuit.

Weight circuit missed: 27 of ACDC's heads, most in early layers (L0-L3) that the weight-space method correctly identified as non-circuit infrastructure.

---

## Cross-Method Discovery: Which Methods Find Which Heads

Source: `experiments/method-venn/data/method_venn.json`, `experiments/eap-ablation/data/eap_ablation.json`, `experiments/actpatch-rti/data/actpatch_rti.json`.

Six circuit discovery methods were compared on RTI, each selecting their top-15 heads (ACDC selects 30). This section documents per-head discovery across methods, organized by the weight circuit's tier structure, then covers notable non-circuit heads.

### Method Legend

| Method | Type | What it measures |
|--------|------|-----------------|
| **Weight** | Structure | OV/QK weight features, bootstrap classification |
| **EAP** | Attribution | Edge-level gradient × activation importance |
| **EAP-IG** | Attribution | Integrated gradient variant of EAP |
| **ActPatch** | Causal (single-node) | LD change when one head's output is corrupted |
| **ACDC** | Causal (greedy pruning) | Backwards head pruning until circuit faithfulness degrades |
| **Wang ABA** | Behavioral | Head output's contribution to repeated token logit |

### Inter-Method Similarity (Jaccard on top-15 sets)

|  | Weight | EAP | EAP-IG | ActPatch | ACDC | Wang |
|--|--------|-----|--------|----------|------|------|
| **Weight** | 1.00 | **0.00** | **0.00** | 0.03 | 0.07 | 0.11 |
| **EAP** | 0.00 | 1.00 | 0.58 | 0.20 | 0.02 | 0.25 |
| **EAP-IG** | 0.00 | 0.58 | 1.00 | 0.15 | 0.05 | 0.43 |
| **ActPatch** | 0.03 | 0.20 | 0.15 | 1.00 | 0.15 | 0.11 |
| **ACDC** | 0.07 | 0.02 | 0.05 | 0.15 | 1.00 | 0.05 |
| **Wang** | 0.11 | 0.25 | 0.43 | 0.11 | 0.05 | 1.00 |

Two method families: (1) **attribution-based** (EAP, EAP-IG, Wang ABA) with Jaccard 0.25-0.58, and (2) **structure-based** (Weight, partly ACDC) with Jaccard 0.00-0.07 to attribution methods.

### Cross-Method Ablation: Are Each Method's Heads Causal?

Baseline LD: +2.243 (200 RTI prompts). Zero ablation and resample ablation of each method's head set.

| Method | Heads | Zero delta | Resample delta | Flips LD? | Zero/Resample ratio |
|--------|-------|-----------|----------------|-----------|---------------------|
| **Weight** | 15 | **-2.464** | -0.355 | **YES** (-0.221) | **6.94** |
| EAP | 15 | -1.566 | -1.711 | No (+0.678) | 0.91 |
| EAP-IG | 15 | -1.731 | -1.932 | No (+0.513) | 0.90 |
| ActPatch | 15 | -1.067 | -1.372 | No (+1.176) | 0.78 |
| ACDC | 30 | -1.942 | -1.029 | No (+0.301) | 1.89 |

The weight circuit is the only set whose ablation flips LD sign. The zero/resample divergence ratio (6.94 for weight vs ~0.9 for EAP) distinguishes task-specific computation from general-purpose importance.

---

### Per-Head Cross-Method Membership: Circuit Heads

#### Backbone tier

| Head | Weight | EAP | EAP-IG | ActPatch | ACDC | Wang | ActPatch rank | Wang score |
|------|--------|-----|--------|----------|------|------|---------------|------------|
| **L0H8** | Y | | | | | | 131/144 | -0.230 |
| **L0H9** | Y | | | | Y | | 37/144 | -0.075 |
| **L0H11** | Y | | | | Y | | 65/144 | -0.449 |

Backbone heads are invisible to attribution methods (EAP, EAP-IG, Wang ABA find zero). They rank poorly in activation patching (37-131). Only ACDC finds 2/3 because its greedy pruning preserves structural dependencies. Wang ABA scores are negative (mean -0.252) — backbone heads write positional context, not token identity.

#### Detector tier

| Head | Weight | EAP | EAP-IG | ActPatch | ACDC | Wang | ActPatch rank | Wang score |
|------|--------|-----|--------|----------|------|------|---------------|------------|
| **L4H11** | Y | | | Y | Y | | **3/144** | +0.073 |

Found by 3/6 methods. EAP ranks it 42-91 across variants. Wang score near zero (it detects, doesn't copy). ActPatch ranks it #3, confirming strong individual importance. But note the DLA sign flip: DLA = -0.214 while total patching = +0.320. Its entire contribution is mediated through downstream copiers.

#### Copier tier

| Head | Weight | EAP | EAP-IG | ActPatch | ACDC | Wang | ActPatch rank | Wang score |
|------|--------|-----|--------|----------|------|------|---------------|------------|
| **L4H0** | Y | | | | | | 117/144 | +0.065 |
| **L5H6** | Y | | | | | | 20/144 | +0.164 |
| **L5H7** | Y | | | | | | 127/144 | +0.129 |
| **L7H0** | Y | | | | | | 64/144 | +1.051 |
| **L8H4** | Y | | | | | | 70/144 | +0.047 |
| **L8H7** | Y | | | | | | 40/144 | +2.271 |
| **L9H3** | Y | | | | | | 21/144 | +2.218 |
| **L9H10** | Y | | | | | | 123/144 | -0.021 |

No copier head is found by ANY non-weight method. This is the sharpest result: 8 heads, 5 other methods, zero overlap. Copier mean activation patching effect is +0.015 — individually below threshold due to 8-way redundancy. Phase 21 shows {L5H6, L9H3} alone preserves 123% of the copier contribution. EAP, ACDC, and ActPatch all fundamentally miss distributed computation.

Wang scores separate copiers into two groups: **high-copy** (L7H0=1.05, L8H7=2.27, L9H3=2.22) and **low-copy** (L4H0=0.065, L5H6=0.16, L8H4=0.05, L9H10=-0.02). High-copy heads do explicit token-identity writing. Low-copy heads amplify via subspace (BOS attention 69-82%), contributing through the direction they write rather than the token they copy.

#### Readout tier

| Head | Weight | EAP | EAP-IG | ActPatch | ACDC | Wang | ActPatch rank | Wang score |
|------|--------|-----|--------|----------|------|------|---------------|------------|
| **L10H11** | Y | | | | | Y | 54/144 | +1.073 |
| **L11H9** | Y | | | | | Y | 52/144 | +2.237 |
| **L11H11** | Y | | | | | | 83/144 | +1.037 |

Wang ABA finds 2/3 readout heads (L10H11 rank 17, L11H9 rank 3) because readout heads have strong per-token copy signals. No other non-weight method finds any readout head. Activation patching ranks readout at 52-83 — removing individual readout heads barely matters because the readout tier is antagonistic (+0.115 improvement when all removed, Phase 19).

---

### Non-Circuit Consensus Heads

These heads are found by multiple methods but are NOT in the weight circuit. They represent what attribution/activation methods find instead.

#### L9H9 — Universal consensus head (5/6 methods)

| Property | Value |
|----------|-------|
| **Found by** | EAP, EAP-IG, ActPatch, ACDC, Wang ABA |
| **NOT found by** | Weight |
| **Wang ABA score** | +1.552 (rank 8/144) |
| **ActPatch effect** | +0.316 (rank 5/144) |
| **EAP rank** | 3/144 |
| **DLA** | high direct logit contribution |

The most-agreed-upon head across all methods except Weight. Strong individual copy score and high marginal importance. Why does the weight method exclude it? Its OV/QK weight features don't match the weight circuit's discriminative signature — it's individually important but not part of the compositional backbone→detector→copier→readout pathway. It likely operates as an independent copying head that works in parallel to the circuit, not through it.

#### L10H0 — Strong output head (4/6 methods)

| Property | Value |
|----------|-------|
| **Found by** | EAP, EAP-IG, ActPatch, Wang ABA |
| **NOT found by** | Weight, ACDC |
| **Wang ABA score** | +1.725 (rank 7/144) |
| **ActPatch effect** | +0.371 (rank 2/144) |
| **W_OV copying score** | +160.793 (strongest copier in weight analysis Exp 1) |

#2 in activation patching and #7 in Wang ABA. Interestingly, the weight analysis eigenvalue test (Phase 10) found it has the strongest OV copying score of ANY head in the model (+160.8) — yet the weight circuit classifier didn't include it. This suggests the discriminative features used for circuit classification (bootstrap from roles.py ground truth) don't overlap with raw OV copying magnitude.

#### L10H6 — Attribution favorite (4/6 methods)

| Property | Value |
|----------|-------|
| **Found by** | EAP, EAP-IG, ActPatch, Wang ABA |
| **NOT found by** | Weight, ACDC |
| **Wang ABA score** | +1.128 (rank 15/144) |
| **ActPatch effect** | +0.195 (rank 10/144) |

Another L10 head with strong marginal importance across attribution methods. Part of the "L10 output cluster" that attribution methods consistently find (L10H0, L10H2, L10H6, L10H10 all appear in multiple methods).

#### L11H10 — EAP's #1 head

| Property | Value |
|----------|-------|
| **Found by** | EAP, EAP-IG |
| **NOT found by** | Weight, ActPatch, ACDC, Wang ABA |
| **EAP rank** | 1/144 |
| **Wang ABA score** | low |

The head EAP considers most important. Only found by the two EAP variants, not by any other method. This exemplifies how EAP can find heads with high gradient×activation products that don't show up in behavioral or causal tests.

#### L8H10 — Activation patching's #1 head

| Property | Value |
|----------|-------|
| **Found by** | ActPatch, ACDC |
| **NOT found by** | Weight, EAP, EAP-IG, Wang ABA |
| **ActPatch effect** | +0.610 (rank **1/144**) |
| **OV eigenvalue** | -138.820 (suppression head, Phase 10) |

The single most important head by activation patching. Its OV eigenvalue is deeply negative — it's a suppression head that inhibits the wrong answer. It operates outside the weight circuit's computational pathway but has enormous individual causal effect. Parallels IOI's Negative Name Movers.

---

### Method Sensitivity by Circuit Tier

Summary: which tiers does each method detect?

| Tier | Weight | EAP | EAP-IG | ActPatch | ACDC | Wang ABA |
|------|--------|-----|--------|----------|------|----------|
| Backbone (3 heads) | **3/3** | 0/3 | 0/3 | 0/3 | 2/3 | 0/3 |
| Detector (1 head) | **1/1** | 0/1 | 0/1 | **1/1** | **1/1** | 0/1 |
| Copier (8 heads) | **8/8** | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| Readout (3 heads) | **3/3** | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 |
| **Total recall** | **15/15** | **0/15** | **0/15** | **1/15** | **3/15** | **2/15** |

The weight method is the only method that finds every tier. All other methods have complete blind spots:
- **EAP/EAP-IG**: Find no circuit heads at all (0/15). They find general-purpose important heads in L9-L11.
- **ActPatch**: Finds only the detector (L4H11, rank 3) because it has strong individual importance. Misses all copiers (distributed) and all readout (antagonistic).
- **ACDC**: Finds 3 structural heads (L0H9, L0H11, L4H11) through its greedy pruning. Misses all copiers and readout.
- **Wang ABA**: Finds 2 readout induction heads (L10H11, L11H9) that have high per-token copy signals. Misses backbone/detector/copier.

This per-tier analysis explains WHY different methods disagree: each method is sensitive to a different aspect of circuit function, and no single non-weight method covers all four functional tiers.

## Cross-Scale Transfer: GPT-2 Small → Medium → Large

Weight-feature formulas trained on small's 15-head circuit transfer zero-shot to larger models. Each tier's top analogue and bootstrap stability:

### Per-Tier Transfer Stability

| Tier | Small GT | Medium top (stab) | Large top (stab) | Trend |
|------|---------|-------------------|-------------------|-------|
| **Backbone** | L0H8, L0H9, L0H11 | L2H0 (0.57), L2H13 (0.57) | **L2H9 (0.97)**, L2H19 (0.93), L4H15 (0.90) | Improves with scale — more early-layer candidates |
| **Detector** | L4H11 | L5H11 (0.53) | L35H17 (0.43) | Degrades — single-head roles don't transfer well |
| **Copier** | 8 heads | **L18H3 (0.97)**, L10H10 (0.93) | L35H0 (0.77), +5 tied at 0.77 | Medium: clear winner; Large: many candidates tied |
| **Readout** | L10H11, L11H9, L11H11 | L21H10 (0.77) | L29H10 (0.67), L29H4 (0.63) | Moderate at both scales |

### Cross-Task Transfer Comparison (best head per role)

| Task | Role | Medium stability | Large stability |
|------|------|-----------------|-----------------|
| RTI | backbone | 0.57 | **0.97** |
| RTI | copier | **0.97** | 0.77 |
| IOI | DTH | **0.87** | 0.40 |
| IOI | PTH | **0.80** | 0.77 |
| IOI | S-Inh | 0.83 | **0.93** |
| IOI | NM | **0.93** | 0.83 |
| IOI | NegNM | 0.60 | **0.70** |
| Induction | IND | **0.93** | 0.37 |
| SVA | embed | **0.80** | 0.33 |
| SVA | encode | 0.73 | **0.83** |
| SVA | route | **0.80** | 0.57 |
| SVA | output | 0.63 | **0.73** |
| GT | early_gt | -- | **0.90** |
| GP | late_ga | **0.90** | 0.87 |
| GP | name_bind | 0.73 | **0.83** |

### Multi-Role Hub Heads (stability ≥ 0.7 in ≥2 tasks)

**Medium hubs:**
| Head | Tasks/Roles |
|------|------------|
| L5H11 | IOI/PTH, induction/PTH |
| L11H1 | IOI/NM, induction/IND |
| L14H2 | IOI/NM, SVA/encode |
| L7H2 | induction/IND, GP/late_ga |
| L9H9 | induction/IND, GP/late_ga |

**Large hubs:**
| Head | Tasks/Roles |
|------|------------|
| L14H1 | RTI/copier, IOI/PTH, induction/PTH |
| L35H17 | RTI/copier, GP/late_ga |
| L34H6 | RTI/copier, IOI/NegNM |
| L0H8 | GT/early_gt, GP/name_bind |

### Scale Observations

- **Backbone: improves dramatically** (0.57 → 0.97). More early-layer heads at scale provide better candidates matching backbone's weight profile.
- **Copier: shifts from one winner to many.** Medium has one clear copier analogue (L18H3 at 0.97). Large has 6 candidates all at 0.77 — copier functionality distributes across more heads at scale.
- **Induction: degrades sharply** (0.93 → 0.37). The formulas can't discriminate among 720 heads for a role defined by only 5 training examples.
- **Detector: consistently weak.** Single-head roles (L4H11) don't give the bootstrap enough training signal. Both medium and large detector transfer is marginal.
- **L35H17 flag:** Appears in 4 tasks in large. Needs transfer-controls to determine if it's a genuine hub or a trivial last-layer artifact.
