# Cross-Model Head Characterization: GPT-2 Medium / Large / XL

Weight-predicted circuit heads transferred from GPT-2 small, validated with EAP and IIA.
Companion to `part4_rigorous_circuit_finding/HEAD_CHARACTERIZATION.md` (GPT-2 small only).

## Method

1. Extract 108 weight features per head (spectral + embedding alignment)
2. Bootstrap transfer: 30 rounds, greedy feature selection on small's ground truth, apply zero-shot to target
3. Stability = fraction of bootstrap rounds where head is selected (threshold >= 0.7)
4. Validate with IIA (activation swap) and EAP (gradient attribution)

## Data Sources

| File | Contents |
|------|----------|
| `data/transfer_results_medium.json` | Weight-predicted heads for medium (stability scores, top features) |
| `data/transfer_results_large.json` | Weight-predicted heads for large |
| `data/transfer_results_xl.json` | Weight-predicted heads for XL |
| `data/control_results_{medium,large,xl}.json` | Depth-random baselines (10 trials) |
| `data/eap_cross_model_small.json` | EAP full head rankings for GPT-2 small |
| `data/eap_cross_model_medium.json` | EAP full head rankings for GPT-2 medium |
| `data/iia_cross_model_{medium,large,xl}.json` | IIA per-role results with diagnostics |

---

## GPT-2 Medium (24 layers, 16 heads, 384 total)

### RTI Circuit (7 heads at stability >= 0.7)

| Tier | Head | Stability | Top Feature | Notes |
|------|------|-----------|-------------|-------|
| Copier | L18H3 | 0.97 | k_align_ica3_w | Highest stability across all medium heads |
| Copier | L10H10 | 0.93 | k_align_ica4_w | |
| Copier | L16H7 | 0.90 | k_align_ica0_w | |
| Copier | L12H8 | 0.90 | ov_top_in_ica1 | |
| Copier | L13H5 | 0.77 | ov_effective_rank | |
| Copier | L4H13 | 0.73 | k_align_ica1_w | |
| Readout | L21H10 | 0.77 | | |

Backbone: no heads >= 0.7 (best: L2H0, L2H13 @ 0.57).
Detector: no heads >= 0.7 (best: L5H11 @ 0.53).

### IOI Circuit (13 heads)

| Tier | Head | Stability | Notes |
|------|------|-----------|-------|
| DTH | L23H4 | 0.87 | |
| PTH | L5H11 | 0.80 | Also in induction PTH |
| S-Inh | L15H10 | 0.83 | |
| S-Inh | L15H1 | 0.83 | |
| S-Inh | L17H12 | 0.80 | Also EAP rank 3 (only overlap) |
| S-Inh | L11H3 | 0.80 | |
| S-Inh | L13H9 | 0.80 | |
| S-Inh | L14H5 | 0.73 | |
| S-Inh | L19H1 | 0.73 | Also EAP rank 13 |
| NM | L14H2 | 0.93 | |
| NM | L11H1 | 0.90 | |
| NM | L16H15 | 0.90 | |
| NM | L20H13 | 0.77 | |

### SVA Circuit (5 heads)

| Tier | Head | Stability |
|------|------|-----------|
| Embed | L14H15 | 0.80 |
| Encode | L3H10 | 0.73 |
| Encode | L14H2 | 0.70 |
| Encode | L5H14 | 0.70 |
| Route | L10H15 | 0.80 |

### Greater-Than Circuit (2 heads)

| Tier | Head | Stability |
|------|------|-----------|
| Early GT | L21H9 | 0.73 |
| Early GT | L10H8 | 0.70 |

### EAP Top-10 (medium)

| Task | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|------|--------|--------|--------|--------|--------|
| RTI | L22H14 (-0.186) | L23H3 (-0.177) | L17H12 (-0.107) | L18H9 (-0.101) | L20H6 (+0.056) |
| IOI | L22H14 (-0.269) | L18H9 (-0.234) | **L17H12** (-0.227) | L23H3 (-0.176) | L17H4 (+0.099) |
| SVA | L18H6 (+0.081) | L20H2 (+0.073) | L19H14 (+0.063) | L17H11 (+0.060) | L16H7 (+0.054) |
| GT | L14H14 (+0.019) | L11H1 (+0.010) | L13H12 (+0.008) | L9H9 (+0.006) | L21H7 (+0.005) |

Bold = also in weight circuit. EAP and weight methods are nearly completely disjoint on medium.

### IIA Results (medium)

| Task | Role | Heads | IIA | Logit Shift | Control Shift |
|------|------|-------|-----|-------------|---------------|
| IOI | DTH | L23H4 | 0.000 | 0.008 | -0.071 |
| IOI | PTH | L5H11 | 0.000 | 0.192 | -0.008 |
| IOI | S-Inh | 7 heads | 0.000 | 0.173 | 0.175 |
| IOI | NM | 4 heads | 0.000 | 0.207 | 0.158 |
| SVA | embed | L14H15 | 0.000 | -0.003 | 0.021 |
| SVA | encode | 3 heads | 0.000 | 0.006 | 0.032 |
| SVA | route | L10H15 | 0.000 | 0.008 | 0.014 |
| GT | early_gt | 2 heads | 0.000 | 0.000 | 0.037 |

All zeros. Logit shifts are tiny and comparable to controls. The weight-transferred heads don't carry task-specific causal information at medium scale via head-level activation swap.

---

## GPT-2 Large (36 layers, 20 heads, 720 total)

### RTI Circuit (17 heads)

| Tier | Head | Stability |
|------|------|-----------|
| Backbone | L2H9 | 0.97 |
| Backbone | L2H19 | 0.93 |
| Backbone | L4H15 | 0.90 |
| Backbone | L3H18 | 0.87 |
| Backbone | L3H10 | 0.70 |
| Copier | L35H0 | 0.77 |
| Copier | L35H12 | 0.77 |
| Copier | L23H10 | 0.77 |
| Copier | L35H6 | 0.77 |
| Copier | L14H1 | 0.77 |
| Copier | L35H14 | 0.77 |
| Copier | L35H17 | 0.77 |
| Copier | L3H8 | 0.73 |
| Copier | L35H1 | 0.73 |
| Copier | L34H18 | 0.73 |
| Copier | L12H12 | 0.70 |
| Copier | L34H6 | 0.70 |

Backbone transfers very well to large (0.70-0.97). Copiers cluster heavily in L34-L35.

### IOI Circuit (5 heads)

| Tier | Head | Stability |
|------|------|-----------|
| PTH | L14H1 | 0.77 |
| S-Inh | L1H6 | 0.93 |
| S-Inh | L29H3 | 0.70 |
| NM | L22H17 | 0.83 |
| NegNM | L34H6 | 0.70 |

Smaller circuit than medium (5 vs 13 heads). S-Inh collapsed from 7 to 2 heads.

### SVA Circuit (7 heads)

| Tier | Head | Stability |
|------|------|-----------|
| Encode | L18H15 | 0.83 |
| Encode | L29H1 | 0.80 |
| Encode | L7H15 | 0.73 |
| Encode | L5H13 | 0.70 |
| Encode | L5H15 | 0.70 |
| Output | L35H13 | 0.73 |
| Output | L29H10 | 0.73 |

### Greater-Than Circuit (5 heads) -- IIA = 0.909!

| Tier | Head | Stability | IIA | Logit Shift |
|------|------|-----------|-----|-------------|
| Early GT | L0H8 | 0.90 | - | - |
| Early GT | L0H0 | 0.83 | - | - |
| Early GT | L13H12 | 0.80 | - | - |
| Early GT | L0H14 | 0.80 | - | - |
| Early GT | L7H6 | 0.77 | - | - |
| **All 5** | | | **0.909** | **1.878** |

The standout result. 5 weight-predicted heads achieve 90.9% IIA on greater-than with a massive logit shift (1.88 vs 0.0 for controls). Only 33/60 valid pairs though (20 skipped because base model got the answer wrong).

Three of five heads are in L0 — backbone-tier heads that write fixed directions. This is the simplest circuit and the one where head-level activation swap works.

### IIA Results (large)

| Task | Role | Heads | IIA | Logit Shift | Control |
|------|------|-------|-----|-------------|---------|
| IOI | PTH | L14H1 | 0.000 | 0.095 | 0.008 |
| IOI | S-Inh | L1H6, L29H3 | 0.000 | 0.000 | 0.003 |
| IOI | NM | L22H17 | 0.000 | 0.005 | 0.024 |
| IOI | NegNM | L34H6 | 0.000 | 0.035 | 0.000 |
| IOI | FULL | 5 heads | 0.000 | 0.134 | - |
| SVA | encode | 5 heads | 0.000 | 0.015 | 0.036 |
| SVA | output | 2 heads | 0.000 | 0.002 | 0.001 |
| **GT** | **early_gt** | **5 heads** | **0.909** | **1.878** | **0.000** |

---

## GPT-2 XL (48 layers, 25 heads, 1200 total)

### RTI Circuit (16 heads)

| Tier | Head | Stability |
|------|------|-----------|
| Backbone | L1H21 | 0.87 |
| Backbone | L3H22 | 0.77 |
| Backbone | L1H12 | 0.77 |
| Copier | L13H20 | 0.83 |
| Copier | L14H12 | 0.80 |
| Copier | L15H19 | 0.80 |
| Copier | L12H21 | 0.80 |
| Copier | L11H2 | 0.77 |
| Copier | L47H19 | 0.73 |
| Copier | L47H20 | 0.73 |
| Copier | L47H0 | 0.73 |
| Copier | L47H6 | 0.70 |
| Readout | L41H15 | 0.90 |
| Readout | L39H17 | 0.83 |
| Readout | L36H5 | 0.70 |
| Readout | L47H0 | 0.70 |

All four tiers present at XL. L47 copiers suggest the circuit extends to the very last layer.

### IOI Circuit (10 heads)

| Tier | Head | Stability |
|------|------|-----------|
| PTH | L12H21 | 0.80 |
| PTH | L15H19 | 0.70 |
| S-Inh | L35H9 | 0.77 |
| S-Inh | L44H21 | 0.73 |
| S-Inh | L41H16 | 0.73 |
| S-Inh | L39H12 | 0.73 |
| S-Inh | L36H9 | 0.70 |
| S-Inh | L37H0 | 0.70 |
| S-Inh | L42H22 | 0.70 |
| NM | L27H23 | 0.83 |

### SVA Circuit (8 heads)

| Tier | Head | Stability |
|------|------|-----------|
| Embed | L33H22 | 0.70 |
| Encode | L6H23 | 0.73 |
| Encode | L6H14 | 0.73 |
| Encode | L12H22 | 0.73 |
| Encode | L6H4 | 0.70 |
| Encode | L1H4 | 0.70 |
| Route | L44H16 | 0.93 |
| Output | L44H11 | 0.70 |

### Greater-Than Circuit (3 heads)

| Tier | Head | Stability |
|------|------|-----------|
| Early GT | L28H24 | 0.73 |
| Early GT | L17H1 | 0.73 |
| Early GT | L7H23 | 0.73 |

### IIA Results (XL)

All zeros. No task achieves nonzero IIA at XL scale, including greater-than (which worked on large).

| Task | Heads | IIA | Logit Shift |
|------|-------|-----|-------------|
| IOI | 10 heads | 0.000 | -0.199 (full) |
| SVA | 8 heads | 0.000 | 0.011 (full) |
| GT | 3 heads | 0.000 | 0.001 |

---

## Cross-Scale Patterns

### Circuit size vs model size

| Task | Small | Medium | Large | XL |
|------|-------|--------|-------|-----|
| RTI | 15 | 7 | 17 | 16 |
| IOI | 15 | 13 | 5 | 10 |
| SVA | 12 | 5 | 7 | 8 |
| GT | 5 | 2 | 5 | 3 |

No monotonic trend. Large has the biggest RTI circuit (17 heads, 12 of which are copiers in L34-L35).

### IIA summary

| Task | Medium | Large | XL |
|------|--------|-------|-----|
| IOI | 0.000 | 0.000 | 0.000 |
| SVA | 0.000 | 0.000 | 0.000 |
| GT | 0.000 | **0.909** | 0.000 |

GT-large is the only successful transfer via head-level activation swap. Three possible explanations (from Perplexity analysis):
1. **Alignment success**: GT's causal variable happens to be axis-aligned with head outputs at large scale
2. **Simpler circuit**: 3/5 heads are L0 (backbone), easiest to swap
3. **Scale window**: GT circuit at this specific scale hasn't developed redundancy yet

### EAP overlap with weight circuit

| Task | Small overlap (top-15) | Medium overlap (top-15) |
|------|----------------------|------------------------|
| RTI | 0/15 | 0/15 |
| IOI | 8/15 | 2/15 |
| SVA | 0/15 | 0/15 |
| GT | 0/15 | 0/15 |

IOI is the only task with meaningful EAP-weight overlap, and it drops from 53% to 13% at medium scale.

### Cross-task hub heads

Some heads appear in multiple tasks' circuits:

**Medium**: L5H11 (IOI PTH + induction PTH), L14H2 (IOI NM + SVA encode)
**Large**: L14H1 (RTI copier + IOI PTH + induction PTH), L34H6 (RTI copier + IOI NegNM)
**XL**: L12H21 (IOI PTH + induction PTH + RTI copier), L15H19 (IOI PTH + induction PTH + RTI copier)
