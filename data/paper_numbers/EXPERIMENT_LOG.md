# Cross-Model Circuit Transfer: Experiment Log

Weight-only circuit discovery on GPT-2 small, transferred zero-shot to medium/large/xl.
This is essentially its own paper: "do weight-space circuits generalize across scale?"

## Phase 1: Weight Feature Transfer (complete)

**Date**: 2026-05-10
**Scripts**: `run_gpt2_medium_transfer.py`, `run_transfer_controls.py`
**W&B runs**: `gpt2-medium-transfer`, `gpt2-large-transfer`, `gpt2-xl-transfer`

Extracted 108 weight features per head on all GPT-2 scales. Bootstrap transfer (30 rounds, greedy feature selection on small's ground truth) identifies heads on larger models with stability scores.

**Key result**: Transfer finds heads at all scales. Circuit sizes range 2-17 heads depending on task and scale. See `HEAD_CHARACTERIZATION_CROSS_MODEL.md` for full tables.

**Control baselines** (`control_results_*.json`): Depth-random controls get stability 0.55-0.93 depending on role. Most weight-predicted heads exceed controls, but copier tier controls are surprisingly high (0.87-0.93), meaning copier-like weight structure is common in later layers.

**Data**: `data/transfer_results_{medium,large,xl}.json`, `data/control_results_*.json`

---

## Phase 2: EAP Comparison (complete for small + medium)

**Date**: 2026-05-11
**Script**: `run_cross_model_eap.py`
**W&B runs**: `eap-small-20260511T025712`, `eap-medium-20260511T025544`

Basic EAP (2 fwd + 1 bwd per batch) on GPT-2 small and medium for RTI, IOI, SVA, greater-than.

**Key result**: EAP and weight circuits are nearly completely disjoint.

| Task | Small overlap (top-15) | Medium overlap (top-15) |
|------|----------------------|------------------------|
| RTI | 0/15 | 0/15 |
| IOI | 8/15 (53%) | 2/15 (13%) |
| SVA | 0/15 | 0/15 |
| GT | 0/15 | 0/15 |

IOI overlap drops from 53% to 13% at medium. This is the complementary framing: weight analysis finds compositional structure (copiers), EAP finds marginal causal importance (name movers).

**EAP large/xl**: OOM'd. Score matrix allocation needs 27+ GiB for gpt2-large (36 layers × 20 heads). Would need chunked accumulation to scale further.

**Data**: `data/eap_cross_model_{small,medium}.json`

---

## Phase 3: IIA Validation (complete)

**Date**: 2026-05-11
**Script**: `run_cross_model_iia.py` (with diagnostics: n_valid, n_skipped, mean_logit_shift)
**W&B runs**: `iia-medium-*`, `iia-large-*`, `iia-xl-*`

Interchange Intervention Accuracy: swap activations at weight-predicted heads from counterfactual source prompts, check if output flips.

**Key result**: All zeros except GT-large (0.909).

| Task | Medium | Large | XL |
|------|--------|-------|-----|
| IOI | 0.000 | 0.000 | 0.000 |
| SVA | 0.000 | 0.000 | 0.000 |
| GT | 0.000 | **0.909** | 0.000 |

Diagnostics confirm this is real, not a data issue: valid pairs range 33-80, logit shifts are tiny except GT-large (1.878).

**GT-large success**: 5 heads (L0H0, L0H8, L0H14, L7H6, L13H12), 3 in layer 0. Simplest circuit, most axis-aligned with head outputs. Control IIA = 0.000.

**Interpretation**: Weight features identify structurally similar heads, but head-level activation swap is too coarse for most tasks at scale. The causal variable may exist in a rotated subspace (not axis-aligned with individual heads). GT-large works because its circuit is simple enough to be axis-aligned.

**Data**: `data/iia_cross_model_{medium,large,xl}.json`

---

## Phase 4: Behavioral Validation (running)

**Date**: 2026-05-11
**Script**: `experiments/batch1/run_cross_model_behavioral_validation.py`
**Pods**: `bval-medium-0314`, `bval-large-0314`, `bval-xl-0315`

Ablation faithfulness: mean-ablate non-circuit heads, measure logit diff preservation. Weaker than IIA but tests whether the predicted heads matter at all.

This is the key test. If faithfulness > random controls even though IIA = 0, the heads carry task-relevant information but in a non-swappable way (subspace misalignment, redundancy, etc.).

**Data**: pending

---

## Phase 5: IIA Improvement Strategies (complete)

**Date**: 2026-05-11
**Script**: `experiments/batch3_iia_improvement/run_strategy_group.py`
**Pods**: 6 pods (fast/greedy/ceiling × medium/large), all completed
**W&B runs**: pod-log artifacts for each pod

### Results Summary

**The original IIA=0 was a threshold artifact.** Lowering the threshold or adding heads fixes it.

**Threshold sweep (fast group):**

| Task | Medium (thresh 0.0) | Large (thresh 0.0) | Large L0 alone |
|------|---------------------|---------------------|----------------|
| IOI | 0.800 (178 heads) | **1.000** (214 heads) | **1.000** |
| SVA | **1.000** (177 heads) | **0.917** (205 heads) | **0.917** |
| GT | 0.968 (109 heads) | **1.000** (106 heads) | **1.000** |

**Ceiling (minimum circuits, unrestricted greedy):**

| Task | Medium | Large |
|------|--------|-------|
| IOI | **7 heads** (IIA=0.900) | 16 heads (IIA=0.812) |
| SVA | **3 heads** (IIA=0.924) | 7 heads (IIA=0.847) |
| GT | **3 heads** (IIA=0.806) | 15 heads (IIA=0.970) |

**L0 bottleneck on large:** Greedy search for IOI adds 8 late-layer heads (IIA=0.000), then 5 L0 heads jump to IIA=0.963. L0H3 alone causes +0.651 IIA jump. GT ceiling: 13 heads at IIA=0.000, then L0H14→0.758, L0H3→0.970.

**EAP comparison (medium only, EAP OOMs on large):**
- EAP top-5: SVA=1.000, GT=0.952, IOI=0.000
- EAP top-20: all tasks IIA=1.000
- Union (weight + EAP) outperforms either alone
- Near-zero overlap between methods (~0-2 shared heads in top-15)

**Data**: `experiments/batch3_iia_improvement/RESULTS_FAST_GROUP.md`, `RESULTS_GREEDY_CEILING.md`

---

## Phase 6: Pythia Transfer (complete)

**Date**: 2026-05-10
**Script**: `run_pythia_transfer.py`
**W&B runs**: `pythia-160m-transfer`, `pythia-410m-transfer`, `pythia-14b-transfer`

Cross-architecture transfer: GPT-2 small's weight features applied to Pythia models.

**Data**: `data/transfer_pythia{160m,410m,1.4b}.json`, `data/features_pythia*.json`, `data/clusters_pythia*.json`, `data/behavioral_pythia*.json`

---

## Phase 7: EAP + EAP-IG at All Scales (complete)

**Date**: 2026-05-11
**Script**: `run_cross_model_eap.py` (updated: `--methods EAP EAP-IG-inputs --ig-steps 10`, compact 2D accumulation)
**Pods**: `eap-medium-1559` (L4), `eap-large-1623` (A40), `eap-xl-1559` (A40)
**W&B runs**: `eap-medium-20260511T160849`, `eap-large-20260511T163607`, `eap-xl-20260511T162749`

### Compact mode (the engineering fix)

The 3D marginal accumulators `(n_fwd, n_bwd, d_model)` use ~58 GB for XL, causing OOM on H100 NVL (94 GB). Added `compact=True`: accumulates 2D edge scores directly via `einsum("b p w d, b p v d -> w v")` instead of the 3D intermediate. Mathematically identical — same einsum contracted during accumulation rather than after. Memory: ~93 GB → ~30 GB for XL. Runs on A40 ($0.44/hr) instead of needing an unavailable H200.

### EAP vs EAP-IG agreement across scale

Both methods run on all 4 tasks (RTI, IOI, SVA, GT) at medium, large, XL.

**Top-5 overlap (EAP vs EAP-IG):**

| Task | Medium | Large | XL |
|------|--------|-------|----|
| RTI | 3/5 | 1/5 | 3/5 |
| IOI | 4/5 | 3/5 | 4/5 |
| SVA | 5/5 | 4/5 | 5/5 |
| GT | 5/5 | 2/5 | 5/5 |

**Top-15 overlap:**

| Task | Medium | Large | XL |
|------|--------|-------|----|
| RTI | 12/15 | 9/15 | 13/15 |
| IOI | 13/15 | 11/15 | 13/15 |
| SVA | 10/15 | 9/15 | 12/15 |
| GT | 13/15 | 6/15 | 12/15 |

Medium and XL: high agreement (10-13/15). Large: lower agreement (6-11/15). The divergence on large is the L0 story.

### KEY FINDING: EAP-IG finds L0 on large, basic EAP does not

**L0 heads in top-15 by method (GPT-2 large):**

| Task | EAP L0 heads | EAP-IG L0 heads |
|------|-------------|-----------------|
| RTI | 0 | L0H14, L0H3, L0H0, L0H10, L0H2 (5) |
| IOI | 0 | L0H14, L0H3, L0H0, L0H10 (4) |
| SVA | 0 | L0H14, L0H10, L0H13, L0H0, L0H3, L0H16 (6) |
| GT | 0 | L0H14, L0H3, L0H0, L0H10 (4) |

Basic EAP's first-order gradient approximation completely misses L0. EAP-IG's integrated gradients captures the nonlinear causal path through L0 that our IIA experiments independently discovered. L0H14, L0H3, L0H0 appear in every task — these are exactly the heads the greedy IIA search identified as the causal bottleneck (Phase 5).

**This is independent confirmation**: two completely different methods (IIA greedy search and EAP-IG) converge on the same L0 heads, while basic EAP misses them entirely. The L0 bottleneck is real and requires nonlinear attribution to detect.

**Why only on large?** On medium, L0 is not a bottleneck (no single layer dominates IIA), so EAP and EAP-IG agree. On XL, L0 may be even more distributed — neither method finds L0 in top-15.

### Weight circuit vs EAP/EAP-IG overlap

**Near-zero overlap persists across all scales and both methods:**

| Scale | Task | WC size | EAP overlap | EAP-IG overlap |
|-------|------|---------|-------------|----------------|
| Medium | RTI | 15 | 0/15 | 0/15 |
| Medium | IOI | 13 | 1/15 | 1/15 |
| Medium | SVA | 5 | 0/15 | 0/15 |
| Medium | GT | 2 | 0/15 | 0/15 |
| Large | RTI | 17 | 0/15 | 0/15 |
| Large | IOI | 5 | 0/15 | 0/15 |
| Large | SVA | 7 | 0/15 | 0/15 |
| Large | GT | 5 | 0/15 | 2/15 |
| XL | RTI | 15 | 0/15 | 0/15 |
| XL | IOI | 10 | 1/15 | 1/15 |
| XL | SVA | 8 | 0/15 | 0/15 |
| XL | GT | 3 | 0/15 | 0/15 |

Weight circuits and activation-based circuits (both EAP and EAP-IG) identify almost entirely disjoint head sets at every scale. The complementarity is not an artifact of using basic EAP — it persists with the more accurate EAP-IG method too.

### EAP head rankings across scale

**EAP top-5 by scale:**

| Task | Medium | Large | XL |
|------|--------|-------|----|
| RTI | L22H14, L23H3, L17H12, L18H9, L20H6 | L32H0, L27H11, L26H0, L34H14, L30H0 | L39H9, L32H13, L26H14, L42H22, L39H12 |
| IOI | L22H14, L18H9, L17H12, L23H3, L17H4 | L32H0, L26H0, L22H0, L30H0, L34H14 | L32H13, L26H14, L39H9, L26H20, L42H22 |
| SVA | L18H6, L20H2, L19H14, L17H11, L16H7 | L24H3, L32H5, L14H0, L25H4, L16H17 | L39H18, L34H24, L26H18, L23H9, L27H0 |
| GT | L14H14, L11H1, L13H12, L9H9, L21H7 | L22H5, L25H14, L18H9, L20H13, L17H19 | L26H23, L25H18, L23H19, L25H16, L22H6 |

**EAP-IG top-5 by scale:**

| Task | Medium | Large | XL |
|------|--------|-------|----|
| RTI | L22H14, L20H6, L15H14, L23H3, L20H0 | **L0H14**, L32H0, **L0H3**, **L0H0**, L29H0 | L39H9, L32H13, L42H22, L24H16, L36H4 |
| IOI | L22H14, L18H9, L17H12, L23H3, L19H1 | L32H0, **L0H14**, L26H0, L22H0, L29H0 | L32H13, L39H9, L26H20, L26H14, L28H22 |
| SVA | L18H6, L19H14, L20H2, L17H11, L16H7 | **L0H14**, L24H3, L32H5, L16H17, L14H0 | L39H18, L26H18, L34H24, L27H0, L23H9 |
| GT | L14H14, L11H1, L13H12, L9H9, L21H7 | **L0H14**, L18H9, L24H8, **L0H3**, L25H14 | L26H23, L25H18, L23H19, L25H16, L22H6 |

L0 heads (bolded) appear exclusively at the large scale in EAP-IG.

### Timing

| Scale | Task | EAP (s) | EAP-IG (s) | Ratio |
|-------|------|---------|------------|-------|
| Medium | RTI | 8 | 108 | 14x |
| Medium | SVA | 6 | 42 | 7x |
| Large | RTI | 29 | 140 | 5x |
| Large | SVA | 20 | 59 | 3x |
| XL | RTI | 51 | 311 | 6x |
| XL | SVA | 26 | 160 | 6x |

EAP-IG is 3-14x slower (10 IG steps). The ratio decreases at larger scale because model forward pass dominates over IG overhead.

**Data**: `data/eap_cross_model_{medium,large,xl}.json`, `data/eap_ig_inputs_cross_model_{medium,large,xl}.json`

---

## Open Questions

1. ~~**Can lowering the threshold or adding EAP heads fix IIA?**~~ **ANSWERED YES.** Threshold 0.0 gives IIA=0.8-1.0 across the board. EAP top-20 gives 1.0 on medium. The original zeros were a threshold artifact.
2. ~~**Why does GT-large work but GT-medium and GT-xl don't?**~~ **PARTIALLY ANSWERED.** GT works at all thresholds on large because its circuit is L0-dominated (3/5 heads in L0). On medium, GT needs more heads (threshold 0.0 → 0.968 with 109 heads). The L0 bottleneck is scale-dependent.
3. **Will behavioral validation (Phase 4) show nonzero faithfulness?** Still pending — bval pods failed (wrong artifact upload path), need to fix and re-launch.
4. ~~**Can we get EAP on large/xl?**~~ **ANSWERED YES.** Compact 2D accumulation bypasses the 58 GB memory wall. EAP + EAP-IG both run on A40 (48 GB) for all scales including XL. Bonus finding: EAP-IG finds L0 heads on large that basic EAP misses.
5. **Why does ceiling find 16-head IOI circuits on large WITHOUT L0, while greedy-from-weight needs L0?** The mid-layer alternative route (L18-L32) achieves IIA=0.812 vs L0-route's 0.963. Are these truly independent circuits or do they share an information bottleneck?
6. **What are the L0 heads actually doing?** They operate on raw token embeddings. Swapping L0 effectively swaps the initial positional/identity encoding. Is this a deep finding or a trivial one (swap the input, swap the output)?
7. **Why does EAP-IG find L0 on large but basic EAP doesn't?** The causal effect of L0 is nonlinear — L0 heads route information that only matters when combined with downstream processing. First-order gradient (EAP) linearizes this away; integrated gradients (EAP-IG) captures the full path. Is this specific to large, or does it appear at other scales with different IG step counts?
8. **Do weight circuits generalize better across example distributions?** Weight features are example-independent; EAP scores depend on specific prompt templates. If you train EAP on IOI template set A and test on template set B, do weight circuits maintain IIA while EAP circuits degrade? This would be a strong argument for the complementarity framing — weight circuits capture structural capacity, EAP captures prompt-specific causal importance.
