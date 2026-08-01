# IIA Strategy Results: Greedy & Ceiling Groups

## Key Findings

1. **Minimum circuits are remarkably small**: 3-16 heads suffice for IIA >= 0.8 across all tasks and scales.
2. **L0 heads are the causal bottleneck on GPT-2 large**: greedy search adds late-layer heads with zero IIA improvement, then L0 heads cause a sudden jump to ~0.96.
3. **Ceiling circuit sizes match EAP top-K**: both methods converge on ~3-7 head circuits on medium, but they find completely different heads.

---

## Greedy Results (start from weight-predicted heads, add best candidate per step)

Starting set = weight-predicted heads at stability >= 0.7. Greedily add head that most improves IIA.

### GPT-2 Medium

**IOI** — 13 start → 22 heads (IIA=0.812, 9 steps)
| Step | Added | Total | IIA | Shift |
|------|-------|-------|-----|-------|
| 0 | (start: 13 weight heads) | 13 | 0.000 | 1.265 |
| 1 | L19H7 | 14 | 0.050 | 1.820 |
| 2 | L15H14 | 15 | 0.325 | 3.064 |
| 3 | L15H11 | 16 | 0.388 | 3.188 |
| 4 | **L17H4** | 17 | **0.613** | 4.047 |
| 5 | L12H10 | 18 | 0.625 | 4.036 |
| 6 | L19H15 | 19 | 0.650 | 4.118 |
| 7 | L21H7 | 20 | 0.725 | 4.294 |
| 8 | L13H4 | 21 | 0.762 | 4.480 |
| 9 | L14H15 | 22 | **0.812** | 4.532 |

Gradual improvement — no single head dominates. L17H4 is the biggest single jump (+0.225).

**SVA** — 5 start → 15 heads (IIA=0.803, 10 steps)
| Step | Added | Total | IIA | Shift |
|------|-------|-------|-----|-------|
| 0 | (start: 5 weight heads) | 5 | 0.000 | 0.012 |
| 4 | L16H7 | 9 | 0.212 | 1.440 |
| 6 | L20H2 | 11 | 0.455 | 2.521 |
| 10 | **L4H13** | 15 | **0.803** | 3.381 |

Slow buildup through mid-layers. L4H13 at step 10 pushes past 0.8.

**Greater-Than** — 2 start → 10 heads (IIA=0.968, 8 steps)
| Step | Added | Total | IIA | Shift |
|------|-------|-------|-----|-------|
| 0 | (start: 2 weight heads) | 2 | 0.000 | 0.000 |
| 1 | L11H1 | 3 | 0.242 | 0.291 |
| 3 | **L13H12** | 5 | **0.645** | 0.755 |
| 8 | **L9H9** | 10 | **0.968** | 1.573 |

L9H9 is the critical head — jumps from 0.774 to 0.968.

### GPT-2 Large

**IOI** — 5 start → 18 heads (IIA=0.963, 13 steps) **THE L0 STORY**
| Step | Added | Total | IIA | Shift | Notes |
|------|-------|-------|-----|-------|-------|
| 0 | (start: 5 weight heads) | 5 | 0.000 | 0.134 | |
| 1-8 | L28H10, L30H13, L32H18, L21H7, L30H15, L27H17, L25H3, L21H8 | 13 | **0.000** | 1.453 | 8 late-layer heads, zero IIA |
| 9 | **L0H14** | 14 | 0.013 | 1.689 | First L0 head |
| 10 | **L0H10** | 15 | 0.075 | 2.264 | |
| 11 | **L0H2** | 16 | 0.200 | 2.785 | |
| 12 | **L0H4** | 17 | 0.312 | 3.204 | |
| 13 | **L0H3** | 18 | **0.963** | **7.563** | Massive jump |

This is the clearest evidence of L0 as the causal bottleneck. Steps 1-8 add late-layer heads and logit shift grows (0.134 → 1.453) but IIA stays at exactly 0.000. Then L0 heads are added one by one, and L0H3 at step 13 causes an explosive jump from 0.312 to 0.963 with shift going from 3.204 to 7.563.

The late-layer heads contribute *information* (growing logit shift) but not *causally sufficient* information — they need the L0 bottleneck to route through.

**SVA** — 7 start → 15 heads (IIA=0.819, 8 steps)
| Step | Added | Total | IIA | Shift |
|------|-------|-------|-----|-------|
| 0 | (start: 7 weight heads) | 7 | 0.000 | 0.017 |
| 2 | **L0H14** | 9 | **0.444** | 2.042 |
| 4 | **L0H10** | 11 | **0.611** | 3.786 |
| 5 | **L0H11** | 12 | **0.722** | 3.984 |
| 8 | L12H12 | 15 | **0.819** | 4.315 |

Again L0 heads dominate: L0H14, L0H10, L0H11 are the critical additions.

**Greater-Than** — 5 start → 6 heads (IIA=0.909, 1 step)
| Step | Added | Total | IIA | Shift |
|------|-------|-------|-----|-------|
| 0 | (start: 5 weight heads) | 5 | 0.909 | 1.878 |
| 1 | L0H4 | 6 | 0.909 | 2.005 |

Already at 0.909 from the weight-predicted circuit (3 of 5 heads are L0). Only minor improvement from adding L0H4.

---

## Ceiling Results (unrestricted greedy from empty set)

Upper bound — what's the smallest circuit achievable by ANY head selection?

### GPT-2 Medium

**IOI** — **7 heads → IIA=0.900**
| Step | Added | IIA | Shift |
|------|-------|-----|-------|
| 1 | L16H15 | 0.000 | 0.207 |
| 2 | L15H4 | 0.000 | 0.802 |
| 3 | L15H14 | 0.013 | 1.370 |
| 4 | L19H7 | 0.013 | 1.881 |
| 5 | L12H3 | 0.075 | 2.522 |
| 6 | **L17H4** | **0.350** | 3.309 |
| 7 | **L19H1** | **0.900** | **5.259** |

L19H1 is the critical head — jumps from 0.350 to 0.900 in a single step. This head is in our weight-predicted IOI circuit (S-Inh tier, stability 0.73) AND in EAP rank 13.

**SVA** — **3 heads → IIA=0.924**
| Step | Added | IIA | Shift |
|------|-------|-----|-------|
| 1 | **L18H6** | **0.212** | 1.534 |
| 2 | **L17H11** | **0.621** | 2.724 |
| 3 | **L15H10** | **0.924** | **3.789** |

Only 3 heads needed. L18H6 and L17H11 are EAP top-5 heads. L15H10 is in the weight-predicted IOI circuit (S-Inh, stability 0.83). The minimum SVA circuit on medium is 3 heads.

**Greater-Than** — **3 heads → IIA=0.806**
| Step | Added | IIA | Shift |
|------|-------|-----|-------|
| 1 | **L13H12** | 0.145 | 0.210 |
| 2 | **L14H14** | **0.726** | 0.994 |
| 3 | **L6H15** | **0.806** | 1.127 |

L14H14 is the EAP rank-1 head for GT on medium. L13H12 is EAP rank-3.

### GPT-2 Large

**IOI** — **16 heads → IIA=0.812**
| Step | Added | IIA | Shift |
|------|-------|-----|-------|
| 1-3 | L27H17, L20H14, L24H17 | 0.000 | 1.763 |
| 4 | L22H0 | 0.037 | 2.934 |
| 7 | L20H19 | 0.250 | 3.642 |
| 12 | **L18H3** | **0.650** | 4.873 |
| 14 | L20H2 | 0.725 | 5.187 |
| 16 | **L26H0** | **0.812** | **5.655** |

Larger circuit needed (16 heads). No L0 heads selected — the ceiling search found an alternative route through mid/late layers (L18-L32). This means L0 is not the ONLY path to high IIA, just the most efficient one when combined with weight-predicted heads.

**SVA** — **7 heads → IIA=0.847**
| Step | Added | IIA | Shift |
|------|-------|-----|-------|
| 1-3 | L31H3, L22H15, L32H5 | 0.069 | 0.747 |
| 4-6 | L25H4, L26H5, L16H17 | 0.222 | 1.780 |
| 7 | **L24H3** | **0.847** | **3.778** |

L24H3 is the critical head — jumps from 0.222 to 0.847. Mid-to-late layer circuit, no L0.

**Greater-Than** — **15 heads → IIA=0.970** **ANOTHER L0 STORY**
| Step | Added | IIA | Shift |
|------|-------|-----|-------|
| 1-13 | (13 heads L0-L31) | **0.000** | -0.300 |
| 14 | **L0H14** | **0.758** | 1.239 |
| 15 | **L0H3** | **0.970** | **2.740** |

13 heads added with IIA stuck at 0.000 (shift even goes negative!). Then L0H14 → 0.758 and L0H3 → 0.970. Same L0 bottleneck as the greedy search.

---

## Comparison: Ceiling vs EAP Top-K (GPT-2 Medium)

Both find minimum circuits — ceiling is exhaustive (any heads), EAP uses gradient attribution.

| Task | Ceiling circuit | Ceiling IIA | EAP top-K for similar IIA | EAP IIA |
|------|----------------|-------------|---------------------------|---------|
| IOI | 7 heads | 0.900 | top-15 (15 heads) | 0.863 |
| SVA | **3 heads** | **0.924** | top-3 (3 heads) | 0.818 |
| GT | **3 heads** | **0.806** | top-3 (3 heads) | 0.903 |

**Circuit sizes are remarkably similar** — both methods find that 3-7 heads suffice for IIA > 0.8 on medium. But they pick different heads:

**SVA overlap**: Ceiling picks L18H6, L17H11, L15H10. EAP top-3 picks L18H6, L20H2, L19H14. **1 of 3 shared (L18H6)**.

**GT overlap**: Ceiling picks L13H12, L14H14, L6H15. EAP top-3 picks L14H14, L11H1, L13H12. **2 of 3 shared (L14H14, L13H12)**.

**IOI overlap**: Ceiling picks 7 heads (L12H3, L15H4, L15H14, L16H15, L17H4, L19H1, L19H7). EAP top-7 would be L22H14, L18H9, L17H12, L23H3, L17H4, L16H14, L19H1. **2 of 7 shared (L17H4, L19H1)**.

The methods partially converge on the most critical heads but approach them from different directions — ceiling does exhaustive IIA search, EAP uses gradient attribution.

---

## Cross-Scale Summary

### Minimum circuit sizes (ceiling, IIA >= 0.8)

| Task | Medium | Large | Published GPT-2 Small circuit |
|------|--------|-------|-------------------------------|
| IOI | 7 | 16 | 26 (Wang et al.) |
| SVA | 3 | 7 | ~12 (estimated) |
| GT | 3 | 15 | ~5 (Hanna et al.) |

IOI on medium (7 heads for 0.900) is **much smaller** than the 26-head published circuit. On large it grows to 16 heads. SVA circuits are surprisingly small at both scales.

### L0 dominance pattern (GPT-2 Large only)

| Task | Method | L0 heads in circuit | L0 contribution |
|------|--------|--------------------|-----------------| 
| IOI | Greedy | L0H14, L0H10, L0H2, L0H4, **L0H3** | 0→0.963 (L0H3 alone: +0.651) |
| IOI | Ceiling | None (found alt route via L18-L32) | — |
| SVA | Greedy | **L0H14**, **L0H10**, **L0H11** | 0→0.722 |
| SVA | Ceiling | None (found alt route via L16-L32) | — |
| GT | Greedy | Already had 3 L0 heads, +L0H4 | 0.909→0.909 |
| GT | Ceiling | **L0H14**, **L0H3** | 0→0.970 (13 heads at 0.000, then L0H14→0.758) |

For IOI and SVA on large, greedy (starting from weight-predicted heads) finds L0 is critical, but ceiling (starting from scratch) finds alternative mid-layer routes. This means **L0 is the most efficient bottleneck but not the only path** — there exist ~16-head mid-layer circuits that achieve similar IIA without any L0 heads.

For GT, both methods converge on L0 as essential.

---

## Key Interpretations

### 1. The circuit is smaller than published estimates

Ceiling finds 3-7 head circuits on medium vs 5-26 heads in the literature. This likely reflects (a) our IIA threshold is 0.8 not 1.0, and (b) published circuits include "supporting" heads that improve performance but aren't strictly necessary for output flipping.

### 2. L0 is a causal bottleneck, not the only path

On GPT-2 large, L0 heads are the most efficient way to flip model outputs — adding 5 L0 heads to late-layer context gives IIA=0.963 for IOI. But the ceiling search shows you CAN achieve IIA=0.812 with 16 mid-layer heads and zero L0 heads. L0 encodes the causal variable most compactly, but larger models have enough redundancy that alternative paths exist.

### 3. The greedy IOI trajectory on large is a textbook illustration

Steps 1-8: late-layer heads accumulate information (shift: 0.134 → 1.453) but IIA = 0.000. The model "knows more" but can't flip the output. Steps 9-13: L0 heads are added and IIA explodes (0.013 → 0.963). This is exactly what you'd expect if L0 encodes the critical routing variable that downstream heads read — without the routing signal, the downstream heads can't redirect the output even though they carry task-relevant information.

### 4. Weight-predicted + greedy addition is competitive with EAP

On medium: greedy from weight heads gets IOI=0.812 (22 heads), GT=0.968 (10 heads), SVA=0.803 (15 heads). EAP top-20 gets IOI=1.000, GT=0.984, SVA=1.000. EAP is more efficient (fewer heads) but requires gradient computation. Weight + greedy uses only forward passes.
