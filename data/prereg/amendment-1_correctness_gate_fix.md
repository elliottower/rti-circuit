# Amendment 1: Correctness gate fix

**Amends**: E5_unsupervised_epistatic_discovery.md (SHA `72bb47b`)

**Date**: 2026-08-01

**Reason**: The original correctness gate planted a synthetic cluster
via mean shift on 5 features. Mean shift translates points without
reducing within-group distances. HDBSCAN detects density (tight
clusters), not location (displaced means). The planted group was
never denser than background at any signal strength — the gate
tested nothing. Additionally, z-standardization over all 144 heads
erases the shift (the shift inflates the global std of signal
features, shrinking the offset back).

Measured on the original construction:

| Signal | within-group dist | background dist | ratio |
|--------|-------------------|-----------------|-------|
| +0.5σ  | 6.61              | 7.01            | 0.94  |
| +2σ    | 6.54              | 6.92            | 0.94  |
| +8σ    | 6.20              | 6.52            | 0.95  |

The ratio is flat at ~0.94 across all signal strengths. No density
cluster exists at any strength.


## Change 1: Correctness gate construction

**Old** (Section "Correctness gate"):
```
Create a 144 × 25 feature matrix where heads 0–7 share a common
signal (mean shift on 5 features) and heads 8–143 are iid standard
normal.
Test at three signal strengths: +0.5σ, +1σ, +2σ.
```

**New**:
```
Create a 144 × 25 feature matrix where heads 8–143 are iid N(0,1).
Heads 0–7 share a planted density cluster: all 25 features are
drawn from N(μ, σ_c²), where μ = 2 (a fixed offset) and σ_c
controls within-cluster spread.

Standardization is applied per-feature over the 136 background
heads only (not all 144), then the 8 signal heads are transformed
using the same mean and std. This prevents the signal from
inflating its own normalization.

Test at three within-cluster spreads: σ_c ∈ {0.5, 0.3, 0.15}.
```

**Rationale**: HDBSCAN finds regions of elevated density, not
displaced means. A density cluster requires reduced within-group
variance, not a location shift. All 25 features must be tight —
reducing spread on only a subset (e.g., 5 of 25) leaves the noise
features dominating pairwise distance, making the cluster
undetectable regardless of how tight the signal features are.
The background-only standardization prevents the signal from
erasing itself.


## Change 2: Gate pass criteria

**Old**:
```
At +2σ: HDBSCAN must recover a cluster containing ≥7 of heads 0–7
with ≤2 false positives.
At +1σ: report recovery. Partial recovery (≥5/8) is expected.
At +0.5σ: report recovery. Failure is acceptable.
```

**New**:
```
At σ_c = 0.15: HDBSCAN must recover a cluster containing ≥7 of
heads 0–7 with ≤2 false positives. Within/background distance
ratio ~0.15. If this fails, the pipeline cannot detect even
extreme density clusters and must not be run on real data.

At σ_c = 0.3: HDBSCAN must recover ≥5/8 signal heads.
Within/background ratio ~0.30.

At σ_c = 0.5: report recovery. Within/background ratio ~0.49.
Failure here calibrates the sensitivity floor — the real copier
tier may fall in this range.

Run each spread 10 times (different random backgrounds) and
report median recovery.
```


## Change 3: Real-data gate (new)

After the synthetic gate passes, add a real-data sanity check:

The 3 backbone heads (L0H8, L0H9, L0H11) occupy a single layer and
have similar positional-encoding function. Run HDBSCAN on the 25
features. If these 3 heads do NOT land in the same cluster (or all
in noise), the 25 features fail to encode even the most obvious
structural similarity. This distinguishes "copiers aren't similar
in feature space" from "these 25 features don't encode similarity
at all."

This is a descriptive check, not a gate — it does not block the
experiment. Report the result either way.


## Change 4: Feature count

The pre-registration lists 25 features. The pre-computed feature
JSON (`features_gpt2_small.json`) contains only 18 of the 25:
the 4 derived ratios (ov_rank_ratio, qk_rank_ratio,
ov_qk_rank_asymmetry, ov_qk_concentration_asymmetry) and
3 cross-matrix features (qk_ov_top_sv_align, qk_ov_top_right_align,
ov_unembed_norm) are absent.

The 4 derived features are computed from existing features (simple
arithmetic). The 3 cross-matrix features require model weight
matrices and are computed at runtime from GPT-2 Small's W_Q, W_K,
W_V, W_O, W_E via SVD.

All 25 features must be present in the clustering matrix. A run
using fewer than 25 features is a deviation from the pre-registration
and must be flagged.
