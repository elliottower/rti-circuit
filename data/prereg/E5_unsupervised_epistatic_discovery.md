# Pre-registration E5: Unsupervised Epistatic Circuit Discovery

**Timestamp**: 2026-08-01T20:00:00Z

**Question**: Can unsupervised clustering of weight features, filtered by
K-composition, recover the RTI copier tier without being told it exists?


## Disclosure: observed vs unobserved data

### Observed (design parameters only)

- RTI circuit definition: 15 heads in 4 tiers (backbone 3, detector 1,
  copier 8, readout 3) from `src/weight_circuits/roles.py`
- Feature set: 33 deterministic features + ~90 direction-alignment features
  computed from weight matrices (`src/weight_circuits/features.py`).
  This experiment clusters on 25 non-composition deterministic features
  (excluding the 8 composition features to keep Step 3 independent).
- That K-composition scores trace inter-head information flow
- That the copier tier was originally discovered from visual similarity
  of W_OV heatmaps (positive diagonals)
- That existing activation-based methods find 0/8 copier heads
- That 63% [49%, 76%] of the full 15-head circuit's causal effect is
  from interactions (LOO over all 15 heads, not copier-only)
- E4 classifier predictions: the supervised classifier is predicted to
  achieve AUROC 0.80–0.90 (from E4 pre-registration). This experiment
  asks whether *unsupervised* clustering can recover similar structure
  without labels. E4 results, if already computed, were not consulted
  for this design.

### Genuinely unobserved (confirmatory)

- The cluster assignments produced by any clustering algorithm on
  the weight-feature matrix
- Whether any cluster overlaps substantially with the copier tier
- K-composition scores between discovered clusters
- Epistasis scores for any discovered cluster
- How many clusters pass the K-composition filter
- Whether the method finds groups not in the known circuit


## Method

### Step 1: Feature extraction (CPU, deterministic)

Compute the weight-feature matrix F ∈ R^{144 × 25} for all heads
in GPT-2 Small. Use the deterministic subset of features from
`src/weight_circuits/features.py`, excluding:
- All direction-alignment features (ICA/cluster-based) — depend on
  random seeds.
- All 8 composition features (k_comp_recv_max, k_comp_recv_mean,
  k_comp_sent_max, k_comp_sent_mean, v_comp_recv_max,
  v_comp_recv_mean, v_comp_sent_max, v_comp_sent_mean) — these
  measure inter-head communication, the same quantity the
  K-composition filter (Step 3) tests. Including them in clustering
  would make the filter circular.

The 25 clustering features are:

**SVD structural (14):**
ov_norm, ov_concentration, ov_sv_gap, ov_effective_rank, ov_top2_ratio,
qk_norm, qk_concentration, qk_sv_gap, qk_effective_rank, qk_top2_ratio,
ov_rank_ratio, qk_rank_ratio, ov_qk_rank_asymmetry,
ov_qk_concentration_asymmetry

**Cross-matrix (3):**
qk_ov_top_sv_align, qk_ov_top_right_align, ov_unembed_norm

**Token-interaction (8):**
qk_same_diff_ratio, qk_same_diff_gap, qk_same_diff_ratio_w, qk_sens_tok,
ov_tok_diag_mean, ov_tok_offdiag_mean, ov_tok_copy_ratio, ov_tok_logit_min

Standardize each feature to zero mean and unit variance.

### Step 2: Clustering (CPU, no labels)

Run HDBSCAN on the standardized feature matrix with:
- distance metric: Euclidean
- min_cluster_size: 3
- min_samples: 3

**Parameter justification**: min_cluster_size=3 is the smallest
value that yields non-trivial clusters (a pair is not a group).
The copier tier has 8 members, so 3 is safely below the expected
cluster size. min_samples=3 matches min_cluster_size (HDBSCAN
default behavior).

**Robustness check**: After the primary run at min_cluster_size=3,
re-run at min_cluster_size ∈ {4, 5}. Report how cluster assignments
change. H1 is evaluated on the primary run (min_cluster_size=3)
only — the robustness runs are descriptive.

HDBSCAN is used instead of DBSCAN to avoid the eps hyperparameter,
which would be a researcher degree of freedom. HDBSCAN determines
density thresholds automatically from the data.

Report: number of clusters found, sizes, which heads are in each,
number of noise points.

**PCA fallback (if HDBSCAN assigns >90% of heads to noise):**
25 features for 144 heads may be too high-dimensional for density
estimation. If >130 heads are classified as noise, apply PCA
retaining 95% of variance before re-running HDBSCAN with the same
parameters. Report both the raw and PCA-reduced results. H1 is
evaluated on whichever run (raw or PCA-reduced) produces clusters —
if both produce clusters, use the raw run.

**Agglomerative follow-up (exploratory, if HDBSCAN produces fewer
than 3 clusters even after PCA fallback):** Run agglomerative
clustering with Ward linkage and cut at the height that maximizes
the silhouette score. One cut, one result — no multiple-cut
exploration. This is a *different clustering method* and its results
are reported as exploratory. H1 is evaluated as "failed for
HDBSCAN" regardless of the agglomerative outcome.

### Step 3: K-composition filtering (CPU)

For each cluster C of size |C| ≥ 3:
1. **Outgoing K-comp**: mean of ||W_O^(i) W_K^(j)||_F for all ordered
   pairs (i,j) where i ∈ C, j ∉ C, and layer(i) < layer(j).
   This measures how strongly cluster members communicate *to*
   downstream non-members.
2. **Background K-comp**: mean of ||W_O^(i) W_K^(j)||_F for pairs
   where both i,j ∉ C, layer(i) < layer(j). This is the baseline
   communication rate between non-cluster heads.
3. Ratio: outgoing / background.

**Rationale for this definition**: A copier-only cluster will have
few internal cross-layer pairs (copier-to-copier communication is
not the functional pathway). The functional signal is
copier→readout, which is *outgoing* from the cluster. Measuring
outgoing vs background communication captures whether cluster
members are preferential senders to specific downstream targets,
even if those targets are outside the cluster.

Retain clusters where ratio > 1.5 (outgoing K-comp at least 50%
higher than background). The 1.5 threshold is confirmatory.

**Robustness report (non-confirmatory)**: Also report which clusters
pass at ratio > 1.2. If the copier cluster fails at 1.5 but passes
at 1.2, this is descriptive information about how close the signal
is to the threshold — it does not count as passing H2.

### Step 4: Epistasis testing (requires model, GPU/CPU)

For each cluster that passes the K-composition filter:
1. Ablate all heads in the cluster simultaneously on the RTI task
   (302 prompts). Record mean logit difference change (cluster effect).
2. For each head in the cluster, ablate that head alone. Record
   mean logit difference change (LOO effect).
3. Epistasis score = 1 - (sum of LOO effects / cluster effect)
4. Bootstrap 95% CI on the epistasis score using ratio-of-means
   (same method as the paper's main analysis).

Ablation method: zero-ablation of head output (same as the paper's
main analysis).

**Random-head baseline**: Sample 50 random groups of |C| heads
(where |C| is the size of the discovered cluster), compute the
epistasis score for each. H3 passes only if the discovered cluster's
epistasis score exceeds *both* the 0.25 absolute threshold *and*
the 95th percentile of the random baseline distribution. This
controls for the possibility that arbitrary groups of |C| heads
show epistasis by chance due to correlated head effects.

**Note on comparability**: The paper's 63% interaction figure is
computed over all 15 circuit heads (LOO each of 15, vs ablating all
15). The epistasis score computed here is *cluster-specific* — LOO
over just the cluster members, vs ablating the whole cluster. These
measure different things. A copier-only cluster with ~8 heads will
yield a copier-specific epistasis score that has no a priori reason
to match 63%. The H3 threshold is therefore set independently.


## Predictions

### H1: Copier recovery (primary)

At least one cluster contains ≥5 of the 8 copier heads
(recall ≥ 62.5%) with cluster size ≤12 (i.e., ≤4 non-copier
heads if all 8 copiers are present, or proportionally fewer for
partial copier recovery; equivalently, precision ≥ 42%).

**Maximum cluster size**: 15 heads. Any cluster larger than 15 is
excluded from H1 evaluation — a cluster containing >10% of all
heads is too diffuse to constitute a discovered circuit.

**Union recovery (descriptive, non-confirmatory)**: If no single
cluster passes H1, also report whether the union of the top-2
clusters by copier overlap achieves ≥6/8 copier recall with
combined size ≤18. This is descriptive only — it does not count
toward H1 or the confirmatory success criteria. It indicates the
method found finer structure (e.g., low-layer vs high-layer
copiers) and motivates follow-up work on hierarchical merging.

**Rationale**: The 8 copier heads share positive-diagonal W_OV
matrices, which produces similar SVD spectra and token-interaction
features. They span layers 4–9 so their feature profiles are not
layer-confounded. HDBSCAN should detect this density cluster in
feature space. The 62.5% recall threshold allows 3 copier heads to
scatter into noise or other clusters (plausible for L4H0, which
sits in the detector's layer and may have a mixed feature profile).

**Falsification**: If no single cluster achieves ≥5/8 copier recall,
H1 fails. The copier tier's weight-feature similarity was
overestimated, or density-based clustering is wrong for this feature
geometry. Union recovery is reported descriptively but does not
rescue H1.

### H2: K-composition filter retains copier cluster

The copier-containing cluster (from H1) passes the K-composition
filter at outgoing/background ratio > 1.5.

**Rationale**: Copier heads communicate downstream to readout heads.
Outgoing K-comp (copier→all downstream non-cluster heads) should
exceed background K-comp because the copier→readout pathway carries
47% of the circuit's effect. The 1.5 threshold is conservative
given that K-composition scores of circuit edges are typically 2–5×
background in the paper's manual analysis.

**Known risk**: If the copier cluster is small (5–6 heads) or the
readout heads have high K-comp with many senders (not just copiers),
the outgoing/background ratio may be close to 1.0. H2 failure with
H1 success would indicate that K-comp filtering is too blunt for
this circuit topology — a meaningful negative result about the
filtering step, not about the clustering step.

### H3: Cluster shows epistasis

The copier-containing cluster's epistasis score exceeds 0.25
*and* exceeds the 95th percentile of the random-head baseline
(50 random groups of matched size; see Step 4).

**Rationale**: This is a *cluster-specific* epistasis score, not
directly comparable to the paper's 63% (which is over all 15 heads).
A cluster of ~8 copier heads should show epistasis because the
copier tier's function depends on distributed redundancy — each
head contributes <5% individually while the ensemble effect is
substantial. The 0.25 threshold means at least 25% of the cluster's
joint effect comes from interactions. This is deliberately lower
than the 63% figure because: (a) the cluster may not contain all
copier heads, reducing interaction opportunities; (b) the cluster
may include non-copier heads whose marginal effects inflate the
LOO sum; (c) the 63% includes interactions across all four tiers,
not just within the copier tier.

**Falsification**: Epistasis < 0.10 would mean the discovered
cluster has marginal-dominated effects — each head contributes
independently — and the cluster is not an epistatic group.

### H4: Null control

Permute each feature column independently (shuffling feature values
across heads within each feature, preserving marginal distributions
but destroying per-head structure) and re-run Steps 2–3. In ≥95%
of 100 permutations, no cluster that passes the K-comp filter should
overlap with the copier tier at ≥5/8 recall.

**Rationale**: Column-wise permutation destroys the correlation
structure between features within each head while preserving each
feature's distribution. If HDBSCAN still finds clusters that
(a) pass K-comp filtering and (b) overlap with the copier tier,
the method has no specificity — it would find "copier clusters"
from random feature assignments.

**Note**: Epistasis testing (Step 4) is not run on permuted data.
The null tests whether the *discovery pipeline* (clustering +
filtering) produces copier-overlapping clusters by chance, not
whether random head groups show epistasis (which is a different,
more expensive null).

### H5: Novel group discovery (exploratory)

At least one non-copier cluster passes the K-composition filter with
≥3 heads and epistasis score > 0.20. This would be a novel epistatic
group not in the known RTI circuit.

**Status**: Exploratory — not confirmatory. A null result does not
falsify the method (the RTI circuit may be the only epistatic group
in GPT-2 Small for this feature set). A positive result would be
the most exciting outcome.


## Correctness gate

Before running Steps 2–4 on real data, verify on synthetic data:
- Create a 144 × 25 feature matrix where heads 0–7 share a common
  signal (mean shift on 5 features) and heads 8–143 are iid standard
  normal.
- Test at three signal strengths: +0.5σ, +1σ, +2σ.
- At +2σ: HDBSCAN must recover a cluster containing ≥7 of heads 0–7
  with ≤2 false positives. If this fails, the HDBSCAN parameters are
  miscalibrated and must be adjusted before touching real data.
- At +1σ: report recovery. Partial recovery (≥5/8) is expected.
- At +0.5σ: report recovery. Failure is acceptable — this calibrates
  the method's sensitivity floor.
- Run each signal strength 10 times (different random backgrounds)
  and report median recovery.


## Analysis plan

### Primary analysis

1. Run Steps 1–3 (feature extraction, clustering, K-comp filtering)
2. Report all clusters: size, head membership, K-comp ratio
3. For each cluster passing K-comp filter: compute overlap with
   known RTI circuit tiers (Jaccard, recall, precision)
4. If no single cluster passes H1, report union recovery
   (descriptive, non-confirmatory)
5. Evaluate H1 and H2 before running Step 4
6. Run Step 4 (epistasis testing) on filtered clusters
7. Evaluate H3
8. Run null control (H4)
9. Report any novel groups (H5)

### Multiple comparisons

If N clusters pass the K-comp filter, the epistasis threshold in H3
applies to each independently — no Bonferroni correction, because
each cluster is a separate hypothesis (different heads, different
function). The null control (H4) handles false discovery at the
method level.

### Reporting

All results reported regardless of outcome. If the method fails to
recover the copier tier, that is the result — it means unsupervised
weight-feature clustering does not capture the structure that visual
inspection of heatmaps revealed, and the gap between human pattern
recognition and automated clustering on weight matrices is real.


## Success criteria

### "Method works" (confirmatory)

H1 AND H2 AND H3 all pass: the unsupervised pipeline recovers a
copier-like cluster, the cluster shows elevated outgoing K-comp,
and ablating it reveals epistasis above 0.25.

This would justify adding an "Automated Recovery" section to the
paper: the circuit discovered by visual inspection can be recovered
blindly by a simple unsupervised pipeline.

### "Method partially works"

H1 passes but H2 or H3 fails: the clustering finds the copier heads
but they don't show elevated K-comp or the epistasis is below
threshold. This is still informative — it means the copier tier is
detectable from weight features but the K-composition or epistasis
criteria need refinement.

Union recovery descriptively succeeds but H1 fails: the copier
tier splits across clusters, indicating the method finds finer
structure than the tier labels. This motivates follow-up work on
hierarchical merging, reported as exploratory.

### "Method fails"

H1 fails: no single cluster overlaps substantially with the copier
tier. This means the feature space does not preserve the copier
similarity that is visible in raw heatmaps, and automated clustering
cannot substitute for human visual inspection of weight matrices.
This is a meaningful negative result. (Union recovery and
agglomerative follow-up are reported descriptively alongside.)
