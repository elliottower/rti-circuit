# Pre-registration E4b: Heatmap-Derived Scalar Metrics for RTI Copier Detection

**Timestamp**: 2026-07-30T09:00:00Z

**Question**: Can 2--3 scalar metrics computed directly from the
W_E @ W_OV @ W_U logit matrix recover the 8 copier heads (and
potentially other RTI circuit members) from GPT-2 Small's 144 heads?


## Disclosure: observed vs unobserved data

### Observed (design parameters only)

- RTI circuit definition: 15 heads in 4 tiers (backbone 3, detector 1,
  copier 8, readout 3) from `src/weight_circuits/roles.py`
- That the circuit was discovered from visual inspection of W_OV
  heatmaps --- the copier tier's "negative diagonal" and "vertical
  banding" patterns
- The paper states "six heads" had visible negative diagonals, not all
  8 copiers
- Known non-circuit copy-suppression heads exist (L10H7, L11H10)
- The three metric definitions:
  1. Diagonal score: mean(diagonal) - mean(off-diagonal)
  2. Vertical band score: var(column means) / overall variance
  3. Effective rank: (sum sigma_i)^2 / sum(sigma_i^2)
- Base rate: 15 circuit heads / 144 total = 10.4%; copiers alone are
  8/144 = 5.6%

### Genuinely unobserved (confirmatory)

- All metric values for all 144 heads
- The distribution shape of each metric across heads
- Which specific heads have extreme values
- Whether the "six heads" with visible negative diagonals correspond
  to the 6 copiers with the most negative diagonal scores
- How many false positives any threshold produces
- Whether vertical band score or effective rank adds signal beyond
  diagonal score alone
- How the three non-copier tiers score on these metrics


## Motivation

E4 (the parent experiment) uses 108 engineered features and machine
learning classifiers. This companion experiment asks a simpler question:
how far do the rawest possible metrics get? The copier tier was
discovered by eyeballing heatmaps. Each metric here formalizes one
visual property of those heatmaps into a single number per head. If
these metrics suffice to flag copiers, it validates the claim that the
circuit's discovery was mechanistic (weight-space patterns) rather than
post-hoc (selected to fit a narrative).


## The three metrics

### Diagonal score

`diag_score(H) = mean(H[i,i]) - mean(H[i,j] for i != j)`

where H = W_E[:k] @ W_V @ W_O @ W_U[:, :k] for the head.

This directly encodes "negative diagonal": when diag_score is negative,
the head suppresses the logit for the token it attends to more than it
affects other tokens' logits. This is the copy-suppression signature.

### Vertical band score

`vband_score(H) = var(column_means(H)) / var(H)`

When this ratio is high, the matrix's variance is dominated by
column-level effects: certain output tokens are systematically boosted
or suppressed regardless of which input token is attended to. Visually,
this produces vertical colored stripes in the heatmap.

### Effective rank

`erank(H) = (sum sigma_i)^2 / sum(sigma_i^2)`

where sigma_i are the singular values of H. A matrix dominated by one
mode has erank near 1; a full-rank matrix with equal singular values
has erank equal to min(rows, cols). Low effective rank means the head's
logit-level behavior is simple (few modes of input-to-output mapping).


## Prediction 1: Diagonal score distribution

### Prediction

The 8 copier heads cluster at the negative extreme of the diagonal
score distribution. Of the 8, **6--8 will rank in the bottom 15 heads**
(most negative diagonal scores), and **all 8 will rank in the bottom
25**.

The population of 144 heads will have diagonal scores roughly centered
near zero, with a slight positive skew (more heads boost than suppress
the attended token's logit, because next-token copying is more common
than next-token suppression).

Non-copier circuit heads will scatter across the distribution:
- **Backbone** (L0H8, L0H9, L0H11): near the population median. Their
  OV matrices encode positional information, not token-level copy or
  suppression. No reason for their diagonal entries to be systematically
  different from off-diagonal entries.
- **Detector** (L4H11): near the population median. L4H11 detects
  repeated tokens via QK attention patterns. Its OV matrix may write a
  generic "repeated token detected" signal rather than token-specific
  suppression. Its diagonal score is determined by what it writes, not
  what it attends to.
- **Readout** (L10H11, L11H9, L11H11): mildly negative to neutral.
  These heads aggregate the suppression signal from upstream copiers.
  If readout heads amplify suppression, they might have weakly negative
  diagonal scores, but the signal will be diffuse (mediated through
  the residual stream rather than directly through their own OV matrix).
  At most 1 readout head will rank in the bottom 15.

### Rationale

The paper identifies "six heads" with visible negative diagonals.
These are almost certainly 6 of the 8 copiers --- the copier tier was
the tier discovered from heatmap inspection. The remaining 2 copiers
either have weaker negative diagonals (detectable by the metric but
not visually striking) or achieve suppression through a different
mechanism (context-dependent rather than weight-encoded).

The diagonal score metric is more sensitive than visual inspection: a
head with a subtly negative diagonal that does not produce a visually
obvious blue stripe might still have a negative diagonal score. So the
metric may catch all 8 copiers even though only 6 had visible patterns.

False positives in the bottom 15: the known copy-suppression heads
L10H7 and L11H10 will likely rank near the copiers. Beyond those, some
induction heads may have weakly negative diagonals (they copy forward
rather than suppress, but partial OV overlap with the suppression
direction could produce a negative diagonal score). I predict 4--9
non-copier heads in the bottom 15.

### Falsification criteria

- **Fewer than 5 copiers in the bottom 15**: would mean the copier
  OV suppression signature is not as uniform as expected, and the
  "six heads" observation from the paper does not translate to a
  quantitative metric.

- **More than 3 non-copier circuit heads in the bottom 15**: would
  mean the negative diagonal is a circuit-wide property, not
  copier-specific. This would challenge the tier taxonomy.

- **Population distribution is bimodal**: would mean there is a
  natural category boundary between "suppression heads" and "other
  heads" in GPT-2, which would be an interesting structural finding
  about the model.


## Prediction 2: Vertical band score distribution

### Prediction

Copier heads will have **moderate vertical band scores**, ranking in
the middle third of the distribution (ranks 40--100 out of 144).

The heads with the highest vertical band scores will be **early-layer
heads** (layers 0--2) and **late-layer heads** (layers 10--11) that
implement frequency-dependent output biases --- heads whose OV matrices
systematically boost or suppress high-frequency tokens regardless of
what is attended to.

### Rationale

Vertical banding reflects column-level consistency: the output
distribution does not depend on the input token. This is a low-rank
property (rank-1 in the extreme case). The copier signature is the
opposite: which output token gets suppressed depends specifically on
which input token is attended to (token j attended to implies token j
suppressed). This diagonal structure requires higher rank and distributes
variance across the diagonal rather than concentrating it in columns.

A copier head could have both a negative diagonal and some vertical
banding (certain tokens might be systematically easier to suppress), but
the vertical banding would be secondary to the diagonal pattern. So
copier heads will have nonzero but unremarkable vertical band scores.

The highest vertical band scores will belong to heads that act as
unigram output biases --- they boost common tokens or suppress rare
ones regardless of context. Early-layer heads often implement such
biases (they process the raw embedding before contextual information
is available). Late-layer heads may implement global output calibration.

### Per-tier predictions

| Tier | Predicted vband rank range | Reasoning |
|---|---|---|
| Copier (8) | 40--100 | Diagonal-dominant structure; moderate banding |
| Backbone (3) | 1--50 (high banding) | Positional heads may have strong column structure: output biases conditioned on position, which for the top-k frequent tokens looks like vertical banding |
| Detector (1) | 50--120 | QK-based head; OV matrix may be unremarkable |
| Readout (3) | 20--80 | Late-layer; may have output calibration biases |

### Falsification criteria

- **Copier heads rank in the top 20 on vertical band score**: would
  mean the copier OV matrix has a stronger column-level structure than
  expected, suggesting the suppression mechanism is partially
  frequency-dependent (suppress high-frequency tokens more).

- **Copier heads rank in the bottom 20**: would mean copier OV
  matrices are unusually row-diverse, with no consistent column-level
  effects. This would suggest each input token produces a unique
  output distribution.

- **Vertical band score and diagonal score are strongly correlated
  (|r| > 0.5) across all 144 heads**: would mean the two metrics are
  not independent, reducing the value of combining them.


## Prediction 3: Effective rank

### Prediction

Copier heads will have **moderate-to-high effective rank**, ranking in
the top half of the distribution (ranks 1--72). They will not be at the
extreme top or bottom.

Heads with the lowest effective rank (simplest structure) will be heads
that implement near-rank-1 operations: pure frequency biases or
single-direction copy operations. Copier heads need to represent
"suppress token j when attending to token j" across many tokens, which
requires multiple modes of variation and therefore higher rank.

### Rationale

The diagonal of a k x k matrix contributes to many singular vectors
(the identity matrix has rank k with all singular values equal to 1).
A head whose logit matrix is dominated by a negative diagonal will have
singular values that are more evenly distributed than a head with a
single dominant mode, yielding higher effective rank.

However, the copier heads' logit matrices are not pure diagonals ---
they also have off-diagonal structure (some tokens may be boosted as
alternatives to the suppressed token). The effective rank measures
the overall complexity, which is a mix of the diagonal structure and
any off-diagonal patterns. I expect effective ranks in the range of
15--35 for copier heads (out of a maximum of k, where k = 50 or 100).

### Falsification criteria

- **Copier heads have the lowest effective ranks in the model**: would
  mean the suppression mechanism is actually low-rank, perhaps
  concentrated on a few principal components. This would suggest the
  suppression operates on a small subspace of token space.

- **Effective rank is essentially uniform across all heads**: would
  mean the metric provides no discriminative signal and should be
  dropped.


## Prediction 4: Simple threshold on diagonal score alone

### Prediction

Setting a threshold to flag the 15 heads with the most negative
diagonal scores recovers **6--8 of the 8 copiers** as true positives,
with **7--9 false positives** (non-circuit heads in the flagged set).

If the threshold is relaxed to the bottom 20 heads, recall rises to
**7--8 copiers** but false positives increase to **12--13**.

If tightened to the bottom 10, recall drops to **5--7 copiers** with
**3--5 false positives**.

### Expected false positive identities

The non-circuit heads most likely to have extreme negative diagonal
scores are:

1. **L10H7** (copy suppression head from McDougall et al.) --- almost
   certainly in the bottom 10. This head is a known copy suppressor
   that is not part of the RTI circuit definition.

2. **L11H10** (another known copy-suppression head) --- likely in the
   bottom 15.

3. **1--3 mid-layer heads** (layers 3--6) that partially implement
   copy suppression for non-RTI tasks (induction, IOI name suppression).

4. **1--2 late-layer heads** (layers 9--11) with OV matrices that
   suppress common tokens as part of output calibration.

### Rationale

The diagonal score is the most direct formalization of the visual
pattern that was used to discover the copier tier. If 6 of 8 copiers
had visible negative diagonals, the metric (which is more sensitive
than visual inspection) should catch at least 6 and possibly all 8.

The false positive count depends on how unique copy suppression is to
the RTI circuit. GPT-2 Small has at least 2 well-documented
non-RTI copy-suppression heads. Beyond those, the false positive count
depends on how many other heads have incidentally negative diagonals
for reasons unrelated to copy suppression (random fluctuations in OV
weights, heads that suppress specific tokens for task-specific reasons).

With 144 heads total and copier heads expected to be in roughly the
bottom 5--10% of the diagonal score distribution, a threshold at the
bottom 15 (10.4%) will include some non-circuit heads by construction
unless copier heads are perfectly separated.

### Falsification criteria

- **Fewer than 5 copiers in the bottom 15**: the diagonal score metric
  fails to capture what was visible in the heatmaps. Either the metric
  definition is wrong (mean diagonal vs mean off-diagonal is not what
  "negative diagonal" means visually) or the copier tier is less
  uniform than the paper implies.

- **More than 12 false positives in the bottom 15**: impossible (only
  15 heads total), but if at threshold 20 there are more than 15 false
  positives (meaning fewer than 5 copiers in the bottom 20), the
  negative diagonal is too common to be discriminative.

- **Zero false positives in the bottom 10**: would mean copy
  suppression is so specific to the RTI circuit that no other head in
  GPT-2 Small implements it. This would be surprising given the known
  existence of L10H7 and L11H10.


## Prediction 5: Combined threshold (diagonal + vertical band)

### Prediction

Adding a vertical band score filter to the diagonal score threshold
produces a **modest improvement in precision** (1--3 fewer false
positives) with **no loss in copier recall**.

Specifically: flagging heads that are both in the bottom 15 on diagonal
score AND in the middle 60% on vertical band score (ranks 30--114)
reduces false positives by 1--3 compared to diagonal score alone, while
keeping 6--8 copiers.

### Rationale

The improvement is modest because:

1. **Low correlation expected**: diagonal score and vertical band score
   measure different properties (row-column interaction vs column
   consistency), so their combination is a genuine 2D filter. But the
   false positives from diagonal score alone (copy-suppression heads)
   likely have similar vertical band profiles to the copier heads ---
   they implement structurally similar operations. The 2D filter removes
   only false positives whose vertical band profiles differ from
   copiers.

2. **The vertical band filter is weak**: copier heads are predicted to
   have middling vertical band scores (prediction 2), which means the
   "middle 60%" filter excludes only heads with extreme vertical band
   scores. Among the diagonal-score false positives, few will have
   extreme vertical band scores.

3. **Copy-suppression heads look alike**: the core problem is that
   L10H7 and L11H10 implement the same operation as the copier heads
   (suppress repeated token logits). No weight-level metric that
   measures the *what* of the computation can distinguish them from
   copier heads --- only metrics that measure the *when* (which
   contexts activate the head) or the *why* (upstream connectivity)
   can separate functionally identical computations that serve
   different circuits.

### Falsification criteria

- **Combined threshold removes > 5 false positives with no recall
  loss**: would mean vertical band score captures a real structural
  difference between RTI copiers and other copy-suppression heads.
  This would be a surprising and useful finding.

- **Combined threshold removes copier heads**: would mean the vertical
  band filter is set wrong, or copier heads have more extreme vertical
  banding than predicted.


## Prediction 6: Effective rank as a discriminator

### Prediction

Effective rank alone is a **weak discriminator** of copier heads,
achieving separation no better than chance-plus-epsilon. The copier
heads' effective ranks will overlap heavily with the population.

However, effective rank may weakly correlate with the other two metrics
across the full population (correlation with vertical band score:
r = -0.2 to -0.5, because high-banding heads have low rank). It will
not improve the combined threshold from prediction 5 by more than 1
false positive.

### Rationale

Effective rank measures global matrix complexity, which is not specific
to any particular computational pattern. Many different heads can have
the same effective rank for different reasons. Copy-suppression heads
have moderate effective rank because the diagonal pattern is moderately
complex --- but so do heads that implement multi-token copying,
attention-head composition, or other mid-complexity operations.

The negative correlation with vertical band score follows from the
definition: a matrix with high column-level variance has a few dominant
singular values (corresponding to the column-mean pattern), yielding
low effective rank.

### Falsification criteria

- **Effective rank separates copiers from non-copiers at AUROC > 0.65**:
  would mean matrix complexity is unexpectedly informative about copy
  suppression, suggesting the copier OV matrices have a distinctive
  spectral profile.


## Summary of predictions

| Prediction | Metric(s) | Predicted outcome | Key number |
|---|---|---|---|
| 1. Diagonal distribution | diag_score | 6--8 copiers in bottom 15 | 6--8 / 8 recall |
| 2. Vertical band distribution | vband_score | Copiers rank mid-pack (40--100) | Not discriminative alone |
| 3. Effective rank | erank | Copiers rank mid-to-high; weak discriminator | AUROC < 0.65 |
| 4. Diagonal threshold | diag_score, top-15 | 6--8 copiers, 7--9 FPs | Precision 40--55% for copiers |
| 5. Combined threshold | diag + vband | 1--3 fewer FPs than diag alone | Precision 45--65% |
| 6. Effective rank addition | diag + vband + erank | At most 1 fewer FP | Negligible improvement |


## Success criteria

### "Heatmap metrics work for copier detection"

All of the following:
1. At least 6 of 8 copiers rank in the bottom 15 on diagonal score
2. At least 5 of 8 copiers rank in the bottom 10 on diagonal score
3. Fewer than 10 false positives at a threshold that catches 6 copiers
4. L10H7 and L11H10 appear as false positives (confirming they are
   structurally similar to copier heads, as the literature suggests)

Meeting these criteria validates the paper's claim that the copier tier
was discovered from weight-space visual inspection: the visual pattern
translates to a quantitative metric that ranks copier heads at one
extreme of the distribution.

### "Heatmap metrics are insufficient"

Any of the following:
1. Fewer than 4 copiers in the bottom 15 on diagonal score
2. More than 15 non-circuit heads have more negative diagonal scores
   than the median copier head
3. The combined threshold (prediction 5) produces more than 12 false
   positives at recall >= 6 copiers

Meeting any of these means the negative-diagonal pattern, while
visually suggestive, does not translate to a reliable quantitative
signal for automated detection.

### Expected intermediate outcome

The most likely result: diagonal score alone flags 6--7 copiers with
5--8 false positives. The combined threshold trims 1--2 false positives.
Effective rank adds nothing. The conclusion will be that a single
scalar metric (diagonal score) captures most of the copier-tier signal
with moderate precision, but cannot distinguish RTI copiers from other
copy-suppression heads in the model. Separating circuit-specific from
circuit-general copy suppression requires activation-level or
connectivity-level analysis.


## Key uncertainty: what "negative diagonal" means quantitatively

The central risk is a mismatch between the visual pattern and the
scalar metric. A heatmap "looks like it has a negative diagonal" when
the diagonal entries are visually darker (more negative) than their
immediate neighbors. The diagonal score metric compares the diagonal
mean to the global off-diagonal mean. These are not the same thing: a
head could have a visually striking negative diagonal (entries below
their row or column neighbors) while having a diagonal score near zero
if the off-diagonal entries are also generally negative.

If this mismatch is large, the diagonal score will fail to rank copier
heads at the extreme, and a local-contrast metric (diagonal entry minus
same-row mean, averaged across the diagonal) would be more appropriate.
I predict this mismatch is small --- the copier heads' heatmaps show
clear blue diagonal stripes against a near-zero or warm-colored
background, suggesting the diagonal entries are negative in absolute
terms, not just relative to neighbors. But this is the prediction I am
least confident about.
