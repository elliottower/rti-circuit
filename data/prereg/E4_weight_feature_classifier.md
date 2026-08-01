# Pre-registration E4: Weight-Feature Classification of RTI Circuit Heads

**Timestamp**: 2026-07-30T08:00:00Z

**Question**: Can 108 weight-space features alone recover the 15-head RTI
circuit in GPT-2 Small?


## Disclosure: observed vs unobserved data

### Observed (design parameters only)

- RTI circuit definition: 15 heads in 4 tiers (backbone 3, detector 1,
  copier 8, readout 3) from `src/weight_circuits/roles.py`
- Feature set: 108 features computed from weight matrices (SVD statistics,
  token interaction, direction alignment, composition norms) from
  `src/weight_circuits/features.py`
- Classifier code: bootstrap greedy selection with AUROC objective from
  `src/weight_circuits/classify.py`
- That the RTI circuit was originally discovered from visual inspection
  of W_OV heatmaps (the copier tier's copy-suppression signature)
- Base rate: 15 circuit heads out of 144 total = 10.4%

### Genuinely unobserved (confirmatory)

- All feature values for all 144 heads
- All AUROC values (univariate or combined)
- All logistic regression coefficients and LOO-CV predictions
- All bootstrap stability scores
- Which specific features rank highest
- How many false positives any method produces
- Whether non-copier tiers are separable at all


## The classification problem

The RTI circuit has a fundamental asymmetry that drives every prediction
below: its four tiers perform different computations and should leave
different weight signatures.

- **Copier** (8 heads, layers 4--9): suppress repeated-token logits.
  The W_E @ W_OV @ W_U matrix should have a negative diagonal ---
  this is literally how the circuit was discovered (visual heatmaps).
  Features `ov_tok_diag_mean`, `ov_tok_copy_ratio`, and `ov_tok_logit_min`
  measure this directly. This tier should be the easiest to classify.

- **Backbone** (3 heads, layer 0): encode positional/token identity for
  downstream consumption. But GPT-2 has 12 heads in layer 0, many of
  which serve similar early-layer roles (previous-token, positional).
  The 3 backbone heads must be separated from 9 other L0 heads that
  share similar SVD profiles and token-interaction features.

- **Detector** (1 head, L4H11): detects repeated tokens via QK matching.
  L4H11 is also part of the induction and IOI circuits (it appears as
  PTH in both). Its QK signature (high same-diff ratio) is shared with
  other previous-token and induction heads. Separating 1 head from 143
  others with shared QK properties is a needle-in-haystack problem.

- **Readout** (3 heads, layers 10--11): read out the suppression signal.
  Late-layer heads have high composition norms by construction (many
  upstream senders). The 3 readout heads compete with 21 other heads
  in layers 10--11 that also have large downstream connectivity.

**Core difficulty**: a single classifier must find something in common
across four tiers that perform four different computations. The features
that identify copiers (OV diagonal) are orthogonal to the features
that identify backbone (QK patterns) or readout (composition norms).
A linear combination can in principle weight different features for
different tiers, but 15 positives out of 144 samples means the model
has very little positive-class training signal.


## Prediction 1: Per-feature AUROC (univariate)

### Prediction

The best single feature achieves AUROC 0.72--0.82, and it is an OV
token-interaction feature (`ov_tok_diag_mean`, `ov_tok_copy_ratio`,
or `ov_tok_logit_min`).

The median feature achieves AUROC 0.45--0.55 (near chance). Most of
the 90 direction-alignment features will cluster near 0.50 because
ICA/cluster directions are generic embedding-space structure and have
no a priori relationship to the RTI circuit.

### Rationale

The AUROC upper bound for a copier-only feature: if all 8 copiers rank
at one extreme and the other 7 circuit heads (backbone, detector,
readout) scatter uniformly among the 129 negatives, the expected AUROC
is approximately (8 x 129 + 7 x 64.5) / (15 x 129) = 0.77. The actual
value depends on (a) whether any non-copier circuit heads also show the
OV suppression signature (readout heads might, pushing AUROC up toward
0.82) and (b) whether any non-circuit heads also suppress (known copy
suppression heads such as L10H7 and L11H10 from other circuits could
bleed into the copier distribution, pushing AUROC down toward 0.72).

No single feature can capture all four tiers because no single weight
property is shared across early-layer positional encoding, mid-layer
QK matching, mid-layer OV suppression, and late-layer readout.

### Feature-family predictions

| Feature family | Count | Predicted AUROC range of best in family | Reasoning |
|---|---|---|---|
| OV token interaction | 4 | 0.72--0.82 | Directly measures the copier signature |
| QK token interaction | 4 | 0.55--0.68 | May capture backbone/detector but not copier/readout |
| SVD structural | 14 | 0.50--0.65 | Generic spectral properties; no tier-specific prediction |
| Cross-matrix | 3 | 0.50--0.62 | OV-unembed norm may partially correlate with copier |
| Composition | 8 | 0.52--0.65 | Aggregate max/mean dilutes circuit-specific connectivity |
| Direction alignment | 90 | 0.50--0.62 | Data-dependent; some may correlate by chance |

### Falsification criteria

- **Best feature AUROC > 0.85**: would mean either (a) one feature
  captures multiple tiers simultaneously, which is surprising given
  the different computations, or (b) a direction-alignment feature
  happened to find a privileged embedding direction that all circuit
  heads share. Either finding would be theoretically interesting.

- **Best feature AUROC < 0.65**: would mean even the copier-tier OV
  signature is not strong enough for univariate classification,
  indicating that copy-suppression patterns are common across many
  GPT-2 heads and not circuit-specific.

- **Best feature is NOT an OV token-interaction feature**: would
  challenge the assumption that the copier tier provides the strongest
  weight-space signal and suggest an unexpected structural commonality
  across tiers.


## Prediction 2: Top-5 features combined

### Prediction

The top-5 features combined (equal weight, best direction, standardized)
achieve AUROC 0.78--0.88.

The improvement over the best single feature is +0.04--0.10.

### Rationale

Five features can potentially capture multiple tiers: feature 1 for
copiers (OV diagonal), feature 2 for backbone/detector (QK same-diff),
feature 3 for readout (composition or OV-unembed), features 4--5 for
refinement. Equal-weight combination with best-direction standardization
is a simple aggregation that can add signal from complementary features.

The improvement is modest because: (a) the 5 features are selected by
univariate AUROC ranking, not by complementarity --- the top 5 features
may all be OV-related and redundant; (b) equal weighting does not allow
the model to weight copier features higher than backbone features; (c)
with 108 candidates and 144 samples, the top 5 by AUROC include some
features that are high by chance rather than by signal.

### Falsification criteria

- **Top-5 AUROC > 0.92**: would mean the top 5 univariate features are
  nearly sufficient to classify the full circuit, implying the tiers
  share more weight-space structure than expected.

- **Top-5 AUROC < best single feature**: would mean the additional
  features add more noise than signal, likely from multiple-testing
  artifacts in the feature ranking.


## Prediction 3: LOO cross-validated logistic regression

### Prediction

LOO-CV logistic regression (L2, C=0.01) achieves AUROC 0.72--0.84.

C=0.01 corresponds to regularization strength lambda = 100, which
is very strong. All 108 coefficients will be shrunk heavily, and the
effective model is a soft average over many features with small weights.
This prevents overfitting but also limits the model's ability to
specialize different features for different tiers.

### Rationale

With 15 positives and 129 negatives (class ratio 1:8.6), each LOO fold
trains on either 14 positives or 15 positives depending on which sample
is held out. When a positive is held out, the training class ratio is
14:129 = 1:9.2. The model must extrapolate from 14 examples what makes
a head a circuit member.

Strong L2 regularization helps by preventing overfitting to the 108
features, but it also prevents the model from learning the multi-tier
structure (which would require different features for different tiers
with different signs and magnitudes). The regularized model will tend
to find a single direction in feature space that best separates
positives from negatives overall --- this direction will be dominated
by the copier signal because copiers are the majority of positives
(8/15) and have the strongest feature signature.

The LOO-CV AUROC may be similar to or slightly below the top-5 combined
AUROC because: the logistic regression uses all 108 features but with
strong shrinkage, effectively creating a noisy ensemble that dilutes
the signal from the best features. However, it can learn a better
weighting than equal-weight combination.

### Per-tier recall prediction at threshold = 0.5

| Tier | Size | Predicted recall | Reasoning |
|---|---|---|---|
| Copier | 8 | 5--8 of 8 (62--100%) | Dominant positive class with strong OV signature |
| Backbone | 3 | 0--2 of 3 (0--67%) | Weak signal; confused with other L0 heads |
| Detector | 1 | 0--1 of 1 (0--100%) | L4H11 may be pulled up by QK features shared with other induction circuits |
| Readout | 3 | 0--2 of 3 (0--67%) | Late-layer heads are not distinctive enough |

### Falsification criteria

- **LOO-CV AUROC > 0.88**: would mean the regularized linear model can
  capture multi-tier structure with 15 positives, which would be
  impressive given the class imbalance and feature dimensionality.

- **LOO-CV AUROC < 0.65**: would mean overfitting to noise dominates
  even with strong regularization, or the feature space genuinely does
  not contain enough signal to separate circuit from non-circuit heads.


## Prediction 4: Copier-seeded threshold

### Prediction

A threshold fit on the 8 copier heads will recover 6--8 of the 8
copiers as true positives while producing 3--8 false positives and
catching 0--2 additional circuit heads from other tiers.

Expected total circuit recall: 6--10 of 15 (40--67%).
Expected precision: 40--65%.

### Rationale

The copier-seeded approach fits a decision boundary on the feature(s)
that best separate the 8 copier heads from the remaining 136 heads.
The copier heads' OV suppression signature is strong and specific ---
negative diagonal of W_E @ W_OV @ W_U is a direct mechanistic property.

**Why 3--8 false positives**: copy suppression is not unique to the
RTI circuit. L10H7 is the famous "anti-induction" / copy suppression
head from McDougall et al. L11H10 is another known copy-suppression
head. Beyond these, several mid-layer heads (particularly in the L5--L9
range) may have partially negative OV diagonals for non-RTI tasks. A
threshold loose enough to catch all 8 copier heads will also catch
these structurally similar heads.

**Why 0--2 non-copier circuit heads**: the backbone heads (L0) write
positional signals into the residual stream, which is mechanistically
unrelated to OV suppression. The detector (L4H11) detects repeated
tokens via QK matching, which is a QK property not an OV property.
The readout heads (L10--11) read the suppression signal; they might
or might not have negative OV diagonals depending on whether they
amplify the suppression or simply attend to the suppressed positions.
At most 1--2 readout heads might fall within the copier threshold.

### Falsification criteria

- **Threshold catches > 4 non-copier circuit heads**: would mean the
  copier OV signature generalizes across tiers, suggesting a shared
  suppression mechanism beyond what the tier labels indicate.

- **Threshold produces > 12 false positives**: would mean copy
  suppression is too common across GPT-2 heads to be circuit-specific.

- **Threshold catches < 4 of 8 copiers**: would mean the copier
  tier's OV suppression signature is not as uniform as expected from
  the heatmap-based discovery. The 8 copier heads may implement
  suppression in diverse ways that a single threshold cannot capture.


## Prediction 5: Bootstrap greedy classifier (global mode)

### Prediction

The bootstrap greedy classifier in global mode (no ground-truth layer
leakage) produces a stability-scored ranking with AUROC 0.80--0.90.

At stability threshold 0.7, the classifier selects 12--22 heads, of
which 8--12 are true circuit members (recall 53--80%, precision 45--70%).

### Per-role stability predictions

| Role | GT heads | Predicted mean stability | Predicted heads with stability > 0.7 | Reasoning |
|---|---|---|---|
| Copier | 8 | 0.75--0.95 | 6--8 of 8 | Copier-specific greedy search with OV features should consistently select the same heads across feature subsamples. The OV suppression signal is strong enough to survive 80% feature subsampling. |
| Backbone | 3 | 0.30--0.65 | 0--2 of 3 | Backbone-specific search must separate 3 L0 heads from 141 others. With only 3 positives, the greedy formula has very little training signal. Feature subsampling across bootstraps will produce unstable selections. |
| Detector | 1 | 0.25--0.60 | 0--1 of 1 | Single-positive classification is inherently unstable. L4H11 has a distinctive QK profile (high same-diff ratio as a PTH head), but this property is shared with other induction heads (L5H1, L5H5, L6H9). The greedy formula may select L4H11 in some bootstraps and substitute another PTH head in others. |
| Readout | 3 | 0.35--0.65 | 0--2 of 3 | Readout search must find 3 late-layer heads. Composition features (v_comp_recv_max, k_comp_recv_max) may provide some signal if readout heads have distinctive upstream connectivity. But aggregate composition norms are noisy and shared across many late-layer heads. |

### Expected false positive profile

The global search runs a separate greedy formula per role, each
selecting top-2k heads. Heads that consistently appear in the top-2k
for any role get high stability. False positives will come from:

- **Copier false positives** (2--5 heads): non-RTI copy-suppression
  heads (L10H7, L11H10) and heads with incidentally negative OV
  diagonals. These heads share the copier OV signature but serve
  different circuits.

- **Backbone false positives** (1--4 heads): other L0 heads (L0H0,
  L0H1, L0H10) that share early-layer SVD/QK profiles with the
  3 backbone heads. With only 3 positives, the greedy formula cannot
  reliably distinguish backbone from other early-layer heads.

- **Detector false positives** (1--3 heads): other PTH/induction
  heads (L2H2, L5H1, L5H5) that share L4H11's QK matching properties.

- **Readout false positives** (1--3 heads): other late-layer heads
  with high composition norms.

### Falsification criteria

- **Stability AUROC > 0.93**: would mean the bootstrap procedure
  achieves near-perfect ranking with weight features alone and no
  layer constraints. This would be the strongest result --- it would
  demonstrate that weight features are practically sufficient for
  circuit discovery.

- **Stability AUROC < 0.75**: would mean the bootstrap procedure
  adds noise rather than robustness, and the greedy feature selection
  overfits within each bootstrap despite feature subsampling.

- **All 8 copier heads have stability < 0.5**: would mean the OV
  suppression signal is not robust to 80% feature subsampling, which
  would undermine the feature set's foundational signal.

- **Backbone or readout mean stability > copier mean stability**:
  would mean the tier I predicted to be hardest is actually easier
  than the tier I predicted to be easiest. Would require re-examining
  the assumption that OV features dominate the classification.


## Summary of predictions

| Test | Primary metric | Predicted range | Success threshold |
|---|---|---|---|
| 1. Best single feature | AUROC | 0.72--0.82 | > 0.70 |
| 2. Top-5 combined | AUROC | 0.78--0.88 | > 0.75 |
| 3. LOO logistic regression | AUROC | 0.72--0.84 | > 0.70 |
| 4. Copier-seeded threshold | Recall / precision | 40--67% / 40--65% | Recall > 40%, precision > 35% |
| 5. Bootstrap greedy (global) | Stability AUROC | 0.80--0.90 | > 0.75 |


## Success criteria

### "Features work" (confirmatory)

All of the following:
1. Best single feature AUROC > 0.70
2. At least one multi-feature method (tests 2--5) achieves AUROC > 0.80
3. The copier tier is the most separable tier (highest per-tier recall)
4. The bootstrap classifier recovers > 50% of circuit heads at
   precision > 40%

Meeting these criteria would demonstrate that weight-space features
carry real circuit-membership signal, dominated by the copier tier's
OV suppression signature but with partial signal for other tiers.

### "Features don't work" (disconfirmatory)

Any of the following:
1. Best single feature AUROC < 0.60
2. All multi-feature methods achieve AUROC < 0.70
3. Copier tier recall < 50% (the strongest expected signal fails)
4. Bootstrap classifier precision < 30% (more false positives than
   true positives)

Meeting any of these would indicate that weight-space features are
insufficient for circuit discovery on the RTI task, either because
the feature set misses the relevant properties or because circuit
membership is not a weight-space property (i.e., it requires
activation-level analysis).

### Intermediate outcome: "features partially work"

The most likely outcome: features successfully identify the copier
tier (recall > 75%) but struggle with backbone, detector, and readout
(combined recall < 50%). This would establish that weight features
capture one tier's signature well --- the tier that was discovered
from weight inspection --- but circuit discovery from weights alone
requires tier-specific feature engineering rather than a single
universal classifier.


## Key uncertainty: the multi-tier problem

My central prediction is that classifier performance will be tier-gated.
The copier tier accounts for 8/15 = 53% of the circuit and has the
strongest weight signature (it was literally discovered by looking at
weight matrices). Any classifier that finds copiers and guesses randomly
on everything else achieves AUROC approximately 0.77. Beating 0.80
requires signal from at least one additional tier.

The question is whether composition features, QK token-interaction
features, or direction-alignment features provide enough signal for
backbone, detector, or readout to push the classifiers above the
copier-only baseline. I predict they do, marginally, yielding the
0.80--0.90 range for the best classifier rather than the 0.75--0.80
that copier-only signal would produce.

If the best classifier achieves AUROC < 0.80, it means the non-copier
tiers contribute essentially no weight-space signal. If it achieves
AUROC > 0.90, it means the feature set captures multi-tier structure
that I did not anticipate.
