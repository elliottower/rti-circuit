# Pre-registration E4f: OV Diagonal Score Sensitivity to k

**Question**: Does the OV diagonal score (which identified the copier tier) depend on the choice of k in the heatmap W_E[:k] @ W_V @ W_O @ W_U[:, :k]? The paper uses k=50 without justification. If the copier-tier signal vanishes at other k values, the finding is fragile.

## Method

Compute OV diagonal score = mean(diag(M)) - mean(offdiag(M)) for all 144 heads at k = 50, 100, 500, 1000. Report:
1. Spearman rank correlation of diagonal scores between k=50 and each other k
2. Whether all 8 copier heads maintain positive diagonal scores at every k
3. Whether the copier heads' rank positions (among 144) are stable

## Disclosure

- **Observed**: OV diagonal scores at k=50 from E4b (copier heads have positive scores, mean +0.65)
- **Unobserved**: All values at k=100, 500, 1000; rank stability; whether any copier head flips sign

## Predictions

1. **Rank correlation**: Spearman rho > 0.90 between k=50 and all other k values. The heads that have strong diagonal structure at k=50 will have it at other k values because the diagonal pattern is a property of the OV circuit, not the token subset.

2. **Sign stability**: All 8 copier heads will have positive diagonal scores at all k values. The positive diagonal reflects a structural property of W_V @ W_O (token-copying capacity), projected through W_E and W_U. Different k values sample different tokens but the diagonal structure is token-agnostic.

3. **Rank stability**: The 8 copier heads will rank in the top 30 (highest diagonal score, since their scores are positive) at all k values. Some rank shuffling within the group is expected.

4. **Score magnitude**: Absolute diagonal scores will decrease as k increases, because larger k includes less frequent tokens with weaker embedding norms, diluting the mean. The relative ordering should be preserved.

## Falsification

- **Any copier head has negative diagonal score at any k**: The positive-diagonal claim is k-dependent and the paper's mechanism story is incomplete.
- **Spearman rho < 0.80 between k=50 and k=1000**: The heatmap patterns are token-subset-dependent and k=50 was a lucky choice.
- **A copier head ranks below median (rank > 72) at any k**: The copier tier is not consistently distinguishable by diagonal score.
