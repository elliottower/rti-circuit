# Pre-registration E4g: K-Composition Threshold Sensitivity

**Question**: Is circuit membership robust to the K-composition threshold? The paper uses composition scores to separate circuit from non-circuit heads but never states an explicit threshold. If the gap between the lowest circuit-edge score and the highest non-circuit-edge score is small, the circuit boundary is arbitrary.

## Method

Compute K-composition scores ||W_O^(i) @ W_K^(j)||_F for all directed head pairs where sender layer < receiver layer. For each key pathway (backbone→detector, backbone→copier, copier→readout):
1. Compute the gap between the minimum circuit-edge score and the maximum non-circuit-edge score (for the same receiver)
2. Plot histogram of all scores with circuit edges highlighted
3. Vary the inclusion threshold from P90 to P99 and report which circuit members are gained/lost

## Disclosure

- **Observed**: Backbone→detector scores (76-130 circuit vs 30-51 non-circuit, from paper). Background mean=10.4, P95=21.0 (from E4b/paper).
- **Unobserved**: Full distribution shape, exact gaps for all pathways, threshold sensitivity results

## Predictions

1. **Backbone→detector gap**: Large and unambiguous. Circuit edges (76-130) are 2-3x above the next highest non-circuit edge (~50). This pathway is robust to any reasonable threshold.

2. **Backbone→copier gap**: Moderate. The paper says backbone→copier mean is 36.2. Non-circuit L0→mid-layer edges will have scores in the 15-30 range. Gap exists but is narrower than backbone→detector.

3. **Copier→readout gap**: Minimal or nonexistent. The paper reports copier→readout mean K-comp of 8.9, near the background (7.6). These edges communicate through the residual stream, not direct composition. K-comp does not separate circuit from non-circuit edges in this pathway.

4. **Threshold sensitivity**: At P95 (score > 21.0), all backbone→detector edges survive. At P90, some backbone→copier edges may be lost. At P99, only backbone→detector edges survive. The circuit's composition-based backbone is robust; the copier→readout connectivity relies on residual-stream communication that K-comp cannot measure.

## Falsification

- **Backbone→detector gap < 10 (score units)**: The flagship pathway is not clearly separated and the P95 threshold is not robust.
- **More than 2 circuit copier heads lost at P90 threshold**: Circuit membership is highly threshold-dependent.
- **Non-circuit edges score above 76 (the minimum circuit backbone→detector score)**: The gap is not a gap at all.
