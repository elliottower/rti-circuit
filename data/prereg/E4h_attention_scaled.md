# Pre-registration E4h: Copier Attention Patterns at Scale

**Question**: Do copier heads consistently attend to non-repeated tokens? E4c showed this on 10 hand-crafted prompts. A reviewer would require 100+ prompts with statistical testing.

## Method

Generate 200 prompts with repeated tokens using template-based construction:
- 12 templates x ~17 name pairs = ~200 prompts
- Each prompt has exactly 2 occurrences of a name, with the repeated token at a known position
- Measure attention from the final position to repeated vs non-repeated source positions for all 15 circuit heads
- Report mean repeat-preference (attn_to_repeated - attn_to_unique) with 95% bootstrap CIs

## Disclosure

- **Observed**: E4c results on 10 prompts. All 8 copier heads had negative repeat preference (range -0.29 to -0.78). Detector L4H11 had +0.40.
- **Unobserved**: Results on the 200 new prompts, bootstrap CIs, whether the effect is consistent across templates

## Predictions

1. **All 8 copier heads**: Negative repeat preference with 95% CI entirely below zero. Expected mean around -0.3 to -0.6 (slightly attenuated from E4c because template variety introduces more noise).

2. **Detector L4H11**: Positive repeat preference with 95% CI entirely above zero. Expected mean around +0.2 to +0.5. The detector's defining property is attending to repeated positions.

3. **Backbone heads**: Near zero repeat preference. They attend broadly (BOS, early positions) and should not discriminate between repeated and non-repeated tokens.

4. **Readout heads**: Negative repeat preference (attending to unique tokens), expected stronger than copier heads (E4c showed L11H9 at -0.94). These are induction heads that attend to the non-repeated next-token prediction.

5. **Effect size vs E4c**: Mean absolute repeat-preference across copier heads will be within 0.15 of the E4c values. Template-based prompts are structurally similar to the hand-crafted ones, so the attention patterns should be comparable.

## Falsification

- **Any copier head has 95% CI overlapping zero**: That head's attention pattern is not reliably unique-preferring, weakening the mechanism story.
- **Detector L4H11 has negative repeat preference**: The detector's role is mischaracterized.
- **Mean copier repeat-preference > -0.10**: The effect is real but too small to support "attend preferentially to non-repeated tokens" language.
