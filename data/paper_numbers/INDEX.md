# Paper Drafts — Source Index

## Paper scope

Five papers from this research:
1. **Paper A: Base weight method** — `paper_a_weight_method.tex` — weight feature extraction, 4-tier classification, 15-head RTI circuit, 19 causal experiments, cross-task transfer, 10-page appendix with full per-head tables
2. **Paper B: Cross-model scaling** — `paper_b_cross_model_scaling.tex` — weight transfer across GPT-2 scales, L0 bottleneck, EAP/EAP-IG comparison, IIA threshold phase transitions, compact 2D accumulation, minimum circuits
3. **Paper C: Anti-repetition circuit** — `paper_c_anti_repetition.tex` — weight-identified degenerate token suppression mechanism (mostly TODOs, needs experimental data from separate chat)
4. **Paper D: Linguistic circuits census** — `paper_d_linguistic_circuits.tex` — systematic weight-space census across many tasks (mostly TODOs, needs additional task experiments)
5. **Paper E: Phonetic/phonological circuits** — `paper_e_phonetic_circuits.tex` — 6 phonological operations (rhyming hypocorism, clipping, initialism, oronym, homophone, folk etymology), cross-boundary phoneme fusion as core novel finding, compositional validation. Pods running, results pending.

Papers A and B may be combined (base method + scaling as one story). Papers C and D are stubs awaiting experimental data. Paper E has full outline with PENDING placeholders for pod results.

---

## Experiment results (primary sources — always check these for latest numbers)

### Cross-model experiments (this folder)
- `../EXPERIMENT_LOG.md` — **master log**, 7 phases, all results with dates and pod IDs
- `../HEAD_CHARACTERIZATION_CROSS_MODEL.md` — all head tables for medium/large/xl

### IIA strategy results (Phase 5)
- `../experiments/batch3_iia_improvement/RESULTS_FAST_GROUP.md` — threshold sweep, layer sweep, EAP top-K, union
- `../experiments/batch3_iia_improvement/RESULTS_GREEDY_CEILING.md` — greedy trajectories, ceiling circuits, L0 bottleneck
- `../experiments/batch3_iia_improvement/PERPLEXITY_EAP_SCALING_ANALYSIS.md` — EAP memory scaling, compact mode rationale

### Base method results (Paper A)
- `../../v2_second_investigation/raw_experiments/v1_role_weight_analysis/part4_rigorous_circuit_finding/HEAD_CHARACTERIZATION.md` — GPT-2 small circuits
- `../../v2_second_investigation/raw_experiments/v1_role_weight_analysis/part4_rigorous_circuit_finding/LAB_NOTEBOOK.md` — full lab notebook
- `../../v2_second_investigation/raw_experiments/v1_role_weight_analysis/part4_rigorous_circuit_finding/FINDINGS.md` — summary findings
- `../../v2_second_investigation/raw_experiments/v1_role_weight_analysis/part4_rigorous_circuit_finding/DISCOVERY_STORY.md` — narrative

### Data files (JSON)
All in `../data/`:
- `transfer_results_{medium,large,xl}.json` — bootstrap stability per head per task
- `eap_cross_model_{small,medium,large,xl}.json` — EAP head rankings, all tasks
- `eap_ig_inputs_cross_model_{medium,large,xl}.json` — EAP-IG head rankings, all tasks
- `iia_cross_model_{medium,large,xl}.json` — original IIA results (threshold 0.7)
- `control_results_{medium,large,xl}.json` — depth-random control baselines
- `features_{gpt2_small,gpt2_medium,large,xl}.json` — extracted weight features
- `transfer_pythia{160m,410m,1.4b}.json` — cross-architecture transfer
- `behavioral_*.json`, `clusters_*.json` — behavioral validation, clustering

---

## Key numbers for the paper

### Headline results
- **IIA=0 was a threshold artifact**: threshold 0.7→0.0 gives IIA 0.8-1.0 across all tasks and scales
- **L0 alone achieves IIA=1.0 on GPT-2 large**: 20 heads (2.8% of 720) for IOI and GT
- **Minimum circuits are 3-16 heads**: ceiling search on medium finds 3-head SVA, 3-head GT, 7-head IOI circuits
- **Weight-EAP disjointness**: 0-1/15 overlap at all scales, both EAP and EAP-IG
- **EAP-IG finds L0 on large, basic EAP does not**: L0H14 is EAP-IG rank 1 for RTI/SVA/GT on large
- **Compact mode**: 2D accumulation drops EAP memory from 93 GB to 30 GB (XL runs on A40)

### Phase transition thresholds (GPT-2 large)
- IOI: IIA=1.0 at threshold 0.2 (71 heads), IIA=0.212 at 0.3 (53 heads)
- SVA: IIA=0.917 at threshold 0.1 (100 heads), IIA=0.0 at 0.2 (50 heads)
- GT: IIA=0.909 from threshold 0.0 to 0.4 (robust, only 17 heads needed)

### L0 greedy trajectory (GPT-2 large IOI)
Steps 1-8: 8 late-layer heads, IIA=0.000, shift grows to 1.453
Step 13: L0H3 added, IIA jumps 0.312→0.963, shift jumps to 7.563

---

## Previous drafts and analyses

### Perplexity LaTeX drafts (pre-IIA results, Phase 1-2 only)
- `reference/perplexity_drafts/claude_pre_iia.tex` — Claude's draft
- `reference/perplexity_drafts/gemini_pre_iia.tex` — Gemini's draft
- `reference/perplexity_drafts/gpt_pre_iia.tex` — GPT's draft
- `reference/paper_draft.tex` — consolidated draft (same as claude.tex in structure)

These are **outdated** — written before the IIA threshold sweep, L0 finding, and EAP-IG results. The framing of "transfer fails" is now wrong. Useful for structure/LaTeX formatting only.

### Model consensus analyses
- `reference/model_consensus/consensus_batch1.txt` — 3-model consensus on framing
- `reference/model_consensus/claude_batch1_analysis.txt` — Claude's analysis of batch 1
- `reference/model_consensus/gpt_batch1_analysis.txt` — GPT's analysis of batch 1
- `reference/model_consensus/claude_initial.txt` — Claude's initial analysis

---

## Blog posts (external repo, for tone/narrative reference)

These live in `/Users/elliottower/Documents/GitHub/elliottower.ai/src/content/blog/v3/`:
- `00-overview.md` / `00-overview_v2.md` — project overview
- `01-the-circuit-rewritten.md` — circuit description
- `02-the-method-rewritten.md` — method description
- `03-mib-benchmark-rewritten.md` — MIB benchmark results
- `04-deep-dive-rewritten.md` — technical deep dive
- `05-what-we-learned-rewritten.md` — lessons learned
- `06-anti-repetition.md` — anti-repetition circuit (Paper C material)

---

## Perplexity brainstorms (new circuits / other models)

These live in `../PERPLEXITY_WRITEUPS/part6_other_circuits_models/`:
- `PERPLEXITY_BRAINSTORM.md` — brainstorm on extending to new circuits
- `PERPLEXITY_V2.md` — v2 brainstorm
- `new_circuits/spec_*.md` — specs for IOI, SVA, GT, induction, gendered pronoun, other tasks
