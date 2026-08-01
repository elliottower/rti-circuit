# EAP Scaling Analysis: Why Weight Transfer is the Natural Alternative

*Perplexity deep research analysis of EAP memory scaling and weight method advantages.*

## Why EAP OOMs on Large

EAP computes a score for every **edge** in the computation graph, not just every node. An edge connects every (sender layer, head) to every (receiver layer, head) downstream — so edge count scales roughly as O(L^2 * H^2) where L is layers and H is heads per layer.

| Model | Nodes (L x H) | Rough edge count | Score matrix memory |
|---|---|---|---|
| GPT-2 Small | 144 | ~10K | ~400 MB |
| GPT-2 Medium | 384 | ~74K | ~2.4 GB |
| GPT-2 Large | 720 | ~259K | ~27+ GB |
| GPT-2 XL | 1200 | ~720K | ~75+ GB |

The score matrix goes quadratic in model size, so even an A40 (48 GB) OOMs on Large. EAP-IG is worse because it also runs K=10-50 forward+backward passes per edge chunk, multiplying compute while keeping the same memory problem.

## It's Engineering, Not Fundamental

You could fix this with **chunked edge accumulation**: compute scores for one source-node slice at a time, stream to CPU, repeat. It'd be slower (~K x L_source batches), but no memory wall. The standard open-source EAP implementations (including the one in our codebase) don't implement this.

**We have access to H200s on RunPod** — the literature runs EAP at these scales routinely. So this is fixable: either chunked accumulation on a 4090/A40, or brute-force on an H200 (80 GB). Worth doing as a reviewer-proofing step.

## Paper Framing

Even if we can run EAP on large GPUs, the scaling argument still holds as a structural observation:

**Weight-based circuit transfer is O(n_heads) with no gradient computation and no memory wall** — extract weight features from each head independently, which scales linearly and runs entirely on CPU. At GPT-2 Large, our weight method ran in minutes while EAP requires special engineering or very large GPUs.

This isn't just practical convenience — it's a structural difference:
- **EAP/EAP-IG**: activation-space methods that need forward passes, perturbation, and backpropagation through the full graph. Memory scales quadratically with model size.
- **Weight methods**: operate on weight matrices directly. No data, no gradients, no memory wall. Scales linearly.

The interesting finding is the **complementarity + scaling divergence**:
- On Medium (where both run): EAP is head-efficient (20 heads -> IIA 1.0) but the union outperforms either alone and the methods find almost entirely different heads
- On Large (where only weight ran so far): weight method gets IIA 1.0 via L0 alone

If a reviewer asks "couldn't you just chunk EAP?": "yes, and we should — but the point is that weight-space methods are architecturally suited to this regime independent of implementation choices."

## Action Items

1. **Run EAP on large via H200 pod** — get the EAP baseline to make the comparison complete, don't leave it as "EAP couldn't run"
2. **Run EAP on XL if H200 has enough memory** — 75+ GB score matrix might still OOM even on H200 (80 GB), but worth trying
3. **Compare EAP head rankings on large vs weight-predicted heads** — does the near-zero overlap from medium persist at large scale?
4. **Union experiment on large** — if EAP works, test union (weight + EAP) on large like we did on medium
