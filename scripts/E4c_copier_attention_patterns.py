"""E4c: What do copier heads attend to on prompts with repeated tokens?

Runs GPT-2 Small on prompts containing repeated tokens and extracts
attention patterns for all 15 RTI circuit heads. Saves per-head attention
to repeated vs non-repeated source tokens.

Usage:
  uv run python paper/E4c_copier_attention_patterns.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

OUT_DIR = Path(__file__).resolve().parent / "paper_numbers" / "E4c_attention_patterns"

RTI_CIRCUIT = {
    "backbone": [(0, 8), (0, 9), (0, 11)],
    "detector": [(4, 11)],
    "copier": [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
    "readout": [(10, 11), (11, 9), (11, 11)],
}

PROMPTS = [
    "The cat sat on the mat. The cat",
    "I went to the store and then I went to the park and then I went",
    "Paris is the capital of France. Paris is a beautiful city. Paris",
    "The dog chased the ball. The dog caught the ball. The dog",
    "She said hello and he said hello back. She said",
    "One two three four five one two three four five one two three",
    "Alice gave Bob a book. Bob gave Alice a gift. Alice gave Bob",
    "The red car and the blue car were parked. The red car",
    "It was a dark and stormy night. It was a dark",
    "They walked and walked and walked until they walked",
]


def find_repeated_positions(tokens, tokenizer):
    """For each position, mark whether that token has appeared before."""
    token_ids = tokens[0].tolist()
    seen = set()
    is_repeat = []
    for i, tid in enumerate(token_ids):
        is_repeat.append(tid in seen)
        seen.add(tid)
    return is_repeat


def run_attention_analysis(model, prompts):
    results = []
    tokenizer = model.tokenizer

    for prompt_text in prompts:
        tokens = model.to_tokens(prompt_text, prepend_bos=True)
        seq_len = tokens.shape[1]

        is_repeat = find_repeated_positions(tokens, tokenizer)
        token_strs = [tokenizer.decode([t]) for t in tokens[0].tolist()]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: "attn.hook_pattern" in n)

        prompt_result = {
            "text": prompt_text,
            "tokens": token_strs,
            "is_repeat": is_repeat,
            "seq_len": seq_len,
            "n_repeated": sum(is_repeat),
            "n_unique": sum(not r for r in is_repeat),
            "heads": {},
        }

        for tier, heads in RTI_CIRCUIT.items():
            for layer, head in heads:
                name = f"L{layer}H{head}"
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # (seq, seq)
                pattern_np = pattern.cpu().numpy()

                last_pos = seq_len - 1
                attn_from_last = pattern_np[last_pos, :]  # what does last position attend to?

                attn_to_repeated = sum(attn_from_last[i] for i in range(seq_len) if is_repeat[i])
                attn_to_unique = sum(attn_from_last[i] for i in range(seq_len) if not is_repeat[i])

                per_position = []
                for dest in range(seq_len):
                    attn_row = pattern_np[dest, :dest + 1]
                    repeat_mask = [is_repeat[j] for j in range(dest + 1)]
                    if any(repeat_mask):
                        attn_rep = sum(attn_row[j] for j in range(dest + 1) if repeat_mask[j])
                    else:
                        attn_rep = 0.0
                    if any(not r for r in repeat_mask):
                        attn_uniq = sum(attn_row[j] for j in range(dest + 1) if not repeat_mask[j])
                    else:
                        attn_uniq = 0.0
                    per_position.append({
                        "dest_token": token_strs[dest],
                        "dest_is_repeat": is_repeat[dest],
                        "attn_to_repeated": float(attn_rep),
                        "attn_to_unique": float(attn_uniq),
                    })

                prompt_result["heads"][name] = {
                    "tier": tier,
                    "last_pos_attn_to_repeated": float(attn_to_repeated),
                    "last_pos_attn_to_unique": float(attn_to_unique),
                    "per_position": per_position,
                }

        results.append(prompt_result)

    return results


def summarize(results):
    summary = {}
    all_heads = set()
    for r in results:
        all_heads.update(r["heads"].keys())

    for head_name in sorted(all_heads):
        repeat_attns = []
        unique_attns = []
        per_pos_repeat_when_dest_is_repeat = []
        per_pos_repeat_when_dest_is_unique = []

        for r in results:
            h = r["heads"][head_name]
            repeat_attns.append(h["last_pos_attn_to_repeated"])
            unique_attns.append(h["last_pos_attn_to_unique"])

            for pp in h["per_position"]:
                if pp["dest_is_repeat"]:
                    per_pos_repeat_when_dest_is_repeat.append(pp["attn_to_repeated"])
                else:
                    per_pos_repeat_when_dest_is_unique.append(pp["attn_to_repeated"])

        summary[head_name] = {
            "tier": results[0]["heads"][head_name]["tier"],
            "mean_attn_to_repeated_from_last": float(np.mean(repeat_attns)),
            "mean_attn_to_unique_from_last": float(np.mean(unique_attns)),
            "repeat_preference": float(np.mean(repeat_attns)) - float(np.mean(unique_attns)),
            "mean_attn_to_repeated_when_dest_is_repeat": float(np.mean(per_pos_repeat_when_dest_is_repeat)) if per_pos_repeat_when_dest_is_repeat else 0,
            "mean_attn_to_repeated_when_dest_is_unique": float(np.mean(per_pos_repeat_when_dest_is_unique)) if per_pos_repeat_when_dest_is_unique else 0,
        }

    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] E4c: Copier attention patterns", flush=True)

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()

    print(f"[{datetime.now().isoformat()}] Running {len(PROMPTS)} prompts...", flush=True)
    results = run_attention_analysis(model, PROMPTS)

    with open(OUT_DIR / "attention_patterns.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{datetime.now().isoformat()}] Summarizing...", flush=True)
    summary = summarize(results)

    with open(OUT_DIR / "attention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== ATTENTION TO REPEATED vs UNIQUE TOKENS (from last position) ===", flush=True)
    for tier in ["backbone", "detector", "copier", "readout"]:
        print(f"\n--- {tier} ---", flush=True)
        for name, s in sorted(summary.items()):
            if s["tier"] != tier:
                continue
            pref = s["repeat_preference"]
            direction = "REPEATED" if pref > 0 else "UNIQUE"
            print(f"  {name}: to_repeated={s['mean_attn_to_repeated_from_last']:.3f}  to_unique={s['mean_attn_to_unique_from_last']:.3f}  pref={pref:+.3f} ({direction})", flush=True)

    print(f"\n[{datetime.now().isoformat()}] Saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
