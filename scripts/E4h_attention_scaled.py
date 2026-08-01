"""E4h: Copier attention patterns at scale (200 prompts).

Generates 200 prompts with repeated tokens from 12 templates x 17 name
pairs. Measures attention to repeated vs non-repeated positions for all
15 RTI circuit heads. Reports means with 95% bootstrap CIs.

Usage:
  uv run python paper/E4h_attention_scaled.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

OUT_DIR = Path(__file__).resolve().parent / "paper_numbers" / "E4h_attention_scaled"

RTI_CIRCUIT = {
    "backbone": [(0, 8), (0, 9), (0, 11)],
    "detector": [(4, 11)],
    "copier": [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
    "readout": [(10, 11), (11, 9), (11, 11)],
}

TEMPLATES = [
    "Then {a} and {b} went to the store. {a} gave a drink to",
    "{a} told {b} a story. Later {b} told {a} about the",
    "The teacher asked {a} and {b} to stay. {a} spoke to",
    "{a} met {b} at the park. {b} walked with {a} to the",
    "First {a} arrived, then {b} came in. {a} sat next to",
    "{a} called {b} on the phone. {b} answered and told {a} about the",
    "Both {a} and {b} were there. {a} looked at {b} and then at",
    "{a} gave {b} a book. {b} thanked {a} and opened the",
    "Everyone liked {a} and {b} equally. {a} talked to {b} about the",
    "{a} saw {b} across the room. {b} waved at {a} and walked to the",
    "The coach picked {a} and {b} for the team. {a} passed to",
    "{a} and {b} shared a meal. Then {a} ordered more for",
]

NAMES = [
    ("Alice", "Bob"), ("Carol", "Dave"), ("Eve", "Frank"),
    ("Grace", "Henry"), ("Iris", "Jack"), ("Kate", "Leo"),
    ("Mary", "Nick"), ("Olivia", "Paul"), ("Quinn", "Ryan"),
    ("Sarah", "Tom"), ("Uma", "Victor"), ("Wendy", "Xavier"),
    ("Yara", "Zack"), ("Diana", "Eric"), ("Fiona", "George"),
    ("Hannah", "Ian"), ("Julia", "Kevin"),
]


def find_repeated_positions(tokens):
    token_ids = tokens[0].tolist()
    seen = set()
    is_repeat = []
    for tid in token_ids:
        is_repeat.append(tid in seen)
        seen.add(tid)
    return is_repeat


def generate_prompts():
    prompts = []
    for template in TEMPLATES:
        for a, b in NAMES:
            prompts.append(template.format(a=a, b=b))
    return prompts


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    values = np.array(values)
    n = len(values)
    boot_means = np.array([
        np.mean(values[np.random.randint(0, n, size=n)])
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lo, hi


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] E4h: Attention patterns at scale", flush=True)

    prompts = generate_prompts()
    print(f"[{datetime.now().isoformat()}] Generated {len(prompts)} prompts", flush=True)

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()
    tokenizer = model.tokenizer

    head_prefs = {}
    for tier, heads in RTI_CIRCUIT.items():
        for layer, head in heads:
            head_prefs[f"L{layer}H{head}"] = {"tier": tier, "preferences": []}

    for prompt_text in tqdm(prompts, desc="Running prompts"):
        tokens = model.to_tokens(prompt_text, prepend_bos=True)
        seq_len = tokens.shape[1]
        is_repeat = find_repeated_positions(tokens)

        n_repeated = sum(is_repeat)
        n_unique = sum(1 for r in is_repeat if not r)
        if n_repeated == 0 or n_unique == 0:
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: "attn.hook_pattern" in n
            )

        last_pos = seq_len - 1
        for tier, heads in RTI_CIRCUIT.items():
            for layer, head in heads:
                name = f"L{layer}H{head}"
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]
                attn_from_last = pattern[last_pos, :].cpu().numpy()

                attn_to_repeated = sum(
                    attn_from_last[i] for i in range(seq_len) if is_repeat[i]
                )
                attn_to_unique = sum(
                    attn_from_last[i] for i in range(seq_len) if not is_repeat[i]
                )
                pref = float(attn_to_repeated - attn_to_unique)
                head_prefs[name]["preferences"].append(pref)

    results = {}
    for name, data in sorted(head_prefs.items()):
        prefs = data["preferences"]
        if len(prefs) == 0:
            continue
        mean_pref = float(np.mean(prefs))
        ci_lo, ci_hi = bootstrap_ci(prefs)
        results[name] = {
            "tier": data["tier"],
            "n_prompts": len(prefs),
            "mean_preference": mean_pref,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
            "direction": "REPEATED" if mean_pref > 0 else "UNIQUE",
            "std": float(np.std(prefs)),
        }

    with open(OUT_DIR / "attention_scaled_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== RESULTS ({len(prompts)} prompts) ===", flush=True)
    for tier in ["backbone", "detector", "copier", "readout"]:
        print(f"\n--- {tier} ---", flush=True)
        for name, r in sorted(results.items()):
            if r["tier"] != tier:
                continue
            ci_str = "EXCLUDES 0" if r["ci_excludes_zero"] else "INCLUDES 0"
            print(
                f"  {name}: pref={r['mean_preference']:+.3f} "
                f"[{r['ci_95_lo']:+.3f}, {r['ci_95_hi']:+.3f}] "
                f"({r['direction']}, {ci_str}, n={r['n_prompts']})",
                flush=True,
            )

    copier_prefs = [r["mean_preference"] for name, r in results.items() if r["tier"] == "copier"]
    if copier_prefs:
        print(f"\nCopier tier summary: mean={np.mean(copier_prefs):+.3f}, "
              f"all_negative={all(p < 0 for p in copier_prefs)}, "
              f"all_ci_exclude_zero={all(r['ci_excludes_zero'] for name, r in results.items() if r['tier'] == 'copier')}",
              flush=True)

    print(f"\n[{datetime.now().isoformat()}] Saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
