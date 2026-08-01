"""Proper edge-level path patching for the RTI circuit (Wang et al. 2022 methodology).

For each (sender, receiver) edge in the circuit:
  1. Run clean and corrupted forward passes, cache hook_result at all layers.
  2. Patch sender head output from corrupted.
  3. Freeze all intermediate attention heads to clean (blocks attention-mediated propagation).
  4. At receiver layer, freeze all non-receiver heads to clean.
  5. Freeze all post-receiver attention heads to clean.
  6. Measure logit diff change.

MLPs recompute at all layers (standard — we isolate attention-to-attention paths).

Two levels of analysis:
  - Per-edge: isolate each intra-circuit edge, report effect sizes with bootstrap CIs.
  - Per-pathway: patch all senders in a tier, let all receivers in target tier recompute,
    report recovery fraction relative to full-circuit corruption.

Usage:
    python paper/E4i_path_patching_proper.py --device cuda
    python paper/E4i_path_patching_proper.py --device cuda --batch-size 1
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get(
    "PP_OUT_DIR",
    str(SCRIPT_DIR / "paper_numbers" / "E4i_path_patching_proper"),
))

RTI_DIR = Path(os.environ.get("RTI_DIR", str(
    Path.home()
    / "Documents/GitHub/factorization-circuits/MIB/MIB-circuit-track"
    / "weight_circuit/experiments/v2_second_investigation/raw_experiments"
    / "v1_role_weight_analysis/part3_l9h3_investigation"
)))
sys.path.insert(0, str(RTI_DIR))

BACKBONE = [(0, 8), (0, 9), (0, 11)]
DETECTOR = [(4, 11)]
COPIER = [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)]
READOUT = [(10, 11), (11, 9), (11, 11)]
ALL_CIRCUIT = BACKBONE + DETECTOR + COPIER + READOUT

TIER_MAP = {}
for h in BACKBONE:
    TIER_MAP[h] = "backbone"
for h in DETECTOR:
    TIER_MAP[h] = "detector"
for h in COPIER:
    TIER_MAP[h] = "copier"
for h in READOUT:
    TIER_MAP[h] = "readout"

PATHWAYS = [
    ("backbone->detector", BACKBONE, DETECTOR),
    ("backbone->copier", BACKBONE, COPIER),
    ("backbone->readout", BACKBONE, READOUT),
    ("detector->copier", DETECTOR, COPIER),
    ("detector->readout", DETECTOR, READOUT),
    ("copier->readout", COPIER, READOUT),
]

N_BOOT = 10_000


def pad_to_match(tok_a, tok_b, pad_id=50256):
    """Right-pad two token tensors to the same sequence length."""
    la, lb = tok_a.shape[1], tok_b.shape[1]
    if la == lb:
        return tok_a, tok_b
    target = max(la, lb)
    if la < target:
        tok_a = torch.nn.functional.pad(tok_a, (0, target - la), value=pad_id)
    if lb < target:
        tok_b = torch.nn.functional.pad(tok_b, (0, target - lb), value=pad_id)
    return tok_a, tok_b


def bootstrap_ci(values, n_boot=N_BOOT, ci=0.95):
    values = np.array(values, dtype=np.float64)
    n = len(values)
    rng = np.random.RandomState(42)
    means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return {
        "mean": float(np.mean(values)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(np.std(means)),
        "n": int(n),
    }


def get_intra_circuit_edges():
    edges = []
    for sl, sh in ALL_CIRCUIT:
        for rl, rh in ALL_CIRCUIT:
            if sl < rl:
                edges.append((sl, sh, rl, rh))
    return edges


def cache_hook_results(model, tokens, n_layers):
    hook_filter = lambda name: "attn.hook_result" in name
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=hook_filter)
    return logits, cache


def path_patch_single_edge(model, clean_tokens, clean_cache, corrupt_cache,
                           sl, sh, rl, rh, n_layers):
    """Isolate edge (sl,sh)->(rl,rh) per Wang et al. (2022) methodology.

    1. Patch sender head output from corrupted.
    2. Freeze all intermediate attention heads (between sender and receiver) to clean,
       blocking attention-mediated propagation. MLPs recompute freely.
    3. At receiver layer, freeze all non-receiver heads to clean — only the receiver
       sees the patched residual stream.
    4. Post-receiver layers recompute normally (IOI paper: "The layers after the
       element from R are recomputed as in a normal forward pass").
    """
    hooks = []

    def make_sender_hook(_sl, _sh):
        def fn(value, hook):
            out = clean_cache[hook.name].clone()
            out[:, :, _sh, :] = corrupt_cache[hook.name][:, :, _sh, :]
            return out
        return fn
    hooks.append((f"blocks.{sl}.attn.hook_result", make_sender_hook(sl, sh)))

    for mid_l in range(sl + 1, rl):
        def make_freeze(_ml):
            def fn(value, hook):
                return clean_cache[hook.name].clone()
            return fn
        hooks.append((f"blocks.{mid_l}.attn.hook_result", make_freeze(mid_l)))

    def make_recv_hook(_rl, _rh):
        def fn(value, hook):
            out = clean_cache[hook.name].clone()
            out[:, :, _rh, :] = value[:, :, _rh, :]
            return out
        return fn
    hooks.append((f"blocks.{rl}.attn.hook_result", make_recv_hook(rl, rh)))

    with torch.no_grad():
        patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

    return patched_logits


def path_patch_pathway(model, clean_tokens, clean_cache, corrupt_cache,
                       sender_heads, receiver_heads, n_layers):
    """Patch all senders, freeze intermediates, let receivers + post-receiver recompute.

    Follows Wang et al. (2022): post-receiver layers recompute normally so the
    receiver's response propagates through the rest of the network.
    """
    min_sender = min(l for l, h in sender_heads)
    max_receiver = max(l for l, h in receiver_heads)

    hooks = []

    for layer in range(min_sender, max_receiver + 1):
        heads_at_layer_s = [(l, h) for l, h in sender_heads if l == layer]
        heads_at_layer_r = [(l, h) for l, h in receiver_heads if l == layer]

        if heads_at_layer_s:
            def make_sender_layer_hook(_layer, _s_heads):
                s_head_indices = {h for l, h in _s_heads}
                def fn(value, hook):
                    out = clean_cache[hook.name].clone()
                    for hi in s_head_indices:
                        out[:, :, hi, :] = corrupt_cache[hook.name][:, :, hi, :]
                    return out
                return fn
            hooks.append((f"blocks.{layer}.attn.hook_result",
                          make_sender_layer_hook(layer, heads_at_layer_s)))

        elif heads_at_layer_r:
            def make_recv_layer_hook(_layer, _r_heads):
                r_head_indices = {h for l, h in _r_heads}
                def fn(value, hook):
                    out = clean_cache[hook.name].clone()
                    for hi in r_head_indices:
                        out[:, :, hi, :] = value[:, :, hi, :]
                    return out
                return fn
            hooks.append((f"blocks.{layer}.attn.hook_result",
                          make_recv_layer_hook(layer, heads_at_layer_r)))

        else:
            def make_freeze_mid(_layer):
                def fn(value, hook):
                    return clean_cache[hook.name].clone()
                return fn
            hooks.append((f"blocks.{layer}.attn.hook_result", make_freeze_mid(layer)))

    with torch.no_grad():
        patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
    return patched_logits


def run_all(device="cpu"):
    from run_rti_task import make_prompts
    from transformer_lens import HookedTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{datetime.now().isoformat()}] Loading GPT-2 Small on {device}...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    model.cfg.use_attn_result = True
    n_layers = model.cfg.n_layers

    print(f"[{datetime.now().isoformat()}] Generating RTI prompts...", flush=True)
    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts
             if p["correct_id"] is not None and p["distractor_id"] is not None]
    print(f"  {len(valid)} valid prompts", flush=True)

    n = len(valid)
    rng = np.random.RandomState(42)
    shuffle_idx = rng.permutation(n)

    edges = get_intra_circuit_edges()
    print(f"  {len(edges)} intra-circuit edges", flush=True)

    # ── Phase 1: baselines ──
    print(f"\n[{datetime.now().isoformat()}] Phase 1: computing baselines...", flush=True)
    baseline_lds = []
    full_corrupt_lds = []

    for i in tqdm(range(n), desc="Baselines"):
        p = valid[i]
        alt_p = valid[shuffle_idx[i]]
        clean_tokens = model.to_tokens(p["text"], prepend_bos=True).to(device)
        corrupt_tokens = model.to_tokens(alt_p["text"], prepend_bos=True).to(device)
        clean_tokens, corrupt_tokens = pad_to_match(clean_tokens, corrupt_tokens)
        clean_seq_len = model.to_tokens(p["text"], prepend_bos=True).shape[1]

        clean_logits, clean_cache = cache_hook_results(model, clean_tokens, n_layers)
        _, corrupt_cache = cache_hook_results(model, corrupt_tokens, n_layers)

        last = clean_logits[0, clean_seq_len - 1]
        baseline_lds.append((last[p["correct_id"]] - last[p["distractor_id"]]).item())

        # Full circuit corruption: patch all 15 heads
        hooks = []
        for layer in range(n_layers):
            circuit_at_layer = [h for l, h in ALL_CIRCUIT if l == layer]
            if circuit_at_layer:
                def make_full_hook(_layer, _heads):
                    def fn(value, hook):
                        out = clean_cache[hook.name].clone()
                        for hi in _heads:
                            out[:, :, hi, :] = corrupt_cache[hook.name][:, :, hi, :]
                        return out
                    return fn
                hooks.append((f"blocks.{layer}.attn.hook_result",
                              make_full_hook(layer, circuit_at_layer)))
        with torch.no_grad():
            full_logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
        full_last = full_logits[0, clean_seq_len - 1]
        full_corrupt_lds.append(
            (full_last[p["correct_id"]] - full_last[p["distractor_id"]]).item()
        )

        del clean_cache, corrupt_cache

    full_circuit_effect = np.mean(baseline_lds) - np.mean(full_corrupt_lds)
    print(f"  Baseline LD: {np.mean(baseline_lds):.4f}", flush=True)
    print(f"  Full corrupt LD: {np.mean(full_corrupt_lds):.4f}", flush=True)
    print(f"  Full circuit effect: {full_circuit_effect:.4f}", flush=True)

    # ── Phase 2: per-pathway path patching ──
    print(f"\n[{datetime.now().isoformat()}] Phase 2: per-pathway path patching...", flush=True)
    pathway_results = {}

    for pw_name, senders, receivers in PATHWAYS:
        valid_receivers = [(l, h) for l, h in receivers
                          if l > max(sl for sl, _ in senders)]
        if not valid_receivers:
            print(f"  {pw_name}: no valid receivers (same or earlier layer), skipping", flush=True)
            continue

        pw_lds = []
        for i in tqdm(range(n), desc=f"Pathway: {pw_name}"):
            p = valid[i]
            alt_p = valid[shuffle_idx[i]]
            clean_tokens = model.to_tokens(p["text"], prepend_bos=True).to(device)
            corrupt_tokens = model.to_tokens(alt_p["text"], prepend_bos=True).to(device)
            clean_tokens, corrupt_tokens = pad_to_match(clean_tokens, corrupt_tokens)
            clean_seq_len = model.to_tokens(p["text"], prepend_bos=True).shape[1]

            clean_logits, clean_cache = cache_hook_results(model, clean_tokens, n_layers)
            _, corrupt_cache = cache_hook_results(model, corrupt_tokens, n_layers)

            patched_logits = path_patch_pathway(
                model, clean_tokens, clean_cache, corrupt_cache,
                senders, valid_receivers, n_layers,
            )

            last = patched_logits[0, clean_seq_len - 1]
            pw_lds.append((last[p["correct_id"]] - last[p["distractor_id"]]).item())

            del clean_cache, corrupt_cache

        pw_effects = [b - p for b, p in zip(baseline_lds, pw_lds)]
        pw_recoveries = []
        for pe, fc in zip(pw_effects,
                          [b - f for b, f in zip(baseline_lds, full_corrupt_lds)]):
            if abs(fc) > 0.01:
                pw_recoveries.append(pe / fc)

        pathway_results[pw_name] = {
            "senders": [[l, h] for l, h in senders],
            "receivers": [[l, h] for l, h in valid_receivers],
            "per_prompt_ld": pw_lds,
            "per_prompt_effect": pw_effects,
            "effect": bootstrap_ci(pw_effects),
            "per_prompt_recovery": pw_recoveries,
            "recovery": bootstrap_ci(pw_recoveries) if len(pw_recoveries) > 10 else None,
        }
        rec = pathway_results[pw_name]["recovery"]
        if rec:
            print(f"  {pw_name}: recovery={rec['mean']:.3f} "
                  f"[{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]", flush=True)

    with open(OUT_DIR / "pathway_results.json", "w") as f:
        json.dump(pathway_results, f, indent=2)
    print(f"  Saved pathway_results.json", flush=True)

    # ── Phase 3: per-edge path patching ──
    print(f"\n[{datetime.now().isoformat()}] Phase 3: per-edge path patching "
          f"({len(edges)} edges x {n} prompts)...", flush=True)

    edge_results = {}
    for sl, sh, rl, rh in edges:
        edge_results[f"L{sl}H{sh}->L{rl}H{rh}"] = []

    for i in tqdm(range(n), desc="Per-edge"):
        p = valid[i]
        alt_p = valid[shuffle_idx[i]]
        clean_tokens = model.to_tokens(p["text"], prepend_bos=True).to(device)
        corrupt_tokens = model.to_tokens(alt_p["text"], prepend_bos=True).to(device)
        clean_tokens, corrupt_tokens = pad_to_match(clean_tokens, corrupt_tokens)
        clean_seq_len = model.to_tokens(p["text"], prepend_bos=True).shape[1]

        clean_logits, clean_cache = cache_hook_results(model, clean_tokens, n_layers)
        _, corrupt_cache = cache_hook_results(model, corrupt_tokens, n_layers)

        for esl, esh, erl, erh in edges:
            patched_logits = path_patch_single_edge(
                model, clean_tokens, clean_cache, corrupt_cache,
                esl, esh, erl, erh, n_layers,
            )
            last = patched_logits[0, clean_seq_len - 1]
            patched_ld = (last[p["correct_id"]] - last[p["distractor_id"]]).item()
            effect = baseline_lds[i] - patched_ld
            edge_results[f"L{esl}H{esh}->L{erl}H{erh}"].append(effect)

        del clean_cache, corrupt_cache

    edge_summary = {}
    for edge_key, effects in edge_results.items():
        edge_summary[edge_key] = {
            "per_prompt_effect": effects,
            "bootstrap": bootstrap_ci(effects),
        }

    with open(OUT_DIR / "edge_results.json", "w") as f:
        json.dump(edge_summary, f, indent=2)
    print(f"  Saved edge_results.json", flush=True)

    # ── Summary ──
    print(f"\n[{datetime.now().isoformat()}] === SUMMARY ===\n")

    print("Per-pathway recovery:")
    for pw_name in pathway_results:
        rec = pathway_results[pw_name].get("recovery")
        if rec:
            print(f"  {pw_name}: {rec['mean']:.1%} [{rec['ci_lo']:.1%}, {rec['ci_hi']:.1%}]")

    print(f"\nTop 20 edges by absolute effect:")
    ranked = sorted(edge_summary.items(),
                    key=lambda x: abs(x[1]["bootstrap"]["mean"]), reverse=True)
    for edge_key, data in ranked[:20]:
        b = data["bootstrap"]
        print(f"  {edge_key}: {b['mean']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")

    # Aggregate edges by pathway
    print(f"\nEdge effects aggregated by pathway:")
    for pw_name, senders, receivers in PATHWAYS:
        sender_set = set(senders)
        receiver_set = set(receivers)
        pw_edges = [
            (k, v) for k, v in edge_summary.items()
            if any(f"L{sl}H{sh}->" in k for sl, sh in sender_set)
            and any(f"->L{rl}H{rh}" in k for rl, rh in receiver_set)
        ]
        if pw_edges:
            total_effect = sum(v["bootstrap"]["mean"] for _, v in pw_edges)
            print(f"  {pw_name}: {len(pw_edges)} edges, "
                  f"sum of effects = {total_effect:+.4f}")

    combined = {
        "timestamp": timestamp,
        "device": device,
        "n_prompts": n,
        "n_edges": len(edges),
        "baseline_ld": bootstrap_ci(baseline_lds),
        "full_corrupt_ld": bootstrap_ci(full_corrupt_lds),
        "full_circuit_effect": float(full_circuit_effect),
        "pathway_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_prompt_ld"
                and kk != "per_prompt_effect" and kk != "per_prompt_recovery"}
            for k, v in pathway_results.items()
        },
        "top_20_edges": [
            {"edge": k, **v["bootstrap"]}
            for k, v in ranked[:20]
        ],
    }
    with open(OUT_DIR / f"summary_{timestamp}.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n[{datetime.now().isoformat()}] Done. Results in {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_all(args.device)
