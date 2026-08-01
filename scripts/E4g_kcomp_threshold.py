"""E4g: K-composition threshold sensitivity.

Computes K-composition scores for all directed head pairs in GPT-2 Small.
Reports gap between circuit and non-circuit edges, threshold sensitivity.

Usage:
  uv run python paper/E4g_kcomp_threshold.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

OUT_DIR = Path(__file__).resolve().parent / "paper_numbers" / "E4g_kcomp_threshold"

RTI_CIRCUIT = {
    "Backbone": [(0, 8), (0, 9), (0, 11)],
    "Detector": [(4, 11)],
    "Copier": [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
    "Readout": [(10, 11), (11, 9), (11, 11)],
}

TIER_MAP = {}
for tier, heads in RTI_CIRCUIT.items():
    for h in heads:
        TIER_MAP[tuple(h)] = tier

CIRCUIT_SET = set(TIER_MAP.keys())

PATHWAYS = {
    "backbone_to_detector": ("Backbone", "Detector"),
    "backbone_to_copier": ("Backbone", "Copier"),
    "backbone_to_readout": ("Backbone", "Readout"),
    "detector_to_copier": ("Detector", "Copier"),
    "detector_to_readout": ("Detector", "Readout"),
    "copier_to_readout": ("Copier", "Readout"),
}


def compute_all_kcomp(model):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = {}

    for sender_layer in tqdm(range(n_layers), desc="K-comp sender layers"):
        for sender_head in range(n_heads):
            W_O = model.W_O[sender_layer, sender_head]
            for recv_layer in range(sender_layer + 1, n_layers):
                for recv_head in range(n_heads):
                    W_K = model.W_K[recv_layer, recv_head]
                    with torch.no_grad():
                        comp = W_O @ W_K
                        score = float(torch.norm(comp, p="fro").item())
                    scores[((sender_layer, sender_head), (recv_layer, recv_head))] = score

    return scores


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] E4g: K-comp threshold sensitivity", flush=True)

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()

    print(f"[{datetime.now().isoformat()}] Computing all K-comp scores...", flush=True)
    all_scores = compute_all_kcomp(model)

    all_values = list(all_scores.values())
    background = {
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "median": float(np.median(all_values)),
        "p90": float(np.percentile(all_values, 90)),
        "p95": float(np.percentile(all_values, 95)),
        "p99": float(np.percentile(all_values, 99)),
        "n_edges": len(all_values),
    }

    pathway_results = {}
    for pathway_name, (sender_tier, recv_tier) in PATHWAYS.items():
        circuit_scores = []
        noncircuit_scores = []

        sender_heads = RTI_CIRCUIT[sender_tier]
        recv_heads = RTI_CIRCUIT[recv_tier]

        for sender in sender_heads:
            for receiver in recv_heads:
                key = (tuple(sender), tuple(receiver))
                if key in all_scores:
                    circuit_scores.append({
                        "sender": f"L{sender[0]}H{sender[1]}",
                        "receiver": f"L{receiver[0]}H{receiver[1]}",
                        "score": all_scores[key],
                    })

        recv_head_set = set(map(tuple, recv_heads))
        sender_head_set = set(map(tuple, sender_heads))
        sender_layers = set(s[0] for s in sender_heads)
        n_heads = model.cfg.n_heads
        for s_layer in sender_layers:
            for s_head in range(n_heads):
                if (s_layer, s_head) in sender_head_set:
                    continue
                for receiver in recv_heads:
                    key = ((s_layer, s_head), tuple(receiver))
                    if key in all_scores:
                        noncircuit_scores.append(all_scores[key])

        circuit_vals = [e["score"] for e in circuit_scores]
        if circuit_vals and noncircuit_scores:
            gap = min(circuit_vals) - max(noncircuit_scores)
            gap_ratio = min(circuit_vals) / max(noncircuit_scores) if max(noncircuit_scores) > 0 else float("inf")
        else:
            gap = None
            gap_ratio = None

        pathway_results[pathway_name] = {
            "circuit_edges": circuit_scores,
            "circuit_min": min(circuit_vals) if circuit_vals else None,
            "circuit_max": max(circuit_vals) if circuit_vals else None,
            "circuit_mean": float(np.mean(circuit_vals)) if circuit_vals else None,
            "noncircuit_max": max(noncircuit_scores) if noncircuit_scores else None,
            "noncircuit_mean": float(np.mean(noncircuit_scores)) if noncircuit_scores else None,
            "noncircuit_p95": float(np.percentile(noncircuit_scores, 95)) if noncircuit_scores else None,
            "gap": gap,
            "gap_ratio": gap_ratio,
            "n_circuit": len(circuit_vals),
            "n_noncircuit": len(noncircuit_scores),
        }

    threshold_sensitivity = {}
    for pct in [90, 92, 95, 97, 99]:
        threshold = float(np.percentile(all_values, pct))
        surviving_circuit = []
        surviving_noncircuit = 0
        for (sender, receiver), score in all_scores.items():
            if score >= threshold:
                s_tier = TIER_MAP.get(sender)
                r_tier = TIER_MAP.get(receiver)
                if s_tier and r_tier:
                    surviving_circuit.append({
                        "sender": f"L{sender[0]}H{sender[1]}",
                        "receiver": f"L{receiver[0]}H{receiver[1]}",
                        "pathway": f"{s_tier}_to_{r_tier}",
                        "score": score,
                    })
                else:
                    surviving_noncircuit += 1

        circuit_heads_involved = set()
        for edge in surviving_circuit:
            circuit_heads_involved.add(edge["sender"])
            circuit_heads_involved.add(edge["receiver"])

        threshold_sensitivity[f"P{pct}"] = {
            "threshold": threshold,
            "n_circuit_edges": len(surviving_circuit),
            "n_noncircuit_edges": surviving_noncircuit,
            "circuit_heads_involved": sorted(circuit_heads_involved),
            "n_circuit_heads": len(circuit_heads_involved),
        }

    results = {
        "background": background,
        "pathways": pathway_results,
        "threshold_sensitivity": threshold_sensitivity,
    }

    with open(OUT_DIR / "kcomp_threshold_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== BACKGROUND ===", flush=True)
    for k, v in background.items():
        print(f"  {k}: {v}", flush=True)

    print(f"\n=== PATHWAY GAPS ===", flush=True)
    for name, pw in pathway_results.items():
        if pw["gap"] is not None:
            print(f"  {name}: circuit_min={pw['circuit_min']:.1f}, "
                  f"noncircuit_max={pw['noncircuit_max']:.1f}, "
                  f"gap={pw['gap']:.1f}, ratio={pw['gap_ratio']:.2f}", flush=True)
        else:
            print(f"  {name}: circuit_min={pw['circuit_min']}, n_circuit={pw['n_circuit']}", flush=True)

    print(f"\n=== THRESHOLD SENSITIVITY ===", flush=True)
    for pct_name, ts in threshold_sensitivity.items():
        print(f"  {pct_name} (>{ts['threshold']:.1f}): "
              f"{ts['n_circuit_edges']} circuit edges, "
              f"{ts['n_noncircuit_edges']} noncircuit edges, "
              f"{ts['n_circuit_heads']}/15 circuit heads involved", flush=True)

    print(f"\n[{datetime.now().isoformat()}] Saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
