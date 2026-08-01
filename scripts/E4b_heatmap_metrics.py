"""E4b: Simple heatmap metrics for RTI circuit head identification.

Computes 2-3 interpretable metrics directly from the W_E @ W_OV @ W_U
logit matrix for all 144 GPT-2 Small heads, then checks whether simple
thresholds recover the visually-discovered copier heads.

Pre-registered in paper/prereg/E4b_heatmap_metrics.md.

Usage:
  uv run python paper/E4b_heatmap_metrics.py
  uv run python paper/E4b_heatmap_metrics.py --top-k 100
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

OUT_DIR = Path(__file__).resolve().parent / "paper_numbers" / "E4b_heatmap_metrics"

RTI_CIRCUIT = {
    "backbone": [(0, 8), (0, 9), (0, 11)],
    "detector": [(4, 11)],
    "copier": [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
    "readout": [(10, 11), (11, 9), (11, 11)],
}

ALL_CIRCUIT = set()
TIER_MAP = {}
for tier, heads in RTI_CIRCUIT.items():
    for h in heads:
        ALL_CIRCUIT.add((h[0], h[1]))
        TIER_MAP[(h[0], h[1])] = tier


def compute_heatmap_metrics(model, top_k=50):
    """Compute diagonal score, vertical band score, and logit effective rank
    for all attention heads from the W_E @ W_OV @ W_U logit matrix."""

    W_E = model.W_E  # (d_vocab, d_model)
    W_U = model.W_U  # (d_model, d_vocab)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = {}

    for layer in tqdm(range(n_layers), desc="Layers"):
        for head in range(n_heads):
            W_V = model.W_V[layer, head]  # (d_model, d_head)
            W_O = model.W_O[layer, head]  # (d_head, d_model)

            # Logit matrix: W_E[:k] @ W_V @ W_O @ W_U[:, :k]
            # Shape: (top_k, top_k)
            # Entry [i,j]: "when attending to token j, contribution to logit for token i"
            with torch.no_grad():
                logit_matrix = W_E[:top_k] @ W_V @ W_O @ W_U[:, :top_k]
                M = logit_matrix.float().cpu().numpy()

            diag = np.diag(M)
            n = M.shape[0]

            # 1. Diagonal score: mean(diag) - mean(offdiag)
            diag_mean = diag.mean()
            offdiag_sum = M.sum() - diag.sum()
            offdiag_mean = offdiag_sum / (n * (n - 1))
            diag_score = float(diag_mean - offdiag_mean)

            # Also: fraction of diagonal entries that are negative
            diag_neg_frac = float((diag < 0).mean())

            # Diag/offdiag ratio
            diag_ratio = float(diag_mean / (abs(offdiag_mean) + 1e-10))

            # 2. Vertical band score: var(column means) / var(all entries)
            col_means = M.mean(axis=0)  # mean across rows for each column (output token)
            row_means = M.mean(axis=1)  # mean across columns for each row (input token)
            overall_var = M.var()
            col_mean_var = col_means.var()
            row_mean_var = row_means.var()
            vertical_band_score = float(col_mean_var / (overall_var + 1e-10))
            horizontal_band_score = float(row_mean_var / (overall_var + 1e-10))

            # 3. Effective rank of the logit matrix
            svs = np.linalg.svd(M, compute_uv=False)
            sv_sum = svs.sum()
            sv_sq_sum = (svs ** 2).sum()
            eff_rank = float(sv_sum ** 2 / (sv_sq_sum + 1e-10))

            # 4. Max absolute column mean (strongest vertical band)
            max_col_mean = float(np.max(np.abs(col_means)))

            # 5. Diagonal negativity: mean(diag) alone (not relative to offdiag)
            diag_mean_raw = float(diag_mean)

            results[(layer, head)] = {
                "head": f"L{layer}H{head}",
                "diag_score": diag_score,
                "diag_mean": diag_mean_raw,
                "offdiag_mean": float(offdiag_mean),
                "diag_ratio": diag_ratio,
                "diag_neg_frac": diag_neg_frac,
                "vertical_band_score": vertical_band_score,
                "horizontal_band_score": horizontal_band_score,
                "effective_rank": eff_rank,
                "max_col_mean_abs": max_col_mean,
            }

    return results


def analyze_separation(results):
    """Check how well simple thresholds separate circuit from non-circuit heads."""

    all_heads = sorted(results.keys())
    metrics = ["diag_score", "diag_mean", "diag_ratio", "diag_neg_frac",
               "vertical_band_score", "effective_rank", "max_col_mean_abs"]

    copier_set = set(RTI_CIRCUIT["copier"])
    analysis = {}

    for metric in metrics:
        vals = [(h, results[h][metric]) for h in all_heads]
        vals.sort(key=lambda x: x[1])

        # Per-tier stats
        tier_stats = {}
        for tier, heads in RTI_CIRCUIT.items():
            tier_vals = [results[tuple(h)][metric] for h in heads]
            tier_stats[tier] = {
                "mean": float(np.mean(tier_vals)),
                "std": float(np.std(tier_vals)),
                "min": float(np.min(tier_vals)),
                "max": float(np.max(tier_vals)),
                "values": {f"L{h[0]}H{h[1]}": float(results[tuple(h)][metric]) for h in heads},
            }

        non_circuit_vals = [results[h][metric] for h in all_heads if h not in ALL_CIRCUIT]
        tier_stats["non_circuit"] = {
            "mean": float(np.mean(non_circuit_vals)),
            "std": float(np.std(non_circuit_vals)),
            "n": len(non_circuit_vals),
        }

        # Ranked list (for checking where circuit heads fall)
        ranked = []
        for h, v in vals:
            ranked.append({
                "head": f"L{h[0]}H{h[1]}",
                "value": v,
                "in_circuit": h in ALL_CIRCUIT,
                "tier": TIER_MAP.get(h),
            })

        # Threshold analysis: try catching heads at each extreme
        # For diag_score: most negative = most "negative diagonal"
        # For vertical_band_score: highest = most vertical banding
        for direction in ["bottom", "top"]:
            if direction == "bottom":
                ordered = vals  # ascending
            else:
                ordered = vals[::-1]  # descending

            threshold_results = []
            for k in [5, 8, 10, 15, 20, 25]:
                topk = {h for h, _ in ordered[:k]}
                tp_all = len(topk & ALL_CIRCUIT)
                tp_copier = len(topk & copier_set)
                fp = len(topk - ALL_CIRCUIT)
                threshold_results.append({
                    "k": k,
                    "tp_circuit": tp_all,
                    "tp_copier": tp_copier,
                    "fp": fp,
                    "recall_circuit": tp_all / 15,
                    "recall_copier": tp_copier / 8,
                    "precision": tp_all / k,
                })

            tier_stats[f"threshold_{direction}"] = threshold_results

        analysis[metric] = {
            "tier_stats": tier_stats,
            "ranked": ranked,
        }

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().isoformat()}] E4b: Heatmap metrics for RTI circuit", flush=True)
    print(f"  top_k={args.top_k}", flush=True)

    print(f"[{datetime.now().isoformat()}] Loading GPT-2 Small...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=args.device)
    model.eval()

    print(f"[{datetime.now().isoformat()}] Computing heatmap metrics...", flush=True)
    results = compute_heatmap_metrics(model, top_k=args.top_k)

    # Save raw metrics
    raw = {f"L{l}H{h}": v for (l, h), v in results.items()}
    with open(OUT_DIR / f"heatmap_metrics_k{args.top_k}.json", "w") as f:
        json.dump(raw, f, indent=2)

    print(f"[{datetime.now().isoformat()}] Analyzing separation...", flush=True)
    analysis = analyze_separation(results)

    with open(OUT_DIR / f"separation_analysis_k{args.top_k}.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print(f"\n=== SUMMARY (top_k={args.top_k}) ===", flush=True)

    for metric in ["diag_score", "vertical_band_score", "effective_rank"]:
        a = analysis[metric]
        print(f"\n--- {metric} ---", flush=True)
        for tier in ["backbone", "detector", "copier", "readout", "non_circuit"]:
            s = a["tier_stats"][tier]
            if tier == "non_circuit":
                print(f"  {tier:12s}: mean={s['mean']:+.4f} +/- {s['std']:.4f} (n={s['n']})", flush=True)
            else:
                print(f"  {tier:12s}: mean={s['mean']:+.4f} +/- {s['std']:.4f} [{s['min']:+.4f}, {s['max']:+.4f}]", flush=True)

        # Best threshold direction for this metric
        for direction in ["bottom", "top"]:
            print(f"  threshold ({direction}):", flush=True)
            for t in a["tier_stats"][f"threshold_{direction}"]:
                if t["k"] in [8, 15, 20]:
                    print(f"    k={t['k']:2d}: copier={t['tp_copier']}/8, circuit={t['tp_circuit']}/15, FP={t['fp']}", flush=True)

    # Ranked lists for key metrics
    print(f"\n=== RANKED: diag_score (ascending = most negative diagonal) ===", flush=True)
    for entry in analysis["diag_score"]["ranked"][:25]:
        marker = f" <<< {entry['tier']}" if entry["in_circuit"] else ""
        print(f"  {entry['head']:8s}: {entry['value']:+.4f}{marker}", flush=True)

    print(f"\n=== RANKED: vertical_band_score (descending = most banding) ===", flush=True)
    ranked_vb = sorted(analysis["vertical_band_score"]["ranked"], key=lambda x: x["value"], reverse=True)
    for entry in ranked_vb[:25]:
        marker = f" <<< {entry['tier']}" if entry["in_circuit"] else ""
        print(f"  {entry['head']:8s}: {entry['value']:.4f}{marker}", flush=True)

    print(f"\n[{datetime.now().isoformat()}] Results saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
