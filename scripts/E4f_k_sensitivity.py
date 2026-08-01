"""E4f: OV diagonal score sensitivity to k.

Computes OV diagonal scores for all 144 GPT-2 Small heads at
k = 50, 100, 500, 1000. Reports rank correlations and sign stability.

Usage:
  uv run python paper/E4f_k_sensitivity.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformer_lens import HookedTransformer

OUT_DIR = Path(__file__).resolve().parent / "paper_numbers" / "E4f_k_sensitivity"

RTI_CIRCUIT = {
    "Backbone": [(0, 8), (0, 9), (0, 11)],
    "Detector": [(4, 11)],
    "Copier": [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
    "Readout": [(10, 11), (11, 9), (11, 11)],
}

COPIER_SET = set(RTI_CIRCUIT["Copier"])
CIRCUIT_SET = set()
for heads in RTI_CIRCUIT.values():
    for h in heads:
        CIRCUIT_SET.add(tuple(h))

K_VALUES = [50, 100, 500, 1000]


def compute_diagonal_scores(model, k):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_E = model.W_E
    W_U = model.W_U

    scores = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            W_V = model.W_V[layer, head]
            W_O = model.W_O[layer, head]
            with torch.no_grad():
                M = W_E[:k] @ W_V @ W_O @ W_U[:, :k]
                M = M.float().cpu().numpy()

            diag = np.diag(M)
            mask = ~np.eye(k, dtype=bool)
            offdiag = M[mask]
            scores[(layer, head)] = float(np.mean(diag) - np.mean(offdiag))

    return scores


def spearman_rank_correlation(x, y):
    rho, pval = spearmanr(x, y)
    return float(rho), float(pval)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] E4f: k sensitivity analysis", flush=True)

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()

    all_scores = {}
    for k in tqdm(K_VALUES, desc="k values"):
        print(f"[{datetime.now().isoformat()}] Computing k={k}...", flush=True)
        scores = compute_diagonal_scores(model, k)
        all_scores[k] = scores

    head_keys = sorted(all_scores[K_VALUES[0]].keys())

    results = {
        "k_values": K_VALUES,
        "per_k": {},
        "rank_correlations": {},
        "copier_sign_stability": {},
        "copier_rank_stability": {},
    }

    for k in K_VALUES:
        scores = all_scores[k]
        values = [scores[h] for h in head_keys]
        sorted_indices = np.argsort(values)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1

        per_head = {}
        for i, h in enumerate(head_keys):
            name = f"L{h[0]}H{h[1]}"
            per_head[name] = {
                "score": scores[h],
                "rank": int(ranks[i]),
                "is_copier": h in COPIER_SET,
                "is_circuit": h in CIRCUIT_SET,
            }
        results["per_k"][str(k)] = per_head

        copier_scores = [scores[h] for h in head_keys if h in COPIER_SET]
        copier_ranks = [int(ranks[i]) for i, h in enumerate(head_keys) if h in COPIER_SET]
        results["copier_sign_stability"][str(k)] = {
            "all_positive": all(s > 0 for s in copier_scores),
            "n_positive": sum(1 for s in copier_scores if s > 0),
            "n_negative": sum(1 for s in copier_scores if s < 0),
            "min_score": min(copier_scores),
            "max_score": max(copier_scores),
            "mean_score": float(np.mean(copier_scores)),
        }
        results["copier_rank_stability"][str(k)] = {
            "ranks": copier_ranks,
            "min_rank": min(copier_ranks),
            "max_rank": max(copier_ranks),
            "mean_rank": float(np.mean(copier_ranks)),
            "all_in_top_30": all(r <= 30 for r in copier_ranks),
        }

    base_k = K_VALUES[0]
    base_values = [all_scores[base_k][h] for h in head_keys]
    for k in K_VALUES[1:]:
        other_values = [all_scores[k][h] for h in head_keys]
        rho, pval = spearman_rank_correlation(base_values, other_values)
        results["rank_correlations"][f"k{base_k}_vs_k{k}"] = {
            "spearman_rho": rho,
            "p_value": pval,
        }

    with open(OUT_DIR / "k_sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== RESULTS ===", flush=True)
    print(f"\nRank correlations (vs k={base_k}):", flush=True)
    for key, val in results["rank_correlations"].items():
        print(f"  {key}: rho={val['spearman_rho']:.4f} (p={val['p_value']:.2e})", flush=True)

    print(f"\nCopier sign stability:", flush=True)
    for k in K_VALUES:
        s = results["copier_sign_stability"][str(k)]
        print(f"  k={k}: {s['n_positive']}/8 positive, min={s['min_score']:.4f}, "
              f"max={s['max_score']:.4f}, mean={s['mean_score']:.4f}", flush=True)

    print(f"\nCopier rank stability:", flush=True)
    for k in K_VALUES:
        r = results["copier_rank_stability"][str(k)]
        print(f"  k={k}: ranks={r['ranks']}, "
              f"range=[{r['min_rank']}, {r['max_rank']}], "
              f"all_in_top_30={r['all_in_top_30']}", flush=True)

    print(f"\n[{datetime.now().isoformat()}] Saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
