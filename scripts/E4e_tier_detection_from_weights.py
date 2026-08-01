"""E4e: Systematic tier detection from weight-space metrics alone.

For all 144 GPT-2 Small heads, computes:
  1. OV diagonal score: mean(diag) - mean(offdiag) of W_E[:k] @ W_V @ W_O @ W_U[:, :k]
  2. QK same-token score: mean(diag) - mean(offdiag) of W_E[:k] @ W_Q @ W_K^T @ W_E[:k]^T
  3. OV Frobenius norm: ||W_V @ W_O||_F (overall magnitude)
  4. QK Frobenius norm: ||W_Q @ W_K^T||_F

Then checks whether simple rules on (OV_diag, QK_same, layer) separate the
four RTI circuit tiers: Backbone, Detector, Copier, Readout.

Usage:
  uv run python paper/E4e_tier_detection_from_weights.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

OUT_DIR = Path(__file__).resolve().parent / "paper_numbers" / "E4e_tier_detection"

RTI_CIRCUIT = {
    "backbone": [(0, 8), (0, 9), (0, 11)],
    "detector": [(4, 11)],
    "copier": [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)],
    "readout": [(10, 11), (11, 9), (11, 11)],
}

TIER_MAP = {}
ALL_CIRCUIT = set()
for tier, heads in RTI_CIRCUIT.items():
    for h in heads:
        TIER_MAP[tuple(h)] = tier
        ALL_CIRCUIT.add(tuple(h))


def compute_all_metrics(model, top_k=50):
    W_E = model.W_E[:top_k]  # (k, d_model)
    W_U = model.W_U[:, :top_k]  # (d_model, k)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = {}

    for layer in tqdm(range(n_layers), desc="Layers"):
        for head in range(n_heads):
            W_Q = model.W_Q[layer, head]  # (d_model, d_head)
            W_K = model.W_K[layer, head]  # (d_model, d_head)
            W_V = model.W_V[layer, head]  # (d_model, d_head)
            W_O = model.W_O[layer, head]  # (d_head, d_model)

            with torch.no_grad():
                # OV logit matrix: (k, k)
                OV = W_E @ W_V @ W_O @ W_U
                OV = OV.float().cpu().numpy()

                # QK token matrix: (k, k)
                # Entry [i,j] = how much token i wants to attend to token j
                QK = W_E @ W_Q @ W_K.T @ W_E.T
                QK = QK.float().cpu().numpy()

                # Frobenius norms of the raw circuits
                ov_fro = float(torch.norm(W_V @ W_O, p='fro').item())
                qk_fro = float(torch.norm(W_Q @ W_K.T, p='fro').item())

            k = OV.shape[0]

            # OV diagonal score
            ov_diag = np.diag(OV)
            ov_diag_mean = float(ov_diag.mean())
            ov_offdiag_mean = float((OV.sum() - ov_diag.sum()) / (k * (k - 1)))
            ov_diag_score = ov_diag_mean - ov_offdiag_mean

            # QK same-token score
            qk_diag = np.diag(QK)
            qk_diag_mean = float(qk_diag.mean())
            qk_offdiag_mean = float((QK.sum() - qk_diag.sum()) / (k * (k - 1)))
            qk_same_score = qk_diag_mean - qk_offdiag_mean

            # OV column variance (vertical band score)
            ov_col_means = OV.mean(axis=0)
            ov_col_var = float(ov_col_means.var())
            ov_overall_var = float(OV.var())
            ov_vertical_band = ov_col_var / (ov_overall_var + 1e-10)

            # QK row variance (does query token identity matter?)
            qk_row_means = QK.mean(axis=1)
            qk_row_var = float(qk_row_means.var())
            qk_overall_var = float(QK.var())
            qk_query_specificity = qk_row_var / (qk_overall_var + 1e-10)

            # QK column variance (does key token identity matter?)
            qk_col_means = QK.mean(axis=0)
            qk_col_var = float(qk_col_means.var())
            qk_key_specificity = qk_col_var / (qk_overall_var + 1e-10)

            results[(layer, head)] = {
                "head": f"L{layer}H{head}",
                "layer": layer,
                "ov_diag_score": float(ov_diag_score),
                "ov_diag_mean": ov_diag_mean,
                "ov_offdiag_mean": ov_offdiag_mean,
                "ov_fro_norm": ov_fro,
                "ov_vertical_band": float(ov_vertical_band),
                "qk_same_score": float(qk_same_score),
                "qk_diag_mean": qk_diag_mean,
                "qk_offdiag_mean": qk_offdiag_mean,
                "qk_fro_norm": qk_fro,
                "qk_query_specificity": float(qk_query_specificity),
                "qk_key_specificity": float(qk_key_specificity),
                "tier": TIER_MAP.get((layer, head)),
                "in_circuit": (layer, head) in ALL_CIRCUIT,
            }

    return results


def compute_auroc(labels, scores):
    """Manual AUROC computation — no sklearn needed."""
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def compute_best_f1(labels, scores):
    """Find threshold maximizing F1."""
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    best_f1 = 0.0
    best_thresh = 0.0
    best_tp = 0
    best_fp = 0
    best_fn = n_pos
    tp = 0
    fp = 0
    for i, (score, label) in enumerate(pairs):
        if label:
            tp += 1
        else:
            fp += 1
        fn = n_pos - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / n_pos if n_pos > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = score
            best_tp = tp
            best_fp = fp
            best_fn = fn
    return best_f1, best_thresh, best_tp, best_fp, best_fn


def analyze_tier_separation(results):
    """Check how well metric combinations separate circuit tiers."""

    tiers = ["backbone", "detector", "copier", "readout"]
    metrics = ["ov_diag_score", "qk_same_score", "ov_fro_norm", "qk_fro_norm",
               "ov_vertical_band", "qk_query_specificity", "qk_key_specificity"]

    tier_stats = {}
    for metric in metrics:
        tier_stats[metric] = {}
        for tier in tiers:
            vals = [results[k][metric] for k in results if results[k]["tier"] == tier]
            tier_stats[metric][tier] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": {results[k]["head"]: results[k][metric] for k in results if results[k]["tier"] == tier},
            }
        non_circuit_vals = [results[k][metric] for k in results if not results[k]["in_circuit"]]
        tier_stats[metric]["non_circuit"] = {
            "mean": float(np.mean(non_circuit_vals)),
            "std": float(np.std(non_circuit_vals)),
            "n": len(non_circuit_vals),
        }

    # Per-metric AUROC and F1 for detecting each tier and full circuit
    detection = {}
    all_keys = sorted(results.keys())

    for target in tiers + ["any_circuit"]:
        detection[target] = {}
        if target == "any_circuit":
            labels = [1 if results[k]["in_circuit"] else 0 for k in all_keys]
        else:
            labels = [1 if results[k]["tier"] == target else 0 for k in all_keys]

        for metric in metrics:
            scores = [results[k][metric] for k in all_keys]

            # Try both directions (high = positive, low = positive)
            auroc_high = compute_auroc(labels, scores)
            auroc_low = compute_auroc(labels, [-s for s in scores])
            f1_high, thresh_h, tp_h, fp_h, fn_h = compute_best_f1(labels, scores)
            f1_low, thresh_l, tp_l, fp_l, fn_l = compute_best_f1(labels, [-s for s in scores])

            if auroc_high >= auroc_low:
                best_auroc = auroc_high
                best_dir = "high"
                best_f1, best_thresh = f1_high, thresh_h
                best_tp, best_fp, best_fn = tp_h, fp_h, fn_h
            else:
                best_auroc = auroc_low
                best_dir = "low"
                best_f1, best_thresh = f1_low, -thresh_l
                best_tp, best_fp, best_fn = tp_l, fp_l, fn_l

            detection[target][metric] = {
                "auroc": float(best_auroc),
                "direction": best_dir,
                "best_f1": float(best_f1),
                "best_threshold": float(best_thresh),
                "tp": best_tp,
                "fp": best_fp,
                "fn": best_fn,
                "n_positive": sum(labels),
            }

    # Multi-metric: OV_diag + QK_same combined score for each tier
    for target in tiers + ["any_circuit"]:
        if target == "any_circuit":
            labels = [1 if results[k]["in_circuit"] else 0 for k in all_keys]
        else:
            labels = [1 if results[k]["tier"] == target else 0 for k in all_keys]

        # Simple combined: |OV_diag| + |QK_same|
        combined_abs = [abs(results[k]["ov_diag_score"]) + abs(results[k]["qk_same_score"]) for k in all_keys]
        auroc_combined = compute_auroc(labels, combined_abs)
        f1_combined, thresh_c, tp_c, fp_c, fn_c = compute_best_f1(labels, combined_abs)

        detection[target]["combined_abs_ov_qk"] = {
            "auroc": float(auroc_combined),
            "direction": "high",
            "best_f1": float(f1_combined),
            "best_threshold": float(thresh_c),
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "n_positive": sum(labels),
        }

    # Per-head detection results
    rules_results = {}
    for key, r in results.items():
        rules_results[r["head"]] = {
            "actual": r["tier"],
            "layer": r["layer"],
            "ov_diag_score": r["ov_diag_score"],
            "qk_same_score": r["qk_same_score"],
            "ov_fro_norm": r["ov_fro_norm"],
            "qk_fro_norm": r["qk_fro_norm"],
            "in_circuit": r["in_circuit"],
        }

    return tier_stats, rules_results, detection


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] E4e: Tier detection from weights", flush=True)

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()

    print(f"[{datetime.now().isoformat()}] Computing OV + QK metrics for 144 heads...", flush=True)
    results = compute_all_metrics(model, top_k=50)

    # Save raw metrics
    raw = {f"L{l}H{h}": v for (l, h), v in results.items()}
    with open(OUT_DIR / "all_head_metrics.json", "w") as f:
        json.dump(raw, f, indent=2)

    print(f"[{datetime.now().isoformat()}] Analyzing tier separation...", flush=True)
    tier_stats, rules_results, detection = analyze_tier_separation(results)

    with open(OUT_DIR / "tier_separation.json", "w") as f:
        json.dump(tier_stats, f, indent=2)

    with open(OUT_DIR / "rules_detection.json", "w") as f:
        json.dump(rules_results, f, indent=2)

    with open(OUT_DIR / "detection_auroc_f1.json", "w") as f:
        json.dump(detection, f, indent=2)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"TIER SEPARATION ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    for metric in ["ov_diag_score", "qk_same_score", "ov_fro_norm", "qk_fro_norm",
                    "ov_vertical_band", "qk_query_specificity", "qk_key_specificity"]:
        print(f"\n--- {metric} ---", flush=True)
        for tier in ["backbone", "detector", "copier", "readout", "non_circuit"]:
            s = tier_stats[metric][tier]
            if tier == "non_circuit":
                print(f"  {tier:12s}: {s['mean']:+8.4f} +/- {s['std']:6.4f}  (n={s['n']})", flush=True)
            else:
                print(f"  {tier:12s}: {s['mean']:+8.4f} +/- {s['std']:6.4f}  [{s['min']:+8.4f}, {s['max']:+8.4f}]", flush=True)
                for name, val in sorted(s["values"].items()):
                    print(f"    {name}: {val:+.4f}", flush=True)

    # 2D scatter data: OV diag vs QK same
    print(f"\n{'='*70}", flush=True)
    print(f"OV_DIAG vs QK_SAME scatter (for all 15 circuit heads)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Head':>8s}  {'Tier':>10s}  {'OV_diag':>10s}  {'QK_same':>10s}  {'Layer':>5s}", flush=True)
    for key in sorted(results.keys()):
        r = results[key]
        if r["in_circuit"]:
            print(f"{r['head']:>8s}  {r['tier']:>10s}  {r['ov_diag_score']:+10.4f}  {r['qk_same_score']:+10.4f}  {r['layer']:5d}", flush=True)

    # Also print the top 20 non-circuit heads by absolute OV diag score
    print(f"\n{'='*70}", flush=True)
    print(f"TOP 20 NON-CIRCUIT HEADS BY |OV_DIAG|", flush=True)
    print(f"{'='*70}", flush=True)
    non_circuit = [(k, r) for k, r in results.items() if not r["in_circuit"]]
    non_circuit.sort(key=lambda x: abs(x[1]["ov_diag_score"]), reverse=True)
    print(f"{'Head':>8s}  {'OV_diag':>10s}  {'QK_same':>10s}  {'Layer':>5s}", flush=True)
    for key, r in non_circuit[:20]:
        print(f"{r['head']:>8s}  {r['ov_diag_score']:+10.4f}  {r['qk_same_score']:+10.4f}  {r['layer']:5d}", flush=True)

    # Detection AUROC/F1 table
    print(f"\n{'='*70}", flush=True)
    print(f"DETECTION AUROC / BEST F1 BY METRIC", flush=True)
    print(f"{'='*70}", flush=True)

    det_metrics = ["ov_diag_score", "qk_same_score", "ov_fro_norm", "qk_fro_norm",
                   "ov_vertical_band", "qk_query_specificity", "qk_key_specificity",
                   "combined_abs_ov_qk"]

    for target in ["backbone", "detector", "copier", "readout", "any_circuit"]:
        print(f"\n--- {target} (n={detection[target]['ov_diag_score']['n_positive']}) ---", flush=True)
        print(f"  {'Metric':>25s}  {'AUROC':>6s}  {'Dir':>4s}  {'F1':>5s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}", flush=True)
        for metric in det_metrics:
            d = detection[target][metric]
            print(f"  {metric:>25s}  {d['auroc']:6.3f}  {d['direction']:>4s}  {d['best_f1']:5.3f}  {d['tp']:3d}  {d['fp']:3d}  {d['fn']:3d}", flush=True)

    print(f"\n[{datetime.now().isoformat()}] Saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
