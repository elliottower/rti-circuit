"""E4: Weight-feature classification of RTI circuit heads.

Pre-registered in paper/prereg/E4_weight_feature_classifier.md (SHA 75e03a4).
Runs all 5 tests and saves results to paper/paper_numbers/E4_weight_classifier/.

Usage:
  uv run python paper/E4_weight_feature_classifier.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from weight_circuits.classify import classify_circuit, auto_roles

OUT_DIR = Path(__file__).resolve().parent / "paper_numbers" / "E4_weight_classifier"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "gpt2-xl-transfer-20260510T051611:v0" / "features_gpt2_small.json"
PREREG_SHA = "75e03a4"

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
        ALL_CIRCUIT.add(f"L{h[0]}H{h[1]}")
        TIER_MAP[f"L{h[0]}H{h[1]}"] = tier

CLASSIFY_DEF = {
    "roles": {t: hs for t, hs in RTI_CIRCUIT.items()},
    "bands": {
        "early": (range(0, 3), ["backbone"]),
        "mid": (range(3, 6), ["detector", "copier"]),
        "mid2": (range(6, 10), ["copier"]),
        "late": (range(10, 12), ["readout"]),
    },
    "pathways": [
        ("backbone", "detector"),
        ("backbone", "copier"),
        ("detector", "copier"),
        ("copier", "readout"),
    ],
}


def load_features():
    with open(FEATURES_PATH) as f:
        feats_raw = json.load(f)
    all_heads = sorted(feats_raw.keys())
    feat_names = sorted(feats_raw[all_heads[0]].keys())
    return feats_raw, all_heads, feat_names


def test1_per_feature_auroc(feats_raw, all_heads, feat_names):
    print(f"[{datetime.now().isoformat()}] Test 1: Per-feature AUROC", flush=True)
    labels = np.array([1 if h in ALL_CIRCUIT else 0 for h in all_heads])
    copier_set = {f"L{l}H{h}" for l, h in RTI_CIRCUIT["copier"]}
    copier_labels = np.array([1 if h in copier_set else 0 for h in all_heads])

    aurocs_full = []
    aurocs_copier = []
    for fname in feat_names:
        vals = np.array([feats_raw[h][fname] for h in all_heads])
        if np.std(vals) < 1e-10:
            continue
        auc = roc_auc_score(labels, vals)
        auc_best = max(auc, 1 - auc)
        direction = "+" if auc > 0.5 else "-"
        aurocs_full.append({"feature": fname, "auroc": auc_best, "auroc_raw": float(auc), "direction": direction})

        auc_c = roc_auc_score(copier_labels, vals)
        auc_c_best = max(auc_c, 1 - auc_c)
        aurocs_copier.append({"feature": fname, "auroc": auc_c_best, "auroc_raw": float(auc_c), "direction": "+" if auc_c > 0.5 else "-"})

    aurocs_full.sort(key=lambda x: x["auroc"], reverse=True)
    aurocs_copier.sort(key=lambda x: x["auroc"], reverse=True)

    median_auroc = float(np.median([x["auroc"] for x in aurocs_full]))

    result = {
        "test": "1_per_feature_auroc",
        "n_features": len(aurocs_full),
        "best_full": aurocs_full[0],
        "top10_full": aurocs_full[:10],
        "median_auroc_full": median_auroc,
        "best_copier": aurocs_copier[0],
        "top10_copier": aurocs_copier[:10],
        "all_full": aurocs_full,
        "all_copier": aurocs_copier,
    }
    print(f"  Best feature (all 15): {aurocs_full[0]['feature']} AUROC={aurocs_full[0]['auroc']:.3f}", flush=True)
    print(f"  Best feature (copier): {aurocs_copier[0]['feature']} AUROC={aurocs_copier[0]['auroc']:.3f}", flush=True)
    print(f"  Median AUROC: {median_auroc:.3f}", flush=True)
    return result


def test2_top5_combined(feats_raw, all_heads, feat_names):
    print(f"[{datetime.now().isoformat()}] Test 2: Top-5 combined", flush=True)
    labels = np.array([1 if h in ALL_CIRCUIT else 0 for h in all_heads])
    X = np.array([[feats_raw[h][f] for f in feat_names] for h in all_heads])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
    fn2i = {f: i for i, f in enumerate(feat_names)}

    aurocs = []
    for fname in feat_names:
        vals = np.array([feats_raw[h][fname] for h in all_heads])
        if np.std(vals) < 1e-10:
            continue
        auc = roc_auc_score(labels, vals)
        aurocs.append((max(auc, 1 - auc), fname, auc))
    aurocs.sort(reverse=True)

    combined = np.zeros(len(all_heads))
    selected_features = []
    for auc_best, fname, auc_raw in aurocs[:5]:
        sign = 1 if auc_raw > 0.5 else -1
        combined += sign * X_s[:, fn2i[fname]]
        selected_features.append({"feature": fname, "auroc": auc_best, "sign": sign})

    auc_combined = roc_auc_score(labels, combined)

    ranked = sorted(zip(all_heads, combined.tolist()), key=lambda x: x[1], reverse=True)
    top_ranked = [{"head": h, "score": s, "in_circuit": h in ALL_CIRCUIT, "tier": TIER_MAP.get(h)} for h, s in ranked[:25]]

    topk_stats = {}
    for k in [10, 15, 20, 25]:
        topk = {h for h, _ in ranked[:k]}
        tp = len(topk & ALL_CIRCUIT)
        fp = len(topk - ALL_CIRCUIT)
        topk_stats[f"top{k}"] = {"tp": tp, "fp": fp, "recall": tp / 15, "precision": tp / k}

    result = {
        "test": "2_top5_combined",
        "auroc": auc_combined,
        "selected_features": selected_features,
        "top25_ranked": top_ranked,
        "topk_stats": topk_stats,
    }
    print(f"  Combined AUROC: {auc_combined:.3f}", flush=True)
    for k, stats in topk_stats.items():
        print(f"  {k}: recall={stats['recall']:.2f}, precision={stats['precision']:.2f}", flush=True)
    return result


def test3_loo_logistic(feats_raw, all_heads, feat_names):
    print(f"[{datetime.now().isoformat()}] Test 3: LOO logistic regression", flush=True)
    labels = np.array([1 if h in ALL_CIRCUIT else 0 for h in all_heads])
    X = np.array([[feats_raw[h][f] for f in feat_names] for h in all_heads])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    clf = LogisticRegression(C=0.01, max_iter=5000, solver="lbfgs")
    loo = LeaveOneOut()
    loo_probs = cross_val_predict(clf, X_s, labels, cv=loo, method="predict_proba")[:, 1]
    auc_loo = roc_auc_score(labels, loo_probs)

    ranked = sorted(zip(all_heads, loo_probs.tolist()), key=lambda x: x[1], reverse=True)
    top_ranked = [{"head": h, "prob": p, "in_circuit": h in ALL_CIRCUIT, "tier": TIER_MAP.get(h)} for h, p in ranked[:25]]

    topk_stats = {}
    for k in [10, 15, 20, 25]:
        topk = {h for h, _ in ranked[:k]}
        tp = len(topk & ALL_CIRCUIT)
        fp = len(topk - ALL_CIRCUIT)
        topk_stats[f"top{k}"] = {"tp": tp, "fp": fp, "recall": tp / 15, "precision": tp / k}

    per_tier_recall = {}
    for tier, heads in RTI_CIRCUIT.items():
        tier_heads = {f"L{l}H{h}" for l, h in heads}
        tier_probs = {h: p for h, p in zip(all_heads, loo_probs.tolist()) if h in tier_heads}
        per_tier_recall[tier] = {
            "heads": {h: p for h, p in tier_probs.items()},
            "mean_prob": float(np.mean(list(tier_probs.values()))),
        }

    result = {
        "test": "3_loo_logistic",
        "C": 0.01,
        "auroc": auc_loo,
        "top25_ranked": top_ranked,
        "topk_stats": topk_stats,
        "per_tier_recall": per_tier_recall,
    }
    print(f"  LOO AUROC: {auc_loo:.3f}", flush=True)
    for tier, info in per_tier_recall.items():
        print(f"  {tier}: mean_prob={info['mean_prob']:.3f}", flush=True)
    return result


def test4_copier_seeded(feats_raw, all_heads, feat_names):
    print(f"[{datetime.now().isoformat()}] Test 4: Copier-seeded threshold", flush=True)
    copier_set = {f"L{l}H{h}" for l, h in RTI_CIRCUIT["copier"]}
    copier_labels = np.array([1 if h in copier_set else 0 for h in all_heads])

    X = np.array([[feats_raw[h][f] for f in feat_names] for h in all_heads])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
    fn2i = {f: i for i, f in enumerate(feat_names)}

    copier_aurocs = []
    for j, fname in enumerate(feat_names):
        vals = X_s[:, j]
        auc = roc_auc_score(copier_labels, vals)
        copier_aurocs.append((max(auc, 1 - auc), fname, j, 1 if auc > 0.5 else -1))
    copier_aurocs.sort(reverse=True)

    seed_score = np.zeros(len(all_heads))
    top5_features = []
    for auc, fname, j, sign in copier_aurocs[:5]:
        seed_score += sign * X_s[:, j]
        top5_features.append({"feature": fname, "auroc": auc, "sign": sign})

    copier_idx = [i for i, h in enumerate(all_heads) if h in copier_set]
    non_copier_idx = [i for i, h in enumerate(all_heads) if h not in copier_set]
    seed_mean = float(np.mean(seed_score[copier_idx]))
    rest_mean = float(np.mean(seed_score[non_copier_idx]))
    threshold = (seed_mean + rest_mean) / 2

    flagged = []
    for h, s in zip(all_heads, seed_score.tolist()):
        if s > threshold:
            flagged.append({"head": h, "score": s, "in_circuit": h in ALL_CIRCUIT, "tier": TIER_MAP.get(h)})
    flagged.sort(key=lambda x: x["score"], reverse=True)

    flagged_set = {f["head"] for f in flagged}
    tp = len(flagged_set & ALL_CIRCUIT)
    fp = len(flagged_set - ALL_CIRCUIT)

    per_tier = {}
    for tier, heads in RTI_CIRCUIT.items():
        tier_heads = {f"L{l}H{h}" for l, h in heads}
        tier_tp = len(flagged_set & tier_heads)
        per_tier[tier] = {"recovered": tier_tp, "total": len(heads), "recall": tier_tp / len(heads)}

    result = {
        "test": "4_copier_seeded",
        "top5_copier_features": top5_features,
        "seed_mean": seed_mean,
        "rest_mean": rest_mean,
        "threshold": threshold,
        "n_flagged": len(flagged),
        "flagged_heads": flagged,
        "tp": tp,
        "fp": fp,
        "total_recall": tp / 15,
        "precision": tp / len(flagged) if flagged else 0,
        "per_tier": per_tier,
    }
    print(f"  Flagged: {len(flagged)} heads, TP={tp}, FP={fp}", flush=True)
    print(f"  Recall: {tp}/15 = {tp/15:.2f}, Precision: {tp/len(flagged):.2f}", flush=True)
    for tier, info in per_tier.items():
        print(f"  {tier}: {info['recovered']}/{info['total']}", flush=True)
    return result


def test5_bootstrap_greedy(feats_raw, all_heads, feat_names):
    print(f"[{datetime.now().isoformat()}] Test 5: Bootstrap greedy classifier (global mode)", flush=True)

    feat_dict = {}
    for name in all_heads:
        l = int(name[1:].split("H")[0])
        h = int(name.split("H")[1])
        feat_dict[(l, h)] = feats_raw[name]

    stability = classify_circuit(CLASSIFY_DEF, feat_dict, feat_names, n_bootstrap=100, search_mode="global")

    stability_scores = {f"L{l}H{h}": s for (l, h), s in stability.items()}
    labels = np.array([1 if h in ALL_CIRCUIT else 0 for h in all_heads])
    stab_vals = np.array([stability_scores.get(h, 0.0) for h in all_heads])
    auc_stab = roc_auc_score(labels, stab_vals)

    ranked = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    top_ranked = [{"head": h, "stability": s, "in_circuit": h in ALL_CIRCUIT, "tier": TIER_MAP.get(h)} for h, s in ranked[:30]]

    thresh_stats = {}
    for thresh in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        selected = {h for h, s in stability_scores.items() if s >= thresh}
        tp = len(selected & ALL_CIRCUIT)
        fp = len(selected - ALL_CIRCUIT)
        thresh_stats[str(thresh)] = {
            "n_selected": len(selected),
            "tp": tp,
            "fp": fp,
            "recall": tp / 15,
            "precision": tp / len(selected) if selected else 0,
        }

    per_tier_stability = {}
    for tier, heads in RTI_CIRCUIT.items():
        tier_stabs = []
        for l, h in heads:
            key = f"L{l}H{h}"
            tier_stabs.append({"head": key, "stability": stability_scores.get(key, 0.0)})
        mean_stab = float(np.mean([t["stability"] for t in tier_stabs]))
        above_07 = sum(1 for t in tier_stabs if t["stability"] >= 0.7)
        per_tier_stability[tier] = {
            "heads": tier_stabs,
            "mean_stability": mean_stab,
            "above_0.7": above_07,
            "total": len(heads),
        }

    result = {
        "test": "5_bootstrap_greedy_global",
        "n_bootstrap": 100,
        "auroc": auc_stab,
        "top30_ranked": top_ranked,
        "threshold_stats": thresh_stats,
        "per_tier_stability": per_tier_stability,
        "all_stability_scores": stability_scores,
    }
    print(f"  Stability AUROC: {auc_stab:.3f}", flush=True)
    for thresh, stats in thresh_stats.items():
        print(f"  threshold={thresh}: n={stats['n_selected']}, recall={stats['recall']:.2f}, precision={stats['precision']:.2f}", flush=True)
    for tier, info in per_tier_stability.items():
        print(f"  {tier}: mean_stab={info['mean_stability']:.3f}, above_0.7={info['above_0.7']}/{info['total']}", flush=True)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] E4: Weight-feature classifier for RTI circuit", flush=True)
    print(f"Pre-reg SHA: {PREREG_SHA}", flush=True)

    feats_raw, all_heads, feat_names = load_features()
    print(f"  {len(all_heads)} heads, {len(feat_names)} features", flush=True)

    results = {"prereg_sha": PREREG_SHA, "timestamp": datetime.now().isoformat(), "n_heads": len(all_heads), "n_features": len(feat_names)}

    results["test1"] = test1_per_feature_auroc(feats_raw, all_heads, feat_names)

    with open(OUT_DIR / "test1_per_feature_auroc.json", "w") as f:
        json.dump(results["test1"], f, indent=2)

    results["test2"] = test2_top5_combined(feats_raw, all_heads, feat_names)

    with open(OUT_DIR / "test2_top5_combined.json", "w") as f:
        json.dump(results["test2"], f, indent=2)

    results["test3"] = test3_loo_logistic(feats_raw, all_heads, feat_names)

    with open(OUT_DIR / "test3_loo_logistic.json", "w") as f:
        json.dump(results["test3"], f, indent=2)

    results["test4"] = test4_copier_seeded(feats_raw, all_heads, feat_names)

    with open(OUT_DIR / "test4_copier_seeded.json", "w") as f:
        json.dump(results["test4"], f, indent=2)

    results["test5"] = test5_bootstrap_greedy(feats_raw, all_heads, feat_names)

    with open(OUT_DIR / "test5_bootstrap_greedy.json", "w") as f:
        json.dump(results["test5"], f, indent=2)

    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{datetime.now().isoformat()}] All results saved to {OUT_DIR}/", flush=True)

    print("\n=== PREREG COMPARISON ===", flush=True)
    t1 = results["test1"]["best_full"]["auroc"]
    t2 = results["test2"]["auroc"]
    t3 = results["test3"]["auroc"]
    t4_recall = results["test4"]["total_recall"]
    t4_prec = results["test4"]["precision"]
    t5 = results["test5"]["auroc"]

    print(f"Test 1 best AUROC:  {t1:.3f}  (predicted 0.72-0.82, success > 0.70)", flush=True)
    print(f"Test 2 top-5 AUROC: {t2:.3f}  (predicted 0.78-0.88, success > 0.75)", flush=True)
    print(f"Test 3 LOO AUROC:   {t3:.3f}  (predicted 0.72-0.84, success > 0.70)", flush=True)
    print(f"Test 4 recall:      {t4_recall:.2f}  (predicted 0.40-0.67)", flush=True)
    print(f"Test 4 precision:   {t4_prec:.2f}  (predicted 0.40-0.65)", flush=True)
    print(f"Test 5 stab AUROC:  {t5:.3f}  (predicted 0.80-0.90, success > 0.75)", flush=True)

    pass_count = 0
    if t1 > 0.70:
        pass_count += 1
    if max(t2, t3, t5) > 0.80:
        pass_count += 1
    copier_stab = results["test5"]["per_tier_stability"]["copier"]["mean_stability"]
    other_mean = np.mean([results["test5"]["per_tier_stability"][t]["mean_stability"] for t in ["backbone", "detector", "readout"]])
    if copier_stab > other_mean:
        pass_count += 1
    best_thresh = max(results["test5"]["threshold_stats"].values(), key=lambda x: x["recall"] if x["precision"] > 0.4 else 0)
    if best_thresh["recall"] > 0.5 and best_thresh["precision"] > 0.4:
        pass_count += 1

    print(f"\nSuccess criteria met: {pass_count}/4", flush=True)
    if pass_count == 4:
        print("VERDICT: Features work", flush=True)
    elif pass_count >= 2:
        print("VERDICT: Features partially work", flush=True)
    else:
        print("VERDICT: Features don't work", flush=True)


if __name__ == "__main__":
    main()
