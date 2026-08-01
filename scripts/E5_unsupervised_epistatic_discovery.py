"""E5: Unsupervised epistatic circuit discovery.

Pre-registered in data/prereg/E5_unsupervised_epistatic_discovery.md (SHA 72bb47b).
Clusters 144 GPT-2 Small heads on 25 weight features, filters by K-composition,
tests for epistasis via ablation.

Usage:
  # Local test (Steps 1-3 only, no ablation):
  uv run --with transformer-lens --with hdbscan --with scikit-learn python scripts/E5_unsupervised_epistatic_discovery.py --steps 1-3

  # Full run (all steps, needs GPU for Step 4):
  uv run --with transformer-lens --with hdbscan --with scikit-learn python scripts/E5_unsupervised_epistatic_discovery.py --device cuda

  # Resume from saved intermediates:
  uv run ... python scripts/E5_unsupervised_epistatic_discovery.py --resume
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUT_DIR = REPO_DIR / "data" / "paper_numbers" / "E5_unsupervised_discovery"

FEATURES_PATH = Path.home() / "Documents/GitHub/weight-circuit-discovery/artifacts/gpt2-xl-transfer-20260510T051611:v0/features_gpt2_small.json"

RTI_DIR = (
    Path.home()
    / "Documents/GitHub/factorization-circuits/MIB/MIB-circuit-track"
    / "weight_circuit/experiments/v2_second_investigation/raw_experiments"
    / "v1_role_weight_analysis/part3_l9h3_investigation"
)

PREREG_SHA = "72bb47b"

COPIER_HEADS = [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)]
COPIER_SET = {f"L{l}H{h}" for l, h in COPIER_HEADS}

BACKBONE = [(0, 8), (0, 9), (0, 11)]
DETECTOR = [(4, 11)]
READOUT = [(10, 11), (11, 9), (11, 11)]
ALL_CIRCUIT = BACKBONE + DETECTOR + list(COPIER_HEADS) + READOUT
ALL_CIRCUIT_SET = {f"L{l}H{h}" for l, h in ALL_CIRCUIT}

CLUSTERING_FEATURES = [
    "ov_norm", "ov_concentration", "ov_sv_gap", "ov_effective_rank", "ov_top2_ratio",
    "qk_norm", "qk_concentration", "qk_sv_gap", "qk_effective_rank", "qk_top2_ratio",
    "ov_rank_ratio", "qk_rank_ratio", "ov_qk_rank_asymmetry",
    "ov_qk_concentration_asymmetry",
    "qk_ov_top_sv_align", "qk_ov_top_right_align", "ov_unembed_norm",
    "qk_same_diff_ratio", "qk_same_diff_gap", "qk_same_diff_ratio_w", "qk_sens_tok",
    "ov_tok_diag_mean", "ov_tok_offdiag_mean", "ov_tok_copy_ratio", "ov_tok_logit_min",
]
assert len(CLUSTERING_FEATURES) == 25

N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
N_BOOT = 10_000


def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}", flush=True)


def parse_head(s):
    l, h = s.replace("L", "").split("H")
    return (int(l), int(h))


def head_str(l, h):
    return f"L{l}H{h}"


# ── Step 1: Feature extraction ──

def load_and_complete_features(model=None):
    """Load pre-computed features, derive missing ones, compute cross-matrix from model."""
    log("Step 1: Loading features...")

    with open(FEATURES_PATH) as f:
        feats_raw = json.load(f)

    all_heads = sorted(feats_raw.keys())
    assert len(all_heads) == 144

    for h in all_heads:
        f = feats_raw[h]
        f["ov_rank_ratio"] = f["ov_effective_rank"] / D_MODEL
        f["qk_rank_ratio"] = f["qk_effective_rank"] / D_MODEL
        f["ov_qk_rank_asymmetry"] = f["ov_effective_rank"] - f["qk_effective_rank"]
        f["ov_qk_concentration_asymmetry"] = f["ov_concentration"] - f["qk_concentration"]

    if model is not None:
        log("  Computing cross-matrix features from model weights...")
        import torch
        W_Q = model.W_Q.detach().cpu().float()
        W_K = model.W_K.detach().cpu().float()
        W_V = model.W_V.detach().cpu().float()
        W_O = model.W_O.detach().cpu().float()
        W_E = model.W_E.detach().cpu().float()

        top_k_tokens = min(5000, W_E.shape[0])
        tok_embeds = W_E[:top_k_tokens]

        for L in range(N_LAYERS):
            for H in range(N_HEADS):
                wq = W_Q[L, H]
                wk = W_K[L, H]
                wv = W_V[L, H]
                wo = W_O[L, H]

                U_ov, S_ov, Vh_ov = torch.linalg.svd(wv @ wo, full_matrices=False)
                U_qk, S_qk, Vh_qk = torch.linalg.svd(wq @ wk.T, full_matrices=False)

                key = head_str(L, H)
                feats_raw[key]["qk_ov_top_sv_align"] = abs((U_ov[:, 0] @ U_qk[:, 0]).item())
                feats_raw[key]["qk_ov_top_right_align"] = abs((Vh_ov[0] @ Vh_qk[0]).item())

                ov_unembed_frob = torch.linalg.norm((tok_embeds @ wv) @ wo).item()
                feats_raw[key]["ov_unembed_norm"] = ov_unembed_frob / (top_k_tokens ** 0.5 + 1e-10)
    else:
        log("  WARNING: No model provided, cross-matrix features set to 0.0")
        for h in all_heads:
            feats_raw[h]["qk_ov_top_sv_align"] = 0.0
            feats_raw[h]["qk_ov_top_right_align"] = 0.0
            feats_raw[h]["ov_unembed_norm"] = 0.0

    X = np.array([[feats_raw[h][f] for f in CLUSTERING_FEATURES] for h in all_heads])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    log(f"  Feature matrix: {X_std.shape} (heads x features)")
    return feats_raw, all_heads, X_std


# ── Correctness gate ──

def correctness_gate():
    """Verify HDBSCAN on synthetic density clusters (amendment-1 construction).

    Plants 8 heads as a tight cluster: all 25 features drawn from N(mu, sigma_c^2).
    Background 136 heads are iid N(0,1). Standardization uses background stats only.
    """
    import hdbscan
    log("Correctness gate: synthetic density cluster verification (amendment-1)...")

    MU = 2.0
    results = {}
    for sigma_c in [0.5, 0.3, 0.15]:
        recoveries = []
        for trial in range(10):
            rng = np.random.RandomState(1000 * int(sigma_c * 100) + trial)
            X_bg = rng.randn(136, 25)
            X_sig = MU + sigma_c * rng.randn(8, 25)

            bg_mean = X_bg.mean(axis=0)
            bg_std = X_bg.std(axis=0) + 1e-10
            X_bg_std = (X_bg - bg_mean) / bg_std
            X_sig_std = (X_sig - bg_mean) / bg_std
            X_std = np.vstack([X_sig_std, X_bg_std])

            clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3, metric="euclidean")
            labels = clusterer.fit_predict(X_std)

            best_recall = 0
            best_fp = 144
            for cid in set(labels):
                if cid == -1:
                    continue
                members = set(np.where(labels == cid)[0])
                true_pos = len(members & set(range(8)))
                false_pos = len(members) - true_pos
                if true_pos > best_recall or (true_pos == best_recall and false_pos < best_fp):
                    best_recall = true_pos
                    best_fp = false_pos

            recoveries.append({"recall": best_recall, "false_positives": best_fp})

        median_recall = float(np.median([r["recall"] for r in recoveries]))
        median_fp = float(np.median([r["false_positives"] for r in recoveries]))
        results[f"sc_{sigma_c}"] = {
            "trials": recoveries,
            "median_recall": median_recall,
            "median_fp": median_fp,
        }
        log(f"  σ_c={sigma_c}: median recall={median_recall}/8, median FP={median_fp}")

    gate_pass = (
        results["sc_0.15"]["median_recall"] >= 7
        and results["sc_0.15"]["median_fp"] <= 2
        and results["sc_0.3"]["median_recall"] >= 5
    )
    results["gate_pass"] = gate_pass
    if not gate_pass:
        log("  CORRECTNESS GATE FAILED. HDBSCAN parameters need adjustment.")
    else:
        log("  Correctness gate PASSED.")

    return results


# ── Step 2: Clustering ──

def run_clustering(X_std, all_heads):
    """HDBSCAN clustering with robustness checks and fallbacks."""
    import hdbscan

    log("Step 2: Clustering...")
    results = {}

    def cluster_and_report(X, min_cs, label):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=min_cs, metric="euclidean")
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        clusters = {}
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            members = [all_heads[i] for i in np.where(labels == cid)[0]]
            clusters[int(cid)] = members
        log(f"  {label}: {n_clusters} clusters, {n_noise} noise points")
        for cid, members in clusters.items():
            copier_overlap = len(set(members) & COPIER_SET)
            log(f"    Cluster {cid}: {len(members)} heads, {copier_overlap}/8 copier overlap")
        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "clusters": clusters,
        }

    primary = cluster_and_report(X_std, 3, "Primary (min_cluster_size=3)")
    results["primary"] = primary

    for mcs in [4, 5]:
        results[f"robustness_mcs{mcs}"] = cluster_and_report(
            X_std, mcs, f"Robustness (min_cluster_size={mcs})"
        )

    used_pca = False
    if primary["n_noise"] > 130:
        log("  >90% noise — applying PCA fallback (95% variance)...")
        pca = PCA(n_components=0.95, svd_solver="full")
        X_pca = pca.fit_transform(X_std)
        n_components = X_pca.shape[1]
        log(f"  PCA reduced to {n_components} components")
        pca_result = cluster_and_report(X_pca, 3, "PCA fallback")
        results["pca_fallback"] = pca_result
        results["pca_n_components"] = n_components
        if pca_result["n_clusters"] > 0 and primary["n_clusters"] == 0:
            used_pca = True
            log("  Using PCA fallback results for H1 evaluation")

    active = results.get("pca_fallback", primary) if used_pca else primary
    results["active_source"] = "pca_fallback" if used_pca else "primary"

    if active["n_clusters"] < 3:
        log("  <3 clusters — running agglomerative follow-up (exploratory)...")
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        best_score = -1
        best_n = 2
        X_active = X_std
        for n_c in range(2, 20):
            agg = AgglomerativeClustering(n_clusters=n_c, linkage="ward")
            agg_labels = agg.fit_predict(X_active)
            score = silhouette_score(X_active, agg_labels)
            if score > best_score:
                best_score = score
                best_n = n_c

        agg = AgglomerativeClustering(n_clusters=best_n, linkage="ward")
        agg_labels = agg.fit_predict(X_active)
        agg_clusters = {}
        for cid in sorted(set(agg_labels)):
            members = [all_heads[i] for i in np.where(agg_labels == cid)[0]]
            agg_clusters[int(cid)] = members
        log(f"  Agglomerative: {best_n} clusters (silhouette={best_score:.3f})")
        results["agglomerative_exploratory"] = {
            "n_clusters": best_n,
            "silhouette": best_score,
            "clusters": agg_clusters,
            "labels": agg_labels.tolist(),
        }

    return results


# ── Step 3: K-composition filtering ──

def kcomp_filter(clusters, model, threshold=1.5):
    """Filter clusters by outgoing/background K-composition ratio."""
    import torch
    log("Step 3: K-composition filtering...")

    W_O = model.W_O.detach().cpu().float()
    W_K = model.W_K.detach().cpu().float()

    def kcomp_norm(l1, h1, l2, h2):
        wo = W_O[l1, h1]
        wk = W_K[l2, h2]
        return torch.linalg.norm(wo @ wk).item()

    all_pairs_cache = {}
    for l1 in range(N_LAYERS):
        for h1 in range(N_HEADS):
            for l2 in range(l1 + 1, N_LAYERS):
                for h2 in range(N_HEADS):
                    all_pairs_cache[(l1, h1, l2, h2)] = kcomp_norm(l1, h1, l2, h2)

    results = {}
    for cid, members in clusters.items():
        if len(members) < 3:
            continue

        member_set = {parse_head(h) for h in members}

        outgoing_scores = []
        background_scores = []

        for (l1, h1, l2, h2), score in all_pairs_cache.items():
            sender_in = (l1, h1) in member_set
            recv_in = (l2, h2) in member_set
            if sender_in and not recv_in:
                outgoing_scores.append(score)
            elif not sender_in and not recv_in:
                background_scores.append(score)

        if not outgoing_scores or not background_scores:
            continue

        outgoing_mean = float(np.mean(outgoing_scores))
        background_mean = float(np.mean(background_scores))
        ratio = outgoing_mean / (background_mean + 1e-10)

        copier_overlap = len(set(members) & COPIER_SET)
        log(f"  Cluster {cid} ({len(members)} heads, {copier_overlap} copiers): "
            f"outgoing={outgoing_mean:.4f}, background={background_mean:.4f}, ratio={ratio:.3f}"
            f" {'PASS' if ratio > threshold else 'FAIL'}")

        results[cid] = {
            "members": members,
            "size": len(members),
            "outgoing_kcomp": outgoing_mean,
            "background_kcomp": background_mean,
            "ratio": ratio,
            "passes_1_5": ratio > 1.5,
            "passes_1_2": ratio > 1.2,
            "copier_overlap": copier_overlap,
        }

    return results


# ── Hypothesis evaluation (H1, H2) ──

def evaluate_h1_h2(cluster_results, kcomp_results):
    """Evaluate H1 (copier recovery) and H2 (K-comp filter retains copier cluster)."""
    log("Evaluating H1 and H2...")

    h1_result = {"pass": False, "best_cluster": None}
    h2_result = {"pass": False}

    active_clusters = cluster_results.get(
        cluster_results["active_source"], cluster_results["primary"]
    )["clusters"]

    for cid, members in active_clusters.items():
        copier_recall = len(set(members) & COPIER_SET)
        size = len(members)
        if copier_recall >= 5 and size <= 12 and size <= 15:
            precision = copier_recall / size
            h1_result = {
                "pass": True,
                "best_cluster": int(cid),
                "copier_recall": copier_recall,
                "cluster_size": size,
                "precision": precision,
                "members": members,
            }
            log(f"  H1 PASS: cluster {cid}, recall={copier_recall}/8, size={size}, precision={precision:.2f}")

            cid_str = str(cid)
            if cid_str in kcomp_results or int(cid) in kcomp_results:
                kc = kcomp_results.get(cid_str, kcomp_results.get(int(cid), {}))
                if kc.get("passes_1_5", False):
                    h2_result = {"pass": True, "ratio": kc["ratio"]}
                    log(f"  H2 PASS: ratio={kc['ratio']:.3f}")
                else:
                    h2_result = {"pass": False, "ratio": kc.get("ratio")}
                    log(f"  H2 FAIL: ratio={kc.get('ratio', 'N/A')}")
            break

    if not h1_result["pass"]:
        log("  H1 FAIL: no single cluster meets criteria")
        top2 = sorted(
            active_clusters.items(),
            key=lambda x: len(set(x[1]) & COPIER_SET),
            reverse=True,
        )[:2]
        if len(top2) == 2:
            union = set(top2[0][1]) | set(top2[1][1])
            union_recall = len(union & COPIER_SET)
            union_size = len(union)
            if union_recall >= 6 and union_size <= 18:
                log(f"  Union recovery (descriptive): recall={union_recall}/8, size={union_size}")
                h1_result["union_recovery"] = {
                    "recall": union_recall,
                    "size": union_size,
                    "clusters": [int(top2[0][0]), int(top2[1][0])],
                }

    return h1_result, h2_result


# ── Step 4: Epistasis testing ──

def run_epistasis(cluster_heads, model, device="cpu"):
    """Compute epistasis score for a group of heads via ablation."""
    import torch
    from tqdm import tqdm

    sys.path.insert(0, str(RTI_DIR))
    from run_rti_task import make_prompts

    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    n = len(valid)

    model.cfg.use_attn_result = True
    head_tuples = [parse_head(h) if isinstance(h, str) else h for h in cluster_heads]

    def ablate_heads(heads_to_ablate):
        per_prompt_effects = []
        for p in valid:
            tokens = model.to_tokens(p["text"], prepend_bos=True).to(device)
            with torch.no_grad():
                clean_logits = model(tokens)
            last_clean = clean_logits[0, -1]
            clean_ld = (last_clean[p["correct_id"]] - last_clean[p["distractor_id"]]).item()

            hooks = []
            for layer in range(N_LAYERS):
                heads_at_layer = [h for l, h in heads_to_ablate if l == layer]
                if heads_at_layer:
                    def make_hook(_heads):
                        def fn(value, hook):
                            for hi in _heads:
                                value[:, :, hi, :] = 0.0
                            return value
                        return fn
                    hooks.append((f"blocks.{layer}.attn.hook_result", make_hook(heads_at_layer)))

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            last_abl = ablated_logits[0, -1]
            ablated_ld = (last_abl[p["correct_id"]] - last_abl[p["distractor_id"]]).item()
            per_prompt_effects.append(clean_ld - ablated_ld)

        return per_prompt_effects

    log(f"  Ablating full cluster ({len(head_tuples)} heads)...")
    full_effects = ablate_heads(head_tuples)
    full_mean = float(np.mean(full_effects))

    loo_sum_effects = np.zeros(len(valid))
    for head in tqdm(head_tuples, desc="LOO ablation"):
        effects = ablate_heads([head])
        loo_sum_effects += np.array(effects)

    loo_sum_mean = float(np.mean(loo_sum_effects))
    epistasis = 1.0 - (loo_sum_mean / (full_mean + 1e-10)) if abs(full_mean) > 1e-10 else 0.0

    rng = np.random.RandomState(42)
    boot_epistasis = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(valid), size=len(valid), replace=True)
        boot_full = float(np.mean(np.array(full_effects)[idx]))
        boot_loo = float(np.mean(loo_sum_effects[idx]))
        if abs(boot_full) > 1e-10:
            boot_epistasis.append(1.0 - boot_loo / boot_full)
    ci_lo, ci_hi = float(np.percentile(boot_epistasis, 2.5)), float(np.percentile(boot_epistasis, 97.5))

    return {
        "epistasis_score": epistasis,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "full_effect_mean": full_mean,
        "loo_sum_mean": loo_sum_mean,
        "n_prompts": len(valid),
        "n_heads": len(head_tuples),
        "heads": [head_str(*h) for h in head_tuples],
    }


def run_random_baseline(cluster_size, model, n_random=50, device="cpu"):
    """Compute epistasis for random head groups of matched size."""
    import torch
    from tqdm import tqdm

    sys.path.insert(0, str(RTI_DIR))
    from run_rti_task import make_prompts

    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]

    model.cfg.use_attn_result = True
    all_heads = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]

    rng = np.random.RandomState(123)
    random_epistasis_scores = []

    for gi in tqdm(range(n_random), desc="Random baseline groups"):
        group = [all_heads[i] for i in rng.choice(len(all_heads), size=cluster_size, replace=False)]

        full_effects = []
        for p in valid:
            tokens = model.to_tokens(p["text"], prepend_bos=True).to(device)
            with torch.no_grad():
                clean_logits = model(tokens)
            last_clean = clean_logits[0, -1]
            clean_ld = (last_clean[p["correct_id"]] - last_clean[p["distractor_id"]]).item()

            hooks = []
            for layer in range(N_LAYERS):
                heads_at_layer = [h for l, h in group if l == layer]
                if heads_at_layer:
                    def make_hook(_heads):
                        def fn(value, hook):
                            for hi in _heads:
                                value[:, :, hi, :] = 0.0
                            return value
                        return fn
                    hooks.append((f"blocks.{layer}.attn.hook_result", make_hook(heads_at_layer)))

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            last_abl = ablated_logits[0, -1]
            ablated_ld = (last_abl[p["correct_id"]] - last_abl[p["distractor_id"]]).item()
            full_effects.append(clean_ld - ablated_ld)

        full_mean = float(np.mean(full_effects))

        loo_sum = np.zeros(len(valid))
        for head in group:
            for pi, p in enumerate(valid):
                tokens = model.to_tokens(p["text"], prepend_bos=True).to(device)
                with torch.no_grad():
                    clean_logits = model(tokens)
                clean_ld = (clean_logits[0, -1, p["correct_id"]] - clean_logits[0, -1, p["distractor_id"]]).item()

                hooks = []
                layer = head[0]
                def make_hook(_h):
                    def fn(value, hook):
                        value[:, :, _h, :] = 0.0
                        return value
                    return fn
                hooks.append((f"blocks.{layer}.attn.hook_result", make_hook(head[1])))

                with torch.no_grad():
                    abl_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                abl_ld = (abl_logits[0, -1, p["correct_id"]] - abl_logits[0, -1, p["distractor_id"]]).item()
                loo_sum[pi] += clean_ld - abl_ld

        loo_mean = float(np.mean(loo_sum))
        eps = 1.0 - (loo_mean / (full_mean + 1e-10)) if abs(full_mean) > 1e-10 else 0.0
        random_epistasis_scores.append(eps)
        log(f"    Random group {gi}: epistasis={eps:.3f}")

    p95 = float(np.percentile(random_epistasis_scores, 95))
    return {
        "scores": random_epistasis_scores,
        "p95": p95,
        "mean": float(np.mean(random_epistasis_scores)),
        "std": float(np.std(random_epistasis_scores)),
        "n_groups": n_random,
        "group_size": cluster_size,
    }


# ── H4: Null control ──

def run_null_control(all_heads, model, n_permutations=100):
    """Permute features, re-run clustering + K-comp, check for copier overlap."""
    import hdbscan
    log("H4: Null control (100 permutations)...")

    with open(FEATURES_PATH) as f:
        feats_raw = json.load(f)

    for h in all_heads:
        f = feats_raw[h]
        f["ov_rank_ratio"] = f["ov_effective_rank"] / D_MODEL
        f["qk_rank_ratio"] = f["qk_effective_rank"] / D_MODEL
        f["ov_qk_rank_asymmetry"] = f["ov_effective_rank"] - f["qk_effective_rank"]
        f["ov_qk_concentration_asymmetry"] = f["ov_concentration"] - f["qk_concentration"]

    present_features = [f for f in CLUSTERING_FEATURES if f in feats_raw[all_heads[0]]]
    X_orig = np.array([[feats_raw[h][f] for f in present_features] for h in all_heads])
    X_orig = np.nan_to_num(X_orig, nan=0.0, posinf=0.0, neginf=0.0)

    copier_indices = {i for i, h in enumerate(all_heads) if h in COPIER_SET}
    false_discovery_count = 0

    rng = np.random.RandomState(42)
    for perm_i in range(n_permutations):
        X_perm = X_orig.copy()
        for col in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, col])

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_perm)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3, metric="euclidean")
        labels = clusterer.fit_predict(X_std)

        clusters = {}
        for cid in set(labels):
            if cid == -1:
                continue
            members_idx = set(np.where(labels == cid)[0])
            clusters[cid] = members_idx

        kcomp_filtered = {}
        if model is not None:
            import torch
            W_O = model.W_O.detach().cpu().float()
            W_K = model.W_K.detach().cpu().float()

            for cid, members_idx in clusters.items():
                if len(members_idx) < 3:
                    continue
                member_heads = {(i // N_HEADS, i % N_HEADS) for i in members_idx}
                outgoing = []
                background = []
                for l1 in range(N_LAYERS):
                    for h1 in range(N_HEADS):
                        for l2 in range(l1 + 1, N_LAYERS):
                            for h2 in range(N_HEADS):
                                score = torch.linalg.norm(W_O[l1, h1] @ W_K[l2, h2]).item()
                                s_in = (l1, h1) in member_heads
                                r_in = (l2, h2) in member_heads
                                if s_in and not r_in:
                                    outgoing.append(score)
                                elif not s_in and not r_in:
                                    background.append(score)
                if outgoing and background:
                    ratio = np.mean(outgoing) / (np.mean(background) + 1e-10)
                    if ratio > 1.5:
                        kcomp_filtered[cid] = members_idx

        for cid, members_idx in kcomp_filtered.items():
            copier_recall = len(members_idx & copier_indices)
            if copier_recall >= 5:
                false_discovery_count += 1
                break

        if (perm_i + 1) % 20 == 0:
            log(f"  {perm_i + 1}/{n_permutations} permutations done, "
                f"{false_discovery_count} false discoveries so far")

    specificity = 1.0 - false_discovery_count / n_permutations
    h4_pass = specificity >= 0.95

    log(f"  H4: {false_discovery_count}/{n_permutations} false discoveries, "
        f"specificity={specificity:.2f} {'PASS' if h4_pass else 'FAIL'}")

    return {
        "n_permutations": n_permutations,
        "false_discoveries": false_discovery_count,
        "specificity": specificity,
        "pass": h4_pass,
    }


# ── Main ──

def save_results(results, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", default="1-5", help="Steps to run (e.g. '1-3' for local test)")
    parser.add_argument("--resume", action="store_true", help="Load saved intermediates")
    args = parser.parse_args()

    step_range = args.steps.split("-")
    min_step = int(step_range[0])
    max_step = int(step_range[-1])

    log(f"E5: Unsupervised Epistatic Circuit Discovery")
    log(f"  Pre-reg SHA: {PREREG_SHA}")
    log(f"  Device: {args.device}")
    log(f"  Steps: {min_step}-{max_step}")

    model = None
    if max_step >= 3 or min_step <= 1:
        log("Loading GPT-2 Small...")
        import torch
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)
        model.eval()

    all_results = {"prereg_sha": PREREG_SHA, "timestamp": datetime.now().isoformat()}

    # Correctness gate
    if min_step <= 1:
        gate = correctness_gate()
        all_results["correctness_gate"] = gate
        save_results(gate, "correctness_gate.json")
        if not gate["gate_pass"]:
            log("ABORTING: correctness gate failed.")
            return

    # Step 1
    if min_step <= 1 and max_step >= 1:
        feats_raw, all_heads, X_std = load_and_complete_features(model)
        save_results({"heads": all_heads, "features": CLUSTERING_FEATURES, "shape": list(X_std.shape)}, "step1_features.json")
        np.save(OUT_DIR / "X_std.npy", X_std)
    elif args.resume:
        X_std = np.load(OUT_DIR / "X_std.npy")
        with open(OUT_DIR / "step1_features.json") as f:
            d = json.load(f)
        all_heads = d["heads"]
        feats_raw = None

    # Step 2
    if min_step <= 2 and max_step >= 2:
        cluster_results = run_clustering(X_std, all_heads)
        save_results(cluster_results, "step2_clustering.json")
        all_results["clustering"] = cluster_results
    elif args.resume:
        with open(OUT_DIR / "step2_clustering.json") as f:
            cluster_results = json.load(f)

    # Step 3
    if min_step <= 3 and max_step >= 3:
        active_source = cluster_results["active_source"]
        active_clusters = cluster_results[active_source]["clusters"]
        kcomp_results = kcomp_filter(active_clusters, model)
        save_results(kcomp_results, "step3_kcomp.json")
        all_results["kcomp"] = kcomp_results

        h1, h2 = evaluate_h1_h2(cluster_results, kcomp_results)
        all_results["H1"] = h1
        all_results["H2"] = h2
        save_results({"H1": h1, "H2": h2}, "h1_h2_results.json")
    elif args.resume:
        with open(OUT_DIR / "step3_kcomp.json") as f:
            kcomp_results = json.load(f)
        with open(OUT_DIR / "h1_h2_results.json") as f:
            h1h2 = json.load(f)
            h1, h2 = h1h2["H1"], h1h2["H2"]

    # Step 4: Epistasis
    if min_step <= 4 and max_step >= 4:
        log("Step 4: Epistasis testing...")
        filtered_clusters = {k: v for k, v in kcomp_results.items()
                            if isinstance(v, dict) and v.get("passes_1_5", False)}

        if not filtered_clusters:
            log("  No clusters passed K-comp filter at 1.5. Checking 1.2...")
            filtered_clusters = {k: v for k, v in kcomp_results.items()
                                if isinstance(v, dict) and v.get("passes_1_2", False)}
            if filtered_clusters:
                log(f"  {len(filtered_clusters)} clusters pass at 1.2 (non-confirmatory)")

        epistasis_results = {}
        for cid, info in filtered_clusters.items():
            log(f"  Cluster {cid}: running epistasis on {info['members']}...")
            ep = run_epistasis(info["members"], model, device=args.device)
            epistasis_results[cid] = ep
            save_results(ep, f"step4_epistasis_cluster_{cid}.json")

            log(f"  Running random baseline (50 groups of {info['size']} heads)...")
            baseline = run_random_baseline(info["size"], model, n_random=50, device=args.device)
            epistasis_results[f"{cid}_random_baseline"] = baseline
            save_results(baseline, f"step4_random_baseline_cluster_{cid}.json")

            passes_absolute = ep["epistasis_score"] > 0.25
            passes_random = ep["epistasis_score"] > baseline["p95"]
            h3_pass = passes_absolute and passes_random
            epistasis_results[f"{cid}_h3"] = {
                "epistasis_score": ep["epistasis_score"],
                "ci": [ep["ci_lo"], ep["ci_hi"]],
                "random_p95": baseline["p95"],
                "passes_absolute": passes_absolute,
                "passes_random": passes_random,
                "pass": h3_pass,
            }
            log(f"  H3 cluster {cid}: epistasis={ep['epistasis_score']:.3f} "
                f"[{ep['ci_lo']:.3f}, {ep['ci_hi']:.3f}], "
                f"random p95={baseline['p95']:.3f}, "
                f"{'PASS' if h3_pass else 'FAIL'}")

        all_results["epistasis"] = epistasis_results
        save_results(epistasis_results, "step4_epistasis.json")

    # Step 5: Null control + novel groups
    if min_step <= 5 and max_step >= 5:
        h4 = run_null_control(all_heads, model, n_permutations=100)
        all_results["H4"] = h4
        save_results(h4, "h4_null_control.json")

        # H5: novel groups
        log("H5: Checking for novel epistatic groups (exploratory)...")
        novel_results = {}
        for cid, info in kcomp_results.items():
            if not isinstance(info, dict):
                continue
            if not info.get("passes_1_5", False):
                continue
            copier_overlap = info.get("copier_overlap", 0)
            if copier_overlap >= 5:
                continue
            if info["size"] >= 3:
                log(f"  Non-copier cluster {cid} passes K-comp. Running epistasis...")
                ep = run_epistasis(info["members"], model, device=args.device)
                novel_results[cid] = {
                    "members": info["members"],
                    "epistasis": ep,
                    "passes": ep["epistasis_score"] > 0.20,
                }
                save_results(novel_results[cid], f"h5_novel_cluster_{cid}.json")

        all_results["H5"] = novel_results if novel_results else {"status": "no_novel_groups_found"}

    # Final summary
    log("\n=== RESULTS SUMMARY ===")
    if "H1" in all_results:
        log(f"H1 (copier recovery): {'PASS' if all_results['H1']['pass'] else 'FAIL'}")
    if "H2" in all_results:
        log(f"H2 (K-comp filter): {'PASS' if all_results['H2']['pass'] else 'FAIL'}")
    if "epistasis" in all_results:
        for cid in all_results["epistasis"]:
            if cid.endswith("_h3"):
                log(f"H3 (epistasis, cluster {cid.replace('_h3', '')}): "
                    f"{'PASS' if all_results['epistasis'][cid]['pass'] else 'FAIL'}")
    if "H4" in all_results:
        log(f"H4 (null control): {'PASS' if all_results['H4']['pass'] else 'FAIL'}")
    if "H5" in all_results:
        if isinstance(all_results["H5"], dict) and "status" not in all_results["H5"]:
            for cid, info in all_results["H5"].items():
                log(f"H5 (novel group {cid}): {'FOUND' if info['passes'] else 'below threshold'}")
        else:
            log("H5 (novel groups): none found (exploratory)")

    save_results(all_results, "E5_all_results.json")
    log("Done.")


if __name__ == "__main__":
    main()
