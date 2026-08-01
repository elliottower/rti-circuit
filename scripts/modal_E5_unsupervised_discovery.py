"""Modal wrapper for E5: Unsupervised epistatic circuit discovery.

Pre-registered in data/prereg/E5_unsupervised_epistatic_discovery.md (SHA 72bb47b).

Usage:
    cd /Users/elliottower/Documents/GitHub/rti-circuit
    modal run scripts/modal_E5_unsupervised_discovery.py --detach
"""

import modal

app = modal.App("rti-e5-unsupervised-epistatic-discovery")

vol = modal.Volume.from_name("e5-unsupervised-discovery-results", create_if_missing=True)

FEATURES_JSON = (
    "/Users/elliottower/Documents/GitHub/weight-circuit-discovery"
    "/artifacts/gpt2-xl-transfer-20260510T051611:v0/features_gpt2_small.json"
)

PART3_DIR = (
    "/Users/elliottower/Documents/GitHub/factorization-circuits"
    "/MIB/MIB-circuit-track/weight_circuit/experiments"
    "/v2_second_investigation/raw_experiments/v1_role_weight_analysis"
    "/part3_l9h3_investigation"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "transformer-lens==2.17.0",
        "transformers==4.51.3",
        "matplotlib==3.9.4",
        "hdbscan==0.8.40",
        "scikit-learn==1.6.1",
    )
    .add_local_file(
        FEATURES_JSON,
        remote_path="/app/features_gpt2_small.json",
    )
    .add_local_dir(
        PART3_DIR,
        remote_path="/app/a/b/c/d/e/part3",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": vol},
)
def run_e5():
    import json
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import torch
    import hdbscan
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    sys.path.insert(0, "/app/a/b/c/d/e/part3")
    from run_rti_task import make_prompts

    RESULTS = Path("/results/E5")
    RESULTS.mkdir(parents=True, exist_ok=True)

    PREREG_SHA = "72bb47b"
    N_LAYERS = 12
    N_HEADS = 12
    D_MODEL = 768
    N_BOOT = 10_000

    COPIER_HEADS = [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)]
    COPIER_SET = {f"L{l}H{h}" for l, h in COPIER_HEADS}
    ALL_CIRCUIT = [(0,8),(0,9),(0,11),(4,11)] + list(COPIER_HEADS) + [(10,11),(11,9),(11,11)]

    CLUSTERING_FEATURES = [
        "ov_norm", "ov_concentration", "ov_sv_gap", "ov_effective_rank", "ov_top2_ratio",
        "qk_norm", "qk_concentration", "qk_sv_gap", "qk_effective_rank", "qk_top2_ratio",
        "ov_rank_ratio", "qk_rank_ratio", "ov_qk_rank_asymmetry",
        "ov_qk_concentration_asymmetry",
        "qk_ov_top_sv_align", "qk_ov_top_right_align", "ov_unembed_norm",
        "qk_same_diff_ratio", "qk_same_diff_gap", "qk_same_diff_ratio_w", "qk_sens_tok",
        "ov_tok_diag_mean", "ov_tok_offdiag_mean", "ov_tok_copy_ratio", "ov_tok_logit_min",
    ]

    def log(msg):
        print(f"[{datetime.now().isoformat()}] {msg}", flush=True)

    def save(data, name):
        path = RESULTS / name
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        vol.commit()
        log(f"  Saved + committed: {path}")

    def parse_head(s):
        l, h = s.replace("L", "").split("H")
        return (int(l), int(h))

    # ── Load model ──
    log("Loading GPT-2 Small on CUDA...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    model.eval()
    model.cfg.use_attn_result = True
    log("Model loaded.")

    # ── Step 1: Features ──
    log("Step 1: Feature extraction...")
    with open("/app/features_gpt2_small.json") as f:
        feats_raw = json.load(f)
    all_heads = sorted(feats_raw.keys())

    for h in all_heads:
        ff = feats_raw[h]
        ff["ov_rank_ratio"] = ff["ov_effective_rank"] / D_MODEL
        ff["qk_rank_ratio"] = ff["qk_effective_rank"] / D_MODEL
        ff["ov_qk_rank_asymmetry"] = ff["ov_effective_rank"] - ff["qk_effective_rank"]
        ff["ov_qk_concentration_asymmetry"] = ff["ov_concentration"] - ff["qk_concentration"]

    W_Q = model.W_Q.detach().cpu().float()
    W_K = model.W_K.detach().cpu().float()
    W_V = model.W_V.detach().cpu().float()
    W_O = model.W_O.detach().cpu().float()
    W_E = model.W_E.detach().cpu().float()
    tok_embeds = W_E[:5000]

    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            wq, wk, wv, wo = W_Q[L,H], W_K[L,H], W_V[L,H], W_O[L,H]
            U_ov, _, Vh_ov = torch.linalg.svd(wv @ wo, full_matrices=False)
            U_qk, _, Vh_qk = torch.linalg.svd(wq @ wk.T, full_matrices=False)
            key = f"L{L}H{H}"
            feats_raw[key]["qk_ov_top_sv_align"] = abs((U_ov[:,0] @ U_qk[:,0]).item())
            feats_raw[key]["qk_ov_top_right_align"] = abs((Vh_ov[0] @ Vh_qk[0]).item())
            feats_raw[key]["ov_unembed_norm"] = torch.linalg.norm((tok_embeds @ wv) @ wo).item() / (5000**0.5)

    X = np.array([[feats_raw[h][f] for f in CLUSTERING_FEATURES] for h in all_heads])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    log(f"  Feature matrix: {X_std.shape}")
    save({"heads": all_heads, "features": CLUSTERING_FEATURES}, "step1_features.json")

    # ── Correctness gate (amendment-1: density clusters, not mean shift) ──
    log("Correctness gate (amendment-1)...")
    gate_results = {}
    MU = 2.0
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
            X_syn_std = np.vstack([X_sig_std, X_bg_std])
            cl = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3, metric="euclidean")
            lab = cl.fit_predict(X_syn_std)
            best = 0
            for cid in set(lab):
                if cid == -1: continue
                tp = len(set(np.where(lab == cid)[0]) & set(range(8)))
                if tp > best: best = tp
            recoveries.append(best)
        gate_results[f"sc_{sigma_c}"] = {
            "median_recall": float(np.median(recoveries)),
            "all_recalls": recoveries,
        }
        log(f"  σ_c={sigma_c}: median recall={np.median(recoveries)}/8")

    gate_results["gate_pass"] = (
        gate_results["sc_0.15"]["median_recall"] >= 7
        and gate_results["sc_0.3"]["median_recall"] >= 5
    )
    save(gate_results, "correctness_gate.json")

    if not gate_results["gate_pass"]:
        log("  CORRECTNESS GATE FAILED. Aborting.")
        return

    # ── Step 2: Clustering ──
    log("Step 2: HDBSCAN clustering...")
    cl = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3, metric="euclidean")
    labels = cl.fit_predict(X_std)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    clusters = {}
    for cid in sorted(set(labels)):
        if cid == -1: continue
        members = [all_heads[i] for i in np.where(labels == cid)[0]]
        clusters[str(cid)] = members
        cop = len(set(members) & COPIER_SET)
        log(f"  Cluster {cid}: {len(members)} heads, copier={cop}/8")

    cluster_result = {
        "primary": {"clusters": clusters, "n_clusters": n_clusters, "n_noise": n_noise, "labels": labels.tolist()},
        "active_source": "primary",
    }

    # Robustness: mcs=4,5
    for mcs in [4, 5]:
        cl2 = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=mcs, metric="euclidean")
        lab2 = cl2.fit_predict(X_std)
        nc2 = len(set(lab2)) - (1 if -1 in lab2 else 0)
        cl2_dict = {}
        for cid in sorted(set(lab2)):
            if cid == -1: continue
            cl2_dict[str(cid)] = [all_heads[i] for i in np.where(lab2 == cid)[0]]
        cluster_result[f"robustness_mcs{mcs}"] = {"clusters": cl2_dict, "n_clusters": nc2}
        log(f"  Robustness mcs={mcs}: {nc2} clusters")

    # PCA fallback
    if n_noise > 130:
        log("  >90% noise — PCA fallback...")
        pca = PCA(n_components=0.95, svd_solver="full")
        X_pca = pca.fit_transform(X_std)
        cl3 = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3, metric="euclidean")
        lab3 = cl3.fit_predict(X_pca)
        pca_clusters = {}
        for cid in sorted(set(lab3)):
            if cid == -1: continue
            pca_clusters[str(cid)] = [all_heads[i] for i in np.where(lab3 == cid)[0]]
        cluster_result["pca_fallback"] = {"clusters": pca_clusters, "n_components": X_pca.shape[1]}
        if not clusters and pca_clusters:
            cluster_result["active_source"] = "pca_fallback"

    # Agglomerative follow-up (exploratory)
    if n_clusters < 3:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        log("  <3 clusters — agglomerative follow-up (exploratory)...")
        best_s, best_n = -1, 2
        for nc in range(2, 20):
            agg = AgglomerativeClustering(n_clusters=nc, linkage="ward")
            agg_l = agg.fit_predict(X_std)
            s = silhouette_score(X_std, agg_l)
            if s > best_s: best_s, best_n = s, nc
        agg = AgglomerativeClustering(n_clusters=best_n, linkage="ward")
        agg_l = agg.fit_predict(X_std)
        agg_clusters = {}
        for cid in sorted(set(agg_l)):
            agg_clusters[str(cid)] = [all_heads[i] for i in np.where(agg_l == cid)[0]]
        cluster_result["agglomerative_exploratory"] = {
            "n_clusters": best_n, "silhouette": best_s, "clusters": agg_clusters,
        }
        log(f"  Agglomerative: {best_n} clusters, silhouette={best_s:.3f}")

    # Real-data sanity check: do the 3 backbone heads cluster together?
    backbone_set = {"L0H8", "L0H9", "L0H11"}
    backbone_same_cluster = False
    backbone_cluster_id = None
    for cid, members in clusters.items():
        if backbone_set.issubset(set(members)):
            backbone_same_cluster = True
            backbone_cluster_id = cid
    cluster_result["backbone_sanity"] = {
        "same_cluster": backbone_same_cluster,
        "cluster_id": backbone_cluster_id,
        "note": "Descriptive check — do the 3 layer-0 backbone heads land together?"
    }
    log(f"  Backbone sanity: {'same cluster' if backbone_same_cluster else 'SPLIT/NOISE'}")

    save(cluster_result, "step2_clustering.json")

    # ── Step 3: K-composition filtering ──
    log("Step 3: K-composition filtering...")
    active_clusters = cluster_result[cluster_result["active_source"]]["clusters"]

    kcomp_cache = {}
    for l1 in range(N_LAYERS):
        for h1 in range(N_HEADS):
            for l2 in range(l1 + 1, N_LAYERS):
                for h2 in range(N_HEADS):
                    kcomp_cache[(l1,h1,l2,h2)] = torch.linalg.norm(W_O[l1,h1] @ W_K[l2,h2]).item()

    kcomp_results = {}
    for cid, members in active_clusters.items():
        if len(members) < 3:
            continue
        member_set = {parse_head(h) for h in members}
        outgoing, background = [], []
        for (l1,h1,l2,h2), score in kcomp_cache.items():
            s_in = (l1,h1) in member_set
            r_in = (l2,h2) in member_set
            if s_in and not r_in:
                outgoing.append(score)
            elif not s_in and not r_in:
                background.append(score)
        if not outgoing or not background:
            continue
        ratio = float(np.mean(outgoing)) / (float(np.mean(background)) + 1e-10)
        cop = len(set(members) & COPIER_SET)
        log(f"  Cluster {cid} ({len(members)} heads, {cop} copiers): ratio={ratio:.3f} "
            f"{'PASS' if ratio > 1.5 else 'FAIL'}")
        kcomp_results[cid] = {
            "members": members, "size": len(members),
            "outgoing_kcomp": float(np.mean(outgoing)),
            "background_kcomp": float(np.mean(background)),
            "ratio": ratio,
            "passes_1_5": ratio > 1.5, "passes_1_2": ratio > 1.2,
            "copier_overlap": cop,
        }
    save(kcomp_results, "step3_kcomp.json")

    # ── H1 & H2 ──
    h1 = {"pass": False}
    h2 = {"pass": False}
    for cid, members in active_clusters.items():
        cop_recall = len(set(members) & COPIER_SET)
        size = len(members)
        if cop_recall >= 5 and size <= 12 and size <= 15:
            h1 = {"pass": True, "cluster": cid, "copier_recall": cop_recall,
                   "size": size, "members": members}
            if cid in kcomp_results and kcomp_results[cid]["passes_1_5"]:
                h2 = {"pass": True, "ratio": kcomp_results[cid]["ratio"]}
            break

    if not h1["pass"]:
        top2 = sorted(active_clusters.items(), key=lambda x: len(set(x[1]) & COPIER_SET), reverse=True)[:2]
        if len(top2) == 2:
            union = set(top2[0][1]) | set(top2[1][1])
            ur = len(union & COPIER_SET)
            if ur >= 6 and len(union) <= 18:
                h1["union_recovery_descriptive"] = {"recall": ur, "size": len(union)}

    log(f"H1: {'PASS' if h1['pass'] else 'FAIL'}")
    log(f"H2: {'PASS' if h2['pass'] else 'FAIL'}")
    save({"H1": h1, "H2": h2}, "h1_h2_results.json")

    # ── Step 4: Epistasis testing ──
    log("Step 4: Epistasis testing...")
    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    n_prompts = len(valid)
    log(f"  {n_prompts} valid prompts")

    def ablate_group(heads_to_ablate):
        effects = []
        for p in valid:
            tokens = model.to_tokens(p["text"], prepend_bos=True).to("cuda")
            with torch.no_grad():
                clean = model(tokens)
            clean_ld = (clean[0,-1,p["correct_id"]] - clean[0,-1,p["distractor_id"]]).item()
            hooks = []
            for layer in range(N_LAYERS):
                hs = [h for l,h in heads_to_ablate if l == layer]
                if hs:
                    def mk(hds):
                        def fn(v, hook):
                            for hi in hds:
                                v[:,:,hi,:] = 0.0
                            return v
                        return fn
                    hooks.append((f"blocks.{layer}.attn.hook_result", mk(hs)))
            with torch.no_grad():
                abl = model.run_with_hooks(tokens, fwd_hooks=hooks)
            abl_ld = (abl[0,-1,p["correct_id"]] - abl[0,-1,p["distractor_id"]]).item()
            effects.append(clean_ld - abl_ld)
        return np.array(effects)

    def compute_epistasis(head_tuples):
        full_eff = ablate_group(head_tuples)
        full_mean = float(np.mean(full_eff))
        loo_sum = np.zeros(n_prompts)
        for head in tqdm(head_tuples, desc="LOO"):
            loo_sum += ablate_group([head])
        loo_mean = float(np.mean(loo_sum))
        eps = 1.0 - (loo_mean / (full_mean + 1e-10)) if abs(full_mean) > 1e-10 else 0.0
        rng = np.random.RandomState(42)
        boot = []
        for _ in range(N_BOOT):
            idx = rng.choice(n_prompts, size=n_prompts, replace=True)
            bf = float(np.mean(full_eff[idx]))
            bl = float(np.mean(loo_sum[idx]))
            if abs(bf) > 1e-10:
                boot.append(1.0 - bl / bf)
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        return {"epistasis": eps, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "full_mean": full_mean, "loo_mean": loo_mean}

    # Test filtered clusters
    filtered = {k: v for k, v in kcomp_results.items()
                if isinstance(v, dict) and v.get("passes_1_5", False)}
    if not filtered:
        filtered = {k: v for k, v in kcomp_results.items()
                    if isinstance(v, dict) and v.get("passes_1_2", False)}
        if filtered:
            log("  No clusters pass at 1.5. Using 1.2 (non-confirmatory).")

    epistasis_results = {}
    for cid, info in filtered.items():
        head_tuples = [parse_head(h) for h in info["members"]]
        if len(head_tuples) > 50:
            log(f"  Cluster {cid} has {len(head_tuples)} heads — too large for LOO, sampling 15")
            rng = np.random.RandomState(42)
            head_tuples = [head_tuples[i] for i in rng.choice(len(head_tuples), 15, replace=False)]
        log(f"  Cluster {cid}: epistasis on {len(head_tuples)} heads...")
        ep = compute_epistasis(head_tuples)
        epistasis_results[cid] = ep
        save(ep, f"step4_epistasis_cluster_{cid}.json")

        log(f"  Random baseline (50 groups of {len(head_tuples)} heads)...")
        all_head_tuples = [(l,h) for l in range(N_LAYERS) for h in range(N_HEADS)]
        rng = np.random.RandomState(123)
        random_eps = []
        for gi in tqdm(range(50), desc="Random baseline"):
            grp = [all_head_tuples[i] for i in rng.choice(144, len(head_tuples), replace=False)]
            full_eff = ablate_group(grp)
            full_m = float(np.mean(full_eff))
            loo_s = np.zeros(n_prompts)
            for hd in grp:
                loo_s += ablate_group([hd])
            loo_m = float(np.mean(loo_s))
            e = 1.0 - (loo_m / (full_m + 1e-10)) if abs(full_m) > 1e-10 else 0.0
            random_eps.append(e)
            if (gi + 1) % 10 == 0:
                log(f"    {gi+1}/50 random groups done")

        p95 = float(np.percentile(random_eps, 95))
        baseline = {"scores": random_eps, "p95": p95, "mean": float(np.mean(random_eps))}
        save(baseline, f"step4_baseline_cluster_{cid}.json")

        h3_pass = ep["epistasis"] > 0.25 and ep["epistasis"] > p95
        epistasis_results[f"{cid}_h3"] = {
            "epistasis": ep["epistasis"], "ci": [ep["ci_lo"], ep["ci_hi"]],
            "random_p95": p95, "pass": h3_pass,
        }
        log(f"  H3 cluster {cid}: eps={ep['epistasis']:.3f} [{ep['ci_lo']:.3f}, {ep['ci_hi']:.3f}], "
            f"random p95={p95:.3f} — {'PASS' if h3_pass else 'FAIL'}")

    save(epistasis_results, "step4_epistasis.json")

    # ── H4: Null control ──
    log("H4: Null control (100 permutations)...")
    copier_indices = {i for i, h in enumerate(all_heads) if h in COPIER_SET}
    false_disc = 0
    rng = np.random.RandomState(42)

    with open("/app/features_gpt2_small.json") as f:
        feats_orig = json.load(f)
    for h in all_heads:
        ff = feats_orig[h]
        ff["ov_rank_ratio"] = ff["ov_effective_rank"] / D_MODEL
        ff["qk_rank_ratio"] = ff["qk_effective_rank"] / D_MODEL
        ff["ov_qk_rank_asymmetry"] = ff["ov_effective_rank"] - ff["qk_effective_rank"]
        ff["ov_qk_concentration_asymmetry"] = ff["ov_concentration"] - ff["qk_concentration"]
        ff["qk_ov_top_sv_align"] = feats_raw[h]["qk_ov_top_sv_align"]
        ff["qk_ov_top_right_align"] = feats_raw[h]["qk_ov_top_right_align"]
        ff["ov_unembed_norm"] = feats_raw[h]["ov_unembed_norm"]

    X_orig = np.array([[feats_orig[h][f] for f in CLUSTERING_FEATURES] for h in all_heads])
    X_orig = np.nan_to_num(X_orig, nan=0.0, posinf=0.0, neginf=0.0)

    for pi in range(100):
        X_perm = X_orig.copy()
        for col in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, col])
        sc = StandardScaler()
        Xp = sc.fit_transform(X_perm)
        cl = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3, metric="euclidean")
        lab = cl.fit_predict(Xp)
        for cid in set(lab):
            if cid == -1: continue
            members_idx = set(np.where(lab == cid)[0])
            if len(members_idx) < 3: continue
            member_heads = {(i // N_HEADS, i % N_HEADS) for i in members_idx}
            out_s, bg_s = [], []
            for (l1,h1_,l2,h2_), score in kcomp_cache.items():
                s_in = (l1,h1_) in member_heads
                r_in = (l2,h2_) in member_heads
                if s_in and not r_in: out_s.append(score)
                elif not s_in and not r_in: bg_s.append(score)
            if out_s and bg_s:
                r = np.mean(out_s) / (np.mean(bg_s) + 1e-10)
                if r > 1.5:
                    cop = len(members_idx & copier_indices)
                    if cop >= 5:
                        false_disc += 1
                        break
        if (pi + 1) % 20 == 0:
            log(f"  {pi+1}/100 permutations, {false_disc} false discoveries")

    h4 = {"n_permutations": 100, "false_discoveries": false_disc,
           "specificity": 1.0 - false_disc / 100, "pass": false_disc <= 5}
    save(h4, "h4_null_control.json")
    log(f"H4: {false_disc}/100 false discoveries — {'PASS' if h4['pass'] else 'FAIL'}")

    # ── H5: Novel groups (exploratory) ──
    log("H5: Novel group discovery...")
    h5 = {}
    for cid, info in kcomp_results.items():
        if not isinstance(info, dict): continue
        if not info.get("passes_1_5"): continue
        if info.get("copier_overlap", 0) >= 5: continue
        if info["size"] >= 3 and info["size"] <= 50:
            head_tuples = [parse_head(h) for h in info["members"]]
            log(f"  Non-copier cluster {cid} ({info['size']} heads): testing epistasis...")
            ep = compute_epistasis(head_tuples)
            h5[cid] = {"members": info["members"], "epistasis": ep, "passes": ep["epistasis"] > 0.20}
            save(h5[cid], f"h5_novel_cluster_{cid}.json")

    if not h5:
        h5 = {"status": "no_novel_groups_found"}
    save(h5, "h5_novel_groups.json")

    # ── Final summary ──
    summary = {
        "prereg_sha": PREREG_SHA,
        "timestamp": datetime.now().isoformat(),
        "correctness_gate": gate_results,
        "H1": h1, "H2": h2,
        "epistasis": epistasis_results,
        "H4": h4, "H5": h5,
        "clustering": {
            "n_clusters": n_clusters, "n_noise": n_noise,
            "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        },
    }
    save(summary, "E5_summary.json")

    log("\n=== RESULTS SUMMARY ===")
    log(f"Correctness gate: {'PASS' if gate_results['gate_pass'] else 'FAIL'}")
    log(f"H1 (copier recovery): {'PASS' if h1['pass'] else 'FAIL'}")
    log(f"H2 (K-comp filter): {'PASS' if h2['pass'] else 'FAIL'}")
    for k, v in epistasis_results.items():
        if isinstance(v, dict) and "pass" in v:
            log(f"H3 ({k}): {'PASS' if v['pass'] else 'FAIL'}")
    log(f"H4 (null control): {'PASS' if h4['pass'] else 'FAIL'}")
    log(f"H5 (novel groups): {len([v for v in h5.values() if isinstance(v, dict) and v.get('passes')])} found")
    log("Done.")


@app.local_entrypoint()
def main():
    import time
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Launching E5 on Modal A10G")
    run_e5.remote()
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Done in {elapsed:.0f}s")
