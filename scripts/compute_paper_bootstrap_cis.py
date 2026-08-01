"""Compute bootstrap CIs for all point estimates in the paper.

Uses per-prompt RTI data (302 prompts) for causal effects,
and existing per-category data for controls.

Output: paper_bootstrap_cis.json with all CIs for paper insertion.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "Documents/GitHub/factorization-circuits/MIB/MIB-circuit-track/weight_circuit/experiments/v2_second_investigation/raw_experiments/v1_role_weight_analysis"
PART3_DATA = DATA_DIR / "part3_l9h3_investigation/data"
PART4_DATA = DATA_DIR / "part4_rigorous_circuit_finding/data"
OUT = Path(__file__).parent / "paper_bootstrap_cis.json"

N_BOOT = 10_000


def bootstrap_ci(values, n_boot=N_BOOT, ci=0.95):
    values = np.array(values)
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


def bootstrap_ci_paired_diff(a, b, n_boot=N_BOOT, ci=0.95):
    a, b = np.array(a), np.array(b)
    diffs = a - b
    return bootstrap_ci(diffs, n_boot, ci)


def main():
    with open(PART3_DATA / "rti_per_prompt.json") as f:
        per_prompt = json.load(f)
    with open(PART4_DATA / "causal_tests.json") as f:
        causal = json.load(f)
    with open(PART3_DATA / "rti_controls.json") as f:
        controls = json.load(f)

    results = {}

    # --- 1. Main causal effects (per-prompt data available) ---

    baseline_lds = [p["baseline_logit_diff"] for p in per_prompt]
    cohort_lds = [p["cohort_logit_diff"] for p in per_prompt]
    full_circuit_lds = [p["full_circuit_logit_diff"] for p in per_prompt]
    random_lds = [p["random_logit_diff"] for p in per_prompt]

    results["baseline_ld"] = bootstrap_ci(baseline_lds)
    results["cohort_ablation_ld"] = bootstrap_ci(cohort_lds)
    results["full_circuit_ablation_ld"] = bootstrap_ci(full_circuit_lds)
    results["random_ablation_ld"] = bootstrap_ci(random_lds)

    # Deltas (paired differences from baseline)
    results["delta_full_circuit"] = bootstrap_ci_paired_diff(
        full_circuit_lds, baseline_lds
    )
    results["delta_cohort"] = bootstrap_ci_paired_diff(
        cohort_lds, baseline_lds
    )
    results["delta_random"] = bootstrap_ci_paired_diff(
        random_lds, baseline_lds
    )

    # --- 2. L0-only effect (per-category, bootstrap over categories) ---
    # We don't have per-prompt for controls, so report per-category stats
    ctrl_conditions = {}
    for cond_name in controls:
        cond = controls[cond_name]
        if "per_category" in cond:
            cat_means = []
            cat_ns = []
            for cat, info in cond["per_category"].items():
                cat_means.append(info["mean_ld"])
                cat_ns.append(info["n"])
            ctrl_conditions[cond_name] = {
                "mean_ld": float(cond["mean_logit_diff"]),
                "delta": float(cond["delta"]),
                "n_categories": len(cat_means),
                "category_mean_ld_values": cat_means,
                "category_ns": cat_ns,
                "note": "per-category means only; no per-prompt data for bootstrap"
            }
    results["controls_per_category"] = ctrl_conditions

    # --- 3. Sufficiency / completeness from causal_tests ---
    results["sufficiency"] = {
        "circuit_only_mean_ld": causal["faithfulness"]["circuit_only_mean_ld"],
        "baseline_mean_ld": causal["faithfulness"]["baseline_mean_ld"],
        "completeness_drop": causal["faithfulness"]["completeness_drop"],
        "bootstrap": causal["bootstrap"],
    }

    # --- 4. Minimality (LOO effects) ---
    minimality = causal["minimality"]
    results["minimality_loo"] = {}
    for head, val in minimality.items():
        results["minimality_loo"][head] = {
            "loo_effect": float(val),
            "note": "single LOO point estimate; bootstrap requires per-prompt LOO data (not available)"
        }

    # --- 5. Path patching ---
    results["path_patching"] = {}
    for path_name, path_data in causal["path_patching"].items():
        if isinstance(path_data, dict):
            results["path_patching"][path_name] = path_data

    # --- 6. IIA ---
    results["iia"] = causal["iia"]

    # --- 7. Faithfulness correlation ---
    results["faithfulness_correlation"] = {
        "r": causal["faithfulness"]["correlation"],
        "note": "Pearson r over prompts; CI requires per-prompt (clean, corrupt) pairs"
    }

    # --- 8. Summary: what has proper CIs vs what doesn't ---
    results["_coverage_summary"] = {
        "has_bootstrap_ci": [
            "baseline_ld", "cohort_ablation_ld", "full_circuit_ablation_ld",
            "random_ablation_ld", "delta_full_circuit", "delta_cohort",
            "delta_random", "sufficiency (via causal_tests bootstrap)"
        ],
        "needs_per_prompt_rerun": [
            "our_L0_only (Δ = −1.009)",
            "minimality LOO (15 values)",
            "degeneration rates (need multi-seed or per-prompt)",
            "path patching recovery fractions",
            "faithfulness correlation r",
        ],
        "deterministic_no_ci_needed": [
            "K-composition Z-scores (from bootstrap null distribution)",
            "OV eigenvalue scores",
            "OV effective rank",
            "Jaccard overlaps",
            "head counts",
        ]
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT}")
    print(f"  {len(results)} top-level sections")

    # Print summary for paper insertion
    print("\n=== KEY RESULTS WITH CIs ===\n")
    for key in ["baseline_ld", "delta_full_circuit", "delta_cohort", "delta_random"]:
        r = results[key]
        print(f"{key}: {r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] (n={r['n']})")

    bs = results["sufficiency"]["bootstrap"]
    print(f"\nSufficiency ablation effect: {bs['ablation_effect']['mean']:.3f} "
          f"[{bs['ablation_effect']['ci_lo']:.3f}, {bs['ablation_effect']['ci_hi']:.3f}]")
    print(f"Sufficiency baseline LD: {bs['baseline_ld']['mean']:.3f} "
          f"[{bs['baseline_ld']['ci_lo']:.3f}, {bs['baseline_ld']['ci_hi']:.3f}]")

    print("\n=== STILL NEED PER-PROMPT RERUNS ===")
    for item in results["_coverage_summary"]["needs_per_prompt_rerun"]:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
