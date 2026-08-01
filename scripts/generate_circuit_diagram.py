"""Generate RTI circuit diagram for the paper.

4-tier node-and-edge diagram with path patching edge widths.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EDGE_FILE = SCRIPT_DIR / "paper_numbers" / "E4i_path_patching_proper" / "edge_results.json"
OUT_DIR = SCRIPT_DIR / "generated_figures"

BACKBONE = [(0, 8), (0, 9), (0, 11)]
DETECTOR = [(4, 11)]
COPIER = [(4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10)]
READOUT = [(10, 11), (11, 9), (11, 11)]

TIER_COLORS = {
    "Backbone": "#4A90D9",
    "Detector": "#E8A838",
    "Copier": "#D96459",
    "Readout": "#6DBD6D",
}

TIER_BG = {
    "Backbone": "#D6E6F5",
    "Detector": "#FDE8C8",
    "Copier": "#F5D5D2",
    "Readout": "#D2EDD2",
}


def head_label(l, h):
    return f"L{l}H{h}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EDGE_FILE) as f:
        edge_data = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(10, 11))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    tier_y = {
        "Backbone": 9.0,
        "Detector": 6.5,
        "Copier": 3.5,
        "Readout": 0.8,
    }

    tier_heads = {
        "Backbone": BACKBONE,
        "Detector": DETECTOR,
        "Copier": COPIER,
        "Readout": READOUT,
    }

    tier_labels = {
        "Backbone": "Backbone (L0)",
        "Detector": "Detector (L4)",
        "Copier": "Copier (L4–L9)",
        "Readout": "Readout (L10–L11)",
    }

    tier_sublabels = {
        "Backbone": "positional encoding",
        "Detector": "previous-token head",
        "Copier": "attend unique, boost logits",
        "Readout": "induction + amplification",
    }

    head_positions = {}

    for tier_name, heads in tier_heads.items():
        y = tier_y[tier_name]
        n = len(heads)
        total_width = n * 1.1
        x_start = 5.0 - total_width / 2 + 0.55

        bg_pad = 0.5
        bg = mpatches.FancyBboxPatch(
            (x_start - bg_pad - 0.15, y - 0.55),
            total_width + 2 * bg_pad - 0.7,
            1.1,
            boxstyle="round,pad=0.15",
            facecolor=TIER_BG[tier_name],
            edgecolor=TIER_COLORS[tier_name],
            linewidth=1.5,
            alpha=0.6,
        )
        ax.add_patch(bg)

        ax.text(
            x_start - bg_pad + 0.1,
            y + 0.75,
            tier_labels[tier_name],
            fontsize=11,
            fontweight="bold",
            color=TIER_COLORS[tier_name],
            va="bottom",
        )
        ax.text(
            x_start - bg_pad + 0.1,
            y + 0.58,
            tier_sublabels[tier_name],
            fontsize=7.5,
            fontstyle="italic",
            color="#666666",
            va="bottom",
        )

        for i, (l, h) in enumerate(heads):
            x = x_start + i * 1.1
            head_positions[(l, h)] = (x, y)

            box = mpatches.FancyBboxPatch(
                (x - 0.42, y - 0.3),
                0.84,
                0.6,
                boxstyle="round,pad=0.08",
                facecolor=TIER_COLORS[tier_name],
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_patch(box)
            ax.text(
                x, y,
                head_label(l, h),
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
            )

    # Draw edges with width proportional to effect
    edge_effects = {}
    for key, data in edge_data.items():
        parts = key.split("->")
        sl = int(parts[0][1:].split("H")[0])
        sh = int(parts[0].split("H")[1])
        rl = int(parts[1][1:].split("H")[0])
        rh = int(parts[1].split("H")[1])
        effect = data["bootstrap"]["mean"]
        edge_effects[(sl, sh, rl, rh)] = effect

    max_effect = max(abs(v) for v in edge_effects.values())

    for (sl, sh, rl, rh), effect in edge_effects.items():
        if (sl, sh) not in head_positions or (rl, rh) not in head_positions:
            continue

        abs_effect = abs(effect)
        if abs_effect < 0.005:
            continue

        x1, y1 = head_positions[(sl, sh)]
        x2, y2 = head_positions[(rl, rh)]

        width = 0.3 + 3.5 * (abs_effect / max_effect)
        alpha = 0.15 + 0.7 * (abs_effect / max_effect)

        color = "#CC3333" if effect < 0 else "#444444"

        ax.annotate(
            "",
            xy=(x2, y2 + 0.32),
            xytext=(x1, y1 - 0.32),
            arrowprops=dict(
                arrowstyle="->,head_width=0.15,head_length=0.1",
                color=color,
                lw=width,
                alpha=alpha,
                connectionstyle="arc3,rad=0.05",
            ),
        )

    # Title
    ax.text(
        5.0, 10.3,
        "RTI Circuit: 15 Heads in 4 Tiers",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold",
    )
    ax.text(
        5.0, 10.0,
        "Edge width proportional to path-patching effect (302 prompts)",
        ha="center", va="bottom",
        fontsize=9, color="#666666",
    )

    # Key stats annotation
    stats_text = (
        "Backbone→Detector: 98% recovery\n"
        "Copier→Readout: 47% recovery\n"
        "63% of total effect from interactions\n"
        "L0H9: dominant hub (top 4 edges)"
    )
    ax.text(
        10.5, 0.0,
        stats_text,
        ha="right", va="bottom",
        fontsize=7.5,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", alpha=0.9),
    )

    # Legend for edge colors
    ax.plot([], [], color="#444444", linewidth=2, label="Positive effect")
    ax.plot([], [], color="#CC3333", linewidth=2, label="Inhibitory effect")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out_path = OUT_DIR / f"rti_circuit_diagram.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved {out_path}")

    plt.close()


if __name__ == "__main__":
    main()
