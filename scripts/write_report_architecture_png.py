"""Generate a compact architecture PNG for the concise delivery report."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT = PROJECT_ROOT / "repair_outputs" / "acceptance_20261023" / "delivery" / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "fig_architecture.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, facecolor, edgecolor):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06",
                linewidth=1.1,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=7.6,
            color="#1c374b",
            wrap=True,
        )

    # top row
    box(0.3, 3.6, 2.0, 1.0, "43 DOCX\nloader + parser", "#eef4f8", "#2c5f7c")
    box(2.6, 3.6, 1.8, 1.0, "chunkers\nfixed/recursive/\nsentence + small_to_big", "#ebf3ee", "#3a6c4f")
    box(4.6, 3.6, 1.7, 1.0, "bge-small-zh-v1.5\n+ lexical spans", "#f7eef2", "#7a4f6c")
    box(6.5, 3.6, 1.6, 1.0, "BM25 / stores\nhybrid fuse", "#fff5e6", "#8a6a30")
    box(8.4, 3.6, 1.3, 1.0, "query opt\nbaseline/rewrite/\nmulti/subquery", "#f2f0f9", "#544e8a")

    # second row
    box(0.3, 1.8, 2.0, 1.0, "D4 results\nfrozen 42 top-level", "#eef4f8", "#2c5f7c")
    box(2.6, 1.8, 1.8, 1.0, "T2/T3/T4/T7/demo\nnew reviewed", "#ebf3ee", "#3a6c4f")
    box(4.6, 1.8, 1.7, 1.0, "rerank/compress\nlatency-quality", "#f7eef2", "#7a4f6c")
    box(6.5, 1.8, 1.6, 1.0, "generation\nglm-5.3-flash", "#fff5e6", "#8a6a30")
    box(8.4, 1.8, 1.3, 1.0, "j u d g e\nLLM + metrics", "#f2f0f9", "#544e8a")

    for x in (1.3, 3.5, 5.45, 7.3):
        ax.annotate(
            "",
            xy=(x + (1.7 if x < 6 else 0.9) / 2, 3.6),
            xytext=(x + 0.85, 2.8),
            arrowprops=dict(arrowstyle="->", color="#5b6b73", lw=1.0),
        )

    ax.text(
        5.0,
        0.35,
        "Offline verify: scripts/verify_acceptance_bundle.py + check_submission_acceptance.py",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#555f66",
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[architecture] {out}")


if __name__ == "__main__":
    main()
