"""Build the delivery-only T5 latency-quality trade-off figure.

This helper never writes to the formal results directory.  It joins rows by
question id and variant, skips rows with an empty/non-finite required value,
and treats direct top-3 reranking as an extra latency of exactly 0 ms.

Examples:
  python -m scripts.t5_delivery_figure
  python -m scripts.t5_delivery_figure --input results/t5_rerank.csv \
      --out-dir repair_outputs/acceptance_20261023/delivery
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = ("rules", "cross_encoder")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261023


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def load_pairs(path):
    """Return (pairs, rejected) using the direct row as the per-question base."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(dict)
    rejected = []
    for row in rows:
        qid = (row.get("id") or "").strip()
        variant = (row.get("variant") or "").strip()
        if not qid or not variant:
            rejected.append({"id": qid, "variant": variant, "reason": "missing id/variant"})
            continue
        if variant in grouped[qid]:
            rejected.append({"id": qid, "variant": variant, "reason": "duplicate id/variant"})
            continue
        grouped[qid][variant] = row

    pairs = []
    for qid in sorted(grouped):
        base = grouped[qid].get("direct")
        if base is None:
            rejected.append({"id": qid, "variant": "direct", "reason": "missing direct baseline"})
            continue
        base_rouge = _finite_number(base.get("rouge_l"))
        if base_rouge is None:
            rejected.append({"id": qid, "variant": "direct", "reason": "missing/non-finite rouge_l"})
            continue
        for variant in VARIANTS:
            row = grouped[qid].get(variant)
            if row is None:
                rejected.append({"id": qid, "variant": variant, "reason": "missing variant row"})
                continue
            rouge = _finite_number(row.get("rouge_l"))
            # direct has no reranking stage; its extra latency is the defined 0 ms.
            extra_ms = _finite_number(row.get("rerank_ms"))
            if rouge is None:
                rejected.append({"id": qid, "variant": variant, "reason": "missing/non-finite rouge_l"})
                continue
            if extra_ms is None or extra_ms < 0:
                rejected.append({"id": qid, "variant": variant, "reason": "missing/non-finite rerank_ms"})
                continue
            pairs.append({
                "id": qid,
                "variant": variant,
                "baseline_rouge_l": base_rouge,
                "rouge_l": rouge,
                "delta_rouge_l": rouge - base_rouge,
                "extra_rerank_ms": extra_ms,
            })
    return pairs, rejected


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize(pairs):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["variant"]].append(pair)
    result = {}
    for variant in VARIANTS:
        rows = grouped[variant]
        deltas = [row["delta_rouge_l"] for row in rows]
        result[variant] = {
            "n_pairs": len(rows),
            "extra_rerank_ms_mean": _mean([row["extra_rerank_ms"] for row in rows]),
            "rouge_l_mean": _mean([row["rouge_l"] for row in rows]),
            "baseline_rouge_l_mean": _mean([row["baseline_rouge_l"] for row in rows]),
            "delta_rouge_l_mean": _mean(deltas),
            "improved_count": sum(delta > 0 for delta in deltas),
            "worse_count": sum(delta < 0 for delta in deltas),
            "tie_count": sum(delta == 0 for delta in deltas),
        }
    return result


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_delta_ci(pairs, variant):
    """Deterministic percentile bootstrap over questions, not generated records."""
    # A tiny linear-congruential PRNG keeps this helper stdlib-only and repeatable.
    state = BOOTSTRAP_SEED

    def random_index(n):
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % n

    deltas = [row["delta_rouge_l"] for row in pairs if row["variant"] == variant]
    if not deltas:
        return {"n_pairs": 0, "mean": None, "ci95": [None, None]}
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [deltas[random_index(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    return {
        "n_pairs": len(deltas),
        "mean": sum(deltas) / len(deltas),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "method": "deterministic percentile bootstrap over paired questions",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def draw(pairs, summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c9c9c9",
        "grid.color": "#e7e7e7",
        "axes.axisbelow": True,
    })
    colors = {"rules": "#d97732", "cross_encoder": "#2474a6"}
    labels = {"rules": "规则重排", "cross_encoder": "CrossEncoder 重排"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for variant in VARIANTS:
        rows = [row for row in pairs if row["variant"] == variant]
        ax.scatter([row["extra_rerank_ms"] for row in rows],
                   [row["rouge_l"] for row in rows],
                   s=18, alpha=0.35, color=colors[variant], edgecolors="none")
    direct_rows = [row for row in pairs if row["variant"] == "rules"]
    if direct_rows:
        ax.scatter([0] * len(direct_rows), [row["baseline_rouge_l"] for row in direct_rows],
                   s=18, alpha=0.28, color="#6d6d6d", edgecolors="none",
                   label="同题 direct top-3（额外延迟定义为 0 ms）")

    for variant in VARIANTS:
        item = summary[variant]
        if item["n_pairs"]:
            ax.scatter([item["extra_rerank_ms_mean"]], [item["rouge_l_mean"]],
                       s=90, color=colors[variant], edgecolors="white", linewidths=1.5,
                       zorder=3, label=f"{labels[variant]}均值 (n={item['n_pairs']})")
            ax.annotate(f"ΔROUGE-L={item['delta_rouge_l_mean']:+.4f}\n"
                        f"+{item['extra_rerank_ms_mean']:.1f} ms",
                        (item["extra_rerank_ms_mean"], item["rouge_l_mean"]),
                        textcoords="offset points", xytext=(9, 9), fontsize=8)

    ax.set_xlabel("相对 direct top-3 的额外重排延迟（ms，逐题配对）")
    ax.set_ylabel("ROUGE-L（字符级代理，不是语义正确率）")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-5)
    ax.grid(True)
    ax.set_title("T5 额外延迟—答案质量权衡（同题配对）")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build(input_path, out_dir):
    pairs, rejected = load_pairs(input_path)
    summary = summarize(pairs)
    summary["direct"] = {
        "n_pairs": len({row["id"] for row in pairs}),
        "extra_rerank_ms_mean": 0.0,
        "definition": "direct top-3 has no reranking stage; extra latency is defined as 0 ms",
    }
    summary["pairing"] = {
        "key": ["id"],
        "baseline_variant": "direct",
        "compared_variants": list(VARIANTS),
        "empty_quality_or_latency_rows_are_not_zero_filled": True,
        "rejected_rows": rejected,
    }
    summary["bootstrap_delta_rouge_l"] = {
        variant: bootstrap_delta_ci(pairs, variant) for variant in VARIANTS
    }
    summary["source"] = {
        "path": os.path.abspath(input_path),
        "note": "Formal results/t5_rerank.csv is read-only input.",
    }

    os.makedirs(out_dir, exist_ok=True)
    pair_path = os.path.join(out_dir, "t5_latency_quality_pairs.csv")
    with open(pair_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "id", "variant", "baseline_variant", "baseline_rouge_l",
            "rouge_l", "delta_rouge_l", "extra_rerank_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairs:
            writer.writerow({**row, "baseline_variant": "direct"})
    summary_path = os.path.join(out_dir, "t5_latency_quality_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    figure_path = os.path.join(out_dir, "fig_t5_latency_quality_delivery.png")
    draw(pairs, summary, figure_path)
    print(f"[T5] pairs={len(pairs)} rejected={len(rejected)}")
    print(f"[T5] figure={figure_path}")
    print(f"[T5] pairs_csv={pair_path}")
    print(f"[T5] summary={summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join("results", "t5_rerank.csv"))
    parser.add_argument(
        "--out-dir",
        default=os.path.join("repair_outputs", "acceptance_20261023", "delivery", "figures"),
    )
    args = parser.parse_args()
    build(args.input, args.out_dir)


if __name__ == "__main__":
    main()
