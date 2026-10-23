"""Offline checks for the delivery-only T5 paired trade-off calculation."""

import csv
import json
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.t5_delivery_figure import load_pairs, summarize

FIELDS = ["id", "variant", "rouge_l", "rerank_ms"]


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t5.csv")
        rows = [
            {"id": "q1", "variant": "direct", "rouge_l": "0.2", "rerank_ms": ""},
            {"id": "q1", "variant": "rules", "rouge_l": "0.3", "rerank_ms": "2"},
            {"id": "q1", "variant": "cross_encoder", "rouge_l": "0.1", "rerank_ms": "200"},
            {"id": "q2", "variant": "direct", "rouge_l": "0.4", "rerank_ms": ""},
            {"id": "q2", "variant": "rules", "rouge_l": "", "rerank_ms": "2"},
            {"id": "q2", "variant": "cross_encoder", "rouge_l": "0.5", "rerank_ms": ""},
            {"id": "q3", "variant": "direct", "rouge_l": "0.6", "rerank_ms": ""},
            {"id": "q3", "variant": "rules", "rouge_l": "0.6", "rerank_ms": "0"},
            {"id": "q3", "variant": "cross_encoder", "rouge_l": "0.6", "rerank_ms": "100"},
        ]
        _write_csv(path, rows)
        pairs, rejected = load_pairs(path)

        assert len(pairs) == 4, pairs
        assert len(rejected) == 2, rejected
        rules = summarize(pairs)["rules"]
        cross = summarize(pairs)["cross_encoder"]
        assert rules["n_pairs"] == 2
        assert cross["n_pairs"] == 2
        assert rules["extra_rerank_ms_mean"] == 1.0
        assert abs(rules["delta_rouge_l_mean"] - 0.05) < 1e-12
        # q1: 0.1 - 0.2 = -0.1; q3: 0.6 - 0.6 = 0.0
        assert abs(cross["delta_rouge_l_mean"] + 0.05) < 1e-12
        assert all(row["baseline_rouge_l"] != 0 for row in pairs
                   if row["id"] in {"q1", "q3"})
        checks.extend([
            "pairing: same id and variant joined; direct baseline",
            "missing quality/latency rows rejected, not filled with zero",
            "direct extra latency explicitly defined as 0 ms",
        ])

    with open(os.path.join("results", "t5_rerank.csv"), encoding="utf-8-sig") as handle:
        formal = list(csv.DictReader(handle))
    summary_path = os.path.join(
        "repair_outputs", "acceptance_20261023", "delivery", "figures",
        "t5_latency_quality_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        assert summary["rules"]["n_pairs"] == 30
        assert summary["cross_encoder"]["n_pairs"] == 30
        assert summary["pairing"]["rejected_rows"] == []
        assert len(formal) == 90
        checks.extend([
            "formal t5_rerank.csv has 90 rows",
            "delivery summary has 30 paired questions per variant and no rejects",
        ])
    else:
        raise AssertionError(f"missing generated summary: {summary_path}")

    print(f"T5 delivery checks: {len(checks)}/{len(checks)} passed")
    for check in checks:
        print(f"  PASS {check}")


if __name__ == "__main__":
    run()
