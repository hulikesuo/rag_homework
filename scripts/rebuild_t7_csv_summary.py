"""Rebuild T7 CSV/summary from checkpoint using last-wins per question id.

This script reads only existing checkpoint rows and writes the derived
summary artifacts.  It never calls the judge API and never modifies the
checkpoint.  Selection rule is documented below.

Selection rule:
- checkpoints are append-only; if multiple rows share an id the LAST appears
  in the checkpoint is the authoritative value (matches ``_read_checkpoint``
  in llm_judge_t7.py).
- for q01 specifically, the first row was an earlier pilot/connection-check
  (total_claims=4, correctness=3); the second/last row was the official batch
  judge (total_claims=7, correctness=2).  Both rows are retained in
  judge_checkpoint.jsonl; only the last is exported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "repair_outputs" / "acceptance_20261023" / "t7"
CHECKPOINT = OUT / "judge_checkpoint.jsonl"
QA = ROOT / "data/eval/qa_set.jsonl"
CACHE = ROOT / "repair_outputs/revised_full_20261022/t1_baseline.jsonl"


def load_last(path: pathlib.Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(vals):
    vals = [float(v) for v in vals if v not in (None, "")]
    return round(sum(vals) / len(vals), 4) if vals else None


def main() -> int:
    by_id = load_last(CHECKPOINT)
    qas = [json.loads(l) for l in QA.read_text(encoding="utf-8").splitlines()
           if l.strip()]
    order = [q["id"] for q in qas]
    missing = [i for i in order if i not in by_id]
    if missing:
        raise SystemExit("missing checkpoint rows: " + ", ".join(missing))
    rows = [by_id[i] for i in order]

    # Duplicate-id guard for provenance.
    raw_lines = [l for l in CHECKPOINT.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    raw_ids = [json.loads(l)["id"] for l in raw_lines]
    dup = {k: v for k, v in Counter(raw_ids).items() if v > 1}

    fields = list(rows[0].keys())
    with (OUT / "t7_llm_judge.csv").open(
            "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out_row = dict(r)
            for k, v in out_row.items():
                if isinstance(v, (dict, list)):
                    out_row[k] = json.dumps(v, ensure_ascii=False)
            w.writerow(out_row)

    levels = {}
    for level in ("L1", "L2", "L3", "L4"):
        rr = [r for r in rows if r["level"] == level]
        u = sum(int(r["unsupported_claims"] or 0) for r in rr)
        c = sum(int(r["total_claims"] or 0) for r in rr)
        macro = mean([r["hallucination_rate"] for r in rr])
        levels[level] = {
            "n": len(rr),
            "scored": sum(1 for r in rr if r["judge_status"] == "ok"),
            "faithfulness_1_5_mean": mean([r["faithfulness_1_5"] for r in rr]),
            "correctness_0_5_mean": mean([r["correctness_0_5"] for r in rr]),
            "l4_rubric_0_5_mean": mean(
                [r["L4_rubric_0_5"] for r in rr]) if level == "L4" else None,
            "hallucination_rate_macro": macro,
            "hallucination_rate_micro": round(u / c, 4) if c else None,
            "unsupported_total": u,
            "claims_total": c,
        }
    u = sum(int(r["unsupported_claims"] or 0) for r in rows)
    c = sum(int(r["total_claims"] or 0) for r in rows)
    levels["ALL"] = {
        "n": len(rows),
        "scored": sum(1 for r in rows if r["judge_status"] == "ok"),
        "faithfulness_1_5_mean": mean([r["faithfulness_1_5"] for r in rows]),
        "correctness_0_5_mean": mean([r["correctness_0_5"] for r in rows]),
        "l4_rubric_0_5_mean": mean(
            [r["L4_rubric_0_5"] for r in rows if r["level"] == "L4"]),
        "hallucination_rate_macro": mean(
            [r["hallucination_rate"] for r in rows]),
        "hallucination_rate_micro": round(u / c, 4) if c else None,
        "unsupported_total": u,
        "claims_total": c,
    }
    summary = {
        "metric_type": "llm_judge_not_human",
        "judge_model": rows[0].get("judge_model"),
        "judge_prompt_version": rows[0].get("prompt_version"),
        "prompt_template_sha256": rows[0].get("prompt_template_sha256")
        if rows[0].get("prompt_template_sha256") else
        "cfbe17242e02a1b1414366ac9017833a4c7fc715774327d66b5c5b56d947ce2c",
        "source_cache": str(CACHE),
        "source_cache_sha256": sha256_file(CACHE),
        "eval_fingerprint": rows[0].get("eval_fingerprint"),
        "n": len(rows),
        "ok": sum(1 for r in rows if r["judge_status"] == "ok"),
        "errors": sum(1 for r in rows if r["judge_status"] == "error"),
        "rejected_before_api": 0,
        "rejected_details": [],
        "levels": levels,
        "failure_stage_counts": dict(Counter(
            r["failure_stage"] for r in rows)),
        "hallucination_rate_micro": round(u / c, 4) if c else None,
    }
    (OUT / "t7_llm_judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    provenance = {
        "rebuild_rule": "last_wins_per_id",
        "duplicate_ids_in_checkpoint": {k: v for k, v in dup.items()},
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "csv_sha256": sha256_file(OUT / "t7_llm_judge.csv"),
        "summary_sha256": sha256_file(OUT / "t7_llm_judge_summary.json"),
        "api_calls_made": False,
    }
    (OUT / "rebuild_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("REBUILT", len(rows), "rows, claims", u, c,
          "meanC", levels["ALL"]["correctness_0_5_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
