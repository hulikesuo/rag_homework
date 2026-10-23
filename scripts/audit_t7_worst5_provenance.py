"""Generate machine-readable worst-5 audit provenance from actual data.

Only reads existing judge/T4/T1/QA outputs; never calls the API and never
modifies judge scores.  Writes JSON+CSV provenance under
repair_outputs/acceptance_20261023/t7/.
"""

from __future__ import annotations

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "repair_outputs" / "acceptance_20261023" / "t7"
QA = ROOT / "data/eval/qa_set.jsonl"
T4 = ROOT / "repair_outputs/revised_full_20261022/t4_per_query.json"
T1 = ROOT / "repair_outputs/revised_full_20261022/t1_baseline.jsonl"
CHECK = OUT / "judge_checkpoint.jsonl"

WORST5 = ["q01", "q22", "q28", "q02", "q13"]
EXTRA = ["q29"]


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError("expected JSON list: " + str(path))
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def select_last(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by[r["id"]] = r
    return by


def main() -> int:
    qas = {q["id"]: q for q in load_jsonl(QA)}
    t4_rows = {r["id"]: r for r in load_jsonl(T4)}
    t1_rows = select_last(load_jsonl(T1))
    judge_rows = select_last(load_jsonl(CHECK))

    records = []
    for rid in WORST5 + EXTRA:
        q = qas[rid]
        gold = q.get("gold_chunk_ids", [])
        top10 = t4_rows[rid]["modes"]["hybrid"].get("top10_ids") or []
        top5 = top10[:5]
        actual_top5 = t1_rows[rid].get("context_chunks") or t1_rows[rid].get("retrieved") or []
        jr = judge_rows[rid]
        record = {
            "id": rid,
            "level": q.get("level"),
            "correctness_0_5": jr.get("correctness_0_5"),
            "faithfulness_1_5": jr.get("faithfulness_1_5"),
            "original_failure_stage": jr.get("failure_stage"),
            "gold_chunk_ids": gold,
            "t4_hybrid_top10_ids": top10,
            "t4_hybrid_top10_hit": sorted(set(gold) & set(top10)),
            "t4_hybrid_top5_ids": top5,
            "t4_hybrid_top5_hit": sorted(set(gold) & set(top5)),
            "actual_top5_ids": actual_top5,
            "actual_top5_hit": sorted(set(gold) & set(actual_top5)),
        }
        records.append(record)

    payload = {
        "provenance_version": 3,
        "source": {
            "qa_set": str(QA),
            "t4_per_query_hybrid_top10_k": 10,
            "t1_cache": str(T1),
            "judge_checkpoint": str(CHECK),
        },
        "worst5": records[:5],
        "supplemental_6th": records[5:],
        "note": (
            "top10/top5 use T4 per-query hybrid mode only (same hybrid "
            "configuration as BASELINE_RETRIEVAL), not dense/bm25 and not "
            "different retrievers. use_top10_ranking_for_failure=False."
        ),
    }
    (OUT / "04_worst5_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    cols = ["id", "level", "correctness_0_5", "faithfulness_1_5",
            "original_failure_stage", "gold_chunk_ids", "top10_ids",
            "top10_hits", "top5_ids", "top5_hits", "actual_top5_ids",
            "actual_top5_hits"]
    with (OUT / "04_worst5_provenance.csv").open(
            "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([
                r["id"], r["level"], r["correctness_0_5"],
                r["faithfulness_1_5"], r["original_failure_stage"],
                "|".join(r["gold_chunk_ids"]),
                "|".join(r["t4_hybrid_top10_ids"]),
                "|".join(r["t4_hybrid_top10_hit"]),
                "|".join(r["t4_hybrid_top5_ids"]),
                "|".join(r["t4_hybrid_top5_hit"]),
                "|".join(r["actual_top5_ids"]),
                "|".join(r["actual_top5_hit"]),
            ])
    print("WROTE", OUT / "04_worst5_provenance.json")
    print("WROTE", OUT / "04_worst5_provenance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
