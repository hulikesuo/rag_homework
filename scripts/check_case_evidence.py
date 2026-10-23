#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only evidence auditor for the revised 30-question project.

This checker does not modify or re-run anything.  It verifies that the
hand-written case evidence is consistent with the real JSON/CSV sources.
The T4 supplement intentionally fails until a validated 5+5 dense/bm25
subset is produced, so a non-zero exit is expected today.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "repair_outputs" / "revised_full_20261022"
QA = ROOT / "data" / "eval" / "qa_set.jsonl"
FROZEN = ROOT / "scripts" / "t4_supplement_queries.jsonl"

DISALLOWED_T1 = {
    "未标注gold", "题集未标注gold文档", "题集缺失gold",
    "生成基本正确但gold口径差异",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> List[dict]:
    """Read CSV while preserving quoted newlines inside cells."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(text) -> str:
    return str(text).replace("\r\n", "\n")


def winner_from_modes(modes: dict) -> str:
    """Strict dense vs bm25: Recall@3 first, then MRR@10, ties stay tie."""
    scores = {}
    for name in ("bm25", "dense"):
        mode = modes.get(name, {})
        try:
            scores[name] = (float(mode["Recall@3"]), float(mode["MRR@10"]))
        except (TypeError, KeyError, ValueError):
            raise ValueError(f"missing numeric Recall@3/MRR@10 for {name}")
    if scores["dense"] == scores["bm25"]:
        return "tie"
    return "dense" if scores["dense"] > scores["bm25"] else "bm25"


def collect_errors(errors: List[str], name: str, condition: bool, message: str) -> None:
    if not condition:
        errors.append(f"{name}: {message}")


def verify_t1(errors: List[str]) -> None:
    t1_path = OUT / "t1_case_analysis.md"
    if not t1_path.is_file():
        errors.append("T1: missing t1_case_analysis.md")
        return
    md = t1_path.read_text(encoding="utf-8")
    for token in ("q06", "q07", "q22", "q28", "q13", "gold_chunk_ids"):
        collect_errors(errors, "T1", token in md, f"md missing {token}")
    for token in DISALLOWED_T1:
        collect_errors(errors, "T1", token not in md, f"md contains forbidden {token!r}")

    qa_rows = read_jsonl(QA)
    t1_rows = read_jsonl(OUT / "t1_baseline.jsonl")
    collect_errors(errors, "T1/ids",
                   len(qa_rows) == 30 and len({r["id"] for r in qa_rows}) == 30,
                   f"QA is not 30 unique rows ({len(qa_rows)} rows, "
                   f"{len({r['id'] for r in qa_rows})} unique ids)")
    collect_errors(errors, "T1/ids",
                   len(t1_rows) == 30 and len({r["id"] for r in t1_rows}) == 30,
                   f"T1 is not 30 unique rows ({len(t1_rows)} rows, "
                   f"{len({r['id'] for r in t1_rows})} unique ids)")
    qa = {r["id"]: r for r in qa_rows}
    t1 = {r["id"]: r for r in t1_rows}
    collect_errors(errors, "T1",
                   sha256(QA) in md,
                   "qa_set.jsonl sha256 not present in md")
    collect_errors(errors, "T1",
                   sha256(OUT / "t1_baseline.jsonl") in md,
                   "t1_baseline.jsonl sha256 not present in md")

    # q13: the QA record has gold; the md must claim the modern 2026 file
    # was missed, not that the gold is absent.
    q13 = qa["q13"]
    t13 = t1["q13"]
    collect_errors(errors, "T1/q13", bool(q13["gold_chunk_ids"]),
                   "QA q13 has empty gold_chunk_ids")
    collect_errors(errors, "T1/q13", norm(q13["question"]) in norm(md),
                   "md does not reproduce q13 question text")
    collect_errors(errors, "T1/q13",
                   t13["gold_chunk_ids"] == q13["gold_chunk_ids"],
                   "T1 record gold_chunk_ids differs from QA")
    context_ids = [str(x) for x in (t13.get("context_chunks") or [])]
    modern_hit = any("05_软件学院2026年接收推免生实施细则.docx" in x
                     for x in context_ids)
    collect_errors(errors, "T1/q13", not modern_hit,
                   "T1 q13 context unexpectedly contains the 2026 file")
    collect_errors(errors, "T1/q13",
                   q13["gold_chunk_ids"] == [
                       "04_保研推免\\05_软件学院2026年接收推免生实施细则.docx__004",
                       "04_保研推免\\05_软件学院2026年接收推免生实施细则.docx__005"],
                   "QA q13 gold ids unexpectedly changed")
    collect_errors(errors, "T1/q13",
                   "检索失败" in md and "2026" in md,
                   "md does not state q13 retrieval failed against 2026 doc")

    # q22: the real model answer uses the wrong column semantics.
    q22 = qa["q22"]
    t22 = t1["q22"]
    collect_errors(errors, "T1/q22",
                   "36和18.0是讲授36学时、实验18学时"
                   in norm(q22["gold_answer"]),
                   "QA q22 gold no longer encodes the official column meaning")
    answer = norm(t22.get("answer") or "")
    collect_errors(errors, "T1/q22",
                   "实验/上机" in answer or "实训" in answer,
                   "T1 q22 answer no longer shows a wrong column interpretation")
    collect_errors(errors, "T1/q22",
                   "36+18=54" in md or "36 + 18 = 54" in md,
                   "md does not explicitly reject 36+18=54 as sufficient")
    collect_errors(errors, "T1/q22",
                   "列义" in md, "md does not flag that the error is column semantics")


def verify_t2(errors: List[str]) -> None:
    md_path = OUT / "t2_chunking_failure_case.md"
    if not md_path.is_file():
        errors.append("T2: missing t2_chunking_failure_case.md")
        return
    md = md_path.read_text(encoding="utf-8")
    for token in ("256/128", "512/128", "__011", "__004", "教学方式",
                  "知识点讲解及案例演示", "位置", "适当拒答"):
        collect_errors(errors, "T2", token in md, f"md missing {token!r}")
    collect_errors(errors, "T2", "基本正确" not in md,
                   "md contains vague '基本正确'")
    csv_path = OUT / "t2_chunking_cases.csv"
    collect_errors(errors, "T2", sha256(csv_path) in md,
                   "t2_chunking_cases.csv sha256 not present in md")
    rows = read_rows(csv_path)
    target = [r for r in rows
              if r["id"] == "q06" and r["method"] == "fixed"
              and r["size"] == "512" and r["overlap"] == "128"]
    if len(target) != 1:
        errors.append(f"T2: expected one fixed/512/128 q06 row, got {len(target)}")
        return
    row = target[0]
    recalled = json.loads(row["retrieved"])
    positions = {chunk: i for i, chunk in enumerate(recalled, 1)}
    gold = json.loads(row["gold_evidence"])[0]["chunk_id"]
    collect_errors(errors, "T2/512-128",
                   positions.get(gold) == 4,
                   f"gold {gold!r} not at rank 4; actual={sorted(positions.items())}")
    collect_errors(errors, "T2/512-128",
                   row["Recall@5_evidence"] == "1.0",
                   "Recall@5_evidence is not 1.0")
    collect_errors(errors, "T2/512-128",
                   "rank#4" in md and "__004" in md,
                   "md does not identify gold chunk __004 at rank4")
    bad = [r for r in rows
           if r["id"] == "q06" and r["method"] == "fixed"
           and r["size"] == "256" and r["overlap"] == "128"]
    if len(bad) != 1:
        errors.append(f"T2: expected one fixed/256/128 q06 row, got {len(bad)}")
    else:
        bad_row = bad[0]
        bad_ids = json.loads(bad_row["retrieved"])
        collect_errors(errors, "T2/256-128",
                       bad_ids[2].endswith("__011"),
                       f"256/128 rank3 is not __011: {bad_ids}")
        collect_errors(errors, "T2/256-128",
                       gold not in bad_ids,
                       "256/128 unexpectedly contains gold __004 in top5")


def parse_audit_rows(errors: List[str], md: str) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    pattern = re.compile(
        r"^\|\s*(t4s_[A-Za-z0-9_]+)\s*\|\s*"
        r"(valid|partial|invalid|ambiguous)\s*\|\s*"
        r"(dense|bm25|tie)\s*\|")
    header_seen = "| id | 审核 | 两路胜 |" in md
    collect_errors(errors, "T4/audit", header_seen,
                   "audit md table header not found")
    for line in md.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rid, cls, winner = match.groups()
        if rid in rows:
            errors.append(f"T4/audit: duplicate row for {rid}")
        rows[rid] = {"class": cls, "winner": winner}
    return rows


def verify_t4(errors: List[str]) -> Dict[str, Dict[str, int]]:
    if not FROZEN.is_file():
        errors.append("T4: missing frozen queries file")
        return {}
    frozen = [json.loads(x) for x in
              FROZEN.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    frozen_ids = [x["id"] for x in frozen]
    if len(frozen_ids) != len(set(frozen_ids)):
        errors.append("T4/frozen: duplicate IDs")

    perq_path = OUT / "t4_supplement_per_query.json"
    if not perq_path.is_file():
        errors.append("T4: missing t4_supplement_per_query.json")
        return {}
    data = json.loads(perq_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        errors.append("T4: per_query is not a list")
        return {}
    per_ids = [x["id"] for x in data]
    if len(per_ids) != 28 or len(set(per_ids)) != 28:
        errors.append(f"T4: per_query unexpected size/duplicates {len(per_ids)}")
    if set(per_ids) != set(frozen_ids):
        errors.append("T4: per_query IDs differ from frozen queries")
    else:
        per_map = {r["id"]: r for r in data}
        frozen_map = {r["id"]: r for r in frozen}
        for idx, rid in enumerate(frozen_ids, 1):
            per = per_map[rid]
            fz = frozen_map[rid]
            for key in ("question", "gold_chunk_id", "expect_winner"):
                if per.get(key) != fz.get(key):
                    errors.append(
                        f"T4/frozen_mismatch/{rid}: {key} differs "
                        f"(per={per.get(key)!r}, frozen={fz.get(key)!r})")
            if per.get("query_index") != idx:
                errors.append(
                    f"T4/frozen_order/{rid}: query_index={per.get('query_index')}, "
                    f"expected {idx}")

    audit_path = OUT / "t4_supplement_audit.md"
    if not audit_path.is_file():
        errors.append("T4: missing t4_supplement_audit.md")
        return {}
    audit_md = audit_path.read_text(encoding="utf-8")
    classification = parse_audit_rows(errors, audit_md)
    missing_cls = set(per_ids) - set(classification)
    extra_cls = set(classification) - set(per_ids)
    if missing_cls:
        errors.append(f"T4/audit: no classification for {sorted(missing_cls)}")
    if extra_cls:
        errors.append(f"T4/audit: unexpected classification for {sorted(extra_cls)}")
    if sha256(FROZEN) not in audit_md:
        errors.append("T4/audit: frozen queries sha256 not present")
    if sha256(perq_path) not in audit_md:
        errors.append("T4/audit: per_query sha256 not present")

    computed_full: Dict[str, int] = {"dense": 0, "bm25": 0, "tie": 0}
    computed_valid: Dict[str, int] = {"dense": 0, "bm25": 0, "tie": 0}
    actual_full: Dict[str, int] = {"dense": 0, "bm25": 0, "tie": 0}
    mismatches = 0
    ties_stored_as_bm25 = 0
    for row in data:
        try:
            computed = winner_from_modes(row.get("modes", {}))
        except ValueError as exc:
            errors.append(f"T4/{row.get('id')}: {exc}")
            continue
        actual = row.get("actual_winner")
        actual_full[actual] = actual_full.get(actual, 0) + 1
        computed_full[computed] += 1
        if actual != computed:
            mismatches += 1
            if actual == "bm25" and computed == "tie":
                ties_stored_as_bm25 += 1
            errors.append(
                f"T4/archived_winner_mismatch/{row['id']}: "
                f"actual_winner={actual!r}, strict winner={computed!r}")
        row["_strict_winner"] = computed
        audit = classification.get(row["id"])
        cls = (audit or {}).get("class")
        if cls == "valid":
            computed_valid[computed] += 1
        if audit and audit.get("winner") != computed:
            errors.append(
                f"T4/audit_winner_mismatch/{row['id']}: "
                f"audit says {audit.get('winner')!r}, strict winner={computed!r}")

    print(f"T4 full  strict: dense={computed_full['dense']} "
          f"bm25={computed_full['bm25']} tie={computed_full['tie']}")
    print(f"T4 archived_winner_mismatch={mismatches} "
          f"actual stored dense={actual_full['dense']} "
          f"bm25={actual_full['bm25']} tie={actual_full['tie']}"
          f" (ties wrongly stored as bm25: {ties_stored_as_bm25})")
    print(f"T4 valid strict: dense={computed_valid['dense']} "
          f"bm25={computed_valid['bm25']} tie={computed_valid['tie']}")
    collect_errors(errors, "T4/archived_winner_mismatch", mismatches == 0,
                   f"{mismatches} rows have actual_winner inconsistent "
                   "with strict dense-vs-bm25 rule")
    collect_errors(errors, "T4/valid", computed_valid["dense"] >= 5,
                   f"valid dense wins {computed_valid['dense']} < 5")
    collect_errors(errors, "T4/valid", computed_valid["bm25"] >= 5,
                   f"valid bm25 wins {computed_valid['bm25']} < 5")
    return {"full": computed_full, "valid": computed_valid}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    errors: List[str] = []
    verify_t1(errors)
    verify_t2(errors)
    verify_t4(errors)
    if not errors:
        print("ALL CHECKS PASSED")
        return 0
    print("\nERRORS:")
    for error in errors:
        print("  FAIL:", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
