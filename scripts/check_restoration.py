#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent verification for scripts/restore_submission_results.py.

The checker is intentionally a separate module so it does not import the
restoring code by accident.  It can be run before and after applying.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE_DIR = ROOT / "repair_outputs" / "revised_full_20261022"
EXPECTED_QA_IDS = [f"q{i:02d}" for i in range(1, 31)]
T1_FINGERPRINT = "9d8055917ee6f4b4a9fe523f09f0917e8cbdab1565ff08fd10926c143fabcf41"

EXPECTED_FILES = {
    "t1_baseline.jsonl", "t1_baseline_summary.csv", "t1_rag_vs_direct.csv",
    "t2_chunking.csv", "t2_chunking_cases.csv", "t2_retrieval_differences.csv",
    "t3_query_opt.csv", "t3_cases_baseline.csv", "t3_cases_multi.csv",
    "t3_cases_rewrite.csv", "t3_cases_subquery.csv",
    "t3_linear_alpha_sweep.csv", "t3_success_cases.csv",
    "t3_success_counts.csv", "t4_retriever.csv", "t4_per_query.csv",
    "t4_per_query.json", "t4_cases.csv", "t4_cases.json", "t4_cases.md",
    "t5_rerank.csv", "t5_compress.csv", "t6_routing.csv", "t6_support.csv",
    "t6_retry.csv", "t6_retry_trace.jsonl", "t6_retry_summary.json",
    "t7_metrics.csv", "t7_faithfulness.csv", "t7_llm_judge.csv",
    "t7_llm_judge_summary.json", "fig_t1_rag_vs_direct.png",
    "fig_t2_chunking.png", "fig_t3_alpha.png", "fig_t3_query_opt.png",
    "fig_t4_retriever.png", "fig_t5_rerank_proxy.png", "fig_t6_flow.png",
    "fig_t7_level.png", "t7_manual_scoring.csv", "t7_generation_status.csv",
    "submission_manifest.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_final_files() -> None:
    actual = {p.name for p in RESULTS.iterdir() if p.is_file()}
    missing = EXPECTED_FILES - actual
    extra = actual - EXPECTED_FILES
    if missing:
        raise ValueError(f"Missing official files: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected official result files: {sorted(extra)}")


def check_source_qa_annotation() -> None:
    qa = read_jsonl(ROOT / "data/eval/qa_set.jsonl")
    if [row["id"] for row in qa] != EXPECTED_QA_IDS:
        raise ValueError("Q/A id order differs")
    annotation = read_rows(ROOT / "scripts" / "qa_annotation.csv")
    if [row["id"] for row in annotation] != EXPECTED_QA_IDS:
        raise ValueError("annotation id order differs")
    qa_map = {row["id"]: row for row in qa}
    for row in annotation:
        q = qa_map[row["id"]]
        for key in ("level", "question", "gold_answer"):
            if str(row[key]).replace("\r\n", "\n") != str(q[key]).replace(
                    "\r\n", "\n"):
                raise ValueError(f"{row['id']} annotation {key} differs")
        if row["gold_chunk_ids"].split("|") != q["gold_chunk_ids"]:
            raise ValueError(f"{row['id']} annotation gold_chunk_ids differs")


def check_t1_manual(manifest: dict) -> None:
    t1_flat = read_jsonl(RESULTS / "t1_baseline.jsonl")
    manual_flat = read_rows(RESULTS / "t7_manual_scoring.csv")
    qa_flat = read_jsonl(ROOT / "data/eval/qa_set.jsonl")
    if len(t1_flat) != 30 or len({row["id"] for row in t1_flat}) != 30:
        raise ValueError("final T1 is not 30 unique rows")
    if len(manual_flat) != 30 or len({row["id"] for row in manual_flat}) != 30:
        raise ValueError("final manual is not 30 unique rows")
    if len(qa_flat) != 30 or len({row["id"] for row in qa_flat}) != 30:
        raise ValueError("final Q/A is not 30 unique rows")
    t1 = {row["id"]: row for row in t1_flat}
    manual_rows = {row["id"]: row for row in manual_flat}
    qa_rows = {row["id"]: row for row in qa_flat}
    if len(t1) != 30 or len(manual_rows) != 30:
        raise ValueError("final T1/manual do not have 30 rows")
    if set(t1) != set(manual_rows) or set(t1) != set(qa_rows):
        raise ValueError("final T1/manual/QA id sets differ")
    t1_fp = {row["eval_fingerprint"] for row in t1.values()}
    if t1_fp != {T1_FINGERPRINT}:
        raise ValueError("final T1 fingerprint differs")
    for rid, row in t1.items():
        qa = qa_rows[rid]
        man = manual_rows[rid]
        for key in ("question", "gold_answer", "level"):
            if str(row[key]).replace("\r\n", "\n") != str(qa[key]).replace(
                    "\r\n", "\n"):
                raise ValueError(f"{rid} final T1 {key} differs")
            if str(man[key]).replace("\r\n", "\n") != str(qa[key]).replace(
                    "\r\n", "\n"):
                raise ValueError(f"{rid} final manual {key} differs")
        if row["gold_chunk_ids"] != qa["gold_chunk_ids"]:
            raise ValueError(f"{rid} final T1 gold_chunk_ids differs")
        if json.loads(man["gold_chunk_ids"]) != qa["gold_chunk_ids"]:
            raise ValueError(f"{rid} final manual gold_chunk_ids differs")
        if str(man["answer"]).replace("\r\n", "\n") != str(row["answer"]).replace(
                "\r\n", "\n"):
            raise ValueError(f"{rid} final manual answer differs")
        if json.loads(man["context_chunks"]) != row["context_chunks"]:
            raise ValueError(f"{rid} final manual context_chunks differs")
        if json.loads(man["context_texts"]) != row["context_texts"]:
            raise ValueError(f"{rid} final manual context_texts differs")
        if str(man["context_recorded"]).replace("\r\n", "\n") != str(
                row["context_recorded"]).replace("\r\n", "\n"):
            raise ValueError(f"{rid} final manual context_recorded differs")
        if str(man["answer_status"]) != str(row["rag_status"]):
            raise ValueError(f"{rid} final manual answer_status differs")
        if man["eval_fingerprint"] != T1_FINGERPRINT:
            raise ValueError(f"{rid} final manual fingerprint differs")
    # check manifest destination hashes match final files
    for filename, info in manifest["files"].items():
        target = RESULTS / filename
        if target.is_file() and info["destination_sha256"] != sha256(target):
            raise ValueError(f"manifest destination hash differs for {filename}")


def check_status() -> None:
    rows = read_rows(RESULTS / "t7_generation_status.csv")
    if len(rows) != 1:
        raise ValueError("status file does not have one row")
    row = rows[0]
    if row["eval_jsonl"] != "data/eval/qa_set.jsonl":
        raise ValueError("status eval_jsonl is not the official relative path")
    if row["manual_worksheet"] != "results/t7_manual_scoring.csv":
        raise ValueError("status manual_worksheet is not official relative path")
    for field, expected in (
            ("status", "heuristic_only_manual_pending"),
            ("matched_answers", "30"),
            ("available_answers", "30"),
            ("eval_fingerprint", T1_FINGERPRINT)):
        if row.get(field) != expected:
            raise ValueError(f"status {field} differs")


def check_manifest() -> dict:
    manifest = json.loads((RESULTS / "submission_manifest.json").read_text(
        encoding="utf-8"))
    if manifest.get("kind") != "submission_results_restoration":
        raise ValueError("manifest kind differs")
    if set(manifest.get("files", {})) != EXPECTED_FILES - {"submission_manifest.json"}:
        raise ValueError("manifest file set differs")
    if manifest["T1"]["fingerprint"] != T1_FINGERPRINT:
        raise ValueError("manifest T1 fingerprint differs")
    qa_path = manifest["qa_set"]["path"]
    if qa_path != "data/eval/qa_set.jsonl":
        raise ValueError("manifest qa_set path differs")
    if manifest["qa_set"]["sha256"] != sha256(ROOT / qa_path):
        raise ValueError("manifest qa_set sha differs")
    return manifest


def check_sources_unchanged() -> None:
    expected = {
        "t1_baseline.jsonl":
            "db39d10b60a00536ced2a6f44cb77a7d1856aa349d10bfc1f64e121a5e22b23d",
        "t7_manual_scoring_1.csv":
            "10cc21f57dd7c6db1d50a47dbf60f11d226af2e620c1595a032db3d65eeeac39",
        "t7_generation_status.csv":
            "87a94bda902e4d5fd65bec857be8e00a72fb67d0a8de69eb91331e775d16254d",
    }
    for name, digest in expected.items():
        if sha256(SOURCE_DIR / name) != digest:
            raise ValueError(f"source {name} changed")


def check_digests_and_status() -> None:
    allow = [
        "t1_baseline.jsonl", "t1_baseline_summary.csv", "t1_rag_vs_direct.csv",
        "t2_chunking.csv", "t2_chunking_cases.csv",
        "t2_retrieval_differences.csv", "t3_query_opt.csv",
        "t3_cases_baseline.csv", "t3_cases_multi.csv", "t3_cases_rewrite.csv",
        "t3_cases_subquery.csv", "t3_linear_alpha_sweep.csv",
        "t3_success_cases.csv", "t3_success_counts.csv", "t4_retriever.csv",
        "t4_per_query.csv", "t4_per_query.json", "t4_cases.csv",
        "t4_cases.json", "t4_cases.md", "t5_rerank.csv", "t5_compress.csv",
        "t6_routing.csv", "t6_support.csv", "t6_retry.csv",
        "t6_retry_trace.jsonl", "t6_retry_summary.json", "t7_metrics.csv",
        "t7_faithfulness.csv", "t7_llm_judge.csv",
        "t7_llm_judge_summary.json", "fig_t1_rag_vs_direct.png",
        "fig_t2_chunking.png", "fig_t3_alpha.png", "fig_t3_query_opt.png",
        "fig_t4_retriever.png", "fig_t5_rerank_proxy.png",
        "fig_t6_flow.png", "fig_t7_level.png",
    ]
    for name in allow:
        if sha256(RESULTS / name) != sha256(SOURCE_DIR / name):
            raise ValueError(f"destination {name} digest differs from source")
    if sha256(RESULTS / "t7_manual_scoring.csv") != sha256(
            SOURCE_DIR / "t7_manual_scoring_1.csv"):
        raise ValueError("manual destination digest differs from source")
    src = read_rows(SOURCE_DIR / "t7_generation_status.csv")[0]
    dst = read_rows(RESULTS / "t7_generation_status.csv")[0]
    if set(src) != set(dst):
        raise ValueError("status field columns differ")
    for key in src:
        if key in ("eval_jsonl", "manual_worksheet"):
            continue
        if src[key] != dst[key]:
            raise ValueError(f"status field {key} changed unexpectedly")
    for name in EXPECTED_FILES:
        if name.startswith("fig_") and name.endswith(".png"):
            if not (SOURCE_DIR / name).is_file() or (SOURCE_DIR / name).stat().st_size == 0:
                raise ValueError(f"source PNG missing or empty: {name}")
            if not (RESULTS / name).is_file() or (RESULTS / name).stat().st_size == 0:
                raise ValueError(f"final PNG missing or empty: {name}")


def check_manifest_matches_sources(manifest: dict) -> None:
    expected_files = EXPECTED_FILES - {"submission_manifest.json"}
    for filename, info in manifest["files"].items():
        source = ROOT / info["source_path"]
        if not source.is_file():
            raise ValueError(f"manifest source missing: {info['source_path']}")
        if sha256(source) != info["source_sha256"]:
            raise ValueError(f"manifest source hash differs for {filename}")
        target = RESULTS / filename
        if target.is_file() and sha256(target) != info["destination_sha256"]:
            raise ValueError(f"manifest destination hash differs for {filename}")
        if info["destination_path"] != "results/" + filename:
            raise ValueError(f"manifest destination_path differs for {filename}")
        if filename not in expected_files:
            raise ValueError(f"manifest has unexpected file {filename}")


def main() -> int:
    check_final_files()
    check_source_qa_annotation()
    manifest = check_manifest()
    check_t1_manual(manifest)
    check_status()
    check_sources_unchanged()
    check_digests_and_status()
    check_manifest_matches_sources(manifest)
    print("CHECK_RESTORATION OK files=42")
    return 0


if __name__ == "__main__":
    sys.exit(main())
