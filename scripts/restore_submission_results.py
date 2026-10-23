#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Restore the 42 official result files that were accidentally deleted.

The restoration source is the read-only experiment bundle
``repair_outputs/revised_full_20261022``. This module never removes,
moves, or overwrites existing project files. A dry run performs all
preflight checks and writes nothing.

Shortcut for the user: the default commands are

    python scripts/restore_submission_results.py --dry-run
    python scripts/restore_submission_results.py --apply
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE_DIR = ROOT / "repair_outputs" / "revised_full_20261022"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_submission_results.py"

MANUAL_SOURCE = "t7_manual_scoring_1.csv"
MANUAL_TARGET = "t7_manual_scoring.csv"
STATUS_TARGET = "t7_generation_status.csv"
MANIFEST_TARGET = "submission_manifest.json"

EXPECTED_QA_IDS = [f"q{i:02d}" for i in range(1, 31)]
EXPECTED_LEVELS = {"L1": 8, "L2": 8, "L3": 7, "L4": 7}
T1_FINGERPRINT = "9d8055917ee6f4b4a9fe523f09f0917e8cbdab1565ff08fd10926c143fabcf41"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_lines(path: Path) -> List[dict]:
    """Read a CSV object while preserving quoted newlines inside cells."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[dict]:
    result = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def allowlist_from_prepare(script: Path) -> List[str]:
    """Read the 39 source names from the official result plan without
    importing that module (it has latent CSV/manifest bugs)."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RESULT_FILES":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or not all(
                            isinstance(x, str) for x in value):
                        raise ValueError("RESULT_FILES is not a string list")
                    return list(value)
    raise ValueError("RESULT_FILES not found")


def exact_json(field):
    return json.loads(field)


def normalize_newlines(text) -> str:
    return str(text).replace("\r\n", "\n")


def validate_qa(qa_path: Path) -> List[dict]:
    rows = read_jsonl(qa_path)
    if [row["id"] for row in rows] != EXPECTED_QA_IDS:
        raise ValueError("Q/A id order differs from the fixed 30-question set")
    if len({row["id"] for row in rows}) != 30:
        raise ValueError("Q/A set has duplicate ids")
    counts = {level: sum(1 for row in rows if row["level"] == level)
              for level in EXPECTED_LEVELS}
    if counts != EXPECTED_LEVELS:
        raise ValueError(f"Unexpected level distribution: {counts}")
    for row in rows:
        for key in ("id", "level", "question", "gold_answer",
                    "gold_chunk_ids"):
            if key not in row or not str(row.get(key, "")).strip():
                raise ValueError(f"{row.get('id')} misses required field {key}")
        if not isinstance(row["gold_chunk_ids"], list) or not row["gold_chunk_ids"]:
            raise ValueError(f"{row['id']} gold_chunk_ids is not a non-empty list")
    return rows


def validate_annotation(annotation: Path, qa: List[dict]) -> None:
    rows = read_lines(annotation)
    if [row["id"] for row in rows] != EXPECTED_QA_IDS:
        raise ValueError("annotation id order differs from fixed set")
    if len({row["id"] for row in rows}) != 30:
        raise ValueError("annotation has duplicate ids")
    if not all(key in rows[0]
               for key in ("id", "level", "question", "gold_answer",
                           "gold_chunk_ids", "evidence_required")):
        raise ValueError("annotation misses required columns")
    qa_by_id = {row["id"]: row for row in qa}
    for row in rows:
        rid = row["id"]
        q = qa_by_id[rid]
        if row["level"] != q["level"]:
            raise ValueError(f"{rid} annotation level differs")
        if normalize_newlines(row["question"]) != normalize_newlines(q["question"]):
            raise ValueError(f"{rid} annotation question differs")
        if normalize_newlines(row["gold_answer"]) != normalize_newlines(
                q["gold_answer"]):
            raise ValueError(f"{rid} annotation gold_answer differs")
        if row["gold_chunk_ids"].split("|") != q["gold_chunk_ids"]:
            raise ValueError(f"{rid} annotation gold_chunk_ids differs")


def validate_t1_and_manual(qa: List[dict]) -> None:
    t1_path = SOURCE_DIR / "t1_baseline.jsonl"
    manual_path = SOURCE_DIR / MANUAL_SOURCE
    t1_rows = read_jsonl(t1_path)
    manual = read_lines(manual_path)
    if len(t1_rows) != 30 or len(manual) != 30:
        raise ValueError("T1 or manual source does not have 30 rows")
    if {row["id"] for row in t1_rows} != set(EXPECTED_QA_IDS):
        raise ValueError("T1 id set differs")
    if {row["id"] for row in manual} != set(EXPECTED_QA_IDS):
        raise ValueError("manual id set differs")
    if len({row["eval_fingerprint"] for row in t1_rows}) != 1:
        raise ValueError("T1 fingerprint is not single")
    if t1_rows[0]["eval_fingerprint"] != T1_FINGERPRINT:
        raise ValueError("T1 fingerprint does not match the revised set")
    t1_map = {row["id"]: row for row in t1_rows}
    man_map = {row["id"]: row for row in manual}
    for qa_row in qa:
        rid = qa_row["id"]
        t1 = t1_map[rid]
        man = man_map[rid]
        for key in ("question", "gold_answer", "level"):
            if normalize_newlines(t1[key]) != normalize_newlines(qa_row[key]):
                raise ValueError(f"{rid} T1 {key} differs from Q/A")
            if normalize_newlines(man[key]) != normalize_newlines(qa_row[key]):
                raise ValueError(f"{rid} manual {key} differs from Q/A")
        if t1["gold_chunk_ids"] != qa_row["gold_chunk_ids"]:
            raise ValueError(f"{rid} T1 gold_chunk_ids differs")
        if exact_json(man["gold_chunk_ids"]) != qa_row["gold_chunk_ids"]:
            raise ValueError(f"{rid} manual gold_chunk_ids differs")
        if normalize_newlines(man["answer"]) != normalize_newlines(t1["answer"]):
            raise ValueError(f"{rid} manual answer differs from T1")
        if exact_json(man["context_chunks"]) != t1["context_chunks"]:
            raise ValueError(f"{rid} manual context_chunks differs")
        if exact_json(man["context_texts"]) != t1["context_texts"]:
            raise ValueError(f"{rid} manual context_texts differs")
        if normalize_newlines(man["context_recorded"]) != normalize_newlines(
                t1["context_recorded"]):
            raise ValueError(f"{rid} manual context_recorded differs")
        if normalize_newlines(man["answer_status"]) != normalize_newlines(
                t1["rag_status"]):
            raise ValueError(f"{rid} manual answer_status differs")
        if man["eval_fingerprint"] != T1_FINGERPRINT:
            raise ValueError(f"{rid} manual fingerprint differs")


def _safe_allowlist() -> List[str]:
    allow = allowlist_from_prepare(PREPARE_SCRIPT)
    if len(allow) != 39:
        raise ValueError("allowlist does not contain 39 files")
    if len(set(allow)) != 39:
        raise ValueError("allowlist has duplicate file names")
    reserved = {MANUAL_TARGET, STATUS_TARGET, MANIFEST_TARGET}
    for name in allow:
        if name != Path(name).name:
            raise ValueError(f"allowlist item is not a bare file name: {name}")
        if name in reserved:
            raise ValueError(f"allowlist item collides with reserved name: {name}")
        source = SOURCE_DIR / name
        if not source.is_file():
            raise ValueError(f"allowlist source is missing: {name}")
        if source.is_symlink():
            raise ValueError(f"allowlist source must not be a symlink: {name}")
        resolved = source.resolve()
        if not resolved.is_relative_to(SOURCE_DIR.resolve()):
            raise ValueError(f"allowlist source escapes source root: {name}")
        target = RESULTS / name
        if not target.resolve().is_relative_to(RESULTS.resolve()):
            raise ValueError(f"allowlist target escapes results root: {name}")
    return allow


def _source_items() -> Dict[str, dict]:
    allow = _safe_allowlist()
    items = {
        name: {"source": SOURCE_DIR / name, "transform": None}
        for name in allow
    }
    items[MANUAL_TARGET] = {
        "source": SOURCE_DIR / MANUAL_SOURCE,
        "transform": None,
    }
    items[STATUS_TARGET] = {
        "source": SOURCE_DIR / STATUS_TARGET,
        "transform": "eval_jsonl/manual_worksheet paths",
    }
    return items


def _status_rows_from_source() -> List[dict]:
    rows = read_lines(SOURCE_DIR / STATUS_TARGET)
    if len(rows) != 1:
        raise ValueError("t7_generation_status source does not have one row")
    row = dict(rows[0])
    if row["eval_jsonl"] != str(ROOT / "repair_outputs" / "qa_set_revised.jsonl"):
        raise ValueError("Unexpected eval_jsonl in status source")
    if row["manual_worksheet"] != str(SOURCE_DIR / MANUAL_SOURCE):
        raise ValueError("Unexpected manual_worksheet in status source")
    expected_fields = {
        "status": "heuristic_only_manual_pending",
        "matched_answers": "30",
        "cached_generation_errors": "0",
        "missing_answers": "0",
        "available_answers": "30",
        "generation_errors": "0",
        "eval_fingerprint": T1_FINGERPRINT,
    }
    for field, expected in expected_fields.items():
        if row.get(field) != expected:
            raise ValueError(f"Unexpected status field {field}={row.get(field)}")
    if row["generated_questions"] not in ("0", "30"):
        raise ValueError("Unexpected generated_questions in status source")
    row["eval_jsonl"] = "data/eval/qa_set.jsonl"
    row["manual_worksheet"] = "results/t7_manual_scoring.csv"
    return [row]


def _csv_bytes(rows: List[dict]) -> bytes:
    if not rows:
        raise ValueError("Cannot write empty CSV")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def _planned_outputs() -> Dict[str, bytes]:
    items = _source_items()
    outputs: Dict[str, bytes] = {}
    for name, cfg in items.items():
        if name == STATUS_TARGET:
            outputs[name] = _csv_bytes(_status_rows_from_source())
        else:
            outputs[name] = cfg["source"].read_bytes()
    return outputs


def _row_count(name: str) -> int:
    path = SOURCE_DIR / name
    if path.suffix == ".jsonl":
        return len([l for l in path.read_text(encoding="utf-8-sig").splitlines()
                    if l.strip()])
    if path.suffix in (".csv", ".json"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return len(list(csv.DictReader(handle))) if path.suffix == ".csv" else 1
    return 0


def _batch_stats() -> dict:
    """Record free-form row counts without inventing timestamps/models."""
    keys = ["t1_baseline.jsonl", "t2_chunking.csv", "t3_query_opt.csv",
            "t4_retriever.csv", "t4_per_query.csv", "t5_rerank.csv",
            "t5_compress.csv", "t6_routing.csv", "t6_retry.csv",
            "t7_faithfulness.csv", "t7_llm_judge.csv"]
    return {name: _row_count(name) for name in keys}


def _manifest_bytes(outputs: Dict[str, bytes]) -> bytes:
    items = _source_items()
    files = {}
    for name in sorted(outputs):
        cfg = items[name]
        files[name] = {
            "source_path": cfg["source"].relative_to(ROOT).as_posix(),
            "destination_path": "results/" + name,
            "source_sha256": sha256(cfg["source"]),
            "destination_sha256": sha256_bytes(outputs[name]),
            "transform": cfg["transform"],
        }
    manifest = {
        "kind": "submission_results_restoration",
        "source_root": "repair_outputs/revised_full_20261022",
        "qa_set": {
            "path": "data/eval/qa_set.jsonl",
            "sha256": sha256(ROOT / "data/eval/qa_set.jsonl"),
        },
        "T1": {
            "fingerprint": T1_FINGERPRINT,
            "source_sha256": sha256(SOURCE_DIR / "t1_baseline.jsonl"),
        },
        "row_counts": _batch_stats(),
        "files": files,
        "notes": [
            "T3 was not rerun after the latest source changes; t3 files are historical.",
            "T7 LLM judge was not rerun; existing t7_llm_judge* are old scores and are not manual scores.",
            "t7_generation_status is heuristic_only_manual_pending.",
            "Old T7 judge rows q02/q13/q23 have full faithfulness with positive unsupported claims; no claim of final acceptance.",
        ],
    }
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n").encode("utf-8")


def planned() -> Dict[str, bytes]:
    qa = validate_qa(ROOT / "data/eval/qa_set.jsonl")
    validate_annotation(ROOT / "scripts" / "qa_annotation.csv", qa)
    validate_t1_and_manual(qa)
    outputs = _planned_outputs()
    outputs[MANIFEST_TARGET] = _manifest_bytes(outputs)
    return outputs


def _outcome() -> (Dict[str, bytes], List[str], List[str]):
    planned_outputs = planned()
    missing = [name for name in planned_outputs
               if not (RESULTS / name).is_file()]
    conflicts = []
    for name, content in planned_outputs.items():
        target = RESULTS / name
        if not target.exists():
            continue
        if target.read_bytes() != content:
            conflicts.append(name)
    return planned_outputs, missing, conflicts


def dry_run() -> int:
    _, missing, conflicts = _outcome()
    print(f"DRYRUN missing={len(missing)} conflicts={len(conflicts)}")
    for name in missing:
        print("missing " + name)
    for name in conflicts:
        print("conflict " + name)
    return 0 if not conflicts else 2


def apply() -> int:
    planned_outputs, missing, conflicts = _outcome()
    if conflicts:
        print(f"CONFLICT {len(conflicts)}", file=sys.stderr)
        for name in conflicts:
            print("conflict " + name, file=sys.stderr)
        return 2
    source_snapshot = {name: sha256((SOURCE_DIR / name))
                       for name in _safe_allowlist() + [MANUAL_SOURCE,
                                                        STATUS_TARGET]}
    RESULTS.mkdir(exist_ok=True)
    written = 0
    for name in missing:
        target = RESULTS / name
        # Exclusive-create is deliberate: identical files are skipped in
        # _outcome() and unexpected races must fail instead of overwrite.
        with open(target, "xb") as handle:
            handle.write(planned_outputs[name])
        if sha256(target) != hashlib.sha256(planned_outputs[name]).hexdigest():
            raise RuntimeError(f"Written file failed hash check: {name}")
        if source_snapshot != {name: sha256((SOURCE_DIR / name))
                               for name in _safe_allowlist() + [MANUAL_SOURCE,
                                                                STATUS_TARGET]}:
            raise RuntimeError("Source files changed during apply")
        written += 1
    final_missing = [name for name in planned_outputs
                     if not (RESULTS / name).is_file()]
    final_conflicts = [name for name, content in planned_outputs.items()
                       if (RESULTS / name).is_file()
                       and (RESULTS / name).read_bytes() != content]
    if final_missing or final_conflicts:
        raise RuntimeError(
            f"Apply left missing={final_missing} conflicts={final_conflicts}")
    print(f"RESTORED {written} files; still_missing=0; conflicts=0")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Restore official results")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true",
                       help="Only run preflight, write nothing")
    group.add_argument("--apply", action="store_true",
                       help="Write only missing non-conflicting files")
    args = parser.parse_args(argv)
    if args.apply:
        return apply()
    return dry_run()


if __name__ == "__main__":
    sys.exit(main())
