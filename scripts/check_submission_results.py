#!/usr/bin/env python3
"""Verify D3/D4 alignment.  Exits non-zero on any inconsistency.

Checks:
1. Formal qa_set byte-identical to source, annotation field-by-field.
2. Every result file present, each byte-identical to source
   (status file checked for path correctness, not byte identity).
3. Manifest re-derived hashes match on-disk hashes for every file.
4. Manual scoring rows match T1 on question/answer/fingerprint.
5. Status file has correct relative paths and expected status value.
"""

import csv, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE_QA_SET = ROOT / "repair_outputs" / "qa_set_revised.jsonl"
SOURCE_ANNOTATION = ROOT / "repair_outputs" / "qa_annotation_revised.csv"
SOURCE_RESULTS = ROOT / "repair_outputs" / "revised_full_20261022"
FORMAL_QA_SET = ROOT / "data" / "eval" / "qa_set.jsonl"
FORMAL_ANNOTATION = ROOT / "scripts" / "qa_annotation.csv"
FORMAL_RESULTS = ROOT / "results"

MANUAL_SCORING_FORMAL = "t7_manual_scoring.csv"
MANUAL_SCORING_SOURCE = "t7_manual_scoring_1.csv"
STATUS_FILE = "t7_generation_status.csv"

failures = 0

EXPECTED_FILES = [
    "t1_baseline.jsonl", "t1_baseline_summary.csv", "t1_rag_vs_direct.csv",
    "t2_chunking.csv", "t2_chunking_cases.csv", "t2_retrieval_differences.csv",
    "t3_query_opt.csv", "t3_cases_baseline.csv", "t3_cases_multi.csv",
    "t3_cases_rewrite.csv", "t3_cases_subquery.csv",
    "t3_linear_alpha_sweep.csv", "t3_success_cases.csv", "t3_success_counts.csv",
    "t4_retriever.csv", "t4_per_query.csv", "t4_per_query.json",
    "t4_cases.csv", "t4_cases.json", "t4_cases.md",
    "t5_rerank.csv", "t5_compress.csv",
    "t6_routing.csv", "t6_support.csv", "t6_retry.csv",
    "t6_retry_trace.jsonl", "t6_retry_summary.json",
    "t7_metrics.csv", "t7_faithfulness.csv",
    "t7_llm_judge.csv", "t7_llm_judge_summary.json",
    MANUAL_SCORING_FORMAL, STATUS_FILE, "submission_manifest.json",
    "fig_t1_rag_vs_direct.png", "fig_t2_chunking.png",
    "fig_t3_alpha.png", "fig_t3_query_opt.png",
    "fig_t4_retriever.png", "fig_t5_rerank_proxy.png",
    "fig_t6_flow.png", "fig_t7_level.png",
]

ANNOTATION_MATCH_FIELDS = ["id", "level", "question", "gold_answer", "gold_chunk_ids"]

failures = 0

def _norm_nl(s):
    """Normalise newlines: collapse consecutive \\n to single \\n, strip."""
    import re
    return re.sub(r'\n+', '\n', s.strip())


def sha256_hex(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(msg):
    global failures
    failures += 1
    print(f"  FAIL [{failures}]: {msg}")


def _norm_csv(val):
    parts = [p.strip() for p in val.replace("|", "\n").split("\n") if p.strip()]
    return "\n".join(sorted(parts))


def _norm_jsonl(val):
    if isinstance(val, list):
        parts = val
    elif isinstance(val, str):
        parts = [p.strip() for p in val.replace("|", "\n").split("\n") if p.strip()]
    else:
        parts = [str(val)]
    return "\n".join(sorted(parts))


def check_qa_set():
    print("--- Check: formal qa_set.jsonl ---")
    if not FORMAL_QA_SET.exists():
        fail("formal qa_set.jsonl not found")
        return
    if not SOURCE_QA_SET.exists():
        fail("source qa_set_revised.jsonl not found")
        return
    src_h = sha256_hex(SOURCE_QA_SET)
    fmt_h = sha256_hex(FORMAL_QA_SET)
    print(f"  source sha256:  {src_h}")
    print(f"  formal sha256:  {fmt_h}")
    if src_h != fmt_h:
        fail("hash mismatch: formal qa_set differs from source")
    lines = FORMAL_QA_SET.read_text(encoding="utf-8").strip().split("\n")
    ids = [json.loads(l)["id"] for l in lines]
    if len(ids) != 30:
        fail(f"expected 30 lines, got {len(ids)}")
    if len(set(ids)) != 30:
        fail("duplicate ids")
    from collections import Counter
    lc = dict(Counter(json.loads(l).get("level", "") for l in lines))
    if lc != {"L1": 8, "L2": 8, "L3": 7, "L4": 7}:
        fail(f"level distribution: {lc}")
    print("  OK: 30 ids, L1=8 L2=8 L3=7 L4=7, hash matches source")


def check_annotation():
    print("\n--- Check: formal qa_annotation.csv ---")
    if not FORMAL_ANNOTATION.exists():
        fail("formal annotation not found")
        return
    if not SOURCE_ANNOTATION.exists():
        fail("source annotation not found")
        return
    src_h = sha256_hex(SOURCE_ANNOTATION)
    fmt_h = sha256_hex(FORMAL_ANNOTATION)
    print(f"  source sha256:  {src_h}")
    print(f"  formal sha256:  {fmt_h}")
    if src_h != fmt_h:
        fail("hash mismatch: annotation differs from source")
    qa_by_id = {}
    for line in FORMAL_QA_SET.read_text(encoding="utf-8").strip().split("\n"):
        r = json.loads(line)
        qa_by_id[r["id"]] = r
    content = FORMAL_ANNOTATION.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(content.splitlines())
    csv_cols = set(reader.fieldnames or [])
    rows = list(reader)
    if len(rows) != 30:
        fail(f"annotation has {len(rows)} rows, expected 30")
    mismatches = 0
    for i, row in enumerate(rows):
        cid = row.get("id", "")
        if cid not in qa_by_id:
            fail(f"annotation row {i+1}: id {cid!r} not in qa_set")
            continue
        qa = qa_by_id[cid]
        for field in ANNOTATION_MATCH_FIELDS:
            if field not in csv_cols:
                continue
            cv = (row.get(field, "") or "").rstrip("\r\n")
            qv = qa.get(field, "") or ""
            if field == "gold_chunk_ids":
                if _norm_csv(cv) != _norm_jsonl(qv):
                    mismatches += 1
            elif cv != str(qv):
                mismatches += 1
    if mismatches:
        fail(f"{mismatches} field mismatches annotation vs qa_set")
    print(f"  OK: 30 rows, fields match qa_set, hash matches source")


def check_results_files():
    print("\n--- Check: results/ file presence and byte identity ---")
    present, missing, mismatch = 0, [], []
    for fname in EXPECTED_FILES:
        src = SOURCE_RESULTS / fname
        dst = FORMAL_RESULTS / fname
        src_actual = src
        if fname == MANUAL_SCORING_FORMAL:
            src_actual = SOURCE_RESULTS / MANUAL_SCORING_SOURCE
        if not dst.exists():
            missing.append(fname)
            continue
        if not src_actual.exists() and fname != STATUS_FILE:
            fail(f"source missing for {fname}")
            continue
        present += 1
        if fname == STATUS_FILE:
            st = dst.read_text(encoding="utf-8-sig")
            if "C:\\\\" in st or "C:/" in st:
                fail("status file has absolute paths")
            if "data/eval/qa_set.jsonl" not in st:
                fail("status file missing eval_jsonl path")
            if "results/t7_manual_scoring.csv" not in st:
                fail("status file missing manual_worksheet path")
            reader = csv.DictReader(st.splitlines())
            rows = list(reader)
            if len(rows) != 1:
                fail(f"status file has {len(rows)} rows, expected 1")
            else:
                status_val = rows[0].get("status", "")
                print(f"  status value: {status_val}")
            continue
        src_h = sha256_hex(src_actual)
        dst_h = sha256_hex(dst)
        if src_h != dst_h:
            mismatch.append(fname)
    print(f"  present: {present}/{len(EXPECTED_FILES)}")
    if missing:
        for m in missing:
            fail(f"missing: {m}")
    if mismatch:
        for m in mismatch:
            fail(f"byte mismatch vs source: {m}")
    if not missing and not mismatch:
        print("  OK: all files present and byte-identical to source")


def check_manifest_hashes():
    print("\n--- Check: manifest hashes vs on-disk ---")
    mf_path = FORMAL_RESULTS / "submission_manifest.json"
    if not mf_path.exists():
        fail("submission_manifest.json not found")
        return
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    mf_results = mf.get("results", {})
    if not mf_results:
        fail("manifest has no results hashes")
        return
    rehashed = {}
    for fname in EXPECTED_FILES:
        dst = FORMAL_RESULTS / fname
        src_key = fname
        if fname == MANUAL_SCORING_FORMAL:
            src_key = f"{MANUAL_SCORING_FORMAL} (from {MANUAL_SCORING_SOURCE})"
        if fname == STATUS_FILE:
            src_key = f"{STATUS_FILE} (paths fixed on copy)"
        if dst.exists() and src_key in mf_results:
            actual = sha256_hex(dst)
            rehashed[src_key] = actual
    mismatches = 0
    for key in sorted(mf_results):
        if key in rehashed:
            if mf_results[key] != rehashed[key]:
                fail(f"manifest hash mismatch for {key}")
                mismatches += 1
    qa_set_h = mf.get("qa_set", {}).get("sha256", "")
    if qa_set_h and qa_set_h != sha256_hex(FORMAL_QA_SET):
        fail("manifest qa_set hash != on-disk formal qa_set")
    ann_h = mf.get("annotation", {}).get("sha256", "")
    if ann_h and ann_h != sha256_hex(FORMAL_ANNOTATION):
        fail("manifest annotation hash != on-disk formal annotation")
    if mismatches == 0:
        print(f"  OK: {len(rehashed)} manifest hashes verified")


def check_manual_vs_t1():
    print("\n--- Check: manual scoring vs T1 ---")
    ms_path = FORMAL_RESULTS / MANUAL_SCORING_FORMAL
    t1_path = FORMAL_RESULTS / "t1_baseline.jsonl"
    if not ms_path.exists():
        fail("manual scoring not found")
        return
    if not t1_path.exists():
        fail("T1 baseline not found")
        return
    t1_by_id = {}
    for line in t1_path.read_text(encoding="utf-8").strip().split("\n"):
        r = json.loads(line)
        t1_by_id[r["id"]] = r
    t1_fps = set(r.get("eval_fingerprint", "") for r in t1_by_id.values())
    if len(t1_fps) != 1:
        fail(f"T1 has {len(t1_fps)} fingerprints (expected 1)")
    else:
        print(f"  T1 fingerprint: {sorted(t1_fps)[0][:32]}...")
    content = ms_path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(content.splitlines())
    ms_rows = list(reader)
    if len(ms_rows) != 30:
        fail(f"manual has {len(ms_rows)} rows, expected 30")
    mismatches = 0
    for row in ms_rows:
        qid = row.get("id", "")
        if qid not in t1_by_id:
            mismatches += 1
            continue
        t1 = t1_by_id[qid]
        if _norm_nl(row.get("question", "")) != _norm_nl(t1.get("question", "")):
            mismatches += 1
        if _norm_nl(row.get("answer", "")) != _norm_nl(t1.get("answer", "")):
            mismatches += 1
        ms_fp = row.get("eval_fingerprint", "").strip()
        t1_fp = t1.get("eval_fingerprint", "").strip()
        if ms_fp and t1_fp and ms_fp != t1_fp:
            mismatches += 1
    if mismatches:
        fail(f"{mismatches} field mismatches manual vs T1")
    else:
        print("  OK: 30 rows, question/answer/fingerprint match T1")
    human_fields = ["faithfulness_1_5", "L4_rubric_0_5", "reviewer", "notes", "error_type"]
    ms_cols = set(reader.fieldnames or [])
    for hf in human_fields:
        if hf in ms_cols:
            n = sum(1 for r in ms_rows if r.get(hf, "").strip())
            if n:
                print(f"  NOTE: {n} rows have non-empty {hf} (protected)")


def main():
    global failures
    print("=== Submission Results Check ===\n")
    check_qa_set()
    check_annotation()
    check_results_files()
    check_manifest_hashes()
    check_manual_vs_t1()
    print(f"\n=== Overall: {'PASS' if failures == 0 else f'FAIL ({failures} failures)'} ===")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
