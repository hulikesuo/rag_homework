#!/usr/bin/env python3
"""D3/D4 alignment: validate and sync qa_set + annotation, copy verified
results to results/, fix t7_generation_status paths, generate manifest.

Dry-run by default (no filesystem writes).  Pass --apply to write.
Pre-checks run before any writes; conflicts or missing source files abort
early without partial writes.  Re-running --apply is idempotent (skips
already-synced files)."""

import argparse, csv, hashlib, json, os, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE_QA_SET = ROOT / "repair_outputs" / "qa_set_revised.jsonl"
SOURCE_ANNOTATION = ROOT / "repair_outputs" / "qa_annotation_revised.csv"
FORMAL_QA_SET = ROOT / "data" / "eval" / "qa_set.jsonl"
FORMAL_ANNOTATION = ROOT / "scripts" / "qa_annotation.csv"

SOURCE_RESULTS = ROOT / "repair_outputs" / "revised_full_20261022"
FORMAL_RESULTS = ROOT / "results"
SOURCE_T1 = SOURCE_RESULTS / "t1_baseline.jsonl"
SOURCE_LLM_SUMMARY = SOURCE_RESULTS / "t7_llm_judge_summary.json"

RESULT_FILES = [
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
    "fig_t1_rag_vs_direct.png", "fig_t2_chunking.png",
    "fig_t3_alpha.png", "fig_t3_query_opt.png",
    "fig_t4_retriever.png", "fig_t5_rerank_proxy.png",
    "fig_t6_flow.png", "fig_t7_level.png",
]

ANNOTATION_MATCH_FIELDS = ["id", "level", "question", "gold_answer", "gold_chunk_ids"]
MANUAL_SCORING_HUMAN_FIELDS = ["faithfulness_1_5", "L4_rubric_0_5", "reviewer", "notes", "error_type"]
MANUAL_SCORING_SOURCE = "t7_manual_scoring_1.csv"
MANUAL_SCORING_FORMAL = "t7_manual_scoring.csv"
STATUS_FILE = "t7_generation_status.csv"

def _norm_nl(s):
    """Normalise newlines: collapse consecutive \\n to single \\n, strip."""
    import re
    return re.sub(r'\n+', '\n', s.strip())


def sha256_hex(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_qa_set(path):
    result = {"ok": True, "errors": []}
    if not path.exists():
        return {"ok": False, "errors": [f"file not found: {path}"]}
    try:
        lines = path.read_text(encoding="utf-8").strip().split("\n")
    except Exception as exc:
        return {"ok": False, "errors": [f"read error: {exc}"]}
    if len(lines) != 30:
        result["errors"].append(f"expected 30 lines, got {len(lines)}")
        result["ok"] = False
    ids, levels, records = [], [], []
    required = {"id", "level", "question", "gold_answer", "gold_chunk_ids"}
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            result["errors"].append(f"line {i+1}: invalid json: {exc}")
            result["ok"] = False
            continue
        missing = required - set(rec.keys())
        if missing:
            result["errors"].append(
                f"line {i+1} ({rec.get('id','?')}): missing {missing}")
            result["ok"] = False
        ids.append(rec.get("id", ""))
        levels.append(rec.get("level", ""))
        records.append(rec)
    if len(set(ids)) != len(ids):
        result["errors"].append("duplicate ids")
        result["ok"] = False
    lc = dict(Counter(levels))
    if lc != {"L1": 8, "L2": 8, "L3": 7, "L4": 7}:
        result["errors"].append(f"level mismatch: {lc}")
        result["ok"] = False
    result["count"] = len(lines)
    result["levels"] = lc
    result["ids"] = ids
    result["records"] = records
    result["hash"] = sha256_hex(path)
    return result


def _norm_chunk_ids_csv(val):
    parts = [p.strip() for p in val.replace("|", "\n").split("\n") if p.strip()]
    return "\n".join(sorted(parts))


def _norm_chunk_ids_jsonl(val):
    if isinstance(val, list):
        parts = val
    elif isinstance(val, str):
        parts = [p.strip() for p in val.replace("|", "\n").split("\n") if p.strip()]
    else:
        parts = [str(val)]
    return "\n".join(sorted(parts))


def validate_annotation(path, qa_records):
    """Validate annotation CSV against qa_set: row count, id order, and
    field-by-field match on id/level/question/gold_answer/gold_chunk_ids."""
    result = {"ok": True, "errors": []}
    if not path.exists():
        return {"ok": False, "errors": [f"not found: {path}"]}
    try:
        content = path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        return {"ok": False, "errors": [f"read error: {exc}"]}
    reader = csv.DictReader(content.splitlines())
    csv_cols = set(reader.fieldnames or [])
    rows = list(reader)
    qa_by_id = {r["id"]: r for r in qa_records}
    if len(rows) != len(qa_records):
        result["errors"].append(
            f"rows {len(rows)} != qa_set {len(qa_records)}")
        result["ok"] = False
    mismatches = []
    for i, row in enumerate(rows):
        cid = row.get("id", "")
        if cid not in qa_by_id:
            result["errors"].append(f"row {i+1}: id {cid!r} not in qa_set")
            result["ok"] = False
            continue
        qa = qa_by_id[cid]
        for field in ANNOTATION_MATCH_FIELDS:
            if field not in csv_cols:
                continue
            cv = (row.get(field, "") or "").rstrip("\r\n")
            qv = qa.get(field, "") or ""
            if field == "gold_chunk_ids":
                if _norm_chunk_ids_csv(cv) != _norm_chunk_ids_jsonl(qv):
                    mismatches.append(f"row {i+1} {cid}: {field}")
            elif cv != str(qv):
                mismatches.append(f"row {i+1} {cid}: {field}")
    if mismatches:
        result["errors"].append(f"{len(mismatches)} field mismatches")
        for m in mismatches[:3]:
            result["errors"].append(f"  {m}")
        result["ok"] = False
    result["rows"] = len(rows)
    result["hash"] = sha256_hex(path)
    return result


def verify_manual_vs_t1(ms_path):
    """Verify manual scoring worksheet matches T1 on question/answer/
    eval_fingerprint for all 30 rows. Also detect non-empty human scores."""
    result = {"ok": True, "errors": [], "has_human_scores": False}
    if not SOURCE_T1.exists():
        return {"ok": False, "errors": ["T1 baseline not found"]}
    if not ms_path.exists():
        return {"ok": False, "errors": [f"manual not found: {ms_path}"]}
    t1_by_id = {}
    for line in SOURCE_T1.read_text(encoding="utf-8").strip().split("\n"):
        r = json.loads(line)
        t1_by_id[r["id"]] = r
    content = ms_path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(content.splitlines())
    ms_rows = list(reader)
    ms_cols = set(reader.fieldnames or [])
    t1_fps = set(r.get("eval_fingerprint", "") for r in t1_by_id.values())
    result["t1_fingerprints"] = sorted(t1_fps)
    if len(t1_fps) != 1:
        result["errors"].append(
            f"T1 has {len(t1_fps)} distinct fingerprints (expected 1)")
        result["ok"] = False
    if len(ms_rows) != 30:
        result["errors"].append(
            f"manual has {len(ms_rows)} rows, expected 30")
        result["ok"] = False
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
        result["errors"].append(
            f"{mismatches} field mismatches between manual and T1")
        result["ok"] = False
    result["mismatches"] = mismatches
    human = {}
    for hf in MANUAL_SCORING_HUMAN_FIELDS:
        if hf in ms_cols:
            n = sum(1 for r in ms_rows if r.get(hf, "").strip())
            if n:
                human[hf] = n
    result["human_score_fields"] = human
    result["has_human_scores"] = len(human) > 0
    return result


def sync_qa_set(apply):
    report = {"step": "D3: sync qa_set + annotation"}
    src_val = validate_qa_set(SOURCE_QA_SET)
    report["source_qa"] = {k: v for k, v in src_val.items()
                           if k != "records"}
    if not src_val["ok"]:
        report["status"] = "FAIL"
        return report
    ann_val = validate_annotation(SOURCE_ANNOTATION, src_val["records"])
    report["source_annotation"] = ann_val
    if not ann_val["ok"]:
        report["status"] = "FAIL"
        return report
    fqe = FORMAL_QA_SET.exists()
    report["formal_qa_exists"] = fqe
    if fqe:
        report["qa_sync_needed"] = (
            sha256_hex(FORMAL_QA_SET) != src_val["hash"])
    else:
        report["qa_sync_needed"] = True
    fae = FORMAL_ANNOTATION.exists()
    report["formal_annotation_exists"] = fae
    if fae:
        report["ann_sync_needed"] = (
            sha256_hex(FORMAL_ANNOTATION) != ann_val["hash"])
    else:
        report["ann_sync_needed"] = True
    if apply:
        written = []
        if report["qa_sync_needed"]:
            FORMAL_QA_SET.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE_QA_SET, FORMAL_QA_SET)
            written.append(str(FORMAL_QA_SET.relative_to(ROOT)))
        if report["ann_sync_needed"]:
            FORMAL_ANNOTATION.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE_ANNOTATION, FORMAL_ANNOTATION)
            written.append(str(FORMAL_ANNOTATION.relative_to(ROOT)))
        report["written"] = written
        report["formal_qa_after"] = sha256_hex(FORMAL_QA_SET)
        report["formal_annotation_after"] = sha256_hex(FORMAL_ANNOTATION)
    report["status"] = "OK"
    return report


def _derive_batch_metadata():
    batches = []
    if SOURCE_T1.exists():
        t1 = [json.loads(l)
              for l in SOURCE_T1.read_text(encoding="utf-8").strip().split("\n")]
        rag = sum(1 for r in t1 if r.get("rag_answered"))
        direct = sum(1 for r in t1 if r.get("direct_answer", "").strip())
        batches.append(
            f"T1: {len(t1)} questions, RAG answered={rag}, direct={direct}")
    else:
        batches.append("T1: results not found")
    fp = SOURCE_RESULTS / "t2_chunking_cases.csv"
    if fp.exists():
        rows = list(csv.DictReader(
            fp.read_text(encoding="utf-8-sig").splitlines()))
        ans = sum(1 for r in rows if r.get("answer", "").strip())
        batches.append(
            f"T2: {len(rows)} case rows, generated answers={ans}")
    else:
        batches.append("T2: results not found")
    t3f = ["t3_cases_baseline.csv", "t3_cases_rewrite.csv",
           "t3_cases_multi.csv", "t3_cases_subquery.csv"]
    t3t, t3a = 0, 0
    for fn in t3f:
        fp = SOURCE_RESULTS / fn
        if fp.exists():
            rows = list(csv.DictReader(
                fp.read_text(encoding="utf-8-sig").splitlines()))
            t3t += len(rows)
            t3a += sum(1 for r in rows if r.get("answer", "").strip())
    batches.append(
        f"T3: {t3t} rows across 4 strategies, generated answers={t3a}")
    for fn, label in [
        ("t4_retriever.csv", "T4 retrieval"),
        ("t5_rerank.csv", "T5 rerank"),
        ("t5_compress.csv", "T5 compress"),
        ("t6_routing.csv", "T6 routing"),
        ("t6_retry.csv", "T6 retry"),
        ("t7_metrics.csv", "T7 metrics"),
        ("t7_llm_judge.csv", "T7 LLM judge"),
    ]:
        fp = SOURCE_RESULTS / fn
        if fp.exists():
            rows = list(csv.DictReader(
                fp.read_text(encoding="utf-8-sig").splitlines()))
            batches.append(f"{label}: {len(rows)} rows")
        else:
            batches.append(f"{label}: results not found")
    return batches


def _precheck_results():
    result = {"ok": True, "missing": [], "conflicts": []}
    for fname in RESULT_FILES:
        src = SOURCE_RESULTS / fname
        if not src.exists():
            result["missing"].append(fname)
            result["ok"] = False
            continue
        dst = FORMAL_RESULTS / fname
        if dst.exists() and sha256_hex(src) != sha256_hex(dst):
            result["conflicts"].append({"file": fname})
    ms_src = SOURCE_RESULTS / MANUAL_SCORING_SOURCE
    if not ms_src.exists():
        result["missing"].append(MANUAL_SCORING_SOURCE)
        result["ok"] = False
    else:
        ms_dst = FORMAL_RESULTS / MANUAL_SCORING_FORMAL
        if ms_dst.exists() and sha256_hex(ms_src) != sha256_hex(ms_dst):
            chk = verify_manual_vs_t1(ms_dst)
            note = ("keeping existing (has human scores)"
                    if chk.get("has_human_scores") else None)
            result["conflicts"].append({
                "file": f"{MANUAL_SCORING_SOURCE} -> {MANUAL_SCORING_FORMAL}",
                "note": note,
            })
    st_src = SOURCE_RESULTS / STATUS_FILE
    if not st_src.exists():
        result["missing"].append(STATUS_FILE)
        result["ok"] = False
    return result


def copy_results(apply):
    report = {"step": "D4: copy results"}
    pre = _precheck_results()
    if pre["missing"]:
        report["missing"] = pre["missing"]
    if pre["conflicts"]:
        report["conflicts"] = pre["conflicts"]
    if pre["missing"] or pre["conflicts"]:
        report["status"] = "BLOCKED"
        report["note"] = (
            "pre-check failed; no files written. "
            "Resolve conflicts/missing then retry.")
        return report
    copied, skipped, hash_pairs = [], [], []
    for fname in RESULT_FILES:
        src = SOURCE_RESULTS / fname
        dst = FORMAL_RESULTS / fname
        if dst.exists():
            skipped.append(fname)
        else:
            if apply:
                FORMAL_RESULTS.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                hash_pairs.append({
                    "file": fname,
                    "src_hash": sha256_hex(src),
                    "dst_hash": sha256_hex(dst),
                })
            copied.append(fname)
    ms_src = SOURCE_RESULTS / MANUAL_SCORING_SOURCE
    ms_dst = FORMAL_RESULTS / MANUAL_SCORING_FORMAL
    if ms_dst.exists():
        skipped.append(MANUAL_SCORING_FORMAL)
    else:
        if apply:
            shutil.copy2(ms_src, ms_dst)
            hash_pairs.append({
                "file": f"{MANUAL_SCORING_SOURCE} -> {MANUAL_SCORING_FORMAL}",
                "src_hash": sha256_hex(ms_src),
                "dst_hash": sha256_hex(ms_dst),
            })
        copied.append(
            f"{MANUAL_SCORING_SOURCE} -> {MANUAL_SCORING_FORMAL}")
    st_src = SOURCE_RESULTS / STATUS_FILE
    st_dst = FORMAL_RESULTS / STATUS_FILE
    if st_dst.exists():
        skipped.append(f"{STATUS_FILE} (paths fixed)")
    else:
        if apply:
            content = st_src.read_text(encoding="utf-8-sig")
            content = content.replace(
                str(SOURCE_QA_SET.resolve()), "data/eval/qa_set.jsonl")
            content = content.replace(
                str((SOURCE_RESULTS / MANUAL_SCORING_SOURCE).resolve()),
                "results/t7_manual_scoring.csv")
            content = content.replace("\\", "/")
            st_dst.write_text(content, encoding="utf-8-sig")
            hash_pairs.append({
                "file": f"{STATUS_FILE} (paths fixed)",
                "src_hash": sha256_hex(st_src),
                "dst_hash": sha256_hex(st_dst),
            })
        copied.append(f"{STATUS_FILE} (paths fixed)")
    report["copied"] = copied
    report["skipped"] = skipped
    report["hash_pairs"] = hash_pairs
    report["status"] = "OK"
    return report


def generate_manifest():
    from datetime import datetime, timezone
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_set": {
            "path": str(FORMAL_QA_SET.relative_to(ROOT)),
            "sha256": sha256_hex(SOURCE_QA_SET),
            "count": 30,
            "levels": {"L1": 8, "L2": 8, "L3": 7, "L4": 7},
        },
        "annotation": {
            "path": str(FORMAL_ANNOTATION.relative_to(ROOT)),
            "sha256": sha256_hex(SOURCE_ANNOTATION),
        },
        "t1_eval_fingerprint": None,
        "results": {},
    }
    if SOURCE_T1.exists():
        t1 = [json.loads(l)
              for l in SOURCE_T1.read_text(encoding="utf-8").strip().split("\n")]
        manifest["t1_eval_fingerprint"] = t1[0].get("eval_fingerprint", "")
    if SOURCE_LLM_SUMMARY.exists():
        s = json.loads(SOURCE_LLM_SUMMARY.read_text(encoding="utf-8"))
        manifest["llm_judge"] = {
            "model": s.get("judge_model", "unknown"),
            "metric_type": s.get("metric_type", "unknown"),
            "note": (
                "LLM judge scores are NOT human scores; "
                "self-evaluation bias (same model family as generator)."),
        }
    else:
        manifest["llm_judge"] = {
            "model": "unknown",
            "note": "LLM judge summary not found.",
        }
    manifest["batches"] = _derive_batch_metadata()
    manifest["t3_pending_rerun"] = (
        "experiments/t3_query_opt.py was modified after T3 results "
        "were generated. T3 results in results/ reflect the "
        "pre-modification run. "
        "Scope: only T3 retrieval+generation needs rerun; "
        "T1/T2/T4/T5/T6/T7 are unaffected.")
    ms_path = SOURCE_RESULTS / MANUAL_SCORING_SOURCE
    if ms_path.exists():
        mv = verify_manual_vs_t1(ms_path)
        manifest["manual_scoring_vs_t1"] = {
            "source": MANUAL_SCORING_SOURCE,
            "mismatches": mv.get("mismatches", -1),
            "has_human_scores": mv.get("has_human_scores", False),
            "human_score_fields": mv.get("human_score_fields", {}),
        }
    for fname in sorted(RESULT_FILES):
        fp = SOURCE_RESULTS / fname
        if fp.exists():
            manifest["results"][fname] = sha256_hex(fp)
    ms_fp = SOURCE_RESULTS / MANUAL_SCORING_SOURCE
    if ms_fp.exists():
        manifest["results"][
            f"{MANUAL_SCORING_FORMAL} (from {MANUAL_SCORING_SOURCE})"
        ] = sha256_hex(ms_fp)
    st_fp = SOURCE_RESULTS / STATUS_FILE
    if st_fp.exists():
        manifest["results"][
            f"{STATUS_FILE} (paths fixed on copy)"
        ] = sha256_hex(st_fp)
    return manifest


def main():
    p = argparse.ArgumentParser(description="D3/D4 alignment for submission")
    p.add_argument("--apply", action="store_true",
                   help="Write files (default: dry-run)")
    p.add_argument("--manifest", type=str, default=None,
                   help="Manifest output path")
    args = p.parse_args()
    if not args.apply:
        print("=== DRY RUN (use --apply to write) ===\n")
    print("--- Pre-check: manual scoring vs T1 ---")
    ms_path = SOURCE_RESULTS / MANUAL_SCORING_SOURCE
    if ms_path.exists():
        mv = verify_manual_vs_t1(ms_path)
        fps = mv.get("t1_fingerprints", [])
        print(f"  T1 distinct fingerprints: {len(fps)} (expect 1)")
        print(f"  manual rows: 30, mismatches vs T1: {mv.get('mismatches','?')}")
        print(f"  has_human_scores: {mv.get('has_human_scores')}, "
              f"fields: {mv.get('human_score_fields')}")
        if not mv["ok"]:
            print("  ERRORS:")
            for e in mv["errors"]:
                print(f"    {e}")
            print("\n=== BLOCKED: manual-vs-T1 verification failed ===")
            sys.exit(1)
    else:
        print(f"  WARNING: {MANUAL_SCORING_SOURCE} not found")
    print()
    print("--- D3: Sync qa_set + annotation ---")
    qa_report = sync_qa_set(args.apply)
    for k, v in qa_report.items():
        if k not in ("source_qa", "source_annotation"):
            print(f"  {k}: {v}")
    if qa_report.get("status") == "FAIL":
        print("ERRORS:", qa_report.get("source_qa", {}).get("errors", []))
        print("ERRORS:",
              qa_report.get("source_annotation", {}).get("errors", []))
        sys.exit(1)
    print("\n--- D4: Copy results ---")
    results_report = copy_results(args.apply)
    for k, v in results_report.items():
        if k == "copied":
            print(f"  copied ({len(v)}):")
            for f in v:
                print(f"    {f}")
        elif k == "skipped":
            if v:
                print(f"  skipped ({len(v)}): {v}")
            else:
                print("  skipped: none")
        elif k == "conflicts":
            if v:
                print(f"  CONFLICTS ({len(v)}):")
                for c in v:
                    extra = f" ({c['note']})" if c.get("note") else ""
                    print(f"    {c['file']}{extra}")
        elif k == "missing":
            if v:
                print(f"  MISSING ({len(v)}): {v}")
        elif k == "hash_pairs":
            if v and args.apply:
                print(f"  source->destination hash pairs ({len(v)}):")
                for hp in v:
                    print(f"    {hp['file']}: "
                          f"src={hp['src_hash'][:16]}..."
                          f" dst={hp['dst_hash'][:16]}...")
        else:
            print(f"  {k}: {v}")
    if results_report.get("status") == "BLOCKED":
        print("\n=== BLOCKED: pre-check failed; no files written ===")
        sys.exit(1)
    manifest = generate_manifest()
    mp = args.manifest or str(ROOT / "results" / "submission_manifest.json")
    if args.apply and results_report.get("status") == "OK":
        Path(mp).parent.mkdir(parents=True, exist_ok=True)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n  manifest written to: {mp}")
    else:
        print(f"\n  manifest (would write to {mp}):")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    label = "APPLIED successfully" if args.apply else "DRY RUN complete"
    print(f"\n=== {label} ===")
    if not args.apply:
        print("Re-run with --apply to write files.")


if __name__ == "__main__":
    main()
