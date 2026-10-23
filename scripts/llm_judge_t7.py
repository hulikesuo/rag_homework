# encoding: utf-8
"""Score T7 answers with the configured LLM; this is not human evaluation.

Input is the current T1 cache validated against real corpus chunks.
Gold answers/rubrics shown only to the judge, never to the generator.
Output explicitly labelled ``llm_judge``.

Design constraints:
- Config/env set before any src import (RAG_RESULTS_DIR / RAG_EVAL_JSONL).
- All cache records validated against corpus chunks before any API call.
- Judge cannot distinguish retrieval vs ranking from actual_context alone;
  those are diagnosed by a separate offline top-10 audit, not by the LLM.
- faithfulness=5 prohibits unsupported>0; contradictory outputs are rejected
  and retried once, not silently accepted.
- total_claims=0 yields empty hallucination_rate; macro excludes empty;
  micro is null when denominator is zero.
- Every scored row carries input/answer/context/prompt provenance hashes.
- Per-question JSONL checkpoints are appended immediately and can be resumed.
- The full CSV/summary are written to an acceptance directory by default;
  historical result directories are read-only inputs.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure functions -- no imports from src, no API calls
# ---------------------------------------------------------------------------

def _sha256_str(s):
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _sha256_json(obj):
    """Deterministic SHA-256 of a JSON-serialisable value."""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _hash_answer(answer):
    """Irreversible binding hash of the candidate answer text."""
    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)
    return _sha256_str(answer)

def _hash_context(context_texts):
    """SHA-256 of ordered context text list for provenance."""
    return _sha256_json(context_texts if isinstance(context_texts, list) else [])

def _input_hash(question, gold, rec, eval_fp, level=None):
    """Full input binding: question + gold/rubric + answer + level +
    gold_chunk_ids + rubric (if present) + context IDs and texts +
    eval fingerprint.  Prevents cross-experiment mismatch."""
    payload = {
        "question": question,
        "gold": gold,
        "level": level if level is not None else rec.get("level"),
        "gold_chunk_ids": rec.get("gold_chunk_ids", []),
        "rubric": rec.get("rubric", None),
        "answer": rec.get("answer", ""),
        "context_ids": rec.get("context_chunks", []),
        "context_texts": rec.get("context_texts", []),
        "eval_fingerprint": eval_fp,
    }
    return _sha256_json(payload)

def _validate_one(q, rec, eval_fp, chunk_texts):
    """Single-record validation against known-good question and chunk map.

    Mirrors the strictness of t7_eval._cached_answers: checks id, level,
    question, gold_answer, gold_chunk_ids, eval_fingerprint, context
    chunk ID-to-text consistency, len(ids)==len(texts), and either ok
    with recorded context or error with empty context.
    """
    if str(q.get("id", "")) != str(rec.get("id", "")):
        raise ValueError("id mismatch")
    if str(q.get("level", "")) != str(rec.get("level", "")):
        raise ValueError(f"level mismatch q={q.get('level')} rec={rec.get('level')}")
    if str(q.get("question", "")) != str(rec.get("question", "")):
        raise ValueError("question mismatch")
    if str(q.get("gold_answer", "")) != str(rec.get("gold_answer", "")):
        raise ValueError("gold_answer mismatch")
    if q.get("gold_chunk_ids", []) != rec.get("gold_chunk_ids", []):
        raise ValueError("gold_chunk_ids mismatch")
    if rec.get("eval_fingerprint") != eval_fp:
        raise ValueError("eval_fingerprint mismatch")

    ids = rec.get("context_chunks")
    texts = rec.get("context_texts")
    status = rec.get("rag_status")
    recorded = rec.get("context_recorded")

    if status != "ok":
        # The judge may only score successful answers. Error/missing records
        # are tracked separately by t7_eval and must never reach the LLM.
        raise ValueError(f"rag_status={status}")
    if not isinstance(rec.get("answer"), str):
        raise ValueError("answer not a string")
    if not isinstance(ids, list) or not isinstance(texts, list):
        raise ValueError("context_chunks or context_texts not lists")
    if len(ids) != len(texts):
        raise ValueError("context_chunks and context_texts length mismatch")

    if not rec["answer"].strip():
        raise ValueError("answer is empty even though rag_status=ok")
    if recorded is not True:
        raise ValueError("context_recorded is not True for ok answer")
    if not all(isinstance(cid, str) and isinstance(t, str)
               and cid in chunk_texts and chunk_texts[cid] == t
               for cid, t in zip(ids, texts)):
        raise ValueError("context chunk id/text mismatch against corpus")

def _extract_json(raw):
    if not isinstance(raw, str):
        raise ValueError("judge returned non-text")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge did not return a JSON object")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("judge JSON is not an object")
    return data

def _score_int(data, key, low, high, nullable=False):
    """Reject bool, float, and out-of-range values. Only true int allowed."""
    value = data.get(key)
    if value is None and nullable:
        return ""
    if isinstance(value, float):
        raise ValueError(f"{key} must be an integer, got float {value}")
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    value = int(value)
    if value < low or value > high:
        raise ValueError(f"{key} outside {low}-{high}")
    return value

def _claim_int(data, key):
    """Reject bool, float, and negative. Only true nonnegative int allowed."""
    value = data.get(key)
    if isinstance(value, float):
        raise ValueError(f"{key} must be an integer, got float {value}")
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{key} must be nonnegative")
    return value


# ---------------------------------------------------------------------------
# Judge prompt -- may not guess retrieval/ranking from actual_context
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "你是严格的RAG评测员。只依据给定的问题、参考答案/评分要点、候选答案和"
    "实际生成上下文评分。不得使用外部知识补全，也不得因为答案没有逐字复述"
    "参考答案而机械扣分。只输出一个JSON对象，不要Markdown代码围栏或解释。"
)

_PROMPT_VERSION = 3  # strict faithfulness/unsupported + positive claims + evidence

def _judge_prompt(q, rec):
    payload = {
        "question": q["question"],
        "level": q["level"],
        "gold_answer_or_rubric": q["gold_answer"],
        "candidate_answer": rec.get("answer", ""),
        "actual_context": rec.get("context_texts", []),
        "instructions": {
            "faithfulness_1_5": (
                "1-5整数；事实陈述是否被actual_context支持。"
                "5=全部有支撑且无矛盾，此时unsupported_claims必须为0"),
            "unsupported_claims": (
                "整数；候选中无上下文依据的事实陈述数。"
                "faithfulness=5时此值必须为0"),
            "total_claims": "整数；候选中的事实陈述总数",
            "total_claims_must_be_positive": (
                "答案若没有可核验的事实断言，则不能报0来获得通过；"
                "应把此视为无法评分，并据实说明。hallucination只统计真claims>0的题目"),
            "correctness_0_5": "0-5整数；L1-L3按参考答案判断，L4按rubric",
            "L4_rubric_0_5": "仅L4填0-5；其他留空",
            "failure_stage": (
                "没有实质失败填none；答案自身错漏填generation_failure；"
                "无法确定原因填eval_ambiguity或insufficient_evidence_unknown。"
                "不得猜测retrieval_failure或ranking_failure；你没有候选"
                "列表信息，不能区分检索漏召回和排序过滤"),
            "evidence": (
                "必填，不超过80字，指出支持或矛盾的具体上下文依据；"
                "禁止只写...或空"),
            "improvement": (
                "必填，不超过80字，可执行改进；禁止只写...或空"),
        },
        "output_schema": {
            "faithfulness_1_5": 1, "unsupported_claims": 0,
            "total_claims": 1, "correctness_0_5": 0,
            "L4_rubric_0_5": None, "failure_stage": "none",
            "evidence": "...", "improvement": "...",
        },
    }
    return (
        "请评分并只返回JSON。factual claim按可独立核验的断言计；"
        "寒暄、标题和不含事实的过渡句不计。\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _prompt_messages(q, rec):
    """The exact messages sent to the judge, used for provenance hashing."""
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _judge_prompt(q, rec)},
    ]


def _prompt_hash(q, rec):
    """Deterministic SHA-256 of the complete judge message set (system +
    user prompt template content), including the prompt version."""
    return _sha256_json({
        "version": _PROMPT_VERSION,
        "messages": _prompt_messages(q, rec),
    })


def _prompt_text(q, rec):
    """The exact user prompt sent to the judge, for audit-only storage."""
    return _judge_prompt(q, rec)

# Allowed failure_stage values for the judge response.
_ALLOWED_STAGES = frozenset({
    "none", "generation_failure", "eval_ambiguity",
    "insufficient_evidence_unknown", "unknown",
})


def _reason_code(exc):
    """Safe fixed reason code for judge validation/API failures.

    Only stable category names are recorded, never exception messages or
    endpoint/credential text.
    """
    exc_type = type(exc).__name__
    if exc_type == "ValueError" and "exceeds total_claims" in str(exc):
        return "claims_unsupported_gt_total"
    if exc_type == "ValueError" and "faithfulness=5" in str(exc):
        return "faithfulness5_with_unsupported"
    if exc_type == "ValueError" and "total_claims must be positive" in str(exc):
        return "claims_zero_invalid"
    if exc_type == "ValueError" and "invalid failure_stage" in str(exc):
        return "invalid_failure_stage"
    if exc_type == "ValueError" and "L4 score is required" in str(exc):
        return "l4_score_missing"
    too_low_evidence = (
        "evidence" in str(exc).lower() or "improvement" in str(exc).lower()
    )
    if exc_type == "ValueError" and too_low_evidence:
        return "evidence_or_improvement_missing"
    if exc_type == "ValueError":
        return "judge_schema_invalid"
    if exc_type.startswith(("Connection", "Timeout", "APIConnection")):
        return "api_connection"
    if exc_type.startswith("RateLimit"):
        return "api_rate_limit"
    return "api_or_runtime_failure"

# ---------------------------------------------------------------------------
# Judge execution with faithfulness/unsupported self-consistency retry
# ---------------------------------------------------------------------------

def judge_one(gen_obj, q, rec, eval_fp, cache_sha):
    """Call the LLM judge. Returns a dict (CSV row) on success or error."""
    last_error = ""
    last_reason = ""
    attempts = 0
    raw_output = ""
    judge_messages = _prompt_messages(q, rec)
    judge_prompt = _prompt_text(q, rec)
    attempt_history = []
    for attempt in range(2):
        attempts += 1
        attempt_raw = ""
        attempt_error_type = ""
        attempt_reason = ""
        try:
            raw = gen_obj.complete(
                judge_prompt, max_tokens=700, temperature=0.0,
                messages=judge_messages,
            )
            raw_output = raw
            attempt_raw = raw
            data = _extract_json(raw)
            total = _claim_int(data, "total_claims")
            if total <= 0:
                raise ValueError("total_claims must be positive; "
                                 "no factual claims cannot be scored as pass")
            unsupported = _claim_int(data, "unsupported_claims")
            if unsupported > total:
                raise ValueError("unsupported_claims exceeds total_claims")
            faithfulness = _score_int(data, "faithfulness_1_5", 1, 5)
            if faithfulness == 5 and unsupported > 0:
                raise ValueError(
                    f"faithfulness=5 but unsupported_claims={unsupported}>0")
            evidence = str(data.get("evidence", ""))
            improvement = str(data.get("improvement", ""))
            if not evidence.strip() or evidence.strip() in ("...", "……"):
                raise ValueError("evidence is required and must be substantive")
            if not improvement.strip() or improvement.strip() in ("...", "……"):
                raise ValueError("improvement is required and must be substantive")
            l4 = _score_int(data, "L4_rubric_0_5", 0, 5, nullable=True)
            if q["level"] == "L4" and l4 == "":
                raise ValueError("L4 score is required for L4")
            if q["level"] != "L4":
                l4 = ""

            stage = str(data.get("failure_stage", "")).strip()
            if stage not in _ALLOWED_STAGES:
                raise ValueError("invalid failure_stage: " + stage)

            hrate = round(unsupported / total, 4) if total else ""

            answer_text = rec.get("answer", "")
            ctx_texts = rec.get("context_texts", [])
            p_hash = _prompt_hash(q, rec)

            return {
                "id": q["id"], "level": q["level"],
                "metric_type": "llm_judge_not_human",
                "judge_model": gen_obj.config.LLM_MODEL,
                "faithfulness_1_5": faithfulness,
                "unsupported_claims": unsupported,
                "total_claims": total,
                "hallucination_rate": hrate,
                "correctness_0_5": _score_int(data, "correctness_0_5", 0, 5),
                "L4_rubric_0_5": l4,
                "failure_stage": stage,
                "evidence": evidence[:240],
                "improvement": improvement[:240],
                "judge_status": "ok",
                "judge_error": "",
                "judge_error_type": "",
                "attempt_history": attempt_history,
                "answer_hash": _hash_answer(answer_text),
                "context_hash": _hash_context(ctx_texts),
                "input_sha256": _input_hash(q["question"], q["gold_answer"],
                                            rec, eval_fp, level=q.get("level")),
                "eval_fingerprint": eval_fp,
                "prompt_version": _PROMPT_VERSION,
                "prompt_hash": p_hash,
                "source_cache_sha": cache_sha,
                "raw_output": raw_output,
                "prompt": judge_prompt,
                "judge_messages_json": json.dumps(
                    judge_messages, ensure_ascii=False),
                "attempts": attempts,
            }
        except Exception as exc:
            last_error = type(exc).__name__
            attempt_error_type = type(exc).__name__
            attempt_reason = _reason_code(exc)
            last_reason = attempt_reason
            attempt_history.append({
                "attempt": attempts,
                "raw_output": attempt_raw,
                "validation_error_type": attempt_error_type,
                "reason_code": attempt_reason,
                "messages_json": json.dumps(
                    judge_messages, ensure_ascii=False),
                "messages_sha256": _sha256_json(judge_messages),
            })

    answer_text = rec.get("answer", "")
    ctx_texts = rec.get("context_texts", [])
    p_hash = _prompt_hash(q, rec)
    return {
        "id": q["id"], "level": q["level"],
        "metric_type": "llm_judge_not_human",
        "judge_model": gen_obj.config.LLM_MODEL,
        "faithfulness_1_5": "", "unsupported_claims": "", "total_claims": "",
        "hallucination_rate": "", "correctness_0_5": "", "L4_rubric_0_5": "",
        "failure_stage": "", "evidence": "", "improvement": "",
        "judge_status": "error", "judge_error": last_error,
        "judge_error_type": last_error,
        "attempt_history": attempt_history,
        "answer_hash": _hash_answer(answer_text),
        "context_hash": _hash_context(ctx_texts),
        "input_sha256": _input_hash(q["question"], q["gold_answer"],
                                    rec, eval_fp, level=q.get("level")),
        "eval_fingerprint": eval_fp,
        "prompt_version": _PROMPT_VERSION,
        "prompt_hash": p_hash,
        "source_cache_sha": cache_sha,
        "raw_output": raw_output,
        "prompt": judge_prompt,
        "judge_messages_json": json.dumps(
            judge_messages, ensure_ascii=False),
        "attempts": attempts,
        "reason_code": last_reason,
    }


def _read_checkpoint(path):
    """Read JSONL checkpoint rows keyed by question id."""
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def _checkpoint_matches(old, q, rec, eval_fp, cache_sha):
    """A successful checkpoint is reusable only with all input bindings."""
    return bool(
        old
        and old.get("judge_status") == "ok"
        and old.get("prompt_hash") == _prompt_hash(q, rec)
        and old.get("input_sha256") == _input_hash(
            q["question"], q["gold_answer"], rec, eval_fp,
            level=q.get("level"))
        and old.get("source_cache_sha") == cache_sha
        and old.get("eval_fingerprint") == eval_fp
    )


def _append_checkpoint(path, row):
    """Immediately persist one judged row so interrupted runs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _checkpoint_sequence(path):
    """Count JSONL checkpoint writes, including superseded rows."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip())


def _write_summary(*, rows, qas, cache_path, cache_sha, eval_path, current_fp,
                   out_dir, checkpoint_path, workers, judge_model):
    """Write CSV plus stratified summary from checkpoint-selected rows."""
    rows = sorted(rows, key=lambda r: r["id"])
    fields = [
        "id", "level", "metric_type", "judge_model",
        "faithfulness_1_5", "unsupported_claims", "total_claims",
        "hallucination_rate", "correctness_0_5", "L4_rubric_0_5",
        "failure_stage", "evidence", "improvement",
        "judge_status", "judge_error",
        "answer_hash", "context_hash", "input_sha256",
        "eval_fingerprint", "prompt_version", "prompt_hash",
        "source_cache_sha",
        "judge_error_type", "reason_code", "raw_output", "prompt",
        "judge_messages_json", "attempts",
    ]
    csv_path = out_dir / "t7_llm_judge.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    levels = {}
    for lv in ("L1", "L2", "L3", "L4", "ALL"):
        subset = rows if lv == "ALL" else [r for r in rows if r["level"] == lv]
        valid = [r for r in subset if r["judge_status"] == "ok"]
        u_sum = sum(int(r["unsupported_claims"]) for r in valid
                    if r["unsupported_claims"] != "")
        c_sum = sum(int(r["total_claims"]) for r in valid
                    if r["total_claims"] != "")
        levels[lv] = {
            "n": len(subset), "scored": len(valid),
            "faithfulness_1_5_mean": _mean(valid, "faithfulness_1_5"),
            "correctness_0_5_mean": _mean(valid, "correctness_0_5"),
            "l4_rubric_0_5_mean": _mean(valid, "L4_rubric_0_5"),
            "hallucination_rate_macro": _mean(valid, "hallucination_rate"),
            "hallucination_rate_micro": (
                round(u_sum / c_sum, 4) if c_sum else None),
            "unsupported_total": u_sum,
            "claims_total": c_sum,
        }

    stage_keys = [
        "none", "generation_failure", "eval_ambiguity",
        "insufficient_evidence_unknown", "unknown",
    ]
    failure = {k: sum(r.get("failure_stage") == k for r in rows)
               for k in stage_keys}
    failure["unknown"] += sum(
        r.get("judge_status") == "ok"
        and r.get("failure_stage") not in stage_keys
        for r in rows
    )

    summary = {
        "metric_type": "llm_judge_not_human",
        "judge_model": judge_model,
        "judge_prompt_version": _PROMPT_VERSION,
        "prompt_template_sha256": _sha256_json({
            "version": _PROMPT_VERSION,
            "system": _JUDGE_SYSTEM,
            "prompt_prefix": (
                "请评分并只返回JSON。factual claim按可独立核验的断言计；"
                "寒暄、标题和不含事实的过渡句不计。\n"),
        }),
        "source_cache": str(cache_path),
        "source_cache_sha256": cache_sha,
        "source_eval": str(eval_path),
        "source_eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "t1_answer_generation": "reused_existing_cache_no_regeneration",
        "checkpoint": str(checkpoint_path),
        "checkpoint_rows": _checkpoint_sequence(checkpoint_path),
        "workers": workers,
        "eval_fingerprint": current_fp,
        "n": len(rows),
        "ok": sum(r["judge_status"] == "ok" for r in rows),
        "errors": sum(r["judge_status"] == "error" for r in rows),
        "requested": len(qas),
        "completed_all_requested": (
            len(rows) == len(qas)
            and all(r["judge_status"] == "ok" for r in rows)
        ),
        "rejected_before_api": 0,
        "rejected_details": [],
        "levels": levels,
        "failure_stage_counts": failure,
        "hallucination_rate_micro": levels["ALL"]["hallucination_rate_micro"],
    }
    summary_path = out_dir / "t7_llm_judge_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path, summary


def _mean(rows, key):
    values = [float(r[key]) for r in rows
              if r[key] != "" and r[key] is not None]
    return round(statistics.mean(values), 4) if values else None


def _check_qa_cache_integrity(qas, raw_records, cache_missing_id,
                              duplicate_ids):
    """Return a list of error strings; empty list means QA<->cache alignment
    is exact (no duplicate, missing, extra, or id-less records)."""
    qa_id_list = [q.get("id") for q in qas]
    qa_ids = set(qa_id_list)
    qa_duplicates = sorted(
        [qid for qid in qa_ids if qa_id_list.count(qid) > 1])
    qa_missing_id = [i for i, q in enumerate(qas) if q.get("id") is None]
    cache_ids = set(raw_records)
    missing_qa = sorted(qid for qid in (qa_ids - cache_ids) if qid is not None)
    extra_cache = sorted(cid for cid in (cache_ids - qa_ids) if cid is not None)
    errors = []
    if qa_duplicates:
        errors.append("Duplicate IDs in QA set: " + ", ".join(qa_duplicates))
    if qa_missing_id:
        errors.append("QA entries missing id at index: " + ", ".join(
            str(i) for i in qa_missing_id))
    if duplicate_ids:
        errors.append("Duplicate IDs in cache: " + ", ".join(
            sorted(set(duplicate_ids))))
    if cache_missing_id:
        errors.append(f"{len(cache_missing_id)} cache entries missing id")
    if missing_qa:
        errors.append("QA IDs missing from cache: " + ", ".join(missing_qa))
    if extra_cache:
        errors.append("Cache IDs not in QA set: " + ", ".join(extra_cache))
    return errors


def _validate_all(qas, raw_records, current_fp, chunk_texts):
    """Return (field_errors, ok) where field_errors is a list of (id, reason)
    and ok is True only when there are zero field errors."""
    field_errors = []
    for q in qas:
        rec = raw_records.get(q["id"])
        try:
            _validate_one(q, rec, current_fp, chunk_texts)
        except ValueError as exc:
            field_errors.append((q["id"], str(exc)))
    return field_errors, not field_errors


# ---------------------------------------------------------------------------
# Main -- deferred imports, full validation, then API
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--eval", dest="eval_path", type=Path,
                        default=Path("data/eval/qa_set.jsonl"))
    parser.add_argument("--results", type=Path,
                        default=Path("repair_outputs/revised_full_20261022"))
    parser.add_argument("--out-dir", type=Path, dest="out_dir",
                        default=Path("repair_outputs/acceptance_20261023/t7"))
    parser.add_argument("--ids", type=str, default="",
                        help="comma-separated question ids, e.g. q01,q02")
    parser.add_argument("--force", action="store_true",
                        help="re-judge existing checkpoint rows")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    root = args.root.resolve()
    eval_path = (root / args.eval_path).resolve() if not args.eval_path.is_absolute() else args.eval_path
    result_dir = (root / args.results).resolve() if not args.results.is_absolute() else args.results
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    if args.workers != 1 and args.workers != 2:
        raise SystemExit("workers must be 1 or 2 (API concurrency limit)")
    selected_ids = [x.strip() for x in args.ids.split(",") if x.strip()]

    # Set env before any src import.
    os.environ["RAG_EVAL_JSONL"] = str(eval_path)
    os.environ["RAG_RESULTS_DIR"] = str(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    # Deferred imports after env set.
    sys.path.insert(0, str(root))
    from src import config, loader, chunker
    from experiments.common import (
        BASELINE_PIPELINE, experiment_signature,
    )

    config.require_llm()
    from src import generator

    # Load full eval questions and corpus. The experiment fingerprint must be
    # computed over the full 30-question set even for a --ids subset run,
    # otherwise cache fingerprints would not match.
    qas_all = [
        json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    docs = loader.read_all_docs(config.CORPUS_DIR)

    # Compute current experiment fingerprint.
    current_fp = experiment_signature(docs, qas_all)

    qas = qas_all
    if selected_ids:
        qas = [q for q in qas_all if q["id"] in selected_ids]
        if not qas:
            raise SystemExit("No matching ids found in eval set: " +
                             ", ".join(selected_ids))

    # Build chunk-text map via the same deterministic chunker used by the
    # cached experiment. This path does not touch embedding/index/model code.
    chunks = chunker.chunk_documents(
        docs,
        method=BASELINE_PIPELINE["chunk_method"],
        size=BASELINE_PIPELINE["chunk_size"],
        overlap=BASELINE_PIPELINE["chunk_overlap"],
    )
    chunk_texts = {c["chunk_id"]: c["text"] for c in chunks}

    # Load T1 cache.
    cache_path = result_dir / "t1_baseline.jsonl"
    if not cache_path.exists():
        raise SystemExit("T1 cache not found: " + str(cache_path))
    cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    raw_records = {}
    duplicate_ids = []
    cache_missing_id = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            rid = rec.get("id")
            if rid is None:
                cache_missing_id.append("<missing-id>")
                continue
            if rid in raw_records:
                duplicate_ids.append(rid)
            raw_records[rid] = rec

    # --- Reject any inconsistency before ever touching the API ---
    # With --ids we intentionally run a subset of the full 30-question cache;
    # alignment must still be checked for the selected subset, not the full set.
    if selected_ids:
        # subset mode: extra cache IDs are normal and not a mismatch
        subset_recs = {k: v for k, v in raw_records.items()
                       if k in selected_ids}
        subset_duplicate_ids = [rid for rid in duplicate_ids
                                if rid in selected_ids]
        errors = _check_qa_cache_integrity(
            qas, subset_recs, cache_missing_id, subset_duplicate_ids)
    else:
        errors = _check_qa_cache_integrity(
            qas, raw_records, cache_missing_id, duplicate_ids)
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        raise SystemExit("Aborting: QA-cache mismatch; zero API calls made.")

    # --- Validate every record against corpus chunks ---
    field_errors, _ = _validate_all(qas, raw_records, current_fp, chunk_texts)
    if field_errors:
        print(f"[validate] {len(field_errors)} field-level failures:")
        for rid, reason in field_errors:
            print(f"  {rid}: {reason}")
        raise SystemExit(
            f"Aborting: {len(field_errors)} records failed validation; "
            "zero API calls made.")

    print(f"[judge] {len(qas)} questions, model={config.LLM_MODEL}, "
          f"workers={args.workers}, prompt_version={_PROMPT_VERSION}")

    # Run judge.
    def _wrap(pair):
        q = pair
        rec = raw_records[q["id"]]
        return judge_one(generator, q, rec, current_fp, cache_sha)

    checkpoint_path = out_dir / "judge_checkpoint.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = _read_checkpoint(checkpoint_path)
    todo = []
    for q in qas:
        rec = raw_records[q["id"]]
        old = rows_by_id.get(q["id"])
        if not args.force and _checkpoint_matches(
                old, q, rec, current_fp, cache_sha):
            continue
        todo.append(q)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_id = {
                pool.submit(_wrap, q): q["id"] for q in todo
            }
            for future in as_completed(future_to_id):
                qid = future_to_id[future]
                row = future.result()
                _append_checkpoint(checkpoint_path, row)
                rows_by_id[qid] = row

    missing = [q["id"] for q in qas if q["id"] not in rows_by_id]
    if missing:
        raise SystemExit("Checkpoint missing after run: " + ", ".join(missing))

    rows = [rows_by_id[q["id"]] for q in qas]
    csv_path, summary_path, summary = _write_summary(
        rows=rows,
        qas=qas,
        cache_path=cache_path,
        cache_sha=cache_sha,
        eval_path=eval_path,
        current_fp=current_fp,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        workers=args.workers,
        judge_model=config.LLM_MODEL,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] {csv_path}")
    print(f"[save] {summary_path}")


if __name__ == "__main__":
    main()
