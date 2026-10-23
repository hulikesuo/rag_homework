"""T7 分层检索评测 + 启发式诊断 + 人工评分工作表。

默认不调用生成 LLM；仅 --generate 显式补生成没有有效缓存记录的题。
合并 T1 与 t7_generated_cache.jsonl 的有效记录，成功/失败/缺失分别计数。
1–5 忠实度和 L4 0–5 rubric 分数必须人工填写，不用启发式数值代填。
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import (
    BASELINE_PIPELINE, baseline_answer, experiment_signature, load_env, save_csv,
)
from src import config, pipeline, evaluate


SCORING_GUIDE = (
    "人工评分：先阅读 question、gold_answer/rubric、answer 和实际 context_texts。"
    "faithfulness_1_5 仅判断答案陈述是否被实际上下文支持："
    "5=全部有支撑且无矛盾；4=基本有支撑，仅轻微无依据细节；"
    "3=部分有支撑，但有实质无依据陈述；2=多数无支撑或有重大矛盾；"
    "1=几乎全部无支撑/编造。不得用 gold 或候选检索片段替代实际 context。"
    "缺答案、生成错误或缺实际 context 时留空并说明原因；空 context 已记录时，"
    "按实际空资料判断，恰当拒答可获高忠实度，正确性另评。"
    "L4_rubric_0_5 仅 L4 填：按题目既定五项 rubric 每项 1 分，"
    "允许有证据且不矛盾的不同结论；无既定五项 rubric 时先人工制定并记录，"
    "勿由 EM、ROUGE 或启发式自动换算。非 L4 留空。"
    "在 reviewer/notes 记录评分人、逐项得分、依据与分歧；"
    "heuristic_* 是字符/数字重合诊断，不是课程要求的人工或 LLM 1–5 评分。"
)


def _cached_answers(path, qas, fingerprint, chunks):
    """分别返回签名/上下文有效的成功答案与生成失败，不把失败当缺失。"""
    current = {q["id"]: q for q in qas}
    chunk_texts = {c["chunk_id"]: c["text"] for c in chunks}
    answers, errors = {}, {}
    if not os.path.exists(path):
        return answers, errors
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or not isinstance(rec.get("id"), (str, int)):
                continue
            q = current.get(rec["id"])
            ids, texts = rec.get("context_chunks"), rec.get("context_texts")
            status = rec.get("rag_status")
            if (q is None or rec.get("eval_fingerprint") != fingerprint
                    or rec.get("question") != q["question"]
                    or rec.get("gold_answer") != q["gold_answer"]
                    or rec.get("gold_chunk_ids") != q["gold_chunk_ids"]
                    or rec.get("level") != q["level"]
                    or status not in ("ok", "error")
                    or not isinstance(rec.get("answer"), str)
                    or not isinstance(ids, list) or not isinstance(texts, list)
                    or len(ids) != len(texts)):
                continue
            recorded = rec.get("context_recorded")
            if status == "ok":
                if not rec["answer"].strip() or recorded is not True:
                    continue
            elif not isinstance(rec.get("rag_error"), str) or not rec["rag_error"]:
                continue
            # 检索失败可以没有快照；不得把非空/不明的快照当作未记录。
            if recorded is not True and not (status == "error" and recorded is False
                                             and not ids and not texts):
                continue
            # 本基线无压缩；核对快照，不以候选 retrieved 重建上下文。
            if not all(isinstance(cid, str) and isinstance(text, str)
                       and cid in chunk_texts and chunk_texts[cid] == text
                       for cid, text in zip(ids, texts)):
                continue
            if status == "ok":
                answers[q["id"]] = rec
                errors.pop(q["id"], None)
            elif q["id"] not in answers:
                errors[q["id"]] = rec
    return answers, errors


def _generation_outputs(qas, answers, fingerprint):
    faith_rows, worksheet = [], []
    for q in qas:
        rec = answers.get(q["id"], {})
        ans = rec.get("answer", "")
        measured = (rec.get("rag_status") == "ok" and bool(ans.strip())
                    and rec.get("context_recorded") is True)
        row = {
            "id": q["id"], "level": q["level"],
            # 保留旧 CSV 列，但歧义分数字段留空，诊断仅写 heuristic_*。
            "faithfulness": "", "unsupported_sents": "", "unsupported_ratio": "",
            "answer_sents": "", "correctness_em": "", "correctness_rouge_l": "",
            "heuristic_faithfulness": "", "heuristic_unsupported_sents": "",
            "heuristic_unsupported_ratio": "", "metric_type": "heuristic_not_human_or_llm_rated",
            "status": "measured_heuristic" if measured else rec.get("rag_status", "missing_answer"),
            "error_type": rec.get("rag_error", ""),
            "eval_fingerprint": fingerprint,
        }
        if measured:
            f, unsup = evaluate.faithfulness_factcheck(rec["context_texts"], ans)
            sents = evaluate.normalized_sents(ans)
            row.update(
                heuristic_faithfulness=round(f, 3),
                heuristic_unsupported_sents=len(unsup), answer_sents=len(sents),
                heuristic_unsupported_ratio=round(len(unsup) / len(sents), 4) if sents else 0.0,
                correctness_em=evaluate.correctness_em(q["gold_answer"], ans),
                correctness_rouge_l=round(evaluate.rouge_l_f1(q["gold_answer"], ans), 4),
            )
        faith_rows.append(row)
        worksheet.append({
            "id": q["id"], "level": q["level"], "question": q["question"],
            "gold_answer": q["gold_answer"],
            "gold_chunk_ids": json.dumps(q["gold_chunk_ids"], ensure_ascii=False),
            "rubric": json.dumps(q.get("rubric", q.get("scoring_rubric", q["gold_answer"])),
                                 ensure_ascii=False),
            "eval_fingerprint": fingerprint, "answer": ans,
            "context_chunks": json.dumps(rec.get("context_chunks", []), ensure_ascii=False),
            "context_texts": json.dumps(rec.get("context_texts", []), ensure_ascii=False),
            "context_recorded": rec.get("context_recorded", False),
            "answer_status": rec.get("rag_status", "missing_or_stale_cache"),
            "error_type": rec.get("rag_error", ""),
            "faithfulness_1_5": "", "L4_rubric_0_5": "",
            "reviewer": "", "notes": "", "scoring_guidance": SCORING_GUIDE,
        })
    fields = ["id", "level", "faithfulness", "unsupported_sents", "answer_sents",
              "unsupported_ratio", "correctness_em", "correctness_rouge_l",
              "heuristic_faithfulness", "heuristic_unsupported_sents",
              "heuristic_unsupported_ratio", "metric_type", "status", "error_type",
              "eval_fingerprint"]
    save_csv("t7_faithfulness.csv", faith_rows, fields)
    worksheet_fields = [
        "id", "level", "question", "gold_answer", "gold_chunk_ids", "rubric",
        "eval_fingerprint", "answer", "context_chunks", "context_texts", "context_recorded",
        "answer_status", "error_type", "faithfulness_1_5", "L4_rubric_0_5",
        "reviewer", "notes", "scoring_guidance",
    ]
    worksheet_path = save_csv("t7_manual_scoring.csv", worksheet, worksheet_fields,
                              overwrite=False)
    print(f"[t7] 人工评分表 {len(worksheet)} 题；评分栏留空，指引见 scoring_guidance。")
    total = sum(r["answer_sents"] for r in faith_rows if isinstance(r["answer_sents"], int))
    unsupported = sum(r["heuristic_unsupported_sents"] for r in faith_rows
                      if isinstance(r["heuristic_unsupported_sents"], int))
    print("[t7] heuristic unsupported sentence ratio:",
          round(unsupported / total, 4) if total else 0.0,
          f"({unsupported}/{total}); not human/LLM-rated")
    return worksheet_path


def run(do_generate=False):
    env = load_env()
    docs, qas = env["docs"], env["qas"]
    fingerprint = experiment_signature(docs, qas)
    pipe = pipeline.RAGPipeline(docs, **BASELINE_PIPELINE) if qas else None

    by_level = defaultdict(list)
    for q in qas:
        hits = pipe.retrieve(q["question"], k=10, mode="hybrid")
        got = [c["chunk_id"] for c, _ in hits]
        by_level[q["level"]].append({
            "recall@5": evaluate.recall_at_k(q["gold_chunk_ids"], got, 5),
            "hit@5": evaluate.hit_at_k(q["gold_chunk_ids"], got, 5),
            "mrr@10": evaluate.mrr_at_k(q["gold_chunk_ids"], got, 10),
            "map@10": evaluate.average_precision_at_k(q["gold_chunk_ids"], got, 10),
            "p@5": evaluate.precision_at_k(q["gold_chunk_ids"], got, 5),
        })
    level_rows = []
    groups = [(lv, by_level[lv]) for lv in sorted(by_level)]
    groups.append(("ALL", [a for arr in by_level.values() for a in arr]))
    for lv, arr in groups:
        row = {"level": lv, "n": len(arr)}
        for label, key in (
            ("Hit@5(课件口径)", "hit@5"), ("Recall@5(分片)", "recall@5"),
            ("P@5", "p@5"), ("MRR@10", "mrr@10"), ("MAP@10", "map@10"),
        ):
            row[label] = round(sum(a[key] for a in arr) / len(arr), 4) if arr else 0.0
        level_rows.append(row)
    save_csv("t7_metrics.csv", level_rows,
             ["level", "n", "Hit@5(课件口径)", "Recall@5(分片)", "P@5", "MRR@10", "MAP@10"])

    chunks = pipe.chunks if pipe else []
    answers, errors = _cached_answers(
        os.path.join(config.RESULTS_DIR, "t1_baseline.jsonl"), qas, fingerprint, chunks)
    generated_cache = os.path.join(config.RESULTS_DIR, "t7_generated_cache.jsonl")
    generated_answers, generated_errors = _cached_answers(
        generated_cache, qas, fingerprint, chunks)
    # 有效成功答案优先，失败记录单独保留；缺失只指没有任何有效记录。
    answers.update(generated_answers)
    errors.update(generated_errors)
    errors = {qid: rec for qid, rec in errors.items() if qid not in answers}
    matched, cached_errors = len(answers), len(errors)
    missing = [q for q in qas if q["id"] not in answers and q["id"] not in errors]
    generated_count = 0
    if do_generate and missing:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        # 逐题追加并 flush，保留历史签名/失败记录，后续离线运行可复用。
        with open(generated_cache, "a", encoding="utf-8") as handle:
            for q in missing:
                rec = {**q, "eval_fingerprint": fingerprint,
                       **baseline_answer(pipe, q["question"])}
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                generated_count += 1
                target = answers if rec["rag_status"] == "ok" else errors
                target[q["id"]] = rec
    available = len(answers)
    missing_count = len(qas) - available - len(errors)
    worksheet_path = _generation_outputs(qas, {**errors, **answers}, fingerprint)
    save_csv("t7_generation_status.csv", [{
        "status": "heuristic_only_manual_pending" if available == len(qas) else "partial_or_not_measured",
        "matched_answers": matched, "cached_generation_errors": cached_errors,
        "missing_answers": missing_count,
        "api_generation_enabled": do_generate, "eval_jsonl": config.EVAL_JSONL,
        "available_answers": available, "generation_errors": len(errors),
        "generated_questions": generated_count,
        "eval_fingerprint": fingerprint, "manual_worksheet": worksheet_path,
    }], ["status", "matched_answers", "cached_generation_errors", "missing_answers",
         "api_generation_enabled", "eval_jsonl", "available_answers", "generation_errors",
         "generated_questions", "eval_fingerprint", "manual_worksheet"])
    print(f"[t7] 签名匹配成功缓存 {matched} 条，失败缓存 {cached_errors} 条；"
          f"可评答案 {available}/{len(qas)}，生成失败 {len(errors)}，缺失 {missing_count}；"
          f"本次补生成 {generated_count}；LLM 补生成 {'已显式开启' if do_generate else '关闭'}。")
    for row in level_rows:
        print(row)
    return level_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="显式允许调用 LLM 补生成缺失答案")
    run(do_generate=parser.parse_args().generate)
