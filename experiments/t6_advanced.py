"""T6：路由、oracle 难度提示分支，以及最多两轮的检索生成循环。

默认 run(use_llm=False) / CLI 不调用 LLM，向量检索仅使用本地缓存模型。
显式 --generate 才启用生成，并在读取数据/构建索引前检查 LLM 配置。
支持分支读取题集 gold level，是 oracle 难度提示实验，不是自动判级。

运行时写出 t6_routing.csv、t6_support.csv、t6_retry.csv、t6_retry_trace.jsonl、
t6_retry_summary.json；返回并打印控制流程 Markdown，不自动绘制任何指标图。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline, evaluate, config, utils, loader, generator  # noqa: E402
from experiments.common import save_csv  # noqa: E402


CONTROL_FLOW_MARKDOWN = """### T6 实际控制流程

1. 默认离线：路由只用启发式，精确问题走 `bm25`，综合问题走 `hybrid`。
   支持分支仅检索；循环为 `not_measured`，不生成、不 judge、不改写。
   循环 attempts/retries 和质量指标留空，不把未测记作失败或成功。
   离线向量模型仅从本地缓存加载，缓存缺失会报错，不自动下载。
2. 显式 `--generate`：先检查 LLM 配置，再读取数据和构建索引。
   路由 LLM 仅接受“精确/综合”，失败或格式异常时退回启发式并记录来源。
3. 支持分支直接使用题集 **gold level（oracle 难度提示）**：
   L1/L2 → top-3 + concise；L3/L4 → top-8 + evidence；不是自动判级。
4. 循环仅抽取题集前六题，每题最多两轮：原问题 → hybrid 检索 →
   基于原问题生成 → 用原问题、完整证据和完整答案 judge。
   仅完整输出 `SUFFICIENT` 才停止为 sufficient；`UNSURE` 为 insufficient，
   其他输出记为 invalid_judgment。首轮未充分才改写检索 query，
   第二轮仍用原问题生成和判定；最后一轮失败直接结束，不再改写。
   API/生成异常记录为 error 并结束该题；空改写不启动第二轮。
   失败轮完整留痕，返回此前最后一个完整成功轮次的答案/质量并记录轮号；
   首轮失败不返回答案，回退答案也不改变 error 状态。
5. 逐轮保存 query、证据（含文本/分数）、答案、judge 原文和解析判定。
   首末质量仅为参考答案包含、ROUGE-L、证据忠实度等代理，不是正确率；
   judge sufficient 也不是人工正确性结论。参考答案仅用于事后评估，不传给 LLM。
   二次检索比例 = 实际执行第二次检索的题数 / 实际进入循环的题数；
   离线或空样本时为空。汇总同时保留候选样本数和实际测量数。
6. 不自动作指标图，主线程可使用返回的数据统一绘图。
"""

QUALITY_FIELDS = ("answered_proxy", "reference_contains_proxy", "rouge_l_proxy",
                  "faithfulness_proxy")


def _has_keyword(q, words):
    return any(w in q for w in words)


def _route_heuristic(q):
    if _has_keyword(q, {"判断", "结合", "比较", "分析", "如何", "影响", "是否"}):
        return "hybrid"
    return "bm25"


def _route_decision(q, use_llm):
    if use_llm:
        try:
            raw = generator.complete(
                f"问题『{q}』需要精确的事实查询还是综合多文档的分析？只回答 精确/综合。",
                max_tokens=8).strip()
            if raw in ("精确", "综合"):
                return {"mode": {"精确": "bm25", "综合": "hybrid"}[raw],
                        "route_source": "llm", "route_raw": raw}
            source = "heuristic_invalid_llm_output"
        except Exception:
            raw, source = None, "heuristic_llm_error"
    else:
        raw, source = None, "heuristic_offline"
    return {"mode": _route_heuristic(q), "route_source": source, "route_raw": raw}


def _route_llm(q):
    """兼容原路由入口；综合始终走 hybrid。"""
    return _route_decision(q, True)["mode"]


def _branch_recall(pipe, q, mode):
    if mode not in ("bm25", "hybrid"):
        raise ValueError(f"未知 T6 路由：{mode}")
    return [c["chunk_id"] for c, _ in pipe.retrieve(q, k=5, mode=mode)]


def _evidence(hits):
    return [{**chunk, "score": float(score)} for chunk, score in hits]


def _quality(answer, context, gold_answer):
    """离线计算生成质量代理；缺参考答案时不填参考答案指标。"""
    faithfulness, _ = evaluate.faithfulness_factcheck(
        [c["text"] for c in context], answer)
    has_gold = bool(gold_answer and gold_answer.strip())
    return {
        "answered_proxy": evaluate.is_answered(answer),
        "reference_contains_proxy": (
            evaluate.correctness_em(gold_answer, answer) if has_gold else None),
        "rouge_l_proxy": (
            evaluate.rouge_l_f1(gold_answer, answer) if has_gold else None),
        "faithfulness_proxy": faithfulness,
    }


def _retry(pipe, question, use_llm=False, max_iter=2, gold_answer=None):
    """返回实际执行记录；只改检索 query，生成/judge 永远使用原问题。"""
    result = {
        "question": question, "status": "not_measured", "answer": None,
        "attempts": None, "extra_retries": None, "second_retrieval": None,
        "initial_quality": None, "final_quality": None,
        "returned_answer_round": None, "rounds": [],
    }
    if not use_llm:
        return result
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter 必须为正整数（实际最多两轮）")
    limit = min(max_iter, 2)
    query = question
    for i in range(limit):
        record = {
            "round": i + 1, "query": query, "evidence": [], "answer": None,
            "judge_raw": None, "judgment": "not_measured", "quality": None,
            "rewrite_query": None, "error_stage": None, "error_type": None,
        }
        result["rounds"].append(record)
        stage = "retrieve"
        try:
            hits = pipe.retrieve(query, k=5, mode="hybrid")
            record["evidence"] = _evidence(hits)
            context = [c for c, _ in hits]
            stage = "generate"
            answer = generator.generate_rag(question, context)
            if not isinstance(answer, str):
                raise ValueError("生成结果必须为文本")
            if not answer.strip():
                raise ValueError("生成结果必须为非空白文本")
            record["answer"] = answer
            stage = "quality"
            record["quality"] = _quality(answer, context, gold_answer)
            stage = "judge"
            context_text = "\n\n".join(
                f"[片段{j + 1}]\n{c['text']}" for j, c in enumerate(context))
            raw = generator.complete(
                "判断回答是否充分回答原始问题且得到所给证据支持。"
                "只回答 SUFFICIENT 或 UNSURE，不输出解释。\n"
                f"原始问题：{question}\n证据：\n{context_text}\n回答：\n{answer}",
                max_tokens=16)
            record["judge_raw"] = raw
            verdict = raw.strip()
            record["judgment"] = (
                "sufficient" if verdict == "SUFFICIENT" else
                "insufficient" if verdict == "UNSURE" else "invalid_judgment")
            result["status"] = record["judgment"]
            if verdict == "SUFFICIENT" or i + 1 == limit:
                break
            stage = "rewrite"
            rewritten = generator.complete(
                "只改写用于检索的 query，保留原始问题的实体、约束和意图，"
                "结合已有证据和回答中的缺口补充检索词，只输出 query。\n"
                f"原始问题：{question}\n当前检索 query：{query}\n"
                f"已有证据：\n{context_text}\n当前回答：\n{answer}",
                max_tokens=128).strip()
            record["rewrite_query"] = rewritten
            if not rewritten:
                result["status"] = "rewrite_empty"
                break
            query = rewritten
        except Exception as exc:
            record["error_stage"] = stage
            record["error_type"] = type(exc).__name__
            result["status"] = "error"
            break

    rounds = result["rounds"]
    # attempts 计实际进入检索的轮数；即使检索抛异常，也保留这次调用。
    # 失败轮只保留在 trace；回退到最后一个完整成功的轮次。
    # 首轮任一阶段失败时，没有可返回的成功答案。
    completed = [r for r in rounds if r["error_stage"] is None]
    returned = completed[-1] if completed else None
    result.update({
        "answer": returned["answer"] if returned else None,
        "returned_answer_round": returned["round"] if returned else None,
        "attempts": len(rounds),
        "extra_retries": len(rounds) - 1, "second_retrieval": len(rounds) > 1,
        "initial_quality": rounds[0]["quality"],
        "final_quality": returned["quality"] if returned else None,
    })
    return result


def _support_branch(pipe, q, use_llm=False):
    """oracle gold level 分支；返回同一次检索的证据，避免重复检索。"""
    level = q["level"]
    if level not in ("L1", "L2", "L3", "L4"):
        raise ValueError(f"未知 gold level：{level}")
    k, template = (3, "concise") if level in ("L1", "L2") else (8, "evidence")
    hits = pipe.retrieve(q["question"], k=k, mode="hybrid")
    context = [c for c, _ in hits]
    answer = None
    if use_llm:
        if template == "concise":
            answer = generator.generate_rag(q["question"], context)
        else:
            ctx_text = "\n".join(
                f"[片段{i + 1}]\n{c['text']}" for i, c in enumerate(context))
            answer = generator.complete(
                "资料：\n" + ctx_text + "\n\n问题：" + q["question"]
                + "\n\n请作答，并在每个结论后标注它来自哪个[片段N]；"
                  "资料不支持的内容明确说'资料未提及'。", max_tokens=900)
    return {"answer": answer, "top_k": k, "template": template,
            "level_source": "oracle_gold_level", "retrieved": [
                c["chunk_id"] for c in context]}


def _retry_row(q, result):
    row = {"id": q["id"], "level": q["level"], "question": q["question"],
           **{key: result[key] for key in (
               "status", "attempts", "extra_retries", "second_retrieval", "answer",
               "returned_answer_round")}}
    for phase in ("initial", "final"):
        quality = result[f"{phase}_quality"] or {}
        for metric in QUALITY_FIELDS:
            row[f"{phase}_{metric}"] = quality.get(metric)
    return row


def _retry_summary(results):
    measured = [r for r in results if r["attempts"] is not None]
    n = len(measured)
    second_n = sum(r["second_retrieval"] for r in measured)
    summary = {
        "status": "measured" if n else "not_measured",
        "sample_size": len(results), "measured_questions": n,
        "second_retrieval_count": second_n if n else None,
        "second_retrieval_rate": second_n / n if n else None,
        "mean_attempts": sum(r["attempts"] for r in measured) / n if n else None,
        "mean_extra_retries": sum(r["extra_retries"] for r in measured) / n if n else None,
        "judge_sufficient_rate": (
            sum(r["status"] == "sufficient" for r in measured) / n if n else None),
        "error_count": sum(r["status"] == "error" for r in measured) if n else None,
    }
    for phase in ("initial", "final"):
        for metric in QUALITY_FIELDS:
            values = [(r[f"{phase}_quality"] or {}).get(metric) for r in measured]
            values = [v for v in values if v is not None]
            summary[f"{phase}_{metric}_n"] = len(values)
            summary[f"{phase}_{metric}_mean"] = (
                sum(values) / len(values) if values else None)
    return summary


def run(use_llm=False):
    if use_llm:
        config.require_llm()  # fail fast：不可静默降级成离线实验
    else:
        # 本文件限定离线行为，不修改公共配置或其他实验。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    docs = loader.read_all_docs(config.CORPUS_DIR)
    qas = utils.load_jsonl(config.EVAL_JSONL)
    pipe = pipeline.RAGPipeline(docs)

    a_rows, b_rows = [], []
    for q in qas:
        route = _route_decision(q["question"], use_llm)
        got = _branch_recall(pipe, q["question"], route["mode"])
        a_rows.append({"id": q["id"], "level": q["level"], **route,
                       "recall@5": round(evaluate.recall_at_k(
                           q["gold_chunk_ids"], got, 5), 3)})
        support = _support_branch(pipe, q, use_llm)
        b_rows.append({"id": q["id"], "level": q["level"],
                       "level_source": support["level_source"],
                       "top_k": support["top_k"], "template": support["template"],
                       "hit@k": round(evaluate.hit_at_k(
                           q["gold_chunk_ids"], support["retrieved"], support["top_k"]), 3),
                       "generation_status": "measured" if use_llm else "not_measured",
                       "answer": support["answer"]})
    traces, c_rows = [], []
    for q in qas[:6]:
        result = _retry(pipe, q["question"], use_llm,
                        gold_answer=q.get("gold_answer"))
        traces.append({"id": q["id"], "level": q["level"], **result})
        c_rows.append(_retry_row(q, result))
    summary = _retry_summary(traces)

    outputs = [
        save_csv("t6_routing.csv", a_rows,
                 ["id", "level", "mode", "route_source", "route_raw", "recall@5"]),
        save_csv("t6_support.csv", b_rows,
                 ["id", "level", "level_source", "top_k", "template", "hit@k",
                  "generation_status", "answer"]),
        save_csv("t6_retry.csv", c_rows,
                 ["id", "level", "question", "status", "attempts", "extra_retries",
                  "second_retrieval", "answer", "returned_answer_round"] + [
                      f"{phase}_{metric}" for phase in ("initial", "final")
                      for metric in QUALITY_FIELDS]),
    ]
    trace_path = os.path.join(config.RESULTS_DIR, "t6_retry_trace.jsonl")
    utils.save_jsonl(trace_path, traces)
    summary_path = os.path.join(config.RESULTS_DIR, "t6_retry_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    outputs += [trace_path, summary_path]

    print(CONTROL_FLOW_MARKDOWN)
    print("[t6] 循环汇总（质量均为代理，不是正确率）：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[t6] 输出：\n" + "\n".join(outputs))
    return {"routing": a_rows, "support": b_rows, "retry": c_rows,
            "retry_traces": traces, "retry_summary": summary,
            "control_flow_markdown": CONTROL_FLOW_MARKDOWN, "outputs": outputs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true",
                        help="显式开启 LLM 路由/生成/judge/改写；缺配置立即报错")
    args = parser.parse_args(argv)
    return run(use_llm=args.generate)


if __name__ == "__main__":
    main()
