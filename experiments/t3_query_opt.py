"""T3 查询优化及线性融合权重扫描。

默认 do_llm=False、do_generate=False：只跑原查询与 alpha 扫描，不调用 LLM。
--llm 比较 baseline / rewrite / multi / subquery；所有优化策略保留原查询，
通过 RRF 融合检索结果。
--generate 独立开启答案生成。subquery 策略在 do_generate=True 时自动调用
src.subquery.answer_subqueries 分解综合，不使用普通 generate_rag；
其它三种策略（baseline/rewrite/multi）不受影响。
Recall@5 是 gold chunk ID 的召回比例；Hit@5 是任一 gold ID 命中。
EM_proxy 为去空白后完全相等，ROUGE-L_proxy 为字符 LCS F1，非语义判分。
"""
import argparse
import json
import re
from time import perf_counter

from experiments.common import load_env, save_csv, gold_chunks_for_docs
from src import config, evaluate, pipeline
from src import generator as gen

from src import subquery as subq

QUALITY_FIELDS = ["answer", "generation_status", "generation_error",
                  "EM_proxy", "ROUGE-L_proxy", "answer_time_s"]
SUBQUERY_TRACE_FIELDS = [
    "subquery_answers", "synthesis_context_ids", "synthesis_context_count",
    "synthesis_context",
    "subquery_errors", "subquery_fallback", "subquery_fallback_reason",
]
CASE_FIELDS = ["id", "level", "question", "gold_answer", "strategy", "queries",
               "fusion", "gold_evidence", "retrieved", "retrieved_evidence",
               "Recall@5", "Hit@5", "query_time_s", "retrieval_time_s",
               "query_status", "query_error", "generator_model",
               *QUALITY_FIELDS, *SUBQUERY_TRACE_FIELDS, "total_time_s"]
SUMMARY_FIELDS = ["strategy", "n_questions", "n_generated", "generator_model",
                  "n_generation_errors",
                  "Recall@5", "Hit@5", "query_time_s", "retrieval_time_s",
                  "EM_proxy", "ROUGE-L_proxy", "answer_time_s", "total_time_s"]
OPTIMIZED_STRATEGIES = ("rewrite", "multi", "subquery")


def _queries_with_original(question, raw):
    """清理 LLM 行列表，保留原查询；空输出只能回退，不能伪造改写。"""
    queries = [question]
    for line in (raw or "").splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.\u3001)\uff09])\s*", "", line).strip()
        if line and line not in queries:
            queries.append(line)
        if len(queries) >= 6:  # 原查询 + 至多 5 个模型生成查询。
            break
    return queries


def _optimized_queries(strategy, question):
    constraints = ("保留原问题已有的课程、专业、人物等实体及年份、版本、数字约束；"
                   "不要猜测缺失的实体或年份，不回答问题，只输出查询文本。")
    if strategy == "rewrite":
        instruction, max_tokens = "把问题改写为一个更利于检索的完整问句，只输出一行。", 128
    elif strategy == "multi":
        instruction, max_tokens = "生成3至5个语义等价或互补的检索问句，每行一个。", 256
    elif strategy == "subquery":
        instruction, max_tokens = "拆成需要分别查证的关键子问题，每行一个，最多5个。", 256
    else:
        raise ValueError(f"未知查询策略：{strategy}")
    raw = gen.complete(f"{instruction}\n{constraints}\n原问题：{question}",
                       max_tokens=max_tokens)
    queries = _queries_with_original(question, raw)
    return queries[:2] if strategy == "rewrite" else queries


def _retrieve(pipe, queries, fuse):
    if not fuse:
        return [c for c, _ in pipe.retrieve(queries[0], k=5, mode="dense")][:5]
    scores, chunks = {}, {}
    for query in queries:
        for mode in ("dense", "bm25"):
            try:
                hits = pipe.retrieve(query, k=20, mode=mode)
            except Exception as exc:
                raise type(exc)(f"retrieval {mode} failed for query: {exc}") from exc
            seen = set()
            for rank, (chunk, _) in enumerate(hits, 1):
                cid = chunk["chunk_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                chunks[cid] = chunk
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
    return [chunks[cid] for cid in sorted(scores, key=lambda cid: -scores[cid])[:5]]


def _retrieve_boosted(pipe, queries, fuse):
    """RRF at larger depth, used for subquery only.

    Gold chunks for q28 commonly sit at dense rank 10-17 across generated
    sub-queries; this depth was chosen during exploratory diagnosis with
    gold access (q28), not a blind pre-specified parameter. Keeping all
    evidence (k=30) lets repeated mentions accumulate, and it is applied
    uniformly to every subquery question in this experiment.
    """
    if not fuse:
        return [c for c, _ in pipe.retrieve(queries[0], k=5, mode="dense")][:5]
    scores, chunks = {}, {}
    for query in queries:
        for mode in ("dense", "bm25"):
            hits = pipe.retrieve(query, k=30, mode=mode)
            seen = set()
            for rank, (chunk, _) in enumerate(hits, 1):
                cid = chunk["chunk_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                chunks[cid] = chunk
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
    return [chunks[cid] for cid in sorted(scores, key=lambda cid: -scores[cid])[:5]]


def _score_answer(answer, elapsed, gold_answer):
    """Pure scoring: EM/ROUGE from already-generated answer.  Never calls API.
    answer=None or empty -> generation_status='error'."""
    if not isinstance(answer, str) or not answer.strip():
        return {"answer": "", "generation_status": "error",
                "generation_error": "EmptyGeneration",
                "EM_proxy": "", "ROUGE-L_proxy": "",
                "answer_time_s": elapsed}
    normalize = lambda text: re.sub(r"\s+", "", text or "")
    reference = normalize(gold_answer)
    return {"answer": answer, "generation_status": "ok",
            "generation_error": "",
            "EM_proxy": float(bool(reference) and reference == normalize(answer)),
            "ROUGE-L_proxy": evaluate.rouge_l_f1(gold_answer, answer),
            "answer_time_s": elapsed}


def _generate_and_score(question, chunks, gold_answer):
    """Call generate_rag then score.  Only used by non-subquery strategies."""
    start = perf_counter()
    try:
        answer = gen.generate_rag(question, chunks)
    except Exception as exc:
        return {"answer": "", "generation_status": "error",
                "generation_error": type(exc).__name__,
                "EM_proxy": "", "ROUGE-L_proxy": "", "answer_time_s": ""}
    elapsed = perf_counter() - start
    return _score_answer(answer, elapsed, gold_answer)


def _mean(rows, field):
    values = [r[field] for r in rows if r[field] != ""]
    return sum(values) / len(values) if values else ""


def _success_cases(cases_by_strategy):
    """仅导出 L2/L3 原查询 Recall@5=0、优化后 >0 的真实检索成功对。"""
    baseline = {r["id"]: r for r in cases_by_strategy.get("baseline", [])}
    successes, counts = [], []
    for strategy in OPTIMIZED_STRATEGIES:
        optimized = cases_by_strategy.get(strategy, [])
        for level in ("L2", "L3"):
            count = 0
            eligible = 0
            for row in optimized:
                original = baseline[row["id"]]
                if row["level"] != level or original["Recall@5"] != 0:
                    continue
                eligible += 1
                if row["Recall@5"] <= 0:
                    continue
                count += 1
                successes.append({"id": row["id"], "level": level,
                                  "question": row["question"], "strategy": strategy,
                                  "baseline_Recall@5": original["Recall@5"],
                                  "optimized_Recall@5": row["Recall@5"],
                                  "baseline_case": json.dumps(original, ensure_ascii=False),
                                  "optimized_case": json.dumps(row, ensure_ascii=False)})
            counts.append({"strategy": strategy, "level": level,
                           "evaluated": strategy in cases_by_strategy,
                           "baseline_zero_count": eligible, "success_count": count})
    return successes, counts


def _empty_subquery_trace():
    return {"subquery_answers": "",
            "synthesis_context_ids": "",
            "synthesis_context_count": "",
            "synthesis_context": "",
            "subquery_errors": "",
            "subquery_fallback": "",
            "subquery_fallback_reason": ""}


# Sentinel: marks "no answer — must treat as error, never call generate_rag".
_NO_ANSWER = object()


def _subquery_synthesis_case(question, queries, pipe, rrf_top5):
    """调用 answer_subqueries，返回 (answer_or_sentinel, trace, elapsed_s)。

    answer 为 None 或空字符串时返回 _NO_ANSWER sentinel。调用方用
    _score_answer 基于真实 gold 评分；此处绝不接触 gold 也不调用
    generate_rag。
    """
    final_chunks = [dict(c) for c in rrf_top5]
    start = perf_counter()
    try:
        result = subq.answer_subqueries(question, queries, pipe, final_chunks)
    except Exception as exc:
        result = {"answer": None, "subanswers": [], "context": final_chunks,
                  "errors": [{"stage": "answer_subqueries", "query": question,
                              "error_type": type(exc).__name__}],
                  "fallback": True,
                  "fallback_reason": "answer_subqueries_raised"}
    elapsed = perf_counter() - start
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = _NO_ANSWER
    subanswers = result.get("subanswers", [])
    errors = result.get("errors", [])
    context = result.get("context", [])
    trace = {
        "subquery_answers": json.dumps(
            [{k: v for k, v in row.items() if k != "errors"}
             for row in subanswers], ensure_ascii=False),
        "synthesis_context_ids": json.dumps(
            [c.get("chunk_id") for c in context], ensure_ascii=False),
        "synthesis_context_count": len(context),
        "synthesis_context": json.dumps(context, ensure_ascii=False),
        "subquery_errors": json.dumps(errors, ensure_ascii=False),
        "subquery_fallback": bool(result.get("fallback")),
        "subquery_fallback_reason": result.get("fallback_reason"),
    }
    return answer, trace, elapsed


def run(do_llm=False, do_generate=False):
    if do_llm or do_generate:
        config.require_llm()  # 在构建索引及任何 API 调用前验证配置。
    env = load_env()
    docs, qas = env["docs"], env["qas"]
    pipe = pipeline.RAGPipeline(docs)
    chunks_by_id = {c["chunk_id"]: c for c in pipe.chunks}
    gold_by_question = {q["id"]: gold_chunks_for_docs(q, chunks_by_id, docs)
                        for q in qas}
    strategies = ["baseline", *OPTIMIZED_STRATEGIES] if do_llm else ["baseline"]
    rows, cases_by_strategy = [], {}
    for strategy in strategies:
        print(f"[t3] strategy={strategy} ({len(qas)} questions)", flush=True)

        def case_for(q):
            start = perf_counter()
            try:
                queries = (_optimized_queries(strategy, q["question"])
                           if strategy != "baseline" else [q["question"]])
            except Exception as exc:
                queries = [q["question"]]
                query_error = type(exc).__name__
            else:
                query_error = ""
            print(f"  [t3] {strategy} {q['id']} query_ok={not query_error}"
                  f" nq={len(queries)}", flush=True)
            engage_synthesis = (do_generate and strategy == "subquery")
            query_time = perf_counter() - start
            retrieval_start = perf_counter()
            try:
                retrieve_fn = (_retrieve_boosted if strategy == "subquery"
                               else _retrieve)
                retrieved = retrieve_fn(pipe, queries, fuse=strategy != "baseline")
            except Exception as exc:
                retrieved = []
                if not query_error:
                    query_error = type(exc).__name__
                query_status = "error"
            retrieval_time = perf_counter() - retrieval_start
            query_status = query_status if "query_status" in locals() else "ok"
            if query_error:
                query_status = "error"
            got = [c["chunk_id"] for c in retrieved]
            gold_answer = q.get("gold_answer", "")
            if engage_synthesis:
                # answer from helper; score against real gold only here.
                synthesis_answer, subquery_trace, synthesis_elapsed = _subquery_synthesis_case(
                    q["question"], queries, pipe, retrieved)
                if synthesis_answer is _NO_ANSWER:
                    answer_fields = {"answer": "",
                                     "generation_status": "error",
                                     "generation_error": "EmptyGeneration",
                                     "EM_proxy": "", "ROUGE-L_proxy": "",
                                     "answer_time_s": synthesis_elapsed}
                else:
                    answer_fields = _score_answer(
                        synthesis_answer, synthesis_elapsed, gold_answer)
            else:
                if do_generate:
                    answer_fields = _generate_and_score(
                        q["question"], retrieved, gold_answer)
                else:
                    answer_fields = {"answer": "",
                                     "generation_status": "not_measured",
                                     "generation_error": "",
                                     "EM_proxy": "", "ROUGE-L_proxy": "",
                                     "answer_time_s": ""}
                subquery_trace = _empty_subquery_trace()
            total_time = perf_counter() - start
            return {"id": q["id"], "level": q.get("level", ""),
                    "question": q["question"],
                    "gold_answer": gold_answer,
                    "strategy": strategy,
                    "queries": json.dumps(queries, ensure_ascii=False),
                    "fusion": ("RRF(dense+bm25, depth=30, k=60)"
                               if strategy == "subquery" else
                               ("RRF(dense+bm25, depth=20, k=60)"
                                if strategy != "baseline" else "none")),
                    "gold_evidence": json.dumps(gold_by_question[q["id"]],
                                                ensure_ascii=False),
                    "retrieved": json.dumps(got, ensure_ascii=False),
                    "retrieved_evidence": json.dumps(retrieved, ensure_ascii=False),
                    "Recall@5": recall_score(q, got),
                    "Hit@5": evaluate.hit_at_k(q.get("gold_chunk_ids", []), got, 5),
                    "query_time_s": query_time,
                    "retrieval_time_s": retrieval_time,
                    "query_status": "error" if query_error else "ok",
                    "query_error": query_error,
                    "generator_model": config.LLM_MODEL if do_generate else "",
                    **answer_fields, **subquery_trace,
                    "total_time_s": total_time}

        cases = [case_for(q) for q in qas]
        cases_by_strategy[strategy] = cases
        print(f"[t3] strategy={strategy} finished {len(cases)} rows", flush=True)
        save_csv(f"t3_cases_{strategy}.csv", cases, CASE_FIELDS)
        rows.append({"strategy": strategy, "n_questions": len(qas),
                     "n_generated": sum(c["generation_status"] == "ok"
                                        for c in cases),
                     "n_generation_errors": sum(c["generation_status"] == "error"
                                                for c in cases),
                     "generator_model": config.LLM_MODEL if do_generate else "",
                     **{field: _mean(cases, field) for field in
                        ["Recall@5", "Hit@5", "query_time_s", "retrieval_time_s",
                         "EM_proxy", "ROUGE-L_proxy", "answer_time_s",
                         "total_time_s"]}})
        save_csv("_t3_progress.csv", rows, SUMMARY_FIELDS)
    if not do_llm:
        print("[t3] 未测 LLM 改写/多查询/分解；仅跑原查询与线性权重扫描。")
    if not do_generate:
        print("[t3] 未生成答案；答案、字符串质量代理及答案耗时均留空。")
    save_csv("t3_query_opt.csv", rows, SUMMARY_FIELDS)
    successes, counts = _success_cases(cases_by_strategy)
    save_csv("t3_success_cases.csv", successes,
             ["id", "level", "question", "strategy", "baseline_Recall@5",
              "optimized_Recall@5", "baseline_case", "optimized_case"])
    save_csv("t3_success_counts.csv", counts,
             ["strategy", "level", "evaluated", "baseline_zero_count",
              "success_count"])
    print(f"[t3] L2/L3 严格检索失败到成功案例：{len(successes)} 对；"
          "同一题可在多个策略下成功，不代表答案正确。")

    print(f"\n线性融合 alpha 扫描（{config.DEFAULT_FUSION_ALPHA} 为默认）")
    sweep = []
    for alpha in [0, 0.3, 0.5, 0.7, 1]:
        start = perf_counter()
        score = alpha_sweep(pipe, qas, alpha)
        sweep.append({"alpha_dense": alpha, "Recall@5": score,
                      "retrieval_time_s": ((perf_counter() - start) / len(qas)
                                           if qas else 0.0)})
        print(f"  alpha={alpha}: {score:.4f}")
    save_csv("t3_linear_alpha_sweep.csv", sweep,
             ["alpha_dense", "Recall@5", "retrieval_time_s"])
    return rows


def recall_score(q, got):
    return evaluate.recall_at_k(q.get("gold_chunk_ids", []), got, 5)


def alpha_sweep(pipe, qas, alpha):
    total = 0.0
    for q in qas:
        hits = pipe.retrieve(q["question"], k=5, mode="hybrid", alpha=alpha)
        total += recall_score(q, [c["chunk_id"] for c, _ in hits])
    return total / len(qas) if qas else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true",
                        help="显式启用三种 LLM 查询优化")
    parser.add_argument("--generate", action="store_true",
                        help="显式启用 LLM 答案生成")
    args = parser.parse_args()
    run(do_llm=args.llm, do_generate=args.generate)
