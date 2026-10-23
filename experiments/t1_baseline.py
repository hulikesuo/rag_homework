"""T1 基础 RAG + 裸 LLM 对比，记录实验签名与实际生成上下文。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import chains, config, generator, pipeline, utils, evaluate
from experiments.common import (
    BASELINE_PIPELINE, BASELINE_RETRIEVAL, EmptyGenerationError, experiment_signature,
    load_env, save_csv,
)


def run(with_direct=True):
    env = load_env()
    docs, qas = env["docs"], env["qas"]
    fingerprint = experiment_signature(docs, qas)
    pipe, setup_error = None, ""
    if qas:
        try:
            pipe = pipeline.RAGPipeline(docs, **BASELINE_PIPELINE)
        except Exception as exc:
            setup_error = type(exc).__name__
    chain = None
    if pipe is not None:
        try:
            chain = chains.build_baseline_chain(pipe, BASELINE_RETRIEVAL)
        except Exception as exc:
            setup_error = type(exc).__name__

    rows = []
    for index, q in enumerate(qas, 1):
        print(f"[t1] {index}/{len(qas)} {q['id']}", flush=True)
        if chain is not None:
            res = chain.invoke(q["question"])
        else:
            res = {
                "answer": "", "retrieved": [], "context_chunks": [],
                "context_texts": [], "context_recorded": False,
                "rag_status": "error", "rag_error": setup_error,
            }
        ok = res["rag_status"] == "ok"
        row = {
            "id": q["id"], "level": q["level"], "question": q["question"],
            "gold_answer": q["gold_answer"], "gold_chunk_ids": q["gold_chunk_ids"],
            "eval_fingerprint": fingerprint, **res,
            "rag_framework": "langchain_core.RunnableLambda",
            "rag_answered": ok and evaluate.is_answered(res["answer"]),
            "rag_em": evaluate.correctness_em(q["gold_answer"], res["answer"]) if ok else 0.0,
            "rag_rouge_l": evaluate.rouge_l_f1(q["gold_answer"], res["answer"]) if ok else 0.0,
            "direct_status": "skipped", "direct_error": "",
        }
        if with_direct:
            row["direct_answer"] = ""
            try:
                answer = generator.generate_direct(q["question"])
                if not isinstance(answer, str):
                    raise TypeError("生成器未返回文本")
                if not answer.strip():
                    raise EmptyGenerationError("生成器返回空白文本")
                row.update(direct_answer=answer, direct_status="ok")
            except Exception as exc:
                row.update(direct_status="error", direct_error=type(exc).__name__)
        rows.append(row)
    utils.save_jsonl(os.path.join(config.RESULTS_DIR, "t1_baseline.jsonl"), rows)

    fields = ["id", "level", "question", "gold_answer", "gold_chunk_ids",
              "eval_fingerprint", "answer", "retrieved", "rag_framework",
              "rag_answered", "rag_em", "rag_rouge_l",
              "context_chunks", "context_texts", "context_recorded", "rag_status", "rag_error",
              "direct_status", "direct_error"]
    if with_direct:
        fields.append("direct_answer")
    save_csv("t1_baseline_summary.csv", rows, fields)
    if with_direct:
        _compare_direct(rows)
    print_cases(rows)
    errors = sum(r["rag_status"] == "error" for r in rows)
    print(f"\n完成 {len(rows)} 题，RAG 异常 {errors} 题。EM/ROUGE 为代理，仍需人工核验。")
    return rows


def _compare_direct(rows):
    """按全题数统计代理指标；异常计零，并单列调用错误率。"""
    rag_em = direct_em = rag_rl = direct_rl = 0.0
    rag_answered = direct_answered = rag_errors = direct_errors = 0
    for r in rows:
        gold = r["gold_answer"]
        if r.get("rag_status") == "ok":
            rag_em += evaluate.correctness_em(gold, r["answer"])
            rag_rl += evaluate.rouge_l_f1(gold, r["answer"])
            rag_answered += evaluate.is_answered(r["answer"])
        else:
            rag_errors += r.get("rag_status") == "error"
        if r.get("direct_status") == "ok":
            da = r.get("direct_answer", "")
            direct_em += evaluate.correctness_em(gold, da)
            direct_rl += evaluate.rouge_l_f1(gold, da)
            direct_answered += evaluate.is_answered(da)
        else:
            direct_errors += r.get("direct_status") == "error"
    n = len(rows)
    summary = []
    for metric, rag, direct in (
        ("EM(精确包含)", rag_em, direct_em),
        ("ROUGE-L", rag_rl, direct_rl),
        ("作答率(非拒答代理)", rag_answered, direct_answered),
        ("调用错误率", rag_errors, direct_errors),
    ):
        summary.append({
            "metric": metric, "RAG": round(rag / n, 4) if n else 0.0,
            "无RAG": round(direct / n, 4) if n else 0.0,
            "差值": round((rag - direct) / n, 4) if n else 0.0,
        })
    save_csv("t1_rag_vs_direct.csv", summary, ["metric", "RAG", "无RAG", "差值"],
             rows_key="__none__")
    print("\n— RAG vs 无 RAG —")
    for s in summary:
        print(f"  {s['metric']:14s} RAG={s['RAG']:.3f}  无RAG={s['无RAG']:.3f}  差={s['差值']:+.3f}")


def print_cases(rows):
    successes = [r for r in rows if r.get("rag_status") == "ok" and r.get("rag_em") == 1.0]
    unresolved = [r for r in rows if r not in successes]
    print("— EM代理成功案例（需人工确认语义正确） —")
    for r in successes[:2]:
        print(f"Q: {r['question']}\nA: {r['answer'][:120]}\n")
    print("— 失败/待人工核验案例 —")
    for r in unresolved[:2]:
        print(f"Q: {r['question']}\n状态: {r['rag_status']} ({r['rag_error']})\n"
              f"检索: {r['retrieved']}\n期望: {r['gold_chunk_ids']}\nA: {r['answer'][:120]}\n")


if __name__ == "__main__":
    run()
