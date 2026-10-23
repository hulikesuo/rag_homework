"""T2 分块实验：跨配置用同文档原文跨度对齐，绝不比较 chunk ID。

Recall@5_evidence = top-5 覆盖的 gold 字符并集长度 / gold 字符并集长度。
Hit@5_evidence = 任一 gold 跨度与 top-5 同文档跨度有非零交集时为 1。
默认只评检索；--generate 显式开启同一生成器的答案对比。
EM_proxy 是去空白后完全相等，ROUGE-L_proxy 是字符 LCS F1；均非语义正确性。
"""
import argparse
import itertools
import json
import re
from time import perf_counter

from experiments.common import (load_env, save_csv, evidence_recall,
                                gold_chunks_for_docs)
from src import chunker, config, embed, evaluate, index_store, retriever
from src import generator as gen


QUALITY_FIELDS = ["answer", "generation_status", "generation_error",
                  "EM_proxy", "ROUGE-L_proxy", "answer_time_s"]
CASE_FIELDS = ["id", "level", "question", "gold_answer", "method", "size",
               "overlap", "retriever", "generator_model", "gold_evidence",
               "retrieved", "retrieved_evidence", "Recall@5_evidence",
               "Hit@5_evidence", "retrieval_time_s", *QUALITY_FIELDS]
SUMMARY_FIELDS = ["method", "size", "overlap", "n_chunks", "retriever",
                  "generator_model", "n_questions", "n_generated",
                  "n_generation_errors",
                  "Recall@5_evidence", "Hit@5_evidence", "retrieval_time_s",
                  "EM_proxy", "ROUGE-L_proxy", "answer_time_s"]


def _grid():
    methods = ["fixed", "recursive", "sentence"]
    sizes = [256, 512, 1024]
    overlaps = [0, 128]
    return [(m, s, ov) for m, s, ov in itertools.product(methods, sizes, overlaps)
            if m != "sentence" or ov == 0]


def hit_evidence(gold, retrieved, k=5):
    """只计正长度交集；相邻端点、不同文档均不算命中。"""
    return float(any(g["doc"] == c["doc"]
                     and min(g["end"], c["end"]) > max(g["start"], c["start"])
                     for g in gold for c in retrieved[:k]))


def _answer_fields(question, chunks, gold_answer, do_generate):
    if not do_generate:
        return {"answer": "", "generation_status": "not_measured",
                "generation_error": "", "EM_proxy": "",
                "ROUGE-L_proxy": "", "answer_time_s": ""}
    start = perf_counter()
    try:
        answer = gen.generate_rag(question, chunks)
    except Exception as exc:
        return {"answer": "", "generation_status": "error",
                "generation_error": type(exc).__name__,
                "EM_proxy": "", "ROUGE-L_proxy": "", "answer_time_s": ""}
    elapsed = perf_counter() - start
    if not isinstance(answer, str):
        return {"answer": "", "generation_status": "error",
                "generation_error": "NonTextGeneration",
                "EM_proxy": "", "ROUGE-L_proxy": "", "answer_time_s": elapsed}
    normalize = lambda text: re.sub(r"\s+", "", text or "")
    reference = normalize(gold_answer)
    return {"answer": answer, "generation_status": "ok", "generation_error": "",
            "EM_proxy": float(bool(reference) and reference == normalize(answer)),
            "ROUGE-L_proxy": evaluate.rouge_l_f1(gold_answer, answer),
            "answer_time_s": elapsed}


def _mean(rows, field):
    values = [r[field] for r in rows if r[field] != ""]
    return sum(values) / len(values) if values else ""


def _retrieval_differences(cases):
    """每题导出跨度覆盖率最低/最高的实际配置；不推断生成答案对错。"""
    grouped = {}
    for case in cases:
        grouped.setdefault(case["id"], []).append(case)
    differences = []
    for qid, records in grouped.items():
        low = min(records, key=lambda r: r["Recall@5_evidence"])
        high = max(records, key=lambda r: r["Recall@5_evidence"])
        if low["Recall@5_evidence"] == high["Recall@5_evidence"]:
            continue
        differences.append({
            "id": qid, "level": low["level"], "question": low["question"],
            "difference_type": ("no_evidence_vs_hit" if low["Recall@5_evidence"] == 0
                                else "partial_coverage_difference"),
            "lower_coverage_case": json.dumps(low, ensure_ascii=False),
            "higher_coverage_case": json.dumps(high, ensure_ascii=False),
        })
    return differences


def run(max_configs=None, do_generate=False):
    if do_generate:
        config.require_llm()  # 在构建索引及任何 API 调用前验证配置。
    env = load_env()
    docs, qas = env["docs"], env["qas"]
    if max_configs is not None and max_configs <= 0:
        raise ValueError("max_configs 必须为正整数或 None")
    cfgs = _grid()[:max_configs] if max_configs is not None else _grid()
    # gold ID 只用于定位默认分块的原文跨度，之后统一按跨度评估。
    baseline_chunks = chunker.chunk_documents(docs)
    chunks_by_id = {c["chunk_id"]: c for c in baseline_chunks}
    gold_by_question = {q["id"]: gold_chunks_for_docs(q, chunks_by_id, docs)
                        for q in qas}
    rows, cases = [], []
    for method, size, overlap in cfgs:
        print(f"分块 {method} size={size} overlap={overlap}")
        chunks = chunker.chunk_documents(docs, method=method, size=size, overlap=overlap)
        idx = index_store.IndexStore(chunks)
        idx.build_dense(embed.encode([c["text"] for c in chunks]))
        idx.build_bm25(retriever.default_tokenize)
        search = retriever.Retriever(idx)
        settings = {"method": method, "size": size, "overlap": overlap,
                    "retriever": "hybrid",
                    "generator_model": config.LLM_MODEL if do_generate else ""}
        def case_for(q):
            start = perf_counter()
            hits = search.retrieve(q["question"], 5, mode="hybrid")
            elapsed = perf_counter() - start
            retrieved = [c for c, _ in hits][:5]
            gold = gold_by_question[q["id"]]
            return {**settings, "id": q["id"], "level": q.get("level", ""),
                    "question": q["question"], "gold_answer": q.get("gold_answer", ""),
                    "gold_evidence": json.dumps(gold, ensure_ascii=False),
                    "retrieved": json.dumps([c["chunk_id"] for c in retrieved],
                                            ensure_ascii=False),
                    "retrieved_evidence": json.dumps(retrieved, ensure_ascii=False),
                    "Recall@5_evidence": evidence_recall(gold, retrieved, 5),
                    "Hit@5_evidence": hit_evidence(gold, retrieved),
                    "retrieval_time_s": elapsed,
                    **_answer_fields(q["question"], retrieved, q.get("gold_answer", ""),
                                     do_generate)}

        # 低并发顺序运行；单题生成失败不会中止整组实验。
        config_cases = [case_for(q) for q in qas]
        cases.extend(config_cases)
        rows.append({**settings, "n_chunks": len(chunks), "n_questions": len(qas),
                     "n_generated": sum(c["generation_status"] == "ok"
                                        for c in config_cases),
                     "n_generation_errors": sum(c["generation_status"] == "error"
                                                for c in config_cases),
                     **{field: _mean(config_cases, field) for field in
                        ["Recall@5_evidence", "Hit@5_evidence", "retrieval_time_s",
                         "EM_proxy", "ROUGE-L_proxy", "answer_time_s"]}})
        # 每组配置即时落盘；进程中断时已完成配置不会丢失。
        save_csv("_t2_progress.csv", rows, SUMMARY_FIELDS)
        save_csv("_t2_cases_progress.csv", cases, CASE_FIELDS)
    save_csv("t2_chunking.csv", rows, SUMMARY_FIELDS)
    save_csv("t2_chunking_cases.csv", cases, CASE_FIELDS)
    differences = _retrieval_differences(cases)
    save_csv("t2_retrieval_differences.csv", differences,
             ["id", "level", "question", "difference_type", "lower_coverage_case",
              "higher_coverage_case"])
    print(f"[t2] 跨配置检索覆盖差异题数：{len(differences)}（不代表生成答案对错）。")
    scored = [r for r in rows if r["Recall@5_evidence"] != ""]
    if scored:
        print("\n最佳分块（hybrid，原文证据跨度召回）：",
              max(scored, key=lambda r: r["Recall@5_evidence"]))
    if not do_generate:
        print("[t2] 未生成答案；答案、字符串质量代理及答案耗时均留空。")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="显式启用 LLM 答案生成")
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args()
    run(max_configs=args.max_configs, do_generate=args.generate)
