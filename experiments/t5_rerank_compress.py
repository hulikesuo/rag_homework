"""T5：共享 hybrid top-5，比 direct/规则/CrossEncoder top-3，再成对比较压缩。

运行：python -m experiments.t5_rerank_compress [--generate] [--compress-subset 6]
EM 为去空白后的完全匹配，ROUGE-L 为字符级 F1；均不代表人工 L4 评分。
Tokenizer 使用嵌入模型的真实 tokenizer，不代表生成 API 的计费 token。
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor

from experiments.common import load_env, save_csv
from src import pipeline, rerank, evaluate, config, embed
from src.compression import InformationCompressor
from src.generator import generate_rag


RERANK_FIELDS = [
    "id", "question", "gold_answer", "variant", "status", "retriever", "reranker",
    "rerank_model", "candidate_k_requested", "top_k_candidates", "top_k_selected",
    "candidate_ids", "retrieved", "recall@3", "hit@3", "tokenizer", "tokenizer_scope",
    "context_text", "context_chars", "context_tokens", "prompt_context_text",
    "prompt_context_tokens", "retrieve_ms", "model_load_state", "model_load_ms",
    "warmup_ms", "rerank_ms", "generation_measured", "generation_ms", "answer",
    "em", "rouge_l", "quality_metric_scope", "l4_score",
]
COMPRESS_FIELDS = [
    "id", "question", "gold_answer", "variant", "retriever", "reranker", "rerank_model",
    "candidate_k_requested", "top_k_candidates", "top_k_selected", "candidate_ids",
    "retrieved", "recall@3", "hit@3", "budget_chars", "tokenizer", "tokenizer_scope",
    "compressor", "compressor_fit_scope",
    "orig_context_text", "after_context_text", "orig_prompt_context_text",
    "after_prompt_context_text", "orig_chars", "after_chars", "orig_tokens", "after_tokens",
    "orig_prompt_context_tokens", "after_prompt_context_tokens", "compression_ratio",
    "retrieve_ms", "rerank_ms", "compress_ms", "generation_measured",
    "orig_generation_ms", "after_generation_ms", "orig_answer", "after_answer",
    "orig_em", "after_em", "orig_rouge_l", "after_rouge_l", "quality_metric_scope", "l4_score",
]
TOKENIZER_SCOPE = "embedding_tokenizer;not_generation_billing;no_truncation;no_special_tokens"
QUALITY_SCOPE = "whitespace_normalized_exact_match_and_character_ROUGE_L;not_L4"


def _elapsed_ms(start):
    # 保留测量精度，不将极短操作四舍五入成伪造的 0ms。
    return (time.perf_counter() - start) * 1000


def _context_stats(chunks):
    text = "\n\n".join(c["text"] for c in chunks)
    # 与 generate_rag 实际插入提示的片段标签和分隔符保持一致。
    prompt_context = "\n\n".join(
        f"[片段{i + 1}]\n{c['text']}" for i, c in enumerate(chunks))
    return {
        "context_text": text, "context_chars": len(text),
        "context_tokens": embed.count_tokens(text),
        "prompt_context_text": prompt_context,
        "prompt_context_tokens": embed.count_tokens(prompt_context),
    }


def _generate_metrics(question, chunks, enabled):
    if not enabled:
        return {"answer": "", "generation_ms": "", "em": "", "rouge_l": ""}
    start = time.perf_counter()
    answer = generate_rag(question["question"], chunks)
    elapsed = _elapsed_ms(start)
    gold = re.sub(r"\s+", "", question["gold_answer"] or "")
    normalized = re.sub(r"\s+", "", answer or "")
    return {
        "answer": answer, "generation_ms": elapsed,
        "em": float(bool(gold) and gold == normalized),
        "rouge_l": evaluate.rouge_l_f1(question["gold_answer"], answer),
    }


def _generate_many(tasks, enabled):
    """仅并行生成请求；候选、重排和压缩仍在主线程逐题完成。"""
    if not enabled or not tasks:
        return [_generate_metrics(q, chunks, False) for q, chunks in tasks]
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(
            lambda task: _generate_metrics(task[0], task[1], True), tasks))


def run(compress_subset=6, do_generate=False, compress_budget=500):
    if compress_subset < 0:
        raise ValueError("compress_subset 必须 >= 0")
    if do_generate:
        config.require_llm()
    env = load_env()
    docs, qas = env["docs"], env["qas"]
    pipe = pipeline.RAGPipeline(docs)
    # 只用原始语料文本估计 unigram 概率，不读取题目 gold 或检索标签。
    compressor = InformationCompressor().fit(docs.values())

    # 首次加载在题目重排计时外进行。若已缓存，不声称测到了冷加载。
    model_name = config.RERANK_MODEL_NAME
    model, load_ms, load_state = None, "", "not_configured"
    if _rerank_model_ok():
        if model_name in rerank._RERANK_CACHE:
            model = rerank._get_reranker(model_name)
            load_state = "already_cached"
        else:
            start = time.perf_counter()
            model = rerank._get_reranker(model_name)
            load_ms = _elapsed_ms(start)
            load_state = "cold_loaded"
    else:
        print("[t5] 未配置 CrossEncoder：显式输出跳过行；direct/规则照常运行。")
    if not do_generate:
        print("[t5] 不调用生成 API；答案、EM/ROUGE-L、生成耗时及 L4 评分留空。")

    rows, comp_rows = [], []
    warmed_up, load_reported = False, False
    for question_index, q in enumerate(qas):
        pending = []
        query = q["question"]
        start = time.perf_counter()
        hits = pipe.retrieve(query, k=5, mode="hybrid")
        retrieve_ms = _elapsed_ms(start)
        variants = [("direct", hits[:3], "", "ok", "")]
        start = time.perf_counter()
        rule_hits = rerank.rerank_rules(query, hits, top_k=3)
        variants.append(("rules", rule_hits, _elapsed_ms(start), "ok", ""))
        warmup_ms = ""
        if model is None:
            variants.append(("cross_encoder", [], "", "skipped_not_configured", ""))
        elif not hits:
            variants.append(("cross_encoder", [], "", "no_candidates", ""))
        else:
            if not warmed_up:
                # 用实际候选做一次不计入热推理的预热，真实耗时单独输出。
                start = time.perf_counter()
                model.predict([(query, chunk["text"]) for chunk, _ in hits])
                warmup_ms = _elapsed_ms(start)
                warmed_up = True
            start = time.perf_counter()
            selected = rerank.rerank_cross_encoder(query, hits, top_k=3)
            variants.append(("cross_encoder", selected, _elapsed_ms(start), "ok", warmup_ms))

        for name, selected, rerank_ms, status, variant_warmup_ms in variants:
            chunks = [chunk for chunk, _ in selected]
            ids = [chunk["chunk_id"] for chunk in chunks]
            skipped = status == "skipped_not_configured"
            gold_ids = q.get("gold_chunk_ids", [])
            shared = {
                "id": q["id"], "question": query, "gold_answer": q.get("gold_answer", ""),
                "variant": name, "retriever": "hybrid",
                "reranker": "none" if name == "direct" else name,
                "rerank_model": model_name if name == "cross_encoder" else "",
                "candidate_k_requested": 5, "top_k_candidates": len(hits),
                "top_k_selected": len(chunks),
                "candidate_ids": [chunk["chunk_id"] for chunk, _ in hits], "retrieved": ids,
                "recall@3": "" if skipped else evaluate.recall_at_k(gold_ids, ids, 3),
                "hit@3": "" if skipped else evaluate.hit_at_k(gold_ids, ids, 3),
                "tokenizer": config.EMBED_MODEL_NAME, "tokenizer_scope": TOKENIZER_SCOPE,
                "retrieve_ms": retrieve_ms, "rerank_ms": rerank_ms,
                "generation_measured": do_generate and not skipped,
                "quality_metric_scope": QUALITY_SCOPE, "l4_score": "",
            }
            stats = _context_stats(chunks) if not skipped else {
                key: "" for key in ("context_text", "context_chars", "context_tokens",
                                    "prompt_context_text", "prompt_context_tokens")}
            pending.append({
                "q": q, "chunks": chunks, "shared": shared, "stats": stats,
                "name": name, "status": status, "skipped": skipped,
                "load_state": load_state if name == "cross_encoder" else "",
                "warmup_ms": variant_warmup_ms,
                "question_index": question_index,
            })

        # 同一题的 direct/rules/cross-encoder 一起并发；不同题仍按顺序。
        generation_tasks = [(item["q"], item["chunks"]) for item in pending
                            if not item["skipped"]]
        generated_results = iter(_generate_many(generation_tasks, do_generate))
        for item in pending:
            generated = (dict.fromkeys(("answer", "generation_ms", "em", "rouge_l"), "")
                         if item["skipped"] else next(generated_results))
            q, chunks, stats = item["q"], item["chunks"], item["stats"]
            shared, name = item["shared"], item["name"]
            row = dict(shared, **stats, **generated, status=item["status"],
                       model_load_state=item["load_state"],
                       model_load_ms="", warmup_ms=item["warmup_ms"])
            if name == "cross_encoder" and not load_reported:
                row["model_load_ms"] = load_ms
                load_reported = True
            rows.append(row)

            if item["skipped"] or question_index >= compress_subset:
                continue
            start = time.perf_counter()
            new_text, orig, after, ratio = compressor.compress(
                q["question"], chunks, budget=compress_budget)
            compress_ms = _elapsed_ms(start)
            compressed_chunks = [{"text": new_text}] if new_text else []
            compressed_stats = _context_stats(compressed_chunks)
            compressed_generated = _generate_metrics(q, compressed_chunks, do_generate)
            comp_rows.append(dict(
                shared, budget_chars=compress_budget,
                compressor="unigram_self_information",
                compressor_fit_scope="raw_corpus_text_only;no_gold;no_eval_labels",
                orig_context_text=stats["context_text"], after_context_text=new_text,
                orig_prompt_context_text=stats["prompt_context_text"],
                after_prompt_context_text=compressed_stats["prompt_context_text"],
                orig_chars=orig, after_chars=after,
                orig_tokens=stats["context_tokens"], after_tokens=compressed_stats["context_tokens"],
                orig_prompt_context_tokens=stats["prompt_context_tokens"],
                after_prompt_context_tokens=compressed_stats["prompt_context_tokens"],
                compression_ratio=ratio, compress_ms=compress_ms,
                orig_generation_ms=generated["generation_ms"],
                after_generation_ms=compressed_generated["generation_ms"],
                orig_answer=generated["answer"], after_answer=compressed_generated["answer"],
                orig_em=generated["em"], after_em=compressed_generated["em"],
                orig_rouge_l=generated["rouge_l"], after_rouge_l=compressed_generated["rouge_l"],
            ))
    save_csv("t5_rerank.csv", rows, RERANK_FIELDS)
    save_csv("t5_compress.csv", comp_rows, COMPRESS_FIELDS)
    return rows, comp_rows


def _rerank_model_ok():
    """仅判断是否配置；真实加载失败时抛错，绝不伪装成成功重排。"""
    return bool(config.RERANK_MODEL_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="调用 LLM，成对测压缩前后答案质量")
    parser.add_argument("--compress-subset", type=int, default=6, help="对前 N 题的各有效分支测压缩")
    parser.add_argument("--compress-budget", type=int, default=500, help="压缩文本字符上限（含换行）")
    args = parser.parse_args()
    run(compress_subset=args.compress_subset, do_generate=args.generate,
        compress_budget=args.compress_budget)
