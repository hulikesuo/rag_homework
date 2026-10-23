"""实验共同工具：加载语料/测试集、跑过往指标、存 CSV。

各 T 文件复用它，减少重复。
"""
import hashlib
import json
import os
import sys
import time

# T1/T7 共用显式配置，避免依赖流水线默认参数漂移。
BASELINE_PIPELINE = {
    "chunk_method": "fixed", "chunk_size": 512, "chunk_overlap": 128,
    "retriever_mode": "dense",
}
BASELINE_RETRIEVAL = {"k": 5, "mode": "hybrid"}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, loader, utils, evaluate  # noqa: E402


def load_env():
    return {
        "docs": loader.read_all_docs(config.CORPUS_DIR),
        "qas": utils.load_jsonl(config.EVAL_JSONL),
    }


def save_csv(path, rows, fieldnames, rows_key="metric", overwrite=True):
    """存 CSV；人工工作表用独占创建，已有文件一律保留。"""
    import csv
    full = os.path.join(config.RESULTS_DIR, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    stem, ext = os.path.splitext(full)
    suffix = 0
    while True:
        target = full if suffix == 0 else f"{stem}_{suffix}{ext}"
        try:
            handle = open(target, "w" if overwrite else "x", newline="",
                          encoding="utf-8-sig")
            break
        except FileExistsError:
            suffix += 1
    with handle as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if rows_key in r:
                r = r[rows_key]
            w.writerow(r)
    print(f"[save] {target}")
    return target


def _sha256(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def experiment_signature(docs, qas):
    """T1/T7 同一实验签名；只返回摘要，不读取 key、不保存原始端点。

    覆盖完整题集（包括 rubric 等扩展字段）、有序语料、实际基线参数、
    关键实现及非敏感模型配置。代码按原始字节摘要，代码变动保守失效。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = {}
    for name in ("loader", "chunker", "embed", "index_store", "retriever",
                 "rerank", "pipeline", "generator", "utils", "evaluate"):
        path = os.path.join(root, "src", name + ".py")
        with open(path, "rb") as handle:
            sources["src/" + name] = hashlib.sha256(handle.read()).hexdigest()
    # common 包含实际基线生成入口；T1/T7 改动也使旧实验缓存失效。
    for name in ("common", "t1_baseline", "t7_eval"):
        with open(os.path.join(root, "experiments", name + ".py"), "rb") as handle:
            sources["experiments/" + name] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(root, "src", "chains.py"), "rb") as handle:
        sources["src/chains"] = hashlib.sha256(handle.read()).hexdigest()
    models = {name: getattr(config, name, "") for name in (
        "EMBED_MODEL_NAME", "LLM_MODEL", "RERANK_MODEL_NAME",
        "DEFAULT_FUSION_ALPHA",
    )}
    # 同名模型切换部署也应失效；端点只参与内存中的 SHA256。
    models["endpoint_sha256"] = _sha256(getattr(config, "LLM_BASE_URL", ""))
    return _sha256({
        "schema": 1, "qas": qas, "docs": list(docs.items()),
        "pipeline": BASELINE_PIPELINE, "retrieval": BASELINE_RETRIEVAL,
        "models": models, "sources": sources,
    })


class EmptyGenerationError(ValueError):
    """生成器返回空白文本；与非文本输出或调用异常区分。"""


def baseline_answer(pipe, question):
    """记录传给 generate_rag 的实际 ID/文本，而非事后重建候选上下文。

    固定基线不重排/压缩，T1 与 T7 --generate 共用本入口。
    异常只保存类型，避免第三方错误消息泄露凭证或端点。
    """
    from src import generator
    result = {"answer": "", "retrieved": [], "context_chunks": [],
              "context_texts": [], "context_recorded": False,
              "rag_status": "error", "rag_error": ""}
    try:
        hits = pipe.retrieve(question, **BASELINE_RETRIEVAL)
        context = [dict(chunk) for chunk, _ in hits]
        result.update(
            retrieved=[chunk["chunk_id"] for chunk, _ in hits],
            context_chunks=[chunk["chunk_id"] for chunk in context],
            context_texts=[chunk["text"] for chunk in context],
            context_recorded=True,
        )
        answer = generator.generate_rag(question, context)
        if not isinstance(answer, str):
            raise TypeError("生成器未返回文本")
        if not answer.strip():
            raise EmptyGenerationError("生成器返回空白文本")
        result.update(answer=answer, rag_status="ok")
    except Exception as exc:
        result["rag_error"] = type(exc).__name__
    return result


def evidence_recall(gold_chunks, retrieved_chunks, k=None):
    """在同文档原文字符跨度上匹配 gold，避免不同分块的 ID 不可比。"""
    if k is not None:
        retrieved_chunks = retrieved_chunks[:k]
    def merge(intervals):
        merged = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return merged

    gold_by_doc = {}
    for gold in gold_chunks:
        gold_by_doc.setdefault(gold["doc"], []).append(
            (gold["start"], gold["end"]))
    relevant = {doc: merge(intervals) for doc, intervals in gold_by_doc.items()}
    total = sum(end - start for spans in relevant.values() for start, end in spans)
    if not total:
        return 0.0
    got_by_doc = {}
    for chunk in retrieved_chunks:
        if chunk.get("doc") in relevant:
            got_by_doc.setdefault(chunk["doc"], []).append(
                (chunk.get("start", 0), chunk.get("end", 0)))
    covered = 0
    for doc, spans in relevant.items():
        got = merge(got_by_doc.get(doc, []))
        for start, end in spans:
            covered += sum(max(0, min(end, right) - max(start, left))
                           for left, right in got if right > start and left < end)
    return covered / total


def gold_chunks_for_docs(q, chunks_by_id, docs):
    """Resolve gold IDs to reference text and source document offsets."""
    out = []
    for cid in q.get("gold_chunk_ids", []):
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            raise ValueError(f"{q['id']}: missing gold chunk {cid}")
        doc_text = docs[chunk["doc"]]
        start = chunk.get("start")
        if start is not None:
            end = chunk.get("end", start + len(chunk["text"]))
            text = chunk["text"]
        else:
            start = doc_text.find(chunk["text"])
            if start < 0:
                raise ValueError(f"{q['id']}: cannot locate gold text {cid}")
            end = start + len(chunk["text"])
            text = chunk["text"]
        out.append({"chunk_id": cid, "doc": chunk["doc"], "start": start,
                    "end": end, "text": text})
    return out


def recall_over_all(pipeline_or_retriever, qas, k, mode="dense", docs=None,
                    gold_chunk_map=None):
    """在所有 query 上算平均证据跨度 Recall@k。"""
    if docs is None or gold_chunk_map is None:
        raise ValueError("证据 Recall 需要 docs 和 gold_chunk_map")
    n, score = len(qas), 0.0
    for q in qas:
        hits = pipeline_or_retriever.retrieve(q["question"], k=k, mode=mode)
        gold = [gold_chunk_map[cid] for cid in q.get("gold_chunk_ids", [])]
        score += evidence_recall(gold, [c for c, _ in hits], k)
    return score / n if n else 0.0


def mrr_over_all(pipeline_or_retriever, qas, k=10, mode="dense"):
    s = 0.0
    for q in qas:
        hits = pipeline_or_retriever.retrieve(q["question"], k=k, mode=mode)
        got = [c["chunk_id"] for c, _ in hits]
        s += evaluate.mrr_at_k(q.get("gold_chunk_ids", []), got, k)
    return s / len(qas) if qas else 0.0
