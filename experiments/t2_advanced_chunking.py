"""T2 advanced chunking experiment: semantic split and Small-to-Big.

This script does not touch ``results/`` or the existing ``src.chunker``.  It
writes isolated outputs under ``repair_outputs/acceptance_20261023/t2/`` so the
formal ``results/`` and manifest are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import load_env, evidence_recall, gold_chunks_for_docs
from src import advanced_chunking, config, embed, evaluate, index_store, retriever
from src import generator as gen


OUT = ROOT / "repair_outputs" / "acceptance_20261023" / "t2"
TMP = OUT / "_tmp"

HF_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / (
    "models--BAAI--bge-small-zh-v1.5"
    "/snapshots/7999e1d3359715c523056ef9478215996d62a620")


def use_local_encoder() -> dict:
    """Set the cached local embedding encoder (no config change)."""
    from sentence_transformers import SentenceTransformer
    if HF_SNAPSHOT.is_dir():
        embed._encoder = SentenceTransformer(
            str(HF_SNAPSHOT), local_files_only=True)
        return {"source": "hf_snapshot", "revision": HF_SNAPSHOT.name}
    # Fall back to configured HF cache path if the snapshot moved away.
    model_dir = Path(config.EMBED_MODEL_NAME)
    if model_dir.is_dir():
        embed._encoder = SentenceTransformer(str(model_dir), local_files_only=True)
        return {"source": "local_dir", "revision": ""}
    raise RuntimeError("local embedding snapshot not found; run in offline mode")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_of(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def source_hashes() -> Dict[str, str]:
    """Return source-code hashes used by this isolated experiment."""
    files = [
        "src/advanced_chunking.py",
        "src/chunker.py",
        "src/config.py",
        "src/embed.py",
        "src/index_store.py",
        "src/retriever.py",
        "src/generator.py",
        "src/evaluate.py",
        "experiments/common.py",
        "experiments/t2_advanced_chunking.py",
    ]
    return {p: _hash_of(ROOT / p) for p in files}


def config_public_meta() -> Dict[str, str]:
    """Non-secret model/url fingerprints for provenance (no endpoint text)."""
    return {
        "embed_model": config.EMBED_MODEL_NAME,
        "llm_model": config.LLM_MODEL,
        "llm_base_url_sha256": hashlib.sha256(
            config.LLM_BASE_URL.encode("utf-8")).hexdigest(),
        "eval_jsonl": str(config.EVAL_JSONL),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _em_proxy(gold: str, answer: str) -> float:
    norm = lambda s: re.sub(r"\s+", "", s or "")
    return float(bool(norm(gold)) and norm(gold) == norm(answer))


def _answer_fields(question: str, chunks: List[dict], gold_answer: str) -> dict:
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
    if not answer.strip():
        return {"answer": "", "generation_status": "error",
                "generation_error": "EmptyGeneration",
                "EM_proxy": "", "ROUGE-L_proxy": "", "answer_time_s": elapsed}
    return {
        "answer": answer,
        "generation_status": "ok",
        "generation_error": "",
        "EM_proxy": _em_proxy(gold_answer, answer),
        "ROUGE-L_proxy": evaluate.rouge_l_f1(gold_answer, answer),
        "answer_time_s": elapsed,
    }


def _hit_evidence(gold: List[dict], retrieved: List[dict], k: int = 5) -> float:
    return float(any(
        g["doc"] == c["doc"]
        and min(g["end"], c["end"]) > max(g["start"], c["start"])
        for g in gold for c in retrieved[:k]))


def build_config(
    method: str, size: int, docs: Dict[str, str]
) -> Tuple[List[dict], Dict[str, dict], Dict[str, str]]:
    """Return (index_chunks, parent_map, small_to_parent).  For semantic, the
    index and generation chunks are the same semantic chunks; for Small-to-Big
    they are small and parent chunks respectively."""
    if method == "semantic":
        chunks = advanced_chunking.semantic_split_docs(docs, size=size)
        return chunks, {}, {}
    # small-to-big
    parent_size = max(size * 4, 1024)
    small_chunks, parents_by_id, mapping = (
        advanced_chunking.small_to_big_split_docs(
            docs, small_size=size, small_overlap=0,
            parent_factor=4, parent_size=parent_size))
    return small_chunks, parents_by_id, mapping


def render_config(cfg: dict) -> str:
    return f"{cfg['method']}/{cfg['size_for_index']}"


def checkpoint_rows(kind: str, rows: List[dict], fields: List[str]) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    with (TMP / f"checkpoint_{kind}.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(max_configs: int | None = None) -> List[dict]:
    embed_meta = use_local_encoder()
    print(f"[t2-advanced] encoder={embed_meta}", flush=True)
    config.require_llm()
    env = load_env()
    docs, qas = env["docs"], env["qas"]
    methods = ["semantic", "small_to_big"]
    sizes = [256, 512, 1024]
    cfgs = [{"method": m, "size_for_index": s} for m in methods for s in sizes]
    if max_configs is not None and max_configs <= 0:
        raise ValueError("max_configs must be None or positive")
    if max_configs is not None:
        cfgs = cfgs[:max_configs]

    baseline_chunks = chunks_fixed(docs)
    chunks_by_id = {c["chunk_id"]: c for c in baseline_chunks}
    gold_by_question = {
        q["id"]: gold_chunks_for_docs(q, chunks_by_id, docs) for q in qas}

    summary_rows: List[dict] = []
    all_rows: List[dict] = []
    for cfg in cfgs:
        cfg_name = render_config(cfg)
        print(f"[t2-advanced] {cfg_name} start")
        index_chunks, parents_by_id, small_map = build_config(
            cfg["method"], cfg["size_for_index"], docs)
        idx = index_store.IndexStore(index_chunks)
        vecs = embed.encode([c["text"] for c in index_chunks])
        idx.build_dense(vecs)
        idx.build_bm25(retriever.default_tokenize)
        search = retriever.Retriever(idx)

        def case_for(q: dict) -> dict:
            start = perf_counter()
            try:
                hits = search.retrieve(q["question"], 5, mode="hybrid")
                retr_elapsed = perf_counter() - start
            except Exception as exc:
                return {
                    "id": q["id"], "level": q.get("level", ""),
                    "question": q["question"], "gold_answer": "",
                    "method": cfg["method"], "size_for_index": cfg["size_for_index"],
                    "retriever": "hybrid",
                    "generator_model": config.LLM_MODEL,
                    "retrieval_status": "error",
                    "retrieval_error": type(exc).__name__,
                    "retrieval_time_s": "",
                    "index_n_chunks": len(index_chunks),
                    "retrieved_small": "", "retrieved_small_evidence": "",
                    "retrieved_parent": "", "retrieved_parent_evidence": "",
                    "Recall@5_small_evidence": "", "Hit@5_small_evidence": "",
                    "Recall@5_parent_evidence": "", "Hit@5_parent_evidence": "",
                    "answer": "", "generation_status": "error",
                    "generation_error": "RetrievalError",
                    "EM_proxy": "", "ROUGE-L_proxy": "", "answer_time_s": "",
                }
            retrieved_small = [c for c, _ in hits][:5]
            gold = gold_by_question[q["id"]]
            small_ids = [c["chunk_id"] for c in retrieved_small]
            if cfg["method"] == "small_to_big":
                parents = []
                for c in retrieved_small:
                    pid = c.get("parent_chunk_id")
                    if pid and pid in parents_by_id:
                        parents.append(parents_by_id[pid])
                # de-duplicate by parent id to avoid feeding the same parent twice
                seen = set()
                uniq_parents = []
                for p in parents:
                    if p["chunk_id"] not in seen:
                        child_ids = [c["chunk_id"] for c in retrieved_small
                                     if c.get("parent_chunk_id") == p["chunk_id"]]
                        copy = dict(p)
                        copy["parent_of"] = child_ids
                        uniq_parents.append(copy)
                        seen.add(p["chunk_id"])
                parents = uniq_parents[:5]
                parent_ids = [p["chunk_id"] for p in parents]
                recall_p = evidence_recall(gold, parents, 5)
                hit_p = _hit_evidence(gold, parents)
            else:
                recall_p, hit_p = "", ""
                parent_ids, parents = [], retrieved_small
            base = {
                "id": q["id"], "level": q.get("level", ""),
                "question": q["question"], "gold_answer": q.get("gold_answer", ""),
                "method": cfg["method"], "size_for_index": cfg["size_for_index"],
                "retriever": "hybrid",
                "generator_model": config.LLM_MODEL,
                "retrieval_status": "ok", "retrieval_error": "",
                "retrieval_time_s": retr_elapsed,
                "index_n_chunks": len(index_chunks),
                "retrieved_small": json.dumps(small_ids, ensure_ascii=False),
                "retrieved_small_evidence": json.dumps(
                    retrieved_small, ensure_ascii=False),
                "retrieved_parent": json.dumps(parent_ids, ensure_ascii=False),
                "retrieved_parent_evidence": json.dumps(
                    parents, ensure_ascii=False),
                "Recall@5_small_evidence": evidence_recall(gold, retrieved_small, 5),
                "Hit@5_small_evidence": _hit_evidence(gold, retrieved_small),
                "Recall@5_parent_evidence": recall_p,
                "Hit@5_parent_evidence": hit_p,
            }
            ans = _answer_fields(q["question"], parents, q.get("gold_answer", ""))
            return {**base, **ans}

        # Small job queue: 3 workers, per-question checkpoint.
        job_results: List[dict] = []
        finished = 0
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(case_for, q): q for q in qas}
            seen = 0
            for fut in as_completed(futures):
                res = fut.result()
                job_results.append(res)
                finished += 1
                seen += 1
                if seen % 5 == 0:
                    partial = all_rows + job_results
                    checkpoint_rows("case", partial, CASE_FIELDS)
                    print(f"[t2-advanced] {cfg_name} completed {seen}/{len(qas)}")
        job_results.sort(key=lambda r: r["id"])
        all_rows.extend(job_results)

        ok = [r for r in job_results
              if r["generation_status"] == "ok"
              and r["retrieval_status"] == "ok"]
        errs = [r for r in job_results if r["generation_status"] == "error"
                or r["retrieval_status"] == "error"]
        summary_rows.append({
            "method": cfg["method"], "size_for_index": cfg["size_for_index"],
            "generator_model": config.LLM_MODEL, "retriever": "hybrid",
            "n_chunks_index": len(index_chunks),
            "n_questions": len(qas), "n_generated": len(ok),
            "n_errors": len(errs),
            "Recall@5_small_evidence": _mean([
                float(r["Recall@5_small_evidence"])
                for r in job_results if r["Recall@5_small_evidence"] != ""]),
            "Hit@5_small_evidence": _mean([
                float(r["Hit@5_small_evidence"])
                for r in job_results if r["Hit@5_small_evidence"] != ""]),
            **({} if cfg["method"] != "small_to_big" else {
                "Recall@5_parent_evidence": _mean([
                    float(r["Recall@5_parent_evidence"])
                    for r in job_results if r["Recall@5_parent_evidence"] != ""]),
                "Hit@5_parent_evidence": _mean([
                    float(r["Hit@5_parent_evidence"])
                    for r in job_results if r["Hit@5_parent_evidence"] != ""])}),
            "EM_proxy": _mean([
                float(r["EM_proxy"]) for r in job_results if r["EM_proxy"] != ""]),
            "ROUGE-L_proxy": _mean([
                float(r["ROUGE-L_proxy"])
                for r in job_results if r["ROUGE-L_proxy"] != ""]),
            "answer_time_s": _mean([
                float(r["answer_time_s"]) for r in job_results
                if r["answer_time_s"] != ""]),
            "retrieval_time_s": _mean([
                float(r["retrieval_time_s"]) for r in job_results
                if r["retrieval_time_s"] != ""]),
        })
        checkpoint_rows("summary", summary_rows, SUMMARY_FIELDS)

    save_csv("t2_advanced_chunking.csv", summary_rows, SUMMARY_FIELDS)
    save_csv("t2_advanced_chunking_cases.csv", all_rows, CASE_FIELDS)
    manifest = {
        "runtime": "experiments/t2_advanced_chunking.py",
        "created_utc_hint": "2026-10-23T00:00:00+08:00",
        "embed_source": embed_meta["source"],
        "embed_revision": embed_meta.get("revision", ""),
        "qa_set": {
            "path": "data/eval/qa_set.jsonl",
            "sha256": _hash_of(ROOT / "data" / "eval" / "qa_set.jsonl"),
            "n": len(qas),
        },
        "docs": {"path": "data/corpus", "n_docs": len(docs)},
        "config": config_public_meta(),
        "sources": source_hashes(),
        "configs": cfgs,
        "summary": summary_rows,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[t2-advanced] done configs={len(cfgs)} cases={len(all_rows)}"
          f" errors={sum(r.get('generation_status')!='ok' for r in all_rows)}")
    return summary_rows


def save_csv(name: str, rows: List[dict], fields: List[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


# Re-export functions used by the checker without mixing modules.
def chunks_fixed(docs):
    from src import chunker as C
    return C.chunk_documents(docs)


CASE_FIELDS = [
    "id", "level", "question", "gold_answer", "method", "size_for_index",
    "retriever", "generator_model", "retrieval_status", "retrieval_error",
    "retrieval_time_s", "index_n_chunks", "retrieved_small",
    "retrieved_small_evidence", "retrieved_parent",
    "retrieved_parent_evidence", "Recall@5_small_evidence",
    "Hit@5_small_evidence", "Recall@5_parent_evidence",
    "Hit@5_parent_evidence", "answer", "generation_status",
    "generation_error", "EM_proxy", "ROUGE-L_proxy", "answer_time_s",
]
SUMMARY_FIELDS = [
    "method", "size_for_index", "generator_model", "retriever", "n_chunks_index",
    "n_questions", "n_generated", "n_errors", "Recall@5_small_evidence",
    "Hit@5_small_evidence",
    *(["Recall@5_parent_evidence", "Hit@5_parent_evidence"]),
    "EM_proxy", "ROUGE-L_proxy", "answer_time_s", "retrieval_time_s",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args()
    run(max_configs=args.max_configs)

