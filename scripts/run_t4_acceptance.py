"""One-shot exploratory T4 focused run: dense vs BM25, hybrid display only.

The 50-query set is frozen before the embedding model is loaded. A claim file
is written immediately after the freeze, before model initialization. Any
existing claim or formal output blocks another retrieval run.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import chunker, config, evaluate, loader  # noqa: E402
from src.index_store import IndexStore  # noqa: E402
from src import retriever  # noqa: E402

QUERY_FILE = PROJECT_ROOT / "scripts" / "t4_acceptance_queries.jsonl"
OUT_DIR = PROJECT_ROOT / "repair_outputs" / "acceptance_20261023" / "t4"
FORMAL_QA = PROJECT_ROOT / "data" / "eval" / "qa_set.jsonl"
OLD_QUERIES = PROJECT_ROOT / "scripts" / "t4_supplement_queries.jsonl"
PILOT_MANIFEST = OUT_DIR / "pilot_recovery_manifest.json"
RESULTS_FILE = OUT_DIR / "t4_acceptance_per_query.json"
TOP10_FILE = OUT_DIR / "t4_acceptance_top10.jsonl"
MANIFEST_FILE = OUT_DIR / "t4_acceptance_manifest.json"
AUDIT_FILE = OUT_DIR / "t4_acceptance_checker.json"
FREEZE_FILE = OUT_DIR / "t4_freeze_final.json"
CLAIM_FILE = OUT_DIR / "t4_run_claim_final.json"
EXPECTED_QUERY_SHA = "c588214360fe684226983951e4bbc74d0430a863f0193fa994588c735fa94395"
EXPECTED_PILOT_SHA = "9dd61c065afeacdc46f4c40fef57f9b87748f384f8ef9a8b059a4d084205d95b"
CHUNKER = "fixed"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MODES = ("dense", "bm25", "hybrid")
KS = (1, 3, 5, 10)
RUN_VERSION = "t4-exploratory-focused-v2"
LABEL = "探索性专项"
SOURCE_FILES = (
    "src/config.py",
    "src/loader.py",
    "src/chunker.py",
    "src/index_store.py",
    "src/embed.py",
    "src/retriever.py",
    "src/evaluate.py",
    "scripts/run_t4_acceptance.py",
    "scripts/check_t4_acceptance.py",
    "scripts/t4_acceptance_queries.jsonl",
)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def strict_winner(dense_metrics, bm25_metrics):
    """Only dense and BM25 participate; hybrid is display-only."""
    dense_key = (dense_metrics["Recall@3"], dense_metrics["MRR@10"])
    bm25_key = (bm25_metrics["Recall@3"], bm25_metrics["MRR@10"])
    if dense_key > bm25_key:
        return "dense"
    if bm25_key > dense_key:
        return "bm25"
    return "tie"


def query_sha(rows):
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")) + "\n" for r in rows)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False,
                                       separators=(",", ":")) + "\n"
                            for r in rows), encoding="utf-8", newline="\n")


def validate_queries(queries, chunks):
    errors = []
    if not 40 <= len(queries) <= 60:
        errors.append("query_count=%d, expected 40..60" % len(queries))
    if len(queries) != 50:
        errors.append("query_count must be 50")
    ids = [q.get("id") for q in queries]
    if len(set(ids)) != len(ids):
        errors.append("duplicate query ids")
    questions = [q.get("question") for q in queries]
    if len(set(questions)) != len(questions):
        errors.append("duplicate questions")
    by_id = {c["chunk_id"]: c for c in chunks}
    required = {"id", "category", "question", "source_doc", "support_quotes",
                "gold_chunk_ids", "gold_texts", "gold_answer", "audit_reason",
                "neighbor_review", "dedup_against_formal",
                "dedup_against_old"}
    for q in queries:
        missing = sorted(required - set(q))
        if missing:
            errors.append("%s missing fields %s" % (q.get("id"), missing))
            continue
        if not q["support_quotes"] or not q["gold_chunk_ids"]:
            errors.append("%s has empty evidence/gold" % q["id"])
            continue
        if q.get("audit_type") != "AI audit (non-human)":
            errors.append("%s audit_type mismatch" % q["id"])
        if len(q["gold_chunk_ids"]) != len(set(q["gold_chunk_ids"])):
            errors.append("%s duplicate gold ids" % q["id"])
        if {g["chunk_id"] for g in q["gold_texts"]} != set(q["gold_chunk_ids"]):
            errors.append("%s gold_texts/id mismatch" % q["id"])
        actual_hits = set()
        for quote in q["support_quotes"]:
            hits = {c["chunk_id"] for c in chunks
                    if c["doc"] == q["source_doc"] and quote in c["text"]}
            if not hits:
                errors.append("%s quote not found in source doc" % q["id"])
            actual_hits |= hits
        if actual_hits != set(q["gold_chunk_ids"]):
            errors.append("%s gold enumeration mismatch" % q["id"])
        for gold in q["gold_texts"]:
            chunk = by_id.get(gold["chunk_id"])
            if chunk is None or chunk["text"] != gold["text"]:
                errors.append("%s frozen gold text mismatch" % q["id"])
    return errors


def metric_block(gold_ids, hits):
    ids = [c["chunk_id"] for c, _ in hits]
    out = {}
    for k in KS:
        out["Hit@%d" % k] = evaluate.hit_at_k(gold_ids, ids, k)
        out["Recall@%d" % k] = evaluate.recall_at_k(gold_ids, ids, k)
    out["MRR@10"] = evaluate.mrr_at_k(gold_ids, ids, 10)
    return out


def top_record(rank, chunk, score, gold_ids):
    return {
        "rank": rank,
        "chunk_id": chunk["chunk_id"],
        "score": float(score),
        "doc": chunk["doc"],
        "text": chunk["text"],
        "seq": chunk["seq"],
        "start": chunk.get("start"),
        "end": chunk.get("end"),
        "is_gold": chunk["chunk_id"] in gold_ids,
    }


def make_analysis(query, modes):
    dense = modes["dense"]
    bm25 = modes["bm25"]
    winner = strict_winner(dense, bm25)
    dense_ids = [e["chunk_id"] for e in dense["top3"]]
    bm25_ids = [e["chunk_id"] for e in bm25["top3"]]
    if winner == "dense":
        cause = ("Dense 的未舍入 Recall@3/MRR@10 严格高于 BM25；"
                 "hybrid 不参与判胜。")
    elif winner == "bm25":
        cause = ("BM25 的未舍入 Recall@3/MRR@10 严格高于 Dense；"
                 "hybrid 不参与判胜。")
    else:
        cause = ("Dense 与 BM25 的未舍入 Recall@3/MRR@10 完全相等，"
                 "按预注册规则记为 tie；hybrid 不参与判胜。")
    return {
        "winner": winner,
        "winner_basis": ("unrounded Recall@3, then unrounded MRR@10; "
                         "dense vs BM25 only"),
        "dense_minus_bm25": {
            "Recall@3": dense["Recall@3"] - bm25["Recall@3"],
            "MRR@10": dense["MRR@10"] - bm25["MRR@10"],
        },
        "top3_id_intersection": [
            cid for cid in dense_ids if cid in bm25_ids],
        "dense_gold_ranks_top3": [
            e["rank"] for e in dense["top3"] if e["is_gold"]],
        "bm25_gold_ranks_top3": [
            e["rank"] for e in bm25["top3"] if e["is_gold"]],
        "hybrid_observed": {
            "Recall@3": modes["hybrid"]["Recall@3"],
            "MRR@10": modes["hybrid"]["MRR@10"],
            "used_for_winner": False,
        },
        "cause": cause,
        "label": LABEL,
        "gold_note": ("全部 gold 均来自同一来源文档中所有包含支持引文的"
                      "512/128 重叠分片；多分片支持时逐条列入。"),
    }


def summarize(records):
    summary = {}
    for mode in MODES:
        row = {"queries": len(records), "mode": mode}
        for key in ("Hit@1", "Hit@3", "Hit@5", "Hit@10",
                    "Recall@1", "Recall@3", "Recall@5", "Recall@10",
                    "MRR@10"):
            vals = [r["modes"][mode][key] for r in records]
            row[key] = sum(vals) / len(vals)
        summary[mode] = row
    counts = {"dense": 0, "bm25": 0, "tie": 0}
    for r in records:
        counts[r["analysis"]["winner"]] += 1
    summary["strict_winner_counts"] = counts
    summary["target"] = {"dense": 5, "bm25": 5}
    summary["meets_dense_target"] = counts["dense"] >= 5
    summary["meets_bm25_target"] = counts["bm25"] >= 5
    summary["meets_both_targets"] = (summary["meets_dense_target"]
                                     and summary["meets_bm25_target"])
    return summary


def collision_report(queries):
    formal = read_jsonl(FORMAL_QA)
    old = read_jsonl(OLD_QUERIES)
    formal_ids = {q["id"] for q in formal}
    old_ids = {q.get("id") for q in old}
    formal_q = {q["question"] for q in formal}
    old_q = {q.get("question") for q in old}
    formal_goldsets = {
        frozenset(g for g in (x.get("gold_chunk_ids")
                              or [x.get("gold_chunk_id")]) if g)
        for x in formal}
    old_goldsets = {
        frozenset(g for g in (x.get("gold_chunk_ids")
                              or [x.get("gold_chunk_id")]) if g)
        for x in old}
    return {
        "blocking_formal_id_collisions": [
            q["id"] for q in queries if q["id"] in formal_ids],
        "blocking_old_id_collisions": [
            q["id"] for q in queries if q["id"] in old_ids],
        "blocking_formal_question_collisions": [
            q["id"] for q in queries if q["question"] in formal_q],
        "blocking_old_question_collisions": [
            q["id"] for q in queries if q["question"] in old_q],
        "report_only_formal_shared_goldsets": [
            q["id"] for q in queries
            if frozenset(q["gold_chunk_ids"]) in formal_goldsets],
        "report_only_old_shared_goldsets": [
            q["id"] for q in queries
            if frozenset(q["gold_chunk_ids"]) in old_goldsets],
        "shared_goldset_policy": (
            "gold 集合相同不等同于重复题；只阻止 ID 或题面完全重复。"),
    }


def exploratory_history():
    pilot = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    history = pilot["exploration_history"]
    return {
        "label": LABEL,
        "not_blind_independent_test": True,
        "pilot_batches": history["pilot_batches"],
        "pilot_query_executions": history["query_executions"],
        "pilot_counts_per_batch": history["counts_per_batch"],
        "pilot_evidence_manifest": str(
            PILOT_MANIFEST.relative_to(PROJECT_ROOT)),
        "pilot_evidence_sha256": sha256_file(PILOT_MANIFEST),
        "pilot_metrics_recovered": True,
        "pilot_top10_recovered": False,
        "pilot_top10_reason": (
            "两轮 pilot 只打印聚合指标与 gold 排名，未落盘 top10；"
            "不得凭记忆或重新检索伪造 top10。"),
        "selection_bias": pilot["selection_bias"],
        "final_focused_batches_max": 1,
        "final_query_count": 50,
        "disclosure": (
            "题集构造、gold 审核及 16 题、27 题两轮 pilot 均在最终冻结前完成。"
            "本结果只能表述为探索性专项，不能表述为盲测独立一次。"),
    }


def canonical_chunk_sha(chunks):
    hashes = sorted(
        hashlib.sha256(
            (c["doc"] + "\0" + str(c["seq"]) + "\0" + c["text"]).encode(
                "utf-8")).hexdigest()
        for c in chunks)
    return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


def build_freeze(queries, docs, chunks):
    source_hashes = {
        rel: sha256_file(PROJECT_ROOT / rel) for rel in SOURCE_FILES}
    corpus = {}
    corpus_bytes = {}
    for doc, text in docs.items():
        corpus[doc] = {
            "chars": len(text),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        corpus_bytes[doc] = sha256_file(Path(config.CORPUS_DIR) / doc)
    return {
        "freeze_version": RUN_VERSION,
        "label": LABEL,
        "frozen_at": utc_now(),
        "query_file": "scripts/t4_acceptance_queries.jsonl",
        "query_file_sha256": sha256_file(QUERY_FILE),
        "query_rows_canonical_sha256": query_sha(queries),
        "query_count": len(queries),
        "query_ids": [q["id"] for q in queries],
        "gold_chunk_count_with_multiplicity": sum(
            len(q["gold_chunk_ids"]) for q in queries),
        "formal_qa_sha256": sha256_file(FORMAL_QA),
        "old_queries_sha256": sha256_file(OLD_QUERIES),
        "source_sha256": source_hashes,
        "embedding_model": config.EMBED_MODEL_NAME,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "chunking": {"method": CHUNKER, "size": CHUNK_SIZE,
                     "overlap": CHUNK_OVERLAP,
                     "document_count": len(docs),
                     "character_count": sum(len(t) for t in docs.values()),
                     "chunk_count": len(chunks),
                     "canonical_chunk_sha256": canonical_chunk_sha(chunks)},
        "corpus_sha256": {
            "files": len(docs),
            "file_sha256": corpus_bytes,
            "canonical_chunk_sha256": canonical_chunk_sha(chunks),
            "canonical_chunk_hash_rule": (
                "sha256 of sorted sha256(doc + NUL + seq + NUL + text) lines"),
        },
        "corpus": corpus,
        "exploratory_history": exploratory_history(),
        "formal_retrieval_started": False,
        "formal_retrieval_run_count": 0,
        "winner_rule": ("dense vs BM25 only: unrounded Recall@3 first, "
                        "then unrounded MRR@10; exact equality is tie"),
        "hybrid_policy": "display only; excluded from winner determination",
        "dedup_check": collision_report(queries),
        "audit_type": "AI audit (non-human)",
    }


def check_freeze(freeze, queries):
    errors = []
    if freeze.get("label") != LABEL:
        errors.append("freeze label mismatch")
    if freeze.get("query_file_sha256") != sha256_file(QUERY_FILE):
        errors.append("freeze query file hash mismatch")
    if freeze.get("query_file_sha256") != EXPECTED_QUERY_SHA:
        errors.append("frozen hash is not the preregistered hash")
    if freeze.get("query_rows_canonical_sha256") != query_sha(queries):
        errors.append("freeze canonical query hash mismatch")
    collisions = freeze.get("dedup_check", {})
    for key, value in collisions.items():
        if key.startswith("blocking_") and value:
            errors.append("%s: %s" % (key, value))
    if freeze.get("formal_retrieval_started"):
        errors.append("freeze already marked as retrieval-started")
    if freeze.get("formal_retrieval_run_count") != 0:
        errors.append("freeze run count is not zero")
    history = freeze.get("exploratory_history", {})
    if history.get("not_blind_independent_test") is not True:
        errors.append("blind-test disclosure mismatch")
    if history.get("pilot_batches") != 2:
        errors.append("pilot batch count mismatch")
    if history.get("pilot_query_executions") != 43:
        errors.append("pilot query execution count mismatch")
    if history.get("pilot_counts_per_batch") != [16, 27]:
        errors.append("pilot counts mismatch")
    if history.get("final_focused_batches_max") != 1:
        errors.append("final focused batch limit mismatch")
    if history.get("final_query_count") != 50:
        errors.append("final query count mismatch")
    if freeze.get("query_count") != 50:
        errors.append("frozen query count mismatch")
    if freeze.get("gold_chunk_count_with_multiplicity") != 72:
        errors.append("frozen gold count mismatch")
    if freeze.get("chunking", {}).get("document_count") != 43:
        errors.append("frozen document count mismatch")
    if freeze.get("chunking", {}).get("chunk_count") != 479:
        errors.append("frozen chunk count mismatch")
    return errors


def verify_frozen_files(freeze):
    errors = []
    for rel, expected in freeze["source_sha256"].items():
        if sha256_file(PROJECT_ROOT / rel) != expected:
            errors.append("source changed after freeze: %s" % rel)
    return errors


def run():
    os.environ["RAG_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in (CLAIM_FILE, RESULTS_FILE, TOP10_FILE,
                                 MANIFEST_FILE, AUDIT_FILE) if p.exists()]
    if existing:
        raise RuntimeError(
            "one-shot run refused; existing final artifacts: %s"
            % ", ".join(existing))
    if FREEZE_FILE.exists():
        raise RuntimeError(
            "final freeze already exists; refusing a second final freeze")
    if sha256_file(QUERY_FILE) != EXPECTED_QUERY_SHA:
        raise RuntimeError("query file does not match the final frozen hash")
    if sha256_file(PILOT_MANIFEST) != EXPECTED_PILOT_SHA:
        raise RuntimeError("pilot recovery manifest changed")

    queries = read_jsonl(QUERY_FILE)
    docs = loader.read_all_docs(config.CORPUS_DIR)
    chunks = chunker.chunk_documents(docs, CHUNKER, CHUNK_SIZE, CHUNK_OVERLAP)
    errors = validate_queries(queries, chunks)
    if errors:
        raise RuntimeError("query validation failed:\n" + "\n".join(errors))

    freeze = build_freeze(queries, docs, chunks)
    freeze_errors = check_freeze(freeze, queries)
    if freeze_errors:
        raise RuntimeError("freeze validation failed:\n"
                           + "\n".join(freeze_errors))
    write_json(FREEZE_FILE, freeze)

    # Write the one-shot claim before importing or loading the model.
    claim = {
        "claim_version": RUN_VERSION,
        "label": LABEL,
        "claimed_at": utc_now(),
        "pid": os.getpid(),
        "freeze_file": FREEZE_FILE.name,
        "freeze_sha256": sha256_file(FREEZE_FILE),
        "query_file_sha256": freeze["query_file_sha256"],
        "query_rows_canonical_sha256": freeze[
            "query_rows_canonical_sha256"],
        "source_sha256": freeze["source_sha256"],
        "formal_retrieval_started": True,
        "formal_retrieval_run_count_claim": 1,
    }
    write_json(CLAIM_FILE, claim)

    frozen_errors = verify_frozen_files(freeze)
    if frozen_errors:
        raise RuntimeError("frozen files changed before model load:\n"
                           + "\n".join(frozen_errors))

    print("Final freeze: %s" % FREEZE_FILE.name)
    print("Claim written before model load: %s" % CLAIM_FILE.name)
    print("Validated exploratory focused queries: %d; chunks: %d" %
          (len(queries), len(chunks)))

    from src import embed

    started = time.perf_counter()
    vectors = embed.encode([c["text"] for c in chunks])
    index = IndexStore(chunks)
    index.build_dense(vectors)
    index.build_bm25(retriever.default_tokenize)
    searcher = retriever.Retriever(index)

    records = []
    top10_rows = []
    for number, query in enumerate(queries, 1):
        query_vec = embed.embed_text_one(query["question"])
        modes = {}
        for mode in MODES:
            hits = list(searcher.retrieve(
                query["question"], 10, mode=mode, query_vec=query_vec))[:10]
            if len(hits) != 10:
                raise RuntimeError("%s/%s returned %d hits" %
                                   (query["id"], mode, len(hits)))
            metrics = metric_block(query["gold_chunk_ids"], hits)
            top10 = [top_record(rank, chunk, score,
                                set(query["gold_chunk_ids"]))
                     for rank, (chunk, score) in enumerate(hits, 1)]
            metrics["top3"] = top10[:3]
            metrics["top10"] = top10
            modes[mode] = metrics
            top10_rows.append({
                "id": query["id"],
                "mode": mode,
                "gold_chunk_ids": query["gold_chunk_ids"],
                "top10": top10,
            })
        record = {
            "query_index": number,
            "id": query["id"],
            "category": query["category"],
            "question": query["question"],
            "source_doc": query["source_doc"],
            "support_quotes": query["support_quotes"],
            "gold_chunk_ids": query["gold_chunk_ids"],
            "gold_texts": query["gold_texts"],
            "gold_answer": query["gold_answer"],
            "audit_reason": query["audit_reason"],
            "audit_type": query["audit_type"],
            "neighbor_review": query["neighbor_review"],
            "dedup_against_formal": query["dedup_against_formal"],
            "dedup_against_old": query["dedup_against_old"],
            "modes": modes,
        }
        record["analysis"] = make_analysis(query, modes)
        records.append(record)
        if number % 5 == 0 or number == len(queries):
            print("  [%d/%d] %s -> %s" %
                  (number, len(queries), query["id"],
                   record["analysis"]["winner"]))

    post_errors = verify_frozen_files(freeze)
    if post_errors:
        raise RuntimeError("frozen files changed during retrieval:\n"
                           + "\n".join(post_errors))
    elapsed = time.perf_counter() - started
    summary = summarize(records)
    manifest = {
        "run_version": RUN_VERSION,
        "label": LABEL,
        "run_status": "completed",
        "formal_run": True,
        "exploratory_focused_run": True,
        "not_blind_independent_test": True,
        "exploratory_pilot_batches": 2,
        "exploratory_pilot_query_executions": 43,
        "formal_retrieval_run_count": 1,
        "formal_retrieval_started": True,
        "started_at": freeze["frozen_at"],
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "freeze_file": FREEZE_FILE.name,
        "claim_file": CLAIM_FILE.name,
        "freeze_sha256": sha256_file(FREEZE_FILE),
        "claim_sha256": sha256_file(CLAIM_FILE),
        "frozen_query_file_sha256": freeze["query_file_sha256"],
        "frozen_query_rows_canonical_sha256": freeze[
            "query_rows_canonical_sha256"],
        "formal_qa_sha256": freeze["formal_qa_sha256"],
        "old_queries_sha256": freeze["old_queries_sha256"],
        "embedding_model": config.EMBED_MODEL_NAME,
        "offline": {"RAG_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
        "chunking": freeze["chunking"],
        "corpus_text_hash_count": len(freeze["corpus"]),
        "corpus_sha256": freeze["corpus_sha256"],
        "source_sha256": freeze["source_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "checker_sha256": sha256_file(
            PROJECT_ROOT / "scripts" / "check_t4_acceptance.py"),
        "modes": list(MODES),
        "retrieval_calls_per_query": {"dense": 1, "bm25": 1, "hybrid": 1},
        "winner_rule": freeze["winner_rule"],
        "hybrid_policy": freeze["hybrid_policy"],
        "summary": summary,
        "evidence": {
            "top10_full_text_and_scores": str(TOP10_FILE.relative_to(
                PROJECT_ROOT)),
            "per_query_full_top3_and_top10": str(RESULTS_FILE.relative_to(
                PROJECT_ROOT)),
            "pilot_recovery_manifest": str(
                PILOT_MANIFEST.relative_to(PROJECT_ROOT)),
        },
        "exploratory_history": freeze["exploratory_history"],
        "selection_bias_note": freeze["exploratory_history"]["disclosure"],
        "audit_type": "AI audit (non-human)",
    }
    write_jsonl(TOP10_FILE, top10_rows)
    write_json(RESULTS_FILE, records)
    write_json(MANIFEST_FILE, manifest)

    print("Exploratory focused retrieval complete in %.1fs" % elapsed)
    print("Strict winners: %s" % summary["strict_winner_counts"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="perform the one formal retrieval run")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
