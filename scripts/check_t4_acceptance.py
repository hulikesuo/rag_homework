"""Independent checker for the frozen T4 acceptance evidence.

Recomputes every metric and strict winner from the saved top10 only. It also
revalidates the frozen query file, gold texts, source quotes, overlap-complete
gold enumeration, top3 prefix, and the one-shot manifest.
"""
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import chunker, config, loader  # noqa: E402

QUERY_FILE = PROJECT_ROOT / "scripts" / "t4_acceptance_queries.jsonl"
FORMAL_QA = PROJECT_ROOT / "data" / "eval" / "qa_set.jsonl"
OLD_QUERIES = PROJECT_ROOT / "scripts" / "t4_supplement_queries.jsonl"
OUT_DIR = PROJECT_ROOT / "repair_outputs" / "acceptance_20261023" / "t4"
FREEZE_FILE = OUT_DIR / "t4_freeze_final.json"
CLAIM_FILE = OUT_DIR / "t4_run_claim_final.json"
RESULTS_FILE = OUT_DIR / "t4_acceptance_per_query.json"
TOP10_FILE = OUT_DIR / "t4_acceptance_top10.jsonl"
MANIFEST_FILE = OUT_DIR / "t4_acceptance_manifest.json"
AUDIT_FILE = OUT_DIR / "t4_acceptance_checker.json"
EXPECTED_QUERY_SHA = "c588214360fe684226983951e4bbc74d0430a863f0193fa994588c735fa94395"
EXPECTED_QA_SHA = "bfcbefc1ba2519c9728451559d30fd55f05151f903a927d836011ba07ae7b969"
EXPECTED_OLD_SHA = "d88df89ef885ad0b8fe76b3f4336f6323e642fa4b1d8b58e44b815bd625ea5f1"
EXPECTED_SOURCE_FILES = (
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


def canonical_sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def query_sha(rows):
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")) + "\n" for r in rows)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def recall_at_k(gold, ids, k):
    g = set(gold)
    return len(g & set(ids[:k])) / len(g) if g else 0.0


def hit_at_k(gold, ids, k):
    return 1.0 if set(gold) & set(ids[:k]) else 0.0


def mrr_at_k(gold, ids, k):
    g = set(gold)
    for rank, cid in enumerate(ids[:k], 1):
        if cid in g:
            return 1.0 / rank
    return 0.0


def expected_winner(dense, bm25):
    a = (dense["Recall@3"], dense["MRR@10"])
    b = (bm25["Recall@3"], bm25["MRR@10"])
    return "dense" if a > b else "bm25" if b > a else "tie"


def add_error(errors, kind, identity=""):
    errors.append({"type": kind, "id": identity})


def main():
    errors = []
    warnings = []
    for path in (QUERY_FILE, FREEZE_FILE, CLAIM_FILE, RESULTS_FILE,
                 TOP10_FILE, MANIFEST_FILE, FORMAL_QA, OLD_QUERIES):
        if not path.exists():
            add_error(errors, "missing_required_file", str(path))
    if errors:
        write_audit(errors, warnings, None, None)
        print("FAIL: required output missing")
        return 1

    query_hashes = {
        "queries": sha256_file(QUERY_FILE),
        "formal": sha256_file(FORMAL_QA),
        "old": sha256_file(OLD_QUERIES),
    }
    if query_hashes["queries"] != EXPECTED_QUERY_SHA:
        add_error(errors, "query_hash_mismatch")
    if query_hashes["formal"] != EXPECTED_QA_SHA:
        add_error(errors, "formal_qa_hash_mismatch")
    if query_hashes["old"] != EXPECTED_OLD_SHA:
        add_error(errors, "old_query_hash_mismatch")

    queries = read_jsonl(QUERY_FILE)
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    claim = json.loads(CLAIM_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    records = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    top10_rows = read_jsonl(TOP10_FILE)

    if len(records) != 50:
        add_error(errors, "record_count_not_fifty")
    if len(top10_rows) != 150:
        add_error(errors, "top10_row_count_not_150")
    if not 40 <= len(queries) <= 60:
        add_error(errors, "query_count_out_of_range")
    if len(queries) != 50:
        add_error(errors, "query_count_not_fifty")
    if len({q["id"] for q in queries}) != len(queries):
        add_error(errors, "duplicate_query_id")
    if len({q["question"] for q in queries}) != len(queries):
        add_error(errors, "duplicate_question")
    if [q["id"] for q in queries] != [r["id"] for r in records]:
        add_error(errors, "record_order_mismatch")

    if freeze["query_file_sha256"] != query_hashes["queries"]:
        add_error(errors, "freeze_query_hash_mismatch")
    if freeze["query_rows_canonical_sha256"] != query_sha(queries):
        add_error(errors, "freeze_canonical_hash_mismatch")
    if freeze["formal_qa_sha256"] != query_hashes["formal"]:
        add_error(errors, "freeze_formal_hash_mismatch")
    if freeze["old_queries_sha256"] != query_hashes["old"]:
        add_error(errors, "freeze_old_hash_mismatch")
    if freeze.get("source_sha256") != manifest.get("source_sha256"):
        add_error(errors, "freeze_manifest_source_hash_mismatch")
    if set(freeze.get("source_sha256", {})) != set(EXPECTED_SOURCE_FILES):
        add_error(errors, "source_file_set_mismatch")
    for rel, expected in freeze["source_sha256"].items():
        if sha256_file(PROJECT_ROOT / rel) != expected:
            add_error(errors, "source_file_hash_mismatch", rel)
    if freeze.get("formal_retrieval_started") is not False:
        add_error(errors, "freeze_not_pre_retrieval")
    if freeze.get("formal_retrieval_run_count") != 0:
        add_error(errors, "freeze_run_count_not_zero")
    if freeze.get("label") != "探索性专项":
        add_error(errors, "freeze_label_mismatch")
    if freeze.get("exploratory_history", {}).get(
            "not_blind_independent_test") is not True:
        add_error(errors, "exploratory_history_not_disclosed")
    if freeze.get("exploratory_history", {}).get(
            "pilot_counts_per_batch") != [16, 27]:
        add_error(errors, "pilot_counts_mismatch")
    if freeze.get("exploratory_history", {}).get(
            "pilot_query_executions") != 43:
        add_error(errors, "pilot_execution_count_mismatch")
    if sum(len(q["gold_chunk_ids"]) > 1 for q in queries) != 22:
        add_error(errors, "multi_gold_query_count_mismatch")
    if claim.get("label") != "探索性专项":
        add_error(errors, "claim_label_mismatch")
    if claim.get("claim_version") != freeze.get("freeze_version"):
        add_error(errors, "claim_version_mismatch")
    if claim.get("freeze_file") != FREEZE_FILE.name:
        add_error(errors, "claim_freeze_filename_mismatch")
    if claim.get("freeze_sha256") != sha256_file(FREEZE_FILE):
        add_error(errors, "claim_freeze_hash_mismatch")
    if claim.get("query_file_sha256") != query_hashes["queries"]:
        add_error(errors, "claim_query_hash_mismatch")
    if claim.get("query_rows_canonical_sha256") != freeze.get(
            "query_rows_canonical_sha256"):
        add_error(errors, "claim_canonical_hash_mismatch")
    if claim.get("source_sha256") != freeze["source_sha256"]:
        add_error(errors, "claim_source_hash_mismatch")
    if claim.get("formal_retrieval_started") is not True:
        add_error(errors, "claim_retrieval_not_started")
    if claim.get("formal_retrieval_run_count_claim") != 1:
        add_error(errors, "claim_run_count_not_one")

    if manifest.get("run_version") != freeze.get("freeze_version"):
        add_error(errors, "manifest_version_mismatch")
    if manifest.get("run_status") != "completed":
        add_error(errors, "manifest_run_not_completed")
    if manifest.get("formal_run") is not True:
        add_error(errors, "manifest_formal_run_missing")
    if manifest.get("exploratory_focused_run") is not True:
        add_error(errors, "manifest_exploratory_run_missing")
    if manifest.get("exploratory_pilot_batches") != 2:
        add_error(errors, "manifest_pilot_batches_mismatch")
    if manifest.get("exploratory_pilot_query_executions") != 43:
        add_error(errors, "manifest_pilot_executions_mismatch")
    if manifest.get("formal_retrieval_run_count") != 1:
        add_error(errors, "manifest_run_count_not_one")
    if manifest.get("formal_retrieval_started") is not True:
        add_error(errors, "manifest_retrieval_not_started")
    if manifest.get("frozen_query_file_sha256") != query_hashes["queries"]:
        add_error(errors, "manifest_query_hash_mismatch")
    if manifest.get("formal_qa_sha256") != query_hashes["formal"]:
        add_error(errors, "manifest_formal_hash_mismatch")
    if manifest.get("old_queries_sha256") != query_hashes["old"]:
        add_error(errors, "manifest_old_hash_mismatch")
    if manifest.get("frozen_query_rows_canonical_sha256") != freeze.get(
            "query_rows_canonical_sha256"):
        add_error(errors, "manifest_canonical_hash_mismatch")
    if manifest.get("freeze_sha256") != sha256_file(FREEZE_FILE):
        add_error(errors, "manifest_freeze_hash_mismatch")
    if manifest.get("runner_sha256") != sha256_file(
            PROJECT_ROOT / "scripts" / "run_t4_acceptance.py"):
        add_error(errors, "runner_hash_mismatch")
    if manifest.get("checker_sha256") != sha256_file(
            PROJECT_ROOT / "scripts" / "check_t4_acceptance.py"):
        add_error(errors, "checker_hash_mismatch")
    if manifest.get("claim_sha256") != sha256_file(CLAIM_FILE):
        add_error(errors, "manifest_claim_hash_mismatch")
    if manifest.get("freeze_file") != FREEZE_FILE.name:
        add_error(errors, "manifest_freeze_filename_mismatch")
    if manifest.get("claim_file") != CLAIM_FILE.name:
        add_error(errors, "manifest_claim_filename_mismatch")
    if manifest.get("label") != "探索性专项":
        add_error(errors, "manifest_label_mismatch")
    if manifest.get("not_blind_independent_test") is not True:
        add_error(errors, "blind_test_mislabel")
    if manifest.get("modes") != ["dense", "bm25", "hybrid"]:
        add_error(errors, "unexpected_modes")
    if manifest.get("embedding_model") != freeze.get("embedding_model"):
        add_error(errors, "embedding_model_mismatch")
    if manifest.get("embedding_model") != config.EMBED_MODEL_NAME:
        add_error(errors, "embedding_model_not_configured")
    if manifest.get("offline") != {
            "RAG_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}:
        add_error(errors, "offline_mode_mismatch")
    if manifest.get("retrieval_calls_per_query") != {
            "dense": 1, "bm25": 1, "hybrid": 1}:
        add_error(errors, "retrieval_call_count_mismatch")
    if manifest.get("runner_sha256") != freeze["source_sha256"].get(
            "scripts/run_t4_acceptance.py"):
        add_error(errors, "manifest_runner_source_mismatch")
    if manifest.get("checker_sha256") != freeze["source_sha256"].get(
            "scripts/check_t4_acceptance.py"):
        add_error(errors, "manifest_checker_source_mismatch")

    docs = loader.read_all_docs(config.CORPUS_DIR)
    chunks = chunker.chunk_documents(docs, "fixed", 512, 128)
    by_id = {c["chunk_id"]: c for c in chunks}
    if manifest["chunking"]["chunk_count"] != len(chunks):
        add_error(errors, "chunk_count_mismatch")
    if manifest["chunking"]["document_count"] != len(docs):
        add_error(errors, "document_count_mismatch")
    if manifest["chunking"]["character_count"] != sum(
            len(text) for text in docs.values()):
        add_error(errors, "character_count_mismatch")
    current_file_hashes = {
        doc: sha256_file(Path(config.CORPUS_DIR) / doc) for doc in docs}
    if freeze["corpus_sha256"]["file_sha256"] != current_file_hashes:
        add_error(errors, "corpus_file_hash_mismatch")
    chunk_hashes = sorted(
        hashlib.sha256(
            (c["doc"] + "\0" + str(c["seq"]) + "\0" + c["text"]).encode(
                "utf-8")).hexdigest()
        for c in chunks)
    canonical_chunk_sha = hashlib.sha256(
        "\n".join(chunk_hashes).encode("utf-8")).hexdigest()
    if freeze["chunking"]["canonical_chunk_sha256"] != canonical_chunk_sha:
        add_error(errors, "canonical_chunk_hash_mismatch")
    if freeze["corpus_sha256"]["canonical_chunk_sha256"] != canonical_chunk_sha:
        add_error(errors, "freeze_corpus_chunk_hash_mismatch")
    if len(by_id) != len(chunks):
        add_error(errors, "duplicate_chunk_id")
    if len(docs) != 43:
        add_error(errors, "document_count_not_forty_three")
    if len(chunks) != 479:
        add_error(errors, "chunk_count_not_479")

    top10_map = {}
    for row in top10_rows:
        key = (row["id"], row["mode"])
        if key in top10_map:
            add_error(errors, "duplicate_top10_group", row["id"])
        top10_map[key] = row
    expected_top10_keys = {
        (q["id"], mode)
        for q in queries
        for mode in ("dense", "bm25", "hybrid")}
    if set(top10_map) != expected_top10_keys:
        add_error(errors, "top10_group_set_mismatch")

    recomputed_counts = Counter()
    strict_records = {}
    for query_index, (query, record) in enumerate(
            zip(queries, records), start=1):
        qid = query["id"]
        if not record.get("audit_type"):
            add_error(errors, "record_audit_type_missing", qid)
        if record.get("audit_type") != "AI audit (non-human)":
            add_error(errors, "record_audit_type_unexpected", qid)
        if record.get("category") != query["category"]:
            add_error(errors, "category_mismatch", qid)
        if record.get("source_doc") != query["source_doc"]:
            add_error(errors, "source_doc_mismatch", qid)
        if record.get("support_quotes") != query["support_quotes"]:
            add_error(errors, "support_quotes_mismatch", qid)
        if record.get("gold_texts") != query["gold_texts"]:
            add_error(errors, "record_gold_texts_mismatch", qid)
        if record.get("gold_answer") != query["gold_answer"]:
            add_error(errors, "gold_answer_mismatch", qid)
        if record.get("audit_reason") != query["audit_reason"]:
            add_error(errors, "audit_reason_mismatch", qid)
        if record.get("neighbor_review") != query["neighbor_review"]:
            add_error(errors, "neighbor_review_mismatch", qid)
        if record.get("dedup_against_formal") != query[
                "dedup_against_formal"]:
            add_error(errors, "dedup_formal_mismatch", qid)
        if record.get("dedup_against_old") != query["dedup_against_old"]:
            add_error(errors, "dedup_old_mismatch", qid)
        if record.get("query_index") != query_index:
            add_error(errors, "record_query_index_mismatch", qid)
        if record["question"] != query["question"]:
            add_error(errors, "question_mismatch", qid)
        if record["gold_chunk_ids"] != query["gold_chunk_ids"]:
            add_error(errors, "record_gold_mismatch", qid)
        gold = query["gold_chunk_ids"]
        if len(gold) != len(set(gold)):
            add_error(errors, "duplicate_gold", qid)
        if {g["chunk_id"] for g in query["gold_texts"]} != set(gold):
            add_error(errors, "gold_text_id_mismatch", qid)
        if len(query["gold_texts"]) != len(gold):
            add_error(errors, "gold_texts_duplicate_or_extra", qid)
        if len({g["chunk_id"] for g in query["gold_texts"]}) != len(
                query["gold_texts"]):
            add_error(errors, "duplicate_gold_texts", qid)
        for gold_id in gold:
            gold_chunk = by_id.get(gold_id)
            if gold_chunk is None:
                add_error(errors, "gold_chunk_missing", qid)
            elif gold_chunk["doc"] != query["source_doc"]:
                add_error(errors, "gold_source_doc_mismatch", qid)
        for gt in query["gold_texts"]:
            actual = by_id.get(gt["chunk_id"])
            if actual is None:
                add_error(errors, "gold_chunk_missing", qid)
            elif actual["text"] != gt["text"]:
                add_error(errors, "gold_text_mismatch", qid)
        expected_hits = set()
        for quote in query["support_quotes"]:
            hits = {c["chunk_id"] for c in chunks
                    if c["doc"] == query["source_doc"] and quote in c["text"]}
            if not hits:
                add_error(errors, "support_quote_missing", qid)
            expected_hits |= hits
        if expected_hits != set(gold):
            add_error(errors, "gold_enumeration_mismatch", qid)

        dense = {}
        bm25 = {}
        for mode in ("dense", "bm25", "hybrid"):
            saved = top10_map.get((qid, mode))
            if saved is None:
                add_error(errors, "top10_group_missing", qid)
                continue
            if saved["gold_chunk_ids"] != gold:
                add_error(errors, "top10_gold_mismatch", qid)
            top10 = saved["top10"]
            if len(top10) != 10:
                add_error(errors, "top10_length_not_ten", qid)
            if [x["rank"] for x in top10] != list(range(1, 11)):
                add_error(errors, "top10_rank_mismatch", qid)
            if len({x["chunk_id"] for x in top10}) != len(top10):
                add_error(errors, "top10_duplicate_chunk", qid)
            scores = [x["score"] for x in top10]
            if not all(isinstance(s, (int, float)) and math.isfinite(s)
                       for s in scores):
                add_error(errors, "top10_score_not_finite", qid)
            if any(scores[i] < scores[i + 1] for i in range(9)):
                add_error(errors, "top10_score_not_descending", qid)
            for item in top10:
                chunk = by_id.get(item["chunk_id"])
                if chunk is None:
                    add_error(errors, "retrieved_chunk_missing", qid)
                else:
                    for key in ("text", "doc", "seq", "start", "end"):
                        if item[key] != chunk.get(key):
                            add_error(errors, "retrieved_payload_mismatch", qid)
                if item["is_gold"] != (item["chunk_id"] in set(gold)):
                    add_error(errors, "is_gold_mismatch", qid)
            saved_top3 = record["modes"][mode]["top3"]
            if saved_top3 != top10[:3]:
                add_error(errors, "top3_not_prefix_of_top10", qid)
            if record["modes"][mode].get("top10") != top10:
                add_error(errors, "record_top10_mismatch", qid)
            ids = [x["chunk_id"] for x in top10]
            metrics = {}
            for k in (1, 3, 5, 10):
                metrics["Hit@%d" % k] = hit_at_k(gold, ids, k)
                metrics["Recall@%d" % k] = recall_at_k(gold, ids, k)
            metrics["MRR@10"] = mrr_at_k(gold, ids, 10)
            for key, value in metrics.items():
                if record["modes"][mode][key] != value:
                    add_error(errors, "metric_recompute_mismatch", qid)
            if mode == "dense":
                dense = metrics
            elif mode == "bm25":
                bm25 = metrics

        if dense and bm25:
            winner = expected_winner(dense, bm25)
            if record["analysis"]["winner"] != winner:
                add_error(errors, "winner_recompute_mismatch", qid)
            if record["analysis"]["winner"] == "hybrid":
                add_error(errors, "hybrid_winner_forbidden", qid)
            if record["analysis"]["hybrid_observed"]["used_for_winner"]:
                add_error(errors, "hybrid_used_for_winner", qid)
            expected_diff = {
                "Recall@3": dense["Recall@3"] - bm25["Recall@3"],
                "MRR@10": dense["MRR@10"] - bm25["MRR@10"],
            }
            if record["analysis"].get("dense_minus_bm25") != expected_diff:
                add_error(errors, "dense_minus_bm25_mismatch", qid)
            recomputed_counts[winner] += 1
            strict_records[qid] = {
                "winner": winner,
                "dense": {"Recall@3": dense["Recall@3"],
                          "MRR@10": dense["MRR@10"]},
                "bm25": {"Recall@3": bm25["Recall@3"],
                         "MRR@10": bm25["MRR@10"]},
                "hybrid": {"Recall@3": record["modes"]["hybrid"]["Recall@3"],
                           "MRR@10": record["modes"]["hybrid"]["MRR@10"],
                           "used_for_winner": False},
            }

    formal_questions = {q["question"] for q in read_jsonl(FORMAL_QA)}
    old_questions = {q["question"] for q in read_jsonl(OLD_QUERIES)}
    for q in queries:
        if q["question"] in formal_questions:
            add_error(errors, "formal_question_collision", q["id"])
        if q["question"] in old_questions:
            add_error(errors, "old_question_collision", q["id"])

    persisted_counts = Counter(manifest["summary"]["strict_winner_counts"])
    if persisted_counts != recomputed_counts:
        add_error(errors, "summary_winner_count_mismatch")
    for mode in ("dense", "bm25", "hybrid"):
        if manifest["summary"][mode].get("queries") != 50:
            add_error(errors, "summary_query_count_mismatch", mode)
        if manifest["summary"][mode].get("mode") != mode:
            add_error(errors, "summary_mode_mismatch", mode)
        for metric in ("Hit@1", "Hit@3", "Hit@5", "Hit@10",
                       "Recall@1", "Recall@3", "Recall@5", "Recall@10",
                       "MRR@10"):
            observed = manifest["summary"][mode][metric]
            recomputed = sum(
                r["modes"][mode][metric] for r in records) / len(records)
            if observed != recomputed:
                add_error(errors, "summary_metric_recompute_mismatch", mode)
                break
    if manifest["summary"].get("target") != {"dense": 5, "bm25": 5}:
        add_error(errors, "summary_target_mismatch")
    if manifest["summary"].get("meets_dense_target") != (
            recomputed_counts["dense"] >= 5):
        add_error(errors, "summary_dense_target_mismatch")
    if manifest["summary"].get("meets_bm25_target") != (
            recomputed_counts["bm25"] >= 5):
        add_error(errors, "summary_bm25_target_mismatch")
    if manifest["summary"].get("meets_both_targets") != (
            recomputed_counts["dense"] >= 5
            and recomputed_counts["bm25"] >= 5):
        add_error(errors, "summary_both_target_mismatch")
    if manifest.get("exploratory_history") != freeze.get(
            "exploratory_history"):
        add_error(errors, "manifest_history_mismatch")
    if manifest.get("label") != freeze.get("label"):
        add_error(errors, "manifest_label_mismatch")
    if freeze.get("winner_rule") != (
            "dense vs BM25 only: unrounded Recall@3 first, then unrounded "
            "MRR@10; exact equality is tie"):
        add_error(errors, "winner_rule_mismatch")
    if freeze.get("hybrid_policy") != (
            "display only; excluded from winner determination"):
        add_error(errors, "hybrid_policy_mismatch")
    targets = {
        "dense": recomputed_counts["dense"] >= 5,
        "bm25": recomputed_counts["bm25"] >= 5,
        "both": recomputed_counts["dense"] >= 5
                and recomputed_counts["bm25"] >= 5,
    }

    summary = {}
    for mode in ("dense", "bm25", "hybrid"):
        summary[mode] = {}
        for metric in ("Hit@1", "Hit@3", "Hit@5", "Hit@10",
                       "Recall@1", "Recall@3", "Recall@5", "Recall@10",
                       "MRR@10"):
            summary[mode][metric] = sum(
                r["modes"][mode][metric] for r in records) / len(records)

    result = {
        "status": "PASS" if not errors else "FAIL",
        "formal_run_count": manifest.get("formal_retrieval_run_count"),
        "query_count": len(queries),
        "query_hash": query_hashes["queries"],
        "formal_qa_hash": query_hashes["formal"],
        "old_queries_hash": query_hashes["old"],
        "chunk_count": len(chunks),
        "gold_chunk_count_with_multiplicity": sum(
            len(q["gold_chunk_ids"]) for q in queries),
        "strict_winner_counts": dict(recomputed_counts),
        "targets": targets,
        "summary": summary,
        "per_query_strict": strict_records,
        "errors": errors,
        "warnings": warnings,
        "note": ("指标与胜者仅从已保存 top10 重算；hybrid 仅展示。"
                 "未达到目标时保留实际计数，不修改题面、gold 或阈值。"),
        "audit_type": "AI audit (non-human)",
    }
    write_audit(errors, warnings, result, strict_records)
    print("Status: %s" % result["status"])
    print("Strict winners: dense=%d bm25=%d tie=%d" % (
        recomputed_counts["dense"], recomputed_counts["bm25"],
        recomputed_counts["tie"]))
    print("Targets: dense=%s bm25=%s both=%s" % (
        targets["dense"], targets["bm25"], targets["both"]))
    if errors:
        print("Errors: %d (see %s)" % (len(errors), AUDIT_FILE))
        return 1
    return 0


def write_audit(errors, warnings, result, strict_records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = result or {
        "status": "FAIL",
        "errors": errors,
        "warnings": warnings,
        "note": "checker stopped before evidence recomputation",
    }
    AUDIT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
