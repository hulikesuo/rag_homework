"""T4 supplement: diagnostic dense/sparse comparison queries.

Design: construct a separate query set to test where dense should win
(paraphrase/semantic) vs where sparse should win (exact terminology).
Does NOT modify the 30-question main eval or any source modules.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, loader, pipeline, utils, evaluate

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "t4_supplement_queries.jsonl")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "repair_outputs", "revised_full_20261022")
os.makedirs(OUT_DIR, exist_ok=True)

MODES = ["bm25", "dense", "hybrid"]
KS = [1, 3, 5, 10]


def _winner(modes):
    """Tiebreak: Recall@3 then MRR@10 (same as main T4 rule)."""
    best = None
    best_recall = -1
    best_mrr = -1
    for mode in MODES:
        r3 = modes[mode].get("Recall@3", -1)
        m10 = modes[mode].get("MRR@10", -1)
        if r3 > best_recall or (r3 == best_recall and m10 > best_mrr):
            best = mode
            best_recall = r3
            best_mrr = m10
    return best


def run():
    queries = utils.load_jsonl(QUERIES_FILE)
    print("T4 supplement: %d diagnostic queries" % len(queries))

    docs = loader.read_all_docs(config.CORPUS_DIR)
    pipe = pipeline.RAGPipeline(docs)

    chunk_map = {chunk["chunk_id"]: chunk for chunk in pipe.chunks}
    records = []

    for qi, q in enumerate(queries, 1):
        gold_id = q.get("gold_chunk_id", "")
        gold_ids = [gold_id] if gold_id else []
        missing = [cid for cid in gold_ids if cid not in chunk_map]
        if missing:
            print("  [%d/%d] %s WARN gold not found: %s" % (qi, len(queries), q["id"], missing))

        record = {
            "query_index": qi,
            "id": q["id"],
            "desc": q.get("desc", ""),
            "question": q["question"],
            "gold_chunk_id": gold_id,
            "expect_winner": q.get("expect_winner", ""),
            "edge_case": q.get("edge_case", ""),
            "modes": {},
        }

        for mode in MODES:
            hits = list(pipe.retrieve(q["question"], k=10, mode=mode))[:10]
            ids = [chunk["chunk_id"] for chunk, _ in hits]
            metrics = {}
            for k in KS:
                metrics["Hit@%d" % k] = evaluate.hit_at_k(gold_ids, ids, k)
                metrics["Recall@%d" % k] = evaluate.recall_at_k(gold_ids, ids, k)
            metrics["MRR@10"] = evaluate.mrr_at_k(gold_ids, ids, 10)
            metrics["top3"] = [
                {"rank": rank, "chunk_id": chunk["chunk_id"],
                 "doc": chunk.get("doc", ""), "text": chunk["text"][:600],
                 "score": float(score), "is_gold": chunk["chunk_id"] in gold_ids}
                for rank, (chunk, score) in enumerate(hits[:3], 1)
            ]
            record["modes"][mode] = metrics

        record["actual_winner"] = _winner(record["modes"])
        records.append(record)

        if qi % 5 == 0 or qi == len(queries):
            print("  [%d/%d] %s winner=%s" % (qi, len(queries), q["id"], record["actual_winner"]))

    # Save results
    out_json = os.path.join(OUT_DIR, "t4_supplement_per_query.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("\nSaved per-query: %s" % out_json)

    # Summary CSV
    summary_rows = []
    for mode in MODES:
        row = {"retriever": mode}
        for metric in ["Recall@1", "Recall@3", "Recall@5", "Recall@10",
                        "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"]:
            row[metric] = round(sum(r["modes"][mode].get(metric, 0)
                                    for r in records) / len(records), 4)
        summary_rows.append(row)

    out_csv = os.path.join(OUT_DIR, "t4_supplement_summary.csv")
    cols = ["retriever", "Recall@1", "Recall@3", "Recall@5", "Recall@10",
            "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)
    print("Saved summary: %s" % out_csv)

    # Win counts
    wins = {}
    for r in records:
        w = r["actual_winner"]
        wins[w] = wins.get(w, 0) + 1

    print("\n--- Supplement Winner Counts ---")
    for mode in sorted(wins):
        n = wins[mode]
        print("  %s: %d wins" % (mode, n))

    # Detail CSV
    detail_rows = []
    for r in records:
        for mode in MODES:
            d = {
                "id": r["id"], "desc": r["desc"],
                "question": r["question"], "mode": mode,
                "expect_winner": r["expect_winner"],
                "Recall@3": r["modes"][mode]["Recall@3"],
                "Recall@5": r["modes"][mode]["Recall@5"],
                "MRR@10": r["modes"][mode]["MRR@10"],
                "actual_winner": r["actual_winner"],
            }
            top3_texts = [e["text"][:200].replace("\n", " ")
                          for e in r["modes"][mode]["top3"]]
            d["top3_texts"] = " ||| ".join(top3_texts)
            detail_rows.append(d)

    out_det = os.path.join(OUT_DIR, "t4_supplement_detail.csv")
    with open(out_det, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)
    print("Saved detail: %s" % out_det)

    # Markdown report
    md_lines = []
    md_lines.append("# T4 补充诊断结果")
    md_lines.append("")
    n_dense_exp = sum(1 for r in records if r["expect_winner"] == "dense")
    n_sparse_exp = sum(1 for r in records if r["expect_winner"] == "sparse")
    n_either = sum(1 for r in records if r["expect_winner"] == "either")
    md_lines.append("**总查询数**: %d  |  dense预期胜: %d  |  sparse预期胜: %d  |  either: %d" % (
        len(records), n_dense_exp, n_sparse_exp, n_either))
    md_lines.append("")
    md_lines.append("## 汇总指标")
    md_lines.append("")
    md_lines.append("| retriever | Recall@3 | Recall@5 | Recall@10 | MRR@10 |")
    md_lines.append("|-----------|----------|----------|-----------|--------|")
    for row in summary_rows:
        md_lines.append("| %s | %.4f | %.4f | %.4f | %.4f |" % (
            row["retriever"], row["Recall@3"], row["Recall@5"],
            row["Recall@10"], row["MRR@10"]))
    md_lines.append("")
    md_lines.append("## 胜负统计")
    md_lines.append("")
    md_lines.append("| Retriever | 胜次 |")
    md_lines.append("|-----------|------|")
    for mode in sorted(wins):
        md_lines.append("| %s | %d |" % (mode, wins[mode]))
    md_lines.append("")
    md_lines.append("## 逐查询结果")
    md_lines.append("")
    for r in records:
        md_lines.append("### %s -- %s (预期: %s, 实际: %s)" % (
            r["id"], r["desc"], r["expect_winner"], r["actual_winner"]))
        md_lines.append("")
        md_lines.append("**问题**: %s" % r["question"])
        md_lines.append("")
        for mode in MODES:
            m = r["modes"][mode]
            md_lines.append("**%s** Recall@3=%.4f Recall@5=%.4f MRR@10=%.4f" % (
                mode, m["Recall@3"], m["Recall@5"], m["MRR@10"]))
            for e in m["top3"]:
                mark = " [GOLD]" if e["is_gold"] else ""
                md_lines.append("- #%d%s `%s` (%.4f) %s" % (
                    e["rank"], mark, e["chunk_id"], e["score"],
                    e["text"][:150]))
            md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    out_md = os.path.join(OUT_DIR, "t4_supplement_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("Saved report: %s" % out_md)

    return records


if __name__ == "__main__":
    os.environ["RAG_OFFLINE"] = "1"
    import time; t0 = time.time()
    records = run()
    wins = {}
    for r in records:
        w = r["actual_winner"]
        wins[w] = wins.get(w, 0) + 1
    print("\n\n========================================")
    print("T4 supplement DONE  (%.1fs)" % (time.time() - t0))
    print("  total queries: %d" % len(records))
    print("  dense wins: %d" % wins.get("dense", 0))
    print("  bm25 wins: %d" % wins.get("bm25", 0))
    print("  hybrid wins: %d" % wins.get("hybrid", 0))
    d_gap = max(0, 5 - wins.get("dense", 0))
    s_gap = max(0, 5 - wins.get("bm25", 0))
    print("  gap to 5 dense + 5 sparse:")
    print("    dense: %s%d more" % ("" if d_gap else "OK (", d_gap))
    print("    sparse: %s%d more" % ("" if s_gap else "OK (", s_gap))
    print("========================================")
