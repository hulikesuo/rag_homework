"""T4：每题每路一次 top-10，报告课件 Hit、分片 Recall 和 MRR。

Recall@k 保留旧的分片召回口径；Hit@k 为课件的至少命中一个 gold。
胜例先比未舍入的分片 Recall@3，再比 MRR@10；相同为平局。
每类按评测集原顺序最多选 5 题，不修改 gold，也不以检索分数跨路比较。
"""
import json
import os

from experiments.common import load_env, save_csv
from src import config, evaluate, pipeline

MODES = ("bm25", "dense", "hybrid")
KS = (1, 3, 5, 10)
CASE_LIMIT = 5
CASE_RULE = (
    "dense 与 bm25 先比较未舍入的分片 Recall@3，再比较 MRR@10；"
    "二者均相同为 tie。各类按评测集原顺序最多取 5 题；hybrid 仅展示，"
    "不参与胜负判定；不同检索器的原始分数不作跨路比较。"
)


def _winner(modes):
    dense = (modes["dense"]["Recall@3"], modes["dense"]["MRR@10"])
    sparse = (modes["bm25"]["Recall@3"], modes["bm25"]["MRR@10"])
    return "dense" if dense > sparse else "sparse" if sparse > dense else "tie"


def _collect(pipe, qas):
    """缓存一次 top-10 的指标和完整 top-3；后续输出不再检索。"""
    chunk_map = {chunk["chunk_id"]: chunk for chunk in pipe.chunks}
    records = []
    for index, q in enumerate(qas, 1):
        gold_ids = list(q.get("gold_chunk_ids", []))
        missing = [cid for cid in gold_ids if cid not in chunk_map]
        if missing:
            raise ValueError(f"第 {index} 题 gold 不在基准分块中：{missing}")
        record = {
            "query_index": index,
            "id": q.get("id", index),
            "question": q["question"],
            "gold_chunk_ids": gold_ids,
            "gold_answer": q.get("gold_answer", ""),
            "gold_chunks": [dict(chunk_map[cid]) for cid in gold_ids],
            "modes": {},
        }
        for mode in MODES:
            hits = list(pipe.retrieve(q["question"], k=10, mode=mode))[:10]
            ids = [chunk["chunk_id"] for chunk, _ in hits]
            metrics = {}
            for k in KS:
                metrics[f"Hit@{k}"] = evaluate.hit_at_k(gold_ids, ids, k)
                metrics[f"Recall@{k}"] = evaluate.recall_at_k(gold_ids, ids, k)
            metrics["MRR@10"] = evaluate.mrr_at_k(gold_ids, ids, 10)
            metrics["top10_ids"] = ids
            metrics["top3"] = [
                {"rank": rank, "chunk_id": chunk["chunk_id"],
                 "doc": chunk.get("doc", ""), "text": chunk["text"],
                 "score": float(score), "is_gold": chunk["chunk_id"] in gold_ids}
                for rank, (chunk, score) in enumerate(hits[:3], 1)
            ]
            record["modes"][mode] = metrics
        record["winner"] = _winner(record["modes"])
        records.append(record)
    return records


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _save_json(name, value):
    path = os.path.join(config.RESULTS_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(f"[save] {path}")


def _detail_rows(records):
    """每题每路一行；JSON 单元格保留文本、分数和完整 gold。"""
    rows = []
    for record in records:
        for mode in MODES:
            metrics = record["modes"][mode]
            row = {key: record[key] for key in
                   ("query_index", "id", "question", "gold_answer", "winner")}
            row.update(
                retriever=mode,
                gold_chunk_ids=_json(record["gold_chunk_ids"]),
                gold_chunks=_json(record["gold_chunks"]),
                top10_ids=_json(metrics["top10_ids"]),
                top3=_json(metrics["top3"]),
            )
            row.update({key: metrics[key] for key in METRIC_FIELDS})
            rows.append(row)
    return rows


METRIC_FIELDS = ([f"Hit@{k}" for k in KS]
                 + [f"Recall@{k}" for k in KS] + ["MRR@10"])
DETAIL_FIELDS = ["query_index", "id", "question", "gold_answer", "winner",
                 "retriever", "gold_chunk_ids", "gold_chunks", "top10_ids",
                 "top3"] + METRIC_FIELDS


def _save_cases(records):
    selected = {}
    counts = {}
    for winner in ("dense", "sparse"):
        candidates = [r for r in records if r["winner"] == winner]
        selected[winner] = candidates[:CASE_LIMIT]
        counts[winner] = {
            "available": len(candidates), "selected": len(selected[winner]),
            "requested": CASE_LIMIT,
            "shortfall": max(0, CASE_LIMIT - len(candidates)),
        }
        label = "稠密赢" if winner == "dense" else "稀疏赢"
        print(f"{label}：可用 {len(candidates)} 例，选择 {len(selected[winner])} 例"
              f"（目标 {CASE_LIMIT} 例，不足 {counts[winner]['shortfall']} 例）")
    payload = {
        "selection_rule": CASE_RULE, "counts": counts,
        "tie_count": sum(r["winner"] == "tie" for r in records),
        "cases": selected,
    }
    _save_json("t4_cases.json", payload)

    # 汇总行保证没有胜例时 CSV 仍明确报告可用数、选择数和缺额。
    csv_rows = []
    for winner in ("dense", "sparse"):
        csv_rows.append(dict(record_type="summary", category=winner,
                             selection_rule=CASE_RULE, **counts[winner]))
        for row in _detail_rows(selected[winner]):
            csv_rows.append(dict(record_type="case", category=winner, **row))
    save_csv("t4_cases.csv", csv_rows,
             ["record_type", "category", "selection_rule", "available",
              "selected", "requested", "shortfall"] + DETAIL_FIELDS)

    parts = ["# T4 检索胜例", "", CASE_RULE, "",
             f"平局：{payload['tie_count']} 题。", ""]
    for winner, label in (("dense", "稠密赢"), ("sparse", "稀疏赢")):
        count = counts[winner]
        parts.extend([f"## {label}", "",
                      f"可用 {count['available']} 例，选择 {count['selected']} 例，"
                      f"目标 {CASE_LIMIT} 例，不足 {count['shortfall']} 例。", ""])
        for record in selected[winner]:
            # 完整 JSON 含三路 top-3 和 gold；动态围栏避免原文中的反引号截断。
            text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
            fence = "`" * max(3, _longest_backticks(text) + 1)
            parts.extend([f"### 第 {record['query_index']} 题", "",
                          fence + "json", text, fence, ""])
    path = os.path.join(config.RESULTS_DIR, "t4_cases.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[save] {path}")


def _longest_backticks(text):
    import re
    return max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)


def run():
    print(f"T4 胜例标准：{CASE_RULE}")
    env = load_env()
    qas = env["qas"]
    if not qas:
        raise ValueError("T4 评测集为空，无法计算平均检索指标")
    pipe = pipeline.RAGPipeline(env["docs"])
    records = _collect(pipe, qas)
    rows = []
    for mode in MODES:
        row = {"retriever": mode}
        for metric in METRIC_FIELDS:
            row[metric] = round(sum(r["modes"][mode][metric] for r in records)
                                / len(records), 4)
        rows.append(row)
    # 旧 Recall 列名和分片口径保留；新增显式课件 Hit 列。
    save_csv("t4_retriever.csv", rows, ["retriever"] + METRIC_FIELDS)
    _save_json("t4_per_query.json", records)
    save_csv("t4_per_query.csv", _detail_rows(records), DETAIL_FIELDS)
    _save_cases(records)
    print("\n三种检索器指标（Hit 为课件命中率，Recall 为分片召回率）：")
    for row in rows:
        print(row)
    return rows


def print_top3_cases(pipe, qas, n=10):
    """兼容人工检查入口；每题每路同样只检索一次 top-10。"""
    for record in _collect(pipe, qas[:n]):
        print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
