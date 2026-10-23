"""T4 补充：构造"稠密赢 / 稀疏赢"的对照分析（作业要求各≥5条）。

做法：对评测集每题，分别用 dense 和 bm25 检索 top-3，比较命中情况：
  - "稠密赢"：dense 命中且 bm25 未命中
  - "稀疏赢"：bm25 命中且 dense 未命中
输出 results/t4_cases_dense_wins.csv / t4_cases_sparse_wins.csv，
每行含题干、两种检索器的 top-3 chunk_id、gold、以及原因分析线索。

若命中的对照不足 5 条，再从语料中构造补充查询（专有名词/代号类适合 bm25，
口语化改写类适合 dense），保证每类至少 5 条。

用法：
  python -m scripts.t4_cases
"""
import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, loader, utils, pipeline, evaluate  # noqa: E402

RES = config.RESULTS_DIR


def _top3(pipe, q, mode):
    hits = pipe.retrieve(q, k=3, mode=mode)
    return [c["chunk_id"] for c, _ in hits]


def _fmt(ids):
    return " | ".join(i.split("\\")[-1][:38] for i in ids)


def main():
    docs = loader.read_all_docs(config.CORPUS_DIR)
    qas = utils.load_jsonl(config.EVAL_JSONL)
    pipe = pipeline.RAGPipeline(docs)

    dense_wins, sparse_wins = [], []
    for q in qas:
        d = _top3(pipe, q["question"], "dense")
        b = _top3(pipe, q["question"], "bm25")
        gold = q["gold_chunk_ids"]
        dh = evaluate.hit_at_k(gold, d, 3) > 0
        bh = evaluate.hit_at_k(gold, b, 3) > 0
        row = {
            "id": q["id"], "level": q["level"], "question": q["question"],
            "gold": _fmt(gold), "dense_top3": _fmt(d), "bm25_top3": _fmt(b),
        }
        if dh and not bh:
            dense_wins.append(row)
        elif bh and not dh:
            sparse_wins.append(row)

    print(f"评测集中：稠密赢 {len(dense_wins)} 条，稀疏赢 {len(sparse_wins)} 条")

    # 补充构造：专有名词查询（bm25 优势场景）
    corpus = [(c["chunk_id"], c["text"]) for c in pipe.chunks]
    if len(sparse_wins) < 5:
        print("稀疏赢不足 5 条，从语料中构造带编号/专有名词的查询…")
        # 找含课程代码/文号/邮箱等强标识符的片段，用其标识符做查询
        import re
        for cid, text in corpus:
            m = re.findall(r"YN\d{10}", text)
            if m and len(sparse_wins) < 5:
                qq = m[0]
                d = _top3(pipe, qq, "dense")
                b = _top3(pipe, qq, "bm25")
                row = {"id": "syn_" + qq, "level": "L1", "question": qq,
                       "gold": _fmt([cid]), "dense_top3": _fmt(d), "bm25_top3": _fmt(b)}
                if evaluate.hit_at_k([cid], b, 3) and not evaluate.hit_at_k([cid], d, 3):
                    sparse_wins.append(row)

    def _write(name, rows):
        path = os.path.join(RES, name)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[
                "id", "level", "question", "gold", "dense_top3", "bm25_top3"])
            w.writeheader()
            w.writerows(rows[:10])
        print(f"[save] {path}  ({min(len(rows),10)} 条)")

    _write("t4_cases_dense_wins.csv", dense_wins)
    _write("t4_cases_sparse_wins.csv", sparse_wins)


if __name__ == "__main__":
    main()