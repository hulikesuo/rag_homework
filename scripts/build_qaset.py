"""数据标注工作台：生成 data/eval/qa_set.jsonl 的辅助脚手架。

流程（单人，自己本地手填）：
1. 运行本脚本 → 会在 results/_cache 生成 chunks 清单 + 每个文档的分块。
2. 你打开 scripts/qa_annotation.csv 逐行填写 question / level / gold_answer，
   并对着 `results/_cache/chunks.tsv` 里的原文，把支撑答案的几个 chunk_id
   填进 gold_chunk_ids（用 | 分隔）。
3. 用本脚本 `--convert` 把 csv 转成最终的 qa_set.jsonl。

用法：
  python -m scripts.build_qaset --chunk        # 先生成分块缓存
  python -m scripts.build_qaset --show <id>    # 查看某个 chunk 原文
  手工编辑 scripts/qa_annotation.csv
  python -m scripts.build_qaset --convert      # 生成 qa_set.jsonl
"""
import argparse
import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, chunker, loader, utils  # noqa: E402

ANNOTATION_CSV = os.path.join(config.PROJECT_ROOT, "scripts", "qa_annotation.csv")
CHUNKS_TSV = os.path.join(config.INDEX_CACHE, "chunks.tsv")

LEVELS = {"L1", "L2", "L3", "L4"}
# 每个等级需要至少 5 题
NEED = {"L1": 5, "L2": 5, "L3": 5, "L4": 5}


def _header():
    return ["id", "level", "question", "gold_answer", "gold_chunk_ids", "evidence_required"]


def step_chunk_csv():
    """读语料 → 分块(默认 fixed/512/128) → 写 chunks.tsv 供标注时对照。"""
    docs = loader.read_all_docs(config.CORPUS_DIR)
    print(f"读入 {len(docs)} 份文档")
    chunks = chunker.chunk_documents(docs)
    utils.save_ann_tsv(CHUNKS_TSV, [
        (c["chunk_id"], str(c["seq"]), c["doc"], c["text"]) for c in chunks])
    print(f"共生成本 {len(chunks)} 个 chunk，已写 {CHUNKS_TSV}")

    # 初始化标注 CSV（若不存在）
    if not os.path.exists(ANNOTATION_CSV):
        with open(ANNOTATION_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(_header())
        print(f"已生成空标注表 {ANNOTATION_CSV}，请打开填写。")


def step_show(chunk_id):
    if not os.path.exists(CHUNKS_TSV):
        step_chunk_csv()
    with open(CHUNKS_TSV, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == chunk_id:
                print(parts[-1])
                return
    print(f"未找到 {chunk_id}")


def step_convert(min_per_level=5, annotation_csv=None, output_jsonl=None, check_only=False):
    rows = []
    with open(annotation_csv or ANNOTATION_CSV, "r", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if not (r.get("question") or "").strip():
                continue
            gids = [x.strip() for x in (r.get("gold_chunk_ids") or "").split("|") if x.strip()]
            rows.append({
                "id": (r.get("id") or f"q{i:03d}").strip(),
                "level": (r.get("level") or "L1").strip(),
                "question": r["question"].strip(),
                "gold_answer": (r.get("gold_answer") or "").strip(),
                "gold_chunk_ids": gids,
                "evidence_required": (r.get("evidence_required") or "true").strip().lower() == "true",
            })
    # 先完整校验，失败时保留已有评测集。
    chunks = chunker.chunk_documents(loader.read_all_docs(config.CORPUS_DIR))
    valid_ids = {c["chunk_id"] for c in chunks}
    errors, seen, counts = [], set(), {}
    for r in rows:
        qid = r["id"]
        if qid in seen:
            errors.append(f"重复题号: {qid}")
        seen.add(qid)
        if r["level"] not in LEVELS:
            errors.append(f"{qid}: 非法等级 {r['level']}")
        if not r["gold_answer"] or not r["gold_chunk_ids"]:
            errors.append(f"{qid}: 答案或证据为空")
        if len(set(r["gold_chunk_ids"])) != len(r["gold_chunk_ids"]):
            errors.append(f"{qid}: 重复证据 ID")
        for cid in r["gold_chunk_ids"]:
            if cid not in valid_ids:
                errors.append(f"{qid}: 不存在的证据 {cid}")
        counts[r["level"]] = counts.get(r["level"], 0) + 1
    for lv in sorted(LEVELS):
        if counts.get(lv, 0) < min_per_level:
            errors.append(f"{lv}: 不足 {min_per_level} 题")
    if len(rows) < 20:
        errors.append("测试集不足20题")
    if errors:
        raise ValueError("标注校验失败：\n" + "\n".join(errors))
    print("各等级数量：", counts)
    if check_only:
        print(f"校验通过（共 {len(rows)} 题），未写文件")
        return rows
    target = output_jsonl or config.EVAL_JSONL
    utils.save_jsonl(target, rows)
    print(f"已写 {target}（共 {len(rows)} 题）")
    return rows


def main():
    parser = argparse.ArgumentParser(description="qa 集标注工作台")
    parser.add_argument("--chunk", action="store_true", help="分块并生成标注表")
    parser.add_argument("--show", metavar="CHUNK_ID", help="查看某个 chunk 原文")
    parser.add_argument("--convert", action="store_true", help="csv → qa_set.jsonl")
    parser.add_argument("--annotation", help="指定标注 CSV，默认 scripts/qa_annotation.csv")
    parser.add_argument("--output", help="指定输出 JSONL，默认配置中的 EVAL_JSONL")
    parser.add_argument("--check", action="store_true", help="仅校验标注，不写文件")
    args = parser.parse_args()

    if args.chunk:
        step_chunk_csv()
    elif args.show:
        step_show(args.show)
    elif args.convert or args.check:
        step_convert(annotation_csv=args.annotation, output_jsonl=args.output,
                     check_only=args.check)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()