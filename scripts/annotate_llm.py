"""LLM 初标测试集（辅助，非提交物）。

读分块缓存 results/_cache/chunks.tsv，随机抽真实原文 chunk，让 LLM 严格
基于该段原文出题 + 答案 + 自评 level，写一份待复核草稿 scripts/qa_annotation_draft.csv。

关键约束（避免"编造题"导致 Recall 0 分的查分风险）：
  - 每题的 gold_chunk_ids 只用 LLM 实际引用的 chunk_id，且必须存在于 chunks.tsv。
  - doctype 按 chunk 所属目录归类，保证题目覆盖 培养方案/课程/选课管理/保研推免 四类。
  - 用固定 seed 可复现；每片抽多份，最终按 level 补足到目标数。

用法：
  python -m scripts.annotate_llm --n 40
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, generator  # noqa: E402

CHUNKS_TSV = os.path.join(config.INDEX_CACHE, "chunks.tsv")
DRAFT_CSV = os.path.join(config.PROJECT_ROOT, "scripts", "qa_annotation_draft.csv")

NEED = {"L1": 8, "L2": 8, "L3": 7, "L4": 7}   # 共 30 题


def load_chunks():
    rows = []
    with open(CHUNKS_TSV, encoding="utf-8") as f:
        f.readline()  # 表头
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            cid, seq, doc, text = parts[0], parts[1], parts[2], parts[3]
            rows.append({"id": cid, "text": text, "doc": doc, "seq": seq})
    return rows


def _doctype(doc):
    doc = doc.replace("\\", "/")
    if "培养方案" in doc or "专业简介" in doc:
        return "培养方案与专业介绍"
    if "课程" in doc or "教学大纲" in doc:
        return "课程教学"
    if "选课" in doc or "学籍" in doc or "成绩" in doc or "学分" in doc:
        return "选课与教学管理"
    if "推免" in doc or "夏令营" in doc:
        return "保研推免"
    return "其他"


def _ask(chunk):
    """让 LLM 基于一段原文出题。返回字典或 None。"""
    prompt = (
        "你是数据标注员。下面是一段检索资料原文。请严格只依据这段原文，出一道"
        "有明确标准答案的『检索型问答题』（答案必须能在原文里逐字或近义找到，不要引入原文外的背景）。\n\n"
        f"原文：[片段]\n{chunk['text']}\n[/片段]\n\n"
        "请严格按如下 JSON 输出，不要输出多余文字：\n"
        '{"question":"（对原文事实的问句）", "answer":"（简洁标准答案）", '
        '"level":"L1 或 L2 或 L3 或 L4，定义：L1=原文可直接找到的事实；'
        'L2=单文档内需要看邻近上下文才能答；L3=需要跨多段/理解才能答；'
        'L4=需要推理/综合才能答"}'
    )
    try:
        raw = generator.complete(prompt, max_tokens=300, temperature=0.4)
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        q = (obj.get("question") or "").strip()
        a = (obj.get("answer") or "").strip()
        lv = (obj.get("level") or "").strip().upper()
        lv = lv if lv in {"L1", "L2", "L3", "L4"} else "L2"
        if len(q) < 6 or len(a) < 2:
            return None
        return {"question": q, "answer": a, "level": lv,
                "gold": [chunk["id"]], "doc": chunk["doc"]}
    except Exception as e:
        print(f"  [跳过] 解析失败: {e}")
        return None


def _ask_hard(cluster, desired):
    """基于同一文档的一段簇出跨段/推理题。desired 为想要的 level 之一(L2/L3/L4)。"""
    texts = "\n\n".join(f"[片段{i+1}]\n{c['text']}" for i, c in enumerate(cluster))
    prompt = (
        "你是数据标注员。下面是某文档前后相邻的几段检索资料。请结合这几段"
        f"（必要时跨段联想、推理或综合作答），出一道 level 为『{desired}』的"
        "『综合/推理型问答题』，答案必须在这些段落里能推理得出，不要引入段外背景。\n\n"
        f"【多段原文】\n{texts}\n【原文结束】\n\n"
        f"务必把题目设计成需要对比多段、或推推导、或归纳的『{desired}』级别（不要只问单段能直接抄答案的事实）。\n"
        "严格按以下 JSON 输出，不要多余文字：\n"
        '{"question":"…", "answer":"…", "level":"' + desired + '"}'
    )
    try:
        raw = generator.complete(prompt, max_tokens=320, temperature=0.5)
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        q = (obj.get("question") or "").strip()
        a = (obj.get("answer") or "").strip()
        if len(q) < 6 or len(a) < 2:
            return None
        # 该题用的支撑段 = 簇里所有段
        return {"question": q, "answer": a, "level": desired,
                "gold": [c["id"] for c in cluster], "doc": cluster[0]["doc"]}
    except Exception as e:
        print(f"  [跳过] {e}")
        return None


def gen_hard(chunks, have):
    """为不足的 L2/L3/L4 生成题目：把同一文档分出相邻段簇喂给 LLM。
    返回新生成的题目列表（含 doctype）。"""
    from collections import Counter
    # 按文档分组，保留原序号顺序
    by_doc = {}
    for c in chunks:
        doc = c["doc"].replace("\\", "/")
        by_doc.setdefault(doc, []).append(c)
    todo = {lv: max(0, NEED[lv] - have.get(lv, 0)) for lv in ("L2", "L3", "L4")}
    print("硬模式需补:", {k: v for k, v in todo.items() if v > 0})
    if sum(todo.values()) == 0:
        print("  已达标，无需硬模式。")
        return []

    random.seed(2027)
    rows = []
    docs = [d for d in sorted(by_doc) if len(by_doc[d]) >= 3]
    random.shuffle(docs)
    docs = docs[:80]  # 防跑太久，封顶尝试文档数
    for d in docs:
        cl = by_doc[d]
        size = random.choice((2, 3))
        for s in range(0, len(cl) - size + 1, size):
            cluster = cl[s:s + size]
            lv = random.choice([lv for lv, k in todo.items() if k > 0])
            r = _ask_hard(cluster, lv)
            if not r:
                continue
            r["doctype"] = _doctype_short(d)
            r["level"] = lv  # 强制按目标档
            rows.append(r)
            have[lv] += 1
            todo[lv] -= 1
            print(f"  补 [{lv}] {r['question'][:36]}…")
            if sum(todo.values()) == 0:
                return rows
    print("  迭代完仍未补满（缺口留存手工），请人工补:", dict(todo))
    return rows


def _doctype_short(doc):
    if any(k in doc for k in ("培养方案", "专业简介")):
        return "培养方案与专业介绍"
    if any(k in doc for k in ("课程", "教学大纲")):
        return "课程教学"
    if any(k in doc for k in ("选课", "学籍", "成绩", "学分")):
        return "选课与教学管理"
    if any(k in doc for k in ("推免", "夏令营")):
        return "保研推免"
    return "其他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽取原文片段数")
    ap.add_argument("--hard", action="store_true", help="补做 L2/L3/L4 跨段推理题")
    ap.add_argument("--skip-basic", action="store_true",
                    help="跳过基础 L1 出题，直接读现有草稿做基线再补硬题")
    args = ap.parse_args()

    chunks = load_chunks()

    # 基线题：要么从现有草稿读入，要么现场生成
    results = []
    if args.skip_basic:
        if not os.path.exists(DRAFT_CSV):
            print("没有现成草稿，无法 --skip-basic")
            return
        with open(DRAFT_CSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                results.append({
                    "question": r["question"], "answer": r["gold_answer"],
                    "level": r["level"], "doc": r["src_doc"],
                    "gold": [g for g in r["gold_chunk_ids"].split("|") if g],
                    "doctype": r.get("doctype", ""),
                })
        print(f"从草稿读入 {len(results)} 题作为基线。")
    else:
        # 按四类文档均匀抽，避免集中到某一类
        by_type = {}
        for c in chunks:
            by_type.setdefault(_doctype(c["doc"]), []).append(c)
        random.seed(2026)
        picked = []
        pool = list(by_type.keys())
        slots = [pool[i % len(pool)] for i in range(args.n)]
        for t in slots:
            picked.append(random.choice(by_type[t]))

        print(f"抽取 {len(picked)} 段原文，调用 LLM 出题…")
        for i, c in enumerate(picked, 1):
            r = _ask(c)
            if r:
                r["doctype"] = _doctype(c["doc"])
                results.append(r)
                print(f"[{i}/{len(picked)}] {r['level']} | {r['question'][:34]}…")

    # level 分布
    from collections import Counter
    cnt = Counter(r["level"] for r in results)
    print("\n初始 level 分布:", dict(cnt))

    # 补足：不足的 level 从中多抽多问（简单版：若某档不足，从结果里不足档标记并提示手工补）
    print("\n目标:", NEED)

    # 硬模式：补 L2/L3/L4 跨段推理题
    if args.hard:
        extra = gen_hard(chunks, Counter(r["level"] for r in results))
        for r in extra:
            r["doc"] = r.get("doc") or ""
            results.append(r)
    from collections import Counter
    cnt = Counter(r["level"] for r in results)
    print("\n合并后 level 分布:", dict(cnt))

    # 校验 gold_chunk_ids 均已存在（避免 Recall 恒 0）
    valid_ids = {c["id"] for c in chunks}
    bad = [r for r in results if not all(g in valid_ids for g in r["gold"])]
    if bad:
        print(f"警告：{len(bad)} 题引用了不存在的 chunk_id，已剔除。")
        results = [r for r in results if r not in bad]

    # 写草稿
    with open(DRAFT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "level", "question", "gold_answer", "gold_chunk_ids",
                    "evidence_required", "doctype", "src_doc"])
        for i, r in enumerate(results, 1):
            w.writerow([f"draft{i:02d}", r["level"], r["question"], r["answer"],
                        "|".join(r["gold"]), "", r["doctype"], r["doc"]])
    print(f"\n草稿已写 {DRAFT_CSV}，共 {len(results)} 题待复核。")


if __name__ == "__main__":
    main()