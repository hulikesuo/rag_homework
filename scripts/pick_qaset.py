"""从 LLM 初标草稿里精选最终测试集，输出可直接转 qa_set 的 scripts/qa_annotation.csv。

规则：
  1. 去重（同 gold_chunk_ids 且题干高度相似只留一条）。
  2. 按 level 配额挑选：L1:8 L2:8 L3:7 L4:7 = 30 题。
  3. 尽量覆盖 4 类文档（培养方案/课程/选课管理/保研推免）。
  4. gold_chunk_ids 保留（硬题的簇段做评测支撑）。

注意：这是"机器初选"，最终仍建议人工抽查每条 gold 是否真支撑答案。

用法：
  python -m scripts.pick_qaset
"""
import csv
import os
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

DRAFT = os.path.join(config.PROJECT_ROOT, "scripts", "qa_annotation_draft.csv")
OUT = os.path.join(config.PROJECT_ROOT, "scripts", "qa_annotation.csv")

QUOTA = {"L1": 8, "L2": 8, "L3": 7, "L4": 7}


def _norm(q):
    """题干归一化，用于近重复判断。"""
    return "".join(ch for ch in q if ch.isalnum())


def main():
    rows = list(csv.DictReader(open(DRAFT, encoding="utf-8-sig")))

    # ---- 去重 ----
    seen_gold, seen_q, uniq = set(), set(), []
    for r in rows:
        g = r["gold_chunk_ids"].strip()
        q = _norm(r["question"])[:40]
        if g in seen_gold or q in seen_q:
            continue
        seen_gold.add(g)
        seen_q.add(q)
        uniq.append(r)
    print(f"去重后 {len(uniq)} 题（原 {len(rows)}）")

    # ---- 按 level 配额 + doctype 下限（每类至少 DT_MIN 题） ----
    # 约束：先满足四类的下限，再按 level 配额补满。
    DT_MIN = 4
    chosen = []
    used = set()

    def take(r):
        chosen.append(r)
        used.add(r["id"])

    # 阶段1：每个 doctype 先保底 DT_MIN 题（优先挑 level 稀缺的，让分布更好）
    by_dt = defaultdict(list)
    for r in uniq:
        by_dt[r["doctype"]].append(r)
    for dt, pool in by_dt.items():
        # 该 doctype 内按 level 稀缺度排序（L4/L3 优先，提升难度覆盖）
        lv_rank = {"L4": 0, "L3": 1, "L2": 2, "L1": 3}
        pool_sorted = sorted(pool, key=lambda r: lv_rank.get(r["level"], 9))
        for r in pool_sorted[:DT_MIN]:
            take(r)

    # 阶段2：按 level 配额补足（跳过已选）
    for lv, need in QUOTA.items():
        have = sum(1 for r in chosen if r["level"] == lv)
        if have >= need:
            continue
        for r in uniq:
            if have >= need:
                break
            if r["level"] != lv or r["id"] in used:
                continue
            take(r)
            have += 1

    # 若总数超过 30，按 level 超额削减（保留 doctype 均衡，削 L1 富余）
    if len(chosen) > 30:
        lv_count = defaultdict(int)
        for r in chosen:
            lv_count[r["level"]] += 1
        kept = []
        for r in chosen:
            if lv_count[r["level"]] > QUOTA.get(r["level"], 0):
                lv_count[r["level"]] -= 1
                continue
            kept.append(r)
        chosen = kept[:30]

    # ---- 写最终标注表 ----
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "level", "question", "gold_answer", "gold_chunk_ids",
                    "evidence_required"])
        for i, r in enumerate(chosen, 1):
            w.writerow([f"q{i:02d}", r["level"], r["question"], r["gold_answer"],
                        r["gold_chunk_ids"], ""])

    from collections import Counter
    print(f"已写 {OUT}，共 {len(chosen)} 题")
    print("level:", dict(Counter(r["level"] for r in chosen)))
    print("doctype:", dict(Counter(r["doctype"] for r in chosen)))


if __name__ == "__main__":
    main()