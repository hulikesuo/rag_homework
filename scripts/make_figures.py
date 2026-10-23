"""从实验 CSV 生成报告图，不运行检索、模型或生成 API。

默认使用 config.RESULTS_DIR，也可指定修复结果目录：
  python -m scripts.make_figures --results-dir repair_outputs/local_results

T2 展示每条 method/size/overlap 配置；T3 仅在存在策略对比时出图，
alpha 扫描独立处理；T5 仅展示可用的检索或阶段耗时代理，不代表答案质量。
T4/T7 的 Recall 为 gold 分片召回率，Hit 为题目命中率。
原始 CSV 是图表数值的表格版本；空白或非法数值不会填成零。
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from statistics import mean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# config 不在此顶层导入：config.py 在导入时无条件 makedirs，若用户用
# --results-dir 指定修复目录，须先解析参数、设置 RAG_RESULTS_DIR 再导入，
# 让 config 把副作用建到自定义目录而不是默认 results。见 main()。

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "text.color": "#252525",
    "axes.labelcolor": "#252525",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 1.4,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RES = None  # main() 中按 --results-dir 解析后设置；fig_* 只读它。
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
# 映射与类别绑定，过滤或重排 CSV 不会改变已有类别的颜色。
RETRIEVER_COLORS = {"dense": BLUE, "bm25": ORANGE, "hybrid": GREEN}
METHOD_COLORS = {"fixed": BLUE, "recursive": ORANGE, "sentence": GREEN}
VARIANT_COLORS = {"direct": BLUE, "baseline": BLUE, "rules": ORANGE,
                  "rerank": GREEN, "cross_encoder": GREEN}
UNKNOWN_COLOR = "#777777"  # 未知类别靠文字识别，不循环分配已有类别颜色。
GENERATED = []


def _read(name):
    path = os.path.join(RES, name)
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, UnicodeError, csv.Error) as exc:
        print(f"[跳过] 无法读取 {name}：{exc}")
        return []


def _number(value, low=None, high=None):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None
    if low is not None and value < low:
        return None
    if high is not None and value > high:
        return None
    return value


def _clean(rows, source, text=(), scores=(), numbers=()):
    """按图所需字段过滤；分数须在 [0, 1]，其他数量须非负。"""
    valid = []
    for row in rows:
        item = dict(row)
        if any(not str(row.get(key) or "").strip() for key in text):
            continue
        for key in text:
            item[key] = str(row[key]).strip()
        for key in (*scores, *numbers):
            item[key] = _number(row.get(key), 0, 1 if key in scores else None)
        if any(item[key] is None for key in (*scores, *numbers)):
            continue
        valid.append(item)
    if len(valid) != len(rows):
        print(f"[数据] {source}：跳过 {len(rows) - len(valid)} 行缺失或非法字段")
    return valid


def _save(fig, name):
    out = os.path.join(RES, name)
    try:
        os.makedirs(RES, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=140, bbox_inches="tight")
        GENERATED.append(out)
        print(f"[图] {out}")
    finally:
        plt.close(fig)


def _remove_stale(name):
    """本图本次无有效数据而早退时，删除上次残留的 PNG，避免误导。"""
    out = os.path.join(RES, name)
    if os.path.exists(out):
        try:
            os.remove(out)
            print(f"[清理] 删除过期 {out}")
        except OSError as exc:
            print(f"[跳过] 无法删除 {out}：{exc}")


def _t5_panel_data(rows):
    """Build independent recall and timing groups from their own valid rows."""
    specs = (
        ("recall@3", "Recall@3（gold 分片召回率）",
         _clean(rows, "T5 recall@3", text=("variant",), scores=("recall@3",))),
        ("rerank_ms", "重排阶段耗时（ms；非端到端延迟）",
         _clean(rows, "T5 rerank_ms", text=("variant",), numbers=("rerank_ms",))),
    )
    panels = []
    for key, label, source in specs:
        groups = defaultdict(list)
        for row in source:
            groups[row["variant"]].append(row[key])
        if len(groups) >= 2:
            panels.append((key, label, groups))
    return panels


def _score_axis(ax, label):
    ax.set_ylabel(label)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y")


def _category_ticks(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20 if len(labels) > 3 else 0,
                       ha="right" if len(labels) > 3 else "center")


def _entity_legend(ax, names, colors):
    names = list(dict.fromkeys(names))
    if len(names) > 1:
        ax.legend(handles=[Patch(facecolor=colors.get(n, UNKNOWN_COLOR), label=n)
                           for n in names], fontsize=8)


def fig_t1():
    rows = _clean(_read("t1_rag_vs_direct.csv"), "T1", text=("metric",),
                  scores=("RAG", "无RAG"))
    if not rows:
        _remove_stale("fig_t1_rag_vs_direct.png")
        return
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.30
    ax.bar([i - 0.18 for i in x], [r["RAG"] for r in rows], w,
           label="RAG", color=BLUE)
    ax.bar([i + 0.18 for i in x], [r["无RAG"] for r in rows], w,
           label="无 RAG", color=ORANGE)
    _category_ticks(ax, [r["metric"] for r in rows])
    ax.set_xlabel("评测指标（作答率不等于正确率）")
    _score_axis(ax, "指标值")
    ax.set_title("T1 基线 RAG 与裸 LLM 对比")
    ax.legend()
    _save(fig, "fig_t1_rag_vs_direct.png")


def fig_t2():
    rows = _clean(_read("t2_chunking.csv"), "T2（需要跨度对齐字段与 overlap）",
                  text=("method", "retriever"), scores=("Recall@5_evidence",),
                  numbers=("size", "overlap"))
    if not rows:
        _remove_stale("fig_t2_chunking.png")
        return
    # 每行一个实际配置，不挑最优、不合并 overlap，也不假定只有 hybrid。
    rows.sort(key=lambda r: (r["method"], r["retriever"], r["size"], r["overlap"]))
    labels = [f"{r['method']} | size={r['size']:g}, overlap={r['overlap']:g}"
              f" | {r['retriever']}" for r in rows]
    fig, ax = plt.subplots(figsize=(11, max(4.5, len(rows) * 0.34 + 1.8)))
    vals = [r["Recall@5_evidence"] for r in rows]
    ax.barh(range(len(rows)), vals, height=0.56,
            color=[METHOD_COLORS.get(r["method"], UNKNOWN_COLOR) for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Recall@5_evidence（gold 原文跨度覆盖率）")
    ax.set_ylabel("实际分块配置（每条 CSV 记录一行）")
    ax.set_title("T2 全部配置对比（保留 size、overlap 与检索器）")
    ax.grid(axis="x")
    best = max(range(len(vals)), key=vals.__getitem__)
    ax.annotate(f"最高 {vals[best]:.3f}", (vals[best], best),
                xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    _entity_legend(ax, [r["method"] for r in rows], METHOD_COLORS)
    _save(fig, "fig_t2_chunking.png")


def fig_t3():
    rows = _clean(_read("t3_query_opt.csv"), "T3", text=("strategy",),
                  scores=("Recall@5",))
    if len({r["strategy"] for r in rows}) < 2:
        print("[跳过] T3 查询策略不足两种（仅 baseline 时不生成对比图）；alpha 扫描独立处理")
        _remove_stale("fig_t3_query_opt.png")  # 仅 baseline 时清掉上次残留对比图
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(rows)), [r["Recall@5"] for r in rows], width=0.35, color=BLUE)
    _category_ticks(ax, [r["strategy"] for r in rows])
    ax.set_xlabel("查询策略")
    _score_axis(ax, "Recall@5（gold 分片召回率）")
    ax.set_title("T3 查询优化策略对比")
    _save(fig, "fig_t3_query_opt.png")


def _read_first(names):
    """按顺序取第一个存在的 CSV 内容，兼顾新旧命名（表头需含所需列）。"""
    for name in names:
        rows = _read(name)
        if rows:
            return rows
    return []


def fig_t3_alpha():
    # 优先新命名线性融合文件，回退到旧 rrf_sweep 文件（含 alpha_dense,Recall@5）。
    rows = _clean(_read_first(["t3_linear_alpha_sweep.csv", "t3_rrf_sweep.csv"]),
                  "T3 alpha", scores=("alpha_dense", "Recall@5"))
    if not rows:
        _remove_stale("fig_t3_alpha.png")
        return
    rows.sort(key=lambda r: r["alpha_dense"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["alpha_dense"] for r in rows], [r["Recall@5"] for r in rows],
            marker="o", color=BLUE)
    best = max(rows, key=lambda r: r["Recall@5"])
    ax.annotate(f"最高 {best['Recall@5']:.3f}",
                (best["alpha_dense"], best["Recall@5"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("α（dense 权重；0=纯 BM25，1=纯 dense）")
    ax.set_xlim(-0.03, 1.03)
    _score_axis(ax, "Recall@5（gold 分片召回率）")
    ax.set_title("T3 dense/BM25 线性融合权重扫描")
    _save(fig, "fig_t3_alpha.png")


def fig_t4():
    rows = _read("t4_retriever.csv")
    ks = [1, 3, 5, 10]
    curves = _clean(rows, "T4 Recall", text=("retriever",),
                    scores=tuple(f"Recall@{k}" for k in ks))
    ranks = _clean(rows, "T4 MRR", text=("retriever",), scores=("MRR@10",))
    if not curves and not ranks:
        _remove_stale("fig_t4_retriever.png")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    markers = {"dense": "o", "bm25": "s", "hybrid": "^"}
    for r in curves:
        name = r["retriever"]
        axes[0].plot(ks, [r[f"Recall@{k}"] for k in ks],
                     marker=markers.get(name, "D"), label=name,
                     color=RETRIEVER_COLORS.get(name, UNKNOWN_COLOR))
    axes[0].set_xlabel("检索结果数 k")
    _score_axis(axes[0], "Recall@k（gold 分片召回率）")
    axes[0].set_title("T4 分片召回率（非题目 Hit@k）")
    axes[0].set_xticks(ks)
    if curves:
        axes[0].legend()  # 只要有一条有效曲线就出图例，单曲线也不丢失图例
    names = [r["retriever"] for r in ranks]
    axes[1].bar(range(len(ranks)), [r["MRR@10"] for r in ranks], width=0.30,
                color=[RETRIEVER_COLORS.get(n, UNKNOWN_COLOR) for n in names])
    _category_ticks(axes[1], names)
    axes[1].set_xlabel("检索器")
    _score_axis(axes[1], "MRR@10（首个相关分片倒数排名均值）")
    axes[1].set_title("T4 平均倒数排名")
    _entity_legend(axes[1], names, RETRIEVER_COLORS)
    for ax, available in zip(axes, (curves, ranks)):
        if not available:
            ax.text(0.5, 0.5, "无有效数据", transform=ax.transAxes, ha="center")
    _save(fig, "fig_t4_retriever.png")


def fig_t5():
    rows = _read("t5_rerank.csv")
    if not rows:
        _remove_stale("fig_t5_rerank_proxy.png")
        return
    # 兼容旧的每题宽表，以及修复后的每题/variant 长表。
    if "variant" not in rows[0]:
        rows = [{"id": r.get("id"), "variant": variant, "recall@3": r.get(key)}
                for r in rows for variant, key in
                (("direct", "recall@3_direct"), ("rerank", "recall@3_rerank"))]
    # 不再按 status 整行剔除：CPU/cross_encoder 无候选时 recall@3=0 是
    # 真实测量，应按数值保留（含 0），否则会抬高均值或让方案整支消失。
    # 两个面板分别按自身数值字段清洗，合法耗时不会因 recall 缺失而被丢弃。
    panels = _t5_panel_data(rows)
    if not panels:
        print("[跳过] T5 缺少至少两个 variant 的有效检索或阶段耗时数据")
        _remove_stale("fig_t5_rerank_proxy.png")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.8),
                             squeeze=False)
    for ax, (key, label, groups) in zip(axes[0], panels):
        names = sorted(groups)
        ax.bar(range(len(names)), [mean(groups[n]) for n in names], width=0.30,
               color=[VARIANT_COLORS.get(n, UNKNOWN_COLOR) for n in names])
        _category_ticks(ax, [f"{n}\n(n={len(groups[n])})" for n in names])
        ax.set_xlabel("重排方案（n 为该指标有效记录数；不保证配对）")
        ax.set_ylabel(label)
        ax.set_title("T5 检索代理：逐记录均值（非答案质量）" if key == "recall@3"
                     else "T5 CSV rerank_ms 均值（非答案质量）")
        if key == "recall@3":
            ax.set_ylim(0, 1.08)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        _entity_legend(ax, names, VARIANT_COLORS)
    _save(fig, "fig_t5_rerank_proxy.png")


def fig_t6():
    """Draw the control flow actually implemented by experiments/t6_advanced.py."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    palette = {
        "start": ("#e8f1fc", "#2a78d6"),
        "route": ("#fff0e8", "#eb6834"),
        "branch": ("#e8f7f1", "#1baf7a"),
        "retry": ("#f6f0e5", "#9a6b2f"),
        "end": ("#eeeeee", "#666666"),
    }

    def box(x, y, w, h, title, detail, kind):
        face, edge = palette[kind]
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=face, edgecolor=edge, linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x, y + 0.16, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#252525")
        ax.text(x, y - 0.18, detail, ha="center", va="center",
                fontsize=8, color="#555555", linespacing=1.35)

    def arrow(x1, y1, x2, y2, label="", bend=0.0):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color="#555555",
            connectionstyle=f"arc3,rad={bend}",
        ))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label,
                    ha="center", va="bottom", fontsize=8, color="#555555")

    box(1.25, 5.95, 1.7, 0.82, "输入问题", "30 题 / L1–L4", "start")
    box(3.85, 5.95, 2.35, 0.92, "路由 Routing",
        "LLM 精确/综合\n失败→关键词启发式", "route")
    box(7.05, 6.25, 1.8, 0.72, "BM25 top-5", "精确查询分支", "branch")
    box(7.05, 5.38, 1.8, 0.72, "Hybrid top-5", "综合分析分支", "branch")
    box(6.35, 3.95, 2.15, 0.9, "支持分支",
        "gold level（oracle）\nL1/L2=3，L3/L4=8", "branch")
    box(9.15, 3.95, 2.2, 0.9, "检索证据",
        "chunk_id + 原文 + score\n不使用 gold answer", "retry")
    box(6.0, 2.55, 2.2, 0.9, "按原问题生成",
        "答案非空才进入下一阶段", "retry")
    box(3.0, 2.55, 2.3, 0.9, "Judge",
        "SUFFICIENT / UNSURE\n其他=invalid_judgment", "retry")
    box(0.9, 2.55, 1.4, 0.78, "返回结果", "最多两轮", "end")
    box(3.0, 0.95, 2.3, 0.82, "改写检索 Query",
        "保留实体、年份与约束\n空改写→停止", "route")
    box(6.0, 0.95, 2.2, 0.82, "第二轮回检+生成",
        "生成/judge 仍用原问题", "retry")
    box(9.25, 2.55, 2.25, 0.9, "输出与留痕",
        "routing/support/retry\n逐轮 trace + CSV", "end")
    box(9.25, 0.95, 2.25, 0.72, "质量代理",
        "EM / ROUGE-L / 忠实度代理", "end")

    arrow(2.1, 5.95, 2.68, 5.95)
    arrow(5.03, 5.8, 6.15, 6.2, "精确")
    arrow(5.03, 5.7, 6.15, 5.45, "综合")
    arrow(7.95, 6.1, 7.25, 4.42)
    arrow(7.95, 5.4, 6.95, 4.42)
    arrow(7.43, 3.95, 8.04, 3.95)
    arrow(8.45, 3.5, 7.1, 2.62)
    arrow(4.9, 2.55, 4.15, 2.55)
    arrow(1.9, 2.55, 1.62, 2.55)
    arrow(3.0, 2.1, 3.0, 1.36, "insufficient")
    arrow(4.15, 0.95, 4.9, 0.95)
    arrow(7.1, 0.95, 8.1, 2.15)
    arrow(9.25, 2.1, 9.25, 1.31)
    arrow(5.4, 1.36, 5.4, 2.1, "第二轮", bend=-0.22)

    ax.text(0.3, 0.26,
            "实线为 experiments/t6_advanced.py 已实现路径；"
            "重新绘自本轮 t6_routing.csv、t6_support.csv、t6_retry_trace.jsonl。",
            fontsize=8, color="#666666")
    ax.set_title("T6 编排流程：路由、oracle 难度分支与最多两轮 Judge 重试",
                 fontsize=13, pad=8)
    _save(fig, "fig_t6_flow.png")


def fig_t7():
    rows = [r for r in _read("t7_metrics.csv")
            if str(r.get("level") or "").strip().upper() != "ALL"]
    # 每个指标独立一幅，避免增加类别颜色，且保留修复结果中的 MAP@10。
    specs = [
        ("Hit@5(课件口径)", "Hit@5（课件：题目命中率）"),
        ("Recall@5(分片)", "Recall@5（gold 分片召回率）"),
        ("P@5", "P@5（返回分片精确率）"),
        ("MRR@10", "MRR@10（平均倒数排名）"),
        ("MAP@10", "MAP@10（平均精确率均值）"),
    ]
    panels = []
    for key, label in specs:
        if not any(key in r for r in rows):
            continue
        valid = _clean(rows, f"T7 {key}", text=("level",), scores=(key,))
        if valid:
            panels.append((key, label, sorted(valid, key=lambda r: r["level"])))
    if not panels:
        print("[跳过] T7 没有有效的分难度指标")
        _remove_stale("fig_t7_level.png")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 4.5),
                             squeeze=False)
    for ax, (key, label, valid) in zip(axes[0], panels):
        ax.bar(range(len(valid)), [r[key] for r in valid], width=0.32, color=BLUE)
        _category_ticks(ax, [r["level"] for r in valid])
        ax.set_xlabel("问题难度等级（不含 ALL）")
        _score_axis(ax, "指标值（逐题均值）")
        ax.set_title(label, fontsize=10)
    fig.suptitle("T7 分难度检索评测（非生成质量）", fontsize=12)
    _save(fig, "fig_t7_level.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认值在解析后再从 config 取，避免导入时序问题：先解析 CLI，
    # 若用户给了 --results-dir 就先把环境变量设成它，再导入 config，
    # 使 config 在导入时把副作用目录建到自定义目录而非默认 results。
    parser.add_argument("--results-dir", default=None,
                        help="CSV 输入与 PNG 输出目录（默认 config.RESULTS_DIR）")
    args = parser.parse_args()
    if args.results_dir is not None:
        os.environ["RAG_RESULTS_DIR"] = os.path.abspath(args.results_dir)
        from src import config as _config  # 延迟导入：让 config 建到自定义目录
    else:
        from src import config as _config
    global RES
    RES = os.path.abspath(args.results_dir if args.results_dir is not None
                          else _config.RESULTS_DIR)
    GENERATED.clear()
    for draw in (fig_t1, fig_t2, fig_t3, fig_t3_alpha, fig_t4, fig_t5, fig_t6,
                 fig_t7):
        draw()
    print(f"本次生成 {len(GENERATED)} 张图，目录：{RES}")


if __name__ == "__main__":
    main()
