"""Plot T7 llm-judge level aggregation from the real CSV/summary."""
from __future__ import annotations

import hashlib
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import font_manager

for _fp in ("C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/Deng.ttf"):
    if pathlib.Path(_fp).exists():
        try:
            font_manager.fontManager.addfont(str(_fp))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=_fp).get_name()] + \
                plt.rcParams.get("font.sans-serif", [])
            break
        except Exception:
            continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "repair_outputs" / "acceptance_20261023" / "t7"
CSV = OUT / "t7_llm_judge.csv"
SUMMARY = OUT / "t7_llm_judge_summary.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    levels = summary["levels"]
    order = ["L1", "L2", "L3", "L4", "ALL"]
    labels = [k for k in order if k in levels]
    if "ALL" not in labels:
        labels += ["ALL"]
    index = pd.Index(labels, name="level")
    faith = pd.Series([levels[k]["faithfulness_1_5_mean"] for k in labels],
                      index=index)
    correct = pd.Series([levels[k]["correctness_0_5_mean"] for k in labels],
                        index=index)
    hallu_macro = pd.Series(
        [levels[k]["hallucination_rate_macro"] for k in labels], index=index)
    hallu_micro = pd.Series(
        [levels[k]["hallucination_rate_micro"] for k in labels], index=index)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.4))
    fig.suptitle(
        "T7 LLM Judge 分层结果（30题正式集）\n"
        "同模型自评 glm-5.3-flash，非独立人工真值",
        fontsize=13)

    ax = axes[0]
    width = 0.35
    x = list(range(len(labels)))
    ax.bar([i - width / 2 for i in x], faith,
           width=width, label="Faithfulness 1-5", color="#2f6fb3")
    ax.bar([i + width / 2 for i in x], correct,
           width=width, label="Correctness 0-5", color="#45a06f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("mean score")
    ax.set_title("Faithfulness / Correctness 均值")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in zip(x, faith):
        ax.text(i - width / 2, v + 0.06, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
    for i, v in zip(x, correct):
        ax.text(i + width / 2, v - 0.12, f"{v:.3f}",
                ha="center", va="top", fontsize=7)

    ax = axes[1]
    ax.plot(labels, hallu_macro, marker="o", label="Hallucination macro",
            color="#c2543a")
    ax.plot(labels, hallu_micro, marker="s", label="Hallucination micro",
            color="#8b5fa8")
    for i, v in enumerate(hallu_macro):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(hallu_micro):
        ax.text(i, v - 0.006, f"{v:.3f}", ha="center", va="top", fontsize=7)
    ax.set_ylim(0, max(hallu_macro.max(), hallu_micro.max()) * 1.25 + 0.01)
    ax.set_ylabel("rate")
    ax.set_title("Hallucination rate（仅真实 claims>0 统计）")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_t7_level.png", dpi=200)

    hashes = {
        "csv_sha256": sha256_file(CSV),
        "summary_sha256": sha256_file(SUMMARY),
        "n": int(df.shape[0]),
        "levels": labels,
    }
    (OUT / "fig_t7_source_hash.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("WROTE", OUT / "fig_t7_level.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
