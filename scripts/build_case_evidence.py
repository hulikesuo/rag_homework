import json, csv, hashlib, os

t2_sha = hashlib.sha256(open("repair_outputs/revised_full_20261022/t2_chunking_cases.csv","rb").read()).hexdigest()

lines = []
lines.append("# T2 切分不当导致生成失败的案例证据")
lines.append("")
lines.append("## 数据来源绑定")
lines.append("")
lines.append("- `t2_chunking_cases.csv` sha256: `%s`" % t2_sha)
lines.append("- 审核方: AI代码代理证据审核，非人工评分")
lines.append("")

# ================================================================
# LOAD T2 data for q06
# ================================================================
rows = []
with open("repair_outputs/revised_full_20261022/t2_chunking_cases.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["id"] == "q06":
            rows.append(r)

r256 = None
r512 = None
for r in rows:
    if r["method"] == "fixed" and r["size"] == "256" and r["overlap"] == "128":
        r256 = r
    if r["method"] == "fixed" and r["size"] == "512" and r["overlap"] == "128":
        r512 = r

lines.append("## 受控配置对: fixed chunker, overlap=128, sizes 256 vs 512")
lines.append("")
lines.append("问题: 《数学建模与实验》教学大纲中，MATLAB部分采用什么教学方式？")
lines.append("gold_answer: 知识点讲解及案例演示，学生动手操作")
lines.append("")

# === 256/128 ===
lines.append("### 配置A: fixed / size=256 / overlap=128")
lines.append("")
lines.append("Recall@5_evidence=%s, EM=%s, ROUGE-L=%s" % (
    r256["Recall@5_evidence"], r256["EM_proxy"], r256["ROUGE-L_proxy"]))
lines.append("")
lines.append("*模型答案*: %s" % r256["answer"])
lines.append("")

ev256 = json.loads(r256["retrieved_evidence"])
lines.append("*检索 top-5 逐 chunk 分析*:")
lines.append("")
lines.append("| rank | chunk_id | len | 含表头 | 含MATLAB | 含答案值 |")
lines.append("|------|----------|-----|--------|----------|----------|")
for ei, e in enumerate(ev256):
    cid = e["chunk_id"]
    txt = e["text"]
    has_hdr = "教学方式" in txt
    has_mat = "MATLAB" in txt
    has_ans = "知识点讲解及案例演示" in txt
    mat_p = txt.find("MATLAB") if has_mat else -1
    ans_p = txt.find("知识点讲解及案例演示") if has_ans else -1
    lines.append("| %d | `%s` | %d | %s | %s(pos=%d) | %s(pos=%d) |" % (
        ei+1, cid, len(txt), has_hdr, has_mat, mat_p, has_ans, ans_p))
lines.append("")

lines.append("*截断证据*:")
lines.append("")
lines.append("- rank#3 chunk `__011` (256 chars) 包含'教学方式'表头(位置202, 离末尾仅54 chars)，")
lines.append("  也包含'MATLAB程序设计及工作环境使用介绍'开头(位置217, 离末尾39 chars)，")
lines.append("  但'知识点讲解及案例演示，学生动手操作'在原始文档中位于该行MATLAB内容之后，")
lines.append("  超出256字符边界，被切到下一个chunk `__004`")
lines.append("- chunk `__004` (512/128配置下出现，但256/128下未被检索到 top-5)")
lines.append("- 模型从已有上下文正确识别了信息不足，给出拒答而非编造")
lines.append("")

lines.append("*模型拒答文本*: '片段3仅提到教学内容包括...但该片段在教学方式一栏处被截断，未显示具体内容'")
lines.append("")

# === 512/128 ===
lines.append("### 配置B: fixed / size=512 / overlap=128")
lines.append("")
lines.append("Recall@5_evidence=%s, EM=%s, ROUGE-L=%s" % (
    r512["Recall@5_evidence"], r512["EM_proxy"], r512["ROUGE-L_proxy"]))
lines.append("")
lines.append("*模型答案*: %s" % r512["answer"])
lines.append("")

ev512 = json.loads(r512["retrieved_evidence"])
lines.append("*检索 top-5 逐 chunk 分析*:")
lines.append("")
lines.append("| rank | chunk_id | len | 含表头 | 含MATLAB | 含答案值 |")
lines.append("|------|----------|-----|--------|----------|----------|")
for ei, e in enumerate(ev512):
    cid = e["chunk_id"]
    txt = e["text"]
    has_hdr = "教学方式" in txt
    has_mat = "MATLAB" in txt
    has_ans = "知识点讲解及案例演示" in txt
    mat_p = txt.find("MATLAB") if has_mat else -1
    ans_p = txt.find("知识点讲解及案例演示") if has_ans else -1
    lines.append("| %d | `%s` | %d | %s | %s(pos=%d) | %s(pos=%d) |" % (
        ei+1, cid, len(txt), has_hdr, has_mat, mat_p, has_ans, ans_p))
lines.append("")

lines.append("*成功证据*:")
lines.append("")
lines.append("- rank#4 chunk `__004` (373 chars) 包含完整答案:")
lines.append("  '教学方式'表头(位置74)，'MATLAB程序设计...'内容(位置89)，")
lines.append("  '知识点讲解及案例演示，学生动手操作'(位置244)")
lines.append("- chunk `__004` 是 gold_chunk_id 本身")
lines.append("- Recall@5=1.0，gold 被成功召回")
lines.append("")

# === Conclusion ===
lines.append("### 结论")
lines.append("")
lines.append("同一文档(chunk_id `__004` 为 gold)，两个fixed配置、相同overlap(128)、仅size不同(256→512):")
lines.append("")
lines.append("|  | size=256 | size=512 |")
lines.append("|--|----------|----------|")
lines.append("| Recall@5 | 0.343 | 1.000 |")
lines.append("| gold `__004` 是否在top-5 | 否 | 是 (rank#4) |")
lines.append("| 答案值 '知识点讲解及案例演示' 可见 | 否(被切分边界截断) | 是 |")
lines.append("| 模型行为 | 适当拒答 | 正确作答 |")
lines.append("| ROUGE-L | 0.032 | 0.447 |")
lines.append("")
lines.append("*关键事实*: 256配置下'教学方式'表头出现在 chunk `__011` 末尾(202/256 chars)，")
lines.append("但该表头的具体值('知识点讲解...')在原文档中紧随其后，跨越256边界后进入下一个chunk。")
lines.append("下一chunk `__004` 未被召回至 top-5，导致模型只能看到表头但看不到值。")
lines.append("这是**切分边界在表格行内切断导致信息丢失**的真实案例。")
lines.append("模型行为是**适当拒答**(明确说明'信息被截断')而非编造。")
lines.append("")

out = "\n".join(lines)
with open("repair_outputs/revised_full_20261022/t2_chunking_failure_case.md", "w", encoding="utf-8") as f:
    f.write(out)
print("T2 analysis written: %d lines" % len(lines))
