# RAG 课程实践作业

## 代码仓库与本地数据

`code-only` 分支只发布 `src/`、`demo/`、`experiments/`、`scripts/` 下的 Python 源码，以及 README、依赖清单、`.gitignore` 和无密钥的 `.env.example`。该分支采用独立根提交，不继承原分支中的数据提交历史。

**不上传**原始文档、题集、标注表、实验结果、报告、压缩包、模型/索引缓存或真实 `.env`。这些文件仍保留在原电脑，本仓库不提供下载地址。下文的实验数值与验收记录描述本地完整版本，不代表代码分支包含相应数据或已生成的验收产物。

新电脑需自行准备并获准使用语料，放入 `data/corpus/`；将兼容的评测题集放入 `data/eval/qa_set.jsonl`。依赖标注表、补充题集、历史结果或报告的脚本，还需自行提供相应本地输入。缺少这些文件时，不能仅凭代码仓库复现完整验收。

Windows PowerShell 启动示例：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
# 在本机编辑 .env，填写自己的 LLM 服务地址、密钥和模型名；不要提交 .env。
# 准备好 data/corpus/ 后启动；首次使用需联网下载向量模型。
& .\.venv\Scripts\python.exe demo\app.py
```

启动后访问 `http://127.0.0.1:7860`。生成回答仍会调用配置的 API，可能产生费用。后续提交前请检查 `git diff --cached --name-only`，不要使用 `git add -f` 绕过数据与密钥忽略规则。

---

面向云南大学计算机/软件学院教务文档的检索增强问答实验。项目将解析、分块、查询优化、检索、重排、压缩、生成和评测拆开，比较各模块的作用及失败原因。

**当前状态（2026-10-23）**：正式顶层 `results/` 仍为冻结 42 文件；已审新结果按课程 D4 要求放入 `results/acceptance_20261023/{t2,t3,t4,t7,demo}`，均保留原始文件名与 runner。`scripts/verify_acceptance_bundle.py` 推荐一键离线验收（T7 71、T3 14+RRF、Demo 4、两份主代理独立审计），当前本地已跑通 7/7 PASS。最终评审包为 `大模型6组_RAG作业_验收版_20261023_v3.zip`。仍未整体 ACCEPTED：真实学号缺失、成员分工被用户删除故不恢复、人工是否核过 gold 需用户确认。

## 1. 课程任务与代码对应

课程要求 **T1–T5 必做，T6 或 T7 至少完成一项**。存在代码入口不等于实验已经完成。

| 任务/课程模块 | 主要代码 | 当前实现与边界 |
|---|---|---|
| T1 基础 RAG、无 RAG 对照 | `src/pipeline.py`、`src/generator.py`、`experiments/t1_baseline.py` | 解析→分块→索引→检索→生成；已通过 `langchain_core.RunnableLambda` 组装链式流水线，30题 RAG+30题裸 LLM 均生成成功。 |
| T2 分块 | `src/chunker.py`、`experiments/t2_chunking.py` | fixed/recursive/sentence；size=256/512/1024、overlap=0/128，sentence 仅 overlap=0，共15组。跨配置按原文跨度评测，450条答案全部生成成功。 |
| T3 查询优化 | `experiments/t3_query_opt.py` | baseline/rewrite/multi/subquery + RRF control + α 扫描均已跑；`results/acceptance_20261023/t3/` 含 30 行 CSV、checkpoint、runner、原始日志。q28 子查询 dense5 命中 gold；需注意 RRF top5 仍为 0、union 不等于 Recall。DAO 分析平均质量不能宣称查询优化提升生成。 |
| T4 稠密/稀疏/混合检索 | `src/embed.py`、`src/index_store.py`、`src/retriever.py`、`experiments/t4_retriever_compare.py` | bge+FAISS、BM25、全库线性融合；输出逐题证据与两类检索胜例。正式 30 题主指标固定；新增 `results/acceptance_20261023/t4/` 含严格 10 case（dense 5 + BM25 5）与 80 账窗口/原始 trace，机验 0 错。 |
| T5 重排与压缩 | `src/rerank.py`、`experiments/t5_rerank_compress.py` | 共享 hybrid top-5，比较 direct/规则/CrossEncoder top-3，再比较压缩前后。**压缩已实际接入 `compression.InformationCompressor().fit(docs.values())` 的语料 unigram 自信息压缩，不是字符重合抽句。** 90条重排、18条成对压缩记录。 |
| T6 高级检索 | `experiments/t6_advanced.py` | 路由（21条 LLM 路由+9条 invalid fallback 退化为启发式）、gold level 的 oracle 支持分支、最多两轮循环。`fig_t6_flow.png` 已生成。 |
| T7 分层评测与忠实度 | `src/evaluate.py`、`experiments/t7_eval.py` | 分难度检索指标、缓存核对、启发式诊断与 30 题 LLM judge（`glm-5.3-flash`，`metric_type=llm_judge_not_human`）。最新批次 168 claims；faithfulness 4.8333、correctness 4.5333、micro 幻觉率 0.0298；最差 5 条证据已审计。 |

两条压缩路径（`src/compression.py` 自信息压缩与 `rerank.compress_extractive` 抽取式）现共用 `src/textutil.py` 的句边界规则。纯函数回归检查不加载模型、不调用 API，也不写实验结果：

```bash
python -I -B scripts/check_compression.py --root .
python -I -B scripts/check_handover_issues.py --root .
```

## 2. 项目结构

```text
rag_homework/
├── data/
│   ├── corpus/                       # 43份原始 docx，按4类存放
│   └── eval/qa_set.jsonl             # 正式题集（已同步修订候选）
├── src/                              # 核心模块，见上表
│   ├── config.py                     # 路径、模型、默认参数、.env 读取
│   ├── loader.py                     # 文档解析，含 docx
│   └── compression.py                # 自信息压缩模块（InformationCompressor）
├── experiments/                      # run_all.py 与 t1–t7 实验编排
├── scripts/
│   ├── build_qaset.py                # 分块清单、标注校验与转换
│   ├── qa_annotation.csv             # 正式标注表
│   ├── llm_judge_t7.py               # LLM 忠实度评分脚本（待真实重跑）
│   ├── check_t7_judge.py             # T7 judge 回归测试（67/67通过）
│   ├── restore_submission_results.py # D4 结果恢复脚本（默认 dry-run，--apply 安全落盘）
│   ├── check_restoration.py          # 恢复后独立校验脚本
│   ├── prepare_submission_results.py # 旧 D3/D4 对齐参考（不推荐运行，见恢复脚本）
│   ├── check_submission_results.py   # 旧对齐校验参考
│   └── make_figures.py               # 根据实际 CSV 字段生成图
├── demo/app.py                       # Gradio 演示
├── repair_outputs/
│   ├── qa_set_revised.jsonl           # 修订候选题集（与正式题集字节一致）
│   ├── qa_annotation_revised.csv      # 对应候选标注表
│   ├── qa_set_review.md               # 逐题复核记录
│   ├── verified_20261022/             # 早期部分复测，保留作历史记录
│   ├── revised_full_20261022/         # 本轮修订题集完整结果与图表（源）
│   ├── t7_judge_audit.json           # T7 审计输出
│   └── t7_worst5_evidence.md         # 最差五题证据审计
├── results/                          # 正式结果目录（42文件含manifest）
├── requirements.txt
├── .env.example
└── 报告.md
```

## 3. 安装与模型配置

Python 3.10+，建议建立虚拟环境。

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

| 配置项 | 用途 |
|---|---|
| `LLM_BASE_URL` | 本地或远程 OpenAI 兼容服务的 `/v1` 地址 |
| `LLM_API_KEY` | 服务凭证；当前代码要求非空 |
| `LLM_MODEL` | 服务实际提供的生成模型名 |
| `RERANK_MODEL_NAME` | 可选本地 CrossEncoder，如 `BAAI/bge-reranker-base` |

生成入口使用 OpenAI 兼容的 `chat.completions` 接口。向量模型固定为 `BAAI/bge-small-zh-v1.5`，CPU 可运行。首次下载需联网，`src/config.py` 默认使用 `https://hf-mirror.com`。

```bash
# 首次联网准备
unset RAG_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -c "from src import embed; embed._get_encoder()"
```

完整缓存后设 `RAG_OFFLINE=1`。离线变量只限制模型 Hub 加载，**不会禁止已显式启用的生成 API**。

## 4. 数据与修订题集

语料共 **43份 docx**，解析文本合计 **176086个字符**。四类为培养方案与专业介绍15份、课程大纲10份、选课与教学管理10份、保研推免8份。基准采用 fixed、size=512、overlap=128，得到 **479个分块**。

正式题集 `data/eval/qa_set.jsonl` 已于 2026-10-23 与修订候选 `repair_outputs/qa_set_revised.jsonl` 逐字节同步（hash 一致），对应标注表 `scripts/qa_annotation.csv` 同步完成。30题分布为 **L1=8、L2=8、L3=7、L4=7**。

L1 为明示事实，L2 为跨片段整合或简单推理，L3 为应用材料给出的规则，L4 为未给解题步骤的情境决策。L4 参考答案提供五项、每项1分的开放 rubric。

## 5. 复现实验

统一运行器在未传 `--generate` 时强制使用离线模型缓存，默认只选 **T2/T4/T5/T7**：

```bash
RAG_OFFLINE=1 python -m experiments.run_all --eval repair_outputs/qa_set_revised.jsonl --results repair_outputs/revised_full_20261022
RAG_OFFLINE=1 python -m experiments.run_all t3 t6 --eval repair_outputs/qa_set_revised.jsonl --results repair_outputs/revised_full_20261022
```

显式启用生成（**会调用远程 API，可能计费**）：

```bash
RAG_OFFLINE=1 python -m experiments.run_all t1 t7 --generate --eval repair_outputs/qa_set_revised.jsonl --results repair_outputs/revised_full_20261022
RAG_OFFLINE=1 python -m experiments.run_all t2 t3 t5 t6 --generate --eval repair_outputs/qa_set_revised.jsonl --results repair_outputs/revised_full_20261022
```

## 6. 指标与实验解释

- **Hit@k（课件口径）**：top-k 至少命中一个 gold 即为1。**Recall@k（分片口径）**：`|gold ∩ top-k| / |gold|`。T7 列名区分 `Hit@5(课件口径)` 与 `Recall@5(分片)`。
- **T2 跨分块比较**：`Recall@5_evidence` 为 top-5 覆盖的 gold 字符并集长度/gold 字符并集长度。
- **检索分数**：dense 为 `1/(1+L2)`；BM25 做 min-max 归一化。hybrid 为 `α·dense+(1−α)·BM25`，默认 α=0.5。
- **T3 两种融合**：查询优化用 RRF 融合 top-20；alpha 扫描用 dense/BM25 线性分数融合，不是同一实验。
- **T5 公平对照**：每题获得一份 hybrid top-5，direct/规则/CrossEncoder 各取前3。压缩模块已实际接入 `InformationCompressor().fit()`。
- **生成指标限制**：EM/ROUGE-L 为字符级代理，不能替代语义正确性与 L4 rubric。
- **T7 评分**：`t7_manual_scoring.csv` 的 `faithfulness_1_5` 和 `L4_rubric_0_5` 全部为空。 `heuristic_*` 只是字符/数字重合诊断（120/283=0.424）。已有 30题 `glm-5.3-flash` LLM judge 评分，但 judge 模型与生成模型相同，存在自评偏乐观风险，**必须标明 LLM 非人工且内部不自洽**（q02,q13,q23 faithfulness=5但unsupported>0）。

## 7. 结果、图表与演示

| 实验 | 有效样本 | 关键结果 |
|---|---:|---|
| T1 | 30 RAG + 30 裸 LLM | RAG/裸 LLM：EM(精确包含) 0.2333/0.1000，ROUGE-L 0.2680/0.1332，作答率 0.9667/0.8333 |
| T2 | 15组×30题=450 | 最佳 fixed 1024/128：Recall@5_evidence 0.8141；fixed 512/128 为 0.8125；450条生成成功 |
| T3 | baseline/rewrite/multi/subquery/controlRRF30/α{0,0.3,0.5,0.7,1} | recall baseline .6972/rewrite .8139/multi .775/subquery .7861/control .7972；控制RRF回答质量 ROUGE .2777 高于查询优化，不能宣称查询优化平均提高质量；α 课件 Hit .8667 在 α=.5/.7 并列 |
| T4 | 3路×30题=90 + 10 case | BM25/dense/hybrid Recall@5 0.7889/0.6972/0.7972；新增严格 10 case（dense 5、BM25 5）机验 0 错；80 账 dense6/bm2535/tie39 为探索窗口 |
| T5 | 90条重排、18条压缩 | rules Recall@3 最高（0.7556）；压缩字符保留率 0.3252；ROUGE-L 0.1509→0.1092 |
| T6 | 30路由/分支，6题循环 | 路由 Recall@5 0.7915；judge sufficient 0.5000；ROUGE-L 0.1855→0.1908 |
| T7 | 30题检索；LLM judge 30/30 | Hit@5 0.8667、Recall@5 0.7972；LLM faithfulness 均值 4.8333、correctness 4.5333、micro 幻觉率 0.0298（最新 batch 168 claims） |

结论边界：T7 的 `heuristic_*` 不替代真实忠实度；T5 压缩 ROUGE-L 下降不能称质量正收益；旧 rewrite 无严格成功案例，新 subquery 的 q28 已用真实子查询找回证据并综合正确，但 RRF top5 仍 0，union19 不能当 Recall；T4 已在补充集中达到严格 5+5，80 账为探索窗口；T7 LLM judge 同模型自评偏差，**不能称人工评分**。压缩回归 17/17、交接 7/7、T3 离线 12/12、T7 离线 67/67 均已通过，但这些只是离线/非整体验收。

脚本按实际字段生成图表：

```bash
python -m scripts.make_figures --results-dir repair_outputs/revised_full_20261022
```

Gradio 演示启动后访问 `http://127.0.0.1:7860`。本轮已完成真实 Gradio HTTP queue L1/L4 端到端并留下 `results/acceptance_20261023/demo/http_e2e_results.json`；浏览器视觉是建议项，不是硬课程阻断。

## 8. 提交材料与评分

D1–D6 交付物：D1 源码库、D2 知识库、D3 测试集、D4 实验结果、D5 实验报告、D6 演示。

课程评分：**工程20、数据15、实验严谨性25、分析25、报告10、演示5**，合计100分；T8 加分上限15分。

当前技术项均已整理并独立审计；真实阻断只剩真实学号、按用户要求删除且不恢复的成员分工、以及人工是否核过 gold 的确认。D5 允许 `.md`/`.pdf` 且 ≤8 页；T6/T7 至少完成一项。

### 一键验收

```bash
# 离线：不调用 API、不加载模型、不修改原始结果；会写一个审计报告
python -I -B scripts/verify_acceptance_bundle.py

# 包计划/打包
python -I -B scripts/build_acceptance_package.py --dry-run   # 先看计划
python -I -B scripts/build_acceptance_package.py --apply     # 产出 v3 zip
```

评审包文件名：`大模型6组_RAG作业_验收版_20261023_v3.zip`。新 T2/T3/T4/T7/Demo 产物按课程 D4 要求放入包内 `results/acceptance_20261023/{t2,t3,t4,t7,demo}`，不覆盖正式顶层 `results/` 的 42 个冻结文件。旧的 `大模型6组_RAG作业_验收版_20261023.zip` 与 `大模型6组_RAG作业_验收版_20261023_v2.zip` 保留不动。
