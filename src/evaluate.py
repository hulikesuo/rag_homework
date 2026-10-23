"""评测指标（T4/T7 用）。

- recall_at_k / precision_at_k / mrr_at_k：基于 gold_chunk_ids 的检索指标。
- faithfulness / correctness：基于 gold_answer 的生成指标（含 LLM-judge 可选）。
"""
import re


def recall_at_k(gold_ids, retrieved_ids, k=None):
    """分片召回率：|gold ∩ retrieved| / |gold|。

    注意：本课程 FAQ 采用的是"命中率"口径（见 hit_at_k），
    两者对单 gold 题目相同，对多 gold 题目 hit@k 更高。
    报告中应同时给出两个口径，避免与课件定义不一致被扣分。
    """
    gold = set(gold_ids)
    if not gold:
        return 0.0
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    hit = len(gold & set(retrieved_ids))
    return hit / len(gold)


def hit_at_k(gold_ids, retrieved_ids, k=None):
    """课程 FAQ 口径的 Recall@k：gold 中至少一个出现在 top-k 即算 1 条命中。

    汇总时对题目取平均，即"命中题数 / 总题数"。
    与课件 9.5.4 / 作业 FAQ 的定义保持一致。
    """
    gold = set(gold_ids)
    if not gold:
        return 0.0
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    return 1.0 if (gold & set(retrieved_ids)) else 0.0


def precision_at_k(gold_ids, retrieved_ids, k=None):
    gold = set(gold_ids)
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not retrieved_ids:
        return 0.0
    return len(gold & set(retrieved_ids)) / len(retrieved_ids)


def mrr_at_k(gold_ids, retrieved_ids, k=None):
    gold = set(gold_ids)
    seq = retrieved_ids[:k] if k else retrieved_ids
    for pos, rid in enumerate(seq, start=1):
        if rid in gold:
            return 1.0 / pos
    return 0.0


def average_precision_at_k(gold_ids, retrieved_ids, k=10):
    """AP@k；跨题取均值为 MAP@k，分母为 min(|gold|, k)。"""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for rank, cid in enumerate(retrieved_ids[:k], 1):
        if cid in gold and cid not in seen:
            hits += 1
            score += hits / rank
        seen.add(cid)
    return score / min(len(gold), k)


def is_answered(answer):
    """保守的非拒答代理，不等于答案正确。"""
    text = (answer or "").strip()
    if not text or text.startswith("<LLM 调用失败"):
        return False
    markers = ("我不知道", "无法回答", "无法确定", "未提供资料", "没有相关资料",
               "资料未提及", "没有看到", "并未看到", "并没有看到")
    return not any(marker in text for marker in markers)


# ---------- 生成侧指标 ----------

def normalized_sents(text):
    sents = re.split(r"(?<=[。！？；!?])", text)
    return [s.strip() for s in sents if len(s.strip()) >= 2]


def faithfulness_factcheck(gold_chunk_texts, answer):
    """近似忠实度：答案中的陈述能否在参考上下文中找到支撑。

    启发式（无 LLM judge 时的近似），综合三路信号：
      1. 词覆盖率：答案里的中文 2-gram 有多大比例出现在上下文中；
      2. 数字一致性：答案出现的数字（学分/年份/比例等）是否能在上下文找到；
      3. 句级支撑：按句评估，句子词覆盖 < 阈值的记为"未支撑句"。
    返回 (得分 0-1, 未支撑句子列表)。

    注意：这是近似指标，报告里应说明其局限（同义改写会被低估）。
    """
    ctx = "".join(gold_chunk_texts)
    if not ctx or not answer:
        return 0.0, normalized_sents(answer or "")

    def grams(text):
        return {run[i:i + 2] for run in re.findall(r"[一-鿿]+", text)
                for i in range(len(run) - 1)}

    ctx_grams = grams(ctx)
    ctx_nums = set(re.findall(r"\d+(?:\.\d+)?", ctx))

    ans_grams = grams(answer)
    ans_nums = set(re.findall(r"\d+(?:\.\d+)?", answer))

    # 1) 词覆盖
    gram_cov = (len(ans_grams & ctx_grams) / len(ans_grams)) if ans_grams else 1.0
    # 2) 数字一致（答案里的数字有多少能溯源）
    num_cov = (len(ans_nums & ctx_nums) / len(ans_nums)) if ans_nums else 1.0

    # 3) 句级支撑
    unsupported = []
    total = 0
    for s in normalized_sents(answer):
        total += 1
        g = grams(s)
        if not g:
            # 纯数字/符号句：看数字能否溯源
            nums = set(re.findall(r"\d+(?:\.\d+)?", s))
            if nums and not nums <= ctx_nums:
                unsupported.append(s)
            continue
        cov = len(g & ctx_grams) / len(g)
        if cov < 0.4:
            unsupported.append(s)
    sent_score = 1.0 - len(unsupported) / max(total, 1)

    score = 0.5 * gram_cov + 0.25 * num_cov + 0.25 * sent_score
    return (round(min(max(score, 0.0), 1.0), 4), unsupported)


def correctness_em(gold_answer, answer):
    gold_n = re.sub(r"\s+", "", gold_answer or "")
    ans_n = re.sub(r"\s+", "", answer or "")
    return 1.0 if gold_n and ans_n and gold_n in ans_n else 0.0


def rouge_l_f1(gold_answer, answer):
    """简化 ROUGE-L：基于最长公共子序列的 F1。"""
    g = gold_answer or ""
    a = answer or ""
    L = _lcs_len(g, a)
    if not g or not a:
        return 0.0
    p = L / len(a)
    r = L / len(g)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _lcs_len(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]