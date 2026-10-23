"""重排 + 压缩（T5）。

- rerank_rules：固定词面规则重排，不使用 gold。
- rerank_cross_encoder：本地 cross-encoder 重排（bge-reranker）。
- rerank_judge_llm：LLM 0-shot 打分重排（弱监督，可选）。
- compress_extractive：压缩——只留与 query 相关的句子，输出更短 context。
"""
import re

from .textutil import split_sentences


def _parse_judge_score(raw):
    """Parse an integer 1-5 judge score without silently accepting garbage."""
    match = re.fullmatch(r"\s*([1-5])\s*(?:分|[.。])?\s*", raw)
    if not match:
        raise ValueError("judge score must be an integer from 1 to 5")
    return int(match.group(1))


def rerank_rules(query, items, top_k=None):
    """用标题、年份和词面覆盖的固定规则重排，不读取 gold 标注。

    保留原检索分数，标题精确命中加0.2，文件名词面覆盖加0.1，
    年份一致加0.1、明确冲突减0.2，片段词面覆盖加0.1。
    这些权重是预设启发式，没有在评测集上调优。
    """
    if top_k is not None and top_k <= 0:
        return []
    from .retriever import default_tokenize
    titles = re.findall(r"《([^》]+)》", query)
    years = set(re.findall(r"(?:19|20)\d{2}", query))
    query_tokens = set(default_tokenize(query))
    out = []
    for chunk, score in items:
        doc = chunk.get("doc", "")
        doc_years = set(re.findall(r"(?:19|20)\d{2}", doc))
        title_match = any(title in doc for title in titles)
        doc_coverage = len(query_tokens & set(default_tokenize(doc))) / max(len(query_tokens), 1)
        text_coverage = len(query_tokens & set(default_tokenize(chunk["text"]))) / max(len(query_tokens), 1)
        year_score = 0.0
        if years and doc_years:
            year_score = 0.1 if years & doc_years else -0.2
        out.append((chunk, float(score) + 0.2 * title_match + 0.1 * doc_coverage
                    + year_score + 0.1 * text_coverage))
    out.sort(key=lambda item: -item[1])
    return out[:top_k] if top_k is not None else out


_RERANK_CACHE = {}


def _get_reranker(model_name):
    """进程内缓存 cross-encoder，避免每题重复加载（30 题会慢到不可用）。"""
    if model_name not in _RERANK_CACHE:
        from sentence_transformers import CrossEncoder
        _RERANK_CACHE[model_name] = CrossEncoder(model_name, max_length=512)
    return _RERANK_CACHE[model_name]


def rerank_cross_encoder(query, items, top_k=None):
    """items: [(chunk, score)] → [(chunk, new_score)]，用 cross-encoder 打分。

    默认读 config.RERANK_MODEL_NAME 本地模型；未配则跳过重排。
    模型在进程内缓存，只加载一次。实验应在计时推理前单独加载/预热。
    """
    if not items or (top_k is not None and top_k <= 0):
        return []
    from . import config
    model_name = config.RERANK_MODEL_NAME
    if not model_name:
        print("[rerank] 未配置 RERANK_MODEL_NAME，跳过重排。")
        return items[:top_k] if top_k is not None else items
    try:
        model = _get_reranker(model_name)
    except ImportError:
        raise RuntimeError("需要 sentence-transformers 运行 cross-encoder")
    pairs = [(query, c["text"]) for c, _ in items]
    scores = model.predict(pairs)
    ranked = sorted(zip(items, scores), key=lambda x: -float(x[1]))
    out = [(c, float(s)) for (c, _), s in ranked]
    return out[:top_k] if top_k is not None else out


def rerank_judge_llm(query, items, top_k=None):
    """LLM 0-shot 打分重排：让模型对每个候选给 1-5 分。

    逐条调用，慢但无需本地 reranker。供没有 cross-encoder 的机器用。
    打分失败时保留原检索分数和原顺序，不把失败伪装成中性分数 3。
    """
    if not items or (top_k is not None and top_k <= 0):
        return []
    from .generator import complete
    scored, failed = [], []
    for c, score in items:
        prompt = f"""请评估以下片段对问题『{query}』的回答相关程度，只输出一个 1-5 的整数。
片段：{c['text'][:600]}
分数："""
        try:
            raw = complete(prompt, max_tokens=8).strip()
            s = _parse_judge_score(raw)
        except Exception:
            # 失败候选单独排在其后，并沿用原检索分数排序；不伪装成中性 3 分。
            failed.append((c, float(score)))
            continue
        scored.append((c, float(s)))
    scored.sort(key=lambda item: (-item[1], str(item[0].get("chunk_id", ""))))
    failed.sort(key=lambda item: (-item[1], str(item[0].get("chunk_id", ""))))
    ranked = scored + failed
    return ranked[:top_k] if top_k is not None else ranked


def compress_extractive(query, chunks, budget):
    """按相关度抽取完整句子，返回文本长度 <= budget（含连接换行）。

    先去除首尾空白并去重，再分配预算；放不下的整句跳过，不截断证据。
    非正预算返回空文本。原字符数按未压缩文本以双换行连接后计算，
    返回 (new_text, 原字符数, 压缩后字符数, 压缩比)。
    """
    orig = len("\n\n".join(c["text"] for c in chunks))
    if budget <= 0:
        return "", orig, 0, 0.0

    best, seen = [], set()
    for c in chunks:
        for sentence in split_sentences(c["text"]):
            sentence = sentence.strip()
            if not sentence or sentence in seen:
                continue
            seen.add(sentence)
            best.append((sentence, _sim_bits(query, sentence)))
    best.sort(key=lambda item: -item[1])
    picked, used = [], 0
    for sentence, _ in best:
        cost = len(sentence) + bool(picked)
        if used + cost > budget:
            continue
        picked.append(sentence)
        used += cost
    new_text = "\n".join(picked)
    return new_text, orig, len(new_text), (len(new_text) / orig if orig else 0.0)


def _sim_bits(query, sent):
    """简单的公共字符重合作为相关度代理（作业够用）。"""
    q = set(query)
    s = set(sent)
    inter = len(q & s)
    return inter / max(len(q) + len(s) - inter, 1)
