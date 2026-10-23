"""LLM 生成（开放 API 兼容 OpenAI 协议）。

单一入口 `complete(prompt)`；支持流式地简单同步返回文本。
若配置了本地 Ollama，base_url 指向它也能跑。
"""
import os
import time
import requests

from . import config


def _client():
    config.require_llm()
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("需要 openai：`pip install openai`")
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def complete(prompt, max_tokens=1024, temperature=0.2, messages=None):
    """messages 可选；默认单条 user 消息。返回文本。

    对瞬时的连接/超时错误有界重试；鉴权与格式错误直接抛出，
    避免把配置问题伪装成模型输出。
    """
    client = _client()
    msgs = messages or [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            retryable = exc.__class__.__name__ in {
                "APITimeoutError", "APIConnectionError", "ConnectTimeout",
                "ReadTimeout", "RemoteProtocolError",
            }
            if not retryable or attempt == 4:
                raise
            last_error = exc
            time.sleep(min(8.0, 1.5 * (2 ** attempt)))
    raise last_error


def generate_rag(query, context_chunks, system=None):
    """把检索到的 context 组装进提示，交给 LLM 生成答案。"""
    default_system = (
        "你是检索增强问答助手。只依据提供的资料回答；"
        "资料中没有的信息，请明确说'资料未提及'，不要编造。"
    )
    context = "\n\n".join(f"[片段{i+1}]\n{c['text']}" for i, c in enumerate(context_chunks))
    prompt = f"""资料：
{context}

问题：{query}

请基于以上资料作答："""
    return complete(prompt, messages=[
        {"role": "system", "content": default_system},
        {"role": "user", "content": prompt},
    ])


def generate_direct(query):
    """无 RAG 的裸 LLM 对比（T1 需要）。"""
    return complete(query, messages=[
        {"role": "system", "content": "你是通用问答助手，直接回答。"},
        {"role": "user", "content": query},
    ])
