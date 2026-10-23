"""D6 演示应用：云南大学教务知识库 RAG 问答（Gradio）。

启动：
  pip install gradio
  python -m demo.app

功能：
  - 输入问题 → 展示 RAG 答案
  - 展示检索到的 top-k 片段（含来源文档、相关度分数）
  - 可切换检索模式（dense / bm25 / hybrid）、是否重排/压缩
  - 侧栏显示本次耗时与上下文压缩率，便于现场演示各模块效果
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr  # noqa: E402

from src import config, loader, pipeline  # noqa: E402

_PIPE = None


def get_pipe():
    """首次调用时构建索引（约 10-20 秒），之后复用。"""
    global _PIPE
    if _PIPE is None:
        docs = loader.read_all_docs(config.CORPUS_DIR)
        if not docs:
            raise RuntimeError("语料为空：请先准备可读取的本地文档。")
        _PIPE = pipeline.RAGPipeline(docs)
    return _PIPE


def answer_fn(question, mode, do_compress, top_k):
    if not question or not question.strip():
        return "请输入问题。", "", ""

    t0 = time.time()
    try:
        pipe = get_pipe()
    except Exception as e:
        return f"初始化失败：{type(e).__name__}", "", ""

    # 检索与压缩放在同一错误边界内，异常时仍返回可读的状态信息。
    try:
        hits = pipe.retrieve(question, k=int(top_k), mode=mode)
        ctx_chunks = [c for c, _ in hits]
        orig = sum(len(c["text"]) for c in ctx_chunks)
        if do_compress:
            from src.rerank import compress_extractive
            new_text, _, after, ratio = compress_extractive(
                question, ctx_chunks, budget=600)
            from src.chunker import new_chunk
            ctx_chunks = [new_chunk("compressed", new_text, 0)]
        else:
            after = orig
            ratio = 1.0
    except Exception as e:
        elapsed = time.time() - t0
        return (f"检索或压缩失败：{type(e).__name__}", "",
                f"耗时 {elapsed:.1f}s | 错误类型：{type(e).__name__}")

    try:
        from src import generator
        ans = generator.generate_rag(question, ctx_chunks)
    except Exception as e:
        ans = f"生成失败：{type(e).__name__}"

    elapsed = time.time() - t0

    # 检索证据展示
    evid_lines = []
    for i, (c, score) in enumerate(hits, 1):
        doc = os.path.basename(c["doc"].replace("\\", "/"))
        snippet = c["text"][:220].replace("\n", " ")
        evid_lines.append(
            f"**[{i}] {doc}** (score={score:.3f})\n\n> {snippet}…\n")
    evidence = "\n".join(evid_lines)

    stats = (f"检索模式：{mode}  |  top-{top_k}  |  耗时 {elapsed:.1f}s\n\n"
             f"上下文：{orig} → {after} 字符（保留 {ratio*100:.1f}%）")

    return ans, evidence, stats


def build_ui():
    with gr.Blocks(title="云南大学教务 RAG 问答") as demo:
        gr.Markdown(
            "# 云南大学计算机/软件学院 教务知识库问答\n"
            "检索增强生成演示 — 回答问题并展示检索证据。")
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="问题", lines=2,
                               placeholder="例如：推免生接收工作对综合测评成绩有什么要求？")
                with gr.Row():
                    mode = gr.Dropdown(["hybrid", "dense", "bm25"],
                                       value="hybrid", label="检索模式")
                    top_k = gr.Slider(1, 10, value=5, step=1, label="top-k")
                    do_comp = gr.Checkbox(label="压缩上下文", value=False)
                btn = gr.Button("提问", variant="primary")
                ans = gr.Markdown(label="答案")
                stats = gr.Textbox(label="运行统计", lines=3)
            with gr.Column(scale=2):
                evid = gr.Markdown(label="检索证据")

        btn.click(answer_fn, inputs=[q, mode, do_comp, top_k],
                  outputs=[ans, evid, stats])
        q.submit(answer_fn, inputs=[q, mode, do_comp, top_k],
                 outputs=[ans, evid, stats])
    return demo


def main():
    """Keep native launches private; Docker explicitly opts into 0.0.0.0."""
    host = os.environ.get("RAG_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("RAG_PORT", "7860"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("RAG_PORT must be between 1 and 65535.") from exc
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, share=False, show_error=False)


if __name__ == "__main__":
    main()
