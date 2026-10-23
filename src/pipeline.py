"""基准 RAG 流水线：组合各模块为一次检索生成（T1）。

可通过参数指定分块方式 / 检索模式 / top-k / 是否重排压缩，
供各实验任务以最小代码复用它对不同配置做对比。
"""
from . import chunker, embed, index_store, generator, retriever, rerank


class RAGPipeline:
    """一次构建索引后可多次 query。保持各模块可替换。"""

    def __init__(self, docs, chunk_method="fixed", chunk_size=512,
                 chunk_overlap=128, retriever_mode="dense"):
        from . import config
        self.docs = docs
        self.chunk_method = chunk_method
        chunks = chunker.chunk_documents(
            docs, method=chunk_method, size=chunk_size, overlap=chunk_overlap)
        self.chunks = chunks

        self.index = index_store.IndexStore(chunks)
        vecs = embed.encode([c["text"] for c in chunks])
        self.index.build_dense(vecs)
        self.index.build_bm25(retriever.default_tokenize)
        self.retriever = retriever.Retriever(self.index)

    def retrieve(self, question, k=5, mode="dense", alpha=None):
        return self.retriever.retrieve(question, k=k, mode=mode, alpha=alpha)

    def answer(self, question, k=5, mode="dense", top_k_final=None,
               do_rerank=False, do_compress=False, compress_budget=800):
        """一次完整回答。返回 dict 含答案、检索结果、token/字符统计。"""
        hits = self.retrieve(question, k, mode=mode)
        context_chunks = [c for c, _ in hits]

        if do_rerank:
            context_chunks = [c for c, _ in rerank.rerank_judge_llm(question, hits, top_k_final)]
        elif top_k_final and top_k_final < len(context_chunks):
            context_chunks = context_chunks[:top_k_final]

        if do_compress:
            from .rerank import compress_extractive
            new_text, orig, after, ratio = compress_extractive(
                question, context_chunks, compress_budget)
            pseudo = chunker.new_chunk("compressed", new_text, 0)
            context_chunks = [pseudo]
            tokens = {"orig_chars": orig, "after_chars": after, "ratio": ratio}
        else:
            tokens = {"orig_chars": sum(len(c["text"]) for c in context_chunks),
                      "after_chars": sum(len(c["text"]) for c in context_chunks),
                      "ratio": 1.0}

        answer = generator.generate_rag(question, context_chunks)
        return {
            "question": question,
            "answer": answer,
            "retrieved": [c["chunk_id"] for c, _ in hits],
            "retrieved_scores": [round(s, 4) for _, s in hits],
            "context_chunks": [c["chunk_id"] for c in context_chunks],
            "token_stats": tokens,
        }


def build_pipeline(docs, **kw):
    return RAGPipeline(docs, **kw)