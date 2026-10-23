"""检索索引：构建 + 查询。

统一接口：`build(chunks)`、`search(query_vec, k)`。
当前实现 Dense（FAISS-L2） + Sparse（BM25）双索引；
后续加混合只用这里的打分，不改 retriever 结构。
"""
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:          # noqa
    faiss = None
    _HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:          # noqa
    _HAS_BM25 = False


class IndexStore:
    """同时维护 dense 和 bm25 两套索引，按 chunks 顺序对齐。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.doc_texts = [c["text"] for c in chunks]
        self._dense = None
        self._bm25 = None

    # ---- 构建 ----
    def build_dense(self, vectors):
        if not _HAS_FAISS:
            raise RuntimeError("需要 faiss：`pip install faiss-cpu`")
        dim = vectors.shape[1]
        self._dense = faiss.IndexFlatL2(dim)
        self._dense.add(vectors.astype(np.float32))
        self._dense_vecs = vectors

    def build_bm25(self, tokenize):
        if not _HAS_BM25:
            raise RuntimeError("需要 rank-bm25：`pip install rank-bm25`")
        self._bm25 = BM25Okapi([tokenize(t) for t in self.doc_texts])

    # ---- Dense 查询（返回已排序 (idx, score)）----
    def search_dense(self, query_vec, k):
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        D, I = self._dense.search(q, min(k, len(self.chunks)))
        # L2 距离小=近，转为"分数高=相关"，便于与 bm25 统一
        return [(int(i), float(1.0 / (1.0 + d))) for i, d in zip(I[0], D[0])]

    # ---- BM25 查询 ----
    def search_bm25(self, query_tokens, k, tokenize):
        scores = self._bm25.get_scores(query_tokens)
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.chunks))]
        # 归一化到 0..1（min-max），与 dense 分数同尺度
        hi, lo = (scores.max(), scores.min()) if len(scores) else (0, 0)
        span = (hi - lo) or 1.0
        return [(int(i), float((scores[i] - lo) / span)) for i in order[:k]]

    def __len__(self):
        return len(self.chunks)