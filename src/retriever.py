"""检索器：组合 dense / sparse / 混合三种模式。

对外只需 `retrieve(question, k, mode='bm25'|'dense'|'hybrid')`，
返回 sorted[(chunk, score)]。混合模式用加权融合（T4 扫 alpha）。
"""
import re


def default_tokenize(text):
    """中文字符二元组 + 英文/数字 token。"""
    tokens = []
    for run in re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", text):
        if re.fullmatch(r"[一-鿿]+", run):
            tokens.extend(run[i:i + 2] for i in range(max(1, len(run) - 1)))
        else:
            tokens.append(run.lower())
    return tokens


class Retriever:
    def __init__(self, index_store, tokenize=None):
        self.index = index_store
        self.tokenize = tokenize or default_tokenize

    def retrieve(self, question, k, mode="dense",
                 alpha=None, query_vec=None):
        """mode: 'dense' | 'bm25' | 'hybrid'。
        alpha: 混合中 dense 的权重，默认 config.DEFAULT_FUSION_ALPHA。
        query_vec: 已算好的稠密向量（避免重复编码）。
        """
        if k <= 0 or not len(self.index):
            return []
        if mode == "dense":
            vec = query_vec if query_vec is not None else index_dense(self.index, question)
            ordered = self.index.search_dense(vec, k)
        elif mode == "bm25":
            ordered = self.index.search_bm25(self.tokenize(question), k, self.tokenize)
        elif mode == "hybrid":
            ordered = self._hybrid(question, k, alpha, query_vec)
        else:
            raise ValueError(mode)

        return [(self.index.chunks[i], score) for i, score in ordered]

    def _hybrid(self, question, k, alpha, query_vec):
        from . import config
        alpha = config.DEFAULT_FUSION_ALPHA if alpha is None else alpha
        if not 0 <= alpha <= 1:
            raise ValueError("alpha 必须在 [0, 1] 内")
        if alpha == 0:
            return self.index.search_bm25(self.tokenize(question), k, self.tokenize)
        vec = query_vec if query_vec is not None else index_dense(self.index, question)
        if alpha == 1:
            return self.index.search_dense(vec, k)
        # 小型语料库对全部分块融合；返回条数不改变候选集合及前缀排序。
        n = len(self.index)
        dense = dict(self.index.search_dense(vec, n))
        bm25 = dict(self.index.search_bm25(self.tokenize(question), n, self.tokenize))
        scores = [(i, alpha * dense[i] + (1 - alpha) * bm25[i]) for i in range(n)]
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:k]


def index_dense(index_store, question):
    """对单个问题做 dense 查询所需的查询向量（供复用）。"""
    from . import embed
    return embed.embed_text_one(question)