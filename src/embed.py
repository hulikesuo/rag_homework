"""向量化接口。

统一暴露 `encode(texts) -> list[np.ndarray]`，使 dense 检索与 index 解耦。
懒加载模型，避免 import 即下载。
"""
import numpy as np

from . import config

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("需要 sentence-transformers：`pip install sentence-transformers`") from e
        _encoder = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _encoder


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def encode(texts):
    """对一批文本编码，返回归一化后的 (n, dim) 数组。"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    enc = _get_encoder()
    vecs = enc.encode(texts, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def count_tokens(text):
    """按嵌入模型 tokenizer 计数，不等同于生成模型的计费 token。"""
    return len(_get_encoder().tokenizer.encode(
        text, add_special_tokens=False, truncation=False, verbose=False))


def embed_text_one(text):
    return encode([text])[0]