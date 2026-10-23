"""分块策略：保留原文字符范围，使不同配置的证据可以对齐。"""
import re


def new_chunk(doc_name, text, seq, start=None, end=None):
    chunk = {"chunk_id": f"{doc_name}__{seq:03d}", "text": text,
             "doc": doc_name, "seq": seq}
    if start is not None:
        chunk.update(start=start, end=end)
    return chunk


def _split(doc, doc_name, size, overlap, separators=()):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("要求 size > 0 且 0 <= overlap < size")
    chunks, start = [], 0
    while start < len(doc):
        end = min(start + size, len(doc))
        if end < len(doc) and separators:
            # 在窗口后半部优先寻找段落/句边界，直接截取原文，不改顺序。
            lower = start + max(overlap + 1, size // 2)
            for separator in separators:
                matches = list(re.finditer(separator, doc[lower:end]))
                if matches:
                    end = lower + matches[-1].end()
                    break
        chunks.append(new_chunk(doc_name, doc[start:end], len(chunks), start, end))
        if end == len(doc):
            break
        start = end - overlap
    return chunks


def chunk_fixed(doc, doc_name, size, overlap):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("要求 size > 0 且 0 <= overlap < size")
    # 保持既有 fixed 的滑窗步长和 chunk ID 顺序。
    starts = range(0, len(doc), size - overlap)
    return [new_chunk(doc_name, doc[start:start + size], seq, start,
                      min(start + size, len(doc)))
            for seq, start in enumerate(starts)]


def chunk_recursive(doc, doc_name, size, overlap):
    return _split(doc, doc_name, size, overlap,
                  (r"\n\n", r"\n", r"[。！？；!?]"))


def chunk_sentence(doc, doc_name, size, overlap):
    return _split(doc, doc_name, size, overlap, (r"[。！？；!?]",))


CHUNKERS = {"fixed": chunk_fixed, "recursive": chunk_recursive,
            "sentence": chunk_sentence}


def chunk_documents(docs, method="fixed", size=512, overlap=128):
    fn = CHUNKERS[method]
    return [chunk for doc_name, text in docs.items()
            for chunk in fn(text, doc_name, size, overlap)]
