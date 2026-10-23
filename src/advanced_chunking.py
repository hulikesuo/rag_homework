"""Advanced chunking utilities for T2 (semantic split and Small-to-Big).

These are intentionally isolated from ``src.chunker`` so existing T1/T7
fingerprints and parallel experiments are not disturbed.

Semantic chunking splits document text at sentence boundaries using the
cosine similarity of adjacent sentence embeddings: a low similarity opens a
new chunk.  Small-to-Big builds small index chunks, retrieves on those small
chunks, and returns larger parent chunks for generation.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import chunker, embed, textutil


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) spans whose text is the stripped sentence."""
    spans: List[Tuple[int, int]] = []
    pos = 0
    for frag in textutil.split_sentences(text):
        if not frag:
            continue
        idx = text.find(frag, pos)
        if idx < 0:
            idx = text.find(frag)
            if idx < 0:
                continue
        raw = text[idx:idx + len(frag)]
        left = len(raw) - len(raw.lstrip())
        right = len(raw) - len(raw.rstrip())
        start, end = idx + left, idx + len(frag) - right
        if end <= start:
            continue
        spans.append((start, end))
        pos = idx + len(frag)
    return spans


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """True cosine similarity, robust to unnormalised embeddings."""
    dots = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dots / (na * nb)


def semantic_split_docs(
    docs: Dict[str, str],
    size: int = 512,
    threshold: float = 0.66,
    embed_fn: Optional[Callable[[Sequence[str]], List[Sequence[float]]]] = None,
) -> List[dict]:
    """Split each document into semantic chunks bounded by ``size`` chars."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if embed_fn is None:
        def embed_fn(texts):
            return embed.encode(list(texts))
    out: List[dict] = []
    for doc_name, text in docs.items():
        spans = _sentence_spans(text)
        if not spans:
            continue
        vecs = list(embed_fn([text[s:e] for s, e in spans]))
        seq = 0

        def emit(start: int, end: int) -> None:
            nonlocal seq
            out.append(chunker.new_chunk(
                doc_name, text[start:end], seq, start, end))
            seq += 1

        def split_long(seg_start: int, seg_end: int) -> None:
            """Split a single long sentence at exact character boundaries."""
            pos = seg_start
            while pos < seg_end:
                nxt = min(pos + size, seg_end)
                emit(pos, nxt)
                pos = nxt

        i = 0
        n = len(spans)
        while i < n:
            s, e = spans[i]
            if e - s > size:
                # A long sentence is a hard split point; keep it whole in
                # exact fixed-length pieces instead of expanding chunk size.
                split_long(s, e)
                i += 1
                continue
            cur_start, cur_end = s, e
            j = i + 1
            while j < n:
                ns, ne = spans[j]
                if ne - ns > size:
                    break
                candidate_len = ne - cur_start
                close = candidate_len > size or (
                    _cosine(vecs[j - 1], vecs[j]) < threshold)
                if close:
                    break
                cur_end = ne
                j += 1
            emit(cur_start, cur_end)
            i = j
    return out


def small_to_big_split_docs(
    docs: Dict[str, str],
    small_size: int = 512,
    small_overlap: int = 0,
    parent_factor: int = 4,
    parent_size: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, dict], Dict[str, str]]:
    """Build small index chunks and larger parent chunks with a chunk_id map.

    Returns ``(small_chunks, parents_by_id, small_id_to_parent_id)``.  Each
    small chunk carries ``parent_chunk_id``.  Retrieval indexes only the small
    chunks; generation should use the mapped parent chunks.
    """
    if small_size <= 0 or small_overlap < 0 or small_overlap >= small_size:
        raise ValueError("require small_size > 0 and 0 <= overlap < size")
    if parent_factor <= 0:
        raise ValueError("parent_factor must be positive")
    small_chunks: List[dict] = []
    parents_by_id: Dict[str, dict] = {}
    mapping: Dict[str, str] = {}
    for doc_name, text in docs.items():
        if not text:
            continue
        target = parent_size or max(small_size * parent_factor, 1024)
        # Segments become parents; every child is cut inside one parent so a
        # sub-chunk can never cross a parent boundary.
        parents = chunker._split(text, doc_name, target, 0, (
            r"\n\n", r"\n", r"[。！？；!?;.]"))
        doc_seq = 0
        for parent in parents:
            parents_by_id[parent["chunk_id"]] = parent
            pstart, ptext = parent["start"], parent["text"]
            step = small_size - small_overlap
            rel = 0
            while rel < len(ptext):
                rel_end = min(rel + small_size, len(ptext))
                c_start = pstart + rel
                c_end = pstart + rel_end
                cid = f"{parent['chunk_id']}-c{doc_seq:03d}"
                small_chunks.append({
                    "chunk_id": cid,
                    "text": ptext[rel:rel_end],
                    "doc": doc_name,
                    "seq": doc_seq,
                    "start": c_start,
                    "end": c_end,
                    "parent_chunk_id": parent["chunk_id"],
                    "parent_start": pstart,
                    "parent_end": parent["end"],
                })
                mapping[cid] = parent["chunk_id"]
                doc_seq += 1
                rel += step
    return small_chunks, parents_by_id, mapping


def build_docs_from_texts(docs: Dict[str, str]) -> Dict[str, str]:
    """Public helper; documents are kept as plain ``{doc_name: text}``."""
    return dict(docs)

