"""公共小工具：路径安全、计时、JSON 读写。"""
import json
import os
import time
from contextlib import contextmanager

from . import config


def join_results(*parts):
    """生成 results/ 下的路径，自动建目录，防路径穿越。"""
    path = os.path.join(config.RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_ann_tsv(path, rows):
    """存标注对照表，用制表符分隔、段间避免歧义。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("chunk_id\tseq\tdoc\ttext\n")
        for cid, seq, doc, text in rows:
            text = text.replace("\t", " ").replace("\n", "⏎")
            f.write(f"{cid}\t{seq}\t{doc}\t{text}\n")


@contextmanager
def timed(tag):
    """计时上下文，用于记录某个步骤耗时（报告里要用）。"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[timed] {tag}: {elapsed:.3f}s")


def load_corpus_text():
    """返回 {doc_filename: full_text}，供分块器使用。"""
    from . import loader
    return loader.read_all_docs(config.CORPUS_DIR)