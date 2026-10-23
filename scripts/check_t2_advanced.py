#!/usr/bin/env python3
"""Read-only acceptance auditor for the advanced T2 chunking experiment.

This checks the isolated new files and their outputs without modifying
``results/`` or any shared source.  It deliberately treats the existing formal
15-config results as untouched history.
"""

from __future__ import annotations

import csv
import os
import hashlib
import json
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data" / "eval" / "qa_set.jsonl"
SOURCES = [
    ROOT / "src" / "advanced_chunking.py",
    ROOT / "experiments" / "t2_advanced_chunking.py",
    ROOT / "src" / "chunker.py",
]

from src import advanced_chunking, chunker, textutil

OUT = ROOT / "repair_outputs" / "acceptance_20261023" / "t2"
DEFAULT_OUT = OUT


def log(msg: str) -> None:
    print(msg, flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(errors: List[str], label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        errors.append(detail or label)


def real_docs() -> Dict[str, str]:
    from src import loader, config
    return loader.read_all_docs(config.CORPUS_DIR)


def verify_semantic_unit(errors: List[str]) -> None:
    """Fast deterministic unit tests; do not load any embedding model."""
    # Adjacent vector similarity controls semantic boundaries.
    tiny = {"d1": "主题甲。主题甲续。主题乙。"}
    size = 128

    def embed_high(texts):
        return [[1.0, 0.0] for _ in texts]

    chunks_high = advanced_chunking.semantic_split_docs(
        tiny, size=size, embed_fn=embed_high)
    fail(errors, "semantic/high", len(chunks_high) == 1,
         f"expected 1 high-similarity chunk, got {len(chunks_high)}")

    def embed_low(texts):
        return [[1.0, 0.0] if "主题甲" in t else [0.0, 1.0] for t in texts]

    chunks_low = advanced_chunking.semantic_split_docs(
        tiny, size=size, embed_fn=embed_low)
    fail(errors, "semantic/low", len(chunks_low) == 2,
         f"expected 2 chunks when adjacent similarity drops, got "
         f"{len(chunks_low)}")
    for c in chunks_low:
        fail(errors, "semantic/low/span",
             tiny[c["doc"]][c["start"]:c["end"]] == c["text"],
             "semantic low span mismatch")

    # Very long sentences must be split into exact ``size`` pieces and the
    # resulting chunks must stay within the requested size.
    long_doc = {"d1": ("很长很长的内容" * 100) + "。第二句。"}
    for size in (64, 128):
        chunks = advanced_chunking.semantic_split_docs(
            long_doc, size=size, embed_fn=embed_high)
        fail(errors, f"semantic/long/{size}", len(chunks) >= 2,
             f"long sentence should split into at least 2 chunks")
        for c in chunks:
            fail(errors, f"semantic/long/{size}/len",
                 len(c["text"]) <= size,
                 f"long chunk length {len(c['text'])} > {size}")
            fail(errors, f"semantic/long/{size}/span",
                 long_doc[c["doc"]][c["start"]:c["end"]] == c["text"],
                 "long semantic span mismatch")

    # Whitespace/newline gaps between sentences are preserved exactly in the
    # chunk text and do not inflate length past the limit.
    gaps = {"d1": "句一。\n\n\n句二。\n\n句三。"}
    for size in (64, 128):
        chunks = advanced_chunking.semantic_split_docs(
            gaps, size=size, embed_fn=embed_high)
        for c in chunks:
            fail(errors, f"semantic/gap/{size}/len",
                 len(c["text"]) <= size,
                 f"gap chunk length {len(c['text'])} > {size}")
            fail(errors, f"semantic/gap/{size}/span",
                 gaps[c["doc"]][c["start"]:c["end"]] == c["text"],
                 "gap semantic span mismatch")

    # Scaled-nonunit vectors must still behave consistently (true cosine).
    # Parallel scaled vectors must be cosine ~1, orthogonal must be ~0 even
    # when magnitudes differ a lot.
    fail(errors, "true_cosine_parallel",
         advanced_chunking._cosine([3.0, 4.0], [6.0, 8.0]) > 0.99,
         "parallel vectors should have cosine ~1")
    fail(errors, "true_cosine_orthogonal",
         abs(advanced_chunking._cosine([12.0, 5.0], [-10.0, 24.0])) < 1e-6,
         "orthogonal vectors should have cosine ~0")


def verify_small_to_big_unit(errors: List[str]) -> None:
    # Sparse doc: multiple parents, every child fully contained in its
    # parent, and children never cross a parent boundary.
    sparse = {"d": "甲。\n\n乙。\n\n丙。\n\n丁。\n\n戊。\n\n己。\n\n庚。\n\n辛。"}
    small, parents, mapping = (
        advanced_chunking.small_to_big_split_docs(
            sparse, small_size=16, parent_size=8))
    fail(errors, "small-to-big sparse", len(parents) > 1,
         "sparse doc has only one parent")
    for c in small:
        p = parents.get(c["parent_chunk_id"], {})
        fail(errors, "small-to-big sparse contain",
             p.get("start", -1) <= c["start"] and c["end"] <= p.get("end", -1),
             f"sparse child {c['chunk_id']} not contained in parent")
        fail(errors, "small-to-big mapping",
             mapping.get(c["chunk_id"]) == c.get("parent_chunk_id"),
             "small-to-big mapping mismatch")
    # Contiguity: all child pieces of one parent must sum, aside from
    # overlap, into the parent's exact text length.
    for pid, parent in parents.items():
        children = [c for c in small if c["parent_chunk_id"] == pid]
        fail(errors, f"small-to-big {pid}",
             len(children) > 0, "parent has no children")
        if children:
            step = 16
            expect = (parent["end"] - parent["start"] + step - 1) // step
            fail(errors, f"small-to-big {pid} count",
                 len(children) == expect,
                 f"parent {pid}: children {len(children)} != expected "
                 f"{expect}")


def verify_real_semantic_integration(errors: List[str]) -> None:
    """Run real-doc structural checks. Loads the embedding model, so call
    with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 to fail fast if not cached."""
    use_local_encoder()
    docs = real_docs()
    log(f"[integration] docs={len(docs)}")
    for size in (256, 512, 1024):
        chunks = advanced_chunking.semantic_split_docs(docs, size=size)
        fail(errors, f"semantic/{size}", len(chunks) > 0,
             "semantic produced no chunks")
        for c in chunks:
            fail(errors, f"semantic/{size}", len(c["text"]) <= size,
                 "semantic chunk too large")
            doc = docs.get(c["doc"])
            if doc is not None:
                fail(errors, f"semantic/{size}/span",
                     doc[c["start"]:c["end"]] == c["text"],
                     "semantic span does not equal text")


def verify_real_stb_integration(errors: List[str]) -> None:
    use_local_encoder()
    docs = real_docs()
    for size in (256, 512, 1024):
        small, parents, mapping = (
            advanced_chunking.small_to_big_split_docs(docs, small_size=size))
        fail(errors, f"small_to_big/{size}",
             len(small) > 0 and len(parents) > 0,
             "small_to_big produced no chunks")
        seen_parents = set()
        for c in small:
            pid = c.get("parent_chunk_id")
            fail(errors, f"small_to_big/{size}",
                 pid and pid in parents
                 and c["chunk_id"] in mapping
                 and mapping[c["chunk_id"]] == pid,
                 f"small chunk missing/invalid parent {pid!r}")
            if pid:
                seen_parents.add(pid)
            parent = parents.get(pid or "")
            if parent:
                # Every child must be fully contained in its parent span.
                contained = (parent["start"] <= c["start"]
                             and c["end"] <= parent["end"])
                fail(errors, f"small_to_big/{size}/contain",
                     contained, f"child {c['chunk_id']} not contained "
                     f"in parent {pid}: child=({c['start']},{c['end']}) "
                     f"parent=({parent['start']},{parent['end']})")
        fail(errors, f"small_to_big/{size}/diversity",
             len(seen_parents) > 1,
             f"only one parent used for {len(small)} small chunks")


def use_local_encoder() -> None:
    snapshot = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--BAAI--bge-small-zh-v1.5"
        "/snapshots/7999e1d3359715c523056ef9478215996d62a620")
    from src import embed
    if not snapshot.is_dir():
        raise RuntimeError(f"offline snapshot missing: {snapshot}")
    from sentence_transformers import SentenceTransformer
    embed._encoder = SentenceTransformer(str(snapshot), local_files_only=True)


def verify_outputs(errors: List[str], out: Path) -> None:
    cases = out / "t2_advanced_chunking_cases_final.csv"
    summary = out / "t2_advanced_chunking_final.csv"
    if not cases.is_file():
        cases = out / "t2_advanced_chunking_cases.csv"
        summary = out / "t2_advanced_chunking.csv"
    manifest = out / "manifest.json"
    for name, path in (("cases", cases), ("summary", summary), ("manifest", manifest)):
        fail(errors, f"missing output {name}", path.is_file(),
             f"missing {path}")
    if summary.is_file():
        rows = read_csv(summary)
        desired = {"semantic", "small_to_big"}
        have = {r["method"] for r in rows}
        fail(errors, "summary methods",
             desired <= have, "summary methods missing")
        for size in (256, 512, 1024):
            for m in ("semantic", "small_to_big"):
                matched = [r for r in rows
                           if r["method"] == m
                           and r["size_for_index"] == str(size)]
                fail(errors, f"{m}/{size} summary",
                     len(matched) == 1, "summary row missing/duplicate")
    if cases.is_file():
        rows = read_csv(cases)
        if len(rows) < 180:
            fail(errors, "case row count", False,
                 f"expected >=180 rows, got {len(rows)}")
        for r in rows:
            if r["generation_status"] != "ok":
                fail(errors, f"case {r.get('id')}/{r.get('method')}",
                     False, "non-ok generation status")
            if r["generation_status"] == "ok" and not str(r.get("answer", "")).strip():
                fail(errors, f"case {r.get('id')}/{r.get('method')}/empty",
                     False, "non-empty check failed")


def verify_manifest(errors: List[str], out: Path) -> None:
    if not (out / "manifest.json").is_file():
        return
    mf = read_json(out / "manifest.json")
    qa = mf.get("qa_set", {})
    fail(errors, "manifest qa hash",
         qa.get("sha256") == sha(QA), "manifest qa hash mismatch")
    src_hash = mf.get("sources", {})
    for path in SOURCES:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        fail(errors, f"manifest src {rel}",
             src_hash.get(rel) == sha(path), "source hash mismatch")


def verify_provenance(errors: List[str], out: Path) -> None:
    prov = out / "recovery_provenance.json"
    if not prov.is_file():
        fail(errors, "recovery_provenance.json", False, "missing provenance")
        return
    data = read_json(prov)
    for fname, field in (
            ("pristine_t2_advanced_chunking_cases.csv",
             "original_cases_sha256"),
            ("pristine_t2_advanced_chunking.csv",
             "original_summary_sha256")):
        p = out / fname
        fail(errors, f"pristine {fname}", p.is_file(),
             f"missing pristine {fname}")
        if p.is_file():
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            fail(errors, f"pristine hash {fname}",
                 actual == data.get(field),
                 f"pristine hash mismatch for {fname}")


def verify_plots(errors: List[str], out: Path) -> None:
    """Check plot-source JSON values match the final summary CSV exactly."""
    adv = out / "fig_t2_advanced_chunking.plot_source.json"
    combined = out / "fig_t2_21_configs.plot_source.json"
    if not adv.is_file() or not combined.is_file():
        fail(errors, "plot source json", False, "missing plot source json")
        return
    summary = read_csv(out / "t2_advanced_chunking_final.csv")
    row = {(r["method"], int(r["size_for_index"])): r for r in summary}
    adv_data = read_json(adv)
    if not adv_data.get("series"):
        fail(errors, "plot source series", False, "empty adv series")
    sem_parent = [x for x in adv_data["series"]
                  if x.get("panel") == "parent_coverage"
                  and x.get("method") == "semantic"]
    if sem_parent:
        for item in sem_parent:
            if item.get("value") is not None:
                fail(errors, "plot source semantic parent null",
                     False, f"semantic parent coverage should be null, got "
                             f"{item['value']}")
            if not item.get("note"):
                fail(errors, "plot source semantic parent note",
                     False, "semantic parent coverage lacks N/A note")
    for item in adv_data["series"]:
        rec = row.get((item["method"], int(item["size"])))
        if rec is None:
            fail(errors, "plot source row missing",
                 False, f"missing {item}")
            continue
        col_map = {
            "small_coverage": "Recall@5_small_evidence",
            "parent_coverage": "Recall@5_parent_evidence",
            "small_hit": "Hit@5_small_evidence",
            "rouge": "ROUGE-L_proxy",
        }
        if item.get("panel") == "parent_coverage" and item["method"] == "semantic":
            if item.get("value") is not None:
                fail(errors, "plot source semantic parent",
                     False, f"semantic parent coverage should be null, got "
                             f"{item['value']}")
            if not item.get("note"):
                fail(errors, "plot source semantic parent note",
                     False, "semantic parent coverage lacks N/A note")
            continue
        actual = rec.get(col_map[item["panel"]], "")
        try:
            expected = float(actual if actual not in ("", None) else 0.0)
        except (TypeError, ValueError):
            expected = None
        if item["value"] != expected:
            fail(errors, "plot source value",
                 False, f"{item} != csv {expected}")
    c_data = read_json(combined)
    cmap = {(r["method"], int(r["size"] or 0)): r for r in c_data.get("rows", [])}
    signatures = {
        ("semantic", 256): ("Recall@5_small_evidence", None),
        ("semantic", 512): ("Recall@5_small_evidence", None),
        ("semantic", 1024): ("Recall@5_small_evidence", None),
        ("small_to_big", 256): ("Recall@5_parent_evidence", "Hit@5_parent_evidence"),
        ("small_to_big", 512): ("Recall@5_parent_evidence", "Hit@5_parent_evidence"),
        ("small_to_big", 1024): ("Recall@5_parent_evidence", "Hit@5_parent_evidence"),
    }
    for (m, s), (cov, hit) in signatures.items():
        rec = row.get((m, s))
        plot_row = cmap.get((m, s))
        if plot_row is None or rec is None:
            continue
        cov_csv = float(rec.get(cov, 0) or 0)
        if abs(plot_row["coverage"] - cov_csv) > 1e-9:
            fail(errors, "21 plot coverage",
                 False, f"{m}/{s}: plot {plot_row['coverage']} != csv {cov_csv}")
        if hit and abs(plot_row["hit"] - float(rec.get(hit, 0) or 0)) > 1e-9:
            fail(errors, "21 plot hit",
                 False, f"{m}/{s}: plot hit mismatch")
        if abs(plot_row["rouge"] - float(rec.get("ROUGE-L_proxy", 0) or 0)) > 1e-9:
            fail(errors, "21 plot rouge",
                 False, f"{m}/{s}: plot rouge mismatch")


def main() -> int:
    global OUT
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-results", action="store_true",
                        help="verify real CSV/JSON outputs under the output dir")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="isolated output directory")
    parser.add_argument("--integration", action="store_true",
                        help="run real-corpus structure checks (loads models; "
                             "use HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)")
    args = parser.parse_args()
    OUT = args.out
    errors: List[str] = []
    log("[t2-advanced] unit: semantic")
    verify_semantic_unit(errors)
    log("[t2-advanced] unit: small-to-big")
    verify_small_to_big_unit(errors)
    if args.integration:
        log("[t2-advanced] integration: semantic")
        verify_real_semantic_integration(errors)
        log("[t2-advanced] integration: small-to-big")
        verify_real_stb_integration(errors)
    if args.verify_results:
        log("[t2-advanced] verify outputs/manifest")
        verify_outputs(errors, OUT)
        verify_manifest(errors, OUT)
        verify_provenance(errors, OUT)
        verify_plots(errors, OUT)
    if errors:
        log("\nFAIL:")
        for e in errors:
            log("  - " + e)
        return 1
    log("ALL T2 ADVANCED CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

