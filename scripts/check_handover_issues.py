"""Checks for the remaining handover issues; no models, APIs or results writes.

Run:
    python -I -B scripts/check_handover_issues.py --root .
"""

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SubqueryTraceChecks(unittest.TestCase):
    def test_failure_omits_exception_message(self):
        try:
            raise RuntimeError("secret-key at http://internal.example/v1")
        except RuntimeError as exc:
            trace = _failure("subquery_retrieval", exc, "问题")
        self.assertEqual(trace, {
            "stage": "subquery_retrieval",
            "query": "问题",
            "error_type": "RuntimeError",
        })
        self.assertNotIn("secret", repr(trace))


class FigureCleanupChecks(unittest.TestCase):
    def test_t5_missing_csv_removes_stale_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_res = make_figures.RES
            make_figures.RES = tmp
            try:
                stale = Path(tmp) / "fig_t5_rerank_proxy.png"
                stale.write_bytes(b"stale")
                make_figures.fig_t5()
                self.assertFalse(stale.exists())
            finally:
                make_figures.RES = old_res

    def test_t5_illegal_values_remove_stale_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_res = make_figures.RES
            make_figures.RES = tmp
            try:
                with open(Path(tmp) / "t5_rerank.csv", "w",
                          encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["id", "variant", "recall@3", "rerank_ms"])
                    writer.writeheader()
                    writer.writerow({"id": "q01", "variant": "direct",
                                     "recall@3": "nan", "rerank_ms": ""})
                stale = Path(tmp) / "fig_t5_rerank_proxy.png"
                stale.write_bytes(b"stale")
                make_figures.fig_t5()
                self.assertFalse(stale.exists())
            finally:
                make_figures.RES = old_res

    def test_t5_timing_panel_keeps_valid_ms_without_recall(self):
        rows = [
            {"variant": "direct", "recall@3": "0", "rerank_ms": "1.5"},
            {"variant": "rerank", "recall@3": "nan", "rerank_ms": "2.5"},
            {"variant": "cross_encoder", "recall@3": "0.5", "rerank_ms": "3.5"},
        ]
        panels = {key: groups for key, _, groups in make_figures._t5_panel_data(rows)}
        self.assertEqual(set(panels["recall@3"]), {"direct", "cross_encoder"})
        self.assertEqual(panels["rerank_ms"],
                         {"direct": [1.5], "rerank": [2.5],
                          "cross_encoder": [3.5]})


class GenerationOrderChecks(unittest.TestCase):
    def test_retrieval_failure_does_not_import_generator(self):
        # 在独立进程内替换导入器，避免污染后续用例的 sys.modules。
        probe = r"""
import builtins
import importlib
import sys
import types

class FailingPipe:
    def retrieve(self, question, **kwargs):
        raise RuntimeError("backend secret")

class FakeRunnableLambda:
    def __init__(self, fn):
        self.fn = fn
    def __or__(self, other):
        first, second = self.fn, other.fn
        return types.SimpleNamespace(
            invoke=lambda question: second(first(question)))

sys.modules["langchain_core"] = types.ModuleType("langchain_core")
sys.modules["langchain_core.runnables"] = types.SimpleNamespace(
    RunnableLambda=FakeRunnableLambda)
sys.path.insert(0, ".")
chains = importlib.import_module("src.chains")
original_import = builtins.__import__

def fail_import(name, *args, **kwargs):
    if name == "src.generator" or name.endswith(".generator"):
        raise AssertionError("generator import must be skipped")
    return original_import(name, *args, **kwargs)

builtins.__import__ = fail_import
try:
    result = chains.build_baseline_chain(
        FailingPipe(), {"k": 5}).invoke("问题")
finally:
    builtins.__import__ = original_import
assert result["rag_error"] == "RuntimeError", result
assert "secret" not in repr(result), result
print("ok")
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", probe],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ok")


class JudgeFallbackChecks(unittest.TestCase):
    def test_bad_llm_score_does_not_become_neutral_three(self):
        original = generator.complete

        def fake_complete(prompt, max_tokens=8):
            if "片段A" in prompt:
                return "not-a-score"
            return "5"

        generator.complete = fake_complete
        try:
            chunks = [{"chunk_id": "A", "text": "片段A"},
                      {"chunk_id": "B", "text": "片段B"}]
            ranked = rerank.rerank_judge_llm("问题", [(chunks[0], 0.99), (chunks[1], 0.1)])
            self.assertEqual([chunk["chunk_id"] for chunk, _ in ranked], ["B", "A"])
            self.assertEqual(ranked[1][1], 0.99)
        finally:
            generator.complete = original

    def test_judge_accepts_valid_score_variants(self):
        original = generator.complete
        generator.complete = lambda prompt, max_tokens=8: "5分"
        try:
            ranked = rerank.rerank_judge_llm(
                "问题", [({"chunk_id": "A", "text": "片段A"}, 0.1)])
        finally:
            generator.complete = original
        self.assertEqual(ranked, [({"chunk_id": "A", "text": "片段A"}, 5.0)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args, unittest_args = parser.parse_known_args()
    root = args.root.resolve(strict=True)
    if not (root / "src" / "subquery.py").is_file():
        parser.error("--root must contain src/subquery.py")
    sys.path.insert(0, str(root))
    from src import generator, rerank, subquery
    from scripts import make_figures

    ROOT = root
    _failure = subquery._failure
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
