"""Offline stub tests for demo.answer_fn edge/error paths.

These tests avoid launching the real Gradio service or consuming real API
credits for deterministic failure/edge inputs.  They patch loader/pipeline,
generator and compress paths so no HF model or network endpoint is touched.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class _StubPipe:
    """In-memory RAG pipeline fake with retrieve returning a chunk list."""

    def retrieve(self, question, k, mode):
        if mode not in ("hybrid", "dense", "bm25"):
            raise ValueError("unsupported mode")
        if int(k or 0) <= 0:
            raise ValueError("top_k must be positive")
        text = "云南大学教务知识库演示片段。"
        return [({"doc": "stub.docx", "text": text,
                  "chunk_id": "stub__001"}, 0.9)]


def _run_case(question: str, mode, do_compress: bool, top_k) -> str:
    sys.path.insert(0, str(ROOT))
    import importlib
    from demo import app
    app._PIPE = _StubPipe()
    return app.answer_fn(question, mode, do_compress, top_k)


class DemoOfflineEdgeTests(unittest.TestCase):
    def setUp(self):
        self._old_sys = list(sys.path)

    def tearDown(self):
        sys.path = self._old_sys
        from demo import app
        app._PIPE = None

    def test_empty(self):
        ans, ev, stats = _run_case("", "hybrid", False, 5)
        self.assertIn("请输入问题", ans)
        self.assertEqual(ev, "")

    def test_blank(self):
        ans, ev, stats = _run_case("   ", "dense", True, 3)
        self.assertIn("请输入问题", ans)

    def test_bad_mode(self):
        ans, ev, stats = _run_case("选课有什么要求？", "bogus", False, 5)
        self.assertNotEqual(ans, "")
        self.assertIn("失败", ans)

    def test_top_k_invalid(self):
        ans, ev, stats = _run_case("选课有什么要求？", "hybrid", False, 0)
        self.assertNotEqual(ans, "")
        self.assertIn("失败", ans)


if __name__ == "__main__":
    unittest.main()
