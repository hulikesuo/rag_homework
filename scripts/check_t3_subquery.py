"""T3 subquery 集成离线测试（不加载模型/不调用真实 API）。

覆盖 run() 三条主路径：综合成功、综合为 None/异常且不额外普通生成、
部分失败保留 trace；同时验证 expanded context 不污染 RRF top5 指标、
gold 不传给答案生成 helper、config.require_llm 先于索引构建、
离线默认不触发任何 LLM 调用。
"""
import csv
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class StubPipeline:
    """RAGPipeline 的最小替身：只提供 chunks 和按查询排序的 hits。"""

    def __init__(self, chunks, hit_func):
        self.chunks = chunks
        self._hit_func = hit_func

    def retrieve(self, question, k=5, mode="dense", alpha=None):
        hits = self._hit_func(question.strip())
        return list(hits[:k])


def _setup_chunks():
    docs = {"docA": "第一段常识正确答案原文可靠证据补充常识"}
    chunks = [
        {"chunk_id": "c1", "doc": "docA", "text": "第一段常识", "start": 0, "end": 6},
        {"chunk_id": "c_gold", "doc": "docA", "text": "正确答案原文", "start": 7, "end": 15},
        {"chunk_id": "c2", "doc": "docA", "text": "可靠证据", "start": 16, "end": 21},
        {"chunk_id": "c3", "doc": "docA", "text": "补充常识", "start": 22, "end": 27},
    ]
    return docs, chunks


def _base_q():
    return {
        "id": "q01", "level": "L3",
        "question": "原问题",
        "gold_answer": "正确答案",
        "gold_chunk_ids": ["c_gold"],
    }


def _hit_for(question, non_gold=True):
    _, chunks = _setup_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = ["c1", "c2", "c3"] if non_gold else ["c_gold", "c1", "c2"]
    return [(dict(by_id[i]), 1.0 / (idx + 1)) for idx, i in enumerate(ids)]


def _fake_pipeline_factory(non_gold=True):
    docs, chunks = _setup_chunks()
    def factory(_docs):
        return StubPipeline(chunks, lambda q: _hit_for(q, non_gold))
    return factory


class T3OfflineTest(unittest.TestCase):
    """t3 模块导入级测试，不调用 run()，校验行为契约。"""

    def setUp(self):
        self.mod = self._fresh_module()

    @staticmethod
    def _fresh_module():
        sys.modules.pop("experiments.t3_query_opt", None)
        sys.modules.pop("experiments.common", None)
        from experiments import t3_query_opt
        return t3_query_opt

    def test_empty_trace_has_no_extra_column(self):
        allowed = set(self.mod.CASE_FIELDS)
        for key in self.mod._empty_subquery_trace():
            self.assertIn(key, allowed)

    def test_score_answer_empty_marks_error(self):
        row = self.mod._score_answer("   ", 0.5, "黄金")
        self.assertEqual(row["generation_status"], "error")
        self.assertEqual(row["answer"], "")
        self.assertEqual(row["EM_proxy"], "")
        self.assertEqual(row["answer_time_s"], 0.5)

    def test_score_answer_uses_gold(self):
        row = self.mod._score_answer("正确答案，包含更多", 1.2, "正确答案")
        self.assertEqual(row["generation_status"], "ok")
        self.assertGreater(row["ROUGE-L_proxy"], 0.0)

    def test_subquery_synthesis_returns_three_tuple_without_time_field(self):
        def answer_subq(question, subqueries, pipe, final_chunks):
            return {"answer": "综合答案", "subanswers": [], "context": [],
                    "errors": [], "fallback": False, "fallback_reason": None}
        with patch.object(self.mod.subq, "answer_subqueries",
                          side_effect=answer_subq), \
             patch.object(self.mod, "perf_counter",
                          side_effect=[1.0, 7.0]) as clock:
            _, trace, elapsed = self.mod._subquery_synthesis_case(
                "Q", ["sq1"], StubPipeline([], lambda q: []), [])
            self.assertNotIn("answer_time_s", trace)
            self.assertEqual(elapsed, 6.0)
            self.assertEqual(clock.call_count, 2)

    def test_dictwriter_default_rejects_extra_column(self):
        # save_csv 用 DictWriter 默认行为，不允许静默忽略额外列。
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=self.mod.CASE_FIELDS)
        row = {k: "" for k in self.mod.CASE_FIELDS}
        row["unexpected"] = "x"
        with self.assertRaises(ValueError):
            w.writerow(row)


class _RunHarness:
    """包裹 run() 所需的所有对象与 mock，复用注入逻辑。"""

    def __init__(self, answer_subq=None, complete_lines=None,
                 generate_answer="普通生成答案", non_gold=True,
                 config_error=None):
        self.mod = T3OfflineTest._fresh_module()
        self.qas = [_base_q()]
        self.docs, self.chunks = _setup_chunks()
        self.answer_subq = answer_subq
        self.complete_lines = complete_lines or {}
        self.generate_answer = generate_answer
        self.captured_final = []
        self.patched = []
        self.answer_subq_calls = 0
        self.complete_calls = 0
        self.generate_rag_calls = 0
        self.load_env_calls = 0
        self.pipeline_calls = 0
        self.tmp = tempfile.TemporaryDirectory(prefix="t3_check_")
        self._apply_patches(config_error=config_error, non_gold=non_gold)
        self.add_patch = None

    def _apply_patches(self, config_error, non_gold):
        m = self.mod
        self.patched.append(patch.object(m.config, "RESULTS_DIR", self.tmp.name))
        self.patched.append(patch.object(m.config, "require_llm",
                                         side_effect=config_error or (lambda: None)))
        self.patched.append(patch.object(m.config, "LLM_MODEL", "stub-model"))
        self.patched.append(patch.object(m, "load_env", side_effect=self._load_env))
        real_factory = _fake_pipeline_factory(non_gold)
        self.patched.append(patch.object(
            m.pipeline, "RAGPipeline",
            side_effect=lambda docs: self._pipeline(docs, real_factory)))
        self.patched.append(patch.object(m.gen, "complete",
                                         side_effect=self._complete))
        self.patched.append(patch.object(m.gen, "generate_rag",
                                         side_effect=self._fail_or_generate))
        if self.answer_subq is None:
            self.patched.append(patch.object(
                m.subq, "answer_subqueries",
                side_effect=self._answer_subq_must_not_call))
        else:
            self.patched.append(patch.object(
                m.subq, "answer_subqueries",
                side_effect=self._answer_subq_fake))
        for p in self.patched:
            p.start()

    def _load_env(self):
        self.load_env_calls += 1
        return {"docs": self.docs, "qas": self.qas}

    def _pipeline(self, docs, factory):
        self.pipeline_calls += 1
        return factory(docs)

    def _answer_subq_must_not_call(self, *args, **kwargs):
        self.answer_subq_calls += 1
        raise RuntimeError("must not call answer_subqueries")

    def _answer_subq_fake(self, *args, **kwargs):
        self.answer_subq_calls += 1
        return self.answer_subq(*args, **kwargs)

    def _complete(self, prompt, max_tokens=1024):
        self.complete_calls += 1
        # 测试不关心真实 prompt 内容，只按策略需要的行数返回。
        for marker, lines in self.complete_lines.items():
            if marker in prompt:
                return lines
        return "默认查询"

    def _fail_or_generate(self, *args, **kwargs):
        self.generate_rag_calls += 1
        return self.generate_answer

    def run(self, do_llm=False, do_generate=False):
        return self.mod.run(do_llm=do_llm, do_generate=do_generate)

    def read_csv(self, name):
        path = Path(self.tmp.name) / name
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def case(self, strategy):
        rows = self.read_csv(f"t3_cases_{strategy}.csv")
        if len(rows) != 1:
            raise AssertionError(f"{strategy} case rows={len(rows)}")
        return rows[0]

    def stop(self):
        for p in reversed(self.patched):
            p.stop()
        self.tmp.cleanup()


class TestRetrieveRRF(unittest.TestCase):
    """验证 _retrieve 的 RRF 分数与去重规则。"""

    @staticmethod
    def _mk(cid):
        return {"chunk_id": cid, "text": "x", "doc": "", "seq": 0}

    @staticmethod
    def _pipe(route_map):
        class _Stub:
            def __init__(self, mp):
                self.mp = mp
            def retrieve(self, query, k, mode, **kw):
                return [(dict(chunk), score)
                        for chunk, score in self.mp.get((query, mode), [])]
        return _Stub(route_map)

    def test_multiple_routes_accumulate(self):
        m = {}
        for query, route, cid, rank, score in [
                ("qA", "dense", "x", 1, 0.9),
                ("qB", "dense", "y", 2, 0.8),
                ("qB", "bm25", "y", 2, 0.7)]:
            m.setdefault((query, route), []).append((self._mk(cid), score))
        mod = T3OfflineTest._fresh_module()
        out = mod._retrieve(self._pipe(m), ["qA", "qB"], True)
        # y 在 qB 的 dense+bm25 两条路线各 rank2，得分应高于 x 的单路 rank1。
        self.assertEqual([c["chunk_id"] for c in out], ["y", "x"])

    def test_same_list_dedup_no_double_count(self):
        m = {("q", "dense"): [(self._mk("x"), 0.9), (self._mk("x"), 0.8),
                               (self._mk("z"), 0.7)],
             ("q", "bm25"): [(self._mk("x"), 0.6)]}
        mod = T3OfflineTest._fresh_module()
        out = mod._retrieve(self._pipe(m), ["q"], True)
        self.assertEqual([c["chunk_id"] for c in out][:2], ["x", "z"])


class TestRunIntegration(unittest.TestCase):
    def test_offline_default_zero_llm_calls(self):
        harness = _RunHarness()
        try:
            harness.run(do_llm=False, do_generate=False)
            # offline 路径不应触碰任何生成/综合函数。
            self.assertEqual(harness.complete_calls, 0)
            self.assertEqual(harness.generate_rag_calls, 0)
            self.assertEqual(harness.answer_subq_calls, 0)
            self.assertEqual(harness.load_env_calls, 1)
            self.assertEqual(harness.pipeline_calls, 1)
        finally:
            harness.stop()

    def test_config_validated_before_pipeline_and_env(self):
        harness = _RunHarness(config_error=RuntimeError("no config"))
        try:
            with self.assertRaises(RuntimeError):
                harness.run(do_llm=True, do_generate=True)
            # require_llm 必须在构建任何模型/索引/环境前失败。
            self.assertEqual(harness.load_env_calls, 0)
            self.assertEqual(harness.pipeline_calls, 0)
            self.assertEqual(harness.complete_calls, 0)
            self.assertEqual(harness.generate_rag_calls, 0)
            self.assertEqual(harness.answer_subq_calls, 0)
        finally:
            harness.stop()

    def test_successful_subquery_keeps_context_out_of_metrics(self):
        captured = {"question": None, "subqueries": None, "pipe": None,
                    "final_chunks": None}

        def answer_subq(question, subqueries, pipe, final_chunks):
            captured["question"] = question
            captured["subqueries"] = list(subqueries)
            captured["pipe"] = pipe
            captured["final_chunks"] = [dict(c) for c in final_chunks]
            return {
                "answer": "正确答案",
                "subanswers": [{"query": "子问题1", "answer": "中间1",
                                "context": [],
                                "errors": []}],
                "context": [],
                "errors": [],
                "fallback": False,
                "fallback_reason": None,
            }

        h = _RunHarness(
            answer_subq=answer_subq,
            complete_lines={"拆成": "子问题1\n子问题2"},
            generate_answer="不应被普通生成覆盖",
        )
        try:
            h.run(do_llm=True, do_generate=True)
            row = h.case("subquery")
            self.assertEqual(row["generation_status"], "ok")
            self.assertEqual(row["answer"], "正确答案")
            self.assertEqual(row["EM_proxy"], "1.0")
            self.assertEqual(row["ROUGE-L_proxy"], "1.0")
            self.assertNotIn("c_gold", json.loads(row["retrieved"]))
            self.assertEqual(row["Recall@5"], "0.0")
            self.assertEqual(row["Hit@5"], "0.0")
            self.assertTrue(float(row["answer_time_s"]) > 0.0)
            # 传给 helper 的就是 RRF top5，不能是 gold。
            self.assertEqual(captured["question"], "原问题")
            self.assertEqual(captured["subqueries"],
                             ["原问题", "子问题1", "子问题2"])
            self.assertIsNotNone(captured["pipe"])
            self.assertNotIn("gold_answer", captured["final_chunks"][0])
            self.assertNotIn("c_gold", [c["chunk_id"] for c in captured["final_chunks"]])
            # subquery 行只允许列内字段，任何额外列都会让 save_csv 失败。
            self.assertEqual(set(row), set(h.mod.CASE_FIELDS))
            # 从落盘 CSV 里确认综合答案确实写出来了（上面 case() 已读取）。
            self.assertEqual(row["answer"], "正确答案")
        finally:
            h.stop()

    def test_empty_subquery_answer_no_extra_generation(self):
        captured = []

        def answer_subq(question, subqueries, pipe, final_chunks):
            captured.append(list(final_chunks))
            return {
                "answer": None,
                "subanswers": [{"query": "子1", "answer": None,
                                "context": [], "errors": [
                                    {"stage": "synthesis_generation",
                                     "query": "原问题",
                                     "error_type": "InvalidGenerationError"}]}],
                "context": [],
                "errors": [{"stage": "synthesis_generation", "query": "原问题",
                            "error_type": "InvalidGenerationError"}],
                "fallback": True,
                "fallback_reason": "synthesis_failed",
            }

        h = _RunHarness(
            answer_subq=answer_subq,
            complete_lines={"拆成": "子1"},
            generate_answer="普通生成",
        )
        try:
            h.run(do_llm=True, do_generate=True)
            row = h.case("subquery")
            self.assertEqual(row["generation_status"], "error")
            self.assertEqual(row["answer"], "")
            self.assertEqual(row["EM_proxy"], "")
            self.assertEqual(row["ROUGE-L_proxy"], "")
            # 没有触发普通 generate_rag 的额外兜底。
            self.assertEqual(h.generate_rag_calls, 3)  # baseline/rewrite/multi
            self.assertEqual(h.answer_subq_calls, 1)
            self.assertIn("synthesis_failed", row["subquery_fallback_reason"])
            self.assertEqual(json.loads(row["subquery_errors"])[0]["stage"],
                             "synthesis_generation")
            # 普通策略答案/状态不受影响。
            for strategy, expected in [("baseline", "普通生成"),
                                       ("rewrite", "普通生成"),
                                       ("multi", "普通生成")]:
                c = h.case(strategy)
                self.assertEqual(c["generation_status"], "ok")
                self.assertEqual(c["answer"], expected)
        finally:
            h.stop()

    def test_partial_failure_preserved_in_trace(self):
        def answer_subq(question, subqueries, pipe, final_chunks):
            return {
                "answer": "部分成功答案",
                "subanswers": [
                    {"query": "子1", "answer": "成功", "context": [],
                     "errors": []},
                    {"query": "子2", "answer": None, "context": [],
                     "errors": [{"stage": "subquery_generation", "query": "子2",
                                 "error_type": "TimeoutError"}]},
                ],
                "context": [],
                "errors": [{"stage": "subquery_generation", "query": "子2",
                            "error_type": "TimeoutError"}],
                "fallback": False,
                "fallback_reason": None,
            }

        h = _RunHarness(
            answer_subq=answer_subq,
            complete_lines={"拆成": "子1\n子2"},
        )
        try:
            h.run(do_llm=True, do_generate=True)
            row = h.case("subquery")
            self.assertEqual(row["generation_status"], "ok")
            self.assertNotEqual(row["ROUGE-L_proxy"], "")
            answers = json.loads(row["subquery_answers"])
            self.assertEqual(answers[0]["answer"], "成功")
            self.assertIsNone(answers[1]["answer"])
            errors = json.loads(row["subquery_errors"])
            self.assertEqual(errors[0]["stage"], "subquery_generation")
            self.assertEqual(h.generate_rag_calls, 3)
            self.assertEqual(h.answer_subq_calls, 1)
        finally:
            h.stop()

    def test_subquery_exception_falls_back_without_hiding_error(self):
        def answer_subq(question, subqueries, pipe, final_chunks):
            raise RuntimeError("boom")

        h = _RunHarness(
            answer_subq=answer_subq,
            complete_lines={"拆成": "子1"},
        )
        try:
            h.run(do_llm=True, do_generate=True)
            row = h.case("subquery")
            self.assertEqual(row["generation_status"], "error")
            self.assertEqual(row["answer"], "")
            self.assertEqual(row["EM_proxy"], "")
            self.assertEqual(row["ROUGE-L_proxy"], "")
            self.assertEqual(row["subquery_fallback_reason"],
                             "answer_subqueries_raised")
            self.assertEqual(h.generate_rag_calls, 3)
            self.assertEqual(h.answer_subq_calls, 1)
        finally:
            h.stop()

    def test_empty_string_subquery_answer_error(self):
        def answer_subq(question, subqueries, pipe, final_chunks):
            return {"answer": "  ", "subanswers": [], "context": [],
                    "errors": [], "fallback": True,
                    "fallback_reason": "synthesis_empty"}

        h = _RunHarness(
            answer_subq=answer_subq,
            complete_lines={"拆成": "子1"},
            generate_answer="普通生成",
        )
        try:
            h.run(do_llm=True, do_generate=True)
            row = h.case("subquery")
            self.assertEqual(row["generation_status"], "error")
            self.assertEqual(row["answer"], "")
            self.assertEqual(row["EM_proxy"], "")
            self.assertEqual(row["ROUGE-L_proxy"], "")
            self.assertEqual(h.generate_rag_calls, 3)
            self.assertEqual(h.answer_subq_calls, 1)
        finally:
            h.stop()


if __name__ == "__main__":
    if sys.stdout and not sys.stdout.encoding.lower().startswith(("utf", "utf8")):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and sys.stderr.encoding.lower().startswith("utf"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    unittest.main(verbosity=2)
