# encoding: utf-8
"""Unit tests for llm_judge_t7 validation / summary / main-level integrity.

Uses standard library and stubs only; no real API calls, no CSV output.
Run: python scripts\check_t7_judge.py
"""

import hashlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "llm_judge_t7", ROOT / "scripts" / "llm_judge_t7.py")
_mod = importlib.util.module_from_spec(spec)
_mod.__dict__["transformers"] = None
_mod.src = MagicMock()
spec.loader.exec_module(_mod)

_sha256_str = _mod._sha256_str
_sha256_json = _mod._sha256_json
_hash_answer = _mod._hash_answer
_hash_context = _mod._hash_context
_input_hash = _mod._input_hash
_validate_one = _mod._validate_one
_extract_json = _mod._extract_json
_score_int = _mod._score_int
_claim_int = _mod._claim_int
_mean = _mod._mean

# ---- sample chunk_texts ----
SAMPLE_CHUNKS = {
    "doc1__001": "chunk text one",
    "doc1__002": "chunk text two",
    "doc2__001": "other chunk",
}


class TestHashFunctions(unittest.TestCase):
    def test_hash_answer_deterministic(self):
        self.assertEqual(_hash_answer("hello"), _hash_answer("hello"))

    def test_hash_answer_different(self):
        self.assertNotEqual(_hash_answer("a"), _hash_answer("b"))

    def test_hash_context_deterministic(self):
        self.assertEqual(_hash_context(["a", "b"]), _hash_context(["a", "b"]))

    def test_hash_context_order_matters(self):
        self.assertNotEqual(_hash_context(["a", "b"]), _hash_context(["b", "a"]))

    def test_input_hash_deterministic(self):
        rec = {"answer": "A", "context_chunks": ["c1"], "context_texts": ["t1"]}
        self.assertEqual(_input_hash("Q", "G", rec, "fp"),
                         _input_hash("Q", "G", rec, "fp"))

    def test_input_hash_changes_with_answer(self):
        r1 = {"answer": "A", "context_chunks": [], "context_texts": []}
        r2 = {"answer": "B", "context_chunks": [], "context_texts": []}
        self.assertNotEqual(_input_hash("Q", "G", r1, "fp"),
                            _input_hash("Q", "G", r2, "fp"))

    def test_sha256_str_len(self):
        self.assertEqual(len(_sha256_str("x")), 64)

    def test_sha256_json_len(self):
        self.assertEqual(len(_sha256_json({"a": 1})), 64)


class TestValidateOne(unittest.TestCase):
    def make_q(self, **kw):
        base = {"id": "q01", "level": "L1", "question": "Q?", "gold_answer": "A",
                "gold_chunk_ids": ["doc1__001"]}
        base.update(kw)
        return base

    def make_rec(self, **kw):
        base = {"id": "q01", "level": "L1", "question": "Q?", "gold_answer": "A",
                "gold_chunk_ids": ["doc1__001"], "eval_fingerprint": "fp",
                "answer": "ok", "context_chunks": ["doc1__001"],
                "context_texts": ["chunk text one"],
                "rag_status": "ok", "context_recorded": True}
        base.update(kw)
        return base

    def test_ok(self):
        _validate_one(self.make_q(), self.make_rec(), "fp", SAMPLE_CHUNKS)

    def test_id_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(id="q01"), self.make_rec(id="q02"),
                          "fp", SAMPLE_CHUNKS)

    def test_level_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(level="L1"), self.make_rec(level="L2"),
                          "fp", SAMPLE_CHUNKS)

    def test_question_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(question="Q?"), self.make_rec(question="X?"),
                          "fp", SAMPLE_CHUNKS)

    def test_gold_answer_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(gold_answer="A"), self.make_rec(gold_answer="B"),
                          "fp", SAMPLE_CHUNKS)

    def test_gold_chunk_ids_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(gold_chunk_ids=["a"]),
                          self.make_rec(gold_chunk_ids=["b"]),
                          "fp", SAMPLE_CHUNKS)

    def test_eval_fingerprint_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(), self.make_rec(eval_fingerprint="fp2"),
                          "fp", SAMPLE_CHUNKS)

    def test_context_chunks_not_list(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(),
                          self.make_rec(context_chunks="not_list"),
                          "fp", SAMPLE_CHUNKS)

    def test_context_texts_not_list(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(),
                          self.make_rec(context_texts="not_list"),
                          "fp", SAMPLE_CHUNKS)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(),
                          self.make_rec(context_chunks=["a"],
                                        context_texts=["t1", "t2"]),
                          "fp", SAMPLE_CHUNKS)

    def test_ok_empty_answer(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(), self.make_rec(answer=""),
                          "fp", SAMPLE_CHUNKS)

    def test_ok_no_context_recorded(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(), self.make_rec(context_recorded=False),
                          "fp", SAMPLE_CHUNKS)

    def test_ok_chunk_id_not_in_corpus(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(),
                          self.make_rec(context_chunks=["unknown__999"],
                                        context_texts=["something"]),
                          "fp", SAMPLE_CHUNKS)

    def test_ok_chunk_text_mismatch(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(),
                          self.make_rec(context_chunks=["doc1__001"],
                                        context_texts=["wrong text"]),
                          "fp", SAMPLE_CHUNKS)

    def test_error_status_without_rag_error(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(),
                          self.make_rec(rag_status="error", rag_error="",
                                        context_recorded=False,
                                        context_chunks=[], context_texts=[]),
                          "fp", SAMPLE_CHUNKS)

    def test_status_not_ok_or_error(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(), self.make_rec(rag_status="unknown"),
                          "fp", SAMPLE_CHUNKS)

    def test_answer_not_string(self):
        with self.assertRaises(ValueError):
            _validate_one(self.make_q(), self.make_rec(answer=42),
                          "fp", SAMPLE_CHUNKS)


class TestScoreInt(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_score_int({"x": 3}, "x", 0, 5), 3)

    def test_nullable_none(self):
        self.assertEqual(_score_int({"x": None}, "x", 0, 5, nullable=True), "")

    def test_rejects_float(self):
        with self.assertRaises(ValueError):
            _score_int({"x": 3.8}, "x", 0, 5)

    def test_rejects_bool(self):
        with self.assertRaises(ValueError):
            _score_int({"x": True}, "x", 0, 5)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            _score_int({"x": 6}, "x", 0, 5)


class TestClaimInt(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_claim_int({"x": 4}, "x"), 4)

    def test_rejects_float(self):
        with self.assertRaises(ValueError):
            _claim_int({"x": 1.7}, "x")

    def test_rejects_bool(self):
        with self.assertRaises(ValueError):
            _claim_int({"x": False}, "x")

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            _claim_int({"x": -1}, "x")


class TestMean(unittest.TestCase):
    def test_ignores_empty_strings(self):
        rows = [{"a": ""}, {"a": "3"}, {"a": ""}, {"a": "5"}]
        self.assertAlmostEqual(_mean(rows, "a"), 4.0)

    def test_returns_none_all_empty(self):
        self.assertIsNone(_mean([{"a": ""}, {"a": ""}], "a"))

    def test_handles_zero(self):
        self.assertEqual(_mean([{"a": "0"}, {"a": "0"}], "a"), 0.0)

    def test_excludes_none(self):
        rows = [{"a": None}, {"a": "3"}, {"a": ""}]
        self.assertAlmostEqual(_mean(rows, "a"), 3.0)

    def test_excludes_none_when_all_none(self):
        self.assertIsNone(_mean([{"a": None}, {"a": None}], "a"))


class TestExtractJson(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_extract_json('{"x": 1}'), {"x": 1})

    def test_code_fence(self):
        self.assertEqual(_extract_json('```json\n{"x": 1}\n```'), {"x": 1})

    def test_not_dict(self):
        with self.assertRaises(ValueError):
            _extract_json('[1,2]')

    def test_not_string(self):
        with self.assertRaises(ValueError):
            _extract_json(42)


class TestJudgeOneSelfConsistency(unittest.TestCase):
    def test_faithfulness5_unsupported_positive_rejected(self):
        gen = MagicMock()
        gen.config = MagicMock()
        gen.config.LLM_MODEL = "stub"
        gen.complete.return_value = json.dumps({
            "faithfulness_1_5": 5, "unsupported_claims": 2, "total_claims": 5,
            "correctness_0_5": 4, "L4_rubric_0_5": None, "failure_stage": "none",
            "evidence": "", "improvement": "",
        })
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        r = _mod.judge_one(gen, q, rec, "fp", "cachesha")
        self.assertEqual(r["judge_status"], "error")
        self.assertGreaterEqual(gen.complete.call_count, 2)

    def test_valid_pass(self):
        gen = MagicMock()
        gen.config = MagicMock()
        gen.config.LLM_MODEL = "stub"
        gen.complete.return_value = json.dumps({
            "faithfulness_1_5": 4, "unsupported_claims": 1, "total_claims": 5,
            "correctness_0_5": 3, "L4_rubric_0_5": None, "failure_stage": "none",
            "evidence": "context支持大部分断言，1项无依据", "improvement": "补上下文依据",
        })
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        r = _mod.judge_one(gen, q, rec, "fp", "cachesha")
        self.assertEqual(r["judge_status"], "ok")
        self.assertEqual(r["faithfulness_1_5"], 4)

    def test_retrieval_failure_stage_rejected(self):
        gen = MagicMock()
        gen.config = MagicMock()
        gen.config.LLM_MODEL = "stub"
        gen.complete.return_value = json.dumps({
            "faithfulness_1_5": 3, "unsupported_claims": 1, "total_claims": 3,
            "correctness_0_5": 2, "L4_rubric_0_5": None,
            "failure_stage": "retrieval_failure",
            "evidence": "", "improvement": "",
        })
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        r = _mod.judge_one(gen, q, rec, "fp", "cachesha")
        self.assertEqual(r["judge_status"], "error")

    def test_row_has_provenance_fields(self):
        gen = MagicMock()
        gen.config = MagicMock()
        gen.config.LLM_MODEL = "stub"
        gen.complete.return_value = json.dumps({
            "faithfulness_1_5": 4, "unsupported_claims": 0, "total_claims": 3,
            "correctness_0_5": 3, "L4_rubric_0_5": None, "failure_stage": "none",
            "evidence": "", "improvement": "",
        })
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        r = _mod.judge_one(gen, q, rec, "fp", "cachesha")
        for key in ("answer_hash", "context_hash", "input_sha256",
                     "eval_fingerprint", "prompt_version", "prompt_hash",
                     "source_cache_sha"):
            self.assertIn(key, r)
            self.assertIsNotNone(r[key])

    def test_total_claims_0_rejected(self):
        gen = MagicMock()
        gen.config = MagicMock()
        gen.config.LLM_MODEL = "stub"
        gen.complete.return_value = json.dumps({
            "faithfulness_1_5": 1, "unsupported_claims": 0, "total_claims": 0,
            "correctness_0_5": 0, "L4_rubric_0_5": None, "failure_stage": "none",
            "evidence": "无事实断言", "improvement": "需说明无法评分",
        })
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        r = _mod.judge_one(gen, q, rec, "fp", "cachesha")
        self.assertEqual(r["judge_status"], "error")
        self.assertEqual(r["reason_code"], "claims_zero_invalid")

    def test_attempt_history_records_each_retry(self):
        from unittest.mock import call
        gen = MagicMock()
        gen.config = MagicMock()
        gen.config.LLM_MODEL = "stub"
        responses = [
            json.dumps({
                "faithfulness_1_5": 5, "unsupported_claims": 1,
                "total_claims": 4, "correctness_0_5": 4,
                "L4_rubric_0_5": None, "failure_stage": "none",
                "evidence": "矛盾上下文", "improvement": "去掉无依据断言",
            }),
            json.dumps({
                "faithfulness_1_5": 4, "unsupported_claims": 0,
                "total_claims": 4, "correctness_0_5": 4,
                "L4_rubric_0_5": None, "failure_stage": "none",
                "evidence": "上下文支持", "improvement": "保持现状",
            }),
        ]
        gen.complete.side_effect = responses
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        r = _mod.judge_one(gen, q, rec, "fp", "cachesha")
        self.assertEqual(r["judge_status"], "ok")
        self.assertEqual(r["attempts"], 2)
        self.assertEqual(r["attempt_history"][0]["reason_code"],
                         "faithfulness5_with_unsupported")
        self.assertEqual(len(r["attempt_history"]), 1)

    def test_missing_evidence_rejected_with_reason(self):
        gen = MagicMock()
        gen.config = MagicMock()
        gen.config.LLM_MODEL = "stub"
        gen.complete.return_value = json.dumps({
            "faithfulness_1_5": 4, "unsupported_claims": 0, "total_claims": 4,
            "correctness_0_5": 4, "L4_rubric_0_5": None,
            "failure_stage": "none", "evidence": "...", "improvement": "改进",
        })
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        r = _mod.judge_one(gen, q, rec, "fp", "cachesha")
        self.assertEqual(r["judge_status"], "error")
        self.assertEqual(r["reason_code"], "evidence_or_improvement_missing")


class TestMicroMacro(unittest.TestCase):
    def compute_micro(self, rows):
        u = sum(int(r["unsupported_claims"]) for r in rows
                if r["unsupported_claims"] != "")
        c = sum(int(r["total_claims"]) for r in rows
                if r["total_claims"] != "")
        return round(u / c, 4) if c else None

    def test_identical(self):
        rows = [
            {"unsupported_claims": 0, "total_claims": 5, "hallucination_rate": 0.0},
            {"unsupported_claims": 0, "total_claims": 5, "hallucination_rate": 0.0},
        ]
        self.assertEqual(_mean(rows, "hallucination_rate"), 0.0)
        self.assertEqual(self.compute_micro(rows), 0.0)

    def test_skewed(self):
        rows = [
            {"unsupported_claims": 1, "total_claims": 2, "hallucination_rate": 0.5},
            {"unsupported_claims": 1, "total_claims": 8, "hallucination_rate": 0.125},
        ]
        self.assertEqual(_mean(rows, "hallucination_rate"), 0.3125)
        self.assertEqual(self.compute_micro(rows), 0.2)

    def test_macro_excludes_empty_rate(self):
        rows = [
            {"unsupported_claims": 0, "total_claims": 0, "hallucination_rate": ""},
            {"unsupported_claims": 1, "total_claims": 4, "hallucination_rate": 0.25},
        ]
        self.assertEqual(_mean(rows, "hallucination_rate"), 0.25)

    def test_micro_null_when_zero_denom(self):
        rows = [
            {"unsupported_claims": 0, "total_claims": 0, "hallucination_rate": ""},
        ]
        self.assertIsNone(self.compute_micro(rows))

    def test_macro_excludes_none(self):
        rows = [
            {"unsupported_claims": 0, "total_claims": 0, "hallucination_rate": None},
            {"unsupported_claims": 0, "total_claims": 5, "hallucination_rate": 0.0},
        ]
        self.assertEqual(_mean(rows, "hallucination_rate"), 0.0)



# =====================================================================
# Main-level integration tests
# =====================================================================

def _write_jsonl(path, items):
    path.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n",
        encoding="utf-8")


def _build_fake_qas(n=3):
    return [
        {"id": "q%02d" % i, "level": "L1", "question": "Q%d" % i,
         "gold_answer": "A%d" % i, "gold_chunk_ids": ["doc1__001"]}
        for i in range(1, n + 1)
    ]

def _build_fake_rec(q):
    return {
        "id": q["id"], "level": q["level"], "question": q["question"],
        "gold_answer": q["gold_answer"], "gold_chunk_ids": q["gold_chunk_ids"],
        "eval_fingerprint": "fp", "answer": "answer text",
        "context_chunks": ["doc1__001"], "context_texts": ["chunk text one"],
        "rag_status": "ok", "context_recorded": True,
    }


class TestQaCacheIntegrityGate(unittest.TestCase):
    """Pure function _check_qa_cache_integrity must reject every alignment
    violation.  This is the exact gate used by main() before any API."""

    def _records(self, qas):
        d = {}
        for q in qas:
            d[q["id"]] = _build_fake_rec(q)
        return d

    def test_clean(self):
        qas = _build_fake_qas(2)
        recs = self._records(qas)
        errs = _mod._check_qa_cache_integrity(qas, recs, [], [])
        self.assertEqual(errs, [])

    def test_qa_duplicate(self):
        qas = _build_fake_qas(2)
        qas.append(dict(qas[0]))  # duplicate q01
        recs = self._records(qas)  # dict will dedupe
        errs = _mod._check_qa_cache_integrity(qas, recs, [], [])
        self.assertTrue(any("Duplicate IDs in QA set" in e for e in errs))

    def test_qa_missing_id(self):
        qas = _build_fake_qas(2)
        qas.append({"level": "L1", "question": "?", "gold_answer": "?",
                    "gold_chunk_ids": []})
        recs = self._records([q for q in qas if q.get("id")])
        errs = _mod._check_qa_cache_integrity(qas, recs, [], [])
        self.assertTrue(any("missing id" in e for e in errs))

    def test_cache_duplicate(self):
        qas = _build_fake_qas(2)
        recs = self._records(qas)
        errs = _mod._check_qa_cache_integrity(qas, recs, [], ["q01"])
        self.assertTrue(any("Duplicate IDs in cache" in e for e in errs))

    def test_cache_missing_id(self):
        qas = _build_fake_qas(2)
        recs = self._records(qas)
        errs = _mod._check_qa_cache_integrity(qas, recs, ["<missing>"], [])
        self.assertTrue(any("missing id" in e for e in errs))

    def test_missing_qa(self):
        qas = _build_fake_qas(3)
        recs = self._records(qas)
        del recs["q03"]
        errs = _mod._check_qa_cache_integrity(qas, recs, [], [])
        self.assertTrue(any("missing from cache" in e for e in errs))

    def test_extra_cache(self):
        qas = _build_fake_qas(2)
        recs = self._records(qas)
        extra = _build_fake_rec({"id": "q99", "level": "L1", "question": "X",
                                  "gold_answer": "Y", "gold_chunk_ids": ["doc1__001"]})
        recs["q99"] = extra
        errs = _mod._check_qa_cache_integrity(qas, recs, [], [])
        self.assertTrue(any("not in QA set" in e for e in errs))


class TestReadCheckpointLastWins(unittest.TestCase):
    def test_duplicate_id_last_row_wins(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "judge_checkpoint.jsonl"
            _write_jsonl(p, [
                {"id": "q01", "correctness_0_5": 3, "total_claims": 4,
                 "sequence": "pilot"},
                {"id": "q01", "correctness_0_5": 2, "total_claims": 7,
                 "sequence": "batch"},
            ])
            rows = _mod._read_checkpoint(p)
            self.assertEqual(rows["q01"]["sequence"], "batch")
            self.assertEqual(rows["q01"]["correctness_0_5"], 2)
            self.assertEqual(rows["q01"]["total_claims"], 7)

    def test_export_uses_checkpoint_last_row(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "judge_checkpoint.jsonl"
            _write_jsonl(p, [
                {"id": "q01", "correctness_0_5": 3, "total_claims": 4,
                 "sequence": "pilot"},
                {"id": "q01", "correctness_0_5": 2, "total_claims": 7,
                 "sequence": "batch"},
            ])
            rows_by_id = _mod._read_checkpoint(p)
            # mirror main()'s ordered export
            qas = [{"id": "q01", "level": "L1"}]
            rows = [rows_by_id[q["id"]] for q in qas]
            self.assertEqual(rows[0]["sequence"], "batch")
            self.assertEqual(rows[0]["correctness_0_5"], 2)
            self.assertEqual(rows[0]["total_claims"], 7)


class TestValidateAll(unittest.TestCase):
    def test_all_ok(self):
        qas = _build_fake_qas(2)
        recs = {q["id"]: _build_fake_rec(q) for q in qas}
        errs, ok = _mod._validate_all(qas, recs, "fp", SAMPLE_CHUNKS)
        self.assertTrue(ok)
        self.assertEqual(errs, [])

    def test_error_record_rejected(self):
        qas = _build_fake_qas(2)
        recs = {q["id"]: _build_fake_rec(q) for q in qas}
        recs["q02"]["rag_status"] = "error"
        recs["q02"]["rag_error"] = "SomeError"
        errs, ok = _mod._validate_all(qas, recs, "fp", SAMPLE_CHUNKS)
        self.assertFalse(ok)
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0][0], "q02")

    def test_mixed_fingerprint_rejected(self):
        qas = _build_fake_qas(2)
        recs = {q["id"]: _build_fake_rec(q) for q in qas}
        recs["q01"]["eval_fingerprint"] = "other-fp"
        errs, ok = _mod._validate_all(qas, recs, "fp", SAMPLE_CHUNKS)
        self.assertFalse(ok)
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0][0], "q01")

    def test_field_mismatch_rejected(self):
        qas = _build_fake_qas(2)
        recs = {q["id"]: _build_fake_rec(q) for q in qas}
        recs["q02"]["gold_chunk_ids"] = ["other"]
        errs, ok = _mod._validate_all(qas, recs, "fp", SAMPLE_CHUNKS)
        self.assertFalse(ok)
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0][0], "q02")


class TestPromptHashChanges(unittest.TestCase):
    def test_prompt_hash_changes_with_prompt_text(self):
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        h1 = _mod._prompt_hash(q, rec)
        # Change prompt: modify _mod._judge_prompt output via question change
        q2 = dict(q, question="Q2")
        h2 = _mod._prompt_hash(q2, rec)
        self.assertNotEqual(h1, h2)

    def test_input_hash_includes_level_and_gold_ids(self):
        rec = {"id": "Q", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"],
               "gold_chunk_ids": ["doc1__001"], "level": "L1"}
        h1 = _mod._input_hash("Q", "G", rec, "fp", level="L1")
        rec2 = dict(rec, level="L2")
        h2 = _mod._input_hash("Q", "G", rec2, "fp", level="L2")
        self.assertNotEqual(h1, h2)
        rec3 = dict(rec, gold_chunk_ids=["other"])
        h3 = _mod._input_hash("Q", "G", rec3, "fp", level="L1")
        self.assertNotEqual(h1, h3)


class TestCheckpointResumeGate(unittest.TestCase):
    """A checkpoint row is reusable only if every binding still matches."""

    def test_resume_requires_input_hash(self):
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "level": "L1", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        row = {
            "id": "Q", "judge_status": "ok",
            "prompt_hash": _mod._prompt_hash(q, rec),
            "source_cache_sha": "cache",
            "eval_fingerprint": "fp",
            "input_sha256": _mod._input_hash("Q", "G", rec, "fp"),
        }
        self.assertTrue(_mod._checkpoint_matches(
            row, q, rec, "fp", "cache"))
        row["input_sha256"] = "stale"
        self.assertFalse(_mod._checkpoint_matches(
            row, q, rec, "fp", "cache"))

    def test_error_row_is_not_reused(self):
        q = {"id": "Q", "level": "L1", "question": "Q", "gold_answer": "G",
             "gold_chunk_ids": ["doc1__001"]}
        rec = {"id": "Q", "level": "L1", "answer": "A", "context_texts": ["t"],
               "context_chunks": ["doc1__001"]}
        row = {
            "id": "Q", "judge_status": "error",
            "prompt_hash": _mod._prompt_hash(q, rec),
            "source_cache_sha": "cache",
            "eval_fingerprint": "fp",
            "input_sha256": _mod._input_hash("Q", "G", rec, "fp"),
        }
        self.assertFalse(_mod._checkpoint_matches(
            row, q, rec, "fp", "cache"))

    def test_summary_marks_partial_runs(self):
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            cache_path = tdir / "t1_baseline.jsonl"
            eval_path = tdir / "qa.jsonl"
            cache_path.write_text("", encoding="utf-8")
            eval_path.write_text("", encoding="utf-8")
            checkpoint = tdir / "judge_checkpoint.jsonl"
            _write_jsonl(checkpoint, [{
                "id": "q01", "level": "L1", "judge_status": "ok",
                "judge_model": "stub", "faithfulness_1_5": 4,
                "unsupported_claims": 0, "total_claims": 3,
                "hallucination_rate": 0.0, "correctness_0_5": 3,
                "L4_rubric_0_5": "", "failure_stage": "none",
            }])
            rows = _mod._read_checkpoint(checkpoint)
            _, _, summary = _mod._write_summary(
                rows=list(rows.values()),
                qas=[{"id": "q01", "level": "L1"},
                     {"id": "q02", "level": "L1"}],
                cache_path=cache_path, cache_sha="cache",
                eval_path=eval_path, current_fp="fp", out_dir=tdir,
                checkpoint_path=checkpoint, workers=1,
                judge_model="stub")
            self.assertFalse(summary["completed_all_requested"])
            self.assertEqual(summary["n"], 1)
            self.assertEqual(summary["requested"], 2)
            self.assertIn("prompt", (tdir / "t7_llm_judge.csv").read_text(
                encoding="utf-8-sig").splitlines()[0])


class TestMainZeroApiOnRejection(unittest.TestCase):
    """Patch sys.modules so a real main() call can run with stubs, and verify
    zero API calls / no CSV on invalid input."""

    def _patch_src_modules(self, tdir, qa_path, cache_recs):
        import types
        import argparse as argparse_mod

        # Build fake modules
        cfg = MagicMock()
        cfg.LLM_MODEL = "stub"
        cfg.CORPUS_DIR = tdir
        cfg.RESULTS_DIR = str(tdir)
        cfg.require_llm = MagicMock()

        ldr = MagicMock()
        ldr.read_all_docs.return_value = {"doc1": "chunk text one"}

        chk = MagicMock()
        chk.chunk_documents.return_value = [
            {"chunk_id": "doc1__001", "text": "chunk text one"}
        ]

        gen = MagicMock()
        gen.config = cfg

        # experiments.common
        common = MagicMock()
        common.BASELINE_PIPELINE = {"chunk_method": "fixed", "chunk_size": 512,
                                    "chunk_overlap": 128, "retriever_mode": "dense"}
        common.experiment_signature.return_value = "fp"

        # Inject into sys.modules so main()'s `from src import ...` works
        src_mod = types.ModuleType("src")
        src_mod.config = cfg
        src_mod.loader = ldr
        src_mod.chunker = chk
        src_mod.generator = gen

        exp_mod = types.ModuleType("experiments")
        common_mod = types.ModuleType("experiments.common")
        common_mod.BASELINE_PIPELINE = common.BASELINE_PIPELINE
        common_mod.experiment_signature = common.experiment_signature
        exp_mod.common = common_mod

        patchers = [
            patch.dict("sys.modules", {"src": src_mod,
                                       "src.config": cfg,
                                       "src.loader": ldr,
                                       "src.chunker": chk,
                                       "src.generator": gen,
                                       "experiments": exp_mod,
                                       "experiments.common": common_mod}),
        ]
        return patchers, gen

    def test_invalid_cache_zero_api_no_csv(self):
        qas = _build_fake_qas(2)
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            qa_path = tdir / "qa.jsonl"
            _write_jsonl(qa_path, qas)
            cache_path = tdir / "t1_baseline.jsonl"
            recs = [_build_fake_rec(q) for q in qas]
            recs[1]["gold_chunk_ids"] = ["corrupt"]  # field mismatch
            _write_jsonl(cache_path, recs)

            patchers, gen = self._patch_src_modules(tdir, qa_path, recs)
            ns = MagicMock()
            ns.root = ROOT
            ns.eval_path = qa_path
            ns.results = tdir
            ns.workers = 1

            with patch("argparse.ArgumentParser.parse_args", return_value=ns):
                for p in patchers:
                    p.start()
                try:
                    with self.assertRaises(SystemExit):
                        _mod.main()
                finally:
                    for p in reversed(patchers):
                        p.stop()

            gen.complete.assert_not_called()
            self.assertFalse((tdir / "t7_llm_judge.csv").exists())
            self.assertFalse((tdir / "t7_llm_judge_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
