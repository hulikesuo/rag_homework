"""Pure-function checks; no models, APIs, corpus files or extra dependencies.

Run with an explicit project root and isolated Python, for example:
    python -I scripts/check_compression.py --root C:/path/to/rag_homework
"""

import argparse
import math
from pathlib import Path
import sys
import unittest


class InformationCompressorChecks(unittest.TestCase):
    def test_smoothed_probabilities_and_unknown_bucket(self):
        compressor = InformationCompressor().fit(["Apple apple banana"])
        # N=3, V=2; observed types plus ONE unknown bucket sum to 1.
        self.assertEqual(compressor.total_tokens, 3)
        self.assertEqual(compressor.token_counts, {"apple": 2, "banana": 1})
        self.assertAlmostEqual(compressor.token_probability("apple"), 3 / 6)
        self.assertAlmostEqual(compressor.token_probability("banana"), 2 / 6)
        self.assertAlmostEqual(compressor.token_probability("unknown"), 1 / 6)
        self.assertAlmostEqual(sum(compressor.token_probability(token)
                                   for token in ("apple", "banana", "unknown")), 1)
        self.assertAlmostEqual(compressor.self_information("apple"), 1)
        self.assertAlmostEqual(compressor.self_information("unknown"), math.log2(6))
        self.assertEqual(len(compressor.token_counts), 2)

    def test_fit_replaces_statistics_and_uses_shared_tokenization(self):
        compressor = InformationCompressor().fit(["discard discard"])
        docs = {"one": "苹果价格 APPLE", "two": "价格 3.14"}
        self.assertIs(compressor.fit(iter(docs.values())), compressor)
        self.assertEqual(compressor.token_counts,
                         {"苹果": 1, "果价": 1, "价格": 2, "apple": 1, "3": 1, "14": 1})
        self.assertEqual(compressor.total_tokens, 7)
        self.assertNotIn("discard", compressor.token_counts)

    def test_fixed_score_formula_and_units(self):
        compressor = InformationCompressor().fit(["apple apple banana"])
        # Query coverage=1/2, H=(1+log2(6))/2 bits/token, B=log2(6).
        expected = 2 + 0.5 + 0.5 * ((1 + math.log2(6)) / 2) / math.log2(6)
        self.assertAlmostEqual(compressor.sentence_score("apple pear", "apple unseen."),
                               expected)
        self.assertEqual(compressor.sentence_score("", "!!!"), 0)

    def test_relevance_beats_rare_irrelevant_words(self):
        compressor = InformationCompressor().fit(["apple price " * 100])
        irrelevant = "xylophone zephyr quokka."
        relevant = "apple price."
        self.assertGreater(compressor.self_information("quokka"),
                           compressor.self_information("apple"))
        self.assertGreater(compressor.sentence_score("apple price", relevant),
                           compressor.sentence_score("apple price", irrelevant))
        chunks = [{"text": irrelevant}, {"text": relevant}]
        self.assertEqual(compressor.compress("apple price", chunks, len(relevant))[0], relevant)

    def test_information_matters_with_equal_relevance(self):
        compressor = InformationCompressor().fit(["apple common " * 100])
        common, rare = "apple common.", "apple rarexx."
        self.assertEqual(len(common), len(rare))
        self.assertGreater(compressor.sentence_score("apple", rare),
                           compressor.sentence_score("apple", common))
        self.assertEqual(compressor.compress("apple", [{"text": common + " " + rare}],
                                             len(rare))[0], rare)
        self.assertEqual(compressor.compress("", [{"text": common + " " + rare}],
                                             len(rare))[0], rare)

    def test_deduplicate_before_budget_and_keep_original_count(self):
        chunks = [{"text": " A.\nA. "}, {"text": "A.\r\nB."}]
        original = len("\n\n".join(chunk["text"] for chunk in chunks))
        result = InformationCompressor().compress("", iter(chunks), 5)
        self.assertEqual(result, ("A.\nB.", original, 5, 5 / original))

    def test_newline_separator_exact_boundary(self):
        compressor = InformationCompressor()
        chunks = [{"text": "A. B."}]
        self.assertEqual(compressor.compress("", chunks, 5)[0], "A.\nB.")
        # Two sentences cost 5 characters, not 4: the separator must fit too.
        self.assertEqual(compressor.compress("", chunks, 4)[0], "A.")
        self.assertEqual(compressor.compress("", chunks, 2)[0], "A.")
        self.assertEqual(compressor.compress("", chunks, 1)[0], "")

    def test_skip_oversized_sentence_without_truncation(self):
        chunks = [{"text": "apple explanation much too long.\nOK."}]
        self.assertEqual(InformationCompressor().compress("apple", chunks, 3)[0], "OK.")
        self.assertEqual(InformationCompressor().compress("apple", chunks, 2)[0], "")

    def test_chinese_sentences_and_decimal_are_whole(self):
        decimal, other = "苹果价格3.14元。", "库存充足！"
        compressor = InformationCompressor().fit([decimal + other])
        chunks = [{"text": decimal + other}]
        self.assertEqual(compressor.compress("苹果价格", chunks, len(decimal))[0], decimal)
        # If the decimal-containing sentence does not fit, none of it leaks out.
        self.assertEqual(compressor.compress("苹果价格", chunks, len(decimal) - 1)[0], other)
        english_decimal = "Price is 3.14 yuan."
        self.assertEqual(compressor.compress("price", [{"text": english_decimal}],
                                             len(english_decimal))[0], english_decimal)
        self.assertEqual(compressor.compress("price", [{"text": english_decimal}],
                                             len(english_decimal) - 1)[0], "")

    def test_empty_inputs_queries_and_nonpositive_budgets(self):
        compressor = InformationCompressor().fit([])
        self.assertEqual(compressor.token_probability("anything"), 1)
        self.assertEqual(compressor.self_information("anything"), 0)
        self.assertEqual(compressor.compress("", [], 100), ("", 0, 0, 0.0))
        self.assertEqual(compressor.compress("", [{"text": ""}], 100), ("", 0, 0, 0.0))
        self.assertEqual(compressor.compress("?", [{"text": " \r\n "}], 100),
                         ("", 4, 0, 0.0))
        for budget in (0, -1, -100):
            self.assertEqual(compressor.compress("a", [{"text": "A."}], budget),
                             ("", 2, 0, 0.0))
        self.assertEqual(compressor.compress("", [{"text": "A."}], 2), ("A.", 2, 2, 1.0))

    def test_ties_are_stable_and_metadata_does_not_affect_output(self):
        compressor = InformationCompressor()
        chunks = [{"text": "B.", "gold_answer": "A.", "score": -999},
                  {"text": "A.", "gold_answer": "B.", "score": 999}]
        expected = ("B.\nA.", 6, 5, 5 / 6)
        for _ in range(3):
            self.assertEqual(compressor.compress("", chunks, 5), expected)
        self.assertEqual(compressor.compress("", [{"text": "A."}, {"text": "B."}], 5)[0],
                         "A.\nB.")

    def test_english_quote_after_period_splits_without_dropping(self):
        # "Apple price." ends with a period followed by a closing quote; without
        # treating that as a boundary the whole sentence would exceed the budget
        # and be dropped, returning empty text.
        s = "\"Apple price.\" Stock available."
        compressor = InformationCompressor().fit(["apple price stock"])
        self.assertEqual(compressor.compress("apple", [{"text": s}], 14)[0],
                         "\"Apple price.\"")
        # Chinese closing quote after the period must not split a complete quote.
        cn = "他说“价格已公布。”库存按日更新。"
        self.assertEqual(compressor.compress("价格", [{"text": cn}], len("他说“价格已公布。”"))[0],
                         "他说“价格已公布。”")

    def test_budget_and_count_invariants_across_boundaries(self):
        sentences = ["苹果价格3.14元。", "Rare quokka!", "库存充足；", "OK."]
        chunks = [{"text": "\n".join(sentences + sentences[:2])}]
        original = len(chunks[0]["text"])
        compressor = InformationCompressor().fit(["苹果价格 库存充足 OK"])
        for budget in range(original + 2):
            with self.subTest(budget=budget):
                text, before, after, ratio = compressor.compress("苹果价格", chunks, budget)
                self.assertEqual(before, original)
                self.assertEqual(after, len(text))
                self.assertLessEqual(after, budget)
                self.assertAlmostEqual(ratio, after / original)
                selected = text.splitlines() if text else []
                self.assertEqual(len(selected), len(set(selected)))
                self.assertTrue(all(sentence in sentences for sentence in selected))


class ExtractivePathChecks(unittest.TestCase):
    """Regression checks for the compressor actually used by T5 and demo."""

    def test_english_quote_after_period_is_kept_whole(self):
        text = '"Apple price." Stock available.'
        new_text, original, after, ratio = compress_extractive(
            "apple", [{"text": text}], len('"Apple price."'))
        self.assertEqual(new_text, '"Apple price."')
        self.assertEqual(original, len(text))
        self.assertEqual(after, len(new_text))
        self.assertAlmostEqual(ratio, after / original)

    def test_chinese_closing_quote_is_kept_whole(self):
        text = "他说“价格已公布。”库存按日更新。"
        quote = "他说“价格已公布。”"
        self.assertEqual(
            compress_extractive("价格", [{"text": text}], len(quote))[0],
            quote,
        )

    def test_decimal_and_nonpositive_budget(self):
        text = "价格3.14元。库存充足。"
        self.assertEqual(
            compress_extractive("价格", [{"text": text}], len("价格3.14元。"))[0],
            "价格3.14元。",
        )
        self.assertEqual(
            compress_extractive("价格", [{"text": text}], 0),
            ("", len(text), 0, 0.0),
        )

    def test_output_never_exceeds_budget_or_truncates_sentence(self):
        text = "价格3.14元。\n库存充足！\nRare quokka."
        original = len(text)
        for budget in range(original + 2):
            with self.subTest(budget=budget):
                new_text, before, after, ratio = compress_extractive(
                    "价格", [{"text": text}], budget)
                self.assertEqual(before, original)
                self.assertEqual(after, len(new_text))
                self.assertLessEqual(after, budget)
                self.assertAlmostEqual(ratio, after / original if original else 0.0)
                if new_text:
                    for sentence in new_text.splitlines():
                        self.assertIn(sentence, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Explicit rag_homework root")
    args, unittest_args = parser.parse_known_args()
    root = args.root.resolve(strict=True)
    if not (root / "src" / "compression.py").is_file():
        parser.error("--root must contain src/compression.py")
    sys.path.insert(0, str(root))
    from src.compression import InformationCompressor
    from src.rerank import compress_extractive

    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
