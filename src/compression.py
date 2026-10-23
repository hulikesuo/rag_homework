"""Corpus unigram self-information for query-aware extractive compression.

This is a statistical unigram estimator, NOT neural language-model surprisal.
It ignores token order, syntax and context; rare tokens can be names, noise or
misspellings rather than useful evidence. Query overlap is lexical, not semantic.
No gold answers, models, APIs or evaluation labels are used.
"""

import math
from collections import Counter

from .retriever import default_tokenize
from .textutil import split_sentences


class InformationCompressor:
    """Fit once on corpus text, then reuse for multiple queries.

    Example::

        compressor = InformationCompressor().fit(docs.values())
        text, original, after, ratio = compressor.compress(query, chunks, 500)

    For corpus token count N and observed vocabulary size V, add-one smoothing is
    p(t) = (count(t) + 1) / (N + V + 1). The extra 1 reserves one unknown-token
    bucket: all unseen spellings share its probability, rather than forming a
    normalized distribution over infinitely many possible unseen spellings.
    Token self-information is -log2(p(t)), in bits.

    Fixed sentence score (dimensionless; no weights tuned on gold answers):
      R = |unique(query tokens) intersect unique(sentence tokens)|
          / max(1, |unique(query tokens)|)
      H = mean token self-information in the sentence, in bits/token
      B = max(1, log2(N + V + 1)), in bits/token
      score = 2 * int(any query overlap) + R + 0.5 * H / B
    H is zero for a tokenless sentence. Since 0 <= H/B <= 1, a sentence with
    query-token overlap always outranks one without it. The information term
    rewards corpus rarity within that constraint; it does not measure factual
    correctness or neural LM surprisal. Repeated tokens contribute to H, while
    query coverage R uses unique tokens. Empty queries fall back to information.

    Ties retain first occurrence in input order. Whitespace-trimmed duplicate
    sentences are removed before allocating the budget. Whole sentences are
    emitted in score order, separated by one newline; oversized sentences are
    skipped, never truncated. The budget uses Python len() character counts,
    including separators, not bytes, tokenizer tokens or grapheme clusters.
    An unfitted instance behaves like one fitted on an empty corpus (p=1, H=0).
    """

    def __init__(self):
        self.token_counts = Counter()
        self.total_tokens = 0

    def fit(self, texts):
        """Replace corpus statistics from an iterable of strings; return self.

        Pass docs.values(), not document metadata or gold annotations. Both fit
        and scoring use retriever.default_tokenize (Chinese bigrams and lowercase
        English/number tokens). This method consumes generators once.
        """
        counts = Counter()
        for text in texts:
            counts.update(default_tokenize(text))
        self.token_counts = counts
        self.total_tokens = sum(counts.values())
        return self

    def token_probability(self, token):
        """Return add-one probability for an already-tokenized token (or UNK)."""
        denominator = self.total_tokens + len(self.token_counts) + 1
        return (self.token_counts[token] + 1) / denominator

    def self_information(self, token):
        """Return unigram self-information in bits for a token, including UNK."""
        return -math.log2(self.token_probability(token))

    def _sentence_score(self, query_tokens, sentence):
        tokens = default_tokenize(sentence)
        overlap = len(query_tokens.intersection(tokens))
        relevance = overlap / max(1, len(query_tokens))
        mean_bits = (sum(self.self_information(token) for token in tokens)
                     / len(tokens)) if tokens else 0.0
        max_bits = max(1.0, math.log2(
            self.total_tokens + len(self.token_counts) + 1))
        return 2 * bool(overlap) + relevance + 0.5 * mean_bits / max_bits

    def sentence_score(self, query, sentence):
        """Expose the fixed relevance/information formula for inspection."""
        return self._sentence_score(set(default_tokenize(query)), sentence)

    def compress(self, query, chunks, budget):
        """Return (text, original_chars, after_chars, retention_ratio).

        chunks is an iterable of mappings with a 'text' string. Original length
        matches compress_extractive: raw chunk texts joined with two newlines,
        before stripping or deduplication. Nonpositive budgets return empty text.
        The retention ratio is after/original (zero when original is empty).
        Only the query and chunk text are read; chunk metadata is ignored.
        """
        texts = [chunk["text"] for chunk in chunks]
        original = len("\n\n".join(texts))
        if budget <= 0:
            return "", original, 0, 0.0

        query_tokens = set(default_tokenize(query))
        candidates, seen = [], set()
        for text in texts:
            for sentence in split_sentences(text):
                sentence = sentence.strip()
                if not sentence or sentence in seen:
                    continue
                seen.add(sentence)
                candidates.append((sentence, self._sentence_score(query_tokens, sentence)))
        # Python's stable sort preserves first occurrence for equal scores.
        candidates.sort(key=lambda item: -item[1])

        picked, used = [], 0
        for sentence, _ in candidates:
            cost = len(sentence) + bool(picked)
            if used + cost > budget:
                continue
            picked.append(sentence)
            used += cost
        text = "\n".join(picked)
        return text, original, used, (used / original if original else 0.0)
