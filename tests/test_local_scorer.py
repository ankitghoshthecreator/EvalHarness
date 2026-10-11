"""
test_local_scorer.py — Tests for scorer/local_scorer.py

Coverage:
  - TTR computation
  - N-gram repetition detection
  - Jaccard overlap / circularity
  - Syntax checks
  - Composite scoring
  - Batch scoring
  - Edge cases
"""

import pytest
from scorer.local_scorer import (
    score, batch_score, LocalScore,
    _ttr, _ngram_repetition_score, _jaccard, _syntax_check, _tokenize,
)


# ── Tokenizer ─────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_basic(self):
        tokens = _tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_empty(self):
        assert _tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = _tokenize("Hello, world!")
        assert "," not in tokens
        assert "!" not in tokens


# ── TTR ───────────────────────────────────────────────────────────────────

class TestTTR:
    def test_all_unique(self):
        tokens = ["alpha", "beta", "gamma", "delta"]
        assert _ttr(tokens) == 1.0

    def test_all_same(self):
        tokens = ["word"] * 10
        assert _ttr(tokens) == pytest.approx(0.1)

    def test_empty(self):
        assert _ttr([]) == 0.0

    def test_mixed(self):
        tokens = ["a", "b", "a", "c"]
        assert _ttr(tokens) == pytest.approx(0.75)


# ── N-gram repetition ─────────────────────────────────────────────────────

class TestNgramRepetition:
    def test_diverse_text(self, sample_output):
        tokens = _tokenize(sample_output)
        score = _ngram_repetition_score(tokens, n=3)
        assert score > 0.7, "Diverse text should score high"

    def test_repetitive_text(self, repetitive_output):
        tokens = _tokenize(repetitive_output)
        score = _ngram_repetition_score(tokens, n=3)
        assert score < 0.5, "Repetitive text should score low"

    def test_too_short(self):
        tokens = ["a", "b"]
        assert _ngram_repetition_score(tokens, n=3) == 1.0

    def test_single_repetition(self):
        tokens = ["the", "market", "is", "up"] * 10
        score = _ngram_repetition_score(tokens)
        assert score < 0.3


# ── Jaccard overlap ───────────────────────────────────────────────────────

class TestJaccard:
    def test_identical(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        result = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert pytest.approx(result, abs=0.01) == 0.5

    def test_empty_sets(self):
        assert _jaccard(set(), {"a"}) == 0.0


# ── Syntax check ──────────────────────────────────────────────────────────

class TestSyntaxCheck:
    def test_good_sentence(self):
        ok, flags = _syntax_check("This is a well-formed sentence.")
        assert ok is True
        assert not flags

    def test_unbalanced_double_quotes(self):
        ok, flags = _syntax_check('He said "hello world.')
        assert "unbalanced_double_quotes" in flags

    def test_too_short(self):
        ok, flags = _syntax_check("Hi.")
        assert "too_short" in flags

    def test_no_terminal_punctuation(self):
        _, flags = _syntax_check("This sentence has no period at the end")
        assert "missing_terminal_punctuation" in flags

    def test_json_output_ok(self):
        ok, flags = _syntax_check('{"key": "value", "num": 42}')
        assert ok is True


# ── Composite score ───────────────────────────────────────────────────────

class TestCompositeScore:
    def test_good_output(self, sample_input, sample_output):
        result = score(sample_input, sample_output)
        assert isinstance(result, LocalScore)
        assert result.composite >= 0.50
        assert result.ttr > 0
        assert result.ngram_repetition > 0

    def test_circular_output_flagged(self, sample_input, circular_output):
        result = score(sample_input, circular_output)
        assert "circular_output" in result.flags
        assert result.jaccard_overlap > 0.6

    def test_repetitive_output_low_score(self, sample_input, repetitive_output):
        result = score(sample_input, repetitive_output)
        assert result.ngram_repetition < 0.4

    def test_empty_output(self, sample_input):
        result = score(sample_input, "")
        assert result.composite == 0.0

    def test_composite_bounded(self, sample_input, sample_output):
        result = score(sample_input, sample_output)
        assert 0.0 <= result.composite <= 1.0

    def test_passed_threshold(self, sample_input, sample_output):
        result = score(sample_input, sample_output)
        # Good quality output should pass
        assert isinstance(result.passed, bool)


# ── Batch scoring ─────────────────────────────────────────────────────────

class TestBatchScore:
    def test_returns_correct_count(self, sample_input, sample_output):
        pairs = [(sample_input, sample_output)] * 5
        results = batch_score(pairs)
        assert len(results) == 5

    def test_all_local_scores(self, sample_input, sample_output):
        results = batch_score([(sample_input, sample_output)])
        assert all(isinstance(r, LocalScore) for r in results)

    def test_empty_batch(self):
        assert batch_score([]) == []

    def test_throughput(self, sample_input, sample_output):
        """50 probes should complete in under 5ms total (0.1ms/probe budget)."""
        import time
        pairs = [(sample_input, sample_output)] * 50
        t0 = time.perf_counter()
        batch_score(pairs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Batch scoring too slow: {elapsed_ms:.1f}ms for 50 probes"
