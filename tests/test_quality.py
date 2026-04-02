"""Unit tests for benchmark/quality.py.

The SentenceTransformer model is mocked throughout so no network download
occurs during CI runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmark.quality import QualityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(values: list[float]) -> np.ndarray:
    """Return a normalised numpy vector."""
    arr = np.array(values, dtype=float)
    return arr / np.linalg.norm(arr)


def _mock_encoder(embeddings: list[np.ndarray]):
    """Return a mock SentenceTransformer whose encode() returns *embeddings*."""
    mock = MagicMock()
    mock.encode.return_value = np.array(embeddings)
    return mock


# ---------------------------------------------------------------------------
# QualityScorer.score
# ---------------------------------------------------------------------------

class TestScore:
    """Tests for QualityScorer.score."""

    def test_identical_strings_score_one(self) -> None:
        """Identical embeddings should produce a cosine similarity of 1.0."""
        vec = _unit_vec([1.0, 0.0, 0.0])
        scorer = QualityScorer()
        scorer._model = _mock_encoder([vec, vec])

        result = scorer.score("hello world", "hello world")

        assert result == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        """Orthogonal embeddings should produce a cosine similarity of 0.0."""
        vec_a = _unit_vec([1.0, 0.0])
        vec_b = _unit_vec([0.0, 1.0])
        scorer = QualityScorer()
        scorer._model = _mock_encoder([vec_a, vec_b])

        result = scorer.score("hello world", "completely different text")

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_different_strings_score_less_than_one(self) -> None:
        """Non-identical embeddings should produce similarity < 1.0."""
        vec_a = _unit_vec([1.0, 0.0, 0.0])
        vec_b = _unit_vec([0.5, 0.5, 0.0])
        scorer = QualityScorer()
        scorer._model = _mock_encoder([vec_a, vec_b])

        result = scorer.score("hello world", "completely different text")

        assert result < 1.0

    def test_zero_vector_returns_zero(self) -> None:
        """A zero-norm vector should return 0.0 without raising."""
        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = _unit_vec([1.0, 0.0, 0.0])
        scorer = QualityScorer()
        scorer._model = _mock_encoder([vec_a, vec_b])

        result = scorer.score("", "hello")

        assert result == 0.0

    def test_score_clamped_to_one(self) -> None:
        """Floating-point noise should never push the result above 1.0."""
        # Simulate a tiny floating-point overshoot.
        vec = _unit_vec([1.0, 0.0])
        scorer = QualityScorer()
        # Make encode return vectors whose dot product slightly exceeds 1.
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([vec * 1.0000001, vec])
        scorer._model = mock_model

        result = scorer.score("x", "x")

        assert result <= 1.0


# ---------------------------------------------------------------------------
# QualityScorer._load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    """Tests for QualityScorer._load_model."""

    def test_loads_once(self) -> None:
        """The SentenceTransformer should only be instantiated once."""
        scorer = QualityScorer()
        mock_st_cls = MagicMock(return_value=MagicMock())

        with patch("benchmark.quality.SentenceTransformer", mock_st_cls):
            scorer._load_model()
            scorer._load_model()  # second call should be a no-op

        mock_st_cls.assert_called_once_with("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# QualityScorer.score_results
# ---------------------------------------------------------------------------

class TestScoreResults:
    """Tests for QualityScorer.score_results."""

    def _make_results(self) -> list[dict]:
        """Build a minimal results list with two model variants."""
        return [
            {
                "name": "llama3:q8_0",
                "quant": "Q8_0",
                "prompts": [
                    {"prompt": "What is 2+2?", "last_response": "4", "avg_ttft_ms": 100.0, "avg_tokens_per_sec": 5.0},
                ],
            },
            {
                "name": "llama3:q4_K_M",
                "quant": "Q4_K_M",
                "prompts": [
                    {"prompt": "What is 2+2?", "last_response": "four", "avg_ttft_ms": 80.0, "avg_tokens_per_sec": 8.0},
                ],
            },
        ]

    def test_quality_score_added_to_prompts(self) -> None:
        """Each prompt dict should gain a 'quality_score' key."""
        scorer = QualityScorer()
        # Mock score() to return a fixed value for non-baseline models.
        scorer.score = MagicMock(return_value=0.85)  # type: ignore[method-assign]

        results = self._make_results()
        updated = scorer.score_results(results, baseline_quant="Q8_0")

        for model in updated:
            for p in model["prompts"]:
                assert "quality_score" in p

    def test_baseline_scores_one(self) -> None:
        """The baseline model should always receive quality_score == 1.0."""
        scorer = QualityScorer()
        scorer.score = MagicMock(return_value=0.75)  # type: ignore[method-assign]

        results = self._make_results()
        updated = scorer.score_results(results, baseline_quant="Q8_0")

        baseline_prompts = updated[0]["prompts"]
        assert baseline_prompts[0]["quality_score"] == pytest.approx(1.0)

    def test_non_baseline_score_comes_from_scorer(self) -> None:
        """Non-baseline models should receive quality_score from scorer.score()."""
        scorer = QualityScorer()
        scorer.score = MagicMock(return_value=0.88)  # type: ignore[method-assign]

        results = self._make_results()
        updated = scorer.score_results(results, baseline_quant="Q8_0")

        non_baseline_prompts = updated[1]["prompts"]
        assert non_baseline_prompts[0]["quality_score"] == pytest.approx(0.88)

    def test_fallback_to_first_model_when_baseline_not_found(self) -> None:
        """If baseline_quant is not in results, the first model is used as baseline."""
        scorer = QualityScorer()
        scorer.score = MagicMock(return_value=0.5)  # type: ignore[method-assign]

        results = self._make_results()
        updated = scorer.score_results(results, baseline_quant="Q6_K")

        # First model is the fallback baseline → quality_score == 1.0.
        assert updated[0]["prompts"][0]["quality_score"] == pytest.approx(1.0)
        # Second model gets scored normally.
        assert updated[1]["prompts"][0]["quality_score"] == pytest.approx(0.5)

    def test_empty_results_returns_empty(self) -> None:
        """Empty input should return empty output without error."""
        scorer = QualityScorer()
        result = scorer.score_results([], baseline_quant="Q8_0")
        assert result == []
