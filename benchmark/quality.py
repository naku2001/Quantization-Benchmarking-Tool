"""Semantic quality scorer using sentence-transformers."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console

# Import at module level so the name is patchable in tests.
# The actual model weights are only downloaded on first use (lazy init).
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

_console = Console()


class QualityScorer:
    """Scores semantic similarity between model responses using embeddings.

    Uses the ``all-MiniLM-L6-v2`` model from ``sentence-transformers``.
    The model is loaded lazily on the first call to :meth:`score` so that
    import time is not penalised when the scorer is instantiated.

    The model download (~90 MB) happens automatically on first use.  A
    :mod:`rich` spinner is shown so the user does not think the tool has hung.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def _load_model(self) -> None:
        """Load ``all-MiniLM-L6-v2`` with a rich spinner.

        No-op if the model is already loaded.
        """
        if self._model is not None:
            return

        with _console.status(
            "[bold cyan]Loading sentence-transformers model "
            "(all-MiniLM-L6-v2, ~90 MB — one-time download)…[/bold cyan]"
        ):
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, baseline: str, candidate: str) -> float:
        """Compute cosine similarity between two strings.

        Args:
            baseline: The reference response (usually from the highest-quality
                quantisation level).
            candidate: The response to evaluate.

        Returns:
            A float in ``[0.0, 1.0]`` where ``1.0`` means identical semantic
            content.
        """
        self._load_model()

        embeddings = self._model.encode([baseline, candidate], convert_to_numpy=True)
        vec_a: np.ndarray = embeddings[0]
        vec_b: np.ndarray = embeddings[1]

        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Clamp to [0.0, 1.0] to handle floating-point edge cases.
        return max(0.0, min(1.0, similarity))

    def score_results(
        self,
        results: list[dict[str, Any]],
        baseline_quant: str,
    ) -> list[dict[str, Any]]:
        """Add ``quality_score`` to every prompt dict in *results*.

        The model whose ``quant`` field matches *baseline_quant* is used as
        the reference.  If no match is found, the first model in *results* is
        used as the baseline.

        Quality is scored once per prompt using the ``last_response`` stored
        in each prompt dict.

        Args:
            results: List of model result dicts as returned by
                :meth:`~benchmark.runner.BenchmarkRunner.run_benchmark`.
            baseline_quant: The quant label to treat as ground truth
                (e.g. ``"Q8"`` or ``"Q8_0"``).

        Returns:
            The same *results* list, mutated in place, with ``"quality_score"``
            added to each prompt dict.
        """
        if not results:
            return results

        # Find the baseline model.
        baseline_model: dict[str, Any] | None = None
        for model in results:
            if model.get("quant", "").upper() == baseline_quant.upper():
                baseline_model = model
                break

        if baseline_model is None:
            baseline_model = results[0]

        # Build a mapping of prompt → baseline response for fast lookup.
        baseline_responses: dict[str, str] = {
            p["prompt"]: p.get("last_response", "")
            for p in baseline_model.get("prompts", [])
        }

        for model in results:
            for prompt_dict in model.get("prompts", []):
                prompt_text: str = prompt_dict["prompt"]
                baseline_text: str = baseline_responses.get(prompt_text, "")
                candidate_text: str = prompt_dict.get("last_response", "")

                if model is baseline_model:
                    # Baseline always scores 1.0 against itself.
                    prompt_dict["quality_score"] = 1.0
                else:
                    prompt_dict["quality_score"] = self.score(
                        baseline_text, candidate_text
                    )

        return results
