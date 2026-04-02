"""Unit tests for benchmark/runner.py.

Uses the `responses` library to mock HTTP calls so no live Ollama instance
is required.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import responses as responses_mock

from benchmark.runner import BenchmarkRunner, OllamaConnectionError

BASE_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ndjson(*chunks: dict) -> bytes:
    """Encode a sequence of dicts as newline-delimited JSON bytes."""
    return b"\n".join(json.dumps(c).encode() for c in chunks)


# ---------------------------------------------------------------------------
# check_connection
# ---------------------------------------------------------------------------

class TestCheckConnection:
    """Tests for BenchmarkRunner.check_connection."""

    @responses_mock.activate
    def test_success(self) -> None:
        """A 200 response from /api/tags should not raise."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": []},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        runner.check_connection()  # Should not raise.

    @responses_mock.activate
    def test_connection_error_raises(self) -> None:
        """A connection error should raise OllamaConnectionError."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            body=ConnectionError("refused"),
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        with pytest.raises(OllamaConnectionError, match="Cannot connect"):
            runner.check_connection()

    @responses_mock.activate
    def test_timeout_raises(self) -> None:
        """A timeout should raise OllamaConnectionError."""
        import requests as req_lib

        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            body=req_lib.exceptions.Timeout("timed out"),
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        with pytest.raises(OllamaConnectionError, match="timed out"):
            runner.check_connection()


# ---------------------------------------------------------------------------
# list_model_variants
# ---------------------------------------------------------------------------

class TestListModelVariants:
    """Tests for BenchmarkRunner.list_model_variants."""

    @responses_mock.activate
    def test_extracts_quant_from_tag(self) -> None:
        """Quant suffix should be extracted from the tag and uppercased."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={
                "models": [
                    {"name": "llama3:8b-q4_K_M"},
                    {"name": "llama3:8b-q8_0"},
                    {"name": "mistral:7b-q5_K_M"},
                ]
            },
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        variants = runner.list_model_variants("llama3")

        assert len(variants) == 2
        names = [v["name"] for v in variants]
        quants = [v["quant"] for v in variants]

        assert "llama3:8b-q4_K_M" in names
        assert "llama3:8b-q8_0" in names
        assert "8B-Q4_K_M" in quants
        assert "8B-Q8_0" in quants

    @responses_mock.activate
    def test_no_match_returns_fallback(self) -> None:
        """If no model matches the prefix, return a single fallback entry."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": [{"name": "mistral:7b-q4_K_M"}]},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        variants = runner.list_model_variants("llama3")

        assert len(variants) == 1
        assert variants[0]["name"] == "llama3"
        assert variants[0]["quant"] == "unknown"

    @responses_mock.activate
    def test_model_without_quant_suffix(self) -> None:
        """A model tag without a colon should be labelled 'unknown'."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": [{"name": "llama3"}]},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        variants = runner.list_model_variants("llama3")

        assert len(variants) == 1
        assert variants[0]["quant"] == "unknown"


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------

class TestRunSingle:
    """Tests for BenchmarkRunner.run_single."""

    @responses_mock.activate
    def test_basic_streaming_response(self) -> None:
        """run_single should return ttft_ms, tokens_per_sec, response, ram_gb."""
        ndjson_body = _make_ndjson(
            {"model": "llama3", "response": "Hello", "done": False},
            {"model": "llama3", "response": " world", "done": False},
            {
                "model": "llama3",
                "response": "",
                "done": True,
                "eval_count": 10,
                "eval_duration": 1_000_000_000,
            },
        )
        responses_mock.add(
            responses_mock.POST,
            f"{BASE_URL}/api/generate",
            body=ndjson_body,
            status=200,
            stream=True,
        )

        runner = BenchmarkRunner(base_url=BASE_URL)
        result = runner.run_single("llama3", "Say hello")

        assert "ttft_ms" in result
        assert result["ttft_ms"] >= 0.0
        assert result["tokens_per_sec"] == pytest.approx(10.0, rel=1e-3)
        assert result["response"] == "Hello world"
        assert result["ram_gb"] > 0.0

    @responses_mock.activate
    def test_connection_error_raises(self) -> None:
        """A connection error during streaming should raise OllamaConnectionError."""
        import requests as req_lib

        responses_mock.add(
            responses_mock.POST,
            f"{BASE_URL}/api/generate",
            body=req_lib.exceptions.ConnectionError("refused"),
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        with pytest.raises(OllamaConnectionError):
            runner.run_single("llama3", "Hello")


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    """Tests for BenchmarkRunner.run_benchmark."""

    def _mock_single(self, runner: BenchmarkRunner, responses_list: list[dict]) -> None:
        """Patch run_single to return items from responses_list in order."""
        # responses_list is consumed in order on each call.
        call_results = iter(responses_list)

        def fake_run_single(model_name: str, prompt: str) -> dict:
            return next(call_results)

        runner.run_single = fake_run_single  # type: ignore[method-assign]

    def test_warmup_is_discarded(self) -> None:
        """With runs=2, only 1 non-warmup run should contribute to averages."""
        runner = BenchmarkRunner(base_url=BASE_URL)

        # runs=2 → index 0 is warmup (ttft=9999), index 1 is real (ttft=100).
        self._mock_single(
            runner,
            [
                {"ttft_ms": 9999.0, "tokens_per_sec": 1.0, "response": "warmup", "ram_gb": 1.0},
                {"ttft_ms": 100.0, "tokens_per_sec": 10.0, "response": "real", "ram_gb": 1.0},
            ],
        )

        result = runner.run_benchmark("llama3:q4", ["What is 2+2?"], runs=2)

        assert len(result["prompts"]) == 1
        prompt_data = result["prompts"][0]

        # Warmup should not influence the average.
        assert prompt_data["avg_ttft_ms"] == pytest.approx(100.0)
        assert prompt_data["avg_tokens_per_sec"] == pytest.approx(10.0)

    def test_averages_are_correct(self) -> None:
        """Averages should be computed correctly across non-warmup runs."""
        runner = BenchmarkRunner(base_url=BASE_URL)

        # runs=3 → index 0 warmup, indices 1 and 2 averaged.
        self._mock_single(
            runner,
            [
                {"ttft_ms": 999.0, "tokens_per_sec": 0.0, "response": "warmup", "ram_gb": 1.0},
                {"ttft_ms": 200.0, "tokens_per_sec": 8.0, "response": "r1", "ram_gb": 1.0},
                {"ttft_ms": 100.0, "tokens_per_sec": 12.0, "response": "r2", "ram_gb": 1.0},
            ],
        )

        result = runner.run_benchmark("llama3:q4", ["Prompt A"], runs=3)
        prompt_data = result["prompts"][0]

        assert prompt_data["avg_ttft_ms"] == pytest.approx(150.0)
        assert prompt_data["avg_tokens_per_sec"] == pytest.approx(10.0)
        assert prompt_data["last_response"] == "r2"

    def test_quant_extracted_from_model_name(self) -> None:
        """Quant should be extracted and uppercased from the model tag."""
        runner = BenchmarkRunner(base_url=BASE_URL)

        self._mock_single(
            runner,
            [
                {"ttft_ms": 100.0, "tokens_per_sec": 5.0, "response": "hi", "ram_gb": 1.0},
                {"ttft_ms": 120.0, "tokens_per_sec": 5.0, "response": "hi", "ram_gb": 1.0},
            ],
        )

        result = runner.run_benchmark("llama3:8b-q4_K_M", ["Hello"], runs=2)
        assert result["quant"] == "8B-Q4_K_M"

    def test_multiple_prompts(self) -> None:
        """Each prompt should produce its own entry in result['prompts']."""
        runner = BenchmarkRunner(base_url=BASE_URL)

        # 2 prompts × 2 runs each = 4 calls.
        self._mock_single(
            runner,
            [
                {"ttft_ms": 10.0, "tokens_per_sec": 5.0, "response": "w1", "ram_gb": 1.0},
                {"ttft_ms": 20.0, "tokens_per_sec": 6.0, "response": "r1", "ram_gb": 1.0},
                {"ttft_ms": 30.0, "tokens_per_sec": 7.0, "response": "w2", "ram_gb": 1.0},
                {"ttft_ms": 40.0, "tokens_per_sec": 8.0, "response": "r2", "ram_gb": 1.0},
            ],
        )

        result = runner.run_benchmark("llama3:q4", ["P1", "P2"], runs=2)
        assert len(result["prompts"]) == 2
        assert result["prompts"][0]["prompt"] == "P1"
        assert result["prompts"][1]["prompt"] == "P2"
