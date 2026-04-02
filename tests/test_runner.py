"""Unit tests for benchmark/runner.py.

Uses the `responses` library to mock HTTP calls so no live Ollama instance
is required.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import responses as responses_mock

from benchmark.runner import (
    BenchmarkRunner,
    OllamaConnectionError,
    _resize_prompt,
)

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


# ---------------------------------------------------------------------------
# _resize_prompt
# ---------------------------------------------------------------------------

class TestResizePrompt:
    """Tests for the _resize_prompt helper."""

    def test_truncates_long_prompt(self) -> None:
        """A prompt longer than the target should be truncated to target_tokens * 4 chars."""
        prompt = "a" * 10_000
        result = _resize_prompt(prompt, 512)
        assert len(result) == 512 * 4

    def test_pads_short_prompt(self) -> None:
        """A prompt shorter than the target should be padded to target_tokens * 4 chars."""
        prompt = "hello "
        result = _resize_prompt(prompt, 100)
        assert len(result) == 100 * 4

    def test_exact_length_unchanged(self) -> None:
        """A prompt exactly at the target length should be returned as-is."""
        prompt = "x" * (256 * 4)
        result = _resize_prompt(prompt, 256)
        assert len(result) == 256 * 4

    def test_padded_content_is_repetition_of_original(self) -> None:
        """Padding should be achieved by repeating the original prompt text."""
        prompt = "abc"
        result = _resize_prompt(prompt, 10)
        # The result should be the first 40 chars of "abcabcabc…"
        expected = ("abc" * ((40 // 3) + 1))[:40]
        assert result == expected


# ---------------------------------------------------------------------------
# run_context_sweep
# ---------------------------------------------------------------------------

class TestRunContextSweep:
    """Tests for BenchmarkRunner.run_context_sweep."""

    def _mock_benchmark(
        self, runner: BenchmarkRunner, return_value: dict
    ) -> None:
        """Patch run_benchmark to always return *return_value*."""
        runner.run_benchmark = lambda _m, prompts, _r=3: dict(  # type: ignore[method-assign]
            return_value, prompts=[{"prompt": p, "runs": [], "avg_ttft_ms": 100.0,
                                    "avg_tokens_per_sec": 5.0, "last_response": "r"} for p in prompts]
        )

    def test_produces_one_result_per_context_size(self) -> None:
        """Each context size should produce a separate result dict."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_benchmark(runner, {"name": "llama3:q4", "quant": "Q4"})

        results = runner.run_context_sweep(
            "llama3:q4", ["Hello"], runs=2, context_sizes=[512, 2048]
        )

        assert len(results) == 2
        assert results[0]["context_size"] == 512
        assert results[1]["context_size"] == 2048

    def test_context_size_tagged_on_each_result(self) -> None:
        """Every result dict should carry the correct context_size key."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_benchmark(runner, {"name": "llama3:q4", "quant": "Q4"})

        results = runner.run_context_sweep(
            "llama3:q4", ["Hi"], runs=2, context_sizes=[512, 4096]
        )

        sizes = [r["context_size"] for r in results]
        assert sizes == [512, 4096]

    def test_prompts_are_resized_to_target(self) -> None:
        """Prompts passed to run_benchmark should be resized to the target length."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        captured: list[list[str]] = []

        def fake_benchmark(model_name: str, prompts: list[str], runs: int = 3) -> dict:
            captured.append(list(prompts))
            return {
                "name": model_name,
                "quant": "Q4",
                "prompts": [{"prompt": p, "runs": [], "avg_ttft_ms": 10.0,
                              "avg_tokens_per_sec": 5.0, "last_response": "r"} for p in prompts],
            }

        runner.run_benchmark = fake_benchmark  # type: ignore[method-assign]

        original_prompt = "x" * 100
        runner.run_context_sweep(
            "llama3:q4", [original_prompt], runs=2, context_sizes=[512]
        )

        assert len(captured) == 1
        resized = captured[0][0]
        assert len(resized) == 512 * 4

    def test_original_prompt_stored_on_prompt_dict(self) -> None:
        """Each prompt dict in the sweep result should carry original_prompt."""
        runner = BenchmarkRunner(base_url=BASE_URL)

        def fake_benchmark(model_name: str, prompts: list[str], runs: int = 3) -> dict:
            return {
                "name": model_name,
                "quant": "Q4",
                "prompts": [{"prompt": p, "runs": [], "avg_ttft_ms": 10.0,
                              "avg_tokens_per_sec": 5.0, "last_response": "r"} for p in prompts],
            }

        runner.run_benchmark = fake_benchmark  # type: ignore[method-assign]

        original = "Hello world"
        results = runner.run_context_sweep(
            "llama3:q4", [original], runs=2, context_sizes=[512]
        )

        assert results[0]["prompts"][0]["original_prompt"] == original

    def test_uses_default_context_sizes_when_none_given(self) -> None:
        """Omitting context_sizes should default to [512, 2048, 4096]."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        call_count = 0

        def fake_benchmark(model_name: str, prompts: list[str], runs: int = 3) -> dict:
            nonlocal call_count
            call_count += 1
            return {
                "name": model_name,
                "quant": "Q4",
                "prompts": [{"prompt": p, "runs": [], "avg_ttft_ms": 10.0,
                              "avg_tokens_per_sec": 5.0, "last_response": "r"} for p in prompts],
            }

        runner.run_benchmark = fake_benchmark  # type: ignore[method-assign]

        results = runner.run_context_sweep("llama3:q4", ["Hi"], runs=2)

        assert call_count == 3  # one call per default context size
        assert [r["context_size"] for r in results] == [512, 2048, 4096]
