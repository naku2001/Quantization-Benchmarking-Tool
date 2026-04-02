"""BenchmarkRunner — core timing logic talking to the Ollama REST API."""

import json
import time
from typing import Any

import requests

from benchmark.metrics import calculate_throughput, calculate_ttft, get_ram_usage_gb

# Approximate character-to-token ratio used for prompt resizing.
CHARS_PER_TOKEN: int = 4

# Default context sizes (in tokens) used by run_context_sweep.
DEFAULT_CONTEXT_SIZES: list[int] = [512, 2048, 4096]


def _resize_prompt(prompt: str, target_tokens: int) -> str:
    """Return a version of *prompt* approximately *target_tokens* tokens long.

    Uses a rough ``CHARS_PER_TOKEN`` approximation (4 chars per token).
    If the prompt is longer than the target it is truncated.  If shorter,
    the prompt text is repeated until the target length is reached.

    Args:
        prompt: The original prompt string.
        target_tokens: Desired length in tokens.

    Returns:
        Resized prompt string.
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    if len(prompt) >= target_chars:
        return prompt[:target_chars]
    # Pad by repeating the prompt.
    reps = (target_chars // len(prompt)) + 1
    return (prompt * reps)[:target_chars]


class OllamaConnectionError(Exception):
    """Raised when the Ollama API is unreachable or returns an unexpected error.

    Callers should catch this exception and display a user-friendly message
    instructing the user to ensure Ollama is running (``ollama serve``).
    """


class BenchmarkRunner:
    """Runs inference benchmarks against a local Ollama instance.

    All communication with Ollama is done over its REST API using the
    ``requests`` library — no Ollama SDK is used.

    Args:
        base_url: Base URL of the Ollama server, e.g. ``http://localhost:11434``.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def check_connection(self) -> None:
        """Verify that the Ollama server is reachable.

        Sends a ``GET /api/tags`` request.  If the request times out or raises
        a connection error, :class:`OllamaConnectionError` is raised with a
        message telling the user to start Ollama.

        Raises:
            OllamaConnectionError: When the server cannot be reached.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except (requests.exceptions.ConnectionError, ConnectionError):
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please make sure Ollama is running (run 'ollama serve')."
            )
        except requests.exceptions.Timeout:
            raise OllamaConnectionError(
                f"Connection to Ollama at {self.base_url} timed out. "
                "Please make sure Ollama is running (run 'ollama serve')."
            )
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Unexpected error connecting to Ollama at {self.base_url}: {exc}. "
                "Please make sure Ollama is running (run 'ollama serve')."
            )

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_model_variants(self, model_name: str) -> list[dict[str, str]]:
        """List all pulled variants of *model_name* that Ollama knows about.

        Queries ``GET /api/tags``, then filters the returned model list to
        those whose tag starts with *model_name*.  The quantisation level is
        extracted from the suffix after the last colon in the tag string (e.g.
        ``llama3:8b-q4_K_M`` → ``Q4_K_M``).  Names without a recognisable
        quant suffix are labelled ``"unknown"``.

        Args:
            model_name: The model name prefix to filter by, e.g. ``"llama3"``.

        Returns:
            A list of dicts, each with keys ``"name"`` (full tag string) and
            ``"quant"`` (upper-cased quant label).  If no matching models are
            found, returns a single entry using *model_name* as-is with quant
            ``"unknown"``.

        Raises:
            OllamaConnectionError: If the API call fails.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Failed to list models from Ollama: {exc}. "
                "Please make sure Ollama is running (run 'ollama serve')."
            )

        data: dict[str, Any] = response.json()
        all_models: list[dict[str, Any]] = data.get("models", [])

        # Filter to models whose name starts with the requested prefix.
        # Accept both exact matches ("llama3") and tagged variants ("llama3:…").
        matching = [
            m for m in all_models
            if m.get("name", "").startswith(model_name)
        ]

        if not matching:
            # Model may be available but not listed (e.g., pulled under a
            # slightly different name).  Return a best-effort entry.
            return [{"name": model_name, "quant": "unknown"}]

        variants: list[dict[str, str]] = []
        for m in matching:
            full_name: str = m["name"]
            # Extract the portion after the last colon as the quant label.
            if ":" in full_name:
                suffix = full_name.rsplit(":", 1)[-1]
                # Normalise to uppercase for display.
                quant = suffix.upper() if suffix else "unknown"
            else:
                quant = "unknown"
            variants.append({"name": full_name, "quant": quant})

        return variants

    # ------------------------------------------------------------------
    # Single inference run
    # ------------------------------------------------------------------

    def run_single(self, model_name: str, prompt: str) -> dict[str, Any]:
        """Run one inference request and return timing + response data.

        Streams the response from ``POST /api/generate``, measuring:

        * **TTFT** — time from request start to first non-empty chunk.
        * **Throughput** — tokens/sec derived from Ollama's ``eval_count``
          and ``eval_duration`` fields in the final done-chunk.
        * **RAM** — process RSS at the moment the response completes.

        Args:
            model_name: Full Ollama model tag, e.g. ``"llama3:8b-q4_K_M"``.
            prompt: The prompt string to send.

        Returns:
            A dict with keys:
            ``ttft_ms`` (float), ``tokens_per_sec`` (float),
            ``response`` (str), ``ram_gb`` (float).

        Raises:
            OllamaConnectionError: If the streaming request fails.
        """
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
        }

        first_chunk_time: float | None = None
        full_response_parts: list[str] = []
        eval_count: int = 0
        eval_duration_ns: int = 0

        try:
            start_time = time.perf_counter()
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue

                    try:
                        chunk: dict[str, Any] = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    chunk_text: str = chunk.get("response", "")

                    # Record TTFT on the first non-empty content chunk.
                    if first_chunk_time is None and chunk_text:
                        first_chunk_time = time.perf_counter()

                    full_response_parts.append(chunk_text)

                    if chunk.get("done", False):
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
                        break

        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Error during inference request to Ollama: {exc}. "
                "Please make sure Ollama is running (run 'ollama serve')."
            )

        # Fallback: if no non-empty chunk arrived, use the end time for TTFT.
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()

        ttft_ms = calculate_ttft(start_time, first_chunk_time)
        tokens_per_sec = calculate_throughput(eval_count, eval_duration_ns)
        ram_gb = get_ram_usage_gb()
        full_response = "".join(full_response_parts)

        return {
            "ttft_ms": ttft_ms,
            "tokens_per_sec": tokens_per_sec,
            "response": full_response,
            "ram_gb": ram_gb,
        }

    # ------------------------------------------------------------------
    # Multi-run benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        model_name: str,
        prompts: list[str],
        runs: int = 3,
    ) -> dict[str, Any]:
        """Benchmark *model_name* across a list of prompts.

        For each prompt, the model is called *runs* times.  Run index 0 is
        treated as a warmup and discarded.  The remaining runs are averaged.
        The last run's response text is stored for quality scoring.

        Args:
            model_name: Full Ollama model tag.
            prompts: List of prompt strings to evaluate.
            runs: Total number of runs per prompt (including the warmup).
                  Must be >= 2 to have at least one non-warmup run.

        Returns:
            A dict matching the benchmark JSON schema::

                {
                    "name": "llama3:8b-q4_K_M",
                    "quant": "Q4_K_M",
                    "prompts": [
                        {
                            "prompt": "...",
                            "runs": [{"ttft_ms": ..., "tokens_per_sec": ..., "ram_gb": ...}],
                            "avg_ttft_ms": 521.0,
                            "avg_tokens_per_sec": 7.3,
                            "last_response": "..."
                        }
                    ]
                }
        """
        # Extract quant label from the model name tag.
        if ":" in model_name:
            suffix = model_name.rsplit(":", 1)[-1]
            quant = suffix.upper() if suffix else "unknown"
        else:
            quant = "unknown"

        prompt_results: list[dict[str, Any]] = []

        for prompt in prompts:
            run_records: list[dict[str, Any]] = []
            last_response: str = ""

            for run_idx in range(runs):
                result = self.run_single(model_name, prompt)
                last_response = result["response"]

                run_records.append(
                    {
                        "ttft_ms": result["ttft_ms"],
                        "tokens_per_sec": result["tokens_per_sec"],
                        "ram_gb": result["ram_gb"],
                    }
                )

            # Discard the first run (index 0 = warmup).
            real_runs = run_records[1:] if len(run_records) > 1 else run_records

            avg_ttft = sum(r["ttft_ms"] for r in real_runs) / len(real_runs)
            avg_tps = sum(r["tokens_per_sec"] for r in real_runs) / len(real_runs)

            prompt_results.append(
                {
                    "prompt": prompt,
                    "runs": run_records,
                    "avg_ttft_ms": avg_ttft,
                    "avg_tokens_per_sec": avg_tps,
                    "last_response": last_response,
                }
            )

        return {
            "name": model_name,
            "quant": quant,
            "prompts": prompt_results,
        }

    # ------------------------------------------------------------------
    # Context-length sweep
    # ------------------------------------------------------------------

    def run_context_sweep(
        self,
        model_name: str,
        prompts: list[str],
        runs: int = 3,
        context_sizes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Benchmark *model_name* at multiple input context sizes.

        For each size in *context_sizes*, every prompt is truncated or padded
        to approximately that many tokens before being sent to Ollama.  The
        model's own context window is unchanged — this measures how input
        length affects TTFT, throughput, and response quality.

        Args:
            model_name: Full Ollama model tag.
            prompts: Original prompt strings.
            runs: Total runs per prompt per context size (first is warmup).
            context_sizes: Token counts to test.
                Defaults to ``DEFAULT_CONTEXT_SIZES`` (512, 2048, 4096).

        Returns:
            A list of result dicts, one per context size.  Each dict is
            identical to the output of :meth:`run_benchmark` but carries an
            additional ``"context_size"`` key (int) and an
            ``"original_prompt"`` key on every prompt dict.
        """
        if context_sizes is None:
            context_sizes = DEFAULT_CONTEXT_SIZES

        sweep_results: list[dict[str, Any]] = []
        for ctx_size in context_sizes:
            resized_prompts = [_resize_prompt(p, ctx_size) for p in prompts]
            result = self.run_benchmark(model_name, resized_prompts, runs)
            result["context_size"] = ctx_size
            # Annotate each prompt entry with the original (unresized) text.
            for i, prompt_dict in enumerate(result["prompts"]):
                prompt_dict["original_prompt"] = prompts[i]
            sweep_results.append(result)

        return sweep_results
