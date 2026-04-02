# quant-bench — LLM Quantization Benchmarking Tool

A CLI tool that benchmarks Ollama models across quantization levels (Q4, Q5, Q6, Q8),
measuring Time to First Token, tokens/sec throughput, and semantic quality degradation.
Designed to run on CPU-only hardware.

---

## Stack

- **Language:** Python 3.10+
- **Ollama API:** REST via `requests` (localhost:11434) — no SDK
- **Quality scoring:** `sentence-transformers` (all-MiniLM-L6-v2)
- **CLI:** `click`
- **Output:** `rich` (terminal tables), `matplotlib` (chart), JSON, Markdown
- **Testing:** `pytest`

---

## Project Structure

```
quant-bench/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── main.py                   # CLI entry point (click)
├── benchmark/
│   ├── __init__.py
│   ├── runner.py             # BenchmarkRunner — core timing logic
│   ├── metrics.py            # TTFT, throughput, memory helpers
│   ├── quality.py            # Semantic similarity scorer
│   └── reporter.py           # JSON, markdown table, chart output
├── prompts/
│   ├── factual.txt           # Short factual questions (one per line)
│   ├── reasoning.txt         # Multi-step reasoning prompts
│   └── creative.txt          # Open-ended generation prompts
├── results/                  # Auto-created on first run (gitignored)
│   ├── results.json
│   ├── report.md
│   └── chart.png
└── tests/
    ├── test_runner.py
    ├── test_metrics.py
    └── test_quality.py
```

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark (basic)
python main.py --model llama3 --runs 3

# Run benchmark (multiple models)
python main.py --model llama3 --model mistral --runs 3

# Run with specific prompt category
python main.py --model llama3 --prompts prompts/reasoning.txt

# Output only JSON
python main.py --model llama3 --format json

# Run tests
pytest tests/ -v

# Lint
ruff check .
```

---

## Architecture & Key Decisions

**Ollama REST API endpoints used:**
- `GET  /api/tags`           — list available models
- `POST /api/generate`       — run inference (stream=true)
- `POST /api/show`           — get model metadata

**TTFT measurement:** Stream the response via `requests` with `stream=True`.
Record `time.perf_counter()` before the request and again when the first
non-empty chunk arrives. This is TTFT in milliseconds.

**Throughput:** Use `eval_count` and `eval_duration` from Ollama's final
done-chunk JSON rather than manually counting tokens. Ollama provides these
natively — do not reimplement token counting.

**Quality scoring:** Use Q8 (or the highest available quant) as the baseline.
Encode both the baseline response and the test response with
`all-MiniLM-L6-v2`, then compute cosine similarity. Score of 1.0 = identical
semantic content.

**Multi-run averaging:** Run each prompt N times. Discard the first run
(warmup) and average the rest. Store all raw run data in JSON so users can
inspect variance.

**Quant level detection:** Query `/api/tags`, filter by model name prefix,
extract quant suffix from the tag string (e.g. `llama3:8b-q4_K_M` → `Q4_K_M`).
If only one tag exists (no quant variants pulled), run it as-is and label it
as the detected quant.

---

## CLI Interface (main.py)

```
python main.py [OPTIONS]

Options:
  --model TEXT       Ollama model name. Repeatable for multiple models.
  --runs INTEGER     Runs per prompt to average. First run is warmup. [default: 3]
  --prompts PATH     Path to prompt file (one prompt per line). [default: prompts/factual.txt]
  --format TEXT      Output: table | json | chart | all [default: all]
  --output PATH      Directory for result files [default: results/]
  --help
```

---

## Output Formats

**Terminal table** (via `rich`): Displayed after each model finishes.
Columns: Model, Quant, TTFT (ms), Tokens/sec, Quality Score, RAM (GB).

**results/results.json:** Full structured data including all raw runs,
averaged metrics, model metadata, and timestamp. Schema:
```json
{
  "run_id": "2026-03-31T14:22:00",
  "models": [
    {
      "name": "llama3:8b-q4_K_M",
      "quant": "Q4_K_M",
      "prompts": [
        {
          "prompt": "...",
          "runs": [...],
          "avg_ttft_ms": 521,
          "avg_tokens_per_sec": 7.3,
          "quality_score": 0.961
        }
      ]
    }
  ]
}
```

**results/report.md:** Markdown table summarising averaged metrics per model/quant.
Include a "Key Finding" line beneath the table (e.g. which quant gives best speed/quality ratio).

**results/chart.png:** Scatter plot. X-axis = tokens/sec, Y-axis = quality score.
One point per model/quant combination, labelled. This visualises the tradeoff curve.

---

## Code Style

- Type hints on all function signatures
- Docstrings on all public classes and methods
- No global state — pass config objects between functions
- Errors from Ollama API should raise a custom `OllamaConnectionError` with
  a clear message telling the user to check that Ollama is running
- Use `rich.console.Console` for all terminal output, not bare `print()`
- All file writes go through `reporter.py` — nothing else writes to disk

---

## Gotchas

- Ollama must be running before the CLI is invoked. Check connectivity with a
  `GET /api/tags` call at startup and fail fast with a clear error if it times out.
- `all-MiniLM-L6-v2` downloads on first use (~90MB). Inform the user with a
  `rich` spinner so they don't think it hung.
- On CPU-only hardware, quality scoring with sentence-transformers is slow.
  Run it once per prompt (not per run) using the last run's response as input.
- The `results/` directory should be gitignored. Add it to `.gitignore`.
- Ollama quant tag naming is inconsistent across models. Normalise to uppercase
  when displaying (q4_K_M → Q4_K_M) but use the raw tag string for API calls.

---

## Testing Strategy

- `test_runner.py`: Mock the Ollama REST API with `responses` library.
  Test TTFT calculation, multi-run averaging, and warmup discard logic.
- `test_metrics.py`: Unit test throughput calculation using known token counts
  and durations.
- `test_quality.py`: Test cosine similarity returns 1.0 for identical strings
  and <1.0 for different strings. Do not download the model in CI — mock the
  encoder.

---

## requirements.txt

```
requests>=2.31.0
sentence-transformers>=2.7.0
matplotlib>=3.8.0
psutil>=5.9.0
click>=8.1.7
rich>=13.7.0
pytest>=8.0.0
responses>=0.25.0
ruff>=0.4.0
```

---

## Build Order

Build in this sequence to avoid integration complexity:

1. `benchmark/metrics.py` — pure functions, no dependencies
2. `benchmark/runner.py` — depends on metrics, mocks Ollama
3. `benchmark/quality.py` — standalone scorer
4. `benchmark/reporter.py` — takes results dict, writes files
5. `main.py` — wires everything together via click
6. Tests for each module
7. `README.md` update with real benchmark numbers from a test run
