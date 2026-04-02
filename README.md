**LLM Quantization Benchmarking Tool**

> Benchmark Ollama models across quantization levels  measuring speed, throughput, and semantic quality degradation entirely on CPU-only hardware.

---

## Overview

**quant-bench** is a CLI tool that runs structured benchmarks against locally served [Ollama](https://ollama.com) models at different quantization levels (Q4, Q5, Q6, Q8). For each model variant it measures:

- **Time to First Token (TTFT)**
- **Throughput** 
- **Semantic quality score**`

Results are written as a rich terminal table, structured JSON, a Markdown report, and a scatter-plot chart visualising the speed/quality tradeoff curve.

---

## Features

- Runs entirely locally
- Designed for CPU-only hardware
- Supports multiple models in a single run
- Warmup-aware: first run per prompt is discarded, remaining runs are averaged
- Auto-discovers quantization variants from Ollama tags
- Outputs results in `table`, `json`, `chart`, or `all` formats

---

## Pulling Model Variants

**Ollama does not auto-quantize.** Each quantization variant is a separate model
file that must be pulled individually before benchmarking.

```bash
# Pull the specific quant variants you want to compare
ollama pull llama3:8b-q4_K_M
ollama pull llama3:8b-q5_K_M
ollama pull llama3:8b-q6_K
ollama pull llama3:8b-q8_0
```

> **Default quant:** Running `ollama pull llama3` with no tag pulls the `Q4_K_M`
> variant. It does **not** pull all quants automatically.

To see which variants you currently have available:

```bash
ollama list
```

quant-bench queries this list at startup, filters by the model name prefix you
pass with `--model`, and benchmarks every matching variant it finds. If you have
only one variant pulled it will benchmark just that one.

### Context windows and quantization

The context window size (maximum input sequence length) is fixed by the model's
architecture and **does not change across quantization levels** — a Q4 and Q8
variant of the same model accept exactly the same maximum number of tokens.

What does change is **attention quality over long inputs**. Lower-quant models
use fewer bits to represent attention weights and activations, which introduces
approximation error that compounds as the sequence grows longer. At short context
lengths the degradation is often imperceptible; at long context lengths it can
cause the model to lose track of earlier content, repeat itself, or miss details
buried in the middle of the input.

The `--context-sweep` mode is designed to measure this effect: it runs the same
prompt at 512, 2048, and 4096 input tokens and plots quality score against
context length — one line per quant level. The resulting `context_sweep.png`
chart visualises exactly where each quantization level starts to degrade.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally at `localhost:11434`
- At least one model pulled (see [Pulling Model Variants](#pulling-model-variants) above)

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/naku2001/Quantization-Bemchmarking-Tool.git
cd Quantization-Bemchmarking-Tool

# Editable install — lets you edit the source and run quant-bench immediately
pip install -e .

# Install dev dependencies (pytest, ruff, etc.)
pip install -e ".[dev]"
```

### From a local build (non-editable)

```bash
pip install build
python -m build        # produces dist/quant_bench-0.1.0-py3-none-any.whl
pip install dist/quant_bench-0.1.0-py3-none-any.whl
```

After either install the `quant-bench` command is available globally:

```bash
quant-bench --model llama3 --runs 3
```

> **Note:** `sentence-transformers` downloads the `all-MiniLM-L6-v2` model
> (~90 MB) on first use.  A rich spinner will indicate this is in progress.

---

## Usage

```bash
# Basic benchmark — single model, 3 runs per prompt
python main.py --model llama3

# Multiple models in one run
python main.py --model llama3 --model mistral --runs 5

# Use a different prompt category
python main.py --model llama3 --prompts prompts/reasoning.txt

# Use long-context prompts
python main.py --model llama3 --prompts prompts/long_context.txt --runs 2

# Output only JSON (skips terminal table)
python main.py --model llama3 --format json

# Context-length sweep (quality vs input size per quant)
python main.py --model llama3 --prompts prompts/long_context.txt --context-sweep

# Specify a custom output directory
python main.py --model llama3 --output my_results/
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--model` | *(required)* | Ollama model name. Repeatable for multiple models. |
| `--runs` | `3` | Runs per prompt. First run is warmup and is discarded. |
| `--prompts` | `prompts/factual.txt` | Path to a prompt file (one prompt per line). |
| `--format` | `all` | Output format: `table` \| `json` \| `chart` \| `all` |
| `--output` | `results/` | Directory for result files. |
| `--context-sweep` | off | Run each prompt at 512 / 2048 / 4096 input tokens and produce a quality-vs-context-length chart. |

---

## Project Structure

```
quant-bench/
├── main.py                   # CLI entry point (click)
├── requirements.txt
├── CLAUDE.md                 # Architecture & implementation notes
├── benchmark/
│   ├── runner.py             # Core timing logic, Ollama API calls
│   ├── metrics.py            # TTFT, throughput, memory helpers
│   ├── quality.py            # Semantic similarity scorer
│   └── reporter.py           # JSON, Markdown, and chart output
├── prompts/
│   ├── factual.txt           # Short factual questions
│   ├── reasoning.txt         # Multi-step reasoning prompts
│   ├── creative.txt          # Open-ended generation prompts
│   └── long_context.txt      # Dense 1000–2000 token passages for context sweep
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

## Output Formats

### Terminal Table

Displayed live after each model completes using `rich`. Columns: Model, Quant, TTFT (ms), Tokens/sec, Quality Score, RAM (GB).

### `results/results.json`

Full structured data including all raw per-run values, averaged metrics, model metadata, and timestamp.

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

### `results/report.md`

Markdown table with averaged metrics per model/quant, plus a **Key Finding** line identifying the best speed/quality ratio.

### `results/chart.png`

Scatter plot with tokens/sec on the X-axis and quality score on the Y-axis. Each point represents one model/quant combination, labelled for easy comparison.

### `results/context_sweep.png`

Line chart produced by `--context-sweep`. X-axis is input context size in tokens (512 / 2048 / 4096); Y-axis is average quality score. One line per quant level — shows where each quantization starts to degrade over longer inputs.

---

## How It Works

**TTFT** is measured by streaming the Ollama `/api/generate` response and recording `time.perf_counter()` before the request and again when the first non-empty chunk arrives.

**Throughput** uses `eval_count` and `eval_duration` from Ollama's final done-chunk JSON — no manual token counting.

**Quality scoring** uses Q8 (or the highest available quant) as a baseline. Both the baseline and test responses are encoded with `all-MiniLM-L6-v2` and compared via cosine similarity. A score of `1.0` means semantically identical output.

**Quant detection** queries `/api/tags`, filters by model name prefix, and extracts the quant suffix from the tag string (e.g. `llama3:8b-q4_K_M` → `Q4_K_M`).

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock the Ollama REST API using the `responses` library — no live Ollama instance required. The quality scorer encoder is also mocked in CI to avoid the model download.

```bash
# Lint
ruff check .
```

---

## Stack

| Component | Library |
|---|---|
| CLI | `click` |
| Terminal output | `rich` |
| Ollama API | `requests` (REST, no SDK) |
| Quality scoring | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Charts | `matplotlib` |
| System metrics | `psutil` |
| Testing | `pytest`, `responses` |
| Linting | `ruff` |

---

## Notes

- Ollama must be running before invoking the CLI. The tool checks connectivity at startup with a `GET /api/tags` call and exits with a clear error if it fails.
- On CPU-only hardware, sentence-transformers scoring is slow. Quality scoring runs once per prompt (not per run) to keep total benchmark time manageable.
- The `results/` directory is gitignored. Raw JSON results are always saved even when `--format table` is specified, so no data is lost.

---

## License

MIT
