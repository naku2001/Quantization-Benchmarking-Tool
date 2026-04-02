"""All output — terminal, JSON, Markdown, and chart — is produced here.

Nothing outside this module should write to disk or render structured output.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

_console = Console()


def _model_averages(model: dict[str, Any]) -> dict[str, float]:
    """Compute per-model averages across all prompts.

    Args:
        model: A single model result dict as returned by
            :class:`~benchmark.runner.BenchmarkRunner`.

    Returns:
        A dict with keys ``avg_ttft_ms``, ``avg_tokens_per_sec``,
        ``avg_quality_score``, and ``avg_ram_gb``.
    """
    prompts: list[dict[str, Any]] = model.get("prompts", [])
    if not prompts:
        return {
            "avg_ttft_ms": 0.0,
            "avg_tokens_per_sec": 0.0,
            "avg_quality_score": 0.0,
            "avg_ram_gb": 0.0,
        }

    ttft_values = [p.get("avg_ttft_ms", 0.0) for p in prompts]
    tps_values = [p.get("avg_tokens_per_sec", 0.0) for p in prompts]
    quality_values = [p.get("quality_score", 0.0) for p in prompts]

    # RAM: average the last run's RAM across prompts (stored per run).
    ram_values: list[float] = []
    for p in prompts:
        runs: list[dict[str, Any]] = p.get("runs", [])
        if runs:
            ram_values.append(runs[-1].get("ram_gb", 0.0))

    return {
        "avg_ttft_ms": sum(ttft_values) / len(ttft_values),
        "avg_tokens_per_sec": sum(tps_values) / len(tps_values),
        "avg_quality_score": sum(quality_values) / len(quality_values) if quality_values else 0.0,
        "avg_ram_gb": sum(ram_values) / len(ram_values) if ram_values else 0.0,
    }


class Reporter:
    """Handles all output for a benchmark run.

    Args:
        output_dir: Directory where result files will be written.
            Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    def print_table(self, results: list[dict[str, Any]]) -> None:
        """Print a rich table summarising benchmark results.

        Columns: Model | Quant | TTFT (ms) | Tokens/sec | Quality Score | RAM (GB)

        One row per model.  Numeric columns show averages across all prompts.

        Args:
            results: List of model result dicts.
        """
        table = Table(
            title="[bold]Quantization Benchmark Results[/bold]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Quant", style="yellow")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("Tokens/sec", justify="right")
        table.add_column("Quality Score", justify="right")
        table.add_column("RAM (GB)", justify="right")

        for model in results:
            avgs = _model_averages(model)
            table.add_row(
                model.get("name", "—"),
                model.get("quant", "—"),
                f"{avgs['avg_ttft_ms']:.1f}",
                f"{avgs['avg_tokens_per_sec']:.2f}",
                f"{avgs['avg_quality_score']:.3f}",
                f"{avgs['avg_ram_gb']:.2f}",
            )

        _console.print(table)

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def save_json(self, run_id: str, results: list[dict[str, Any]]) -> None:
        """Write the full benchmark results to ``results/results.json``.

        Args:
            run_id: ISO-8601 timestamp string identifying this run.
            results: List of model result dicts.
        """
        output = {"run_id": run_id, "models": results}
        path = os.path.join(self.output_dir, "results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        _console.print(f"[green]JSON saved →[/green] {path}")

    def save_markdown(self, results: list[dict[str, Any]]) -> None:
        """Write a Markdown summary report to ``results/report.md``.

        The report contains a table of averaged metrics per model/quant and a
        "Key Finding" line identifying the best speed/quality tradeoff.

        Args:
            results: List of model result dicts.
        """
        lines: list[str] = []
        lines.append("# Quantization Benchmark Report\n")
        lines.append(
            "| Model | Quant | Avg TTFT (ms) | Avg Tokens/sec | Avg Quality |"
        )
        lines.append(
            "|-------|-------|---------------|----------------|-------------|"
        )

        best_model: str = ""
        best_quant: str = ""
        best_tps: float = -1.0

        for model in results:
            avgs = _model_averages(model)
            name = model.get("name", "—")
            quant = model.get("quant", "—")
            lines.append(
                f"| {name} | {quant} | {avgs['avg_ttft_ms']:.1f} "
                f"| {avgs['avg_tokens_per_sec']:.2f} "
                f"| {avgs['avg_quality_score']:.3f} |"
            )

            # Track best tokens/sec among models with quality >= 0.9.
            if (
                avgs["avg_quality_score"] >= 0.9
                and avgs["avg_tokens_per_sec"] > best_tps
            ):
                best_tps = avgs["avg_tokens_per_sec"]
                best_model = name
                best_quant = quant

        lines.append("")
        if best_model:
            lines.append(
                f"**Key Finding:** `{best_model}` (quant: `{best_quant}`) delivers "
                f"the best speed/quality ratio with {best_tps:.2f} tokens/sec at "
                "quality ≥ 0.9."
            )
        else:
            lines.append(
                "**Key Finding:** No model achieved a quality score ≥ 0.9 in this run."
            )

        path = os.path.join(self.output_dir, "report.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        _console.print(f"[green]Markdown saved →[/green] {path}")

    def save_chart(self, results: list[dict[str, Any]]) -> None:
        """Save a scatter plot of tokens/sec vs quality score to ``results/chart.png``.

        Each point represents one model/quant combination, labelled with the
        model name and quant.

        Args:
            results: List of model result dicts.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in results:
            avgs = _model_averages(model)
            x = avgs["avg_tokens_per_sec"]
            y = avgs["avg_quality_score"]
            label = f"{model.get('name', '?')} ({model.get('quant', '?')})"

            ax.scatter(x, y, s=80, zorder=5)
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

        ax.set_xlabel("Tokens / sec", fontsize=12)
        ax.set_ylabel("Quality Score (cosine similarity)", fontsize=12)
        ax.set_title("Quantization Speed vs Quality Tradeoff", fontsize=14)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "chart.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _console.print(f"[green]Chart saved →[/green] {path}")

    def save_all(self, run_id: str, results: list[dict[str, Any]]) -> None:
        """Save JSON, Markdown, and chart outputs.

        Args:
            run_id: ISO-8601 timestamp string identifying this run.
            results: List of model result dicts.
        """
        self.save_json(run_id, results)
        self.save_markdown(results)
        self.save_chart(results)
