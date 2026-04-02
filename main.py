"""quant-bench — CLI entry point.

Run ``python main.py --help`` for usage information.
"""

from __future__ import annotations

import sys
from datetime import datetime
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from benchmark.quality import QualityScorer
from benchmark.reporter import Reporter
from benchmark.runner import BenchmarkRunner, OllamaConnectionError

console = Console()


def _load_prompts(path: str) -> list[str]:
    """Read prompts from *path*, one per line.

    Lines that are blank or start with ``#`` are skipped.

    Resolution order:
    1. The path as given (absolute, or relative to the current working directory).
    2. Bundled package data under ``benchmark/prompts/`` — used when the tool
       is installed via pip and the prompts directory is not on disk.

    Args:
        path: Path to the prompt file.

    Returns:
        List of non-empty, non-comment prompt strings.

    Raises:
        SystemExit: If the file cannot be found by either method.
    """
    p = Path(path)
    content: str | None = None

    if p.exists():
        content = p.read_text(encoding="utf-8")
    else:
        # Fall back to bundled package data (works for pip-installed package).
        try:
            pkg_file = _pkg_files("benchmark.prompts").joinpath(p.name)
            content = pkg_file.read_text(encoding="utf-8")
        except Exception:
            pass

    if content is None:
        console.print(f"[bold red]Error:[/bold red] Prompt file not found: {path}")
        sys.exit(1)

    prompts: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            prompts.append(stripped)

    if not prompts:
        console.print(
            f"[bold red]Error:[/bold red] No prompts found in {path}. "
            "Check that the file has at least one non-blank, non-comment line."
        )
        sys.exit(1)

    return prompts


def _pick_baseline_quant(results: list[dict[str, Any]]) -> str:
    """Choose the best quant label to use as the quality baseline.

    Prefers Q8 (or variants like Q8_0).  Falls back to the quant whose label
    sorts last lexicographically (a rough heuristic for "highest quality").

    Args:
        results: List of model result dicts.

    Returns:
        The quant label string to use as baseline.
    """
    quants = [r.get("quant", "") for r in results]

    # Prefer any Q8 variant.
    for q in quants:
        if q.upper().startswith("Q8"):
            return q

    # Fall back to highest lexicographic quant (e.g. Q6 > Q5 > Q4).
    if quants:
        return max(quants)

    return "unknown"


@click.command()
@click.option(
    "--model",
    multiple=True,
    required=True,
    help="Ollama model name. Repeatable for multiple models.",
)
@click.option(
    "--runs",
    default=3,
    show_default=True,
    help="Runs per prompt to average. The first run is a warmup and is discarded.",
)
@click.option(
    "--prompts",
    default="prompts/factual.txt",
    show_default=True,
    help="Path to a prompt file (one prompt per line).",
)
@click.option(
    "--format",
    "output_format",
    default="all",
    show_default=True,
    help="Output format: table | json | chart | all",
)
@click.option(
    "--output",
    default="results",
    show_default=True,
    help="Directory for result files.",
)
@click.option(
    "--context-sweep",
    "context_sweep",
    is_flag=True,
    default=False,
    help=(
        "Run each prompt at multiple input context sizes (512, 2048, 4096 tokens) "
        "and produce a quality-vs-context-length chart."
    ),
)
def main(
    model: tuple[str, ...],
    runs: int,
    prompts: str,
    output_format: str,
    output: str,
    context_sweep: bool,
) -> None:
    """Benchmark Ollama models across quantization levels.

    Measures Time to First Token, throughput (tokens/sec), and semantic
    quality degradation relative to the highest available quantization.
    """
    # ------------------------------------------------------------------ #
    # 1. Set up runner and verify Ollama is reachable.                    #
    # ------------------------------------------------------------------ #
    runner = BenchmarkRunner()

    console.print("[bold cyan]quant-bench[/bold cyan] — LLM Quantization Benchmarking Tool")
    console.print(f"Connecting to Ollama at {runner.base_url} …")

    try:
        runner.check_connection()
    except OllamaConnectionError as exc:
        console.print(f"[bold red]Connection error:[/bold red] {exc}")
        sys.exit(1)

    console.print("[green]Ollama is reachable.[/green]\n")

    # ------------------------------------------------------------------ #
    # 2. Load prompts.                                                    #
    # ------------------------------------------------------------------ #
    prompt_list = _load_prompts(prompts)
    console.print(
        f"Loaded [bold]{len(prompt_list)}[/bold] prompt(s) from [cyan]{prompts}[/cyan]."
    )
    console.print(
        f"Runs per prompt: [bold]{runs}[/bold] "
        f"(run 1 is warmup; averaging runs 2–{runs}).\n"
    )

    scorer = QualityScorer()
    run_id = datetime.now().isoformat(timespec="seconds")
    reporter = Reporter(output_dir=output)

    # ------------------------------------------------------------------ #
    # 3a. Context-sweep mode.                                             #
    # ------------------------------------------------------------------ #
    if context_sweep:
        console.print(
            "[bold cyan]Context-sweep mode:[/bold cyan] running each prompt at "
            "512, 2048, and 4096 input tokens.\n"
        )
        all_sweep_results: list[dict[str, Any]] = []

        for model_name in model:
            console.rule(f"[bold yellow]Model: {model_name}[/bold yellow]")
            variants = runner.list_model_variants(model_name)
            console.print(
                f"Found [bold]{len(variants)}[/bold] variant(s): "
                + ", ".join(v["name"] for v in variants)
            )

            for variant in variants:
                console.print(
                    f"\nContext sweep: [cyan]{variant['name']}[/cyan] "
                    f"(quant: [yellow]{variant['quant']}[/yellow]) …"
                )
                with console.status("Running sweep across 3 context sizes…"):
                    sweep = runner.run_context_sweep(variant["name"], prompt_list, runs)
                all_sweep_results.extend(sweep)
                console.print(f"  [green]Done.[/green] ({len(sweep)} context sizes)")

        if not all_sweep_results:
            console.print("[bold red]No results collected. Exiting.[/bold red]")
            sys.exit(1)

        console.rule("[bold]Quality Scoring[/bold]")
        baseline_quant = _pick_baseline_quant(all_sweep_results)
        console.print(
            f"Baseline quant: [yellow]{baseline_quant}[/yellow]"
        )
        all_sweep_results = scorer.score_sweep_results(all_sweep_results, baseline_quant)
        console.print("[green]Quality scoring complete.[/green]\n")

        want_table = output_format in ("table", "all")
        want_json = output_format in ("json", "all")
        want_chart = output_format in ("chart", "all")

        if want_table:
            console.rule("[bold]Context Sweep Results[/bold]")
            reporter.print_context_sweep_table(all_sweep_results)

        if want_json or want_chart:
            console.rule("[bold]Saving Files[/bold]")

        if want_json:
            reporter.save_json(run_id, all_sweep_results)

        if want_chart:
            reporter.save_context_sweep_chart(all_sweep_results)

        if output_format == "table":
            reporter.save_json(run_id, all_sweep_results)

        console.print(
            f"\n[bold green]Context sweep complete.[/bold green] "
            f"Results written to [cyan]{output}/[/cyan]"
        )
        return

    # ------------------------------------------------------------------ #
    # 3b. Standard benchmark mode.                                        #
    # ------------------------------------------------------------------ #
    all_results: list[dict[str, Any]] = []

    for model_name in model:
        console.rule(f"[bold yellow]Model: {model_name}[/bold yellow]")

        variants = runner.list_model_variants(model_name)
        console.print(
            f"Found [bold]{len(variants)}[/bold] variant(s): "
            + ", ".join(v["name"] for v in variants)
        )

        for variant in variants:
            console.print(
                f"\nBenchmarking [cyan]{variant['name']}[/cyan] "
                f"(quant: [yellow]{variant['quant']}[/yellow]) …"
            )
            with console.status(f"Running {runs} × {len(prompt_list)} prompt(s)…"):
                result = runner.run_benchmark(variant["name"], prompt_list, runs)
            all_results.append(result)
            console.print(
                f"  [green]Done.[/green] Avg TTFT: "
                f"{sum(p['avg_ttft_ms'] for p in result['prompts']) / len(result['prompts']):.1f} ms"
            )

    if not all_results:
        console.print("[bold red]No results collected. Exiting.[/bold red]")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 4. Quality scoring.                                                 #
    # ------------------------------------------------------------------ #
    console.rule("[bold]Quality Scoring[/bold]")
    baseline_quant = _pick_baseline_quant(all_results)
    console.print(
        f"Baseline quant for quality scoring: [yellow]{baseline_quant}[/yellow]"
    )

    all_results = scorer.score_results(all_results, baseline_quant)
    console.print("[green]Quality scoring complete.[/green]\n")

    # ------------------------------------------------------------------ #
    # 5. Output.                                                          #
    # ------------------------------------------------------------------ #
    want_table = output_format in ("table", "all")
    want_json = output_format in ("json", "all")
    want_chart = output_format in ("chart", "all")

    if want_table:
        console.rule("[bold]Results Table[/bold]")
        reporter.print_table(all_results)

    if want_json or want_chart:
        console.rule("[bold]Saving Files[/bold]")

    if want_json:
        reporter.save_json(run_id, all_results)
        reporter.save_markdown(all_results)

    if want_chart:
        reporter.save_chart(all_results)

    # If only "table" was requested, still save JSON so results aren't lost.
    if output_format == "table":
        reporter.save_json(run_id, all_results)
        reporter.save_markdown(all_results)

    console.print(
        f"\n[bold green]Benchmark complete.[/bold green] "
        f"Results written to [cyan]{output}/[/cyan]"
    )


if __name__ == "__main__":
    main()
