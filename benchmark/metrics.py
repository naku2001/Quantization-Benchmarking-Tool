"""Pure metric calculation functions — no external dependencies beyond stdlib and psutil."""

import psutil


def calculate_ttft(start_time: float, first_chunk_time: float) -> float:
    """Return Time To First Token in milliseconds.

    Args:
        start_time: perf_counter value recorded just before the request was sent.
        first_chunk_time: perf_counter value recorded when the first non-empty
            chunk arrived from the streaming response.

    Returns:
        TTFT in milliseconds as a float.
    """
    return (first_chunk_time - start_time) * 1000.0


def calculate_throughput(eval_count: int, eval_duration_ns: int) -> float:
    """Return tokens per second using Ollama's native counters.

    Args:
        eval_count: Number of tokens evaluated, taken from Ollama's final
            done-chunk (``eval_count`` field).
        eval_duration_ns: Evaluation duration in nanoseconds, taken from
            Ollama's final done-chunk (``eval_duration`` field).

    Returns:
        Throughput in tokens/sec.  Returns 0.0 if eval_duration_ns is zero
        to avoid division by zero.
    """
    if eval_duration_ns == 0:
        return 0.0
    return eval_count / (eval_duration_ns / 1_000_000_000)


def get_ram_usage_gb() -> float:
    """Return the current process RSS memory usage in gigabytes.

    Uses :mod:`psutil` to query the running process's resident set size so
    that callers get a snapshot of actual physical RAM consumed at the moment
    of the call.

    Returns:
        RAM usage in GB as a float.
    """
    process = psutil.Process()
    rss_bytes = process.memory_info().rss
    return rss_bytes / (1024 ** 3)
