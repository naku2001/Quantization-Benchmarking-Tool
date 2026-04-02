"""Unit tests for benchmark/metrics.py."""

from unittest.mock import MagicMock, patch

import pytest

from benchmark.metrics import calculate_throughput, calculate_ttft, get_ram_usage_gb


class TestCalculateTtft:
    """Tests for calculate_ttft."""

    def test_basic_calculation(self) -> None:
        """TTFT should be (first_chunk - start) * 1000 ms."""
        result = calculate_ttft(start_time=0.0, first_chunk_time=0.521)
        assert result == pytest.approx(521.0, rel=1e-6)

    def test_zero_delta(self) -> None:
        """Zero delta should return 0.0 ms."""
        result = calculate_ttft(start_time=1.0, first_chunk_time=1.0)
        assert result == pytest.approx(0.0)

    def test_small_delta(self) -> None:
        """A 100 ms gap should return 100.0."""
        result = calculate_ttft(start_time=5.0, first_chunk_time=5.1)
        assert result == pytest.approx(100.0, rel=1e-6)


class TestCalculateThroughput:
    """Tests for calculate_throughput."""

    def test_basic_calculation(self) -> None:
        """100 tokens in 10 seconds → 10.0 tokens/sec."""
        result = calculate_throughput(eval_count=100, eval_duration_ns=10_000_000_000)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_zero_duration(self) -> None:
        """Zero duration should return 0.0 to avoid division by zero."""
        result = calculate_throughput(eval_count=50, eval_duration_ns=0)
        assert result == 0.0

    def test_high_throughput(self) -> None:
        """200 tokens in 1 second → 200.0 tokens/sec."""
        result = calculate_throughput(eval_count=200, eval_duration_ns=1_000_000_000)
        assert result == pytest.approx(200.0, rel=1e-6)

    def test_fractional_result(self) -> None:
        """7 tokens in 1 second → 7.0 tokens/sec."""
        result = calculate_throughput(eval_count=7, eval_duration_ns=1_000_000_000)
        assert result == pytest.approx(7.0, rel=1e-6)


class TestGetRamUsageGb:
    """Tests for get_ram_usage_gb."""

    def test_returns_positive_float(self) -> None:
        """RAM usage should always be a positive float."""
        result = get_ram_usage_gb()
        assert isinstance(result, float)
        assert result > 0.0

    def test_uses_psutil(self) -> None:
        """Should delegate to psutil.Process().memory_info().rss."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 2 * (1024 ** 3)  # 2 GB

        with patch("benchmark.metrics.psutil.Process", return_value=mock_process):
            result = get_ram_usage_gb()

        assert result == pytest.approx(2.0, rel=1e-6)
