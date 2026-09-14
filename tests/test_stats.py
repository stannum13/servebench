from servebench.results import RequestMeasurement
from servebench.stats import bootstrap_ci, summarize_measurements


def item(index: int, *, status: str = "ok") -> RequestMeasurement:
    start = float(index)
    return RequestMeasurement(
        run_id="x", request_id=str(index), workload="short", policy="fifo",
        prompt_tokens=100, output_tokens=2, scheduled_at=start, started_at=start,
        first_token_at=start + 0.1, token_timestamps=[start + 0.1, start + 0.2],
        completed_at=start + 0.3, status=status,
    )


def test_summary_includes_latency_throughput_and_failures() -> None:
    summary = summarize_measurements([item(0), item(1), item(2, status="timeout")])
    assert summary["requests"] == 3
    assert summary["failures"] == 1
    assert summary["ttft_ms"]["p95"] == 100.0
    assert summary["output_tokens_per_second"] > 0


def test_bootstrap_confidence_interval_is_deterministic() -> None:
    first = bootstrap_ci([1.0, 2.0, 3.0, 4.0], seed=22, samples=200)
    assert first == bootstrap_ci([1.0, 2.0, 3.0, 4.0], seed=22, samples=200)
    assert first.low <= first.estimate <= first.high
