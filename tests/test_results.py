from pathlib import Path

from servebench.results import JsonlWriter, RequestMeasurement, read_jsonl, write_parquet


def measurement() -> RequestMeasurement:
    return RequestMeasurement(
        run_id="run-1", request_id="r1", workload="short", policy="fifo",
        prompt_tokens=256, output_tokens=3, scheduled_at=1.0, started_at=1.1,
        first_token_at=1.4, token_timestamps=[1.4, 1.5, 1.7], completed_at=1.8,
        status="ok", queue_time_ms=100.0, token_timing_exact=True,
    )


def test_measurement_derives_streaming_latencies() -> None:
    item = measurement()
    assert item.ttft_ms == 300.0
    assert item.inter_token_latencies_ms == [100.0, 200.0]
    assert item.e2e_latency_ms == 700.0


def test_jsonl_round_trip_and_parquet(tmp_path: Path) -> None:
    jsonl = tmp_path / "requests.jsonl"
    JsonlWriter(jsonl).write(measurement())
    records = read_jsonl(jsonl)
    assert records[0].request_id == "r1"
    parquet = tmp_path / "requests.parquet"
    write_parquet(jsonl, parquet)
    assert parquet.exists() and parquet.stat().st_size > 0
