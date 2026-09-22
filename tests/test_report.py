from pathlib import Path

from servebench.report import generate_report


def rows(repeats: int = 3) -> list[dict[str, object]]:
    result = []
    for repeat in range(repeats):
        for concurrency in (1, 4, 8):
            result.append({
                "policy": "fifo", "repeat": repeat, "concurrency": concurrency,
                "p95_ttft_ms": concurrency * 100, "requests_per_second": concurrency * 2,
                "queue_p95_ms": concurrency * 20, "gpu_p95": 70 + concurrency,
                "kv_p95": 0.1 * concurrency, "power_watts": 250,
                "gpu_memory_peak_mib": 20000, "evidence_kind": "gpu",
                "vllm_queue_mean_ms": 20, "vllm_prefill_mean_ms": 80,
                "model": "open/model",
                "cache_isolation": "paired-alternating-shared-engine",
                "cache_state_initial": "warm-or-unknown",
            })
        result.append({
            "policy": "slo", "repeat": repeat, "concurrency": 8,
            "p95_ttft_ms": 600, "requests_per_second": 15.5,
            "queue_p95_ms": 80, "gpu_p95": 77, "kv_p95": 0.7, "power_watts": 245,
            "gpu_memory_peak_mib": 20000, "evidence_kind": "gpu",
            "vllm_queue_mean_ms": 15, "vllm_prefill_mean_ms": 70,
            "model": "open/model",
            "cache_isolation": "paired-alternating-shared-engine",
            "cache_state_initial": "warm-or-unknown",
        })
    return result


def test_report_generates_three_graphs_and_evidence_tables(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    generate_report(rows(), report, tmp_path / "figures")
    names = ["saturation.png", "latency-decomposition.png", "scheduler-comparison.png"]
    assert all((tmp_path / "figures" / name).exists() for name in names)
    text = report.read_text()
    assert "Latency / throughput / cost" in text
    assert "$ / 1M output tokens" in text
    assert "repeated measurements" in text
    assert "95% CI" in text
    assert "TTFT improved within throughput constraint" in text
    assert "completion %" in text
    assert "rejection %" in text
    assert "stream-event ITL" in text
    assert "TPOT" in text


def test_report_refuses_optimization_claim_with_too_few_repeats(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    generate_report(rows(2), report, tmp_path / "figures")
    assert "insufficient repeated measurements" in report.read_text().lower()


def test_report_accepts_native_runner_summary_schema(tmp_path: Path) -> None:
    native = [{
        "policy": "fifo", "repeat": repeat, "concurrency": 2,
        "p95_ttft_ms": 120, "requests_per_second": 4.0,
        "queue_time_ms": {"p95": 30}, "gpu_utilization_peak": 88,
        "kv_cache_peak": 0.7, "gpu_power_average_watts": 240,
    } for repeat in range(3)]
    report = tmp_path / "REPORT.md"
    generate_report(native, report, tmp_path / "figures")
    assert "240.00" in report.read_text()


def test_report_never_claims_mock_policy_improvement(tmp_path: Path) -> None:
    mock_rows = [{**row, "evidence_kind": "mock"} for row in rows()]
    report = tmp_path / "REPORT.md"
    generate_report(mock_rows, report, tmp_path / "figures")
    text = report.read_text().lower()
    assert "mock evidence only" in text
    assert "no optimization claim" in text


def test_report_never_claims_improvement_without_provenance(tmp_path: Path) -> None:
    unknown_rows = [{key: value for key, value in row.items() if key != "evidence_kind"}
                    for row in rows()]
    report = tmp_path / "REPORT.md"
    generate_report(unknown_rows, report, tmp_path / "figures")
    text = report.read_text().lower()
    assert "no optimization claim" in text


def test_report_never_claims_without_cache_provenance(tmp_path: Path) -> None:
    unknown_rows = [{key: value for key, value in row.items()
                     if key not in {"cache_isolation", "cache_state_initial"}}
                    for row in rows()]
    report = tmp_path / "REPORT.md"
    generate_report(unknown_rows, report, tmp_path / "figures")
    assert "no optimization claim" in report.read_text().lower()


def test_report_never_claims_without_vllm_queue_and_prefill_telemetry(tmp_path: Path) -> None:
    incomplete = [{key: value for key, value in row.items()
                   if key not in {"vllm_queue_mean_ms", "vllm_prefill_mean_ms"}}
                  for row in rows()]
    report = tmp_path / "REPORT.md"
    generate_report(incomplete, report, tmp_path / "figures")
    assert "no optimization claim" in report.read_text().lower()


def test_report_distinguishes_client_queue_from_server_ttft_stages(tmp_path: Path) -> None:
    stage_rows = [{
        **row,
        "client_queue_time_ms": {"p95": 200},
        "router_admission_ms": {"p95": 10},
        "post_header_ttft_ms": {"p95": 90},
        "vllm_queue_mean_ms": 25,
        "vllm_prefill_mean_ms": 50,
    } for row in rows()]
    report = tmp_path / "REPORT.md"
    generate_report(stage_rows, report, tmp_path / "figures")
    text = report.read_text().lower()
    assert "client queue" in text
    assert "router admission" in text
    assert "post-header" in text
    assert "vllm queue" in text
    assert "prefill" in text


def test_report_does_not_label_legacy_client_queue_as_server_queue(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    generate_report(rows(), report, tmp_path / "figures")
    text = report.read_text().lower()
    assert "stage measurements unavailable" in text


def test_report_handles_policy_with_only_rejections(tmp_path: Path) -> None:
    rejected = [
        {**row, "p95_ttft_ms": None, "requests_per_second": 0, "rejections": 100}
        if row["policy"] == "slo" else row
        for row in rows()
    ]
    report = tmp_path / "REPORT.md"
    generate_report(rejected, report, tmp_path / "figures")
    text = report.read_text().lower()
    assert "unavailable" in text
    assert "no optimization claim" in text
