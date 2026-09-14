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
            })
        result.append({
            "policy": "slo", "repeat": repeat, "concurrency": 8,
            "p95_ttft_ms": 600, "requests_per_second": 15.5,
            "queue_p95_ms": 80, "gpu_p95": 77, "kv_p95": 0.7, "power_watts": 245,
        })
    return result


def test_report_generates_three_graphs_and_evidence_tables(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    generate_report(rows(), report, tmp_path / "figures")
    names = ["saturation.png", "latency-decomposition.png", "scheduler-comparison.png"]
    assert all((tmp_path / "figures" / name).exists() for name in names)
    text = report.read_text()
    assert "Latency / throughput / cost" in text
    assert "repeated measurements" in text


def test_report_refuses_optimization_claim_with_too_few_repeats(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    generate_report(rows(2), report, tmp_path / "figures")
    assert "insufficient repeated measurements" in report.read_text().lower()
