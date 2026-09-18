import json
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.app import create_app


def test_results_site_exposes_runs_status_and_dashboard(tmp_path: Path) -> None:
    results = tmp_path / "results" / "baseline"
    results.mkdir(parents=True)
    (results / "runs.json").write_text(
        json.dumps([{"run_id": "r1", "policy": "fifo", "evidence_kind": "mock"}])
    )
    report = tmp_path / "REPORT.md"
    report.write_text("Generated benchmark report. No optimization claim is made.")
    figures = tmp_path / "figures"
    figures.mkdir()
    client = TestClient(create_app(tmp_path / "results", report, figures))
    assert client.get("/").status_code == 200
    assert "servebench" in client.get("/").text.lower()
    assert client.get("/api/runs").json() == [
        {"run_id": "r1", "policy": "fifo", "evidence_kind": "mock"}
    ]
    assert client.get("/api/status").json()["gpu_evidence"] is False


def test_status_rejects_gpu_label_without_complete_telemetry(tmp_path: Path) -> None:
    results = tmp_path / "results" / "baseline"
    results.mkdir(parents=True)
    (results / "runs.json").write_text(json.dumps([{
        "run_id": "r1", "evidence_kind": "gpu", "kv_cache_peak": 0.8,
    }]))
    client = TestClient(create_app(tmp_path / "results", tmp_path / "REPORT.md", tmp_path))
    assert client.get("/api/status").json()["gpu_evidence"] is False


def test_status_accepts_only_complete_gpu_telemetry(tmp_path: Path) -> None:
    results = tmp_path / "results" / "baseline"
    results.mkdir(parents=True)
    (results / "runs.json").write_text(json.dumps([{
        "run_id": "r1", "evidence_kind": "gpu", "kv_cache_peak": 0.8,
        "gpu_utilization_peak": 95, "gpu_memory_peak_mib": 22000,
        "vllm_queue_mean_ms": 12, "vllm_prefill_mean_ms": 30,
    }]))
    client = TestClient(create_app(tmp_path / "results", tmp_path / "REPORT.md", tmp_path))
    assert client.get("/api/status").json()["gpu_evidence"] is True


def test_status_rejects_non_numeric_or_non_finite_gpu_telemetry(tmp_path: Path) -> None:
    results = tmp_path / "results" / "baseline"
    results.mkdir(parents=True)
    record = {
        "run_id": "r1", "evidence_kind": "gpu", "kv_cache_peak": "0.8",
        "gpu_utilization_peak": 95, "gpu_memory_peak_mib": 22000,
        "vllm_queue_mean_ms": 12, "vllm_prefill_mean_ms": float("nan"),
    }
    (results / "runs.json").write_text(json.dumps([record]))
    client = TestClient(create_app(tmp_path / "results", tmp_path / "REPORT.md", tmp_path))
    assert client.get("/api/status").json()["gpu_evidence"] is False


def test_partial_result_file_does_not_take_down_public_api(tmp_path: Path) -> None:
    results = tmp_path / "results" / "active-run"
    results.mkdir(parents=True)
    (results / "runs.json").write_text('[{"run_id": "unfinished"')
    client = TestClient(create_app(tmp_path / "results", tmp_path / "REPORT.md", tmp_path))
    assert client.get("/api/runs").json() == []
    assert client.get("/api/status").json()["invalid_result_files"] == 1
