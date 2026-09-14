import json
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.app import create_app


def test_results_site_exposes_runs_status_and_dashboard(tmp_path: Path) -> None:
    results = tmp_path / "results" / "baseline"
    results.mkdir(parents=True)
    (results / "runs.json").write_text(json.dumps([{"run_id": "r1", "policy": "fifo"}]))
    report = tmp_path / "REPORT.md"
    report.write_text("GPU measurements are pending. No optimization claim is made.")
    figures = tmp_path / "figures"
    figures.mkdir()
    client = TestClient(create_app(tmp_path / "results", report, figures))
    assert client.get("/").status_code == 200
    assert "servebench" in client.get("/").text.lower()
    assert client.get("/api/runs").json() == [{"run_id": "r1", "policy": "fifo"}]
    assert client.get("/api/status").json()["gpu_evidence"] is False
