"""Theme-neutral results API and dashboard shell."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>servebench</title></head><body>
<main><h1>servebench</h1><p id="status">Loading benchmark status…</p>
<h2>Runs</h2><pre id="runs"></pre>
<h2>Figures</h2><p><a href="/figures/saturation.png">Saturation</a> ·
<a href="/figures/latency-decomposition.png">Latency decomposition</a> ·
<a href="/figures/scheduler-comparison.png">Scheduler comparison</a></p>
<p><a href="/report">Read the report</a></p></main>
<script>Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/runs').then(r=>r.json())])
.then(([s,r])=>{document.querySelector('#status').textContent=s.gpu_evidence?
'GPU evidence available':'GPU evidence pending';document.querySelector('#runs').textContent=
JSON.stringify(r,null,2)});</script></body></html>"""

REQUIRED_GPU_FIELDS = (
    "kv_cache_peak",
    "gpu_utilization_peak",
    "gpu_memory_peak_mib",
    "vllm_queue_mean_ms",
    "vllm_prefill_mean_ms",
)


def _has_gpu_evidence(record: dict[str, object]) -> bool:
    values = [record.get(field) for field in REQUIRED_GPU_FIELDS]
    numeric = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        for value in values
    )
    return (
        record.get("evidence_kind") == "gpu"
        and isinstance(record.get("run_id"), str)
        and bool(record["run_id"])
        and numeric
        and float(record["kv_cache_peak"]) <= 1
        and float(record["gpu_utilization_peak"]) <= 100
    )


def _load_runs(results_dir: Path) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    invalid_files = 0
    for path in sorted(results_dir.glob("*/runs.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_files += 1
            continue
        if not isinstance(payload, list):
            invalid_files += 1
            continue
        records.extend(item for item in payload if isinstance(item, dict))
    return records, invalid_files


def create_app(results_dir: Path, report_path: Path, figures_dir: Path) -> FastAPI:
    app = FastAPI(title="servebench results", docs_url=None, redoc_url=None)
    figures_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/figures", StaticFiles(directory=figures_dir), name="figures")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX

    @app.get("/api/runs")
    async def runs() -> list[dict[str, object]]:
        records, _ = _load_runs(results_dir)
        return records

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        records, invalid_files = _load_runs(results_dir)
        gpu_evidence = any(_has_gpu_evidence(record) for record in records)
        return {
            "gpu_evidence": gpu_evidence,
            "runs": len(records),
            "invalid_result_files": invalid_files,
        }

    @app.get("/report", response_class=PlainTextResponse)
    async def report() -> str:
        if report_path.exists():
            return report_path.read_text(encoding="utf-8")
        return "Report pending."

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app(
    Path(os.getenv("RESULTS_DIR", "results")),
    Path(os.getenv("REPORT_PATH", "REPORT.md")),
    Path(os.getenv("FIGURES_DIR", "figures")),
)
