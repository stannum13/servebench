"""Evidence-gated Markdown report and figure generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.runner import compare_policies  # noqa: E402
from servebench.stats import bootstrap_ci  # noqa: E402


def _normalize(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    queue = row.get("queue_time_ms")
    if "queue_p95_ms" not in normalized and isinstance(queue, dict):
        normalized["queue_p95_ms"] = queue.get("p95")
    aliases = {
        "gpu_p95": "gpu_utilization_peak",
        "kv_p95": "kv_cache_peak",
        "power_watts": "gpu_power_average_watts",
    }
    for target, source in aliases.items():
        if target not in normalized:
            normalized[target] = row.get(source)
    return normalized


def _plot(frame: pd.DataFrame, x: str, ys: list[str], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for column in ys:
        axis.plot(frame[x], frame[column], marker="o", label=column.replace("_", " "))
    axis.set_title(title)
    axis.set_xlabel(x.replace("_", " "))
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_report(rows: list[dict[str, object]], report: Path, figures: Path) -> None:
    if not rows:
        raise ValueError("report requires measurements")
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(_normalize(row) for row in rows)
    fifo = (
        frame[frame.policy == "fifo"]
        .groupby("concurrency", as_index=False)
        .mean(numeric_only=True)
    )
    _plot(
        fifo, "concurrency", ["p95_ttft_ms", "requests_per_second"],
        figures / "saturation.png", "Saturation curve",
    )
    _plot(
        fifo, "concurrency", ["p95_ttft_ms", "queue_p95_ms"],
        figures / "latency-decomposition.png", "Latency decomposition",
    )
    policies = frame.groupby("policy", as_index=False).mean(numeric_only=True)
    _plot(policies, "policy", ["p95_ttft_ms"], figures / "scheduler-comparison.png", "FIFO vs SLO")
    table_lines = [
        "| policy | p95 TTFT ms (95% CI) | requests/s (95% CI) | power (W) |",
        "|---|---:|---:|---:|",
    ]
    for policy_name, group in frame.groupby("policy"):
        ttft = bootstrap_ci(group["p95_ttft_ms"].astype(float).tolist(), seed=7)
        throughput = bootstrap_ci(group["requests_per_second"].astype(float).tolist(), seed=8)
        power = group["power_watts"].astype(float).mean()
        table_lines.append(
            f"| {policy_name} | {ttft.estimate:.2f} [{ttft.low:.2f}, {ttft.high:.2f}] | "
            f"{throughput.estimate:.2f} [{throughput.low:.2f}, {throughput.high:.2f}] | "
            f"{power:.2f} |"
        )
    table = "\n".join(table_lines)
    repeats = int(frame.groupby("policy")["repeat"].nunique().min())
    verdict = "Insufficient repeated measurements: no optimization claim is made."
    policy_names = set(frame["policy"])
    if repeats >= 3 and {"fifo", "slo"} <= policy_names:
        common = set(frame[frame.policy == "fifo"].concurrency) & set(
            frame[frame.policy == "slo"].concurrency
        )
        if common:
            concurrency = max(common)
            fifo_rows = frame[(frame.policy == "fifo") & (frame.concurrency == concurrency)]
            slo_rows = frame[(frame.policy == "slo") & (frame.concurrency == concurrency)]
            fifo_rows = fifo_rows.sort_values("repeat")
            slo_rows = slo_rows.sort_values("repeat")
            if len(fifo_rows) == len(slo_rows) and len(fifo_rows) >= 3:
                comparison = compare_policies(
                    fifo_rows.p95_ttft_ms.tolist(), slo_rows.p95_ttft_ms.tolist(),
                    fifo_rows.requests_per_second.tolist(),
                    slo_rows.requests_per_second.tolist(),
                )
                verdict = f"Based on repeated measurements: {comparison.reason}."
    content = f"""# Servebench Report

> Generated from machine-readable measurements. Smoke/mock results are not GPU evidence.

## Latency / throughput / cost

{table}

{verdict}

## Key graphs

![Saturation curve](figures/saturation.png)

![Latency decomposition](figures/latency-decomposition.png)

![Scheduler comparison](figures/scheduler-comparison.png)

Queue growth, KV-cache pressure, and GPU utilization must be interpreted together
to locate saturation.
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, default=Path("REPORT.md"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    generate_report(rows, args.report, args.figures)
