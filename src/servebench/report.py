"""Evidence-gated Markdown report and figure generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


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
    frame = pd.DataFrame(rows)
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
    repeats = int(frame.groupby("policy")["repeat"].nunique().min())
    verdict = (
        "The comparison has repeated measurements; consult confidence intervals "
        "before claiming an improvement."
        if repeats >= 3
        else "Insufficient repeated measurements: no optimization claim is made."
    )
    table_lines = [
        "| policy | p95 TTFT (ms) | requests/s | power (W) |",
        "|---|---:|---:|---:|",
    ]
    for row in policies.itertuples():
        table_lines.append(
            f"| {row.policy} | {row.p95_ttft_ms:.2f} | "
            f"{row.requests_per_second:.2f} | {row.power_watts:.2f} |"
        )
    table = "\n".join(table_lines)
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
