"""Evidence-gated Markdown report and figure generation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.runner import compare_policies  # noqa: E402
from servebench.stats import bootstrap_ci  # noqa: E402


def _normalize(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized.setdefault("suite", "legacy")
    requests = int(row.get("requests", 0))
    if requests:
        normalized["completion_rate"] = float(row.get("successful", 0)) / requests
        normalized["rejection_rate"] = float(row.get("rejections", 0)) / requests
    else:
        normalized.setdefault("completion_rate", 1.0)
        normalized.setdefault("rejection_rate", 0.0)
    queue = row.get("queue_time_ms")
    if "queue_p95_ms" not in normalized and isinstance(queue, dict):
        normalized["queue_p95_ms"] = queue.get("p95")
    for field in ("client_queue_time_ms", "router_admission_ms", "post_header_ttft_ms"):
        value = row.get(field)
        if isinstance(value, dict):
            normalized[f"{field}_p95"] = value.get("p95")
    aliases = {
        "gpu_p95": "gpu_utilization_peak",
        "kv_p95": "kv_cache_peak",
        "power_watts": "gpu_power_average_watts",
        "gpu_utilization_peak": "gpu_p95",
        "kv_cache_peak": "kv_p95",
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


def generate_report(
    rows: list[dict[str, object]],
    report: Path,
    figures: Path,
    gpu_hourly_cost_usd: float | None = None,
) -> None:
    if not rows:
        raise ValueError("report requires measurements")
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(_normalize(row) for row in rows)
    suites = set(frame["suite"].astype(str))
    baseline_suite = next(
        (name for name in ("baseline-saturation", "mock-saturation") if name in suites),
        None,
    )
    if baseline_suite is None:
        eligible = frame[
            (~frame["suite"].astype(str).str.startswith("engine-"))
            & (frame["policy"] == "fifo")
        ]
        if eligible.empty:
            raise ValueError("report requires a FIFO baseline suite")
        baseline_suite = str(
            eligible.groupby("suite")["concurrency"].nunique().sort_values().index[-1]
        )
    fifo = (
        frame[(frame.suite == baseline_suite) & (frame.policy == "fifo")]
        .groupby("concurrency", as_index=False)
        .mean(numeric_only=True)
    )
    _plot(
        fifo, "concurrency", ["p95_ttft_ms", "requests_per_second"],
        figures / "saturation.png", "Saturation curve",
    )
    stages_available = all(
        field in fifo and fifo[field].notna().all()
        for field in ("router_admission_ms_p95", "post_header_ttft_ms_p95")
    )
    stage_columns = (
        ["p95_ttft_ms", "router_admission_ms_p95", "post_header_ttft_ms_p95"]
        if stages_available else ["p95_ttft_ms", "queue_p95_ms"]
    )
    _plot(
        fifo, "concurrency", stage_columns,
        figures / "latency-decomposition.png",
        "Observed TTFT stages" if stages_available else "Client queue and TTFT (legacy)",
    )
    policy_sets = {
        str(suite): set(group["policy"].astype(str))
        for suite, group in frame.groupby("suite")
    }
    scheduler_candidates = [
        suite for suite, policies_in_suite in policy_sets.items()
        if {"fifo", "slo"} <= policies_in_suite
    ]
    scheduler_suite = next(
        (name for name in ("fifo-vs-slo", "mock-scheduler") if name in scheduler_candidates),
        scheduler_candidates[0] if len(scheduler_candidates) == 1 else None,
    )
    comparison_source = (
        frame[frame.suite == scheduler_suite] if scheduler_suite is not None else frame
    )
    comparison_frame = comparison_source
    policy_names = set(comparison_source["policy"])
    if {"fifo", "slo"} <= policy_names:
        fifo_levels = set(
            comparison_source[comparison_source.policy == "fifo"].concurrency
        )
        slo_levels = set(
            comparison_source[comparison_source.policy == "slo"].concurrency
        )
        common_levels = fifo_levels & slo_levels
        if common_levels:
            comparison_frame = comparison_source[
                comparison_source.concurrency == max(common_levels)
            ]
    policies = comparison_frame.groupby("policy", as_index=False).mean(numeric_only=True)
    _plot(policies, "policy", ["p95_ttft_ms"], figures / "scheduler-comparison.png", "FIFO vs SLO")
    table_lines = [
        "| policy | p95 TTFT ms (95% CI) | requests/s (95% CI) | "
        "completion % | rejection % | power (W) | $ / 1M output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name, group in comparison_frame.groupby("policy"):
        ttft_values = group["p95_ttft_ms"].dropna().astype(float).tolist()
        ttft = bootstrap_ci(ttft_values, seed=7) if ttft_values else None
        throughput = bootstrap_ci(group["requests_per_second"].astype(float).tolist(), seed=8)
        power = group["power_watts"].astype(float).mean()
        power_text = "unavailable" if math.isnan(power) else f"{power:.2f}"
        output_rate = group.get("output_tokens_per_second")
        cost_text = "not configured"
        if gpu_hourly_cost_usd is not None and output_rate is not None:
            tokens_per_second = output_rate.astype(float).mean()
            if tokens_per_second > 0:
                cost = gpu_hourly_cost_usd * 1_000_000 / (tokens_per_second * 3600)
                cost_text = f"{cost:.4f}"
        table_lines.append(
            f"| {policy_name} | "
            + (
                f"{ttft.estimate:.2f} [{ttft.low:.2f}, {ttft.high:.2f}]"
                if ttft is not None else "unavailable"
            )
            + " | "
            f"{throughput.estimate:.2f} [{throughput.low:.2f}, {throughput.high:.2f}] | "
            f"{group['completion_rate'].astype(float).mean() * 100:.2f} | "
            f"{group['rejection_rate'].astype(float).mean() * 100:.2f} | "
            f"{power_text} | {cost_text} |"
        )
    table = "\n".join(table_lines)
    repeats = int(comparison_frame.groupby("policy")["repeat"].nunique().min())
    verdict = "Insufficient repeated measurements: no optimization claim is made."
    missing_first_tokens = comparison_frame["p95_ttft_ms"].isna().any()
    required_gpu_fields = (
        "gpu_utilization_peak", "gpu_memory_peak_mib", "kv_cache_peak",
        "vllm_queue_mean_ms", "vllm_prefill_mean_ms",
    )
    required_provenance_fields = (
        "model", "cache_isolation", "cache_state_initial",
    )
    gpu_evidence = (
        "evidence_kind" in comparison_frame
        and set(comparison_frame["evidence_kind"]) == {"gpu"}
        and all(
            field in comparison_frame and comparison_frame[field].notna().all()
            for field in required_gpu_fields
        )
        and all(
            field in comparison_frame
            and comparison_frame[field].notna().all()
            and not comparison_frame[field].astype(str).str.lower().eq("unknown").any()
            for field in required_provenance_fields
        )
    )
    paired_provenance = True
    if {"fifo", "slo"} <= policy_names:
        pairing_fields = (
            "repeat", "seed", "model", "workload", "cache_isolation",
            "cache_state_initial",
        )
        fifo_pairs = comparison_frame[comparison_frame.policy == "fifo"].sort_values("repeat")
        slo_pairs = comparison_frame[comparison_frame.policy == "slo"].sort_values("repeat")
        paired_provenance = len(fifo_pairs) == len(slo_pairs) and all(
            field in comparison_frame
            and fifo_pairs[field].astype(str).tolist() == slo_pairs[field].astype(str).tolist()
            for field in pairing_fields
        )
    if repeats >= 3 and not paired_provenance:
        verdict = "Mismatched paired provenance: no optimization claim is made."
    elif repeats >= 3 and missing_first_tokens:
        verdict = "One or more policy runs had no first token: no optimization claim is made."
    elif repeats >= 3 and not gpu_evidence:
        is_mock = (
            "evidence_kind" in comparison_frame
            and set(comparison_frame["evidence_kind"]) == {"mock"}
        )
        if is_mock:
            verdict = (
                "Mock evidence only: repeated measurements exist, "
                "but no optimization claim is made."
            )
        else:
            verdict = "GPU provenance or telemetry unverified: no optimization claim is made."
    if (
        repeats >= 3 and paired_provenance and not missing_first_tokens and gpu_evidence
        and {"fifo", "slo"} <= policy_names
    ):
        common = set(comparison_frame[comparison_frame.policy == "fifo"].concurrency) & set(
            comparison_frame[comparison_frame.policy == "slo"].concurrency
        )
        if common:
            concurrency = max(common)
            fifo_rows = comparison_source[
                (comparison_source.policy == "fifo")
                & (comparison_source.concurrency == concurrency)
            ]
            slo_rows = comparison_source[
                (comparison_source.policy == "slo")
                & (comparison_source.concurrency == concurrency)
            ]
            fifo_rows = fifo_rows.sort_values("repeat")
            slo_rows = slo_rows.sort_values("repeat")
            if len(fifo_rows) == len(slo_rows) and len(fifo_rows) >= 3:
                comparison = compare_policies(
                    fifo_rows.p95_ttft_ms.tolist(), slo_rows.p95_ttft_ms.tolist(),
                    fifo_rows.requests_per_second.tolist(),
                    slo_rows.requests_per_second.tolist(),
                    fifo_completion=fifo_rows.completion_rate.tolist(),
                    slo_completion=slo_rows.completion_rate.tolist(),
                )
                verdict = f"Based on repeated measurements: {comparison.reason}."
    if stages_available:
        queue_mean = frame.get("vllm_queue_mean_ms")
        prefill_mean = frame.get("vllm_prefill_mean_ms")
        queue_text = (
            f"{queue_mean.astype(float).mean():.2f} ms"
            if queue_mean is not None and queue_mean.notna().any() else "unavailable"
        )
        prefill_text = (
            f"{prefill_mean.astype(float).mean():.2f} ms"
            if prefill_mean is not None and prefill_mean.notna().any() else "unavailable"
        )
        stage_text = (
            "Client queue is loadgen semaphore wait and is outside TTFT. Router admission is "
            "measured inside the router; post-header TTFT includes backend queue, prefill, "
            "and transport. Run-scoped vLLM queue and prefill means are "
            f"{queue_text} and {prefill_text}, respectively. "
            "These p95 stage percentiles are not additive."
        )
    else:
        stage_text = (
            "Stage measurements unavailable in these legacy runs. The queue series in the "
            "figure is client queue (loadgen semaphore wait), not router or vLLM queue."
        )
    content = f"""# Servebench Report

> Generated from machine-readable measurements. Smoke/mock results are not GPU evidence.

## Latency / throughput / cost

{table}

{verdict}

ITL is measured between non-empty streamed output events; it is stream-event ITL,
not a fabricated per-token gap when one event contains multiple token IDs. TPOT is
the elapsed time after the first token divided by the remaining output-token count.

## Key graphs

![Saturation curve](figures/saturation.png)

![Latency decomposition](figures/latency-decomposition.png)

{stage_text}

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
    parser.add_argument(
        "--gpu-hourly-cost-usd",
        type=float,
        default=float(os.getenv("GPU_HOURLY_COST_USD", "nan")),
    )
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    hourly_cost = None if math.isnan(args.gpu_hourly_cost_usd) else args.gpu_hourly_cost_usd
    generate_report(rows, args.report, args.figures, hourly_cost)
