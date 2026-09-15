"""Confidence-aware experiment comparison CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import yaml

from servebench.stats import ConfidenceInterval, bootstrap_ci

from .state import append_state
from .telemetry import PrometheusTelemetry


@dataclass(frozen=True, slots=True)
class PolicyComparison:
    keep: bool
    ttft_delta_ms: ConfidenceInterval
    throughput_ratio: ConfidenceInterval
    reason: str


@dataclass(frozen=True, slots=True)
class RunSpec:
    suite: str
    workload: str
    requests: int
    concurrency: int
    repeat: int
    seed: int
    policy: str

    @property
    def run_id(self) -> str:
        return f"{self.policy}-c{self.concurrency:04d}-r{self.repeat:02d}"


@dataclass(frozen=True, slots=True)
class SaturationTransition:
    saturation_concurrency: int | None
    refinement_concurrency: int | None


def validate_repeats(repeats: int) -> None:
    if repeats < 3:
        raise ValueError("optimization comparisons require at least three repeats")


def compare_policies(
    fifo_ttft: list[float],
    slo_ttft: list[float],
    fifo_throughput: list[float],
    slo_throughput: list[float],
    throughput_floor: float = 0.95,
) -> PolicyComparison:
    lengths = {len(fifo_ttft), len(slo_ttft), len(fifo_throughput), len(slo_throughput)}
    if len(lengths) != 1:
        raise ValueError("policy measurements must be paired")
    repeats = lengths.pop()
    validate_repeats(repeats)
    ttft_delta = bootstrap_ci([new - old for old, new in zip(fifo_ttft, slo_ttft, strict=True)])
    ratios = [new / old for old, new in zip(fifo_throughput, slo_throughput, strict=True)]
    throughput_ratio = bootstrap_ci(ratios, seed=1)
    keep = ttft_delta.high < 0 and throughput_ratio.low >= throughput_floor
    reason = (
        "TTFT improved within throughput constraint"
        if keep
        else "confidence or throughput constraint failed"
    )
    return PolicyComparison(keep, ttft_delta, throughput_ratio, reason)


def build_run_specs(config: dict[str, object]) -> list[RunSpec]:
    repeats = int(config.get("repeats", 0))
    validate_repeats(repeats)
    concurrency = config.get("concurrency")
    if not isinstance(concurrency, dict) or not isinstance(concurrency.get("coarse"), list):
        raise ValueError("config requires concurrency.coarse")
    raw_policies = config.get("policies", [config.get("policy", "fifo")])
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("policies must be a non-empty list")
    return [
        RunSpec(
            suite=str(config["name"]),
            workload=str(config.get("workload", "mixed")),
            requests=int(config.get("requests", 100)),
            concurrency=int(level),
            repeat=repeat,
            seed=int(config.get("seed", 0)) + repeat,
            policy=str(policy),
        )
        for level in concurrency["coarse"]
        for policy in raw_policies
        for repeat in range(repeats)
    ]


def build_refinement_specs(
    config: dict[str, object], rows: list[dict[str, object]]
) -> list[RunSpec]:
    concurrency = config.get("concurrency")
    if not isinstance(concurrency, dict) or not concurrency.get("refine_transition", False):
        return []
    transition = detect_saturation(rows)  # type: ignore[arg-type]
    if transition.refinement_concurrency is None:
        return []
    refined = {**config, "concurrency": {"coarse": [transition.refinement_concurrency]}}
    return build_run_specs(refined)


def loadgen_command(
    spec: RunSpec, config: dict[str, object], url: str, output: Path
) -> list[str]:
    command = [
        "uv", "run", "servebench-loadgen", "--url", url,
        "--workload", spec.workload, "--requests", str(spec.requests),
        "--concurrency", str(spec.concurrency), "--seed", str(spec.seed),
        "--policy", spec.policy, "--output", str(output),
    ]
    if model := config.get("model"):
        command.extend(["--model", str(model)])
    if tokenizer_url := config.get("tokenizer_url", os.getenv("TOKENIZER_URL")):
        command.extend(["--tokenizer-url", str(tokenizer_url)])
    if request_rate := config.get("request_rate"):
        command.extend(["--request-rate", str(request_rate)])
    return command


def classify_evidence(requested: str, summary: dict[str, object]) -> str:
    if requested == "mock":
        return "mock"
    required = (
        "kv_cache_peak", "gpu_utilization_peak", "gpu_memory_peak_mib",
        "vllm_queue_mean_ms", "vllm_prefill_mean_ms",
    )
    return "gpu" if all(summary.get(name) is not None for name in required) else "unknown"


def detect_saturation(rows: list[dict[str, float]]) -> SaturationTransition:
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["concurrency"]), []).append(row)
    levels = sorted(grouped)
    if len(levels) < 2:
        raise ValueError("saturation detection requires two concurrency levels")
    medians = {}
    for level, values in grouped.items():
        ordered_rps = sorted(item["requests_per_second"] for item in values)
        ordered_ttft = sorted(
            item["p95_ttft_ms"] for item in values
            if item["p95_ttft_ms"] is not None
        )
        ttft = ordered_ttft[len(ordered_ttft) // 2] if ordered_ttft else None
        medians[level] = (ordered_rps[len(ordered_rps) // 2], ttft)
    saturation: int | None = None
    for previous, current in pairwise(levels):
        previous_rps, previous_ttft = medians[previous]
        current_rps, current_ttft = medians[current]
        gain = (current_rps - previous_rps) / max(previous_rps, 1e-9)
        if (
            current_ttft is not None and previous_ttft is not None
            and gain < 0.10 and current_ttft > previous_ttft * 1.5
        ):
            saturation = current
            break
    if saturation is None:
        return SaturationTransition(None, None)
    index = levels.index(saturation)
    refinement = (levels[index - 1] + saturation) // 2 if index > 0 else None
    if refinement in levels:
        refinement = None
    return SaturationTransition(saturation, refinement)


def execute_suite(
    config_path: Path,
    url: str,
    results_root: Path,
    prometheus_url: str | None = None,
    evidence_kind: str = "gpu",
) -> list[dict[str, object]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = build_run_specs(config)
    rows: list[dict[str, object]] = []
    telemetry = PrometheusTelemetry(prometheus_url) if prometheus_url else None
    def run(spec: RunSpec) -> dict[str, object]:
        output_dir = results_root / spec.suite / spec.run_id
        output = output_dir / "requests.jsonl"
        command = loadgen_command(spec, config, url, output)
        started_at = time.time()
        subprocess.run(command, check=True)
        completed_at = time.time()
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        if telemetry:
            summary.update(telemetry.collect(started_at, completed_at))
        summary.update({
            "run_id": spec.run_id, "policy": spec.policy, "repeat": spec.repeat,
            "seed": spec.seed, "concurrency": spec.concurrency,
            "p95_ttft_ms": summary["ttft_ms"]["p95"],
            "evidence_kind": classify_evidence(evidence_kind, summary),
            "evidence_requested": evidence_kind,
        })
        return summary

    try:
        for spec in specs:
            rows.append(run(spec))
        for spec in build_refinement_specs(config, rows):
            rows.append(run(spec))
    finally:
        if telemetry:
            telemetry.close()
    suite_dir = results_root / str(config["name"])
    (suite_dir / "runs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    refresh_report_input(results_root)

    levels = {int(row["concurrency"]) for row in rows}
    if len(levels) >= 2:
        transition = detect_saturation(rows)  # type: ignore[arg-type]
        (suite_dir / "transition.json").write_text(
            json.dumps(asdict(transition), indent=2) + "\n", encoding="utf-8"
        )
        append_state(
            Path("BENCH_STATE.md"),
            run_id=str(config["name"]),
            bottleneck=(
                f"transition near concurrency {transition.saturation_concurrency}"
                if transition.saturation_concurrency is not None
                else f"saturation not observed through concurrency {max(levels)}"
            ),
            hypothesis=(
                "queue, KV, and GPU telemetry at the transition identify "
                "the limiting resource"
            ),
            variable="concurrency",
            decision="inconclusive",
        )
    policies = {str(row["policy"]) for row in rows}
    if {"fifo", "slo"} <= policies:
        verified = "gpu" if all(row["evidence_kind"] == "gpu" for row in rows) else "unknown"
        compare_and_record_policies(config, rows, suite_dir, verified)
    return rows


def refresh_report_input(results_root: Path) -> None:
    report_rows: list[dict[str, object]] = []
    for path in results_root.glob("*/runs.json"):
        report_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    (results_root / "report-input.json").write_text(
        json.dumps(report_rows, indent=2) + "\n", encoding="utf-8"
    )


def compare_and_record_policies(
    config: dict[str, object],
    rows: list[dict[str, object]],
    suite_dir: Path,
    evidence_kind: str,
) -> None:
    common = sorted(
        {int(row["concurrency"]) for row in rows if row["policy"] == "fifo"}
        & {int(row["concurrency"]) for row in rows if row["policy"] == "slo"}
    )
    if not common:
        return
    level = common[-1]
    fifo = sorted(
        (row for row in rows if row["policy"] == "fifo" and row["concurrency"] == level),
        key=lambda row: int(row["repeat"]),
    )
    slo = sorted(
        (row for row in rows if row["policy"] == "slo" and row["concurrency"] == level),
        key=lambda row: int(row["repeat"]),
    )
    if any(row.get("p95_ttft_ms") is None for row in fifo + slo):
        (suite_dir / "comparison.json").write_text(
            json.dumps({
                "keep": False,
                "ttft_delta_ms": None,
                "throughput_ratio": None,
                "reason": "one or more policy runs had no successful first token",
            }, indent=2) + "\n", encoding="utf-8",
        )
        append_state(
            Path("BENCH_STATE.md"),
            run_id=str(config["name"]),
            bottleneck="one or more policies produced no successful first token",
            hypothesis="SLO admission reduces p95 TTFT with at least 95% FIFO throughput",
            variable="scheduling policy",
            decision="inconclusive",
        )
        return
    comparison = compare_policies(
        [float(row["p95_ttft_ms"]) for row in fifo],
        [float(row["p95_ttft_ms"]) for row in slo],
        [float(row["requests_per_second"]) for row in fifo],
        [float(row["requests_per_second"]) for row in slo],
    )
    (suite_dir / "comparison.json").write_text(
        json.dumps(comparison_as_dict(comparison), indent=2) + "\n", encoding="utf-8"
    )
    decision = "keep" if comparison.keep else "revert"
    if evidence_kind != "gpu":
        decision = "inconclusive"
    append_state(
        Path("BENCH_STATE.md"),
        run_id=str(config["name"]),
        bottleneck="mixed-workload p95 TTFT at saturation",
        hypothesis="SLO admission reduces p95 TTFT while preserving 95% of FIFO throughput",
        variable="scheduling policy",
        decision=decision,
        measurement=(
            "p95 TTFT delta: "
            f"{comparison.ttft_delta_ms.estimate:.2f} ms, 95% CI "
            f"[{comparison.ttft_delta_ms.low:.2f}, {comparison.ttft_delta_ms.high:.2f}] ms; "
            "throughput ratio: "
            f"{comparison.throughput_ratio.estimate:.3f}, 95% CI "
            f"[{comparison.throughput_ratio.low:.3f}, "
            f"{comparison.throughput_ratio.high:.3f}]"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_files", nargs="*", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--prometheus-url")
    parser.add_argument("--engine-sweep-at", type=int)
    parser.add_argument("--vllm-health-url", default="http://localhost:8000/health")
    parser.add_argument("--evidence-kind", choices=["gpu", "mock"], default="gpu")
    parser.add_argument("--output", type=Path, default=Path("results/experiment-manifest.json"))
    args = parser.parse_args()
    if args.engine_sweep_at:
        if not args.config or not args.prometheus_url:
            parser.error("--engine-sweep-at requires --config and --prometheus-url")
        from .engine import execute_engine_sweeps

        rows = execute_engine_sweeps(
            args.config,
            args.engine_sweep_at,
            args.url,
            args.prometheus_url,
            args.vllm_health_url,
            args.results_root,
        )
        print(json.dumps({"engine_runs": len(rows)}))
        return
    if args.config:
        rows = execute_suite(
            args.config, args.url, args.results_root, args.prometheus_url, args.evidence_kind
        )
        print(json.dumps({"runs": len(rows), "suite": args.config.stem}))
        return
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary_files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(summaries)}))


def comparison_as_dict(value: PolicyComparison) -> dict[str, object]:
    return asdict(value)
