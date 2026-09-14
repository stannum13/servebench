"""Confidence-aware experiment comparison CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import yaml

from servebench.stats import ConfidenceInterval, bootstrap_ci

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
    saturation_concurrency: int
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
        ordered_ttft = sorted(item["p95_ttft_ms"] for item in values)
        medians[level] = (ordered_rps[len(ordered_rps) // 2], ordered_ttft[len(ordered_ttft) // 2])
    saturation = levels[-1]
    for previous, current in pairwise(levels):
        previous_rps, previous_ttft = medians[previous]
        current_rps, current_ttft = medians[current]
        gain = (current_rps - previous_rps) / max(previous_rps, 1e-9)
        if gain < 0.10 and current_ttft > previous_ttft * 1.5:
            saturation = current
            break
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
) -> list[dict[str, object]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = build_run_specs(config)
    rows: list[dict[str, object]] = []
    telemetry = PrometheusTelemetry(prometheus_url) if prometheus_url else None
    for spec in specs:
        output_dir = results_root / spec.suite / spec.run_id
        output = output_dir / "requests.jsonl"
        command = [
            "uv", "run", "servebench-loadgen", "--url", url,
            "--workload", spec.workload, "--requests", str(spec.requests),
            "--concurrency", str(spec.concurrency), "--seed", str(spec.seed),
            "--policy", spec.policy, "--output", str(output),
        ]
        if request_rate := config.get("request_rate"):
            command.extend(["--request-rate", str(request_rate)])
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
        })
        rows.append(summary)
    if telemetry:
        telemetry.close()
    suite_dir = results_root / str(config["name"])
    (suite_dir / "runs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    report_rows: list[dict[str, object]] = []
    for path in results_root.glob("*/runs.json"):
        report_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    (results_root / "report-input.json").write_text(
        json.dumps(report_rows, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_files", nargs="*", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--prometheus-url")
    parser.add_argument("--output", type=Path, default=Path("results/experiment-manifest.json"))
    args = parser.parse_args()
    if args.config:
        rows = execute_suite(args.config, args.url, args.results_root, args.prometheus_url)
        print(json.dumps({"runs": len(rows), "suite": args.config.stem}))
        return
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary_files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(summaries)}))


def comparison_as_dict(value: PolicyComparison) -> dict[str, object]:
    return asdict(value)
