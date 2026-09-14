"""Confidence-aware experiment comparison CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from servebench.stats import ConfidenceInterval, bootstrap_ci


@dataclass(frozen=True, slots=True)
class PolicyComparison:
    keep: bool
    ttft_delta_ms: ConfidenceInterval
    throughput_ratio: ConfidenceInterval
    reason: str


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_files", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-manifest.json"))
    args = parser.parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary_files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(summaries)}))


def comparison_as_dict(value: PolicyComparison) -> dict[str, object]:
    return asdict(value)
