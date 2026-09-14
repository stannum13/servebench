"""Command-line entry point for a single reproducible load run."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from servebench.results import write_parquet
from servebench.stats import summarize_measurements
from servebench.workloads import WorkloadGenerator

from .client import LoadOptions, run_load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--workload", default="short")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--policy", default="fifo")
    parser.add_argument("--output", type=Path, default=Path("results/latest/requests.jsonl"))
    args = parser.parse_args()
    run_id = str(uuid.uuid4())
    specs = WorkloadGenerator().generate(args.workload, args.requests, args.seed)
    options = LoadOptions(run_id, args.url, args.policy, args.concurrency)
    results = asyncio.run(run_load(specs, options, args.output))
    summary = summarize_measurements(results)
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_parquet(args.output, args.output.with_suffix(".parquet"))
    print(json.dumps(summary, indent=2))

