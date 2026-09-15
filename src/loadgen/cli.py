"""Command-line entry point for a single reproducible load run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from servebench.results import write_parquet
from servebench.stats import summarize_measurements
from servebench.workloads import WorkloadGenerator, arrival_schedule

from .client import LoadOptions, run_load
from .tokenize import calibrate_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--workload", default="short")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--policy", default="fifo")
    parser.add_argument("--model", default=os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--tokenizer-url", default=os.getenv("TOKENIZER_URL"))
    parser.add_argument("--api-key", default=os.getenv("ROUTER_API_KEY"))
    parser.add_argument(
        "--request-rate", type=float, default=0,
        help="Open-loop requests/second; zero uses immediate closed-loop submission",
    )
    parser.add_argument("--output", type=Path, default=Path("results/latest/requests.jsonl"))
    args = parser.parse_args()
    run_id = str(uuid.uuid4())
    specs = WorkloadGenerator().generate(args.workload, args.requests, args.seed)
    if args.tokenizer_url:
        specs = asyncio.run(calibrate_requests(specs, args.tokenizer_url, args.model))
    options = LoadOptions(
        run_id, args.url, args.policy, args.concurrency, model=args.model, api_key=args.api_key
    )
    offsets = (
        arrival_schedule(args.workload, args.requests, args.request_rate, args.seed)
        if args.request_rate > 0
        else None
    )
    results = asyncio.run(run_load(specs, options, args.output, arrival_offsets=offsets))
    summary = summarize_measurements(results)
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_parquet(args.output, args.output.with_suffix(".parquet"))
    print(json.dumps(summary, indent=2))
