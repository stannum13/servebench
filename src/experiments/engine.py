"""Independent vLLM engine-variant planning and Compose environment mapping."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import yaml

from servebench.stats import bootstrap_ci

from .runner import PolicyComparison, compare_policies
from .state import append_state

SUPPORTED_VARIABLES = {
    "max_num_batched_tokens",
    "max_num_seqs",
    "prefix_caching",
    "chunked_prefill",
    "precision",
}


@dataclass(frozen=True, slots=True)
class EngineVariant:
    name: str
    variable: str
    settings: dict[str, object]


def compare_engine_variant(
    baseline_ttft: list[float],
    variant_ttft: list[float],
    baseline_throughput: list[float],
    variant_throughput: list[float],
    throughput_floor: float = 0.95,
    *,
    baseline_completion: list[float] | None = None,
    variant_completion: list[float] | None = None,
) -> PolicyComparison:
    return compare_policies(
        baseline_ttft,
        variant_ttft,
        baseline_throughput,
        variant_throughput,
        throughput_floor,
        fifo_completion=baseline_completion,
        slo_completion=variant_completion,
    )


def engine_evidence_verified(
    baseline_rows: list[dict[str, object]], variant_rows: list[dict[str, object]]
) -> bool:
    return all(
        row.get("evidence_kind") == "gpu" and row.get("p95_ttft_ms") is not None
        for row in baseline_rows + variant_rows
    )


def _provenance(row: dict[str, object]) -> dict[str, object]:
    return {
        field: row.get(field)
        for field in (
            "model", "workload", "seed", "cache_isolation",
            "cache_state_initial", "engine_variant", "changed_variable",
            "engine_settings",
        )
    }


def _slug(value: object) -> str:
    return str(value).lower().replace("_", "-")


def build_engine_variants(
    baseline: dict[str, object], sweeps: dict[str, list[object]]
) -> list[EngineVariant]:
    missing = SUPPORTED_VARIABLES - set(baseline)
    if missing:
        raise ValueError(f"baseline is missing settings: {sorted(missing)}")
    unsupported = set(sweeps) - SUPPORTED_VARIABLES
    if unsupported:
        raise ValueError(f"unsupported engine variable: {sorted(unsupported)}")
    variants: list[EngineVariant] = []
    for variable, values in sweeps.items():
        for value in values:
            if value == baseline[variable]:
                continue
            variants.append(
                EngineVariant(
                    name=f"{variable}-{_slug(value)}",
                    variable=variable,
                    settings={**baseline, variable: value},
                )
            )
    return variants


def compose_environment(
    settings: dict[str, object], model: str, quantized_model: str
) -> dict[str, str]:
    precision = settings["precision"]
    if precision not in {"bf16", "awq"}:
        raise ValueError("precision must be bf16 or awq")
    return {
        "MODEL": quantized_model if precision == "awq" else model,
        "DTYPE": "auto" if precision == "awq" else "bfloat16",
        "MAX_BATCHED_TOKENS": str(settings["max_num_batched_tokens"]),
        "MAX_SEQS": str(settings["max_num_seqs"]),
        "PREFIX_CACHE_FLAG": (
            "--enable-prefix-caching"
            if settings["prefix_caching"]
            else "--no-enable-prefix-caching"
        ),
        "CHUNKED_PREFILL_FLAG": (
            "--enable-chunked-prefill"
            if settings["chunked_prefill"]
            else "--no-enable-chunked-prefill"
        ),
    }


def restart_vllm(environment: dict[str, str], health_url: str, timeout: float = 600) -> None:
    subprocess.run(
        ["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "vllm"],
        check=True,
        env={**os.environ, **environment},
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(health_url, timeout=2).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout:g}s")


def execute_engine_sweeps(
    config_path: Path,
    saturation_concurrency: int,
    router_url: str,
    prometheus_url: str,
    health_url: str,
    results_root: Path,
) -> list[dict[str, object]]:
    from .runner import refresh_report_input

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline = config["engine_baseline"]
    variants = build_engine_variants(baseline, config["independent_sweeps"])
    model = str(os.getenv("MODEL", config.get("model", "Qwen/Qwen2.5-7B-Instruct")))
    quantized = str(os.getenv("QUANTIZED_MODEL", config["quantized_model"]))
    baseline_environment = compose_environment(baseline, model, quantized)
    rows: list[dict[str, object]] = []
    try:
        for variant in variants:
            restart_vllm(baseline_environment, health_url)
            baseline_name = f"baseline-for-{variant.name}"
            baseline_rows = _execute_variant_suite(
                baseline_name, None, baseline, config,
                saturation_concurrency, router_url, prometheus_url, results_root, model,
            )
            rows.extend(baseline_rows)
            restart_vllm(compose_environment(variant.settings, model, quantized), health_url)
            variant_rows = _execute_variant_suite(
                variant.name, variant.variable, variant.settings, config,
                saturation_concurrency, router_url, prometheus_url, results_root,
                quantized if variant.settings["precision"] == "awq" else model,
            )
            first_tokens_present = all(
                row.get("p95_ttft_ms") is not None
                for row in baseline_rows + variant_rows
            )
            comparison = (
                compare_engine_variant(
                    [float(row["p95_ttft_ms"]) for row in baseline_rows],
                    [float(row["p95_ttft_ms"]) for row in variant_rows],
                    [float(row["requests_per_second"]) for row in baseline_rows],
                    [float(row["requests_per_second"]) for row in variant_rows],
                    baseline_completion=[_completion_rate(row) for row in baseline_rows],
                    variant_completion=[_completion_rate(row) for row in variant_rows],
                ) if first_tokens_present else None
            )
            baseline_completion = [_completion_rate(row) for row in baseline_rows]
            variant_completion = [_completion_rate(row) for row in variant_rows]
            completion_ratio = (
                comparison.completion_ratio if comparison else bootstrap_ci([
                    candidate / baseline if baseline else 1.0
                    for baseline, candidate in zip(
                        baseline_completion, variant_completion, strict=True
                    )
                ], seed=2)
            )
            verified = engine_evidence_verified(baseline_rows, variant_rows)
            reason = (
                comparison.reason if verified and comparison is not None
                else "GPU telemetry or successful first tokens missing; no engine claim"
            )
            variant_dir = results_root / f"engine-{variant.name}"
            (variant_dir / "decision.json").write_text(
                json.dumps({
                    "baseline_suite": f"engine-{baseline_name}",
                    "candidate_suite": f"engine-{variant.name}",
                    "baseline_provenance": _provenance(baseline_rows[0]),
                    "candidate_provenance": _provenance(variant_rows[0]),
                    "keep": bool(verified and comparison and comparison.keep),
                    "ttft_delta_ms": (
                        asdict(comparison.ttft_delta_ms) if comparison else None
                    ),
                    "throughput_ratio": (
                        asdict(comparison.throughput_ratio) if comparison else None
                    ),
                    "completion_ratio": (
                        asdict(completion_ratio)
                    ),
                    "reason": reason,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            append_state(
                Path("BENCH_STATE.md"),
                run_id=f"engine-{variant.name}",
                bottleneck="p95 TTFT at the measured saturation point",
                hypothesis=f"changing only {variant.variable} improves p95 TTFT",
                variable=variant.variable,
                decision=(
                    "inconclusive" if not verified else
                    "keep" if comparison and comparison.keep else "revert"
                ),
                measurement=(
                    "p95 TTFT delta: "
                    f"{comparison.ttft_delta_ms.estimate:.2f} ms, 95% CI "
                    f"[{comparison.ttft_delta_ms.low:.2f}, "
                    f"{comparison.ttft_delta_ms.high:.2f}] ms; throughput ratio: "
                    f"{comparison.throughput_ratio.estimate:.3f}, 95% CI "
                    f"[{comparison.throughput_ratio.low:.3f}, "
                    f"{comparison.throughput_ratio.high:.3f}]; completion ratio: "
                    f"{comparison.completion_ratio.estimate:.3f}, 95% CI "
                    f"[{comparison.completion_ratio.low:.3f}, "
                    f"{comparison.completion_ratio.high:.3f}]"
                ) if comparison else (
                    "No successful first token in one or more repeats; completion ratio: "
                    f"{completion_ratio.estimate:.3f}, 95% CI "
                    f"[{completion_ratio.low:.3f}, {completion_ratio.high:.3f}]"
                ),
            )
            rows.extend(variant_rows)
        refresh_report_input(results_root)
    finally:
        restart_vllm(baseline_environment, health_url)
    return rows


def _completion_rate(row: dict[str, object]) -> float:
    if "requests" not in row:
        return 1.0
    requests = int(row["requests"])
    return float(row.get("successful", 0)) / requests if requests else 0.0


def _execute_variant_suite(
    name: str,
    variable: str | None,
    settings: dict[str, object],
    config: dict[str, object],
    saturation_concurrency: int,
    router_url: str,
    prometheus_url: str,
    results_root: Path,
    served_model: str,
) -> list[dict[str, object]]:
    from .runner import execute_suite

    suite_name = f"engine-{name}"
    suite_config = {
        "name": suite_name,
        "workload": config.get("workload", "mixed"),
        "seed": config["seed"],
        "repeats": config["repeats"],
        "requests": config.get("requests", 100),
        "policy": "fifo",
        "model": served_model,
        "cache_isolation": "vllm-restart-before-suite",
        "cache_state_initial": "cold",
        "engine_variant": name,
        "changed_variable": variable,
        "engine_settings": settings,
        "concurrency": {"coarse": [saturation_concurrency]},
    }
    with tempfile.TemporaryDirectory(prefix="servebench-engine-") as directory:
        path = Path(directory) / "suite.yaml"
        path.write_text(yaml.safe_dump(suite_config), encoding="utf-8")
        variant_rows = execute_suite(path, router_url, results_root, prometheus_url)
    for row in variant_rows:
        row.update({
            "engine_variant": name,
            "changed_variable": variable,
            "engine_settings": settings,
        })
    variant_dir = results_root / suite_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "runs.json").write_text(
        json.dumps(variant_rows, indent=2) + "\n", encoding="utf-8"
    )
    return variant_rows
