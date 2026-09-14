from pathlib import Path

import pytest

from experiments.runner import (
    build_refinement_specs,
    build_run_specs,
    compare_policies,
    detect_saturation,
    validate_repeats,
)
from experiments.state import append_state


def test_experiments_require_three_repeats() -> None:
    with pytest.raises(ValueError, match="at least three"):
        validate_repeats(2)


def test_policy_improvement_requires_ttft_and_throughput() -> None:
    result = compare_policies([100, 110, 105], [80, 82, 78], [10, 10, 10], [9.7, 9.8, 9.9])
    assert result.keep is True
    failed = compare_policies([100, 110, 105], [80, 82, 78], [10, 10, 10], [8, 8, 8])
    assert failed.keep is False


def test_state_ledger_records_hypothesis_and_decision(tmp_path: Path) -> None:
    path = tmp_path / "BENCH_STATE.md"
    append_state(
        path,
        run_id="r1",
        bottleneck="queue",
        hypothesis="delay long prompts",
        variable="policy",
        decision="keep",
    )
    text = path.read_text()
    assert "queue" in text and "delay long prompts" in text and "keep" in text


def test_baseline_specs_repeat_every_concurrency_with_distinct_seeds() -> None:
    specs = build_run_specs(
        {"name": "baseline", "workload": "mixed", "seed": 10, "repeats": 3,
         "requests": 50, "policy": "fifo", "concurrency": {"coarse": [1, 4]}}
    )
    assert len(specs) == 6
    assert {(spec.concurrency, spec.repeat) for spec in specs} == {
        (1, 0), (1, 1), (1, 2), (4, 0), (4, 1), (4, 2)
    }
    assert len({spec.seed for spec in specs}) == 3


def test_scheduler_specs_cross_policies_but_not_engine_settings() -> None:
    specs = build_run_specs({
        "name": "scheduler", "workload": "mixed", "seed": 1, "repeats": 3,
        "requests": 50, "policies": ["fifo", "slo"], "concurrency": {"coarse": [8]},
    })
    assert len(specs) == 6
    assert {spec.policy for spec in specs} == {"fifo", "slo"}


def test_saturation_detects_throughput_plateau_with_ttft_growth() -> None:
    rows = [
        {"concurrency": 1, "requests_per_second": 2, "p95_ttft_ms": 100},
        {"concurrency": 2, "requests_per_second": 3.8, "p95_ttft_ms": 110},
        {"concurrency": 4, "requests_per_second": 4.0, "p95_ttft_ms": 240},
        {"concurrency": 8, "requests_per_second": 4.1, "p95_ttft_ms": 500},
    ]
    transition = detect_saturation(rows)
    assert transition.saturation_concurrency == 4
    assert transition.refinement_concurrency == 3


def test_refinement_specs_measure_detected_midpoint() -> None:
    config = {
        "name": "baseline", "workload": "mixed", "seed": 10, "repeats": 3,
        "requests": 50, "policy": "fifo",
        "concurrency": {"coarse": [1, 2, 4], "refine_transition": True},
    }
    rows = [
        {"concurrency": 1, "requests_per_second": 2, "p95_ttft_ms": 100},
        {"concurrency": 2, "requests_per_second": 3.8, "p95_ttft_ms": 110},
        {"concurrency": 4, "requests_per_second": 4.0, "p95_ttft_ms": 240},
    ]
    specs = build_refinement_specs(config, rows)
    assert len(specs) == 3
    assert {spec.concurrency for spec in specs} == {3}
    assert {spec.seed for spec in specs} == {10, 11, 12}


def test_refinement_specs_are_opt_in() -> None:
    config = {
        "name": "baseline", "seed": 10, "repeats": 3,
        "concurrency": {"coarse": [1, 2, 4]},
    }
    rows = [
        {"concurrency": 1, "requests_per_second": 2, "p95_ttft_ms": 100},
        {"concurrency": 2, "requests_per_second": 3.8, "p95_ttft_ms": 110},
        {"concurrency": 4, "requests_per_second": 4.0, "p95_ttft_ms": 240},
    ]
    assert build_refinement_specs(config, rows) == []
