from pathlib import Path

import pytest

from experiments.runner import (
    build_refinement_specs,
    build_run_specs,
    classify_evidence,
    compare_and_record_policies,
    compare_policies,
    detect_saturation,
    execute_suite,
    loadgen_command,
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


def test_policy_cannot_win_by_rejecting_requests() -> None:
    result = compare_policies(
        [100, 105, 110], [60, 65, 70],
        [10, 10, 10], [10, 10, 10],
        fifo_completion=[1.0, 1.0, 1.0],
        slo_completion=[0.8, 0.82, 0.79],
    )
    assert result.keep is False
    assert result.completion_ratio.high < 0.95
    assert "completion" in result.reason.lower()


def test_policy_cannot_win_with_any_completion_rate_regression() -> None:
    result = compare_policies(
        [100, 105, 110], [60, 65, 70],
        [10, 10, 10], [10, 10, 10],
        fifo_completion=[1.0, 1.0, 1.0],
        slo_completion=[0.96, 0.96, 0.96],
    )
    assert result.keep is False
    assert result.completion_ratio.estimate == 0.96


def test_policy_cannot_win_when_both_policies_have_failures() -> None:
    result = compare_policies(
        [100, 105, 110], [60, 65, 70],
        [10, 10, 10], [10, 10, 10],
        fifo_completion=[0.9, 0.9, 0.9],
        slo_completion=[0.9, 0.9, 0.9],
    )
    assert result.keep is False
    assert "clean" in result.reason.lower()


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


def test_state_ledger_records_measured_confidence_intervals(tmp_path: Path) -> None:
    path = tmp_path / "BENCH_STATE.md"
    append_state(
        path, run_id="r2", bottleneck="TTFT", hypothesis="delay long prompts",
        variable="policy", decision="revert",
        measurement="p95 TTFT delta: -5.0 ms, 95% CI [-10.0, 2.0] ms",
    )
    assert "95% CI [-10.0, 2.0]" in path.read_text()


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
        "name": "scheduler", "workload": "mixed", "seed": 1, "repeats": 4,
        "requests": 50, "policies": ["fifo", "slo"], "concurrency": {"coarse": [8]},
    })
    assert len(specs) == 8
    assert {spec.policy for spec in specs} == {"fifo", "slo"}
    assert [(spec.repeat, spec.policy) for spec in specs] == [
        (0, "fifo"), (0, "slo"),
        (1, "slo"), (1, "fifo"),
        (2, "fifo"), (2, "slo"),
        (3, "slo"), (3, "fifo"),
    ]
    assert specs[0].seed == specs[1].seed
    assert specs[2].seed == specs[3].seed


def test_scheduler_specs_require_balanced_policy_order() -> None:
    with pytest.raises(ValueError, match="balanced policy order"):
        build_run_specs({
            "name": "scheduler", "seed": 1, "repeats": 3,
            "policies": ["fifo", "slo"], "concurrency": {"coarse": [8]},
        })


def test_suite_rows_record_reproducibility_and_cache_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    import yaml

    config = {
        "name": "scheduler", "model": "open/model", "workload": "mixed",
        "seed": 7, "repeats": 4, "requests": 5,
        "policies": ["fifo", "slo"], "concurrency": {"coarse": [2]},
        "cache_isolation": "paired-alternating-shared-engine",
        "cache_state_initial": "warm-or-unknown",
    }
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def fake_run(command, check):
        assert check is True
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True)
        (output.parent / "summary.json").write_text(json.dumps({
            "requests": 5, "successful": 5, "rejections": 0,
            "failures": 0, "timeouts": 0,
            "ttft_ms": {"p95": 100}, "requests_per_second": 10,
        }))

    monkeypatch.setattr("experiments.runner.subprocess.run", fake_run)
    monkeypatch.setattr("experiments.runner.append_state", lambda *args, **kwargs: None)
    rows = execute_suite(config_path, "http://router", tmp_path / "results", evidence_kind="mock")
    assert all(row["model"] == "open/model" for row in rows)
    assert all(row["workload"] == "mixed" for row in rows)
    assert all(row["requested_requests"] == 5 for row in rows)
    assert all(row["cache_isolation"] == "paired-alternating-shared-engine" for row in rows)
    assert all(row["cache_state_initial"] == "warm-or-unknown" for row in rows)
    assert [row["policy"] for row in rows] == [
        "fifo", "slo", "slo", "fifo", "fifo", "slo", "slo", "fifo",
    ]


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


def test_saturation_is_not_invented_when_throughput_keeps_rising() -> None:
    rows = [
        {"concurrency": 1, "requests_per_second": 2, "p95_ttft_ms": 100},
        {"concurrency": 2, "requests_per_second": 4, "p95_ttft_ms": 110},
        {"concurrency": 4, "requests_per_second": 8, "p95_ttft_ms": 120},
    ]
    transition = detect_saturation(rows)
    assert transition.saturation_concurrency is None
    assert transition.refinement_concurrency is None


def test_saturation_handles_levels_without_first_tokens() -> None:
    rows = [
        {"concurrency": 1, "requests_per_second": 2, "p95_ttft_ms": 100},
        {"concurrency": 2, "requests_per_second": 4, "p95_ttft_ms": 140},
        {"concurrency": 4, "requests_per_second": 0, "p95_ttft_ms": None},
    ]
    transition = detect_saturation(rows)
    assert transition.saturation_concurrency is None


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


def test_suite_command_passes_model_to_loadgen() -> None:
    spec = build_run_specs({
        "name": "awq", "seed": 1, "repeats": 3, "requests": 10,
        "concurrency": {"coarse": [1]},
    })[0]
    command = loadgen_command(
        spec, {"model": "quantized/model"}, "http://router", Path("results/requests.jsonl")
    )
    assert command[command.index("--model") + 1] == "quantized/model"


def test_gpu_evidence_requires_cache_and_dcgm_telemetry() -> None:
    assert classify_evidence("gpu", {"kv_cache_peak": 0.8}) == "unknown"
    assert classify_evidence("gpu", {
        "kv_cache_peak": 0.8,
        "gpu_utilization_peak": 92,
        "gpu_memory_peak_mib": 24000,
    }) == "unknown"
    assert classify_evidence("gpu", {
        "kv_cache_peak": 0.8,
        "gpu_utilization_peak": 92,
        "gpu_memory_peak_mib": 24000,
        "vllm_queue_mean_ms": 30,
        "vllm_prefill_mean_ms": 55,
    }) == "gpu"
    assert classify_evidence("mock", {
        "kv_cache_peak": 0.8,
        "gpu_utilization_peak": 92,
        "gpu_memory_peak_mib": 24000,
    }) == "mock"


def test_all_rejected_policy_run_is_inconclusive_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    entries = []
    def capture_state(*args, **kwargs):
        entries.append(kwargs)
    monkeypatch.setattr("experiments.runner.append_state", capture_state)
    rows = [
        {"policy": "fifo", "concurrency": 8, "repeat": repeat,
         "p95_ttft_ms": 100, "requests_per_second": 10,
         "requests": 100, "successful": 100}
        for repeat in range(3)
    ] + [
        {"policy": "slo", "concurrency": 8, "repeat": repeat,
         "p95_ttft_ms": None, "requests_per_second": 0,
         "requests": 100, "successful": 0, "rejections": 100}
        for repeat in range(3)
    ]
    compare_and_record_policies(
        {"name": "scheduler"}, rows, tmp_path, "gpu"
    )
    import json
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["keep"] is False
    assert comparison["completion_ratio"]["estimate"] == 0.0
    assert entries[0]["decision"] == "inconclusive"


def test_mock_policy_artifact_cannot_serialize_keep_true(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.runner.append_state", lambda *args, **kwargs: None)
    rows = []
    for repeat in range(3):
        rows.extend([
            {"policy": "fifo", "concurrency": 8, "repeat": repeat,
             "p95_ttft_ms": 100, "requests_per_second": 10},
            {"policy": "slo", "concurrency": 8, "repeat": repeat,
             "p95_ttft_ms": 50, "requests_per_second": 10},
        ])
    compare_and_record_policies({"name": "scheduler"}, rows, tmp_path, "mock")
    import json
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["statistical_keep"] is True
    assert comparison["keep"] is False


def test_policy_artifact_records_completion_and_rejection_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("experiments.runner.append_state", lambda *args, **kwargs: None)
    rows = []
    for repeat in range(3):
        rows.extend([
            {
                "policy": "fifo", "concurrency": 8, "repeat": repeat,
                "p95_ttft_ms": 100, "requests_per_second": 10,
                "requests": 100, "successful": 100, "rejections": 0, "timeouts": 0,
            },
            {
                "policy": "slo", "concurrency": 8, "repeat": repeat,
                "p95_ttft_ms": 70, "requests_per_second": 10,
                "requests": 100, "successful": 80, "rejections": 20, "timeouts": 0,
            },
        ])
    compare_and_record_policies(
        {"name": "scheduler", "throughput_floor_ratio": 0.95},
        rows, tmp_path, "gpu",
    )
    import json
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["keep"] is False
    assert comparison["completion_ratio"]["estimate"] == 0.8
    assert comparison["policy_totals"]["slo"]["rejections"] == 60
