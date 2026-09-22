import json
from pathlib import Path

import pytest

from experiments.engine import (
    _execute_variant_suite,
    build_engine_variants,
    compare_engine_variant,
    compose_environment,
    engine_evidence_verified,
    execute_engine_sweeps,
)


def baseline() -> dict[str, object]:
    return {
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "prefix_caching": True,
        "chunked_prefill": True,
        "precision": "bf16",
    }


def test_engine_variants_change_one_setting_at_a_time() -> None:
    variants = build_engine_variants(
        baseline(),
        {"max_num_seqs": [128, 512], "prefix_caching": [False], "precision": ["awq"]},
    )
    assert len(variants) == 4
    for variant in variants:
        differences = {key for key, value in variant.settings.items() if baseline()[key] != value}
        assert differences == {variant.variable}


def test_compose_environment_maps_bf16_and_awq() -> None:
    bf16 = compose_environment(baseline(), "base/model", "quant/model")
    assert bf16["MODEL"] == "base/model" and bf16["DTYPE"] == "bfloat16"
    awq_settings = {**baseline(), "precision": "awq"}
    awq = compose_environment(awq_settings, "base/model", "quant/model")
    assert awq["MODEL"] == "quant/model" and awq["DTYPE"] == "auto"


def test_unknown_engine_variable_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_engine_variants(baseline(), {"mystery": [1]})


def test_engine_variant_requires_confident_latency_win_without_throughput_loss() -> None:
    comparison = compare_engine_variant(
        [100, 105, 110], [75, 80, 85], [10, 10, 10], [9.8, 9.9, 10.0]
    )
    assert comparison.keep is True
    assert comparison.ttft_delta_ms.high < 0


def test_engine_variant_reverts_when_throughput_floor_fails() -> None:
    comparison = compare_engine_variant(
        [100, 105, 110], [75, 80, 85], [10, 10, 10], [8, 8, 8]
    )
    assert comparison.keep is False


def test_engine_variant_reverts_when_completion_regresses() -> None:
    comparison = compare_engine_variant(
        [100, 105, 110], [75, 80, 85], [10, 10, 10], [10, 10, 10],
        baseline_completion=[1.0, 1.0, 1.0],
        variant_completion=[0.9, 0.9, 0.9],
    )
    assert comparison.keep is False


def test_engine_variant_suite_requests_actual_served_model(tmp_path: Path, monkeypatch) -> None:
    captured = []

    def fake_execute_suite(path, *args):
        import yaml
        captured.append(yaml.safe_load(path.read_text()))
        return [{"p95_ttft_ms": 100, "requests_per_second": 10}]

    monkeypatch.setattr("experiments.runner.execute_suite", fake_execute_suite)
    _execute_variant_suite(
        "awq", "precision", {**baseline(), "precision": "awq"},
        {"seed": 1, "repeats": 3, "requests": 10}, 1,
        "http://router", "http://prometheus", tmp_path, "quantized/model",
    )
    assert captured[0]["model"] == "quantized/model"
    assert captured[0]["cache_isolation"] == "vllm-restart-before-suite"
    assert captured[0]["cache_state_initial"] == "cold"
    assert captured[0]["engine_settings"]["precision"] == "awq"


def test_engine_candidate_cannot_be_kept_with_missing_gpu_or_first_tokens() -> None:
    complete = {"evidence_kind": "gpu", "p95_ttft_ms": 100}
    assert engine_evidence_verified([complete], [complete]) is True
    assert engine_evidence_verified([complete], [{**complete, "evidence_kind": "unknown"}]) is False
    assert engine_evidence_verified([complete], [{**complete, "p95_ttft_ms": None}]) is False


def test_each_engine_candidate_gets_a_fresh_named_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    import yaml

    config = {
        "seed": 1, "repeats": 3, "requests": 2,
        "engine_baseline": baseline(),
        "independent_sweeps": {"max_num_seqs": [128, 512]},
        "quantized_model": "quant/model",
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    calls: list[str] = []
    events: list[tuple[str, str]] = []

    def fake_suite(name, *args):
        calls.append(name)
        events.append(("suite", name))
        result_dir = tmp_path / "results" / f"engine-{name}"
        result_dir.mkdir(parents=True)
        return [
            {
                "repeat": repeat, "evidence_kind": "gpu", "p95_ttft_ms": 100,
                "requests_per_second": 10,
                "model": "base/model",
                "cache_isolation": "vllm-restart-before-suite",
                "cache_state_initial": "cold",
                "engine_settings": baseline(),
            }
            for repeat in range(3)
        ]

    monkeypatch.setattr(
        "experiments.engine.restart_vllm",
        lambda environment, *args: events.append(("restart", environment["MAX_SEQS"])),
    )
    monkeypatch.setattr("experiments.engine._execute_variant_suite", fake_suite)
    monkeypatch.setattr("experiments.engine.append_state", lambda *args, **kwargs: None)
    monkeypatch.setattr("experiments.runner.refresh_report_input", lambda *args: None)
    execute_engine_sweeps(
        config_path, 8, "http://router", "http://prometheus", "http://vllm/health",
        tmp_path / "results",
    )
    assert calls == [
        "baseline-for-max_num_seqs-128", "max_num_seqs-128",
        "baseline-for-max_num_seqs-512", "max_num_seqs-512",
    ]
    assert events == [
        ("restart", "256"), ("suite", "baseline-for-max_num_seqs-128"),
        ("restart", "128"), ("suite", "max_num_seqs-128"),
        ("restart", "256"), ("suite", "baseline-for-max_num_seqs-512"),
        ("restart", "512"), ("suite", "max_num_seqs-512"),
        ("restart", "256"),
    ]
    decision = json.loads(
        (tmp_path / "results/engine-max_num_seqs-128/decision.json").read_text()
    )
    assert decision["baseline_suite"] == "engine-baseline-for-max_num_seqs-128"
    assert decision["completion_ratio"]["estimate"] == 1.0
    assert decision["baseline_provenance"]["cache_state_initial"] == "cold"
    assert decision["candidate_provenance"]["cache_isolation"] == "vllm-restart-before-suite"
