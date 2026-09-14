import pytest

from experiments.engine import build_engine_variants, compare_engine_variant, compose_environment


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
