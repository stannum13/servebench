import pytest

from experiments.sweep import independent_variants, refine_concurrency_transition


def test_transition_refinement_avoids_full_grid() -> None:
    assert refine_concurrency_transition([1, 2, 4, 8, 16], saturation_index=3) == [1, 2, 4, 6, 8]


def test_variants_change_exactly_one_engine_setting() -> None:
    baseline = {"max_num_batched_tokens": 8192, "max_num_seqs": 256, "prefix_caching": True}
    variants = independent_variants(baseline, {"max_num_seqs": [64, 128]})
    assert [item["max_num_seqs"] for item in variants] == [64, 128]
    with pytest.raises(ValueError, match="exactly one"):
        independent_variants(baseline, {"max_num_seqs": [64], "prefix_caching": [False]})

