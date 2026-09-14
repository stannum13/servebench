from itertools import pairwise

from servebench.workloads import WorkloadGenerator, arrival_schedule


def test_generation_is_exactly_reproducible() -> None:
    generator = WorkloadGenerator()
    assert generator.generate("mixed", 20, 7) == generator.generate("mixed", 20, 7)
    assert generator.generate("mixed", 20, 7) != generator.generate("mixed", 20, 8)


def test_short_and_long_prefill_sizes_match_contract() -> None:
    generator = WorkloadGenerator()
    short = generator.generate("short", 1, 1)[0]
    long = generator.generate("long-prefill", 1, 1)[0]
    assert short.prompt_tokens == 256
    assert short.max_tokens == 128
    assert long.prompt_tokens == 4096
    assert long.max_tokens == 128


def test_shared_prefix_is_byte_identical() -> None:
    requests = WorkloadGenerator().generate("shared-prefix", 3, 4)
    prefixes = [request.prompt.rsplit(" UNIQUE_SUFFIX ", 1)[0] for request in requests]
    assert len(set(prefixes)) == 1
    assert len({request.prompt for request in requests}) == 3


def test_bursty_schedule_contains_concurrency_spike() -> None:
    times = arrival_schedule("bursty", count=30, rate=2.0, seed=3)
    gaps = [later - earlier for earlier, later in pairwise(times)]
    assert times == sorted(times)
    assert sum(gap == 0 for gap in gaps) >= 9


def test_mixed_schedule_preserves_a_burst_component() -> None:
    times = arrival_schedule("mixed", count=30, rate=2.0, seed=3)
    gaps = [later - earlier for earlier, later in pairwise(times)]
    assert sum(gap == 0 for gap in gaps) >= 9


def test_mixed_workload_honors_weights() -> None:
    requests = WorkloadGenerator().generate("mixed", 100, 9)
    counts = {kind: sum(request.kind == kind for request in requests) for kind in {
        "short", "long-prefill", "shared-prefix", "bursty"
    }}
    assert counts == {"short": 40, "long-prefill": 20, "shared-prefix": 20, "bursty": 20}
