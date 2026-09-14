from pathlib import Path

import pytest

from experiments.runner import compare_policies, validate_repeats
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
