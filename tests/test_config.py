from pathlib import Path

import pytest
from pydantic import ValidationError

from servebench.config import load_config


def test_default_config_uses_ungated_configurable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL", raising=False)
    config = load_config(Path("configs/default.yaml"))
    assert config.model == "Qwen/Qwen2.5-7B-Instruct"


def test_environment_overrides_model_and_backend_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "local/model")
    monkeypatch.setenv("VLLM_BASE_URL", "http://worker-a:9000,http://worker-b:9000")
    config = load_config(Path("configs/default.yaml"))
    assert config.model == "local/model"
    assert config.backend.urls == ["http://worker-a:9000", "http://worker-b:9000"]


def test_mixed_weights_must_sum_to_one(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "workloads:\n"
        "  mixed_weights:\n"
        "    short: 0.8\n"
        "    long-prefill: 0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="sum to 1"):
        load_config(path)


def test_seed_is_stable_across_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERVEBENCH_SEED", raising=False)
    first = load_config(Path("configs/default.yaml"))
    second = load_config(Path("configs/default.yaml"))
    assert first.workloads.seed == second.workloads.seed == 20260914


def test_seed_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVEBENCH_SEED", "42")
    config = load_config(Path("configs/default.yaml"))
    assert config.workloads.seed == 42
