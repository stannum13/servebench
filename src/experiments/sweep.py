"""Transition-focused sweep construction."""

from __future__ import annotations

from typing import Any


def refine_concurrency_transition(levels: list[int], saturation_index: int) -> list[int]:
    if not 0 < saturation_index < len(levels):
        raise ValueError("saturation_index must have a preceding level")
    lower, upper = levels[saturation_index - 1 : saturation_index + 1]
    midpoint = (lower + upper) // 2
    return sorted(set(levels[:saturation_index] + [midpoint, upper]))


def independent_variants(
    baseline: dict[str, Any], changes: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    if len(changes) != 1:
        raise ValueError("a sweep must change exactly one variable")
    key, values = next(iter(changes.items()))
    if key not in baseline:
        raise ValueError(f"unknown baseline setting: {key}")
    return [{**baseline, key: value} for value in values]

