"""Strict YAML document loading shared by the standalone metric modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


_TOP_LEVEL_KEYS = frozenset({"metric", "data_dim", "parameters"})
_SUPPORTED_METRICS = frozenset({"apls", "tlts"})


@dataclass(frozen=True)
class MetricConfig:
    """The validated top-level values from one metric YAML document."""

    metric: str
    data_dim: int
    parameters: Mapping[str, Any]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _check_exact_keys(mapping: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(mapping.keys())
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys {sorted(missing)!r}")
        if unknown:
            details.append(f"unknown keys {sorted(unknown)!r}")
        raise ValueError(f"{name} has invalid keys: {', '.join(details)}")


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def load_config(config_path: str | Path, expected_metric: str) -> MetricConfig:
    """Load one metric YAML document without supplying implicit values.

    The loader owns only the common document envelope. Each metric validates
    its own parameter keys, value types, ranges, and supported dimensions
    after receiving this immutable top-level object.
    """

    if expected_metric not in _SUPPORTED_METRICS:
        raise ValueError(f"unsupported expected metric {expected_metric!r}")

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    root = _require_mapping(document, "configuration")
    _check_exact_keys(root, _TOP_LEVEL_KEYS, "configuration")

    metric = _require_string(root["metric"], "metric")
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"metric must be 'apls' or 'tlts', got {metric!r}")
    if metric != expected_metric:
        raise ValueError(
            f"configuration metric {metric!r} does not match expected {expected_metric!r}"
        )

    data_dim = _require_integer(root["data_dim"], "data_dim", 1)
    parameters = _require_mapping(root["parameters"], "parameters")
    return MetricConfig(
        metric=metric,
        data_dim=data_dim,
        parameters=MappingProxyType(dict(parameters)),
    )


__all__ = ["MetricConfig", "load_config"]
