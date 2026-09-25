"""Graph-level Too-Long/Too-Short (TLTS) metric.

TLTS consumes NetworkX graphs.  Graph construction belongs to the caller, so
the public metric has no mask, tensor, or threshold arguments; dimensionality
is selected in YAML:

.. code-block:: python

    metric = TLTS("apls_tlts/configs/tlts.yaml")
    scores = metric.score(ground_truth_graph, prediction_graph)

The graph nodes must either be coordinate tuples or carry a numeric ``pos``
node attribute.  The graph may be 2-D or 3-D; its coordinate dimension must
match ``data_dim`` in the YAML configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import networkx as nx
import numpy as np

from .config import MetricConfig, load_config


__all__ = ["TLTS"]


_TLTS_PARAMETER_KEYS = frozenset(
    {
        "num_paths",
        "min_path_length",
        "radius_match",
        "length_deviation",
        "max_attempts",
        "random_seed",
    }
)


def _validate_tlts_config(config: MetricConfig) -> MetricConfig:
    """Validate TLTS-specific keys and normalize their numeric values."""

    if config.metric != "tlts":
        raise ValueError(f"configuration metric must be 'tlts', got {config.metric!r}")
    if config.data_dim not in (2, 3):
        raise ValueError("TLTS configuration requires data_dim=2 or data_dim=3")

    parameters = config.parameters
    actual_keys = set(parameters)
    missing_keys = _TLTS_PARAMETER_KEYS - actual_keys
    unknown_keys = actual_keys - _TLTS_PARAMETER_KEYS
    if missing_keys or unknown_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing keys {sorted(missing_keys)!r}")
        if unknown_keys:
            details.append(f"unknown keys {sorted(unknown_keys)!r}")
        raise ValueError(f"parameters has invalid keys: {', '.join(details)}")

    def require_integer(name: str, minimum: int) -> int:
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"parameters.{name} must be an integer at least {minimum}"
            )
        return int(value)

    def require_number(name: str) -> float:
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"parameters.{name} must be a number")
        normalized = float(value)
        if not np.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                f"parameters.{name} must be finite and non-negative"
            )
        return normalized

    normalized_parameters = {
        "num_paths": require_integer("num_paths", 0),
        "min_path_length": require_integer("min_path_length", 1),
        "radius_match": require_number("radius_match"),
        "length_deviation": require_number("length_deviation"),
        "max_attempts": require_integer("max_attempts", 0),
        "random_seed": require_integer("random_seed", 0),
    }
    return MetricConfig(
        metric="tlts",
        data_dim=config.data_dim,
        parameters=MappingProxyType(normalized_parameters),
    )


def _node_coordinate(graph: nx.Graph, node: Any) -> np.ndarray:
    """Return one graph node's numeric coordinate."""

    node_data = graph.nodes[node]
    if "pos" in node_data:
        raw_coordinate = node_data["pos"]
    else:
        raw_coordinate = node
    coordinate = np.asarray(raw_coordinate, dtype=float)
    if coordinate.ndim != 1:
        raise ValueError(
            f"graph node coordinate must be one-dimensional: {raw_coordinate!r}"
        )
    if coordinate.size not in (2, 3):
        raise ValueError(
            f"TLTS graph node coordinates must be 2-D or 3-D, got {coordinate.size}-D"
        )
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"graph node coordinate must be finite: {raw_coordinate!r}")
    return coordinate


def _path_endpoint(graph: nx.Graph, node: Any) -> tuple[Any, ...]:
    """Return a tuple endpoint while preserving coordinate tuple identifiers."""

    if isinstance(node, tuple):
        try:
            coordinate = np.asarray(node, dtype=float)
        except (TypeError, ValueError):
            coordinate = np.empty(0, dtype=float)
        if (
            coordinate.ndim == 1
            and coordinate.size in (2, 3)
            and np.all(np.isfinite(coordinate))
        ):
            return node
    return tuple(_node_coordinate(graph, node).tolist())


def _choice(random_state: Any, high: int, size: int) -> np.ndarray:
    """Draw indices from a NumPy Generator or RandomState-like object."""

    if not hasattr(random_state, "choice"):
        raise TypeError("random_state must provide NumPy's choice method")
    selected = random_state.choice(high, size=size, replace=False)
    return np.asarray(selected, dtype=int)


def _extract_gt_paths(
    graph_gt: nx.Graph,
    N: int,
    min_path_length: int,
    random_state: Any,
    max_attempts: int,
) -> list[dict[str, Any]]:
    """Sample finite ground-truth shortest paths from a NetworkX graph.

    ``s_gt`` and ``t_gt`` retain tuple node identifiers when the source graph
    uses coordinate tuples.  For identifier-based graphs with a ``pos``
    attribute, tuple coordinates are returned instead.  The shortest path
    itself always retains the graph's node identifiers.
    """

    if type(N) is not int or N < 0:
        raise ValueError(f"N must be a non-negative integer, got {N!r}")
    if type(min_path_length) is not int or min_path_length < 1:
        raise ValueError(
            f"min_path_length must be a positive integer, got {min_path_length!r}"
        )
    if type(max_attempts) is not int or max_attempts < 0:
        raise ValueError(
            f"max_attempts must be a non-negative integer, got {max_attempts!r}"
        )
    if N == 0 or max_attempts == 0 or graph_gt.number_of_nodes() < 2:
        return []

    components = []
    for component in nx.connected_components(graph_gt):
        if len(component) < 2:
            continue
        # connected_components yields sets; graph insertion order matches the
        # framework sampler and keeps seeded endpoint selection reproducible.
        components.append(tuple(node for node in graph_gt if node in component))
    if not components:
        return []

    paths: list[dict[str, Any]] = []
    for _ in range(max_attempts):
        if len(paths) >= N:
            break
        component_index = int(_choice(random_state, len(components), size=1)[0])
        component = components[component_index]
        endpoint_indices = _choice(random_state, len(component), size=2)
        source = component[int(endpoint_indices[0])]
        target = component[int(endpoint_indices[1])]
        try:
            shortest_path = list(nx.shortest_path(graph_gt, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(shortest_path) < min_path_length:
            continue
        paths.append(
            {
                "s_gt": _path_endpoint(graph_gt, source),
                "t_gt": _path_endpoint(graph_gt, target),
                "shortest_path_gt": shortest_path,
            }
        )
    return paths


def _prediction_nodes(graph_pred: nx.Graph) -> tuple[list[Any], np.ndarray]:
    """Return prediction node identifiers and a coordinate matrix."""

    nodes = list(graph_pred.nodes())
    if not nodes:
        return nodes, np.empty((0, 0), dtype=float)
    coordinates = np.stack(
        [_node_coordinate(graph_pred, node) for node in nodes], axis=0
    )
    return nodes, coordinates


def _toolong_tooshort_score(
    paths_gt: list[Mapping[str, Any]],
    graph_pred: nx.Graph,
    radius_match: float,
    length_deviation: float,
) -> tuple[int, float, float, float, float, list[dict[str, Any]]]:
    """Score sampled paths against a prediction graph.

    The tuple is ``(total, correct, tooshort, toolong, infeasible, details)``.
    Distances are measured using node coordinates while route lengths follow
    NetworkX shortest paths and count path nodes, matching the framework TLTS
    implementation.
    """

    if not np.isfinite(float(radius_match)) or radius_match < 0:
        raise ValueError(f"radius_match must be non-negative, got {radius_match!r}")
    if not np.isfinite(float(length_deviation)) or length_deviation < 0:
        raise ValueError(
            f"length_deviation must be non-negative, got {length_deviation!r}"
        )

    total = len(paths_gt)
    if total == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, []

    pred_nodes, pred_coordinates = _prediction_nodes(graph_pred)
    if not pred_nodes:
        return total, 0.0, 0.0, 0.0, 1.0, []

    counter_correct = 0
    counter_tooshort = 0
    counter_toolong = 0
    counter_infeasible = 0
    details: list[dict[str, Any]] = []

    for path in paths_gt:
        if "s_gt" not in path or "t_gt" not in path or "shortest_path_gt" not in path:
            raise ValueError(
                "each ground-truth path must contain s_gt, t_gt, and shortest_path_gt"
            )
        source_gt = np.asarray(path["s_gt"], dtype=float)
        target_gt = np.asarray(path["t_gt"], dtype=float)
        shortest_path_gt = list(path["shortest_path_gt"])
        if source_gt.ndim != 1 or target_gt.ndim != 1:
            raise ValueError("s_gt and t_gt must be one-dimensional coordinates")
        if pred_coordinates.shape[1] != source_gt.shape[0]:
            raise ValueError(
                "ground-truth and prediction graph coordinates have different dimensions"
            )

        source_distance = np.linalg.norm(pred_coordinates - source_gt[None, :], axis=1)
        target_distance = np.linalg.norm(pred_coordinates - target_gt[None, :], axis=1)
        source_candidates = np.flatnonzero(source_distance < radius_match)
        target_candidates = np.flatnonzero(target_distance < radius_match)
        if len(source_candidates) == 0 or len(target_candidates) == 0:
            counter_infeasible += 1
            details.append(
                {
                    "line_gt": shortest_path_gt,
                    "line_pred": None,
                    "s_gt": source_gt,
                    "t_gt": target_gt,
                    "s_pred": None,
                    "t_pred": None,
                    "tooshort": False,
                    "toolong": False,
                    "correct": False,
                    "infeasible": True,
                }
            )
            continue

        source_index = int(source_candidates[np.argmin(source_distance[source_candidates])])
        target_index = int(target_candidates[np.argmin(target_distance[target_candidates])])
        source_pred = pred_nodes[source_index]
        target_pred = pred_nodes[target_index]
        try:
            shortest_path_pred = list(
                nx.shortest_path(graph_pred, source_pred, target_pred)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            counter_infeasible += 1
            details.append(
                {
                    "line_gt": shortest_path_gt,
                    "line_pred": None,
                    "s_gt": source_gt,
                    "t_gt": target_gt,
                    "s_pred": source_pred,
                    "t_pred": target_pred,
                    "tooshort": False,
                    "toolong": False,
                    "correct": False,
                    "infeasible": True,
                }
            )
            continue

        gt_length = len(shortest_path_gt)
        pred_length = len(shortest_path_pred)
        is_toolong = pred_length > gt_length * (1.0 + length_deviation)
        is_tooshort = pred_length < gt_length * (1.0 - length_deviation)
        is_correct = not is_toolong and not is_tooshort
        counter_toolong += int(is_toolong)
        counter_tooshort += int(is_tooshort)
        counter_correct += int(is_correct)
        details.append(
            {
                "line_gt": shortest_path_gt,
                "line_pred": shortest_path_pred,
                "s_gt": source_gt,
                "t_gt": target_gt,
                "s_pred": source_pred,
                "t_pred": target_pred,
                "tooshort": is_tooshort,
                "toolong": is_toolong,
                "correct": is_correct,
                "infeasible": False,
            }
        )

    return (
        total,
        counter_correct / total,
        counter_tooshort / total,
        counter_toolong / total,
        counter_infeasible / total,
        details,
    )


class TLTS:
    """Strict-YAML-configured graph-level TLTS metric."""

    def __init__(self, config_path: str | Path) -> None:
        path = Path(config_path)
        self._config = _validate_tlts_config(load_config(path, "tlts"))

    @property
    def config(self) -> MetricConfig:
        """Return the immutable validated YAML configuration."""

        return self._config

    @staticmethod
    def _validate_graph(graph: nx.Graph, name: str, data_dim: int) -> None:
        if not isinstance(graph, nx.Graph):
            raise TypeError(
                f"{name} must be a NetworkX graph, got {type(graph).__name__}"
            )
        if graph.is_directed():
            raise TypeError(f"{name} must be an undirected NetworkX graph")
        for node in graph.nodes:
            coordinate = _node_coordinate(graph, node)
            if coordinate.size != data_dim:
                raise ValueError(
                    f"{name} node coordinates must be {data_dim}-D, "
                    f"got {coordinate.size}-D"
                )

    def score(self, gt_graph: nx.Graph, pred_graph: nx.Graph) -> dict[str, float]:
        """Score one ground-truth graph against one prediction graph."""

        config = self._config
        parameters = config.parameters
        self._validate_graph(gt_graph, "gt_graph", config.data_dim)
        self._validate_graph(pred_graph, "pred_graph", config.data_dim)
        random_state = np.random.default_rng(parameters["random_seed"])
        paths = _extract_gt_paths(
            gt_graph,
            N=parameters["num_paths"],
            min_path_length=parameters["min_path_length"],
            random_state=random_state,
            max_attempts=parameters["max_attempts"],
        )
        if not paths:
            return {
                "tlts_correct": 0.0,
                "tlts_tooshort": 0.0,
                "tlts_toolong": 0.0,
                "tlts_infeasible": 1.0,
            }
        _, correct, tooshort, toolong, infeasible, _ = _toolong_tooshort_score(
            paths,
            pred_graph,
            radius_match=parameters["radius_match"],
            length_deviation=parameters["length_deviation"],
        )
        return {
            "tlts_correct": correct,
            "tlts_tooshort": tooshort,
            "tlts_toolong": toolong,
            "tlts_infeasible": infeasible,
        }

    def __call__(self, gt_graph: nx.Graph, pred_graph: nx.Graph) -> dict[str, float]:
        """Delegate callable use to :meth:`score`; both inputs remain graphs."""

        return self.score(gt_graph, pred_graph)
