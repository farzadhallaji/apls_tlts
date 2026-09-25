"""Standalone graph-level Average Path Length Similarity (APLS).

``APLS`` loads all score-affecting values from a strict YAML configuration and
scores two NetworkX graphs. Each graph must provide coordinates matching the
configured dimension through a ``pos`` attribute, or through a coordinate tuple
node identifier. The graph adapter creates the MultiGraph shape expected by
the local APLS path-matching core and derives missing edge lengths from those
coordinates.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

import networkx as nx
import numpy as np

from . import apls_core
from .config import MetricConfig, load_config


_APLS_PARAMETER_KEYS = frozenset(
    {
        "max_nodes",
        "max_snap_dist",
        "dist_close_node",
        "allow_renaming",
        "select_intersections",
        "min_path_length",
        "random_seed",
    }
)


def _validate_apls_config(config: MetricConfig) -> MetricConfig:
    """Validate and normalize the APLS-specific part of the shared config."""

    if config.metric != "apls":
        raise ValueError(f"configuration metric must be 'apls', got {config.metric!r}")
    if config.data_dim not in {2, 3}:
        raise ValueError("APLS configuration requires data_dim=2 or data_dim=3")
    parameters: Mapping[str, Any] = config.parameters
    actual_keys = set(parameters)
    missing_keys = _APLS_PARAMETER_KEYS - actual_keys
    unknown_keys = actual_keys - _APLS_PARAMETER_KEYS
    if missing_keys or unknown_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing keys {sorted(missing_keys)!r}")
        if unknown_keys:
            details.append(f"unknown keys {sorted(unknown_keys)!r}")
        raise ValueError(f"parameters has invalid keys: {', '.join(details)}")

    max_nodes = parameters["max_nodes"]
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("parameters.max_nodes must be an integer at least 1")
    random_seed = parameters["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("parameters.random_seed must be an integer at least 0")

    def require_number(name: str) -> float:
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"parameters.{name} must be a number")
        normalized = float(value)
        if not isfinite(normalized) or normalized < 0.0:
            raise ValueError(f"parameters.{name} must be finite and non-negative")
        return normalized

    for name in ("max_snap_dist", "dist_close_node", "min_path_length"):
        require_number(name)
    for name in ("allow_renaming", "select_intersections"):
        if not isinstance(parameters[name], bool):
            raise ValueError(f"parameters.{name} must be a boolean")

    normalized_parameters = {
        "max_nodes": int(max_nodes),
        "max_snap_dist": require_number("max_snap_dist"),
        "dist_close_node": require_number("dist_close_node"),
        "allow_renaming": parameters["allow_renaming"],
        "select_intersections": parameters["select_intersections"],
        "min_path_length": require_number("min_path_length"),
        "random_seed": int(random_seed),
    }
    return MetricConfig(
        metric="apls",
        data_dim=config.data_dim,
        parameters=MappingProxyType(normalized_parameters),
    )


def _node_position(
    node: Any,
    properties: dict[str, Any],
    dimension: int,
) -> tuple[float, ...]:
    """Read one finite position with the configured graph dimension."""

    if "pos" in properties:
        raw_position = properties["pos"]
    elif isinstance(node, tuple) and len(node) == dimension:
        raw_position = node
    else:
        raise ValueError(
            f"APLS graph nodes must have a {dimension}-dimensional 'pos' attribute "
            f"or a {dimension}-coordinate tuple identifier"
        )
    try:
        position = np.asarray(raw_position, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("APLS node positions must contain numeric coordinates") from exc
    if position.shape != (dimension,) or not np.isfinite(position).all():
        raise ValueError(
            f"APLS node positions must be finite {dimension}-dimensional coordinates"
        )
    return tuple(float(value) for value in position)


def _edge_records(graph: nx.Graph) -> list[tuple[Any, Any, Any, dict[str, Any]]]:
    """Copy edge records from either a simple or multigraph."""

    if graph.is_multigraph():
        return [
            (u, v, key, dict(data))
            for u, v, key, data in graph.edges(keys=True, data=True)
        ]
    return [(u, v, 0, dict(data)) for u, v, data in graph.edges(data=True)]


def _as_apls_graph(graph: nx.Graph, role: str, dimension: int) -> nx.MultiGraph:
    """Convert a user graph to the local APLS MultiGraph representation.

    The graph's coordinate positions are copied exactly. They are never
    interpreted as geographic latitude/longitude values.
    """

    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise TypeError("APLS score inputs must be NetworkX graphs")
    if graph.is_directed():
        raise ValueError("APLS score inputs must be undirected NetworkX graphs")
    if role not in {"gt", "pred"}:
        raise ValueError(f"unsupported APLS graph role {role!r}")

    converted = nx.MultiGraph()
    source_nodes = list(graph.nodes())
    node_mapping = {
        source_node: f"{role}:{index}"
        for index, source_node in enumerate(source_nodes)
    }
    positions: dict[Any, tuple[float, ...]] = {}
    for source_node in source_nodes:
        position = _node_position(source_node, graph.nodes[source_node], dimension)
        positions[source_node] = position
        converted.add_node(
            node_mapping[source_node],
            coord=position,
            pos=np.asarray(position, dtype=float),
        )

    for source_u, source_v, _, source_data in _edge_records(graph):
        if source_u == source_v:
            raise ValueError("APLS graphs must not contain self-loop edges")
        u_position = positions[source_u]
        v_position = positions[source_v]
        segment = apls_core.EuclideanSegment(u_position, v_position)
        if segment.length <= 0.0:
            raise ValueError("APLS edges must connect distinct node positions")
        edge_data = copy.deepcopy(source_data)
        edge_data["geometry"] = segment
        if "length" in edge_data:
            try:
                length = float(edge_data["length"])
            except (TypeError, ValueError) as exc:
                raise ValueError("APLS edge 'length' values must be numeric") from exc
            if not np.isfinite(length) or length < 0.0:
                raise ValueError("APLS edge 'length' values must be finite and non-negative")
        else:
            edge_data["length"] = float(segment.length)
        converted.add_edge(
            node_mapping[source_u],
            node_mapping[source_v],
            **edge_data,
        )
    return converted


class APLS:
    """Graph-level two- or three-dimensional APLS metric."""

    def __init__(self, config_path: str | Path) -> None:
        config = load_config(config_path, "apls")
        self._config = _validate_apls_config(config)

    @property
    def config(self) -> MetricConfig:
        """Return the immutable validated metric configuration."""

        return self._config

    def score(self, ground_truth: nx.Graph, proposal: nx.Graph) -> float:
        """Return APLS for two NetworkX graphs."""

        parameters = self._config.parameters
        dimension = self._config.data_dim
        ground_truth_apls = _as_apls_graph(ground_truth, "gt", dimension)
        proposal_apls = _as_apls_graph(proposal, "pred", dimension)
        if ground_truth.number_of_nodes() == 0:
            return float(proposal.number_of_nodes() == 0)
        if proposal.number_of_nodes() == 0:
            return 0.0
        return apls_core.score_graphs(
            ground_truth_apls,
            proposal_apls,
            max_nodes=parameters["max_nodes"],
            max_snap_dist=parameters["max_snap_dist"],
            dist_close_node=parameters["dist_close_node"],
            allow_renaming=parameters["allow_renaming"],
            select_intersections=parameters["select_intersections"],
            min_path_length=parameters["min_path_length"],
            seed=parameters["random_seed"],
        )


__all__ = ["APLS"]
