"""End-to-end graph perturbation checks for the standalone metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import yaml

from apls_tlts import APLS, TLTS

_GRAPH_PARAMETER_KEYS = frozenset(
    {
        "seed",
        "node_count",
        "chord_count",
        "position_extent",
        "chord_span",
        "coordinate_dim",
    }
)


def _load_graph_parameters(config_path: Path) -> dict[str, int | float]:
    """Load the graph fixture parameters from one exact YAML schema."""

    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError("graph perturbation configuration must be a YAML mapping")

    actual_keys = set(document)
    missing_keys = _GRAPH_PARAMETER_KEYS - actual_keys
    unknown_keys = actual_keys - _GRAPH_PARAMETER_KEYS
    if missing_keys or unknown_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing keys {sorted(missing_keys)!r}")
        if unknown_keys:
            details.append(f"unknown keys {sorted(unknown_keys)!r}")
        raise ValueError(", ".join(details))

    integer_parameters = (
        "seed",
        "node_count",
        "chord_count",
        "chord_span",
        "coordinate_dim",
    )
    parameters: dict[str, int | float] = {}
    for name in integer_parameters:
        value = document[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"graph perturbation parameter {name!r} must be an integer"
            )
        parameters[name] = int(value)
    position_extent = document["position_extent"]
    if isinstance(position_extent, bool) or not isinstance(
        position_extent, (int, float)
    ):
        raise ValueError(
            "graph perturbation parameter 'position_extent' must be a number"
        )
    parameters["position_extent"] = float(position_extent)

    if parameters["seed"] < 0:
        raise ValueError("graph perturbation parameter 'seed' must be non-negative")
    if parameters["node_count"] < 2:
        raise ValueError("graph perturbation parameter 'node_count' must be at least 2")
    if parameters["chord_count"] < 0:
        raise ValueError(
            "graph perturbation parameter 'chord_count' must be non-negative"
        )
    if (
        not np.isfinite(parameters["position_extent"])
        or parameters["position_extent"] <= 0.0
    ):
        raise ValueError(
            "graph perturbation parameter 'position_extent' must be finite and positive"
        )
    if parameters["chord_span"] < 3:
        raise ValueError("graph perturbation parameter 'chord_span' must be at least 3")
    if parameters["coordinate_dim"] not in (2, 3):
        raise ValueError("graph perturbation parameter 'coordinate_dim' must be 2 or 3")
    return parameters


def _random_connected_graph(
    seed: int,
    node_count: int,
    chord_count: int,
    position_extent: float,
    chord_span: int,
    coordinate_dim: int,
) -> nx.Graph:
    """Build a seeded, connected graph with finite 2D or 3D positions.

    A randomized Hamiltonian backbone guarantees connectivity and long routes;
    local random chords make the topology non-trivial without removing all
    bridge edges.  The caller supplies every generation parameter explicitly.
    """

    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if chord_count < 0:
        raise ValueError("chord_count must be non-negative")
    if not np.isfinite(position_extent) or position_extent <= 0.0:
        raise ValueError("position_extent must be finite and positive")
    if chord_span < 3:
        raise ValueError("chord_span must be at least 3")
    if coordinate_dim not in (2, 3):
        raise ValueError("coordinate_dim must be 2 or 3")

    random_state = np.random.default_rng(seed)
    path_order = random_state.permutation(node_count).tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(zip(path_order, path_order[1:]))

    positions = random_state.uniform(
        0.0, position_extent, size=(node_count, coordinate_dim)
    )
    graph.add_nodes_from(
        (node, {"pos": tuple(float(value) for value in positions[node])})
        for node in range(node_count)
    )

    candidates = [
        (path_order[left], path_order[right])
        for left in range(node_count)
        for right in range(left + 2, min(node_count, left + chord_span))
    ]
    if chord_count > len(candidates):
        raise ValueError("chord_count exceeds the available local non-edges")
    random_state.shuffle(candidates)
    graph.add_edges_from(candidates[:chord_count])
    return graph


def _bridge_with_largest_split(graph: nx.Graph) -> tuple[int, int]:
    """Select a deterministic bridge whose removal separates many nodes."""

    candidates: list[tuple[int, tuple[int, int]]] = []
    for left, right in nx.bridges(graph):
        trial = graph.copy()
        trial.remove_edge(left, right)
        component_sizes = [
            len(component) for component in nx.connected_components(trial)
        ]
        edge = tuple(sorted((left, right)))
        candidates.append((min(component_sizes), edge))
    if not candidates:
        raise ValueError("graph must contain a bridge")
    return max(candidates)[1]


def _longest_non_edge(graph: nx.Graph) -> tuple[int, int]:
    """Select a deterministic non-edge with the longest current route."""

    candidates = [
        (
            nx.shortest_path_length(graph, left, right),
            tuple(sorted((left, right))),
        )
        for left, right in nx.non_edges(graph)
    ]
    if not candidates:
        raise ValueError("graph must contain a non-edge")
    return max(candidates)[1]


def _assert_tlts_fractions(scores: dict[str, float]) -> None:
    """Check that a scored, sampled case is a complete TLTS partition."""

    assert sum(scores.values()) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def _assert_graph_coordinates(graph: nx.Graph, coordinate_dim: int) -> None:
    """Check that every fixture node has a finite YAML-sized coordinate."""

    assert all(
        isinstance(graph.nodes[node]["pos"], tuple)
        and len(graph.nodes[node]["pos"]) == coordinate_dim
        and all(np.isfinite(value) for value in graph.nodes[node]["pos"])
        for node in graph
    )


def test_random_graph_scores_identical_deleted_and_added_edge_cases() -> None:
    """Score one graph unchanged and after one topology edit of each kind."""

    config_directory = Path(__file__).with_name("configs")
    apls = APLS(config_directory / "apls.yaml")
    tlts = TLTS(config_directory / "tlts.yaml")

    graph_parameters = _load_graph_parameters(
        config_directory / "graph_perturbations.yaml"
    )
    graph = _random_connected_graph(**graph_parameters)
    assert nx.is_connected(graph)
    assert graph.number_of_edges() > graph.number_of_nodes()
    _assert_graph_coordinates(graph, graph_parameters["coordinate_dim"])

    deleted_edge = _bridge_with_largest_split(graph)
    added_edge = _longest_non_edge(graph)
    deleted_prediction = graph.copy()
    deleted_prediction.remove_edge(*deleted_edge)
    added_prediction = graph.copy()
    added_prediction.add_edge(*added_edge)

    identical_apls = apls.score(graph, graph)
    identical_tlts = tlts.score(graph, graph)
    deleted_apls = apls.score(graph, deleted_prediction)
    deleted_tlts = tlts.score(graph, deleted_prediction)
    added_apls = apls.score(graph, added_prediction)
    added_tlts = tlts.score(graph, added_prediction)

    print(f"identical: apls={identical_apls:.12f}, tlts={identical_tlts}")
    print(f"deleted {deleted_edge}: apls={deleted_apls:.12f}, " f"tlts={deleted_tlts}")
    print(f"added {added_edge}: apls={added_apls:.12f}, " f"tlts={added_tlts}")

    assert identical_apls == pytest.approx(1.0)
    assert identical_tlts == {
        "tlts_correct": 1.0,
        "tlts_tooshort": 0.0,
        "tlts_toolong": 0.0,
        "tlts_infeasible": 0.0,
    }
    _assert_tlts_fractions(identical_tlts)

    assert not nx.is_connected(deleted_prediction)
    assert deleted_apls < identical_apls
    assert deleted_tlts["tlts_correct"] < identical_tlts["tlts_correct"]
    assert deleted_tlts["tlts_infeasible"] > identical_tlts["tlts_infeasible"]
    _assert_tlts_fractions(deleted_tlts)

    assert nx.is_connected(added_prediction)
    assert added_apls < identical_apls
    assert added_tlts["tlts_correct"] < identical_tlts["tlts_correct"]
    assert added_tlts["tlts_tooshort"] > 0.0
    _assert_tlts_fractions(added_tlts)


def test_random_graph_scores_identical_deleted_and_added_edge_cases_3d() -> None:
    """Score the same perturbations with the YAML-configured 3D TLTS metric."""

    config_directory = Path(__file__).with_name("configs")
    apls = APLS(config_directory / "apls_3d.yaml")
    tlts = TLTS(config_directory / "tlts_3d.yaml")
    graph_parameters = _load_graph_parameters(
        config_directory / "graph_perturbations_3d.yaml"
    )
    graph = _random_connected_graph(**graph_parameters)
    assert nx.is_connected(graph)
    assert graph.number_of_edges() > graph.number_of_nodes()
    _assert_graph_coordinates(graph, graph_parameters["coordinate_dim"])

    deleted_edge = _bridge_with_largest_split(graph)
    added_edge = _longest_non_edge(graph)
    deleted_prediction = graph.copy()
    deleted_prediction.remove_edge(*deleted_edge)
    added_prediction = graph.copy()
    added_prediction.add_edge(*added_edge)

    identical_apls = apls.score(graph, graph)
    deleted_apls = apls.score(graph, deleted_prediction)
    added_apls = apls.score(graph, added_prediction)
    identical_tlts = tlts.score(graph, graph)
    deleted_tlts = tlts.score(graph, deleted_prediction)
    added_tlts = tlts.score(graph, added_prediction)

    print(f"3d identical: apls={identical_apls:.12f}, tlts={identical_tlts}")
    print(
        f"3d deleted {deleted_edge}: apls={deleted_apls:.12f}, " f"tlts={deleted_tlts}"
    )
    print(f"3d added {added_edge}: apls={added_apls:.12f}, " f"tlts={added_tlts}")

    assert identical_apls == pytest.approx(1.0)
    assert identical_tlts == {
        "tlts_correct": 1.0,
        "tlts_tooshort": 0.0,
        "tlts_toolong": 0.0,
        "tlts_infeasible": 0.0,
    }
    _assert_tlts_fractions(identical_tlts)

    assert not nx.is_connected(deleted_prediction)
    assert deleted_apls < identical_apls
    assert deleted_tlts["tlts_correct"] < identical_tlts["tlts_correct"]
    assert deleted_tlts["tlts_infeasible"] > identical_tlts["tlts_infeasible"]
    _assert_tlts_fractions(deleted_tlts)

    assert nx.is_connected(added_prediction)
    assert added_apls < identical_apls
    assert added_tlts["tlts_correct"] < identical_tlts["tlts_correct"]
    assert added_tlts["tlts_tooshort"] > 0.0
    _assert_tlts_fractions(added_tlts)


def test_random_graph_tlts_yaml_dimensions_are_enforced() -> None:
    """Reject each random graph when scored with the other dimension's YAML."""

    config_directory = Path(__file__).with_name("configs")
    graph_2d = _random_connected_graph(
        **_load_graph_parameters(config_directory / "graph_perturbations.yaml")
    )
    graph_3d = _random_connected_graph(
        **_load_graph_parameters(config_directory / "graph_perturbations_3d.yaml")
    )

    with pytest.raises(ValueError, match="2-D"):
        TLTS(config_directory / "tlts.yaml").score(graph_3d, graph_3d)
    with pytest.raises(ValueError, match="3-D"):
        TLTS(config_directory / "tlts_3d.yaml").score(graph_2d, graph_2d)
