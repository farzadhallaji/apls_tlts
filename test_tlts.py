"""Graph-only TLTS behavior and strict-YAML configuration checks."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from apls_tlts.tlts import TLTS, _extract_gt_paths, _toolong_tooshort_score


TLTS_CONFIG = {
    "num_paths": 24,
    "min_path_length": 2,
    "radius_match": 0.5,
    "length_deviation": 0.05,
    "max_attempts": 500,
    "random_seed": 7,
}


def _write_config(
    directory: Path,
    values: dict[str, int | float | str],
    data_dim: int,
) -> Path:
    path = directory / "tlts.yaml"

    def yaml_scalar(value: int | float | str) -> str:
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return str(value)

    lines = ["metric: tlts", f"data_dim: {data_dim}", "parameters:"]
    lines.extend(f"  {key}: {yaml_scalar(value)}" for key, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _line_graph(node_count: int) -> nx.Graph:
    graph = nx.Graph()
    nodes = [(index, 0) for index in range(node_count)]
    graph.add_nodes_from((node, {"pos": np.asarray(node, dtype=int)}) for node in nodes)
    graph.add_edges_from(zip(nodes, nodes[1:]))
    return graph


def _three_d_target_graph() -> nx.Graph:
    graph = nx.Graph()
    nodes = [(1, 1, depth) for depth in range(5)]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(zip(nodes, nodes[1:]))
    graph.add_edges_from(((nodes[index], nodes[index + 2]) for index in range(3)))
    return graph


def _three_d_prediction_graph() -> nx.Graph:
    graph = nx.Graph()
    nodes = [(1, 1, 0), (2, 1, 1), (2, 2, 2), (1, 2, 3), (1, 1, 4)]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(zip(nodes, nodes[1:]))
    return graph


def test_graph_level_perfect_score_and_isolated_copy(tmp_path: Path):
    config_path = _write_config(tmp_path, TLTS_CONFIG, data_dim=2)
    ground_truth = _line_graph(16)
    metric = TLTS(config_path)

    expected = {
        "tlts_correct": 1.0,
        "tlts_tooshort": 0.0,
        "tlts_toolong": 0.0,
        "tlts_infeasible": 0.0,
    }
    assert metric.score(ground_truth, ground_truth) == expected


def test_yaml_parameters_are_not_publicly_writable(tmp_path: Path):
    metric = TLTS(_write_config(tmp_path, TLTS_CONFIG, data_dim=2))

    assert not hasattr(metric, "num_paths")
    assert not hasattr(metric, "min_path_length")
    assert not hasattr(metric, "radius_match")
    assert not hasattr(metric, "length_deviation")
    assert not hasattr(metric, "max_attempts")
    assert not hasattr(metric, "random_seed")
    assert metric.config.parameters["num_paths"] == TLTS_CONFIG["num_paths"]
    with pytest.raises(TypeError):
        metric.config.parameters["num_paths"] = 1  # type: ignore[index]


def test_graph_level_score_detects_missing_link(tmp_path: Path):
    config_path = _write_config(tmp_path, TLTS_CONFIG, data_dim=2)
    ground_truth = _line_graph(16)
    prediction = ground_truth.copy()
    prediction.remove_edge((7, 0), (8, 0))

    result = TLTS(config_path).score(ground_truth, prediction)

    assert result["tlts_infeasible"] > 0.0


def test_three_d_local_clique_path_lengths_change_fraction(tmp_path: Path):
    values = {
        "num_paths": 1,
        "min_path_length": 2,
        "radius_match": 0.5,
        "length_deviation": 0.05,
        "max_attempts": 1,
        "random_seed": 11,
    }
    config_path = _write_config(tmp_path, values, data_dim=3)
    ground_truth = _three_d_target_graph()
    prediction = _three_d_prediction_graph()

    assert ground_truth.number_of_edges() == 7
    assert prediction.number_of_edges() == 4
    assert len(nx.shortest_path(ground_truth, (1, 1, 0), (1, 1, 4))) == 3
    assert len(nx.shortest_path(prediction, (1, 1, 0), (1, 1, 4))) == 5
    result = TLTS(config_path).score(ground_truth, prediction)

    assert result == {
        "tlts_correct": 0.0,
        "tlts_tooshort": 0.0,
        "tlts_toolong": 1.0,
        "tlts_infeasible": 0.0,
    }


def test_identifier_nodes_use_pos_coordinates(tmp_path: Path):
    graph = nx.Graph()
    nodes = [10, 11, 12, 13]
    graph.add_nodes_from(
        (node, {"pos": (index, 0)}) for index, node in enumerate(nodes)
    )
    graph.add_edges_from(zip(nodes, nodes[1:]))
    paths = _extract_gt_paths(
        graph,
        N=1,
        min_path_length=2,
        random_state=np.random.default_rng(7),
        max_attempts=100,
    )

    assert paths
    assert isinstance(paths[0]["s_gt"], tuple)
    assert isinstance(paths[0]["t_gt"], tuple)


def test_coordinate_tuple_identifiers_are_preserved():
    graph = _line_graph(8)
    paths = _extract_gt_paths(
        graph,
        N=1,
        min_path_length=2,
        random_state=np.random.default_rng(5),
        max_attempts=100,
    )

    assert paths
    assert paths[0]["s_gt"] in graph
    assert paths[0]["t_gt"] in graph


def test_score_rejects_non_graph_inputs(tmp_path: Path):
    metric = TLTS(_write_config(tmp_path, TLTS_CONFIG, data_dim=2))
    with pytest.raises(TypeError, match="NetworkX graph"):
        metric.score(np.zeros((4, 4), dtype=bool), _line_graph(4))


def test_yaml_schema_rejects_missing_unknown_and_wrong_types(tmp_path: Path):
    missing = dict(TLTS_CONFIG)
    del missing["random_seed"]
    with pytest.raises(ValueError, match="missing keys"):
        TLTS(_write_config(tmp_path, missing, data_dim=2))

    unknown = dict(TLTS_CONFIG)
    unknown["threshold"] = 0.5
    with pytest.raises(ValueError, match="unknown keys"):
        TLTS(_write_config(tmp_path, unknown, data_dim=2))

    wrong_type = dict(TLTS_CONFIG)
    wrong_type["num_paths"] = "24"
    with pytest.raises(ValueError, match="num_paths"):
        TLTS(_write_config(tmp_path, wrong_type, data_dim=2))


def test_low_level_graph_scorer_returns_four_fractions():
    graph = _line_graph(8)
    paths = _extract_gt_paths(
        graph,
        N=8,
        min_path_length=2,
        random_state=np.random.default_rng(3),
        max_attempts=100,
    )
    result = _toolong_tooshort_score(
        paths,
        graph,
        radius_match=0.5,
        length_deviation=0.05,
    )

    assert result[0] == len(paths) > 0
    assert result[1:5] == (1.0, 0.0, 0.0, 0.0)
