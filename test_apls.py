"""Behavior checks for the standalone graph-level APLS metric."""

from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from apls_tlts import apls_core
from apls_tlts.apls import APLS, _as_apls_graph


CONFIG_TEXT = """\
metric: apls
data_dim: 2
parameters:
  max_nodes: 64
  max_snap_dist: 4.0
  dist_close_node: 0.1
  allow_renaming: true
  select_intersections: false
  min_path_length: 2.0
  random_seed: 19
"""

CONFIG_3D_TEXT = """\
metric: apls
data_dim: 3
parameters:
  max_nodes: 64
  max_snap_dist: 4.0
  dist_close_node: 0.1
  allow_renaming: true
  select_intersections: false
  min_path_length: 2.0
  random_seed: 19
"""


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "apls.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _line_graph(node_count: int, missing_node: int) -> nx.Graph:
    graph = nx.Graph()
    for index in range(node_count):
        if index == missing_node:
            continue
        graph.add_node((float(index * 10), 0.0))
    nodes = list(graph.nodes())
    for left, right in zip(nodes, nodes[1:]):
        graph.add_edge(left, right, length=10.0)
    return graph


def _branch_graph() -> nx.Graph:
    graph = _line_graph(node_count=9, missing_node=-1)
    graph.add_node((40.0, 10.0))
    graph.add_node((40.0, 20.0))
    graph.add_edge((40.0, 0.0), (40.0, 10.0), length=10.0)
    graph.add_edge((40.0, 10.0), (40.0, 20.0), length=10.0)
    return graph


def _line_graph_3d(node_count: int, z_value: float, missing_node: int) -> nx.Graph:
    graph = nx.Graph()
    for index in range(node_count):
        if index == missing_node:
            continue
        position = (float(index * 10), 0.0, float(z_value))
        graph.add_node(position, pos=position)
    nodes = list(graph.nodes())
    for left, right in zip(nodes, nodes[1:]):
        graph.add_edge(left, right)
    return graph


def _branch_graph_3d() -> nx.Graph:
    graph = _line_graph_3d(node_count=3, z_value=0.0, missing_node=-1)
    graph.add_node((10.0, 0.0, 10.0), pos=(10.0, 0.0, 10.0))
    graph.add_edge((10.0, 0.0, 0.0), (10.0, 0.0, 10.0))
    return graph


def test_apls_scores_identical_tuple_coordinate_graphs(tmp_path: Path) -> None:
    metric = APLS(_config(tmp_path, CONFIG_TEXT))
    graph = _line_graph(node_count=9, missing_node=-1)

    score = metric.score(graph, graph.copy())

    assert score == pytest.approx(1.0, abs=1e-12)


def test_apls_detects_missing_graph_segment(tmp_path: Path) -> None:
    metric = APLS(_config(tmp_path, CONFIG_TEXT))
    target = _line_graph(node_count=9, missing_node=-1)
    proposal = _line_graph(node_count=9, missing_node=4)

    score = metric.score(target, proposal)

    assert 0.0 <= score < 1.0


def test_apls_empty_graph_semantics_match_metric_wrapper(tmp_path: Path) -> None:
    metric = APLS(_config(tmp_path, CONFIG_TEXT))
    empty = nx.Graph()
    nonempty = _line_graph(node_count=3, missing_node=-1)

    assert metric.score(empty, empty) == 1.0
    assert metric.score(empty, nonempty) == 0.0
    assert metric.score(nonempty, empty) == 0.0


def test_select_intersections_branch_path_runs_without_undefined_symbol(
    tmp_path: Path,
) -> None:
    config = CONFIG_TEXT.replace("select_intersections: false", "select_intersections: true")
    metric = APLS(_config(tmp_path, config))
    graph = _branch_graph()

    score = metric.score(graph, graph.copy())

    assert np.isfinite(score)


def test_seeded_sampling_does_not_use_global_random_state(tmp_path: Path) -> None:
    config = CONFIG_TEXT.replace("max_nodes: 64", "max_nodes: 4")
    metric = APLS(_config(tmp_path, config))
    target = _line_graph(node_count=14, missing_node=-1)
    proposal = _line_graph(node_count=14, missing_node=-1)

    random.seed(1)
    first = metric.score(target, proposal)
    random.seed(987654)
    second = metric.score(target, proposal)

    assert first == second


def test_apls_rejects_invalid_yaml_and_non_graph_inputs(tmp_path: Path) -> None:
    bad_dimension = CONFIG_TEXT.replace("data_dim: 2", "data_dim: 4")
    with pytest.raises(ValueError, match="data_dim=2 or data_dim=3"):
        APLS(_config(tmp_path, bad_dimension))

    unknown_key = CONFIG_TEXT.replace(
        "  random_seed: 19", "  random_seed: 19\n  unexpected: true"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        APLS(_config(tmp_path, unknown_key))

    metric = APLS(_config(tmp_path, CONFIG_TEXT))
    with pytest.raises(TypeError, match="NetworkX graphs"):
        metric.score(np.zeros((4, 4)), np.zeros((4, 4)))


def test_graph_validation_errors_are_not_converted_to_zero(tmp_path: Path) -> None:
    metric = APLS(_config(tmp_path, CONFIG_TEXT))
    invalid = nx.Graph()
    invalid.add_edge("left", "right", length=1.0)

    with pytest.raises(ValueError, match="2-dimensional"):
        metric.score(invalid, invalid)


def test_apls_rejects_directed_graphs(tmp_path: Path) -> None:
    metric = APLS(_config(tmp_path, CONFIG_TEXT))
    directed = nx.DiGraph()
    directed.add_node((0.0, 0.0))
    directed.add_node((10.0, 0.0))
    directed.add_edge((0.0, 0.0), (10.0, 0.0), length=10.0)

    with pytest.raises(ValueError, match="undirected"):
        metric.score(directed, directed)


def test_apls_3d_graph_cases(tmp_path: Path) -> None:
    metric = APLS(_config(tmp_path, CONFIG_3D_TEXT))
    target = _branch_graph_3d()
    deleted = _line_graph_3d(node_count=3, z_value=0.0, missing_node=-1)

    assert metric.score(target, target.copy()) == pytest.approx(1.0, abs=1e-12)
    assert metric.score(target, deleted) < 1.0
    assert metric.score(deleted, target) < 1.0


def test_apls_3d_uses_true_euclidean_edge_length_and_splitting() -> None:
    diagonal = nx.Graph()
    diagonal.add_node((0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0))
    diagonal.add_node((3.0, 4.0, 12.0), pos=(3.0, 4.0, 12.0))
    diagonal.add_edge((0.0, 0.0, 0.0), (3.0, 4.0, 12.0))
    converted_diagonal = _as_apls_graph(diagonal, "gt", 3)
    diagonal_edge = next(iter(converted_diagonal.edges(data=True)))[2]
    assert diagonal_edge["length"] == pytest.approx(13.0)
    assert diagonal_edge["geometry"].length == pytest.approx(13.0)

    vertical = nx.Graph()
    vertical.add_node((0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0))
    vertical.add_node((0.0, 0.0, 10.0), pos=(0.0, 0.0, 10.0))
    vertical.add_edge((0.0, 0.0, 0.0), (0.0, 0.0, 10.0))
    metric = APLS(Path(__file__).parent / "configs" / "apls_3d.yaml")
    assert metric.score(vertical, vertical.copy()) == pytest.approx(1.0)
    converted_vertical = _as_apls_graph(vertical, "gt", 3)
    split_graph, _, _ = apls_core.insert_control_points(
        converted_vertical,
        [["control", (0.0, 0.0, 5.0)]],
        max_distance_meters=1.0,
        dist_close_node=0.1,
        allow_renaming=False,
        weight="length",
        verbose=False,
        super_verbose=False,
        zero_length_tolerance=1e-12,
    )
    split_lengths = [
        data["length"]
        for _, _, _, data in split_graph.edges("control", keys=True, data=True)
    ]
    assert sorted(split_lengths) == pytest.approx([5.0, 5.0])


def test_apls_3d_z_separation_changes_score(tmp_path: Path) -> None:
    metric = APLS(_config(tmp_path, CONFIG_3D_TEXT))
    ground_truth = nx.Graph()
    point_a = (0.0, 0.0, 0.0)
    point_b = (3.0, 4.0, 12.0)
    ground_truth.add_node(point_a, pos=point_a)
    ground_truth.add_node(point_b, pos=point_b)
    ground_truth.add_edge(point_a, point_b)

    proposal = nx.Graph()
    point_p = (1.5, 2.0, 0.0)
    proposal.add_node(point_a, pos=point_a)
    proposal.add_node(point_p, pos=point_p)
    proposal.add_node(point_b, pos=point_b)
    proposal.add_edge(point_a, point_p)
    proposal.add_edge(point_p, point_b)

    assert metric.score(ground_truth, ground_truth.copy()) == pytest.approx(1.0)
    assert metric.score(ground_truth, proposal) < 1.0


def test_apls_rejects_both_dimension_mismatches(tmp_path: Path) -> None:
    metric_2d = APLS(_config(tmp_path, CONFIG_TEXT))
    metric_3d = APLS(_config(tmp_path, CONFIG_3D_TEXT))
    graph_2d = _line_graph(node_count=3, missing_node=-1)
    graph_3d = _line_graph_3d(node_count=3, z_value=0.0, missing_node=-1)

    with pytest.raises(ValueError, match="2-dimensional"):
        metric_2d.score(graph_3d, graph_3d)
    with pytest.raises(ValueError, match="3-dimensional"):
        metric_3d.score(graph_2d, graph_2d)


def test_apls_3d_config_file_is_usable() -> None:
    metric = APLS(Path(__file__).parent / "configs" / "apls_3d.yaml")
    graph = _line_graph_3d(node_count=3, z_value=0.0, missing_node=-1)

    assert metric.config.data_dim == 3
    assert metric.score(graph, graph.copy()) == pytest.approx(1.0)
