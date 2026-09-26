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


def _snapshot_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return (
            "array",
            str(value.dtype),
            tuple(value.shape),
            _snapshot_value(value.tolist()),
        )
    if isinstance(value, dict):
        return tuple((key, _snapshot_value(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _graph_snapshot(graph: nx.MultiGraph) -> tuple[object, ...]:
    return (
        tuple(
            (node, _snapshot_value(dict(data))) for node, data in graph.nodes(data=True)
        ),
        tuple(
            (u, v, key, _snapshot_value(dict(data)))
            for u, v, key, data in graph.edges(keys=True, data=True)
        ),
        _snapshot_value(dict(graph.graph)),
    )


def test_owned_relabel_matches_copy_iteration_order_and_preserves_source() -> None:
    graph = nx.MultiGraph()
    graph.graph["metadata"] = {"frame": "retained"}
    graph.add_node("before", coord=(0.0, 0.0), pos=(0.0, 0.0))
    graph.add_node("rename", coord=(1.0, 0.0), pos=(1.0, 0.0))
    graph.add_node("after", coord=(2.0, 0.0), pos=(2.0, 0.0))
    graph.add_node("tail", coord=(3.0, 0.0), pos=(3.0, 0.0))
    graph.add_edge(
        "after",
        "tail",
        key="later-insertion",
        length=1.0,
        geometry=apls_core.EuclideanSegment((2.0, 0.0), (3.0, 0.0)),
    )
    graph.add_edge(
        "before",
        "rename",
        key="earlier-iteration",
        length=1.0,
        geometry=apls_core.EuclideanSegment((0.0, 0.0), (1.0, 0.0)),
    )
    graph.add_edge(
        "rename",
        "after",
        key="parallel-a",
        length=1.0,
        geometry=apls_core.EuclideanSegment((1.0, 0.0), (2.0, 0.0)),
    )
    graph.add_edge(
        "rename",
        "after",
        key="parallel-b",
        length=1.0,
        geometry=apls_core.EuclideanSegment((1.0, 0.0), (2.0, 0.0)),
    )
    graph.add_edge(
        "rename",
        "rename",
        key="self-loop",
        length=0.0,
        geometry=apls_core.EuclideanSegment((1.0, 0.0), (1.0, 0.0)),
    )
    source_before = _graph_snapshot(graph)
    expected = nx.relabel_nodes(graph.copy(), {"rename": "selected"}, copy=True)
    actual = apls_core._relabel_node_in_place_preserving_order(
        graph.copy(), "rename", "selected"
    )

    assert _graph_snapshot(actual) == _graph_snapshot(expected)
    assert _graph_snapshot(graph) == source_before
    assert all(actual._adj[u][v] is actual._adj[v][u] for u, v in actual.edges())


def test_bulk_insertion_preserves_tie_order_and_public_inputs() -> None:
    graph = nx.MultiGraph()
    graph.graph["frame"] = "full-image"
    coordinates = {
        "a": (0.0, 0.0),
        "b": (10.0, 0.0),
        "c": (0.0, 2.0),
        "d": (10.0, 2.0),
    }
    for node, position in coordinates.items():
        graph.add_node(node, coord=position, pos=position)
    graph.add_edge(
        "c",
        "d",
        key=0,
        length=10.0,
        geometry=apls_core.EuclideanSegment(coordinates["c"], coordinates["d"]),
    )
    graph.add_edge(
        "a",
        "b",
        key=0,
        length=10.0,
        geometry=apls_core.EuclideanSegment(coordinates["a"], coordinates["b"]),
    )
    source_before = _graph_snapshot(graph)
    tied_edge, tied_distance, _ = apls_core.get_closest_edge_from_G(
        graph, (0.0, 1.0), nearby_nodes=set(), verbose=False
    )
    assert tied_edge == ["a", "b", 0]
    assert tied_distance == 1.0

    controls = [["selected_a", (0.0, 1.0)], ["selected_b", (10.0, 1.0)]]
    expected = graph.copy()
    for node_id, point in controls:
        expected, _, _, _ = apls_core.insert_point_into_G(
            expected,
            point,
            node_id=node_id,
            max_distance_meters=2.0,
            dist_close_node=0.1,
            nearby_nodes=set(),
            allow_renaming=True,
            weight="length",
            verbose=False,
            super_verbose=False,
            zero_length_tolerance=1.0e-12,
        )
    actual, inserted, skipped = apls_core.insert_control_points(
        graph,
        controls,
        max_distance_meters=2.0,
        dist_close_node=0.1,
        allow_renaming=True,
        weight="length",
        verbose=False,
        super_verbose=False,
        zero_length_tolerance=1.0e-12,
    )

    assert _graph_snapshot(actual) == _graph_snapshot(expected)
    assert _graph_snapshot(graph) == source_before
    assert inserted == [(0.0, 0.0), (10.0, 0.0)]
    assert skipped == [(0.0, 0.0), (10.0, 0.0)]

    public_result, _, _, _ = apls_core.insert_point_into_G(
        graph,
        (0.0, 1.0),
        node_id="public_selected",
        max_distance_meters=2.0,
        dist_close_node=0.1,
        nearby_nodes=set(),
        allow_renaming=True,
        weight="length",
        verbose=False,
        super_verbose=False,
        zero_length_tolerance=1.0e-12,
    )
    assert "a" in graph
    assert "public_selected" not in graph
    assert "a" not in public_result
    assert _graph_snapshot(graph) == source_before


def test_cached_segment_geometry_matches_uncached_2d_and_3d_formulas() -> None:
    cases = [
        ((1.0, -2.0), (8.0, 3.0), (4.0, 6.0), 2.75),
        ((1.0, -2.0, 4.0), (8.0, 3.0, -1.0), (4.0, 6.0, 3.0), 2.75),
        ((2.0, 1.0), (2.0, 1.0), (5.0, 4.0), 0.0),
    ]
    for start, end, point, interpolation_distance in cases:
        segment = apls_core.EuclideanSegment(start, end)
        start_array = np.asarray(segment.start)
        direction = np.subtract(segment.end, segment.start)
        expected_length = float(np.linalg.norm(direction))
        point_array = np.asarray(tuple(point), dtype=float)
        squared_length = float(np.dot(direction, direction))
        if squared_length == 0.0:
            projected_distance = 0.0
        else:
            fraction = float(np.dot(np.subtract(point_array, start_array), direction))
            fraction /= squared_length
            fraction = min(1.0, max(0.0, fraction))
            projected_distance = fraction * expected_length
        interpolation_fraction = (
            0.0
            if expected_length == 0.0
            else min(1.0, max(0.0, interpolation_distance / expected_length))
        )
        expected_interpolated = tuple(
            float(value) for value in start_array + interpolation_fraction * direction
        )
        projected_fraction = (
            0.0
            if expected_length == 0.0
            else min(1.0, max(0.0, projected_distance / expected_length))
        )
        expected_projection = tuple(
            float(value) for value in start_array + projected_fraction * direction
        )
        expected_distance = float(
            np.linalg.norm(np.subtract(tuple(point), expected_projection))
        )

        assert segment.length == expected_length
        assert segment.project(point) == projected_distance
        assert segment.interpolate(interpolation_distance) == expected_interpolated
        assert segment.distance(point) == expected_distance
        assert not segment._start_array.flags.writeable
        assert not segment._direction.flags.writeable


@pytest.mark.parametrize("near_end", [False, True])
@pytest.mark.parametrize("allow_renaming", [False, True])
@pytest.mark.parametrize("dist_close_node", [0.0, 10.0])
def test_control_point_within_split_tolerance_uses_exact_endpoint(
    near_end: bool, allow_renaming: bool, dist_close_node: float,
) -> None:
    graph = nx.MultiGraph()
    graph.add_node("left", coord=(0.0, 0.0))
    graph.add_node("right", coord=(100.0, 0.0))
    graph.add_edge(
        "left", "right", length=100.0,
        geometry=apls_core.EuclideanSegment((0.0, 0.0), (100.0, 0.0)),
    )
    x = 100.0 - 5e-13 if near_end else 5e-13
    endpoint = "right" if near_end else "left"
    expected_point = graph.nodes[endpoint]["coord"]
    result, _, inserted, _ = apls_core.insert_point_into_G(
        graph, (x, 0.0), node_id="control",
        max_distance_meters=1.0, dist_close_node=dist_close_node,
        nearby_nodes=set(), allow_renaming=allow_renaming, weight="length",
        verbose=False, super_verbose=False, zero_length_tolerance=1e-12,
    )
    assert inserted == expected_point
    assert result.nodes["control"]["coord"] == expected_point
    opposite = "left" if near_end else "right"
    assert nx.shortest_path_length(result, "control", opposite, weight="length") == 100.0
    if allow_renaming:
        assert endpoint not in result
        assert result.number_of_edges() == 1
    else:
        assert result["control"][endpoint][0]["length"] == 0.0
        assert result["control"][endpoint][0]["geometry"].length == 0.0
        assert result.number_of_edges() == 2


def test_repeated_control_insertion_preserves_reversed_undirected_geometry() -> None:
    graph = nx.MultiGraph()
    graph.add_node("right", coord=(100.0, 0.0))
    graph.add_node("left", coord=(0.0, 0.0))
    graph.add_edge(
        "left", "right", length=100.0,
        geometry=apls_core.EuclideanSegment((0.0, 0.0), (100.0, 0.0)),
    )
    for index, x in enumerate((25.0, 75.0, 50.0, 100.0)):
        graph, _, _, _ = apls_core.insert_point_into_G(
            graph, (x, 0.0), node_id=f"control_{index}",
            max_distance_meters=1.0, dist_close_node=0.0,
            nearby_nodes=set(), allow_renaming=False, weight="length",
            verbose=False, super_verbose=False, zero_length_tolerance=0.0,
        )
        for u, v, data in graph.edges(data=True):
            geometry = data["geometry"]
            assert {geometry.start, geometry.end} == {
                graph.nodes[u]["coord"], graph.nodes[v]["coord"]
            }
        assert nx.shortest_path_length(graph, "left", "right", weight="length") == 100.0
        assert graph.nodes[f"control_{index}"]["coord"] == (x, 0.0)
