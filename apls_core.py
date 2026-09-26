"""Dimension-aware graph path matching for the standalone APLS metric.

The same control-point and path-similarity algorithm is used for two- and
three-dimensional graphs. Edge snapping and edge splitting operate on finite
Euclidean segments instead of Shapely geometries, whose distance operations
ignore a third coordinate.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy import stats


def _point_array(point: Iterable[float]) -> np.ndarray:
    """Convert one coordinate sequence to a one-dimensional float array."""

    array = np.asarray(tuple(point), dtype=float)
    if array.ndim != 1 or len(array) not in {2, 3}:
        raise ValueError("APLS internal coordinates must be two- or three-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("APLS internal coordinates must be finite")
    return array


@dataclass(frozen=True)
class EuclideanSegment:
    """A straight segment with true two- or three-dimensional distances."""

    start: tuple[float, ...]
    end: tuple[float, ...]
    _start_array: np.ndarray = field(init=False, repr=False, compare=False)
    _direction: np.ndarray = field(init=False, repr=False, compare=False)
    _squared_length: float = field(init=False, repr=False, compare=False)
    _length: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        start_array = _point_array(self.start)
        end_array = _point_array(self.end)
        if start_array.shape != end_array.shape:
            raise ValueError("APLS segment endpoints must have the same dimension")
        start_array.setflags(write=False)
        direction = np.subtract(end_array, start_array)
        direction.setflags(write=False)
        object.__setattr__(self, "start", tuple(float(value) for value in start_array))
        object.__setattr__(self, "end", tuple(float(value) for value in end_array))
        object.__setattr__(self, "_start_array", start_array)
        object.__setattr__(self, "_direction", direction)
        object.__setattr__(self, "_squared_length", float(np.dot(direction, direction)))
        object.__setattr__(self, "_length", float(np.linalg.norm(direction)))

    @property
    def dimension(self) -> int:
        return len(self.start)

    @property
    def length(self) -> float:
        return self._length

    def project(self, point: Iterable[float]) -> float:
        """Return the clamped distance along the segment nearest ``point``."""

        point_array = _point_array(point)
        if self._squared_length == 0.0:
            return 0.0
        fraction = float(
            np.dot(np.subtract(point_array, self._start_array), self._direction)
        )
        fraction /= self._squared_length
        fraction = min(1.0, max(0.0, fraction))
        return fraction * self._length

    def interpolate(self, distance: float) -> tuple[float, ...]:
        """Return the point at a clamped distance along the segment."""

        segment_length = self._length
        if segment_length == 0.0:
            return self.start
        if distance <= 0.0:
            return self.start
        if distance >= segment_length:
            return self.end
        fraction = min(1.0, max(0.0, float(distance) / segment_length))
        point = self._start_array + fraction * self._direction
        return tuple(float(value) for value in point)

    def distance(self, point: Iterable[float]) -> float:
        """Return true Euclidean point-to-segment distance."""

        projected = self.interpolate(self.project(point))
        return float(np.linalg.norm(np.subtract(tuple(point), projected)))

    def split(self, distance: float) -> list["EuclideanSegment"]:
        """Split at a distance, retaining endpoint behavior."""

        if distance <= 0.0 or distance >= self.length:
            return [self]
        middle = self.interpolate(distance)
        return [EuclideanSegment(self.start, middle), EuclideanSegment(middle, self.end)]


def is_intersection(graph: nx.Graph, node: Any) -> bool:
    """Return whether ``node`` has at least three incident edges."""

    return graph.degree(node) >= 3


def _edge_records(graph: nx.Graph) -> Iterable[tuple[Any, Any, Any, dict[str, Any]]]:
    """Yield ``(u, v, key, data)`` records for either graph kind."""

    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
        return
    for u, v, data in graph.edges(data=True):
        yield u, v, 0, data


def _segment_from_edge(data: dict[str, Any]) -> EuclideanSegment:
    """Read one validated internal segment from an APLS edge."""

    if "geometry" not in data:
        raise ValueError("APLS edges must contain an internal Euclidean segment")
    segment = data["geometry"]
    if not isinstance(segment, EuclideanSegment):
        raise ValueError("APLS edge geometry must be an EuclideanSegment")
    return segment


def _finite_weight(data: dict[str, Any], weight: str) -> float:
    """Read one finite non-negative edge weight."""

    if weight not in data:
        raise ValueError(f"APLS edge is missing configured weight {weight!r}")
    value = data[weight]
    if isinstance(value, bool):
        raise ValueError(f"APLS edge weight {weight!r} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"APLS edge weight {weight!r} must be numeric") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"APLS edge weight {weight!r} must be finite and non-negative")
    return numeric


def get_closest_edge_from_G(
    graph: nx.Graph,
    point: tuple[float, ...],
    nearby_nodes: set[Any],
    verbose: bool,
) -> tuple[list[Any], float, EuclideanSegment]:
    """Find the closest edge using true N-dimensional distance."""

    distances: list[float] = []
    edges: list[list[Any]] = []
    geometries: list[EuclideanSegment] = []
    for u, v, key, data in _edge_records(graph):
        if nearby_nodes and u not in nearby_nodes and v not in nearby_nodes:
            continue
        geometry = _segment_from_edge(data)
        if geometry.length <= 0.0:
            continue
        geometries.append(geometry)
        distances.append(geometry.distance(point))
        edges.append([u, v, key])

    if len(edges) == 0 and nearby_nodes:
        for u, v, key, data in _edge_records(graph):
            geometry = _segment_from_edge(data)
            if geometry.length <= 0.0:
                continue
            geometries.append(geometry)
            distances.append(geometry.distance(point))
            edges.append([u, v, key])
    if len(edges) == 0:
        raise ValueError("cannot snap a control point onto a graph without edges")

    closest_index = int(np.argmin(np.asarray(distances)))
    return edges[closest_index], distances[closest_index], geometries[closest_index]


def _edge_data(graph: nx.Graph, u: Any, v: Any, key: Any) -> dict[str, Any]:
    """Read one edge data mapping from either graph kind."""

    if graph.is_multigraph():
        return graph.edges[u, v, key]
    return graph.edges[u, v]


def insert_point_into_G(
    graph: nx.MultiGraph,
    point: tuple[float, ...],
    node_id: Any,
    max_distance_meters: float,
    dist_close_node: float,
    nearby_nodes: set[Any],
    allow_renaming: bool,
    weight: str,
    verbose: bool,
    super_verbose: bool,
    zero_length_tolerance: float,
) -> tuple[nx.MultiGraph, dict[str, Any], tuple[float, ...], tuple[float, ...]]:
    """Insert a control point by snapping or splitting its closest edge."""

    return _insert_point_into_G(
        graph,
        point,
        node_id,
        max_distance_meters,
        dist_close_node,
        nearby_nodes,
        allow_renaming,
        weight,
        verbose,
        super_verbose,
        zero_length_tolerance,
        rename_owned_graph_in_place=False,
    )


def _relabel_node_in_place_preserving_order(
    graph: nx.MultiGraph,
    old_node: Any,
    new_node: Any,
) -> nx.MultiGraph:
    """Rename one absent target on a privately owned graph, keeping iteration order.

    NetworkX's copy-based relabel is the public insertion behavior. The bulk
    insertion path already owns a graph copy, so rebuilding that graph for
    every renamed endpoint is unnecessary. Re-keying its ordered internal
    dictionaries in place gives the same node and edge iteration order as the
    copy-based operation while preserving node and edge attributes.
    """

    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError("ordered in-place relabeling requires an undirected MultiGraph")
    if old_node not in graph:
        raise ValueError(f"node to relabel is absent: {old_node!r}")
    if new_node in graph:
        raise ValueError(f"new node ID already exists: {new_node!r}")

    node_entries = [
        (new_node if node == old_node else node, properties)
        for node, properties in graph._node.items()
    ]
    adjacency_entries = [
        (new_node if node == old_node else node, adjacent)
        for node, adjacent in graph._adj.items()
    ]
    for adjacent in graph._adj.values():
        if old_node in adjacent:
            renamed_neighbors = [
                (new_node if neighbor == old_node else neighbor, edge_keys)
                for neighbor, edge_keys in adjacent.items()
            ]
            adjacent.clear()
            adjacent.update(renamed_neighbors)

    graph._node.clear()
    graph._node.update(node_entries)
    graph._adj.clear()
    graph._adj.update(adjacency_entries)
    return graph


def _insert_point_into_G(
    graph: nx.MultiGraph,
    point: tuple[float, ...],
    node_id: Any,
    max_distance_meters: float,
    dist_close_node: float,
    nearby_nodes: set[Any],
    allow_renaming: bool,
    weight: str,
    verbose: bool,
    super_verbose: bool,
    zero_length_tolerance: float,
    rename_owned_graph_in_place: bool,
) -> tuple[nx.MultiGraph, dict[str, Any], tuple[float, ...], tuple[float, ...]]:
    """Insert one point with an explicit ownership policy for endpoint renames."""

    best_edge, minimum_distance, best_geometry = get_closest_edge_from_G(
        graph,
        point,
        nearby_nodes=nearby_nodes,
        verbose=super_verbose,
    )
    if minimum_distance > max_distance_meters:
        return graph, {}, tuple(), tuple()

    u, v, key = best_edge
    if node_id in graph:
        return graph, {}, tuple(), tuple()

    # Undirected edge iteration need not preserve stored geometry orientation.
    start = tuple(graph.nodes[u]["coord"])
    end = tuple(graph.nodes[v]["coord"])
    if best_geometry.start == end and best_geometry.end == start:
        best_geometry = EuclideanSegment(start, end)
    elif best_geometry.start != start or best_geometry.end != end:
        raise ValueError("APLS edge geometry endpoints do not match its graph nodes")

    projected_distance = best_geometry.project(point)
    projected_point = best_geometry.interpolate(projected_distance)
    endpoint: Any = None
    endpoint_distance: float | None = None
    for candidate in (u, v):
        candidate_position = tuple(graph.nodes[candidate]["coord"])
        candidate_distance = float(
            np.linalg.norm(np.subtract(projected_point, candidate_position))
        )
        endpoint_is_exact = candidate_distance <= zero_length_tolerance
        endpoint_is_close = candidate_distance < dist_close_node
        if (endpoint_is_exact or endpoint_is_close) and (
            endpoint_distance is None or candidate_distance < endpoint_distance
        ):
            endpoint = candidate
            endpoint_distance = candidate_distance

    # A projection within the split tolerance of either endpoint cannot form
    # two non-degenerate segments. Use that endpoint exactly, including at the
    # far end where distance along the edge is large.
    endpoint_is_degenerate = (
        endpoint_distance is not None
        and endpoint_distance <= zero_length_tolerance
    )
    if endpoint_is_degenerate:
        projected_point = tuple(graph.nodes[endpoint]["coord"])

    node_properties: dict[str, Any] = {
        "coord": projected_point,
        "pos": np.asarray(projected_point, dtype=float),
    }

    if endpoint is not None and (
        endpoint_is_degenerate or projected_distance <= dist_close_node
    ):
        if allow_renaming:
            endpoint_properties = dict(graph.nodes[endpoint])
            if rename_owned_graph_in_place:
                renamed = _relabel_node_in_place_preserving_order(
                    graph, endpoint, node_id
                )
            else:
                renamed = nx.relabel_nodes(graph, {endpoint: node_id}, copy=True)
            return renamed, endpoint_properties, projected_point, projected_point
        graph.add_node(node_id, **node_properties)
        edge_data = copy.deepcopy(_edge_data(graph, u, v, key))
        edge_data[weight] = 0.0
        edge_data["geometry"] = EuclideanSegment(
            projected_point,
            tuple(graph.nodes[endpoint]["coord"]),
        )
        graph.add_edge(node_id, endpoint, **edge_data)
        return graph, node_properties, projected_point, projected_point

    edge_data = copy.deepcopy(_edge_data(graph, u, v, key))
    original_weight = _finite_weight(edge_data, weight)
    split_segments = best_geometry.split(projected_distance)
    if len(split_segments) != 2:
        if endpoint is None:
            raise ValueError("APLS could not split the closest edge at the control point")
        graph.add_node(node_id, **node_properties)
        edge_data[weight] = 0.0
        edge_data["geometry"] = EuclideanSegment(
            projected_point,
            tuple(graph.nodes[endpoint]["coord"]),
        )
        graph.add_edge(node_id, endpoint, **edge_data)
        return graph, node_properties, projected_point, projected_point

    first_segment, second_segment = split_segments
    original_length = best_geometry.length
    first_weight = original_weight * (first_segment.length / original_length)
    second_weight = original_weight * (second_segment.length / original_length)
    if (
        first_segment.length <= zero_length_tolerance
        or second_segment.length <= zero_length_tolerance
    ):
        raise ValueError("APLS control-point insertion produced a zero-length edge")

    graph.remove_edge(u, v, key)
    graph.add_node(node_id, **node_properties)
    first_data = copy.deepcopy(edge_data)
    first_data["geometry"] = first_segment
    first_data[weight] = first_weight
    second_data = copy.deepcopy(edge_data)
    second_data["geometry"] = second_segment
    second_data[weight] = second_weight
    graph.add_edge(u, node_id, **first_data)
    graph.add_edge(node_id, v, **second_data)
    return graph, node_properties, projected_point, projected_point


def insert_control_points(
    graph: nx.MultiGraph,
    control_points: list[list[Any]],
    max_distance_meters: float,
    dist_close_node: float,
    allow_renaming: bool,
    weight: str,
    verbose: bool,
    super_verbose: bool,
    zero_length_tolerance: float,
) -> tuple[nx.MultiGraph, list[tuple[float, ...]], list[tuple[float, ...]]]:
    """Insert all control points into a graph copy."""

    output = graph.copy()
    inserted_positions: list[tuple[float, ...]] = []
    skipped_positions: list[tuple[float, ...]] = []
    for control_point in control_points:
        node_id, point = control_point
        output, _, inserted, skipped = _insert_point_into_G(
            output,
            tuple(point),
            node_id=node_id,
            max_distance_meters=max_distance_meters,
            dist_close_node=dist_close_node,
            nearby_nodes=set(),
            allow_renaming=allow_renaming,
            weight=weight,
            verbose=verbose,
            super_verbose=super_verbose,
            zero_length_tolerance=zero_length_tolerance,
            rename_owned_graph_in_place=True,
        )
        if len(inserted) > 0:
            inserted_positions.append(inserted)
        if len(skipped) > 0:
            skipped_positions.append(skipped)
    return output, inserted_positions, skipped_positions


def _selected_nodes(graph: nx.Graph, select_intersections: bool) -> list[Any]:
    """Select all nodes or only intersection nodes in graph order."""

    nodes = list(graph.nodes())
    if select_intersections:
        return [node for node in nodes if is_intersection(graph, node)]
    return nodes


def _sample_nodes(
    nodes: list[Any],
    max_nodes: int,
    rng: random.Random,
) -> list[Any]:
    """Sample control nodes with an explicit local RNG."""

    sample_size = min(max_nodes, len(nodes))
    return rng.sample(nodes, sample_size)


def _pairwise_lengths(
    graph: nx.Graph,
    sources: list[Any],
    source_set: set[Any],
    weight: str,
) -> dict[Any, dict[Any, float]]:
    """Compute weighted paths restricted to sampled control nodes."""

    lengths: dict[Any, dict[Any, float]] = {}
    for source in sources:
        if source not in graph:
            continue
        paths = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
        lengths[source] = {
            node: float(distance)
            for node, distance in paths.items()
            if node in source_set
        }
    return lengths


def make_graphs_yuge(
    ground_truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    weight: str,
    max_nodes: int,
    max_snap_dist: float,
    dist_close_node: float,
    allow_renaming: bool,
    select_intersections: bool,
    seed: int,
    verbose: bool,
    super_verbose: bool,
) -> tuple[
    nx.MultiGraph,
    nx.MultiGraph,
    nx.MultiGraph,
    nx.MultiGraph,
    list[list[Any]],
    list[list[Any]],
    dict[Any, dict[Any, float]],
    dict[Any, dict[Any, float]],
    dict[Any, dict[Any, float]],
    dict[Any, dict[Any, float]],
]:
    """Match graph control points and return native and snapped path tables."""

    rng = random.Random(seed)
    ground_truth_native = ground_truth.to_undirected()
    proposal_native = proposal.to_undirected()

    selected_ground_truth = _selected_nodes(ground_truth_native, select_intersections)
    selected_proposal = _selected_nodes(proposal_native, select_intersections)
    sampled_ground_truth = _sample_nodes(selected_ground_truth, max_nodes, rng)
    sampled_proposal = _sample_nodes(selected_proposal, max_nodes, rng)
    sampled_ground_truth_set = set(sampled_ground_truth)
    sampled_proposal_set = set(sampled_proposal)

    control_points_ground_truth = [
        [node, tuple(ground_truth_native.nodes[node]["coord"])]
        for node in sampled_ground_truth
    ]
    control_points_proposal = [
        [node, tuple(proposal_native.nodes[node]["coord"])]
        for node in sampled_proposal
    ]

    ground_truth_lengths_native = _pairwise_lengths(
        ground_truth_native,
        sampled_ground_truth,
        sampled_ground_truth_set,
        weight,
    )
    proposal_lengths_native = _pairwise_lengths(
        proposal_native,
        sampled_proposal,
        sampled_proposal_set,
        weight,
    )

    insertion_parameters = {
        "max_distance_meters": max_snap_dist,
        "dist_close_node": dist_close_node,
        "allow_renaming": allow_renaming,
        "weight": weight,
        "verbose": super_verbose,
        "super_verbose": super_verbose,
        "zero_length_tolerance": 1e-12,
    }
    proposal_with_ground_truth, _, _ = insert_control_points(
        proposal_native,
        control_points_ground_truth,
        **insertion_parameters,
    )
    ground_truth_with_proposal, _, _ = insert_control_points(
        ground_truth_native,
        control_points_proposal,
        **insertion_parameters,
    )

    ground_truth_lengths_prime = _pairwise_lengths(
        ground_truth_with_proposal,
        sampled_proposal,
        sampled_proposal_set,
        weight,
    )
    proposal_lengths_prime = _pairwise_lengths(
        proposal_with_ground_truth,
        sampled_ground_truth,
        sampled_ground_truth_set,
        weight,
    )
    return (
        ground_truth_native,
        proposal_native,
        ground_truth_with_proposal,
        proposal_with_ground_truth,
        control_points_ground_truth,
        control_points_proposal,
        ground_truth_lengths_native,
        proposal_lengths_native,
        ground_truth_lengths_prime,
        proposal_lengths_prime,
    )


def single_path_metric(len_gt: float, len_prop: float, diff_max: float) -> float:
    """Compute the bounded relative error for one route."""

    if len_gt <= 0.0:
        return 0.0
    if len_prop < 0.0 and len_gt > 0.0:
        return float(diff_max)
    diff_raw = abs(len_gt - len_prop) / len_gt
    return float(min(diff_max, diff_raw))


def path_sim_metric(
    all_pairs_lengths_gt: dict[Any, dict[Any, float]],
    all_pairs_lengths_prop: dict[Any, dict[Any, float]],
    control_nodes: list[Any],
    min_path_length: float,
    diff_max: float,
    missing_path_len: float,
    normalize: bool,
) -> tuple[float, list[float], list[list[Any]], dict[Any, dict[Any, float]]]:
    """Compare route lengths for one direction of the APLS score."""

    diffs: list[float] = []
    routes: list[list[Any]] = []
    difference_by_start: dict[Any, dict[Any, float]] = {}
    gt_start_nodes = set(all_pairs_lengths_gt)
    proposal_start_nodes = set(all_pairs_lengths_prop)
    if len(gt_start_nodes) == 0:
        return 0.0, [], [], {}

    good_nodes = list(all_pairs_lengths_gt) if len(control_nodes) == 0 else control_nodes
    good_node_set = set(good_nodes)
    for start_node in good_nodes:
        differences_for_start: dict[Any, float] = {}
        if start_node not in gt_start_nodes:
            if start_node in proposal_start_nodes:
                for end_node in all_pairs_lengths_prop[start_node]:
                    differences_for_start[end_node] = diff_max
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
            difference_by_start[start_node] = differences_for_start
            continue

        paths_gt = all_pairs_lengths_gt[start_node]
        if start_node not in proposal_start_nodes:
            for end_node, length_gt in paths_gt.items():
                if end_node != start_node and end_node in good_node_set:
                    differences_for_start[end_node] = diff_max
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
            difference_by_start[start_node] = differences_for_start
            continue

        paths_proposal = all_pairs_lengths_prop[start_node]
        for end_node in set(paths_gt).intersection(good_node_set):
            length_gt = paths_gt[end_node]
            if length_gt < min_path_length:
                continue
            length_proposal = (
                paths_proposal[end_node]
                if end_node in paths_proposal
                else missing_path_len
            )
            difference = single_path_metric(
                length_gt,
                length_proposal,
                diff_max=diff_max,
            )
            differences_for_start[end_node] = difference
            diffs.append(difference)
            routes.append([start_node, end_node])
        difference_by_start[start_node] = differences_for_start

    if len(diffs) == 0:
        return 0.0, [], [], {}
    difference_total = float(np.sum(diffs))
    if normalize:
        return 1.0 - difference_total / len(diffs), diffs, routes, difference_by_start
    return difference_total, diffs, routes, difference_by_start


def compute_apls_metric(
    all_pairs_lengths_gt_native: dict[Any, dict[Any, float]],
    all_pairs_lengths_prop_native: dict[Any, dict[Any, float]],
    all_pairs_lengths_gt_prime: dict[Any, dict[Any, float]],
    all_pairs_lengths_prop_prime: dict[Any, dict[Any, float]],
    control_points_gt: list[list[Any]],
    control_points_prop: list[list[Any]],
    min_path_length: float,
) -> tuple[float, float, float]:
    """Compute the harmonic mean of both directed path similarities."""

    if len(all_pairs_lengths_gt_native) == 0 or len(all_pairs_lengths_prop_native) == 0:
        return 0.0, 0.0, 0.0

    control_nodes_gt = [point[0] for point in control_points_gt]
    score_gt_to_prop, _, _, _ = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes_gt,
        min_path_length=min_path_length,
        diff_max=1.0,
        missing_path_len=-1.0,
        normalize=True,
    )
    control_nodes_prop = [point[0] for point in control_points_prop]
    score_prop_to_gt, _, _, _ = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes_prop,
        min_path_length=min_path_length,
        diff_max=1.0,
        missing_path_len=-1.0,
        normalize=True,
    )
    if (
        score_gt_to_prop <= 0.0
        or score_prop_to_gt <= 0.0
        or np.isnan(score_gt_to_prop)
        or np.isnan(score_prop_to_gt)
    ):
        return 0.0, score_gt_to_prop, score_prop_to_gt
    total = float(stats.hmean([score_gt_to_prop, score_prop_to_gt]))
    if np.isnan(total):
        total = 0.0
    return total, score_gt_to_prop, score_prop_to_gt


def score_graphs(
    ground_truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    max_nodes: int,
    max_snap_dist: float,
    dist_close_node: float,
    allow_renaming: bool,
    select_intersections: bool,
    min_path_length: float,
    seed: int,
) -> float:
    """Score two prepared graphs with explicit algorithm parameters."""

    result = make_graphs_yuge(
        ground_truth,
        proposal,
        weight="length",
        max_nodes=max_nodes,
        max_snap_dist=max_snap_dist,
        dist_close_node=dist_close_node,
        allow_renaming=allow_renaming,
        select_intersections=select_intersections,
        seed=seed,
        verbose=False,
        super_verbose=False,
    )
    (
        _,
        _,
        _,
        _,
        control_points_gt,
        control_points_prop,
        lengths_gt_native,
        lengths_prop_native,
        lengths_gt_prime,
        lengths_prop_prime,
    ) = result
    score, _, _ = compute_apls_metric(
        lengths_gt_native,
        lengths_prop_native,
        lengths_gt_prime,
        lengths_prop_prime,
        control_points_gt,
        control_points_prop,
        min_path_length=min_path_length,
    )
    return float(score)


__all__ = [
    "EuclideanSegment",
    "compute_apls_metric",
    "is_intersection",
    "make_graphs_yuge",
    "score_graphs",
]
