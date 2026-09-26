# apls_tlts

apls_tlts is a standalone package for graph-based APLS and TLTS scoring. All
metric code required at runtime lives in this directory.

## Public API

TLTS supports `parameters.num_paths: all` for exhaustive endpoint-pair scoring.
Set `max_attempts: null` and `random_seed: null` in that mode. Every unordered
pair of distinct vertices within a connected ground-truth component contributes
one unweighted shortest path when it meets `min_path_length` (vertex count).
Alternative paths for the same pair are not enumerated. Correct, too-short,
too-long, and infeasible fractions share the full eligible-pair denominator.
When no pair is eligible, the public score returns zero for the first three
fractions and one for infeasible. Integer `num_paths` retains seeded sampling.
Exhaustive evaluation can examine n*(n-1)/2 pairs in a connected n-node graph.

Both metrics load every result-affecting parameter from a YAML file when they
are constructed. A score call accepts two prebuilt undirected NetworkX graphs
and no mask, tensor, threshold, or channel arguments:

    import networkx as nx

    from apls_tlts import APLS, TLTS

    ground_truth = nx.Graph()
    for index in range(11):
        ground_truth.add_node(index, pos=(float(index), 0.0))
    ground_truth.add_edges_from((index, index + 1) for index in range(10))

    prediction = ground_truth.copy()

    apls_2d_score = APLS("apls_tlts/configs/apls.yaml").score(
        ground_truth, prediction
    )
    tlts_2d_scores = TLTS("apls_tlts/configs/tlts.yaml").score(
        ground_truth, prediction
    )

    apls_3d = APLS("apls_tlts/configs/apls_3d.yaml")
    tlts_3d = TLTS("apls_tlts/configs/tlts_3d.yaml")

The graph coordinate dimension is selected by YAML. APLS and TLTS both accept
2-D and 3-D graphs. Each node must either carry a finite numeric pos sequence
or itself be a finite numeric coordinate tuple. The coordinate length must
equal data_dim in the selected YAML file. Node identifiers may be arbitrary
hashable values when pos is present.

APLS converts each edge into a finite Euclidean segment. Missing edge lengths
are derived from the endpoint coordinates, and supplied finite non-negative
edge lengths are preserved for route weighting. Its 3-D distances include the
third coordinate during edge length, projection, snapping, and path
comparison. TLTS uses coordinates for endpoint matching and unweighted
shortest-path hop counts for route-length comparison in both dimensions.

The metric parameter files are:

- configs/apls.yaml for 2-D APLS.
- configs/apls_3d.yaml for 3-D APLS.
- configs/tlts.yaml for 2-D TLTS.
- configs/tlts_3d.yaml for 3-D TLTS.

The common loader rejects missing or unknown root keys and wrong root scalar
types. Each metric validates its parameter keys, types, ranges, and supported
dimensions. Change a value in YAML explicitly when a run needs a different
policy; no parameter is filled in by the package.

To run the focused metric, dimension, and graph-perturbation checks from the
project root:

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
      apls_tlts/test_apls.py \
      apls_tlts/test_tlts.py \
      apls_tlts/test_graph_perturbations.py

The package has no root-level packaging metadata. Install the dependencies
listed in requirements.txt, then use the project-root checkout on PYTHONPATH
or import it from code running at the repository root.
