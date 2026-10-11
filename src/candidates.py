import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

def generate(dataset, rho=50.0):
    """All directed arcs within rho metres; nearest fallback is recorded."""
    tree = STRtree(dataset.lines)
    candidates, distances, fallback = [], [], []
    for xy in dataset.xy:
        point = Point(xy)
        arcs = np.sort(tree.query(point, predicate='dwithin', distance=rho))
        fallback.append(len(arcs) == 0)
        if not len(arcs):
            nearest = int(tree.nearest(point))
            # Include both directions if present for the fallback edge.
            arcs = np.flatnonzero(dataset.edge_ids == dataset.edge_ids[nearest])
        candidates.append(arcs)
        distances.append(np.array([point.distance(dataset.lines[i]) for i in arcs]))
    return candidates, distances, np.array(fallback)
