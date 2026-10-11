import numpy as np
from shapely.geometry import Point

def emissions(dataset, t, arcs, distances, sigma=10.):
    values = np.ones((len(arcs), 3))
    # Gaussian density, matching the reference formula (units: 1/metre).
    values[:, 0] = np.exp(-distances**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    for j, arc in enumerate(arcs):
        line = dataset.lines[arc]
        along = line.project(Point(dataset.xy[t]))
        a = np.array(line.interpolate(max(0, along-1)).coords[0])
        b = np.array(line.interpolate(min(line.length, along+1)).coords[0])
        tangent = (b-a)/max(np.linalg.norm(b-a), 1e-12)
        values[j, 1] = np.dot(tangent, dataset.heading[t]) if dataset.speed[t] >= 1 else 0
    values[:, 2] = ~((dataset.speed[t] > 17.8816) & (dataset.speed[t] >= 1.2*dataset.limits[arcs]))
    return values

def transitions(dataset, t, previous, current, visited):
    displacement = np.linalg.norm(dataset.xy[t]-dataset.xy[t-1])
    midpoint_dist = np.linalg.norm(dataset.midpoints[previous, None]-dataset.midpoints[current][None], axis=2)
    discrepancy = 1/(np.abs(displacement-midpoint_dist)+1)
    # A committed-history approximation keeps Viterbi first-order and causal.
    backtrack = np.broadcast_to(~np.isin(current, list(visited)), discrepancy.shape).astype(float).copy()
    backtrack[previous[:, None] == current[None, :]] = 1
    return np.stack([backtrack, discrepancy], axis=-1)
