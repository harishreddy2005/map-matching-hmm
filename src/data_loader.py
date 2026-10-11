from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import numpy as np
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Dataset:
    xy: np.ndarray
    timestamps: np.ndarray
    speed: np.ndarray
    heading: np.ndarray
    lines: list
    edge_ids: np.ndarray
    directions: np.ndarray
    limits: np.ndarray
    midpoints: np.ndarray
    truth: np.ndarray
    route: np.ndarray
    route_position: np.ndarray
    counts: dict

def load_dataset():
    """Parse original files, project longitude/latitude to UTM 10N.

    Speed/heading use only present and past readings (causal 5-point average).
    Route-constrained labels are derived proxies, not independent point truth.
    """
    project = Transformer.from_crs(4326, 32610, always_xy=True)
    with (ROOT / 'data/gps_data.txt').open() as f:
        rows = list(csv.reader(f, delimiter='\t'))[1:]
    lonlat = np.array([[float(r[3]), float(r[2])] for r in rows])
    xy = np.column_stack(project.transform(lonlat[:, 0], lonlat[:, 1]))
    timestamps = np.array([datetime.strptime(r[0]+' '+r[1], '%d-%b-%Y %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() for r in rows])
    velocity = np.zeros_like(xy)
    for i in range(1, len(xy)):
        j = max(0, i-5)
        velocity[i] = (xy[i]-xy[j]) / max(timestamps[i]-timestamps[j], 1)
    speed = np.linalg.norm(velocity, axis=1)
    heading = velocity / np.maximum(speed[:, None], 1e-12)
    lines, ids, directions, limits = [], [], [], []
    lookup = {}
    edge_count = 0
    with (ROOT / 'data/road_network.txt').open() as f:
        next(f)
        for row in csv.reader(f, delimiter='\t'):
            if not row:
                continue
            edge_count += 1
            edge, two_way, limit = int(row[0]), int(row[3]), float(row[4])
            coords = np.array(wkt.loads(row[6]).coords)
            if abs(coords[0, 0]) < 90 and abs(coords[0, 1]) > 90:
                coords = coords[:, ::-1]
            projected = np.column_stack(project.transform(coords[:, 0], coords[:, 1]))
            for direction in ([1, 0] if two_way else [1]):
                lookup[(edge, direction)] = len(lines)
                lines.append(LineString(projected if direction else projected[::-1]))
                ids.append(edge)
                directions.append(direction)
                limits.append(limit)
    with (ROOT / 'data/ground_truth_route.txt').open() as f:
        route_rows = list(csv.reader(f, delimiter='\t'))[1:]
    route = np.array([lookup[(int(r[0]), int(r[1]))] for r in route_rows if r], dtype=int)
    truth, route_position = align_route(xy, [lines[i] for i in route], route)
    midpoints = np.array([line.interpolate(.5, normalized=True).coords[0] for line in lines])
    return Dataset(xy, timestamps, speed, heading, lines, np.array(ids), np.array(directions),
                   np.array(limits), midpoints, truth, route, route_position,
                   {'gps_points': len(xy), 'road_edges': edge_count, 'directed_arcs': len(lines),
                    'route_edges': len(route), 'elapsed_seconds': float(timestamps[-1]-timestamps[0]),
                    'sampling_median_seconds': float(np.median(np.diff(timestamps))),
                    'sampling_gap_count': int(np.sum(np.diff(timestamps) != 1))})

def align_route(xy, route_lines, route):
    """Global monotone distance alignment, preserving repeated route occurrences.

    Allowed step: stay or advance up to 8 route edges. All route positions are
    initial candidates; a 1 m advance penalty breaks ties at shared junctions.
    No model features/learned predictions are used. This is offline label creation.
    """
    distances = np.array([[Point(p).distance(line) for line in route_lines] for p in xy])
    prev = distances[0].copy()
    back = np.zeros(distances.shape, dtype=np.int16)
    indices = np.arange(len(route))
    for t in range(1, len(xy)):
        options = np.full((9, len(route)), np.inf)
        options[0] = prev
        for jump in range(1, 9):
            options[jump, jump:] = prev[:-jump] + jump
        jump = options.argmin(axis=0)
        back[t] = indices-jump
        prev = distances[t] + options[jump, indices]
    positions = np.zeros(len(xy), dtype=int)
    positions[-1] = prev.argmin()
    for t in range(len(xy)-1, 0, -1):
        positions[t-1] = back[t, positions[t]]
    return route[positions], positions
