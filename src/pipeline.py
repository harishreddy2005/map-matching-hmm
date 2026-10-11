"""Shared preparation, cached local data, and feature arrays."""
from pathlib import Path
import pickle
import numpy as np
from .data_loader import ROOT, load_dataset
from .candidates import generate
from .features import emissions, transitions

def prepare(rebuild=False):
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    cache = results/'prepared.pkl'
    if cache.exists() and not rebuild:
        # Local cache produced by this program only; never load untrusted pickle.
        with cache.open('rb') as f:
            return pickle.load(f)
    print('Loading benchmark and aligning route labels...', flush=True)
    data = load_dataset()
    print(data.counts, flush=True)
    candidates, distances, fallback = generate(data)
    em = [emissions(data, t, c, distances[t]) for t, c in enumerate(candidates)]
    discrepancy = [None] + [transitions(data, t, candidates[t-1], candidates[t], set())[:, :, 1]
                            for t in range(1, len(candidates))]
    payload = data, candidates, distances, fallback, em, discrepancy
    with cache.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload

def transition_function(candidates, discrepancy, offset=0):
    def features(k, visited):
        prev, curr = candidates[k+offset-1], candidates[k+offset]
        backtrack = np.broadcast_to(~np.isin(curr, list(visited)), (len(prev),len(curr))).astype(float).copy()
        backtrack[prev[:, None] == curr[None, :]] = 1
        return np.stack([backtrack, discrepancy[k+offset]], axis=-1)
    return features
