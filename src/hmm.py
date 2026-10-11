import numpy as np

def decode(candidates, emissions, transitions, T=10, N=5, emission_weights=None,
           transition_weights=None, ablate=None):
    """Fixed-lag, finite-window Viterbi; every output is irrevocable.

    Committed states that overlap the window are clamped to their actual time
    indices, rather than clamping the first state to a different time's state.
    Remaining N points are flushed at EOF and marked separately in evaluation.
    Scores are positive potentials, not calibrated HMM probabilities.
    """
    if T < 1 or N < 0 or N >= T:
        raise ValueError('Require T >= 1 and 0 <= N < T')
    ew = np.array(emission_weights if emission_weights is not None else [.8, .15, .05])
    tw = np.array(transition_weights if transition_weights is not None else [.2, .8])
    if ablate is not None:
        (ew if ablate < 3 else tw)[ablate if ablate < 3 else ablate-3] = 0
    logem = [np.log(np.maximum(f @ ew, 1e-12)) for f in emissions]
    committed = []
    visited = set()
    last_path = []
    for t in range(len(candidates)):
        start = max(0, t-T+1)
        visited = set(committed[start:])
        delta = logem[start].copy()
        if start < len(committed):
            delta[candidates[start] != committed[start]] = -np.inf
        back = []
        for k in range(start+1, t+1):
            features = transitions(k, visited)
            scores = delta[:, None] + np.log(np.maximum(features @ tw, 1e-12))
            parents = scores.argmax(axis=0)
            delta = scores[parents, np.arange(len(candidates[k]))] + logem[k]
            if k < len(committed):
                delta[candidates[k] != committed[k]] = -np.inf
            back.append(parents)
        idx = int(delta.argmax())
        last_path = [int(candidates[t][idx])]
        for k in range(t-1, start-1, -1):
            idx = int(back[k-start][idx])
            last_path.append(int(candidates[k][idx]))
        last_path.reverse()
        if t >= N:
            output = last_path[t-N-start]
            committed.append(output)
            visited.add(output)
        # Retain only committed states within the current finite window.
        visited = set(committed[max(0, start):])
    if len(committed) < len(candidates):
        committed.extend(last_path[len(committed)-start:])
    return np.array(committed, dtype=int)
