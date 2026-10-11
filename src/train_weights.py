"""Train on time-ordered blocks; no test labels enter fitting or selection."""
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from .data_loader import ROOT
from .pipeline import prepare, transition_function

def fit_weights(X, y, groups):
    unique = np.unique(groups)
    fold_results = []
    for C in [.1, 1., 10.]:
        scores = []
        for train, valid in TimeSeriesSplit(n_splits=10, gap=15).split(unique):
            ti, vi = np.isin(groups, unique[train]), np.isin(groups, unique[valid])
            if len(np.unique(y[ti])) < 2 or len(np.unique(y[vi])) < 2:
                continue
            scale = StandardScaler().fit(X[ti])
            model = LinearSVC(penalty='l1', dual=False, C=C, class_weight='balanced', max_iter=20000,
                              random_state=0).fit(scale.transform(X[ti]), y[ti])
            scores.append(float(f1_score(y[vi], model.predict(scale.transform(X[vi])), zero_division=0)))
        fold_results.append({'C': C, 'mean_f1': float(np.mean(scores)), 'folds': scores})
    C = max(fold_results, key=lambda r:r['mean_f1'])['C']
    scale = StandardScaler().fit(X)
    model = LinearSVC(penalty='l1', dual=False, C=C, class_weight='balanced', max_iter=20000,
                      random_state=0).fit(scale.transform(X), y)
    coefficients = model.coef_[0]/scale.scale_
    weights = np.maximum(coefficients, 0)
    fallback = not np.any(weights)
    if fallback:
        weights = np.ones(X.shape[1])
    weights /= weights.sum()
    return weights.tolist(), {'C': C, 'cv': fold_results, 'raw_coefficients': coefficients.tolist(),
                             'positive_fraction': float(np.mean(y)), 'examples':len(y),
                             'fallback_uniform': fallback, 'intercept_used_in_decoder': False}

def main():
    data, candidates, distances, fallback, em, discrepancy = prepare()
    ntrain = int(.6*len(candidates))
    X, y, groups = [], [], []
    for t in range(ntrain):
        X.extend(em[t]); y.extend(candidates[t] == data.truth[t]); groups.extend([t]*len(candidates[t]))
    ew, einfo = fit_weights(np.array(X), np.array(y,dtype=int), np.array(groups))
    print('Emission weights:', ew, flush=True)
    X, y, groups = [], [], []
    tf = transition_function(candidates, discrepancy)
    for t in range(1, ntrain):
        visited = set(data.truth[max(0,t-15):t])
        feat = tf(t,visited)
        labels = (candidates[t-1][:,None] == data.truth[t-1]) & (candidates[t][None,:] == data.truth[t])
        X.extend(feat.reshape(-1,2)); y.extend(labels.ravel()); groups.extend([t]*labels.size)
    tw, tinfo = fit_weights(np.array(X), np.array(y,dtype=int), np.array(groups))
    payload = {'emission':ew, 'transition':tw, 'emission_training':einfo, 'transition_training':tinfo,
               'training_end':ntrain, 'validation_end':int(.7*len(candidates)), 'sigma_metres':10,
               'rho_metres':50, 'split':'60% train / 10% validation / 30% test',
               'score_conversion':'positive raw-space coefficients, L1 normalised; intercept discarded; epsilon floor',
               'backtrack_training':'15-observation reference history; inference uses committed window history'}
    (ROOT/'results/weights.json').write_text(json.dumps(payload,indent=2))
    print('Transition weights:',tw,flush=True)

if __name__ == '__main__':
    main()
