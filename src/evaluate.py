import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point
from .data_loader import ROOT
from .pipeline import prepare, transition_function
from .hmm import decode

def match_range(candidates, em, discrepancy, weights, start, stop, T, N, ablate=None):
    begin = max(0,start-T)
    tic=time.perf_counter()
    pred=decode(candidates[begin:stop], em[begin:stop], transition_function(candidates,discrepancy,begin),
                T,N,weights['emission'],weights['transition'],ablate)
    runtime=time.perf_counter()-tic
    return pred[start-begin:], runtime

def metrics(data, prediction, start, N=0):
    truth=data.truth[start:start+len(prediction)]
    mismatch=prediction!=truth
    online=len(prediction)-N
    return {'arc_error':float(np.mean(mismatch)),
            'edge_error':float(np.mean(data.edge_ids[prediction]!=data.edge_ids[truth])),
            'online_arc_error':float(np.mean(mismatch[:online])) if online>0 else None,
            'flush_points':min(N,len(prediction)), 'points':len(prediction)}

def save_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    data,candidates,distances,fallback,em,discrepancy=prepare()
    weights=json.loads((ROOT/'results/weights.json').read_text())
    va,te=weights['training_end'],weights['validation_end']
    validation=[]
    for T in [5,10,15]:
        for N in range(T):
            pred,runtime=match_range(candidates,em,discrepancy,weights,va,te,T,N)
            validation.append({'T':T,'N':N,**metrics(data,pred,va,N)})
    best=min(validation,key=lambda r:(r['arc_error'],r['N'],r['T']))
    save_csv(ROOT/'results/validation.csv',validation)
    rows=[]
    for T in [5,10,15]:
        for N in range(T):
            pred,runtime=match_range(candidates,em,discrepancy,weights,te,len(candidates),T,N)
            row={'T':T,'N':N,**metrics(data,pred,te,N),'seconds_per_1000':runtime/len(pred)*1000}
            rows.append(row)
        print('Test sweep complete: T =',T,flush=True)
    save_csv(ROOT/'results/metrics.csv',rows)
    baseline=np.array([c[d.argmin()] for c,d in zip(candidates[te:],distances[te:])])
    baseline_metrics=metrics(data,baseline,te)
    ablations=[]
    for drop,name in [(None,'full'),(0,'without distance'),(1,'without heading'),(2,'without speed'),
                      (3,'without backtrack'),(4,'without discrepancy')]:
        pred,runtime=match_range(candidates,em,discrepancy,weights,te,len(candidates),best['T'],best['N'],drop)
        ablations.append({'feature':name,**metrics(data,pred,te,best['N'])})
        if drop is None:
            np.savez_compressed(ROOT/'results/predictions.npz',prediction=pred,truth=data.truth[te:],start=te)
    save_csv(ROOT/'results/ablation.csv',ablations)
    fig,ax=plt.subplots(figsize=(7,3.4))
    for T in [5,10,15]:
        r=[r for r in rows if r['T']==T]
        ax.plot([x['N'] for x in r],[100*x['arc_error'] for x in r],'-o',label=f'T = {T}',markersize=3)
    ax.axhline(100*baseline_metrics['arc_error'],ls='--',color='grey',label='Nearest segment (direction tie: forward)')
    ax.set(xlabel='Buffer N (observations; approximately seconds)',ylabel='Directed-arc proxy error (%)')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/error_vs_buffer.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3))
    ax.barh([a['feature'] for a in ablations],[a['arc_error']*100 for a in ablations],color='#2563eb')
    ax.set_xlabel('Directed-arc proxy error (%)');fig.tight_layout();fig.savefig(ROOT/'results/ablation.png',dpi=180);plt.close(fig)
    coverage=np.array([truth in c for truth,c in zip(data.truth,candidates)])
    selected=next(r for r in rows if r['T']==best['T'] and r['N']==best['N'])
    summary={'dataset':data.counts,'training_points':va,'validation_points':te-va,'test_points':len(candidates)-te,
             'candidate_coverage':float(coverage.mean()),'test_candidate_coverage':float(coverage[te:].mean()),
             'candidate_fallbacks':int(fallback.sum()),'mean_candidates':float(np.mean([len(c) for c in candidates])),
             'selected_by_validation':{'T':best['T'],'N':best['N'],'validation_arc_error':best['arc_error']},
             'selected_test':selected,'baseline_test':baseline_metrics,'ablations':ablations,
             'route_alignment_median_distance_metres':float(np.median([data.lines[int(a)].distance(Point(p)) for a,p in zip(data.truth,data.xy)])),
             'limitations':['Per-point labels are offline monotone distance alignment to published route, not independent GPS labels.',
                            'No shortest paths, connectivity enforcement, or route-length mismatch metric; disconnected predictions are possible.',
                            'SVM coefficients become uncalibrated positive potentials; negative coefficients clipped and intercept discarded.',
                            'End-of-stream flush has less future context; online-only error also reported.',
                            'Dataset includes sampling gaps; N is observation count, delay is only approximately N seconds.',
                            'Nearest-segment directed baseline breaks ties with forward direction; edge error is fairer for geometric baseline.']}
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':main()
