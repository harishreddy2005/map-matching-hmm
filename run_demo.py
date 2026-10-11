"""Run the trained matcher and save a fully offline demonstration."""
import argparse
import json
from pathlib import Path
from src.data_loader import ROOT
from src.pipeline import prepare
from src.evaluate import match_range, metrics
from src.visualize import render

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--T',type=int,default=10)
    parser.add_argument('--N',type=int,default=5)
    parser.add_argument('--slice',type=int,nargs=2,metavar=('START','STOP'))
    args=parser.parse_args()
    weights_path=ROOT/'results/weights.json'
    if not weights_path.exists():
        parser.error('Train first: python -m src.train_weights')
    weights=json.loads(weights_path.read_text())
    data,candidates,distances,fallback,em,discrepancy=prepare()
    start,stop=args.slice or (weights['validation_end'],len(candidates))
    if not 0<=start<stop<=len(candidates):parser.error('Invalid slice boundaries')
    if not 0<=args.N<args.T:parser.error('Require 0 <= N < T')
    pred,runtime=match_range(candidates,em,discrepancy,weights,start,stop,args.T,args.N)
    print(json.dumps({'T':args.T,'N':args.N,**metrics(data,pred,start,args.N),'matcher_seconds':runtime},indent=2))
    render(data,pred,start,'demo_map.html','demo_map.png')
    print('Open results/demo_map.html (fully offline).')

if __name__=='__main__':main()
