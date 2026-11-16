"""Cycle-sensitive KL safe-improvement decision problem.

This small *two-hypothesis* example proves a narrow Borda-information
separation. It does NOT demonstrate world-first novelty over unrestricted
adaptive preference algorithms, and it is not a trained LLM experiment.

Copyright 2026 Ahmad Mohammadi. Research prototype.
"""
from __future__ import annotations
import argparse
import json
import math
import random

B = (1/3, 1/3, 1/3)
KAPPA = 0.10
THRESHOLD = 0.07
CYCLE = 0.40
A = 0.10
DELTA = 0.05
H = (0, 1)

def matrix(cycle_sign: int):
    if cycle_sign not in (-1, 1):
        raise ValueError("cycle_sign must be +/-1")
    c = CYCLE * cycle_sign
    s = (A, 0.0, -A)
    w = [[s[i]-s[j] for j in range(3)] for i in range(3)]
    for i,j in ((0,1),(1,2),(2,0)):
        w[i][j] += c
        w[j][i] -= c
    assert all(abs(w[i][j])<=1 for i in range(3) for j in range(3))
    assert (w[0][1]>0)==(w[1][2]>0)==(w[2][0]>0) # all three edges agree in sign, either orientation
    return w

def score(w):
    return tuple(sum(w[i][j]*B[j] for j in range(3)) for i in range(3))

def kl(q):
    return sum(qi*math.log(qi/bi) for qi,bi in zip(q,B) if qi>0)

def gains(q,w):
    d=[q[i]-B[i] for i in range(3)]
    return tuple(sum(d[i]*w[i][j] for i in range(3)) for j in H)

def margin(q,w):
    return min(gains(q,w))

def binary_kl(x,p=1/3):
    if x<=0:return math.log(1/(1-p))
    if x>=1:return math.log(1/p)
    return x*math.log(x/p)+(1-x)*math.log((1-x)/(1-p))

def min_q0(kappa=KAPPA):
    if not 0<=kappa<math.log(1.5):
        raise ValueError("supported kappa is below log(3/2)")
    a,b=0.,1/3
    for _ in range(80):
        m=(a+b)/2
        if binary_kl(m)>kappa:a=m
        else:b=m
    return (a+b)/2

def no_world_upper_bound(kappa=KAPPA):
    """For W_minus, (g_0/4+3g_1/4)=.3*(1/3-q_0).
    Thus min(g_0,g_1)<=.3*(1/3-q_0)<=this upper bound.
    The second inequality uses data processing for KL.
    """
    return 0.3*(1/3-min_q0(kappa))

def plus_witness():
    return (0.5,1/6,1/3)

def needed_labels(delta=DELTA):
    """Hoeffding on Y in {-1,+1}, threshold midway between means."""
    if not 0<delta<1: raise ValueError("delta must be in (0,1)")
    midpoint=0.10
    separation=min(abs(matrix(+1)[0][1]-midpoint),abs(matrix(-1)[0][1]-midpoint))
    return math.ceil(2*math.log(1/delta)/separation**2)

def predict_from_labels(wins,n):
    if n<=0 or not 0<=wins<=n:raise ValueError('invalid counts')
    return +1 if 2*wins/n-1 > .1 else -1

def run_trials(repeats=10000,delta=DELTA,seed=20261116):
    n=needed_labels(delta)
    out={}
    for sign in (-1,1):
        w=matrix(sign)
        p=(1+w[0][1])/2
        rng=random.Random(seed+sign*101)
        errors=0
        for _ in range(repeats):
            wins=sum(rng.random()<p for __ in range(n))
            errors+=predict_from_labels(wins,n)!=sign
        out[str(sign)]={'mean_edge_payoff':w[0][1], 'n':n,
                       'errors':errors,'trials':repeats,'error_rate':errors/repeats}
    return {'baseline_borda_scores_plus':score(matrix(+1)),
            'baseline_borda_scores_minus':score(matrix(-1)),
            'kl_budget':KAPPA,'required_margin':THRESHOLD,
            'plus_witness_kl':kl(plus_witness()),
            'plus_witness_margin':margin(plus_witness(),matrix(+1)),
            'minus_upper_bound':no_world_upper_bound(),
            'delta':delta,'labels_for_binary_case':n,
            'per_world_error_Hoeffding_bound':math.exp(-n*0.4**2/2),
            'worlds':out,
            'interpretation':'Only 2 known hypotheses; not a universal unknown-W safe certificate.'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261116)
    parser.add_argument('--out',default='cycle_result.json')
    args=parser.parse_args()
    if args.trials<=0:raise ValueError('trials must be positive')
    result=run_trials(args.trials,seed=args.seed)
    with open(args.out,'w',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
