"""Two-world sequential likelihood test and universal adaptive information lower bound.

ONLY for the two *known* cyclic preference matrices W+ and W- in cycle_gap.py.
SPRT and change-of-measure are established methods, not novel by themselves.
"""
import math
import numpy as np

WPLUS=np.array([[0.,.5,-.2],[-.5,0.,.5],[.2,-.5,0.]])
WMINUS=np.array([[0.,-.3,.6],[.3,0.,-.3],[-.6,.3,0.]])
EDGES=[(0,1),(0,2),(1,2)]


def bern_kl(p,q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


def universal_lower(delta=.05):
    """Applies to ANY delta-correct adaptive edge chooser with finite E[T]."""
    if not 0<delta<.5:raise ValueError('delta')
    dp=max(bern_kl((1+WPLUS[i,j])/2,(1+WMINUS[i,j])/2) for i,j in EDGES)
    dm=max(bern_kl((1+WMINUS[i,j])/2,(1+WPLUS[i,j])/2) for i,j in EDGES)
    kldecision=bern_kl(1-delta,delta)
    return {'all_adaptive_Eplus_lower':kldecision/dp,
            'all_adaptive_Eminus_lower':kldecision/dm,
            'max_div_plus':dp,'max_div_minus':dm}


def sequential_expected_upper(delta=.05):
    """Wald bound for fixed most-diagnostic edge (0,1)."""
    if not 0<delta<.5:raise ValueError('delta')
    p,q=.75,.35;A=math.log(1/delta)
    Dp=bern_kl(p,q);Dm=bern_kl(q,p)
    up=math.log(p/q); down=math.log((1-p)/(1-q))
    vp=(A+up)/Dp;vm=(A-down)/Dm
    lower=universal_lower(delta)
    return {'Eplus_upper':vp,'Eminus_upper':vm,
            'competitive_plus':vp/lower['all_adaptive_Eplus_lower'],
            'competitive_minus':vm/lower['all_adaptive_Eminus_lower'],
            'pplus':p,'pminus':q,'threshold':A}


def run_one(seed=0,world=+1,delta=.05,max_labels=20000):
    if world not in (-1,+1) or not 0<delta<.5:raise ValueError('world or delta')
    p=.75 if world==1 else .35
    pos=math.log(.75/.35);neg=math.log(.25/.65);A=math.log(1/delta)
    rng=np.random.default_rng(seed); loglr=0.
    for n in range(1,max_labels+1):
        loglr+=pos if rng.random()<p else neg
        if loglr>=A:return {'n':n,'selected':1,'correct':world==1}
        if loglr<=-A:return {'n':n,'selected':-1,'correct':world==-1}
    return {'n':max_labels,'selected':0,'correct':False}


def experiment(seeds=10000,delta=.05):
    out={'delta':delta,'universal_adaptive_lower':universal_lower(delta),
         'sequential_analytic_upper':sequential_expected_upper(delta),'families':{}}
    for sign in [-1,+1]:
        rows=[run_one(seed=20261116+z,world=sign,delta=delta) for z in range(seeds)]
        out['families'][str(sign)]={'runs':seeds,
            'errors':sum(x['selected']!=sign for x in rows),
            'mean_labels':float(np.mean([x['n'] for x in rows])),
            'median_labels':float(np.median([x['n'] for x in rows])),
            'max_labels':max(x['n'] for x in rows)}
    return out


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    pa=argparse.ArgumentParser();pa.add_argument('--seeds',type=int,default=10000)
    pa.add_argument('--out',default='results/two_world_sequential.json')
    args=pa.parse_args();data=experiment(args.seeds)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))
