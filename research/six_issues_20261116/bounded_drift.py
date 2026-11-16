"""Anytime certificate for bounded-drift Bernoulli preference streams.

Assumption: each pair e at global time t has a predictable conditional
probability p_e(t); |p_e(t)-p_e(t-1)| <= rho for a KNOWN rho. The query schedule
may adapt to all past outcomes. The confidence bound is on the CURRENT mean
at any calendar time, not on a stationary historical average.

For observations on e at times tau_1,...,tau_n, Azuma-Hoeffding over the
bounded martingale differences of Y in {-1,1} gives the usual Hoeffding
anytime radius for the average conditional means. Drift bias at global
time t is <= 2*rho * (t - mean(tau_s)), capped by 2. Union across n and e.
Avoid using the stationary Beta-mixture martingale after drift!
"""
from __future__ import annotations
import math,argparse,json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

def bound(wins,n,sum_times,now,rho,delta,E):
    if not(0<delta<1) or E<1 or rho<0 or n<0 or wins<0 or wins>n or now<0:
        raise ValueError("parameters")
    if n==0:return (-1.,1.)
    if sum_times/n>now+1e-8:raise ValueError("future sample time")
    r=math.sqrt(2*math.log(E*math.pi**2*n*n/(3*delta))/n)
    drift=min(2.,2*rho*(now-sum_times/n))
    m=2*wins/n-1
    return max(-1.,m-r-drift),min(1.,m+r+drift)

def experiment(seed,cycles=500,k=5,delta=.05,amp=.12,period=400):
    if k<3 or amp<=0 or period<=0:raise ValueError("shape")
    edges=list(combinations(range(k),2));E=len(edges)
    # In mean scale |mu(t+1)-mu(t)| <= amp/period,
    # in Bernoulli probability scale rho=amp/(2*period).
    rho=amp/(2*period)
    rng=np.random.default_rng(seed);n=np.zeros(E,int);s=np.zeros(E,int);ts=np.zeros(E,float)
    rng2=np.random.default_rng(seed+991871)
    phases=rng2.uniform(0,2*np.pi,E)
    pot=np.linspace(.22,-.22,k)
    base=np.array([.6*(pot[i]-pot[j]) for i,j in edges])
    if np.any(np.abs(base)+amp>=1):raise ValueError("probability outside 0..1")
    breaks=0;min_halfwidth=1.;widths=[]
    for t in range(1,cycles+1):
        # Adaptive pair-query: query least sampled most recent ambiguous edge,
        # incorporating previous observations; tie broken pseudorandomly.
        if t<=E:
            e=t-1
        else:
            estimates=(2*s/np.maximum(n,1)-1)
            scores=(1-np.abs(estimates)) / np.sqrt(n+.1)
            e=int(np.argmax(scores))
        mu=base[e]+amp*np.sin(t/period+phases[e])
        s[e]+=rng.binomial(1,(1+mu)/2);n[e]+=1;ts[e]+=t
        # Evaluate simultaneous CURRENT coverage on all e, including those
        # not sampled at calendar time t (drift bias grows for stale pairs).
        for ix in range(E):
            lo,hi=bound(int(s[ix]),int(n[ix]),ts[ix],t,rho,delta,E)
            truth=base[ix]+amp*np.sin(t/period+phases[ix])
            if not lo-1e-12<=truth<=hi+1e-12:breaks+=1
            if n[ix]:widths.append(hi-lo)
    return {"seed":seed,"covered_all_pairs_all_times":breaks==0,
            "violations":breaks,"mean_width":float(np.mean(widths)),
            "last_total_queries":int(n.sum()),"max_rho":rho}

def audit(seeds=40,offset=20261116,cycles=500):
    rows=[experiment(offset+i,cycles) for i in range(seeds)]
    success=sum(x["covered_all_pairs_all_times"] for x in rows)
    # This estimates actual empirical simultaneous-path coverage; it does not
    # prove 95% empirical coverage from only 40 paths.
    low=float(binomtest(success,len(rows)).proportion_ci(confidence_level=.95,method="exact").low)
    return {"seeds":seeds,"cycles":cycles,"all_path_coverage_count":success,
            "path_coverage_rate":success/seeds,"exact_binomial_lower_95":low,
            "mean_interval_width":float(np.mean([x["mean_width"] for x in rows])),
            "rows":rows}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",type=int,default=40)
    ap.add_argument("--cycles",type=int,default=500);ap.add_argument("--out",default="results/drift_coverage.json")
    a=ap.parse_args();v=audit(a.seeds,cycles=a.cycles)
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
    print(json.dumps({k:z for k,z in v.items() if k!="rows"},indent=2))
