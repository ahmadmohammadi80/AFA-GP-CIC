"""Known epoch boundaries, bounded within-epoch preference variation.

For each edge e and epoch j, conditionally independent signs Y_t have
predictable conditional means mu_e(t) in [-1,1]. Assume for any times u,v
within one epoch |mu_e(u)-mu_e(v)| <= rho. Then the Hoeffding-Azuma
confidence radius around observed means plus rho covers CURRENT mu_e(t),
for every sample count and known epoch (via a double union bound).

This is not a certificate for arbitrary unbounded or unknown drift.
"""
from math import log,pi,sqrt,sin,cos
from itertools import combinations
import numpy as np
from repair_core import problem


def confidence(wins,n,epoch,delta,E,rho):
    if n==0:return -1.,1.
    delta_j=delta*6/(pi*pi*(epoch+1)**2)
    rad=sqrt(2*log(E*pi*pi*n*n/(3*delta_j))/n)
    mu=2*wins/n-1
    return max(-1.,mu-rad-rho),min(1.,mu+rad+rho)


def drift_truth(base_W,t,epoch_size,epoch,amp=.018):
    W=base_W.copy()
    phase=2*pi*(t/epoch_size)+epoch*.37
    # In each epoch, each edge changes in a bounded range <=2*amp.
    for i,j,offset in [(0,2,0.),(2,3,1.1),(3,0,2.4)]:
        d=amp*sin(phase+offset)
        W[i,j]+=d;W[j,i]-=d
    assert np.max(np.abs(W))<=1.
    return W


def experiment(seeds=40,epochs=3,epoch_size=2500,delta=.05,threshold=.015,amp=.018):
    b,Q,H,W0,edges,A,mu0=problem(cyclic=True);E=len(edges);rho=2*amp+1e-12
    out=[]
    for trial in range(seeds):
        rng=np.random.default_rng(20261116+trial)
        certified=0;coverall=True;bad_cert=0;rows=[]
        for j in range(epochs):
            N=np.zeros(E,int);wins=np.zeros(E,int)
            for t in range(epoch_size):
                e=t%E
                W=drift_truth(W0,t,epoch_size,j,amp)
                i,k=edges[e]
                wins[e]+=int(rng.random()<(1+W[i,k])/2)
                N[e]+=1
                if t%100==0 or t==epoch_size-1:
                    ci=np.array([confidence(int(w),int(n),j,delta,E,rho) for w,n in zip(wins,N)])
                    current=np.array([W[i,k] for i,k in edges])
                    coverall=coverall and bool(np.all(ci[:,0]<=current+1e-12) and np.all(current<=ci[:,1]+1e-12))
            # Only the final epoch assessment uses the displayed current W.
            lo=np.array([confidence(int(w),int(n),j,delta,E,rho)[0] for w,n in zip(wins,N)])
            hi=np.array([confidence(int(w),int(n),j,delta,E,rho)[1] for w,n in zip(wins,N)])
            low=np.where(A>=0,A*lo,A*hi).sum(axis=-1).min(axis=1)
            winner=int(low.argmax());is_cert=bool(low[winner]>threshold)
            true=float((A[winner]@np.array([W[i,k] for i,k in edges])).min())
            certified+=int(is_cert)
            bad_cert+=int(is_cert and true<=threshold)
            rows.append({'epoch':j,'certified':is_cert,'lower':float(low[winner]),'true':true})
        out.append({'trial':trial,'all_checked_intervals_cover':bool(coverall),
                    'certificates':certified,'false_certificates':bad_cert,'epochs':rows})
    return {'trials':seeds,'epochs':epochs,'epoch_size':epoch_size,'variation_bound_mu':rho,
            'all_epochs_covered_runs':sum(x['all_checked_intervals_cover'] for x in out),
            'total_certificates':sum(x['certificates'] for x in out),
            'false_certificates':sum(x['false_certificates'] for x in out),'rows':out,
            'scope':'known epoch boundaries; conditional Bernoulli labels; within-epoch variation <=rho'}

if __name__=='__main__':
 import json,sys
 n=int(sys.argv[1]) if len(sys.argv)>1 else 40
 data=experiment(seeds=n)
 from pathlib import Path
 Path(__file__).with_name('drift40.json').write_text(json.dumps(data,indent=2))
 print(json.dumps({k:v for k,v in data.items() if k!='rows'},indent=2))