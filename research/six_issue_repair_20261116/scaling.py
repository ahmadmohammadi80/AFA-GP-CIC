"""Exact matched computation for one-step Beta-binomial lookahead.

Original naive: for each e, outcome v, recalculate every candidate's full
E-dimensional certificate: O(E^2*B*P*H). Vectorized update: O(E*B*P*H).
Both still perform O(E*B) beta-mixture interval root inversions (cached).
"""
from pathlib import Path
import sys, time,tracemalloc,json
import numpy as np
from scipy.stats import betabinom
from repair_core import problem
from anytime_mixture import edge_cs


def make_inputs(K,P=3):
    b=np.ones(K)/K;Q=[]
    for j in range(P):
        q=b.copy();q[j%K]+=.025;q[(j+1)%K]-=.025
        Q.append(q)
    H=np.eye(K);E=K*(K-1)//2
    A=np.zeros((P,K,E));edges=[(i,j) for i in range(K) for j in range(i+1,K)]
    for p,q in enumerate(Q):
        d=q-b
        for e,(i,j) in enumerate(edges):
            A[p,:,e]=d[i]*H[:,j]-d[j]*H[:,i]
    rng=np.random.default_rng(247)
    N=rng.integers(20,60,size=E)
    wins=np.array([rng.binomial(n,.5) for n in N])
    return A,N,wins


def scan(A,N,wins,delta=.05,block=4,optimized=True):
    P,H,E=A.shape
    z=[edge_cs(int(w),int(n),delta/E) for w,n in zip(wins,N)]
    lo=np.array([x[0] for x in z]);hi=np.array([x[1] for x in z])
    current=np.where(A>=0,A*lo,A*hi)
    base=current.sum(axis=2) # P,H
    scores=[]
    for e in range(E):
        xs=np.arange(block+1)
        prob=betabinom.pmf(xs,block,wins[e]+.5,N[e]-wins[e]+.5)
        if optimized:
            nextlo=np.empty(block+1);nexthi=np.empty(block+1)
            for j,y in enumerate(xs):
                nextlo[j],nexthi[j]=edge_cs(int(wins[e]+y),int(N[e]+block),delta/E)
            repl=np.where(A[:,:,e,None]>=0,A[:,:,e,None]*nextlo[None,None,:],
                          A[:,:,e,None]*nexthi[None,None,:])
            new=base[:,:,None]-current[:,:,e,None]+repl
            val=np.max(np.min(new,axis=1),axis=0)
        else:
            vals=[]
            for y in xs:
                l,h=edge_cs(int(wins[e]+y),int(N[e]+block),delta/E)
                ll=lo.copy();hh=hi.copy();ll[e]=l;hh[e]=h
                new=np.where(A>=0,A*ll,A*hh).sum(axis=2)
                vals.append(np.max(np.min(new,axis=1)))
            val=np.asarray(vals)
        scores.append(float(prob@val))
    return np.array(scores)


def scaling(dims=(5,10,15,20),repeats=3,P=3,block=4):
    rows=[]
    for K in dims:
        A,N,wins=make_inputs(K,P)
        rec={'K':K,'E':len(N),'P':P,'H':K,'block':block}
        for mode,fast in [('naive',False),('vectorized',True)]:
            scan(A,N,wins,block=block,optimized=fast) # warm the common CS cache
            times=[];peaks=[]
            for rep in range(repeats):
                tracemalloc.start(); t=time.perf_counter()
                v=scan(A,N,wins,block=block,optimized=fast)
                times.append(time.perf_counter()-t)
                _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                peaks.append(peak/1024/1024)
            rec[mode+'_time_median_s']=float(np.median(times))
            rec[mode+'_python_peak_mib']=float(np.max(peaks))
            if not fast:ref=v
            else:
                err=float(np.max(abs(ref-v)))
                rec['max_absolute_value_error']=err
                assert err<1e-10
        rec['speedup']=rec['naive_time_median_s']/rec['vectorized_time_median_s']
        rows.append(rec)
    return rows


def charts(rows,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=Path(outdir);d.mkdir(parents=True,exist_ok=True)
    for key,ylab,filename in [('time_median_s','Seconds (median)','scaling_time.png'),
                              ('python_peak_mib','Tracemalloc peak MiB (excludes native NumPy memory)','scaling_memory.png')]:
        fig,ax=plt.subplots(figsize=(7,4.4))
        for m in ('naive','vectorized'):
            ax.plot([r['K'] for r in rows],[r[m+'_'+key] for r in rows],marker='o',label=m)
        ax.set(xlabel='Number of actions K',ylabel=ylab,title='Exact one-step pair-query lookahead')
        ax.legend();fig.tight_layout();fig.savefig(d/filename,dpi=150);plt.close(fig)

if __name__=='__main__':
 rows=scaling()
 d=Path(__file__).parent
 (d/'scaling.json').write_text(json.dumps(rows,indent=2))
 charts(rows,d)
 print(json.dumps(rows,indent=2))