"""AFA: threshold-directed joint query and post-query policy selection.

This prototype evaluates strong anytime-beta-mixture with an optional one-step
threshold-crossing policy recourse gate. No claim of global originality.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
import numpy as np
from anytime_mixture import edge_cs,edge_coefficients,certificate
from itertools import combinations


def problem(cyclic=False):
    k=5;b=np.ones(k)/k
    q1=b.copy();q1[0]+=.08;q1[1]-=.08
    q2=b.copy();q2[2]+=.08;q2[3]-=.08
    Q=np.array([q1,q2]);H=np.eye(k)
    potential=np.array([.45,-.35,.25,-.15,-.2]);W=potential[:,None]-potential[None,:]
    for i,j in ((0,2),(2,3),(3,0)):
        W[i,j]+=.04;W[j,i]-=.04
    if cyclic:
        for i,j in ((2,3),(3,4),(4,2)):
            W[i,j]+=.5;W[j,i]-=.5
    E=list(combinations(range(k),2));A=np.array([edge_coefficients(b,q,H)[1] for q in Q])
    mu=np.array([W[i,j] for i,j in E]);return b,Q,H,W,E,A,mu


def greedy_scores(A,lo,hi,N):
    L=np.where(A>=0,A*lo,A*hi).sum(axis=-1).min(axis=1)
    winner=int(L.argmax())
    Ac=A[winner];mean=(lo+hi)/2
    worst=int(np.argmin(Ac@mean - np.abs(Ac)@((hi-lo)/2)))
    priority=np.abs(Ac[worst])*(hi-lo)/(np.sqrt(N)+.2)
    return winner,L,int(np.argmax(priority)) if priority.max()>0 else int(np.argmin(N))


def threshold_recourse_edge(A,lo,hi,wins,N,delta,threshold,greedy_edge):
    """Use threshold crossing probability only when an actual next-label certificate can cross.
    Conditional Jeffreys Beta posterior predictive is for PLANNING, not coverage.
    """
    E=len(N);old=np.where(A>=0,A*lo,A*hi).sum(axis=-1) # (P,H)
    best=-1.;picked=greedy_edge;trigger=False
    for e in range(E):
        # Exact no-crossing pruning: an opponent constraint that is below
        # threshold and unaffected by this edge makes any one-step
        # certificate for that candidate impossible.
        possible=np.all((old>threshold) | (np.abs(A[:,:,e])>1e-15),axis=1)
        if not np.any(possible):
            continue
        prob1=(wins[e]+.5)/(N[e]+1.)
        pacc=0.; expected_margin=0.
        oldpart=np.where(A[:,:,e]>=0,A[:,:,e]*lo[e],A[:,:,e]*hi[e])
        for y,pr in ((0,1-prob1),(1,prob1)):
            l,h=edge_cs(int(wins[e]+y),int(N[e]+1),delta/E)
            newpart=np.where(A[:,:,e]>=0,A[:,:,e]*l,A[:,:,e]*h)
            after=np.min(old-oldpart+newpart,axis=1).max()
            pacc+=pr*float(after>threshold)
            expected_margin+=pr*max(0,after-threshold)
        value=pacc + .01*expected_margin
        if value>best+1e-12:
            best=value;picked=e
    if best>0:
        return picked,True
    return greedy_edge,False


def simulate(seed,method='joint_threshold',budget=1510,delta=.05,threshold=.015,cyclic=False):
    if method not in ('joint_threshold','greedy'):
        raise ValueError(method)
    b,Q,H,W,E,A,mu=problem(cyclic=cyclic)
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(len(E))]
    N=np.ones(len(E),int);wins=np.array([s.binomial(1,(1+v)/2) for s,v in zip(streams,mu)])
    lo=np.array([edge_cs(int(w),1,delta/len(E))[0] for w in wins]);hi=np.array([edge_cs(int(w),1,delta/len(E))[1] for w in wins])
    triggers=0;changes=0
    for t in range(len(E),budget+1):
        win,L,e0=greedy_scores(A,lo,hi,N)
        if L[win]>threshold:
            return {'cert':True,'labels':int(t),'selected':int(win),'triggers':triggers,'changes':changes,
                    'true_gain':float(np.min(A[win]@mu)), 'kl':float(np.sum(Q[win]*np.log(Q[win]/b))),
                    'max_policy_width':float(np.max(np.sum(np.abs(A)*(hi-lo),axis=-1)))}
        if t==budget:break
        if method=='joint_threshold':
            e,on=threshold_recourse_edge(A,lo,hi,wins,N,delta,threshold,e0)
            triggers+=int(on);changes+=int(on and e!=e0)
        else:e=e0
        wins[e]+=streams[e].binomial(1,(1+mu[e])/2);N[e]+=1
        lo[e],hi[e]=edge_cs(int(wins[e]),int(N[e]),delta/len(E))
    w,L,_=greedy_scores(A,lo,hi,N)
    return {'cert':False,'labels':int(budget),'selected':int(w),'triggers':triggers,'changes':changes,
            'true_gain':float(np.min(A[w]@mu)), 'kl':float(np.sum(Q[w]*np.log(Q[w]/b))),
            'max_policy_width':float(np.max(np.sum(np.abs(A)*(hi-lo),axis=-1)))}


def experiment(seeds=40,offset=20261116):
    rows=[]
    for i in range(seeds):
        seed=offset+i
        d={m:simulate(seed,m) for m in ('joint_threshold','greedy')}
        rows.append(d)
    out={m:{'certs':sum(r[m]['cert'] for r in rows),'labels':sum(r[m]['labels'] for r in rows)/len(rows),
       'triggered':sum(r[m]['triggers'] for r in rows),'switched':sum(r[m]['changes'] for r in rows)}
       for m in ('joint_threshold','greedy')}
    return out,rows

if __name__=='__main__':
 import json,sys
 n=int(sys.argv[1]) if len(sys.argv)>1 else 40
 z,r=experiment(n)
 print(json.dumps(z,indent=2))