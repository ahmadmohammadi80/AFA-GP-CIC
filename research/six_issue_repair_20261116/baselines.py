"""Literature-inspired reference adaptations, NOT authors' original code.

optimistic_edge: UCB-style preference uncertainty querying (cf. RUCB/dueling).
pb_dirac: finite-policy PAC-Bayes/union-bound confidence control with uniform
randomized edge queries and importance-corrected policy returns. Different
sampling model from adaptive methods; results only demonstrate protocol.
"""
import numpy as np
from scipy.stats import binomtest
from repair_core import problem,greedy_scores
from anytime_mixture import edge_cs


def adapted_ucb(seed,budget=1510,delta=.05,threshold=.015,cyclic=False):
    b,Q,H,W,E,A,mu=problem(cyclic=cyclic);m=len(E)
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(m)]
    N=np.ones(m,int); wins=np.array([s.binomial(1,(1+x)/2) for s,x in zip(streams,mu)])
    lo=np.array([edge_cs(int(w),1,delta/m)[0] for w in wins]); hi=np.array([edge_cs(int(w),1,delta/m)[1] for w in wins])
    for t in range(m,budget+1):
        winner,L,_=greedy_scores(A,lo,hi,N)
        if L[winner]>threshold:return {'cert':True,'labels':t,'selected':winner,'max_policy_width':float(np.max(np.sum(np.abs(A)*(hi-lo),axis=-1)))}
        if t==budget:break
        optimistic=np.where(A>=0,A*hi,A*lo).sum(axis=-1)
        q=int(optimistic.min(axis=1).argmax())
        # Credible worst-response under the upper confidence matrix.
        h=int(optimistic[q].argmin())
        score=np.abs(A[q,h])*(hi-lo)/(np.sqrt(N)+.2)
        e=int(score.argmax()) if score.max()>0 else int(N.argmin())
        wins[e]+=streams[e].binomial(1,(1+mu[e])/2);N[e]+=1
        lo[e],hi[e]=edge_cs(int(wins[e]),int(N[e]),delta/m)
    return {'cert':False,'labels':budget,'selected':int(winner),'max_policy_width':float(np.max(np.sum(np.abs(A)*(hi-lo),axis=-1)))}


def pac_bayes_dirac(seed,budget=1510,delta=.05,threshold=.015,cyclic=False):
    """PAC-Bayes fixed uniform policy prior, point-mass posterior; finite T.

    E randomized edges chosen independently with known probability 1/E.
    For every candidate q, opponent h: Z_t=E*A[q,h,e_t]*Y_t in [-C,C],
    so E[Z_t|past]=G(q,h), and simultaneous Hoeffding/PAC-Bayes-Dirac
    lower at fixed known t is mean(Z)-C*sqrt(2log(PH/delta)/t).
    No adaptive stopping; bound evaluated only at the fixed budget.
    """
    b,Q,H,W,E,A,mu=problem(cyclic=cyclic);m=len(E);P,R=A.shape[:2]
    rng=np.random.default_rng(np.random.SeedSequence([seed,900123]))
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(m)]
    accum=np.zeros((P,R))
    for t in range(budget):
        e=int(rng.integers(m))
        y=2*int(streams[e].binomial(1,(1+mu[e])/2))-1
        accum+=m*A[:,:,e]*y
    C=m*np.max(np.abs(A),axis=2)
    # P*R hypotheses -> exactly Dirac KL=log(P*R) w.r.t. uniform prior.
    radius=C*np.sqrt(2*np.log(P*R/delta)/budget)
    lowers=((accum/budget)-radius).min(axis=1)
    w=int(lowers.argmax())
    return {'cert':bool(lowers[w]>threshold),'labels':budget,'selected':w,
            'best_lower':float(lowers[w]),'max_policy_width':float(2*np.max(radius))}


def exact_mcnemar(a,b):
    plus=sum(x and not y for x,y in zip(a,b))
    minus=sum(y and not x for x,y in zip(a,b))
    n=plus+minus
    p=binomtest(min(plus,minus),n,.5).pvalue if n else 1.
    return {'wins_A_only':plus,'wins_B_only':minus,'discordant':n,'p_raw':float(p)}


def holm(pvals):
    ps=np.asarray(pvals,float);order=np.argsort(ps);out=np.zeros(len(ps));run=0.
    for rank,i in enumerate(order):
        run=max(run,min(1.,(len(ps)-rank)*ps[i]));out[i]=run
    return out.tolist()


def benchmark(seeds=40,offset=20261116,cyclic=False):
    from repair_core import simulate
    rows=[]
    for t in range(seeds):
        s=offset+t
        rows.append({'seed':s,'joint_threshold':simulate(s,'joint_threshold',cyclic=cyclic),
                     'greedy':simulate(s,'greedy',cyclic=cyclic),
                     'ucb_adapt':adapted_ucb(s,cyclic=cyclic),
                     'pac_bayes_dirac':pac_bayes_dirac(s,cyclic=cyclic),
                     'rucb_query':rucb_query(s,cyclic=cyclic)})
    keys=['joint_threshold','greedy','ucb_adapt','pac_bayes_dirac','rucb_query']
    summary={k:{'certified':sum(r[k]['cert'] for r in rows),
                'mean_labels':float(np.mean([r[k]['labels'] for r in rows])),
                'mean_policy_width':float(np.mean([r[k]['max_policy_width'] for r in rows]))}
                for k in keys}
    comps={k:exact_mcnemar([r[k]['cert'] for r in rows],
                           [r['greedy']['cert'] for r in rows])
           for k in keys if k!='greedy'}
    padj=holm([v['p_raw'] for v in comps.values()])
    for v,p in zip(comps.values(),padj):v['p_holm']=p
    return {'cyclic':cyclic,'methods':summary,'comparisons_vs_greedy':comps,'rows':rows,
      'note':'RUCB sampling follows a published query principle but not RUCB output; PAC-Bayes-Dirac and UCB adaptations are NOT reproductions of published end-to-end algorithms.'}


def rucb_query(seed,budget=1510,delta=.05,threshold=.015,alpha=.51,cyclic=False):
    """Query component of Zoghi et al. (2014) RUCB; our shared certificate.

    Classic RUCB assumes a Condorcet winner and returns an arm, not a KL policy
    certificate. We use RUCB's champion/challenger sampling and test the same
    anytime multi-opponent lower bound as AFA; this ADAPTATION is not RUCB itself.
    """
    b,Q,H,W,E,A,mu=problem(cyclic=cyclic);K=len(b);m=len(E)
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(m)]
    rng=np.random.default_rng(np.random.SeedSequence([seed,700919]))
    N=np.ones(m,int);wins=np.array([s.binomial(1,(1+x)/2) for s,x in zip(streams,mu)])
    lo=np.array([edge_cs(int(w),1,delta/m)[0] for w in wins]);hi=np.array([edge_cs(int(w),1,delta/m)[1] for w in wins])
    idx={e:i for i,e in enumerate(E)};champ_prev=None;self_duels=0
    for t in range(m,budget+1):
        winner,L,_=greedy_scores(A,lo,hi,N)
        if L[winner]>threshold:return {'cert':True,'labels':t,'selected':winner,'self_duels':self_duels,'max_policy_width':float(np.max(np.sum(np.abs(A)*(hi-lo),axis=-1)))}
        if t==budget:break
        up=np.ones((K,K));np.fill_diagonal(up,.5)
        bonus=np.sqrt(alpha*np.log(t+1)/N)
        for e,(i,j) in enumerate(E):
            ph=wins[e]/N[e]
            up[i,j]=min(1.,ph+bonus[e]);up[j,i]=min(1.,1-ph+bonus[e])
        C=[i for i in range(K) if all(up[i,j]>=.5 for j in range(K))]
        if not C:C=list(range(K)) # necessary extension for cyclic non-Condorcet cases
        if champ_prev in C and len(C)>1 and rng.random()<.5:
            champ=champ_prev
        else:
            champ=int(rng.choice(C))
        # At most one remembered optimistic champion B in the original scheme.
        champ_prev=champ if len(C)==1 else champ_prev
        challenger=int(np.argmax(up[:,champ]))
        if challenger==champ:
            self_duels+=1;continue
        e=idx[(min(champ,challenger),max(champ,challenger))]
        wins[e]+=streams[e].binomial(1,(1+mu[e])/2);N[e]+=1
        lo[e],hi[e]=edge_cs(int(wins[e]),int(N[e]),delta/m)
    return {'cert':False,'labels':budget,'selected':int(winner),'self_duels':self_duels,'max_policy_width':float(np.max(np.sum(np.abs(A)*(hi-lo),axis=-1)))}

if __name__=='__main__':
 import json,sys
 n=int(sys.argv[1]) if len(sys.argv)>1 else 40
 out=benchmark(seeds=n)
 print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
 from pathlib import Path
 Path(__file__).with_name('benchmark40.json').write_text(json.dumps(out,indent=2))
