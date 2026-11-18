"""Paired full online paths: existing greedy vs gated exact static two-step AFA planner.

Same 5 arms, 2 KL-feasible policies, 5 opponents, beta-mixture confidence
sequences, Bernoulli edge-prefix streams and budget as issue1_ablation.py.
Every sample is followed by a new decision for both algorithms.  AFA calls
its exact two-query lookahead only inside the predeclared margin gate; at
other times or when the two-query robust crossing probability is zero, it
uses the original greedy rule.  This is a *gated* policy, not a claim of
full-horizon optimality.  It is the same planner and certificate as the
existing static_two_step_certificate_20261118.js, not a new algorithm.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from integrated_two_step_certificate_20261118 import (
    canonical_problem, intervals, greedy_edge, plan_two, certificate
)

METHODS=('greedy','joint_mix','afa_gated_two_step')

def joint_mix_edge(s,n,As,delta=.05):
    """Unchanged 70/30 scoring rule of issue1_ablation.py (not claimed novel)."""
    ci=intervals(s,n,delta);lo=ci[:,0];hi=ci[:,1]
    mid=(lo+hi)/2;half=(hi-lo)/2
    lower=np.array([certificate(A,lo,hi) for A in As]);top=int(np.argmax(lower))
    priority=[]
    for A in As:
        worst=int(np.argmin(A@mid-np.abs(A)@half))
        priority.append(np.abs(A[worst])*half/(np.sqrt(n)+.2))
    P=np.array(priority)
    scores=.7*P[top]+.3*np.max(P,axis=0)
    return int(np.argmax(scores)) if scores.max()>0 else int(np.argmin(n))


def run_one(seed,mode,budget=1510,delta=.05,threshold=.015,gate=.0005,cyclic=False,null=False):
    if mode not in METHODS:raise ValueError('method')
    if budget<12 or not 0<delta<1 or gate<0:raise ValueError('setup')
    s,n,streams,mu,As,W,qs,b,H=canonical_problem(seed,cyclic=cyclic)
    if null:
        W=np.zeros_like(W);mu=np.zeros_like(mu)
        streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(len(n))]
        s=np.array([rng.binomial(1,.5) for rng in streams],dtype=int)
    true_gains=[min(float((q-b)@W@h) for h in H) for q in qs]
    assert max(float(sum(q*np.log(q/b))) for q in qs)<.05
    calls=positive=overrides=0;clock=0
    cover_breached=False
    query_counts=np.zeros(len(n),dtype=int)
    t0=time.perf_counter()
    while True:
        ci=intervals(s,n,delta)
        cover_breached |= bool(np.any(ci[:,0]>mu+1e-12) or np.any(ci[:,1]<mu-1e-12))
        lb=[certificate(A,ci[:,0],ci[:,1]) for A in As]
        win=int(np.argmax(lb));is_cert=bool(lb[win]>threshold)
        if is_cert or n.sum()>=budget:break
        greedy=greedy_edge(s,n,As,delta)
        edge=joint_mix_edge(s,n,As,delta) if mode=='joint_mix' else greedy
        margin=threshold-max(lb)
        if mode=='afa_gated_two_step' and 0<margin<=gate:
            calls+=1
            tree=plan_two(s,n,As,delta,threshold)
            if tree['already_certified']:raise AssertionError('planner disagrees with certificate')
            v=tree['optimal']['robust_probability']
            if v>1e-12:
                positive+=1
                edge=tree['optimal']['e']
                overrides+=int(edge!=greedy)
        y=int(streams[edge].binomial(1,(1+mu[edge])/2))
        s[edge]+=y;n[edge]+=1;query_counts[edge]+=1;clock+=1
    return {'seed':int(seed),'method':mode,'certified':is_cert,
        'false_certificate':bool(is_cert and true_gains[win]<=threshold),
        'labels':int(n.sum()),'steps':clock,'winner':win if is_cert else None,
        'max_lower':float(max(lb)),'true_gain_winner':float(true_gains[win]),
        'coverage_breached':cover_breached,'planner_calls':calls,
        'positive_plans':positive,'root_overrides':overrides,
        'online_decisions':clock,'query_counts':query_counts.tolist(),
        'elapsed_sec':round(time.perf_counter()-t0,5)}

def paired_audit(seeds=40,offset=20261116,budget=1510,gate=.0005,cyclic=False,null=False):
    if seeds<=0:raise ValueError('seeds')
    rows={m:[] for m in METHODS}
    for i in range(seeds):
        seed=offset+i
        for m in METHODS:
            rows[m].append(run_one(seed,m,budget=budget,gate=gate,cyclic=cyclic,null=null))
    gs=rows['greedy'];ps=rows['afa_gated_two_step']
    def pair(a,b):
        ra,rb=rows[a],rows[b]
        win=sum(p['certified'] and not g['certified'] for p,g in zip(ra,rb))
        loss=sum(g['certified'] and not p['certified'] for p,g in zip(ra,rb))
        pv=float(binomtest(min(win,loss),win+loss,p=.5).pvalue) if win+loss else 1.
        diff=np.array([p['labels']-g['labels'] for p,g in zip(ra,rb)],float)
        rng=np.random.default_rng(271828)
        boot=np.mean(diff[rng.integers(0,len(diff),(10000,len(diff)))],axis=1)
        low,high=np.quantile(boot,[.025,.975])
        return {'first_only':win,'second_only':loss,'mcnemar_exact_2sided_p':pv,
                'mean_labels_first_minus_second':float(diff.mean()),
                'paired_bootstrap_95pct_CI_mean_label_diff':[float(low),float(high)],
                'label_differences':diff.tolist()}
    pairs={a+'_vs_'+b:pair(a,b) for a,b in (
        ('afa_gated_two_step','greedy'),('joint_mix','greedy'),
        ('afa_gated_two_step','joint_mix'))}
    # Holm FWER on the two predeclared comparisons to greedy.
    holm=sorted(['afa_gated_two_step_vs_greedy','joint_mix_vs_greedy'],
                key=lambda z:pairs[z]['mcnemar_exact_2sided_p'])
    running=0.
    for i,name in enumerate(holm):
        running=max(running,min(1.,(len(holm)-i)*pairs[name]['mcnemar_exact_2sided_p']))
        pairs[name]['holm_vs_greedy_p']=running
    s={m:{'certified':sum(x['certified'] for x in rows[m]),
          'false_certificates':sum(x['false_certificate'] for x in rows[m]),
          'coverage_breaches':sum(x['coverage_breached'] for x in rows[m]),
          'mean_labels':float(np.mean([x['labels'] for x in rows[m]])),
          'mean_elapsed_sec':float(np.mean([x['elapsed_sec'] for x in rows[m]]))} for m in METHODS}
    s['afa_gated_two_step'].update({
        'total_planner_calls':sum(x['planner_calls'] for x in ps),
        'positive_plans':sum(x['positive_plans'] for x in ps),
        'root_overrides':sum(x['root_overrides'] for x in ps)})
    return {'protocol':{'offset':offset,'seeds':seeds,'budget':budget,'gate':gate,
           'cyclic':cyclic,'null':null,'delta':.05,'threshold':.015,'KL_cap':.05,'k':5,
           'candidate_policies':2,'opponents':5,
           'preference_oracle':'shared edge-specific Bernoulli prefixes',
           'confidence':'beta-mixture anytime per edge, delta/E',
           'clock':'decision after each added comparison for all methods',
           'planning':'gated, fallback greedy if robust horizon-2 crossing probability = 0',
           'scope':'full online 1510-label trajectories, stationary synthetic oracle'},
           'summary':s,'paired':pairs,'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=40)
    p.add_argument('--offset',type=int,default=20261116)
    p.add_argument('--budget',type=int,default=1510)
    p.add_argument('--gate',type=float,default=.0005)
    p.add_argument('--cyclic',action='store_true')
    p.add_argument('--null',action='store_true')
    p.add_argument('--out',default='results/online_two_step_primary.json')
    a=p.parse_args()
    r=paired_audit(a.seeds,a.offset,a.budget,a.gate,a.cyclic,a.null)
    f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps({'summary':r['summary'],'paired':{k:{j:w for j,w in v.items() if j!='label_differences'} for k,v in r['paired'].items()}},ensure_ascii=False,indent=2))
