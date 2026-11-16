"""Preregistered in code: B=24, 5000/grid world, 30000/prior, seed=20261116.
No true p reaches any expert. Primary CSWM uses horizon 2; full horizon is an
explicit diagnostic ablation. No parameter search, reused exact fixed baselines.
"""
import sys,json,math,time,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'core_gate_20261116'))
from exact_budget_gate import Gate
from cswm_allocator import CSWMAllocator, BellmanExpert, GateState, certificate_uncertainty

MODES=('greedy','joint_mix','bellman','bellman_h2','thompson','uniform','cswm','cswm_full')

def tables(g):
    side=g.B+1
    size=side**4
    tab={k:np.zeros(size,dtype=d) for k,d in [('terminal',bool),('winner',int),('u',float),('greedy',int),('joint_mix',int),('bellman',int),('bellman_h2',int),('forced',bool),('n0',int),('s0',int),('n1',int),('s1',int)]}
    tab['next']=np.zeros((size,2,2),dtype=np.int32)
    code=lambda s:((s[0]*side+s[1])*side+s[2])*side+s[3]
    h2=BellmanExpert(g,2)
    for n0 in range(1,g.B):
      for n1 in range(1,g.B-n0+1):
       for s0 in range(n0+1):
        for s1 in range(n1+1):
         s=(n0,s0,n1,s1);i=code(s)
         for k,v in zip(('n0','s0','n1','s1'),s):tab[k][i]=v
         tab['terminal'][i]=g.terminal(s)
         tab['winner'][i]=int(np.argmax(g.lower(s)))
         tab['u'][i]=certificate_uncertainty(GateState(g,s))
         if n0+n1==g.B or tab['terminal'][i]:continue
         tab['forced'][i]=len(g.allowed(s))==1
         for m in ('greedy','joint_mix','bellman'):tab[m][i]=g.choice(s,m)
         tab['bellman_h2'][i]=h2.value(s,2)[1]
         for e in (0,1):
          z,o=g.successors(s,e);tab['next'][i,e]=[code(z),code(o)]
    return tab,code


def simulation(g,tab,code,p,streams,mode,seed):
    N=len(p);rng=np.random.default_rng(seed)
    pilot=(streams[:,:,0]<p).astype(int)
    state=np.array([code((1,int(s[0]),1,int(s[1]))) for s in pilot])
    weights=np.full((N,3),1/3.)
    logw=np.zeros((N,3));labels=np.full(N,g.B,dtype=int)
    active=np.ones(N,dtype=bool)
    reward_tot=np.zeros(N)
    expert_reward_tot=np.zeros((N,3))
    rounds=zero_rewards=disagreements=0
    eta=CSWMAllocator.tuned_eta(3,g.B-2,.05)
    for t in range(2,g.B):
        stop=active & tab['terminal'][state];labels[stop]=t;active[stop]=False
        ids=np.flatnonzero(active)
        if not len(ids):break
        s=state[ids]
        if mode in ('greedy','joint_mix','bellman','bellman_h2'):
            edge=tab[mode][s]
        else:
            free=~tab['forced'][s]
            p0=np.full(len(s),.5);p1=np.full(len(s),.5)
            sf=s[free]
            p0[free]=rng.beta(tab['s0'][sf]+g.a,tab['n0'][sf]-tab['s0'][sf]+g.b)
            p1[free]=rng.beta(tab['s1'][sf]+g.a,tab['n1'][sf]-tab['s1'][sf]+g.b)
            nxt=tab['next'][s]
            score0=(1-p0)*tab['terminal'][nxt[:,0,0]]+p0*tab['terminal'][nxt[:,0,1]]
            score1=(1-p1)*tab['terminal'][nxt[:,1,0]]+p1*tab['terminal'][nxt[:,1,1]]
            th=np.where(np.maximum(score0,score1)>0,(score1>score0).astype(int),(p1<p0).astype(int))
            th=np.where(tab['forced'][s],tab['greedy'][s],th)
            if mode=='thompson':edge=th
            else:
                props=np.column_stack((tab['greedy'][s],tab['bellman' if mode=='cswm_full' else 'bellman_h2'][s],th))
                probs=weights[ids] if mode!='uniform' else np.full((len(ids),3),1/3.)
                chosen=(rng.random(len(ids))[:,None]>np.cumsum(probs,axis=1)[:,:-1]).sum(axis=1)
                edge=props[np.arange(len(ids)),chosen]
        n=np.where(edge==0,tab['n0'][s],tab['n1'][s])
        y=(streams[ids,edge,n]<p[ids,edge]).astype(int)
        after=tab['next'][s,edge,y]
        reward=tab['u'][s]-tab['u'][after]
        reward_tot[ids]+=reward
        rounds+=len(ids);zero_rewards+=int(np.sum(np.abs(reward)<1e-14))
        if mode in ('cswm','cswm_full'):
            disagreements+=int(np.sum(np.any(props!=props[:,0,None],axis=1)))
            # Evaluation-only potential outcomes. They NEVER enter the allocator.
            ncf=np.where(props==0,tab['n0'][s,None],tab['n1'][s,None])
            ycf=(streams[ids[:,None],props,ncf]<p[ids[:,None],props]).astype(int)
            cf=tab['next'][s[:,None],props,ycf]
            expert_reward_tot[ids]+=tab['u'][s,None]-tab['u'][cf]
            loss=(1-reward)/2
            logw[ids,chosen]-=eta*loss/(weights[ids,chosen]+eta/2)
            lw=logw[ids];lw-=lw.max(axis=1)[:,None]
            w=np.exp(lw);weights[ids]=w/w.sum(axis=1)[:,None]
        state[ids]=after
    cert=tab['terminal'][state]
    return dict(successes=int(cert.sum()),replicates=N,success_probability=float(cert.mean()),
        success_se=float(np.sqrt(cert.mean()*(1-cert.mean())/N)),expected_labels=float(labels.mean()),
        label_se=float(labels.std(ddof=1 if N>1 else 0)/np.sqrt(N)),label_variance=float(labels.var(ddof=1 if N>1 else 0)),
        certificate_variance=float(cert.var(ddof=1 if N>1 else 0)),mean_final_weights=weights.mean(axis=0).tolist(),
        mean_cumulative_uncertainty_reduction=float(reward_tot.mean()),
        labels_sum=int(labels.sum()),labels_squared_sum=int((labels**2).sum()),
        observed_reward_zero_fraction=zero_rewards/rounds,
        proposal_disagreement_fraction=(disagreements/rounds if mode in ('cswm','cswm_full') else None),
        false_certificates=int(np.sum(cert & (g.shift*(1-2*p[np.arange(N),tab['winner'][state]])<=g.tau))),
        mean_local_expert_regret=(float(np.mean(expert_reward_tot.max(axis=1)-reward_tot)) if mode in ('cswm','cswm_full') else None),
        local_regret_95th_percentile=(float(np.quantile(expert_reward_tot.max(axis=1)-reward_tot,.95)) if mode in ('cswm','cswm_full') else None))


def main():
    start=time.time();g=Gate();tab,code=tables(g)
    exact=json.loads((HERE.parent/'core_gate_20261116/exact_budget_results.json').read_text())
    out={'protocol':dict(budget=24,pilot=2,grid_replicates=5000,prior_replicates=30000,seed=20261116,
        regret_delta=.05,certificate_delta=.05,eta=CSWMAllocator.tuned_eta(3,22,.05),
        primary_experts=['greedy','bellman_h2','thompson_decision'],
        posterior_prior=[.5,.5],paired_edge_streams=True,prior_objective='independent Beta(.5,.5) truth per episode',
        grid_objective='uniform 49 specified worlds; distinct from beta prior',
        threshold=.015,forced_period=20,tuning='none; theorem eta fixed before evaluation'),
        'grid':[],'exact_bayes':exact['bayes']}
    valid=(tab['n0']>0)&(tab['n1']>0)&((tab['n0']+tab['n1'])<g.B)&(~tab['terminal'])
    out['expert_diagnostic']={'nonterminal_count_states':int(valid.sum()),
        'h2_greedy_disagreement_states':int(np.sum(valid & (tab['greedy']!=tab['bellman_h2']))),
        'full_greedy_disagreement_states':int(np.sum(valid & (tab['greedy']!=tab['bellman'])))}
    for j,row in enumerate(exact['grid']):
        N=5000;rng=np.random.default_rng(20261116+j)
        p=np.tile(row['p'],(N,1));streams=rng.random((N,2,g.B))
        result={m:simulation(g,tab,code,p,streams,m,20270000+j) for m in MODES}
        out['grid'].append({'p':row['p'],'methods':result,'exact_fixed':row['methods']})
        if j%10==0:print('completed grid',j+1,flush=True)
    rng=np.random.default_rng(20280000);p=rng.beta(.5,.5,(30000,2));streams=rng.random((len(p),2,g.B))
    out['bayes_mc']={m:simulation(g,tab,code,p,streams,m,20290000) for m in MODES}
    radius=math.sqrt(math.log(2*49/.05)/(2*5000))
    out['gate_A']={}
    for mode in ('cswm','cswm_full'):
        wins=sum(r['methods'][mode]['success_probability']>max(r['exact_fixed'][m]['success_probability'] for m in ('greedy','bellman')) for r in out['grid'])
        robust=sum(r['methods'][mode]['success_probability']-radius>max(r['exact_fixed'][m]['success_probability'] for m in ('greedy','bellman')) for r in out['grid'])
        out['gate_A'][mode]={'point_estimate_wins_both':wins,'simultaneous_95pct_lower_bound_wins_both':robust,'hoeffding_radius':radius,'required':30,'passed':robust>=30,
            'uniform_grid_mean_certificate':float(np.mean([r['methods'][mode]['success_probability'] for r in out['grid']])),
            'uniform_grid_mean_labels':float(np.mean([r['methods'][mode]['expected_labels'] for r in out['grid']]))}
    out['variance_definition']='Within-world Bernoulli certificate and label variance. Lower Bernoulli variance alone can mean worse success. Across-method variation is not the same estimand.'
    out['elapsed_seconds']=time.time()-start
    out['sources_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'cswm_allocator.py',Path(__file__),HERE.parent/'core_gate_20261116/exact_budget_results.json')}
    (HERE/'cswm_results.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'gate_A':out['gate_A'],'bayes_mc':out['bayes_mc'],'elapsed':out['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
