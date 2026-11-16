"""AFA-CIC-CPL: finite-horizon certificate planning in a cyclic pairwise-preference model.

Prototype written and independently tested for this research exercise. It is NOT
claimed to be globally novel. Certificates are frequentist anytime-valid;
query planning uses a separate Beta-binomial prior.
"""
import numpy as np
from scipy.stats import betabinom
from scipy.special import xlogy
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    n: int = 400  # comparisons per query block
    delta: float = 0.05
    M: int = 3     # unknown (0,i) edges
    lam: float = 0.06

B = np.array([.94, .02, .02, .02])
TARGET_A = np.array([0., 1., 0., 0.])
TARGET_B = np.array([0., 0., .5, .5])
PRIORS = ((58.,42.), (18.,2.), (18.,2.))  # design prior, not safety
TRUE_P = np.array([.58, .90, .90])
# An explicitly known cycle on actions {1,2,3}, W[1,2]=W[2,3]=W[3,1]=0.55,
# contributes zero to both certificate directions because b[1]=b[2]=b[3].
# Only the three (i,0) edge probabilities are uncertain. This is a limited
# cyclic example; unknown cycle edges would require additional confidence bounds.


def kl(q,p):
    return float(np.sum(xlogy(q,q/p)))


def policy_list(cfg: Settings):
    qa=(1-cfg.lam)*B+cfg.lam*TARGET_A
    qb=(1-cfg.lam)*B+cfg.lam*TARGET_B
    return qa,qb


def radius(n, cfg:Settings):
    if n==0:return float('inf')
    return float(np.sqrt(np.log(np.pi**2*cfg.M*n**2/(3*cfg.delta))/(2*n)))


def lower_w(successes:int, n:int, cfg:Settings):
    if n==0:return -1.
    return float(2*max(0., successes/n-radius(n,cfg))-1)


def certifiable(counts, wins, cfg):
    """Check safe certificate for two policies against full-support reference.
    This model sets the cyclic W[1,2]=W[2,3]=W[3,1]=0.55 as known.
    Because b[1]=b[2]=b[3], these contributions cancel exactly for
    both policy directions. Only (i,0) edges are uncertain.
    """
    l=[lower_w(int(wins[i]),int(counts[i]),cfg) for i in range(3)]
    # qa^T W b = lam*b0*W[1,0]; qb^T W b=lam*b0/2*(W[2,0]+W[3,0])
    return l[0]>0 or (l[1]+l[2]>0)


def _predict(x, n, prior, block):
    a,b=prior
    return betabinom.pmf(np.arange(block+1),block,a+x,b+n-x)


def conditional_future_success(counts,wins, chosen, cfg):
    """Exact Bayesian probability of certifying after one more block, but
    *the certificate itself does not use the prior*."""
    n=cfg.n
    if certifiable(counts,wins,cfg):return 1.
    counts2=np.array(counts,dtype=int); counts2[chosen]+=n
    xs=np.arange(n+1,dtype=int)
    probs=_predict(int(wins[chosen]),int(counts[chosen]),PRIORS[chosen],n)
    lv=np.array([lower_w(int(wins[i]),int(counts2[i]),cfg) if i!=chosen else 0. for i in range(3)])
    lo_new=2*np.maximum(0.,(wins[chosen]+xs)/counts2[chosen]-radius(int(counts2[chosen]),cfg))-1
    if chosen==0:valid=lo_new>0
    elif chosen==1:valid=(lv[0]>0) | ((lo_new+lv[2])>0)
    else:valid=(lv[0]>0) | ((lo_new+lv[1])>0)
    return float(np.dot(probs,valid))


def plan(cfg=Settings(), horizon=2):
    """Exact finite-horizon two-block Bellman optimizer for a given prior;
    returns first pair and contingent second action for all first outcomes."""
    assert horizon==2
    n=cfg.n; root={}; actions={}; first_step={}
    for first in range(3):
        prior_probs=_predict(0,0,PRIORS[first],n)
        action_for_x=np.empty(n+1,dtype=int)
        conditional_values=np.empty(n+1,dtype=float)
        for x in range(n+1):
            counts=np.zeros(3,dtype=int);wins=np.zeros(3,dtype=int)
            counts[first]=n;wins[first]=x
            vals=np.array([conditional_future_success(counts,wins,s,cfg) for s in range(3)])
            j=int(np.argmax(vals));action_for_x[x]=j;conditional_values[x]=vals[j]
        root[first]=float(np.dot(prior_probs,conditional_values))
        actions[first]=action_for_x
        # myopic first-step certificate probability, exact under the prior
        first_step[first]=float(np.dot(prior_probs, np.array([certifiable([n if k==first else 0 for k in range(3)], [x if k==first else 0 for k in range(3)],cfg) for x in range(n+1)])))
    lookahead_first=max(root,key=root.get)
    greedy_first=max(first_step,key=first_step.get)
    return {'root_values':root,'myopic_values':first_step,
            'lookahead_first':int(lookahead_first),'greedy_first':int(greedy_first),
            'second_actions':actions}


def experiment(planner, cfg=Settings(), repetitions=5000, seed=42):
    rng=np.random.default_rng(seed)
    data={}
    for name,first in [('lookahead',planner['lookahead_first']),('myopic',planner['greedy_first'])]:
        successful=0
        spent=[]
        for _ in range(repetitions):
            counts=np.zeros(3,dtype=int);wins=np.zeros(3,dtype=int)
            x=int(rng.binomial(cfg.n,TRUE_P[first]))
            counts[first]=cfg.n;wins[first]=x
            if certifiable(counts,wins,cfg):
                successful+=1;spent.append(cfg.n);continue
            second=int(planner['second_actions'][first][x])
            y=int(rng.binomial(cfg.n,TRUE_P[second]))
            counts[second]+=cfg.n;wins[second]+=y
            successful+=int(certifiable(counts,wins,cfg));spent.append(2*cfg.n)
        data[name]={'certification_rate':successful/repetitions,'certified':successful,
                    'repetitions':repetitions,'mean_comparisons':float(np.mean(spent))}
    return data


def true_win_matrix(p=TRUE_P):
    W=np.zeros((4,4),float)
    for i in range(1,4): W[i,0]=2*p[i-1]-1; W[0,i]=-W[i,0]
    W[1,2]=W[2,3]=W[3,1]=.55
    W[2,1]=W[3,2]=W[1,3]=-.55
    return W

def exact_true_success(planner, first, cfg=Settings()):
    """Enumerate exact frequentist success for a given first action and
    its contingent continuation policy; no Monte Carlo variance."""
    from scipy.stats import binom
    n=cfg.n
    root=binom.pmf(np.arange(n+1),n,TRUE_P[first])
    total=0.
    for x in range(n+1):
        if root[x]<1e-17:continue
        counts=np.zeros(3,dtype=int);wins=np.zeros(3,dtype=int)
        counts[first]=n;wins[first]=x
        if certifiable(counts,wins,cfg):total+=root[x];continue
        f=int(planner['second_actions'][first][x])
        xs=np.arange(n+1,dtype=int)
        updated_counts=counts.copy();updated_counts[f]+=n
        # For each possible y in the next Bernoulli batch, compute exact event.
        success=np.array([certifiable(updated_counts, wins+np.eye(3,dtype=int)[f]*y,cfg) for y in xs])
        total+=float(root[x]*np.dot(binom.pmf(xs,n,TRUE_P[f]),success))
    return float(total)

def exact_static_success(first, second, cfg=Settings()):
    """Exact final certification probability of a preset two-block design."""
    from scipy.stats import binom
    n=cfg.n
    x=np.arange(n+1)
    if first==second:
        ns=np.zeros(3,dtype=int); ns[first]=2*n
        probs=binom.pmf(np.arange(2*n+1),2*n,TRUE_P[first])
        flags=np.array([certifiable(ns, np.eye(3,dtype=int)[first]*int(k),cfg) for k in range(2*n+1)])
        return float(probs@flags)
    nvec=np.zeros(3,dtype=int);nvec[first]=n;nvec[second]=n
    px=binom.pmf(x,n,TRUE_P[first]); py=binom.pmf(x,n,TRUE_P[second]);
    total=0.
    for i in x:
        if px[i]<1e-17:continue
        wins=np.zeros(3,dtype=int); wins[first]=i
        flags=np.array([certifiable(nvec,wins+np.eye(3,dtype=int)[second]*int(j),cfg) for j in x])
        total+=px[i]*float(py@flags)
    return float(total)

if __name__=='__main__':
    import json, argparse
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=10000);p.add_argument('--out',default='results.json')
    args=p.parse_args();cfg=Settings()
    planned=plan(cfg)
    s={ 'config':cfg.__dict__,
        'baseline':B.tolist(),
        'policy_KL': [kl(q,B) for q in policy_list(cfg)],
        'true_p':TRUE_P.tolist(),
        'true_W':true_win_matrix().tolist(),
        'cycle_W12_W23_W31':0.55,
        'true_gain_A_B':[float(q@true_win_matrix()@B) for q in policy_list(cfg)],
        'exact_true_success':{'lookahead':exact_true_success(planned,planned['lookahead_first'],cfg), 'myopic':exact_true_success(planned,planned['greedy_first'],cfg)},
        'exact_fixed_success':{'A_A':exact_static_success(0,0,cfg),'B_C':exact_static_success(1,2,cfg)},
        'root_values':{str(k):v for k,v in planned['root_values'].items()},
        'myopic_values':{str(k):v for k,v in planned['myopic_values'].items()},
        'lookahead_first':planned['lookahead_first'],
        'greedy_first':planned['greedy_first'],
        'experiment':experiment(planned,cfg,args.repetitions)}
    with open(args.out,'w') as f:json.dump(s,f,indent=2)
    print(json.dumps(s,indent=2))
