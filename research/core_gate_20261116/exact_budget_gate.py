"""Exact finite-budget falsification gate; standard Bayesian DP, NOT new method.

Both schedulers see the same b,Q,H, beta prior, support, pilot outcomes,
confidence intervals, per-label stopping, budget and forced-cover schedule.
DP optimizes prior-averaged chance of the SAME anytime certificate by B.
Fixed-world evaluations enumerate Bernoulli histories (no random seeds).
"""
import sys,json,math,hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'six_issues_20261116'))
from theory_and_guarantees import clipped_intersection

class Gate:
 def __init__(self,budget=24,delta=.05,threshold=.015,shift=.12,prior=(.5,.5),period=20):
  self.B=budget;self.delta=delta;self.tau=threshold;self.shift=shift
  self.a,self.b=prior;self.period=period;self.evaluated=0
  self.intervals={(n,s):clipped_intersection(s,n,delta,2) for n in range(1,budget+1) for s in range(n+1)}
 def lower(self,state):
  n0,s0,n1,s1=state
  return np.array([-self.shift*self.intervals[n,s][1] for n,s in ((n0,s0),(n1,s1))])
 def terminal(self,state):return bool(max(self.lower(state))>self.tau)
 def successors(self,state,e):
  v=list(state);v[2*e]+=1;zero=tuple(v);v[2*e+1]+=1
  return zero,tuple(v)
 def allowed(self,state):
  t=state[0]+state[2];post=t-2
  if (post+1)%self.period==0:return (((post+1)//self.period-1)%2,)
  return (0,1)
 def heuristic(self,state,mode):
  allowed=self.allowed(state)
  if len(allowed)==1:return allowed[0]
  L=self.lower(state);top=int(np.argmax(L));P=np.zeros((2,2))
  for e,(n,s) in enumerate(((state[0],state[1]),(state[2],state[3]))):
   lo,hi=self.intervals[n,s];P[e,e]=self.shift*(hi-lo)/2/(math.sqrt(n)+.2)
  score=P[top] if mode=='greedy' else .7*P[top]+.3*np.max(P,axis=0)
  return int(np.argmax(score))
 @lru_cache(None)
 def optimum(self,state):
  self.evaluated+=1
  if self.terminal(state):return (1.,-1)
  if state[0]+state[2]>=self.B:return (0.,-1)
  values=[]
  for e in self.allowed(state):
   n,s=state[2*e:2*e+2];p=(s+self.a)/(n+self.a+self.b)
   z,o=self.successors(state,e)
   values.append(((1-p)*self.optimum(z)[0]+p*self.optimum(o)[0],e))
  return max(values,key=lambda v:(v[0],-v[1]))
 def choice(self,state,mode):return self.optimum(state)[1] if mode=='bellman' else self.heuristic(state,mode)
 def distribution(self,mode,truth=None):
  # Shared one-observation-per-edge pilot: exact joint law of four outcomes.
  prior_mean=self.a/(self.a+self.b);p=(prior_mean,prior_mean) if truth is None else truth
  mass={(1,s0,1,s1):(p[0] if s0 else 1-p[0])*(p[1] if s1 else 1-p[1]) for s0 in (0,1) for s1 in (0,1)}
  success=cost=0.;false=0.;terminal_mass=0.
  for t in range(2,self.B+1):
   nxt={}
   for state,prob in mass.items():
    stop=self.terminal(state)
    if stop or t==self.B:
     success+=prob*stop;cost+=prob*t;terminal_mass+=prob
     if stop and truth is not None:
      q=int(np.argmax(self.lower(state)))
      false+=prob*(self.shift*(1-2*truth[q])<=self.tau)
     continue
    e=self.choice(state,mode);n,s=state[2*e:2*e+2]
    pe=(s+self.a)/(n+self.a+self.b) if truth is None else truth[e]
    z,o=self.successors(state,e)
    nxt[z]=nxt.get(z,0.)+prob*(1-pe);nxt[o]=nxt.get(o,0.)+prob*pe
   mass=nxt
  assert abs(terminal_mass-1)<1e-11
  return {'success_probability':success,'expected_labels':cost,'false_certificate_probability':false if truth is not None else None,'terminal_mass':terminal_mass}

def audit():
 g=Gate();names=('greedy','joint_mix','bellman')
 bayes={m:g.distribution(m) for m in names}
 grid=[]
 for p0 in (.05,.2,.4,.5,.6,.8,.95):
  for p1 in (.05,.2,.4,.5,.6,.8,.95):
   grid.append({'p':[p0,p1],'methods':{m:g.distribution(m,(p0,p1)) for m in names}})
 prior_mean=g.a/(g.a+g.b)
 root=sum((prior_mean if s0 else 1-prior_mean)*(prior_mean if s1 else 1-prior_mean)*g.optimum((1,s0,1,s1))[0] for s0 in (0,1) for s1 in (0,1))
 assert abs(root-bayes['bellman']['success_probability'])<1e-11
 return {'setup':{'budget':g.B,'delta':g.delta,'tau':g.tau,'shift':g.shift,'prior':[g.a,g.b],'period':g.period,
 'b':[1/3]*3,'Q':[[1/3-g.shift,1/3+g.shift,1/3],[1/3-g.shift,1/3,1/3+g.shift]],'kappa':.05,'H':[[1.,0.,0.]],
 'support_edges':[[0,1],[0,2]],'pilot_labels':2,'frozen_base':'fa5b5b20ad8d7d9cf264d1c7e000b61c1afa6469'},
 'bayes':bayes,'bellman_root_backward_check':root,'dp_states':g.evaluated,'grid':grid,
 'floating_point_tolerance':1e-11,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'scope':'Exact enumeration in float64; Bayesian objective, fixed finite B; not an independent-novelty claim, not Sticky-TaS reproduction.'}
if __name__=='__main__':
 r=audit();p=Path(__file__).with_name('exact_budget_results.json');p.write_text(json.dumps(r,indent=2))
 print(json.dumps({k:v for k,v in r.items() if k!='grid'},indent=2))
