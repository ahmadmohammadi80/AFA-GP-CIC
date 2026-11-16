"""Exact rational probabilities for the configured Beta(1/2,1/2) experiment.

Acceptance decisions are independently checked without confidence-root finding:
M_n(p_threshold)>2E/delta and sample proportion below threshold. In this small
configured horizon the Hoeffding branch adds no extra acceptance (checked).
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import json,math
from exact_budget_gate import Gate

def rising(x,n):
 out=F(1)
 for k in range(n):out*=x+k
 return out

def solve():
 g=Gate();p0=(1-F(str(g.tau))/F(str(g.shift)))/2
 cut=F(4)/F(str(g.delta));flags={}
 for n in range(1,g.B+1):
  for s in range(n+1):
   M=rising(F(1,2),s)*rising(F(1,2),n-s)/rising(F(1),n)/(p0**s*(1-p0)**(n-s))
   flag=F(s,n)<p0 and M>cut
   float_flag=-g.shift*g.intervals[n,s][1]>g.tau
   assert flag==float_flag,(n,s,M,float_flag)
   flags[n,s]=flag
 def terminal(st):return flags[st[0],st[1]] or flags[st[2],st[3]]
 @lru_cache(None)
 def value(st,mode):
  if terminal(st):return F(1)
  if st[0]+st[2]>=g.B:return F(0)
  options=g.allowed(st) if mode=='bellman' else (g.heuristic(st,mode),)
  vals=[]
  for e in options:
   n,s=st[2*e:2*e+2];p=F(2*s+1,2*n+2);z,o=g.successors(st,e)
   vals.append((1-p)*value(z,mode)+p*value(o,mode))
  return max(vals)
 out={}
 for m in ('greedy','joint_mix','bellman'):
  v=sum(value((1,a,1,b),m) for a in (0,1) for b in (0,1))/4
  out[m]={'rational':str(v),'decimal':float(v)}
 out['strict_bellman_advantage']=F(out['bellman']['rational'])>F(out['joint_mix']['rational'])
 out['strict_greedy_advantage']=F(out['greedy']['rational'])>F(out['joint_mix']['rational'])
 out['checked_acceptance_states']=len(flags)
 return out
if __name__=='__main__':
 out=solve();Path(__file__).with_name('rational_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
