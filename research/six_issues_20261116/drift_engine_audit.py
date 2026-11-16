"""End-to-end current-time certificates under a specified known drift bound."""
import json
from pathlib import Path
import numpy as np
from guarded_engine import GuardedEngine
from published_baselines import preference_problem

def audit(seeds=40,steps=500):
 rows=[]
 for seed in range(seeds):
  b,Q,H,W=preference_problem(True);g=GuardedEngine(b,Q,H,threshold=.015,rho=.0002)
  rng=np.random.default_rng(729193+seed);bad=0;false=0;positive=0
  for t in range(1,steps+1):
   e=g.next_edge();current=W*(.8+.1*np.sin(t/250))
   g.observe(e,int(rng.binomial(1,(1+current[g.edges[e]])/2)))
   lo,hi=g.bounds();truth=np.array([min((q-b)@current@H.T) for q in Q])
   bad+=int(np.any(truth<lo-1e-12) or np.any(truth>hi+1e-12))
   out=g.result();positive+=int(out['certified']);false+=int(out['certified'] and truth[out['policy_index']]<=.015)
  rows.append({'seed':seed,'coverage_failures':bad,'false_acceptances':false,'positive_timepoints':positive})
 return {'seeds':seeds,'steps':steps,'known_rho':.0002,
         'covered_paths':sum(x['coverage_failures']==0 for x in rows),
         'false_acceptances':sum(x['false_acceptances'] for x in rows),
         'positive_timepoints':sum(x['positive_timepoints'] for x in rows),
         'note':'Coverage calibration only; zero acceptances is not evidence of useful power. W amplitude slope implies probability drift <= 0.0002.', 'rows':rows}
if __name__=='__main__':
 r=audit();p=Path('results/drift_engine_audit.json');p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='rows'})
