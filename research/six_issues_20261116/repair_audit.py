"""Reproduction, not a newly registered or untouched significance experiment."""
import json,time,platform,hashlib
from pathlib import Path
import numpy as np
from guarded_engine import GuardedEngine
from theory_and_guarantees import guarded_joint
from published_baselines import preference_problem,published_trial

def engine_trial(seed,cyclic=False):
 b,Q,H,W=preference_problem(cyclic);g=GuardedEngine(b,Q,H,kappa=.05)
 streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(len(g.edges))]
 out=g.run(lambda e,t:int(streams[e].binomial(1,(1+W[g.edges[e]])/2)),1510)
 gains=np.array([min((q-b)@W@H.T) for q in Q])
 out.update(seed=seed,false_certificate=bool(out['certified'] and gains[out['policy_index']]<=.015),
            actual_simple_regret=float(max(gains)-gains[out['policy_index']]))
 return out

def run(seeds=200):
 results={}
 for cyclic in (False,True):
  xs=[engine_trial(20290101+i,cyclic) for i in range(seeds)]
  refs=[guarded_joint(20290101+i,cyclic=cyclic) for i in range(seeds)]
  results[str(cyclic)]={'certificates':sum(x['certified'] for x in xs),
   'false_certificates':sum(x['false_certificate'] for x in xs),
   'mean_labels':float(np.mean([x['labels'] for x in xs])),
   'reference_decision_cost_mismatches':sum((x['labels'],x['certified'])!=(r['labels'],r['certified']) for x,r in zip(xs,refs)),
   'regret_bound_violations':sum(x['actual_simple_regret']>x['simple_regret_certificate']+1e-12 for x in xs),
   'rows':xs}
 baseline={m:[published_trial(20261116+i,m) for i in range(40)] for m in ('rucb','dts')}
 times=[]
 for k in (5,10,20,40):
  rng=np.random.default_rng(918+k);b=np.ones(k)/k;Q=.9*b+.1*rng.dirichlet(np.ones(k),size=16);H=rng.dirichlet(np.ones(k),size=12)
  schedules=[];timing={}
  steps=k*(k-1)//2+500
  for mode in (False,True):
   g=GuardedEngine(b,Q,H,threshold=1,incremental=mode)
   stream=np.random.default_rng(717);edges=[];start=time.perf_counter()
   for _ in range(steps):
    e=g.next_edge();edges.append(e);g.observe(e,int(stream.integers(2)));g.result()
   timing[str(mode)]=time.perf_counter()-start;schedules.append(edges)
  times.append({'k':k,'q':16,'h':12,'labels':steps,'post_pilot_labels':500,'identical_queries':schedules[0]==schedules[1],
                'seconds':timing,'speedup':timing['False']/timing['True']})
 root=Path(__file__).parent
 return {'provenance':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},
  'note':'Same historical seeds: reproducibility audit only, not new independent confirmation'},
  'guarded_reproduction':results,'corrected_adapted_baselines':{m:{'certificates':sum(x['certified'] for x in rows),'mean_labels':float(np.mean([x['labels'] for x in rows])),'rows':rows} for m,rows in baseline.items()},'general_opponent_end_to_end_scaling':times}
if __name__=='__main__':
 r=run();p=Path('results/repair_audit.json');p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(r,indent=2))
 print(json.dumps({**{k:{a:b for a,b in v.items() if a!='rows'} for k,v in r['guarded_reproduction'].items()},'scaling':r['general_opponent_end_to_end_scaling'],'baselines':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in r['corrected_adapted_baselines'].items()}},indent=2))
