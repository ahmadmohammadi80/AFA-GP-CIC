"""Supplementary bounded-drift safety/adaptation diagnostic; not gate A.
B=120, 40 paths, p0 linearly .2->.4 and p1 .4->.2, known rho=.2/119.
Stationary Beta-mixture intervals are NOT used under drift. The experts' Beta
predictive planning remains a heuristic under drift, not a calibrated posterior.
"""
import json
from pathlib import Path
import numpy as np
from cswm_allocator import *

def main():
    B=120;rows=[]
    for seed in range(40):
        rng=np.random.default_rng(303000+seed)
        s=FiniteGameState(-.12*np.eye(2).reshape(2,1,2),((0,1),(0,2)),[0,0],[0,0],B,
                          rho=.2/(B-1),sample_times=[0,0])
        a=CSWMAllocator([GreedyExpert(),BellmanExpert(horizon=2),ThompsonExpert(seed+1)],
                        CSWMAllocator.tuned_eta(3,B,.05),.05,seed=seed+4)
        valid=True;accepted=False;weights_mid=None
        for t in range(1,B+1):
            p=np.array([.2+.2*(t-1)/(B-1),.4-.2*(t-1)/(B-1)])
            e=a.select_edge(s);idx=s.active_edges.index(e)
            o=BinaryObservation(e,int(rng.random()<p[idx]));a.update(o);s=s.observe(o)
            valid &= bool(np.all((s.lower_bound<=2*p-1)&(2*p-1<=s.upper_bound)))
            if t==B//2:weights_mid=a.weights.tolist()
            if s.terminal():accepted=True;break
        rows.append(dict(seed=seed,covered_all_times=valid,certified=accepted,labels=t,
                         midpoint_weights=weights_mid,final_weights=a.weights.tolist()))
    out=dict(budget=B,paths=40,rho=.2/(B-1),p_start=[.2,.4],p_end=[.4,.2],
             confidence_mode='known bounded drift Hoeffding; no stationary beta CS',
             all_path_coverage_count=sum(x['covered_all_times'] for x in rows),
             certificates=sum(x['certified'] for x in rows),rows=rows)
    Path(__file__).with_name('cswm_drift_results.json').write_text(json.dumps(out,indent=2))
    print({k:v for k,v in out.items() if k!='rows'})
if __name__=='__main__':main()
