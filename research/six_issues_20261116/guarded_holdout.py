"""New untouched 200-seed paired test: proved GUARDED joint vs beta-mixture greedy.

Same worlds, KL candidates, pairwise streams, total delta, tau, budget,
and per-label stopping. Guarded method uses a split-delta intersection
and deterministic coverage queries; it may have lower empirical power
than unguarded joint_mix. Never conflate their theorems/results.
"""
import json,argparse
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from issue1_ablation import trial as baseline
from theory_and_guarantees import guarded_joint

def audit(seeds=200,offset=20290101,cyclic=False):
    gs=[guarded_joint(offset+i,cyclic=cyclic) for i in range(seeds)]
    bs=[baseline(offset+i,"greedy",cyclic=cyclic) for i in range(seeds)]
    win=sum(a["certified"] and not b["certified"] for a,b in zip(gs,bs))
    loss=sum(b["certified"] and not a["certified"] for a,b in zip(gs,bs))
    p_power=float(binomtest(min(win,loss),win+loss,p=.5).pvalue) if win+loss else 1.
    diff=np.array([a["labels"]-b["labels"] for a,b in zip(gs,bs)],float)
    more=int(sum(diff>0));less=int(sum(diff<0))
    p_cost=float(binomtest(min(more,less),more+less,p=.5).pvalue) if more+less else 1.
    ps=sorted([p_power,p_cost])
    a=min(1.,2*ps[0]);bb=max(a,ps[1])
    adj_power=a if p_power<=p_cost else bb
    adj_cost=a if p_cost<=p_power else bb
    return {"setup":{"seeds":seeds,"offset":offset,"cyclic":cyclic,"budget":1510,
                     "guard_period":20,"confidence":.05},
            "summary":{"guarded":{"positive":sum(x["certified"] for x in gs),
                                   "false_certificates":sum(x["false_certificate"] for x in gs),
                                   "mean_labels":float(np.mean([x["labels"] for x in gs])),
                                   "regret_bound_violations_observed":sum(not x["within_bound"] for x in gs)},
                       "greedy":{"positive":sum(x["certified"] for x in bs),
                                 "false_certificates":sum(x["false_certificate"] for x in bs),
                                 "mean_labels":float(np.mean([x["labels"] for x in bs]))}},
            "paired":{"guard_only":win,"greedy_only":loss,
                      "p_power":p_power,"p_power_holm":adj_power,
                      "guard_more_labels":more,"guard_less_labels":less,
                      "p_label_sign":p_cost,"p_label_sign_holm":adj_cost,
                      "mean_label_difference":float(diff.mean())},
            "rows":{"guarded":gs,"greedy":bs}}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=200)
    p.add_argument("--offset",type=int,default=20290101)
    p.add_argument("--cyclic",action="store_true")
    p.add_argument("--out",default="results/guarded_200.json")
    a=p.parse_args();v=audit(a.seeds,a.offset,a.cyclic)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(v,indent=2))
    print(json.dumps({k:z for k,z in v.items() if k!="rows"},indent=2))
