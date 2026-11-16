"""Independent larger powered paired confirmation of joint_mix vs greedy.
Frozen hyperparameter 0.70/0.30 from the earlier 40-seed pilot.
No seed selection or resampling until finding a favorable p-value.
"""
import json,argparse
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from issue1_ablation import trial

def audit(seeds=200,offset=20280101,cyclic=False):
    methods=("greedy","joint_mix")
    rows={m:[trial(offset+i,m,cyclic=cyclic) for i in range(seeds)] for m in methods}
    a,b=rows["joint_mix"],rows["greedy"]
    plus=sum(x["certified"] and not y["certified"] for x,y in zip(a,b))
    minus=sum(y["certified"] and not x["certified"] for x,y in zip(a,b))
    p_power=float(binomtest(min(plus,minus),plus+minus,p=.5).pvalue) if plus+minus else 1.
    dif=np.array([x["labels"]-y["labels"] for x,y in zip(a,b)],float)
    extra=int((dif>0).sum());saved=int((dif<0).sum())
    p_cost=float(binomtest(min(extra,saved),extra+saved,p=.5).pvalue) if extra+saved else 1.
    # Familywise correction over two prespecified co-primary comparisons.
    adjusted_power=min(1.,p_power*2) if p_power<=p_cost else max(p_power,p_cost)
    adjusted_cost=min(1.,p_cost*2) if p_cost<=p_power else max(p_cost,p_power)
    return {"setup":{"seeds":seeds,"offset":offset,"cyclic":cyclic,"budget":1510,
                     "frozen_joint_weight":.7},
            "summary":{m:{"certified":sum(r["certified"] for r in rs),
                          "mean_labels":float(np.mean([r["labels"] for r in rs])),
                          "false_certificates":sum(r["false_certificate"] for r in rs)}
                       for m,rs in rows.items()},
            "matched":{"joint_only":plus,"greedy_only":minus,
                       "p_power_exact":p_power,"p_power_holm_two":adjusted_power,
                       "joint_uses_more_labels_in":extra,"joint_uses_less_labels_in":saved,
                       "p_label_sign_exact":p_cost,"p_label_sign_holm_two":adjusted_cost,
                       "mean_label_difference_joint_minus_greedy":float(dif.mean())},
            "rows":rows}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=200)
    p.add_argument("--offset",type=int,default=20280101)
    p.add_argument("--cyclic",action="store_true")
    p.add_argument("--out",default="results/extended_power.json")
    a=p.parse_args();r=audit(a.seeds,a.offset,a.cyclic)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2))
    print(json.dumps({"setup":r["setup"],"summary":r["summary"],"matched":r["matched"]},indent=2))
