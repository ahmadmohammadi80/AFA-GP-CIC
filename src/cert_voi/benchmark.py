import json
import numpy as np
from afa_cic_joint import run_trial

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--end",type=int,default=60)
    parser.add_argument("--out",default="results.json")
    args=parser.parse_args()
    out={}
    for method in ['voi','voi_fixed','joint','fixed','variance','uniform']:
        rows=[run_trial(i,method,1500,25,20) for i in range(args.start,args.end)]
        out[method]={'rows':rows,'mean_queries':float(np.mean([r['comparisons'] for r in rows])),
            'median_queries':float(np.median([r['comparisons'] for r in rows])),
            'certified':int(sum(r['certified'] for r in rows)),
            'mean_true_gain':float(np.mean([r['true_gain'] for r in rows])),
            'coverage':int(sum(r['covers'] for r in rows)),
            'violations':int(sum(r['true_gain']<-1e-9 or r['returned_kl']>.250001 for r in rows))}
    with open(args.out,'w') as f:json.dump(out,f,indent=2)
    for m,v in out.items():print(m,{k:round(x,4) if isinstance(x,float) else x for k,x in v.items() if k!='rows'})
if __name__=='__main__':main()
