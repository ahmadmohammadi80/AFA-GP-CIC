"""Independent finite-grid cross-check for 3-action KL robust optimization."""
import json
import numpy as np
from certified_solver import solve

def brute_grid(b,lo,hi,kappa,resolution=400):
    a=np.arange(resolution+1)/resolution
    x,y=np.meshgrid(a,a,indexing='ij')
    x,y=x.ravel(),y.ravel()
    keep=x+y<=1.
    q=np.column_stack((x[keep],y[keep],1-x[keep]-y[keep]))
    m=q>0; logp=np.zeros_like(q)
    logp[m]=np.log(q[m]/np.broadcast_to(b,q.shape)[m])
    q=q[(q*logp).sum(axis=1)<=kappa+1e-12]
    d=q-b; val=np.where(d>=0,d*lo,d*hi).sum(axis=1)
    return float(val.max()),int(len(q))

def main():
    results=[]
    for seed in range(16):
        rng=np.random.default_rng(seed+1140)
        b=rng.dirichlet([2.,2.,2.]);z=rng.uniform(-.7,.7,3);e=rng.uniform(.01,.2,3)
        lo=np.maximum(-1,z-e);hi=np.minimum(1,z+e)
        r=solve(b,lo,hi,.08)
        v,num=brute_grid(b,lo,hi,.08)
        results.append({'seed':seed,'primal':r.primal,'dual':r.dual_upper,'grid':v,'gap':r.gap,
                        'grid_feasible':num,'passed':v<=r.dual_upper+1e-9 and r.primal-v<.003})
    assert all(r['passed'] for r in results)
    with open('independent_grid_audit.json','w') as f:json.dump(results,f,indent=2)
    print({'grid_cases':len(results),'all_passed':True,'max_diff':max(r['primal']-r['grid'] for r in results)})
if __name__=='__main__':main()
