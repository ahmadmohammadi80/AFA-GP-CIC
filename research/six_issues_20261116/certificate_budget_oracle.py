"""Exact continuous and integer allocation for a single active certificate row.
Oracle benchmark: known nonnegative weights w_e=|A_e|c_e, fixed radius c_e/sqrt(n_e).
This does NOT establish an adaptive sample-complexity improvement or novelty over prior work.
"""
import numpy as np
def allocate(weights,budget):
    w=np.asarray(weights,float)
    if w.ndim!=1 or not np.all(np.isfinite(w)) or np.any(w<0) or budget<=0:raise ValueError("input")
    active=w>0;n=np.zeros_like(w)
    if not active.any():return n,0.
    z=w[active]**(2/3);n[active]=budget*z/z.sum()
    return n,float(np.sum(w[active]/np.sqrt(n[active])))
def optimal_width(weights,budget):
    w=np.asarray(weights,float)
    if w.ndim!=1 or np.any(w<0) or budget<=0:raise ValueError("input")
    return float(np.sum(w**(2/3))**1.5/np.sqrt(budget))
def integer_allocation(weights,budget):
    w=np.asarray(weights,float)
    if w.ndim!=1 or np.any(w<0) or not np.all(np.isfinite(w)) or int(budget)!=budget:raise ValueError("input")
    active=np.flatnonzero(w>0);B=int(budget);m=len(active)
    if m==0:return np.zeros(len(w),int),0.
    if B<m:raise ValueError("insufficient budget")
    dp=np.full((m+1,B+1),np.inf);dp[0,0]=0
    parent=np.zeros((m+1,B+1),int)
    for i,e in enumerate(active,1):
        for b in range(i,B+1):
            for n in range(1,b-i+2):
                v=dp[i-1,b-n]+w[e]/np.sqrt(n)
                if v<dp[i,b]:dp[i,b]=v;parent[i,b]=n
    out=np.zeros(len(w),int);b=B
    for i in range(m,0,-1):
        n=parent[i,b];out[active[i-1]]=n;b-=n
    return out,float(dp[m,B])
if __name__=="__main__":
    import json
    w=np.array([4.,1.,0.,.5]);B=24;n,v=allocate(w,B);ni,vi=integer_allocation(w,B)
    print(json.dumps({"weights":w.tolist(),"budget":B,"continuous_n":n.tolist(),
      "continuous_width":v,"integer_n":ni.tolist(),"integer_width":vi,
      "equal_active_width":float(np.sum(w[w>0]/np.sqrt(B/np.sum(w>0))))},indent=2))
