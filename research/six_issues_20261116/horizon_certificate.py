"""HCQA: exact finite-horizon minimax certificate-width allocation.
Only fixed, known, nonnegative coefficient magnitudes A and deterministic
r_e(n)=c_e/sqrt(n) are covered. No optimality of random stopping time claimed.
"""
import itertools
import numpy as np

def width(A,n,c):
    A=np.asarray(A,float);n=np.asarray(n,int);c=np.asarray(c,float)
    return float(np.max((A*c/np.sqrt(n)).sum(axis=1)))

def oracle(A,n,c,budget):
    A=np.asarray(A,float);n=np.asarray(n,int);c=np.asarray(c,float)
    if A.ndim!=2 or len(n)!=A.shape[1] or len(c)!=len(n) or np.any(A<0) or np.any(n<1) or np.any(c<0) or budget<0:
        raise ValueError("invalid inputs")
    e=len(n);best=[float("inf"),None];nodes=[0,0]
    def recurse(i,left,chosen):
        nodes[0]+=1
        if i==e-1:
            v=np.array(chosen+[left],int);obj=width(A,n+v,c)
            if obj<best[0]-1e-13 or (abs(obj-best[0])<=1e-13 and (best[1] is None or tuple(v)<tuple(best[1]))):
                best[:]=[obj,v]
            return
        optimistic=n.copy()
        optimistic[:i]+=np.array(chosen,int)
        optimistic[i:]+=left
        if width(A,optimistic,c)>best[0]+1e-12:
            nodes[1]+=1;return
        for x in range(left+1):recurse(i+1,left-x,chosen+[x])
    recurse(0,budget,[])
    return {"allocation":best[1].tolist(),"minimax_width":best[0],"nodes":nodes[0],"pruned":nodes[1]}

def greedy(A,n,c,budget):
    n=np.asarray(n,int).copy();alloc=np.zeros(len(n),int)
    for _ in range(budget):
        costs=[width(A,n+np.eye(len(n),dtype=int)[i],c) for i in range(len(n))]
        i=int(np.argmin(costs));n[i]+=1;alloc[i]+=1
    return {"allocation":alloc.tolist(),"minimax_width":width(A,n,c)}

def exhaustive(A,n,c,budget):
    E=len(n);best=float("inf")
    for seq in itertools.product(range(E),repeat=budget):
        alloc=np.bincount(seq,minlength=E)
        best=min(best,width(A,n+alloc,c))
    return best

def audit():
    rng=np.random.default_rng(20261116)
    for _ in range(80):
        A=rng.uniform(0,2,(3,3));n=rng.integers(1,6,3)
        c=rng.uniform(.2,1.5,3);b=4
        out=oracle(A,n,c,b)
        assert abs(out["minimax_width"]-exhaustive(A,n,c,b))<1e-10
        assert sum(out["allocation"])==b
    witness=None
    for seed in range(1000):
        r=np.random.default_rng(seed);A=r.uniform(.1,3,(3,4))
        n=np.ones(4,int);c=np.ones(4);b=6
        o=oracle(A,n,c,b);g=greedy(A,n,c,b)
        if o["minimax_width"]+1e-7<g["minimax_width"]:
            witness={"seed":seed,"A":A.tolist(),"budget":b,"oracle":o,"greedy":g}
            break
    assert witness is not None
    return {"random_exact_checks":80,"greedy_counterexample":witness}

if __name__=="__main__":
    import json
    print(json.dumps(audit(),indent=2))
