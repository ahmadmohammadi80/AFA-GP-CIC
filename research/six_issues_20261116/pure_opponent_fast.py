"""Issue 6: exact pure-opponent certificate in O(Q*K^2), not O(Q*K^3).

For H=I_K, each edge (i,j) contributes ONLY to opponent i or j:
 A(q,h=j,e)=q_i-b_i and A(q,h=i,e)=-(q_j-b_j).
Computing all H rows as a dense Q*K*E tensor is unnecessary. This method
is exact, uses O(Q*K) auxiliary score memory, and retains cyclic W.
"""
import numpy as np
from itertools import combinations
import time,tracemalloc,json,argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
from anytime_mixture import edge_coefficients,certificate

def fast_pure(b,Q,lo,hi):
    b=np.asarray(b,float);Q=np.asarray(Q,float)
    if Q.ndim!=2 or Q.shape[1]!=len(b):raise ValueError("Q shape")
    k=len(b);E=k*(k-1)//2
    lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if len(lo)!=E or len(hi)!=E or np.any(lo>hi):raise ValueError("interval")
    d=Q-b;score=np.zeros((len(Q),k),dtype=float)
    for e,(i,j) in enumerate(combinations(range(k),2)):
        ai=d[:,i];aj=-d[:,j]
        score[:,j]+=np.where(ai>=0,ai*lo[e],ai*hi[e])
        score[:,i]+=np.where(aj>=0,aj*lo[e],aj*hi[e])
    return score.min(axis=1)

def dense_reference(b,Q,lo,hi):
    H=np.eye(len(b))
    return np.array([certificate(edge_coefficients(b,q,H)[1],lo,hi) for q in Q])

def run(seed=2026,k=10,q_count=5,repetitions=15):
    rng=np.random.default_rng(seed);b=rng.dirichlet(np.ones(k))
    Q=rng.dirichlet(np.ones(k),size=q_count)
    E=k*(k-1)//2;center=rng.uniform(-.6,.6,E)
    half=rng.uniform(.04,.22,E);lo=np.maximum(-1.,center-half);hi=np.minimum(1.,center+half)
    a=fast_pure(b,Q,lo,hi);z=dense_reference(b,Q,lo,hi)
    if not np.allclose(a,z,rtol=0,atol=1e-12):raise AssertionError("different certificate")
    # CPU and peak Python allocation measurements; record hardware in GitHub
    # Actions provenance. Not a deterministic algorithmic guarantee.
    measures={}
    for name,func in (("dense",dense_reference),("pure_fast",fast_pure)):
        tracemalloc.start();start=time.perf_counter()
        for _ in range(repetitions):func(b,Q,lo,hi)
        seconds=time.perf_counter()-start
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        measures[name]={"seconds_per_call":seconds/repetitions,"peak_python_bytes":peak}
    return {"seed":seed,"k":k,"edges":E,"q":q_count,
            "max_numeric_error":float(np.max(np.abs(a-z))),
            "speedup":float(measures["dense"]["seconds_per_call"]/max(1e-12,measures["pure_fast"]["seconds_per_call"])),
            "memory_reduction":float(measures["dense"]["peak_python_bytes"]/max(1,measures["pure_fast"]["peak_python_bytes"])),
            "measures":measures}

def audit():
    rows=[run(k=k,repetitions=(12 if k<=10 else (4 if k<=20 else 1))) for k in [5,10,20,40]]
    return {"runs":rows,"scope":"pure opponents only; exact O(QK^2) rather than dense O(QK^3)"}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/scale_pure.json")
    a=p.parse_args();r=audit();dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(r,indent=2))
    # Measured scaling plot; values and hardware provenance accompany the PNG.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks=[z["k"] for z in r["runs"]]
    for m in ("dense","pure_fast"):
        plt.plot(ks,[1000*z["measures"][m]["seconds_per_call"] for z in r["runs"]],
                 marker="o",label=m)
    plt.xlabel("Number of actions K")
    plt.ylabel("Measured milliseconds per certificate")
    plt.yscale("log");plt.legend();plt.tight_layout()
    plt.savefig(dest.with_suffix(".png"),dpi=180)
    plt.close()
    print(json.dumps(r,indent=2))
