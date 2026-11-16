"""Global multiplicity adjustment over all 8 planned 200-seed comparisons.

Two algorithms (unguarded and guarded) x two environment families
(original and sign-cycle) x two outcomes (certificate power and labels).
This is more conservative than adjusting only within each experiment.
"""
from __future__ import annotations
import json,argparse
from pathlib import Path

def holm(raw):
    if not raw or any(not 0<=p<=1 for p in raw.values()):raise ValueError("p-values")
    seq=sorted(raw,key=raw.get);out={};running=0.
    for i,name in enumerate(seq):
        running=max(running,min(1.,(len(seq)-i)*raw[name]))
        out[name]=running
    return out

def summarize(root="results"):
    root=Path(root)
    spec=[("unguarded","original","extended_power_200.json"),
          ("unguarded","cycle","extended_power_200_cyclic.json"),
          ("guarded","original","guarded_original_200.json"),
          ("guarded","cycle","guarded_cyclic_200.json")]
    raw={};details={}
    for method,family,filename in spec:
        data=json.loads((root/filename).read_text())
        for outcome,field in (("power","p_power_exact" if method=="unguarded" else "p_power"),
                              ("labels","p_label_sign_exact" if method=="unguarded" else "p_label_sign")):
            name=f"{method}_{family}_{outcome}"
            raw[name]=float(data["matched" if method=="unguarded" else "paired"][field])
        details[f"{method}_{family}"]=data["summary"]
    return {"n_planned_tests":len(raw),"raw_p":raw,"holm_fwer_p":holm(raw),
            "details":details,"preregistered_family":"2 algorithms x 2 worlds x 2 outcomes"}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",default="results")
    p.add_argument("--out",default="results/global_200_holm.json")
    a=p.parse_args();r=summarize(a.root)
    f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(r,indent=2))
    print(json.dumps({"raw_p":r["raw_p"],"holm_fwer_p":r["holm_fwer_p"]},indent=2))
