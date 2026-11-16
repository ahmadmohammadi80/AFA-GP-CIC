"""Anytime false-positive stress test on a nontransitive rock-paper-scissors game.
All q have q^T W uniform = 0 exactly; ANY positive certificate is a false claim.
"""
import json
import numpy as np
from afa_cic_joint import Certificate,frontier_candidates

def trial(seed,budget=1500,pilot=25,block=25,delta=.05):
    ref=np.ones(3)/3
    W=np.array([[0,.40,-.40],[-.40,0,.40],[.40,-.40,0.]])
    assert np.allclose(W@ref,0)
    rng=np.random.default_rng(seed+724931)
    cert=Certificate(ref,delta)
    for e,(i,j) in enumerate(cert.edges):
        cert.observe(e,int(rng.binomial(pilot,(W[i,j]+1)/2)),pilot)
    cand=frontier_candidates(ref,.25)
    fixed=cand[int(np.argmax([cert.empirical(q) for q in cand]))]
    t=len(cert.edges)*pilot
    accepted=cert.certify(cand,.25)
    while accepted is None and t<budget:
        batch=min(block,budget-t)
        e=cert.query(cand,method='voi',fixed_candidate=fixed,block=batch)
        i,j=cert.edges[e]
        cert.observe(e,int(rng.binomial(batch,(W[i,j]+1)/2)),batch)
        t+=batch
        accepted=cert.certify(cand,.25)
    l,h=cert.interval()
    covers=all(l[e]<=W[i,j]<=h[e] for e,(i,j) in enumerate(cert.edges))
    return bool(accepted is not None),bool(covers),int(t)

if __name__=='__main__':
    rs=[trial(i) for i in range(200)]
    report={'seed_range':[0,199],'trials':len(rs),'false_positive':sum(x[0] for x in rs),
            'joint_confidence_coverage':sum(x[1] for x in rs),'average_labels':sum(x[2] for x in rs)/len(rs),
            'limit_note':'Observed false-positive count does not prove exact empirical <= delta; theory supplies the bound.'}
    with open('null_results.json','w') as f:json.dump(report,f,indent=2)
    print(report)
