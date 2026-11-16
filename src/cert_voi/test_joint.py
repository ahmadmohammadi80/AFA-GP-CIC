import unittest
import numpy as np
from afa_cic_joint import Certificate,coeffs,pair_probability_matrix,generated_problem,frontier_candidates,kl,run_trial

class Checks(unittest.TestCase):
    def test_identity_and_cycle(self):
        b,W=generated_problem(13)
        q=frontier_candidates(b,.25)[1]
        a=coeffs(q,b)
        x=np.array([W[i,j] for i in range(len(b)) for j in range(i+1,len(b))])
        self.assertAlmostEqual(a@x,q@W@b,places=12)
        self.assertTrue(W[0,1]>0 and W[1,2]>0 and W[2,0]>0)
    def test_kl(self):
        b,_=generated_problem(1)
        for q in frontier_candidates(b,.25):
            self.assertLessEqual(kl(q,b),.25000001)
    def test_uncertainty(self):
        b,W=generated_problem(1)
        c=Certificate(b,.01)
        l,h=c.interval()
        self.assertTrue(np.all(l==-1)); self.assertTrue(np.all(h==1))
        for e,(i,j) in enumerate(c.edges): c.observe(e,int(100000*(1+W[i,j])/2),100000)
        l,h=c.interval()
        self.assertTrue(all(l[e]<=W[i,j]<=h[e] for e,(i,j) in enumerate(c.edges)))
    def test_certification_safety(self):
        for seed in range(12):
            b,W=generated_problem(seed)
            r=run_trial(seed,budget=900)
            self.assertLessEqual(r['returned_kl'],.250001)
            self.assertGreaterEqual(r['true_gain'],-1e-12)
    def test_confidence_bound_arbitrary_q(self):
        b,W=generated_problem(29)
        c=Certificate(b,.05)
        rng=np.random.default_rng(2026)
        for e,(i,j) in enumerate(c.edges):
            n=1000
            c.observe(e,int(rng.binomial(n,(1+W[i,j])/2)),n)
        lo,hi=c.interval()
        self.assertTrue(all(lo[e]<=W[i,j]<=hi[e] for e,(i,j) in enumerate(c.edges)))
        for q in frontier_candidates(b,.3):
            self.assertLessEqual(c.lower(q),q@W@b+1e-10)
            self.assertGreaterEqual(c.upper(q),q@W@b-1e-10)
    def test_zero_budget(self):
        b,_=generated_problem(4)
        self.assertTrue(np.allclose(frontier_candidates(b,0),b))
    def test_posterior_query_valid(self):
        b,_=generated_problem(11)
        c=Certificate(b,.05)
        qs=frontier_candidates(b,.3)
        for e in range(len(c.edges)):c.observe(e,9,20)
        edge=c.query(qs,'voi',block=4)
        self.assertTrue(0<=edge<len(c.edges))
    def test_direct_cyclic_not_scalar(self):
        b,W=generated_problem(11)
        self.assertGreater(W[0,1]+W[1,2]+W[2,0],.1)

if __name__=='__main__':unittest.main()
