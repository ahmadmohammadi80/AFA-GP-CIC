import unittest
import numpy as np
from finite_horizon import (Settings,B,TRUE_P,PRIORS,policy_list,kl,true_win_matrix,
    radius,lower_w,certifiable,plan,exact_true_success,exact_static_success)

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=Settings(n=400)
        cls.p=plan(cls.cfg)

    def test_valid_cycle(self):
        W=true_win_matrix()
        self.assertAlmostEqual(W[1,2],.55)
        self.assertAlmostEqual(W[2,3],.55)
        self.assertAlmostEqual(W[3,1],.55)
        self.assertTrue(np.allclose(W,-W.T))

    def test_certificate_algebra_including_cycle(self):
        W=true_win_matrix()
        qa,qb=policy_list(self.cfg)
        self.assertAlmostEqual(qa@W@B,self.cfg.lam*B[0]*(2*TRUE_P[0]-1),places=12)
        self.assertAlmostEqual(qb@W@B,self.cfg.lam*B[0]*((2*TRUE_P[1]-1)+(2*TRUE_P[2]-1))/2,places=12)

    def test_kl(self):
        for q in policy_list(self.cfg):
            self.assertAlmostEqual(sum(q),1.,places=12)
            self.assertLessEqual(kl(q,B),.06)

    def test_preference_cycle_not_scalar(self):
        W=true_win_matrix()
        self.assertGreater(W[1,2]+W[2,3]+W[3,1],1.)

    def test_myopic_and_nonmyopic_distinct(self):
        self.assertEqual(self.p['greedy_first'],0)
        self.assertIn(self.p['lookahead_first'],(1,2))
        self.assertGreater(self.p['root_values'][1],self.p['root_values'][0])

    def test_true_success_gap(self):
        v=exact_true_success(self.p,self.p['lookahead_first'],self.cfg)
        g=exact_true_success(self.p,self.p['greedy_first'],self.cfg)
        self.assertGreater(v,.999)
        self.assertLess(g,.06)

    def test_strong_static_competitor_matches(self):
        v=exact_true_success(self.p,self.p['lookahead_first'],self.cfg)
        static=exact_static_success(1,2,self.cfg)
        self.assertAlmostEqual(v,static,places=10)

    def test_no_one_batch_certificate_for_partial_B(self):
        for e in (1,2):
            for x in (0,100,200,300,400):
                counts=np.zeros(3,dtype=int);wins=np.zeros(3,dtype=int)
                counts[e]=400;wins[e]=x
                self.assertFalse(certifiable(counts,wins,self.cfg))

    def test_radii(self):
        self.assertGreater(radius(400,self.cfg),0)
        self.assertAlmostEqual(lower_w(0,0,self.cfg),-1)

    def test_simultaneous_interval_event_guarantees_acceptance(self):
        rng=np.random.default_rng(10)
        W=true_win_matrix()
        for _ in range(100):
            ns=np.array([rng.integers(0,1001) for _ in range(3)])
            xs=np.array([rng.binomial(int(ns[i]),float(TRUE_P[i])) for i in range(3)])
            good=True
            for i in range(3):
                lo=lower_w(int(xs[i]),int(ns[i]),self.cfg)
                if lo>W[i+1,0]+1e-12:good=False
            if good and certifiable(ns,xs,self.cfg):
                self.assertGreater(max([q@W@B for q in policy_list(self.cfg)]),0)

if __name__=='__main__':unittest.main()
