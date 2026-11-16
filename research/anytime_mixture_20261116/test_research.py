import unittest
import numpy as np
from anytime_mixture import edge_cs,hoeffding_radius,edge_coefficients,run_one,experiment
from two_world_sequential import universal_lower,sequential_expected_upper,run_one as seq_run,experiment as seq_exp


class ResearchTests(unittest.TestCase):
    def test_beta_inversion(self):
        for n in [1,2,7,30,151,1000]:
            for s in [0,n//4,n//2,n]:
                lo,hi=edge_cs(s,n,.005)
                self.assertLess(lo,hi)
                self.assertLessEqual(lo,2*s/n-1+1e-12)
                self.assertGreaterEqual(hi,2*s/n-1-1e-12)

    def test_initial_interval(self):
        self.assertEqual(edge_cs(0,0,.05),(-1.,1.))
        with self.assertRaises(ValueError):edge_cs(1,0,.05)

    def test_pairwise_coefficient_identity(self):
        rng=np.random.default_rng(2026)
        for k in [3,5,9]:
            b=rng.dirichlet(np.ones(k));q=rng.dirichlet(np.ones(k))
            H=rng.dirichlet(np.ones(k),size=4)
            z=rng.uniform(-.3,.3,(k,k));W=z-z.T
            edges,A=edge_coefficients(b,q,H)
            mu=np.array([W[i,j] for i,j in edges])
            np.testing.assert_allclose(A@mu,[(q-b)@W@h for h in H],
                                       rtol=0,atol=1e-12)

    def test_adaptive_coverage_and_safety(self):
        for method in ['mixture','hoeffding']:
            for i in range(15):
                v=run_one(seed=i,max_labels=2000,mode=method,threshold=.015)
                self.assertLessEqual(v['labels'],2000)
                self.assertFalse(v['certified'] and v['true_gain']<=.015)
                self.assertLess(v['KL'],.05)

    def test_stationary_null_control(self):
        for method in ['mixture','hoeffding']:
            rows=[run_one(seed=20+i,max_labels=400,mode=method,null=True,threshold=.015)
                  for i in range(25)]
            self.assertEqual(sum(v['certified'] for v in rows),0)

    def test_coupled_matched_budget_power(self):
        r=experiment(seeds=40,max_labels=1510,threshold=.015)
        self.assertGreaterEqual(r['methods']['mixture']['certified_count'],20)
        self.assertLessEqual(r['methods']['hoeffding']['certified_count'],10)
        self.assertEqual(r['methods']['mixture']['false_certificates'],0)
        self.assertEqual(r['methods']['hoeffding']['false_certificates'],0)

    def test_two_world_adaptive_information_lower(self):
        v=universal_lower(.05)
        s=sequential_expected_upper(.05)
        self.assertAlmostEqual(v['all_adaptive_Eplus_lower'],6.93882192847592,places=10)
        self.assertAlmostEqual(v['all_adaptive_Eminus_lower'],7.478817756977083,places=10)
        self.assertLess(s['competitive_plus'],1.7)
        self.assertLess(s['competitive_minus'],1.6)

    def test_sequential_worlds(self):
        out=seq_exp(seeds=1000)
        for world in ['-1','1']:
            self.assertLessEqual(out['families'][world]['errors'],75)
            self.assertLess(out['families'][world]['mean_labels'],12)

    def test_hoeffding_radius_range(self):
        r=hoeffding_radius([1,2,10,100],.05,10)
        self.assertTrue(np.all(np.isfinite(r)))
        self.assertTrue(np.all(r>0))
        self.assertLessEqual(float(r.max()),2.)

    def test_sequential_stopping(self):
        for world in (-1,1):
            for delta in (.05,.02):
                r=seq_run(seed=2026,world=world,delta=delta)
                self.assertLess(r['n'],20000)
                self.assertIn(r['selected'],(-1,1))


if __name__=='__main__':
    unittest.main()
