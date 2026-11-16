import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import unittest
import numpy as np
from certified_solver import solve,kl,dual,lower,mix_to_kl
from strong_adaptive_study import cycle_instance,sample_confidence,oracle_kl_policy,run

class VerifiedOptimizer(unittest.TestCase):
    def test_input_rejections(self):
        b=np.array([.5,.5])
        for bad in [np.array([.5,0.5,0.]),np.array([0.,1.]),np.array([.3,.3]),np.array([float('nan'),.5])]:
            with self.assertRaises(ValueError):solve(bad,np.full(len(bad),-.5),np.full(len(bad),.5),.1)
        with self.assertRaises(ValueError):solve(b,[-1,0],[0,-.1],.2)
        with self.assertRaises(ValueError):solve(b,[-1,0],[0,1],-.01)

    def test_exact_zero_budget(self):
        b=np.array([.1,.2,.7]);r=solve(b,[-1,-.2,.3],[.3,.9,.9],0.)
        self.assertTrue(np.allclose(r.q,b));self.assertEqual(r.primal,0.)
        self.assertEqual(r.dual_upper,0.)

    def test_exact_zero_overlapping_intervals(self):
        b=np.array([.1,.2,.7]);r=solve(b,[-.5,-.4,-.3],[.2,.3,.4],.2)
        self.assertTrue(r.success);self.assertEqual(r.primal,0.)
        self.assertEqual(r.dual_upper,0.);self.assertEqual(r.gap,0.)

    def test_random_certified_gap(self):
        for seed in range(45):
            rng=np.random.default_rng(seed)
            k=int(rng.integers(3,28));b=rng.dirichlet(.5+np.ones(k)*rng.random())
            center=rng.uniform(-.8,.8,k); width=rng.uniform(.001,.3,k)
            lo=np.maximum(-1.,center-width);hi=np.minimum(1.,center+width)
            kappa=float(rng.uniform(.002,.3))
            sol=solve(b,lo,hi,kappa)
            self.assertTrue(np.all(sol.q>=0.))
            self.assertAlmostEqual(sol.q.sum(),1.,places=8)
            self.assertLessEqual(kl(sol.q,b),kappa+1e-10)
            self.assertLessEqual(sol.primal,sol.dual_upper+2e-8)
            self.assertLess(sol.gap,2e-4,f'seed {seed}, k={k}, gap={sol.gap}')

    def test_known_scores_exact_oracle(self):
        for k in [6,12,24,48]:
            b,W,s=cycle_instance(k)
            res=solve(b,s,s,.04)
            oracle=oracle_kl_policy(s,b,.04)
            self.assertLess(res.gap,1e-5)
            self.assertAlmostEqual(res.primal,float(oracle@s),places=5)

    def test_unknown_scores_gap(self):
        for k in [6,12,24,48]:
            b,W,s=cycle_instance(k)
            lo=s-.09;hi=s+.09
            res=solve(b,lo,hi,.04)
            self.assertTrue(res.success);self.assertLess(res.gap,2e-4)
            self.assertLessEqual(res.primal,float((res.q-b)@s)+1e-10)
            self.assertAlmostEqual(res.primal,lower(res.q,b,lo,hi),places=9)
            self.assertAlmostEqual(res.dual_upper,dual(res.s_dual,res.eta,b,.04),places=9)

    def test_any_dual_certificate_is_upper(self):
        rng=np.random.default_rng(123)
        for k in [3,5,20]:
            b=rng.dirichlet(np.ones(k));s=rng.uniform(-.5,.5,k)
            q=oracle_kl_policy(s,b,.1)
            for z in [.01,.1,1.,10.]:
                self.assertLessEqual(float((q-b)@s),dual(s,z,b,.1)+1e-10)

    def test_empirical_time_uniform_bounds(self):
        # Statistical regression check. The formal guarantee is proven separately.
        b,W,s=cycle_instance(12);violations=0
        for seed in range(80):
            rng=np.random.default_rng(seed+10)
            for n in [5,25,120,700]:
                _,lo,hi=sample_confidence(rng,W,b,n,.05)
                if np.any(lo>s+1e-12) or np.any(hi<s-1e-12):violations+=1
        self.assertEqual(violations,0)

    def test_synthetic_old_api_compatibility(self):
        r=run(k=12,n=1200,seeds=4,kappa=.04,gamma=.04)
        self.assertEqual(r['number_trials'],4)
        self.assertEqual(r['numerically_certified_optimality_count'],4)
        self.assertLess(r['max_primal_dual_gap'],2e-4)

    def test_positive_certificate_necessary_and_sufficient(self):
        b=np.array([.25,.25,.25,.25]);lo=np.array([-.3,.12,-.2,-.3]);hi=np.array([.03,.5,.01,.02])
        res=solve(b,lo,hi,.03)
        self.assertGreater(lo.max(),hi.min());self.assertGreater(res.primal,0.)
        self.assertLess(res.gap,2e-4)

    def test_project_to_kl(self):
        b=np.array([.2,.3,.5]);q=np.array([.95,.03,.02]);p=mix_to_kl(q,b,.04)
        self.assertLessEqual(kl(p,b),.04+1e-11)
        self.assertAlmostEqual(sum(p),1.,places=10)

if __name__=='__main__':unittest.main()
