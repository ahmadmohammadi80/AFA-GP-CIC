import unittest
import numpy as np
from edge_allocation import (coefficients,one_opponent_exact,dual_bound,design,certificate,run)

class CertificateAllocationTests(unittest.TestCase):
    def test_edge_identity(self):
        r=np.random.default_rng(12)
        for k in [4,8,12]:
            b=np.ones(k)/k;q=r.dirichlet(np.ones(k))
            H=r.dirichlet(np.ones(k),size=5)
            X=r.uniform(-.5,.5,(k,k));W=X-X.T
            edges,A=coefficients(b,q,H)
            mu=np.array([W[i,j] for i,j in edges])
            np.testing.assert_allclose(A@mu,[(q-b)@W@h for h in H],atol=1e-12)
    def test_exact_one_opponent_kkt(self):
        w=np.array([1.,.2,.05,.01])
        n=one_opponent_exact(w,.025)
        self.assertAlmostEqual(float(np.sum(w/np.sqrt(n))),.025,places=10)
        S=float(np.sum(w**(2/3)))
        self.assertAlmostEqual(float(n.sum()),S**3/.025**2,places=8)
    def test_design_matches_one_opponent(self):
        A=np.array([[.5,.15,.04,.005]])
        B=np.array([.025])
        r=design(A,B)
        self.assertLess(abs(r['continuous_cost']-one_opponent_exact(A[0],B[0]).sum()),.2)
        self.assertEqual(int(r['optimal_counts'].sum()),int(r['uniform_counts'].sum()))
        self.assertLessEqual(r['global_lower'],r['continuous_cost']+1e-5)
    def test_multiopponent_feasible_and_weak_duality(self):
        A=np.array([[.42,.07,.01,.04],[.05,.35,.15,.07],[.12,.08,.19,.2]])
        B=np.array([.012,.019,.014])
        o=design(A,B)
        self.assertTrue(np.all(np.abs(A)@o['optimal_counts'].astype(float)**(-.5)<=B+1e-8))
        for lam in [[0,0,0],[1,2,3],[50,1,20],[10000,15000,5000]]:
            self.assertLessEqual(dual_bound(A,B,lam),o['continuous_cost']+1e-4)
    def test_independent_certificates_conservative(self):
        r=np.random.default_rng(191)
        A=np.array([[.07,.01,.02],[-.02,.03,.04]])
        mu=np.array([.2,-.25,.5]);N=np.array([900,500,800])
        true=float((A@mu).min())
        for _ in range(100):
            m=np.array([2*r.binomial(int(n),(v+1)/2)/n-1 for n,v in zip(N,mu)])
            self.assertLessEqual(certificate(A,m,N,.05),true+1e-12)
    def test_cycle_information_not_borda(self):
        k=6;b=np.ones(k)/k;q=b.copy()
        q[0]+=.05;q[1]-=.05;q[2]+=.02;q[3]-=.02
        H=np.eye(k)[2:5]
        W=np.zeros((k,k));C=W.copy()
        for i,j in [(2,3),(3,4),(4,2)]:
            C[i,j]+=.1;C[j,i]-=.1
        np.testing.assert_allclose(C@b,0,atol=1e-12)
        self.assertFalse(np.allclose([(q-b)@W@h for h in H],
                                     [(q-b)@(W+C)@h for h in H]))
    def test_run_reproducible(self):
        a=run(trials=10,seed=35,pilot=100)
        b=run(trials=10,seed=35,pilot=100)
        self.assertEqual(a,b)
        self.assertGreater(a['planned_trials'],0)
        self.assertEqual(a['trial_count'],10)
        for r in a['rows']:
            if r['planned']:
                self.assertGreater(r['confirm_budget'],0)
                self.assertTrue(np.isfinite(r['our_lower']))
    def test_input_rejections(self):
        with self.assertRaises(ValueError):one_opponent_exact([.2,-.1],.1)
        with self.assertRaises(ValueError):design([[.1,.2]],[0])
        with self.assertRaises(ValueError):certificate([[1]],np.array([.1]),np.array([0]),.05)

if __name__=="__main__":unittest.main()
