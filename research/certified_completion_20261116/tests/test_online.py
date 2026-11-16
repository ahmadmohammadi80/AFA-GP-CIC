import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from adaptive_certification import Controller
from strong_adaptive_study import cycle_instance

class OnlineTests(unittest.TestCase):
    def test_stop_on_confirmed_certificate(self):
        b,W,s=cycle_instance(6)
        r=Controller(b,block=500,kappa=.04,gamma=.015).fit(W,seed=7,budget=20000)
        self.assertTrue(r.certified);self.assertLessEqual(r.kl,.04000001)
        self.assertGreater(r.true_gain,r.lower_bound-1e-10)
        self.assertLess(r.labels,20000)

    def test_budget_and_indices(self):
        b,W,s=cycle_instance(6)
        for budget in [1,41,199,204]:
            c=Controller(b,block=17)
            r=c.fit(W,seed=13,budget=budget)
            self.assertLessEqual(r.labels,budget)
            self.assertEqual(sum(r.counts),r.labels)

    def test_negative_cycle_has_no_genuine_gain(self):
        b=np.ones(3)/3
        W=np.array([[0,.4,-.4],[-.4,0,.4],[.4,-.4,0]])
        self.assertTrue(np.allclose(W@b,0))
        for seed in range(16):
            r=Controller(b,block=40,kappa=.08,gamma=.015).fit(W,seed=seed,budget=800)
            self.assertFalse(r.certified)
            self.assertAlmostEqual(r.true_gain,0.,places=12)

    def test_observe_validation(self):
        c=Controller(np.array([.3,.7]))
        for j,y in [(5,[1]),(-1,[0]),(0,[2]),(0,[])]:
            with self.assertRaises(ValueError):c.observe(j,y)

    def test_zero_budget(self):
        b,W,s=cycle_instance(6)
        r=Controller(b).fit(W,seed=0,budget=0)
        self.assertEqual(r.labels,0);self.assertFalse(r.certified)
        self.assertAlmostEqual(r.kl,0.)

if __name__=='__main__':unittest.main()
