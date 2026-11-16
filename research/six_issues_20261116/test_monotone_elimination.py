import unittest
import numpy as np
from monotone_elimination import MonotoneEliminator,make_coefficients,witness

class MonotoneTests(unittest.TestCase):
    def test_dynamic_support_witness(self):
        x=witness()
        self.assertEqual(x["survivors"],[0])
        self.assertEqual(x["final_edges"],[0])
    def test_never_eliminate_optimal_under_coverage(self):
        rng=np.random.default_rng(721)
        for _ in range(120):
            k=5;b=np.ones(k)/k
            Q=rng.dirichlet(np.ones(k),size=6);H=rng.dirichlet(np.ones(k),size=4)
            A,edges=make_coefficients(b,Q,H)
            W=rng.uniform(-1,1,len(edges))
            gain=np.min(A@W,axis=1)
            optimal=int(np.argmax(gain))
            state=MonotoneEliminator(A)
            for step in range(90):
                e=int(rng.integers(len(edges)))
                # Arbitrarily varying valid confidence intervals, not
                # necessarily nested; intersection must preserve true W.
                rad=float(rng.uniform(.001,.4))
                center=W[e]+rng.uniform(-rad,rad)
                lo=max(-1,center-rad);hi=min(1,center+rad)
                self.assertTrue(lo<=W[e]<=hi)
                self.assertTrue(state.update(e,lo,hi))
                self.assertTrue(state.alive[optimal])
                L,U=state.bounds()
                self.assertTrue(np.all(L<=gain+1e-12))
                self.assertTrue(np.all(gain<=U+1e-12))
                self.assertTrue(state.result()["simple_regret_upper"]>=-1e-12)
    def test_monotone_support_and_candidates(self):
        A=np.eye(4).reshape(4,1,4)
        x=MonotoneEliminator(A);last=4;last_edges=4
        for e in range(4):
            x.update(e,.7 if e==0 else -.9,.8 if e==0 else -.8)
            self.assertLessEqual(int(x.alive.sum()),last)
            self.assertLessEqual(int(x.active_edges().sum()),last_edges)
            last=int(x.alive.sum());last_edges=int(x.active_edges().sum())
    def test_conflicting_interval_is_not_accepted(self):
        A=np.ones((1,1,1))
        x=MonotoneEliminator(A)
        x.update(0,.2,.4)
        self.assertFalse(x.update(0,-.8,-.7))
        self.assertEqual(x.lo[0],.2)
    def test_eliminated_policy_can_not_revive(self):
        x=MonotoneEliminator(np.eye(3).reshape(3,1,3))
        x.update(0,.8,.9)
        x.update(1,-.9,-.8)
        self.assertFalse(x.alive[1])
        x.update(1,.5,.9) # contradictory; fail closed
        self.assertFalse(x.alive[1])

if __name__=="__main__":unittest.main()
