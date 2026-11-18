"""Tests for the actual beta-mixture integrated exact two-step planner."""
import unittest
import numpy as np
from integrated_two_step_certificate_20261118 import (
    exact_static_min, intervals, plan_two, leaf_certifier,
    greedy_edge, edge_coefficients, canonical_problem, boundary_audit,
)

class IntegratedTwoStepTests(unittest.TestCase):
    def test_exact_inner_vs_dense_grid(self):
        rng=np.random.default_rng(87)
        for _ in range(15):
            low=rng.uniform(.05,.45,size=2)
            high=np.minimum(.95,low+rng.uniform(.15,.4,size=2))
            box=np.column_stack((low,high))
            e,a,b=rng.integers(0,2,3)
            flags=rng.integers(0,2,4)
            opt=exact_static_min(box,int(e),int(a),int(b),flags)
            grid=float('inf')
            for x in np.linspace(*box[0],51):
                for y in np.linspace(*box[1],51):
                    p=[x,y];v=(1-p[e])*((1-p[a])*flags[0]+p[a]*flags[1])+p[e]*((1-p[b])*flags[2]+p[b]*flags[3])
                    grid=min(grid,v)
            self.assertLessEqual(opt,grid+1e-11)
            self.assertLess(grid-opt,.0005)

    @staticmethod
    def setup_three():
        r=np.ones(3)/3;q=np.array([.6,.2,.2])
        assert float(sum(q*np.log(q/r)))<.2
        A=edge_coefficients(r,q,np.array([r]))[1]
        return [A]

    def test_strict_win_over_greedy_same_two_queries(self):
        As=self.setup_three()
        s=np.array([52,82,50]);n=np.array([100,100,100])
        p=plan_two(s,n,As,.05,0.)
        self.assertFalse(p['already_certified'])
        g=greedy_edge(s,n,As,.05)
        v_g=p['best_for_root'][g]['robust_probability']
        v_opt=p['optimal']['robust_probability']
        self.assertEqual(g,0)
        self.assertEqual(p['optimal']['e'],1)
        self.assertAlmostEqual(v_g,0.3422988340611028,places=10)
        self.assertAlmostEqual(v_opt,0.6561798896134118,places=10)
        self.assertGreater(v_opt,v_g)

    def test_actual_two_step_crossing_when_all_one_step_fail(self):
        As=self.setup_three()
        s=np.array([31,95,50]);n=np.array([100,100,100])
        t=leaf_certifier(s,n,As,.05,0.)
        self.assertTrue(all(not t(((e,y),)) for e in range(3) for y in (0,1)))
        v=plan_two(s,n,As,.05,0.)['optimal']['robust_probability']
        self.assertAlmostEqual(v,0.13525227658227387,places=10)
        self.assertGreater(v,0.)

    def test_canonical_5_action_kl_and_prefix_reproducibility(self):
        a=canonical_problem(20261116);b=canonical_problem(20261116)
        self.assertEqual(len(a[0]),10)
        self.assertEqual(len(a[4]),2)
        self.assertEqual(a[4][0].shape,(5,10))
        self.assertTrue(np.array_equal(a[0],b[0]))
        self.assertTrue(np.array_equal(a[3],b[3]))
        self.assertEqual(a[4][0].tolist(),b[4][0].tolist())
        x=boundary_audit(20261116,budget=250)
        self.assertFalse(x['already_certified'])
        self.assertEqual(x['labels'],248)
        self.assertGreaterEqual(x['optimal']['robust_probability'],x['greedy_with_optimal_recourse_value']-1e-12)

if __name__=='__main__':unittest.main()
