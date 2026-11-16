"""Check mathematical claims and simulation reproducibility (stdlib only)."""
import math
import unittest
from cycle_gap import (matrix,score,kl,gains,margin,binary_kl,
                       plus_witness,no_world_upper_bound,min_q0,
                       needed_labels,predict_from_labels,run_trials,
                       THRESHOLD,B,KAPPA)

class ProofChecks(unittest.TestCase):
    def test_two_worlds_are_valid_and_cyclic(self):
        for c in (-1,1):
            W=matrix(c)
            self.assertTrue(all(abs(W[i][j])<=1 for i in range(3) for j in range(3)))
            self.assertTrue(all(abs(W[i][j]+W[j][i])<1e-12 for i in range(3) for j in range(3)))
            self.assertGreater(W[0][1]*W[1][2]*W[2][0],0)

    def test_exact_score_collapse(self):
        splus=score(matrix(+1))
        sminus=score(matrix(-1))
        for x,y in zip(splus,sminus):
            self.assertAlmostEqual(x,y,places=13)
        self.assertAlmostEqual(splus[0],.1)
        self.assertAlmostEqual(splus[1],0.)
        self.assertAlmostEqual(splus[2],-.1)

    def test_witness_passes_in_plus_world(self):
        q=plus_witness()
        self.assertAlmostEqual(sum(q),1)
        self.assertLessEqual(kl(q),KAPPA)
        self.assertAlmostEqual(margin(q,matrix(+1)),1/12)
        self.assertGreater(margin(q,matrix(+1)),THRESHOLD)

    def test_minus_upper_bound_is_below_threshold(self):
        u=no_world_upper_bound()
        self.assertAlmostEqual(u,.0586518106657468,places=9)
        self.assertLess(u,THRESHOLD)

    def test_dual_identity_minus_world(self):
        W=matrix(-1)
        q=(.2,.45,.35)
        gs=gains(q,W)
        self.assertAlmostEqual(.25*gs[0]+.75*gs[1],.3*(1/3-q[0]),places=13)
        self.assertLessEqual(min(gs),.25*gs[0]+.75*gs[1]+1e-12)

    def test_data_processing_numeric(self):
        q0min=min_q0()
        self.assertAlmostEqual(binary_kl(q0min),KAPPA,places=12)
        for q0 in (q0min,.2,.3,1/3,.45,.6):
            r=(1-q0)/2
            q=(q0,r,r)
            self.assertGreaterEqual(kl(q)+1e-12,binary_kl(q0))
            if kl(q)<=KAPPA:
                self.assertGreaterEqual(q0+1e-12,q0min)

    def test_sample_size_bound(self):
        n=needed_labels()
        self.assertEqual(n,38)
        self.assertLessEqual(math.exp(-n*.4**2/2),.05)

    def test_true_world_and_false_world(self):
        self.assertEqual(predict_from_labels(38,38),1)
        self.assertEqual(predict_from_labels(0,38),-1)
        with self.assertRaises(ValueError):predict_from_labels(0,0)

    def test_reproducible_monte_carlo(self):
        a=run_trials(2000,seed=101)
        b=run_trials(2000,seed=101)
        self.assertEqual(a,b)
        self.assertLess(a['worlds']['-1']['error_rate'],.05)
        self.assertLess(a['worlds']['1']['error_rate'],.05)

if __name__=='__main__':unittest.main()
