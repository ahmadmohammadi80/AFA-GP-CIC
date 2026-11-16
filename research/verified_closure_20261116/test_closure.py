import math
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
root=Path(__file__).resolve().parent.parent
base=root/'certified_completion_20261116'
if not base.exists():base=root/'afa_certified_completion_20261116'
sys.path.insert(0,str(base))
from certified_solver import solve
from validated_enclosures import certify,kl_interval,lower_exact,interval,from_mpf_tuple
from verified_solver import solve_verified
from sample_complexity import confidence_radius,n_for_radius,safe_certificate_budget
from strong_adaptive_study import cycle_instance

class ClosureTests(unittest.TestCase):
    def test_audit_random_feasible_and_dual(self):
        rng=np.random.default_rng(20261116)
        for k in [2,3,6,12,24]:
            for seed in range(8):
                b=rng.dirichlet(np.ones(k))
                mid=rng.uniform(-.6,.6,k)
                width=rng.uniform(.001,.25,k)
                lo=np.maximum(-1,mid-width); hi=np.minimum(1,mid+width)
                kappa=float(rng.choice([1e-12,1e-8,.0001,.01,.04,.5]))
                r=solve(b,lo,hi,kappa)
                ev=certify(b,r.q,lo,hi,kappa,r.s_dual,r.eta)
                self.assertTrue(ev.exact_feasible)
                self.assertTrue(ev.optimality_tol_pass)
                self.assertLess(float(ev.kl_upper_decimal),kappa+1e-15)
                self.assertLessEqual(float(ev.primal_lower_decimal),float(ev.dual_upper_decimal)+1e-12)

    def test_null_and_zero_budget(self):
        b=np.array([.23,.31,.46]);L=np.array([-.1,-.2,-.3]);U=np.array([.3,.2,.1])
        for kappa in [0.,1e-12,.04,5.]:
            r=solve(b,L,U,kappa)
            ev=certify(b,r.q,L,U,kappa,r.s_dual,r.eta)
            self.assertEqual(float(ev.primal_lower_decimal),0.)
            self.assertTrue(ev.optimality_tol_pass)

    def test_extreme_probability_and_many_actions(self):
        b=np.array([1e-8,.02,.18,.8-1e-8]);L=np.array([-.9,-.3,.2,.4]);U=L+.01
        r=solve(b,L,U,.4)
        ev=certify(b,r.q,L,U,.4,r.s_dual,r.eta)
        self.assertTrue(ev.optimality_tol_pass)
        b,W,s=cycle_instance(48);L=s-.12;U=s+.12
        r=solve(b,L,U,.04)
        ev=certify(b,r.q,L,U,.04,r.s_dual,r.eta)
        self.assertTrue(ev.optimality_tol_pass)

    def test_reject_inconsistent_certificates(self):
        b=[.5,.5];q=[.8,.2];L=[-.5,.5];U=[-.3,.8]
        with self.assertRaises(ValueError):
            certify(b,q,L,U,.1,[.8,.5],1.)
        with self.assertRaises(ValueError):
            certify(b,q,L,U,.1,[-.4,.6],0.,digits=15)

    def test_fail_closed_wrapper(self):
        b=[.4,.6];L=[-.8,.5];U=[-.7,.6]
        verified=solve_verified(b,L,U,.04)
        self.assertTrue(verified.audit.exact_feasible)
        self.assertEqual(len(verified.policy_exact_rationals),2)
        self.assertLessEqual(float(verified.audit.gap_upper_decimal),.0002)
        with self.assertRaises(ArithmeticError):
            solve_verified(b,L,U,.04,tolerance="0")

    def test_radius_explicit_bound(self):
        for k in [2,4,20,100]:
            for delta in [.05,.005]:
                for eps in [.02,.1,.3,.9,1.]:
                    n=n_for_radius(k,delta,eps)
                    self.assertLessEqual(confidence_radius(n,k,delta),eps)
                    self.assertGreater(n,1)

    def test_end_to_end_sample_budget(self):
        p=safe_certificate_budget(6,.05,.04,.03,block=25,explore_every=5)
        self.assertGreater(p.forced_exploration_labels,p.uniform_labels)
        self.assertLessEqual(2*p.radius_target*math.sqrt(2*.04)+p.certified_solver_gap_allowed,.03/2+1e-12)
        with self.assertRaises(ValueError):safe_certificate_budget(6,.05,0.,.03)

if __name__=='__main__':unittest.main()
