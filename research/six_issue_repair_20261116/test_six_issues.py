"""Core mathematical and implementation checks for six-issue audit."""
import unittest,sys,itertools
import numpy as np
from scipy.stats import betabinom
from repair_core import problem,simulate,threshold_recourse_edge
from baselines import benchmark,holm,pac_bayes_dirac,rucb_query
from drift_cert import confidence,experiment as drift_experiment
from scaling import make_inputs,scan
from anytime_mixture import edge_cs,edge_coefficients,certificate

class SixIssueTests(unittest.TestCase):
    def test_actual_cycle_and_borda_invisibility(self):
        b,Q,H,W,edges,A,mu=problem(cyclic=False)
        _,_,_,C,_,_,_=problem(cyclic=True)
        gap=C-W
        self.assertLess(np.max(np.abs(gap@b)),1e-12)
        self.assertTrue(any(C[i,j]>0 and C[j,k]>0 and C[k,i]>0
                       for i,j,k in itertools.permutations(range(5),3)))
        self.assertFalse(any(W[i,j]>0 and W[j,k]>0 and W[k,i]>0
                       for i,j,k in itertools.permutations(range(5),3)))
        self.assertGreater(abs(np.min((Q[1]-b)@C@H.T)-np.min((Q[1]-b)@W@H.T)),.05)

    def test_kl_feasible_and_triggered(self):
        b,Q,H,W,E,A,mu=problem()
        self.assertTrue(all(sum(q*np.log(q/b))<=.05 for q in Q))
        a=simulate(20261116,'joint_threshold')
        self.assertGreaterEqual(a['labels'],len(E))
        self.assertLessEqual(a['labels'],1510)
        self.assertFalse(a['cert'] and a['true_gain']<=.015)

    def test_replay_seed(self):
        x=simulate(20261122,'joint_threshold')
        y=simulate(20261122,'joint_threshold')
        self.assertEqual(x,y)

    def test_40_original_benchmark(self):
        report=benchmark(seeds=40,cyclic=False)
        self.assertEqual(report['methods']['joint_threshold']['certified'],34)
        self.assertEqual(report['methods']['greedy']['certified'],34)
        self.assertGreater(sum(r['joint_threshold']['changes'] for r in report['rows']),0)
        self.assertEqual(report['comparisons_vs_greedy']['joint_threshold']['p_holm'],1.)

    def test_40_true_cycle_benchmark(self):
        report=benchmark(seeds=40,cyclic=True)
        self.assertEqual(report['methods']['joint_threshold']['certified'],37)
        self.assertEqual(report['methods']['greedy']['certified'],37)

    def test_pac_dirac_fixed_budget(self):
        x=pac_bayes_dirac(11,budget=300)
        self.assertEqual(x['labels'],300)
        self.assertGreater(x['max_policy_width'],0)

    def test_holm_monotone_adjustment(self):
        a=holm([.03,.001,.1])
        for q,p in zip(a,[.03,.001,.1]): self.assertGreaterEqual(q,p)
        self.assertAlmostEqual(a[1],.003)

    def test_drift_coverage_and_certificate(self):
        result=drift_experiment(seeds=40,epochs=3,epoch_size=10000)
        self.assertEqual(result['all_epochs_covered_runs'],40)
        self.assertEqual(result['false_certificates'],0)
        self.assertGreater(result['total_certificates'],0)
        self.assertEqual(result['total_certificates'],116)

    def test_lookahead_vectorization(self):
        for k in [5,8,12]:
            A,N,wins=make_inputs(k,P=3)
            a=scan(A,N,wins,block=5,optimized=False)
            b=scan(A,N,wins,block=5,optimized=True)
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)

    def test_one_step_switching_probability_strict(self):
        b=np.array([.5,.5]);Q=[np.array([.7,.3]),np.array([.3,.7])];H=np.eye(2)
        AA=[edge_coefficients(b,q,H)[1] for q in Q]
        B=100;xs=np.arange(B+1);prob=betabinom.pmf(xs,B,15.5,15.5)
        ok=[]
        for y in xs:
            l,h=edge_cs(int(15+y),30+B,.05)
            ok.append([certificate(A,np.array([l]),np.array([h]))>.001 for A in AA])
        ok=np.asarray(ok)
        joint=float((prob*ok.any(axis=1)).sum());fixed=float((prob[:,None]*ok).sum(axis=0).max())
        self.assertGreater(joint-fixed,.02)

    def test_regret_width_pathwise(self):
        b,Q,H,W,E,A,mu=problem(cyclic=True)
        lo=np.maximum(-1,mu-.02);hi=np.minimum(1,mu+.02)
        lowers=np.where(A>=0,A*lo,A*hi).sum(axis=-1).min(axis=1)
        best=int(np.argmax((A@mu).min(axis=1)));chosen=int(lowers.argmax())
        gap=float((A[best]@mu).min()-(A[chosen]@mu).min())
        width=float(np.max(np.sum(np.abs(A)*(hi-lo),axis=-1)))
        self.assertLessEqual(gap,width+1e-12)

if __name__=='__main__':unittest.main()