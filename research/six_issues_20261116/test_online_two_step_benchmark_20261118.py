"""Reproducible regression tests for the full online comparison."""
import unittest
import numpy as np
from online_two_step_benchmark_20261118 import (
    METHODS,run_one,paired_audit,joint_mix_edge
)
from integrated_two_step_certificate_20261118 import (
    canonical_problem,greedy_edge,intervals,certificate
)


def without_timing(row):
    return {k:v for k,v in row.items() if k not in ('elapsed_sec','method')}

class OnlineTests(unittest.TestCase):
    def test_matched_prefix_and_reproducibility(self):
        for seed in (20261116,20270101):
            for mode in METHODS:
                a=run_one(seed,mode,budget=300)
                b=run_one(seed,mode,budget=300)
                self.assertEqual(without_timing(a),without_timing(b))
                self.assertEqual(a['steps'],a['labels']-10)
                self.assertFalse(a['false_certificate'])

    def test_zero_gate_matches_greedy_exactly(self):
        for seed in (20261116,20270101,20261121):
            a=run_one(seed,'greedy',budget=500)
            b=run_one(seed,'afa_gated_two_step',budget=500,gate=0.)
            self.assertEqual(a['labels'],b['labels'])
            self.assertEqual(a['certified'],b['certified'])
            self.assertEqual(a['max_lower'],b['max_lower'])
            self.assertEqual(a['query_counts'],b['query_counts'])
            self.assertEqual(b['planner_calls'],0)

    def test_reference_greedy_scoring(self):
        s,n,_,_,As,*_=canonical_problem(101)
        self.assertEqual(greedy_edge(s,n,As),joint_mix_edge(s,n,As))
        ci=intervals(s,n,.05)
        self.assertEqual(len(ci),10)
        self.assertTrue(all(np.isfinite(certificate(A,ci[:,0],ci[:,1])) for A in As))

    def test_paired_output_and_null(self):
        r=paired_audit(seeds=3,offset=20261116,budget=250,null=True)
        self.assertEqual(set(r['rows']),set(METHODS))
        self.assertEqual(r['protocol']['null'],True)
        for m in METHODS:
            self.assertEqual(r['summary'][m]['false_certificates'],0)
            self.assertEqual(len(r['rows'][m]),3)
        self.assertIn('holm_vs_greedy_p',r['paired']['joint_mix_vs_greedy'])

    def test_nontransitive_problem(self):
        _,_,_,_,_,W,*_=canonical_problem(33,cyclic=True)
        self.assertGreater(W[2,3],0)
        self.assertGreater(W[3,4],0)
        self.assertGreater(W[4,2],0)
        r=run_one(33,'afa_gated_two_step',budget=350,cyclic=True)
        self.assertFalse(r['false_certificate'])

if __name__=='__main__':unittest.main()
