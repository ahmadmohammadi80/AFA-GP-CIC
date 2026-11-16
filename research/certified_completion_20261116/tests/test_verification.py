import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import unittest
import numpy as np
from strong_adaptive_study import cycle_instance,oracle_kl_policy,strongest_pair_oracle,pair_transfer,kl,robust_gain,bound,robust_full_policy,run
from real_llm_pilot import true_preferences, simulate,QA

class Theorems(unittest.TestCase):
    def test_cyclic_W_b(self):
        for k in [6,12,24,48]:
            b,W,s=cycle_instance(k)
            self.assertTrue(np.allclose(W@b,s))
            self.assertGreater(W[0,1],0)
            self.assertGreater(W[1,2],0)
            self.assertGreater(W[2,0],0)

    def test_pair_lower_bound(self):
        for k in [6,12,24,48]:
            b,W,s=cycle_instance(k)
            pair,_=strongest_pair_oracle(b,s,.04)
            self.assertLessEqual(pair,.6*min(1/k,np.sqrt(.04/k))+1e-10)
            full=oracle_kl_policy(s,b,.04)
            self.assertGreaterEqual(float(full@s),.3*np.sqrt(.04)-1e-9)
            self.assertGreater(float(full@s),pair)

    def test_exact_policy_KL(self):
        b,W,s=cycle_instance(24)
        q=oracle_kl_policy(s,b,.04)
        self.assertAlmostEqual(sum(q),1)
        self.assertLessEqual(kl(q,b),.040000000001)

    def test_robust_certificate_is_lower_bound(self):
        b,W,s=cycle_instance(12)
        lo=s-.2;hi=s+.2
        q=robust_full_policy(b,lo,hi,.04)
        self.assertLessEqual(robust_gain(q,b,lo,hi),float((q-b)@s)+1e-8)
        self.assertLessEqual(kl(q,b),.04+1e-8)

    def test_bounded_preference_oracle(self):
        for _,_,correct in QA:
            W=true_preferences(correct)
            self.assertTrue(np.allclose(W,-W.T))
            self.assertLessEqual(abs(W).max(),1.)

    def test_protocol_dry_run_12_questions(self):
        rng=np.random.default_rng(0)
        x=[{'question':p,'choices':choices,'correct_index':index,'reference_policy':rng.dirichlet(np.ones(6)*2).tolist()} for p,choices,index in QA]
        out=simulate(x,n=500)
        self.assertEqual(len(out),12)
        self.assertTrue(all(v['policy_kl']<=.150001 for v in out))

    def test_all_labels_consistent(self):
        row=run(k=12,n=3000,seeds=3)
        self.assertLess(row['oracle_one_transfer_gain'],.04)
        self.assertGreater(row['oracle_unrestricted_gain'],.04)
        self.assertTrue(all(x['kl']<=.040001 for x in row['trials']))

if __name__=='__main__':unittest.main()
