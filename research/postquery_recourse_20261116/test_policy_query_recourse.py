import unittest
import numpy as np
from policy_query_recourse import one_step_scan,replay,experiment

class JointPolicyQuery(unittest.TestCase):
    def test_exact_one_step_inequality_on_symmetric_preference(self):
        b=np.array([.5,.5]);Q=np.array([[.7,.3],[.3,.7]])
        H=np.eye(2)
        for n in (20,30,100):
            s=one_step_scan(b,Q,H,[n//2],[n],.05,20)
            self.assertGreater(s['global_joint'],s['global_precommit']+1e-4)
            self.assertEqual(s['best_joint_edge'],0)

    def test_multiedge_recourse_dominates_precommit_one_step(self):
        b=np.ones(4)/4;Q=np.array([[.45,.05,.25,.25],[.1,.4,.25,.25],[.25,.25,.4,.1]])
        H=np.eye(4)
        scans=one_step_scan(b,Q,H,[10,20,14,18,27,12],[30,30,30,40,40,20],.05,10)
        self.assertTrue(np.all(scans['recourse']>=scans['precommit']-1e-12))
        self.assertGreaterEqual(scans['global_joint'],scans['global_precommit']-1e-12)

    def test_no_fake_global_sample_advantage(self):
        r=experiment(seeds=8,budget=1510,block=50)
        for method in ["joint","precommit","greedy_online"]:
            self.assertEqual(r['methods'][method]['false_certificates'],0)
        self.assertEqual(len(r['rows']),8)
        self.assertLessEqual(r['methods']['joint']['certificates'],8)

    def test_replay_determinism_and_kl(self):
        for method in ("joint","precommit"):
            a=replay(543,method,budget=310,block=50)
            b=replay(543,method,budget=310,block=50)
            self.assertEqual(a,b)
            self.assertLessEqual(max(a['KL']),.05)
            self.assertLessEqual(a['labels'],310)

    def test_invalid_inputs(self):
        b=np.array([.5,.5]);q=np.array([[.7,.3],[.3,.7]])
        with self.assertRaises(ValueError):
            one_step_scan(b,q,np.eye(2),[0],[1],block=0)
        with self.assertRaises(ValueError):
            one_step_scan(b,q,np.eye(2),[4],[1])
        with self.assertRaises(ValueError):
            replay(1,mode="fiction")

if __name__=="__main__":unittest.main()
