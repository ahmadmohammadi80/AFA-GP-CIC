"""Maximin one-step probability of a valid positive linear-gain certificate.

Fixed policy and coefficients; adaptive Bernoulli edge sampling; rectangular
anytime confidence box. This is a scoped candidate, NOT an originality claim.
"""
from math import log, sqrt
import unittest

def confidence_box(successes, counts, alpha=0.05):
    m = len(counts)
    if not 0 < alpha < 1 or not m or len(successes) != m:
        raise ValueError("invalid inputs")
    out = []
    for s,n in zip(successes,counts):
        if type(s) is not int or type(n) is not int or not 0 <= s <= n:
            raise ValueError("invalid counts")
        if n == 0:
            out.append((0.,1.))
        else:
            r = sqrt(log(2*m*n*(n+1)/alpha)/(2*n))
            out.append((max(0.,s/n-r),min(1.,s/n+r)))
    return out

def gain_lower(weights, offset, box):
    if len(weights) != len(box):
        raise ValueError("dimension mismatch")
    return offset + sum(w*(lo if w>=0 else hi) for w,(lo,hi) in zip(weights,box))

def robust_one_step(weights, offset, successes, counts, alpha=0.05):
    """Return (best_edge, robust_probability, all_edge_scores).

    For edge e, I_y is 1 iff the recomputed next-step gain lower bound
    exceeds zero after observing y. The exact worst-case one-step
    certification probability is min at the two current-box endpoints.
    """
    box = confidence_box(successes,counts,alpha)
    if len(weights) != len(counts):
        raise ValueError("dimension mismatch")
    scores = []
    for e,(lo,hi) in enumerate(box):
        indicators = []
        for y in (0,1):
            ss,nn = list(successes),list(counts)
            ss[e] += y
            nn[e] += 1
            indicators.append(int(gain_lower(weights,offset,confidence_box(ss,nn,alpha))>0))
        i0,i1 = indicators
        score = min((1-lo)*i0+lo*i1,(1-hi)*i0+hi*i1)
        scores.append((score,(i0,i1)))
    best = max(range(len(scores)),key=lambda e:(scores[e][0],-e))
    return best,scores[best][0],scores

class Tests(unittest.TestCase):
    def test_unobserved(self):
        self.assertEqual(robust_one_step([1.],-.5,[0],[0])[:2],(0,0.))
    def test_already_positive(self):
        self.assertEqual(robust_one_step([0.,0.],.1,[0,0],[0,0])[:2],(0,1.))
    def test_endpoint_bound(self):
        for lo,hi in ((.1,.8),(.3,.3),(0.,1.)):
            for i0,i1 in ((0,0),(0,1),(1,0),(1,1)):
                exact=min((1-lo)*i0+lo*i1,(1-hi)*i0+hi*i1)
                grid=min((1-(lo+(hi-lo)*k/100))*i0+(lo+(hi-lo)*k/100)*i1 for k in range(101))
                self.assertAlmostEqual(exact,grid)
    def test_argmax(self):
        _,v,s=robust_one_step([1.,-1.],.2,[4,1],[8,8])
        self.assertEqual(v,max(x[0] for x in s))
    def test_invalid(self):
        with self.assertRaises(ValueError):
            robust_one_step([1.],0.,[2],[1])

if __name__=="__main__":
    unittest.main()
