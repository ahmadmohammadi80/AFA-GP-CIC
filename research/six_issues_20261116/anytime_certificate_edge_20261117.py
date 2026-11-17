"""Anytime-valid certificate-directed preference edge selection (finite arms).

Assumes conditional i.i.d. Bernoulli responses per edge, adaptively chosen edges,
fixed bounded gain coefficients, and a fixed baseline. No novelty claim.
"""
from math import log, sqrt
import unittest

def confidence_box(successes, counts, alpha=0.05):
    m=len(counts)
    if not 0 < alpha < 1 or m == 0 or len(successes)!=m:
        raise ValueError('invalid inputs')
    box=[]
    for s,n in zip(successes,counts):
        if type(n) is not int or type(s) is not int or n<0 or not 0<=s<=n:
            raise ValueError('invalid counts')
        if n==0:
            box.append((0.,1.))
        else:
            radius=sqrt(log(2*m*n*(n+1)/alpha)/(2*n))
            box.append((max(0.,s/n-radius),min(1.,s/n+radius)))
    return box

def gain_lower_bound(weights, offset, box):
    if len(weights)!=len(box):
        raise ValueError('dimension mismatch')
    return offset+sum(w*(lo if w>=0 else hi) for w,(lo,hi) in zip(weights,box))

def select_certificate_edge(weights, successes, counts, alpha=0.05):
    """Choose edge with greatest current contribution to gain-bound width.

    A deterministic heuristic; not claimed to maximize certificate probability.
    """
    box=confidence_box(successes,counts,alpha)
    widths=[abs(w)*(hi-lo) for w,(lo,hi) in zip(weights,box)]
    return max(range(len(widths)),key=lambda i:(widths[i],-i))

class Tests(unittest.TestCase):
    def test_unobserved_cannot_certify(self):
        self.assertEqual(confidence_box([0],[0]),[(0.,1.)])
        self.assertEqual(gain_lower_bound([1.],-.5,[(0.,1.)]),-.5)
    def test_sign_correct(self):
        self.assertAlmostEqual(gain_lower_bound([2.,-3.],1.,[(.2,.4),(.6,.8)]),-1.)
    def test_certify_known_box(self):
        self.assertGreater(gain_lower_bound([1.,-1.],0.,[(.8,1.),(0.,.1)]),0)
    def test_choice(self):
        self.assertEqual(select_certificate_edge([1.,4.],[0,0],[0,0]),1)
    def test_invalid(self):
        with self.assertRaises(ValueError):confidence_box([2],[1])
if __name__=='__main__':unittest.main()
