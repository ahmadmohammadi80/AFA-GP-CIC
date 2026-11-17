"""Exact robust probability for a single Bernoulli edge queried at most twice.

This extends the read-once vertex method to one repeated edge with horizon 2.
It does NOT establish a new scientific novelty claim or certify policy improvement.
"""
import unittest

def event_probability_two(policy, p, event):
    def rec(h, remaining):
        e = policy(h) if remaining else None
        if e is None:
            return float(bool(event(h)))
        if e != 0:
            raise ValueError("only one edge supported")
        return (1-p)*rec(h+((0,0),), remaining-1)+p*rec(h+((0,1),), remaining-1)
    return rec((), 2)

def exact_minimum_two(policy, interval, event):
    lo, hi = interval
    if not 0 <= lo <= hi <= 1:
        raise ValueError("invalid interval")
    f0 = event_probability_two(policy, 0., event)
    f1 = event_probability_two(policy, 1., event)
    fm = event_probability_two(policy, .5, event)
    a = 2*(f0+f1-2*fm)
    b = f1-f0-a
    candidates = [lo, hi]
    if a > 0:
        stationary = -b/(2*a)
        if lo < stationary < hi:
            candidates.append(stationary)
    pstar = min(candidates, key=lambda p:event_probability_two(policy,p,event))
    return event_probability_two(policy,pstar,event), pstar

class Tests(unittest.TestCase):
    def test_interior_minimum(self):
        pi = lambda h: 0
        ev = lambda h: h[0][1] == h[1][1]
        val, p = exact_minimum_two(pi,(0.,1.),ev)
        self.assertAlmostEqual(val,.5)
        self.assertAlmostEqual(p,.5)
        self.assertEqual(event_probability_two(pi,0.,ev),1.)
        self.assertEqual(event_probability_two(pi,1.,ev),1.)
    def test_boundary(self):
        pi = lambda h: 0
        ev = lambda h: h[0][1] == h[1][1]
        val,p = exact_minimum_two(pi,(.7,.9),ev)
        self.assertAlmostEqual(p,.7)
        self.assertAlmostEqual(val,.58)
    def test_linear(self):
        pi = lambda h: 0 if not h else None
        ev = lambda h: h[0][1] == 1
        val,p = exact_minimum_two(pi,(.2,.8),ev)
        self.assertAlmostEqual(val,.2)
        self.assertAlmostEqual(p,.2)
    def test_invalid(self):
        with self.assertRaises(ValueError):
            exact_minimum_two(lambda h:0,(.9,.1),lambda h:True)
    def test_grid(self):
        pi = lambda h:0
        ev = lambda h:h[0][1] == h[1][1]
        val,_ = exact_minimum_two(pi,(.1,.9),ev)
        grid = min(event_probability_two(pi,.1+.008*i,ev) for i in range(101))
        self.assertAlmostEqual(val,grid,places=12)

if __name__ == "__main__":
    unittest.main()
