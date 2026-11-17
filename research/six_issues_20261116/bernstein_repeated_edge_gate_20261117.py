"""Bernstein certificate for repeated adaptive queries of one Bernoulli edge.

For a fixed history-dependent policy and terminal Boolean event, its event
probability is a Bernstein polynomial of degree at most horizon. Restricting
the polynomial to a parameter interval and recursively subdividing yields
rigorous enclosing bounds on its minimum. No independent novelty is claimed.
"""
from math import comb
import unittest

def coefficients(policy, event, horizon):
    if type(horizon) is not int or not 0 <= horizon <= 16:
        raise ValueError("horizon must be integer in [0,16]")
    sums = [0.] * (horizon + 1)
    def visit(history, successes):
        action = policy(history) if len(history) < horizon else None
        if action is None:
            result = event(history)
            if type(result) is not bool:
                raise ValueError("event must return bool")
            if result:
                remaining = horizon - len(history)
                for j in range(remaining + 1):
                    sums[successes+j] += comb(remaining,j)
            return
        if action != 0:
            raise ValueError("only one Bernoulli edge (0) supported")
        visit(history+((0,0),),successes)
        visit(history+((0,1),),successes+1)
    visit((),0)
    return [sums[k]/comb(horizon,k) for k in range(horizon+1)]

def split(c,t):
    rows=[list(c)]
    while len(rows[-1])>1:
        a=rows[-1]
        rows.append([(1-t)*a[i]+t*a[i+1] for i in range(len(a)-1)])
    return [r[0] for r in rows], [r[-1] for r in rows[::-1]]

def restricted(c,low,high):
    if not 0 <= low <= high <= 1:
        raise ValueError("invalid interval")
    if low==1:
        return [c[-1]]*len(c)
    _,right=split(c,low)
    left,_=split(right,(high-low)/(1-low))
    return left

def bound_minimum(policy,event,horizon,interval=(0.,1.),tol=1e-6,max_nodes=100000):
    if tol<=0 or max_nodes<1:
        raise ValueError("invalid tolerance/max_nodes")
    pending=[restricted(coefficients(policy,event,horizon),*interval)]
    lower=float("inf")
    upper=float("inf")
    visited=0
    while pending:
        c=pending.pop()
        visited+=1
        l=min(c)
        u=min(c[0],c[-1])
        upper=min(upper,u)
        if u-l<=tol or visited>=max_nodes:
            lower=min(lower,l)
        else:
            a,b=split(c,.5)
            pending.extend((a,b))
    return lower,upper,visited

class Tests(unittest.TestCase):
    def test_interior(self):
        p=lambda h:0
        e=lambda h:h[0][1]==h[1][1]
        l,u,_=bound_minimum(p,e,2,tol=1e-7)
        self.assertLessEqual(l,.5)
        self.assertGreaterEqual(u,.5)
        self.assertLessEqual(u-l,1e-7)
    def test_restricted(self):
        p=lambda h:0
        e=lambda h:h[0][1]==h[1][1]
        l,u,_=bound_minimum(p,e,2,(.7,.9),tol=1e-7)
        self.assertLessEqual(l,.58)
        self.assertGreaterEqual(u,.58)
    def test_early_stop(self):
        p=lambda h:0 if not h else None
        e=lambda h:bool(h[0][1])
        l,u,_=bound_minimum(p,e,5,(.2,.8))
        self.assertAlmostEqual(l,.2)
        self.assertAlmostEqual(u,.2)
    def test_four(self):
        p=lambda h:0
        e=lambda h:sum(v for _,v in h)==2
        l,u,_=bound_minimum(p,e,4)
        self.assertLessEqual(l,0)
        self.assertAlmostEqual(u,0)
    def test_invalid(self):
        with self.assertRaises(ValueError):
            bound_minimum(lambda h:0,lambda h:True,2,(.9,.1))
if __name__=="__main__":
    unittest.main()
