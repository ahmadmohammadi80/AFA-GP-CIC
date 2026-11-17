"""Tensor-Bernstein robust event gate for adaptive multi-edge Bernoulli queries.

A fixed history-dependent policy can choose different edges based on outcomes.
For independent stationary edge parameters in a rectangular uncertainty set,
the event probability has a tensor Bernstein representation. Coefficient
minima provide valid lower bounds, corner evaluations upper bounds.
This does not establish independent novelty or policy improvement.
"""
from itertools import product
from math import comb, prod
import unittest

def gate(policy, event, edges, horizon, intervals):
    if type(horizon) is not int or not 0 <= horizon <= 8:
        raise ValueError("invalid horizon")
    if type(edges) is not int or not 1 <= edges <= 4 or len(intervals) != edges:
        raise ValueError("invalid edges")
    if any(not 0 <= lo <= hi <= 1 for lo, hi in intervals):
        raise ValueError("invalid intervals")
    keys = list(product(range(horizon + 1), repeat=edges))
    sums = dict.fromkeys(keys, 0.0)
    def visit(history, counts, successes):
        action = policy(history) if len(history) < horizon else None
        if action is None:
            result = event(history)
            if type(result) is not bool:
                raise ValueError("event must be Boolean")
            if result:
                for padding in product(*(range(horizon-counts[e]+1) for e in range(edges))):
                    key = tuple(successes[e]+padding[e] for e in range(edges))
                    sums[key] += prod(comb(horizon-counts[e],padding[e]) for e in range(edges))
            return
        if type(action) is not int or not 0 <= action < edges:
            raise ValueError("invalid edge")
        for outcome in (0,1):
            c=list(counts); s=list(successes)
            c[action]+=1; s[action]+=outcome
            visit(history+((action,outcome),),tuple(c),tuple(s))
    visit((),(0,)*edges,(0,)*edges)
    coeff={k:v/prod(comb(horizon,i) for i in k) for k,v in sums.items()}
    def split(c,t):
        rows=[list(c)]
        while len(rows[-1])>1:
            a=rows[-1]
            rows.append([(1-t)*a[i]+t*a[i+1] for i in range(len(a)-1)])
        return [r[0] for r in rows],[r[-1] for r in rows[::-1]]
    for axis,(lo,hi) in enumerate(intervals):
        for others in product(range(horizon+1),repeat=edges-1):
            def key(i):
                a=list(others);a.insert(axis,i);return tuple(a)
            values=[coeff[key(i)] for i in range(horizon+1)]
            if lo==1:
                values=[values[-1]]*(horizon+1)
            else:
                _,right=split(values,lo)
                values,_=split(right,(hi-lo)/(1-lo))
            for i,value in enumerate(values):
                coeff[key(i)]=value
    return min(coeff.values()), min(coeff[k] for k in product((0,horizon),repeat=edges))

class Tests(unittest.TestCase):
    def test_adaptive_switch(self):
        policy=lambda h: 0 if not h else (1 if h[0][1] else 0)
        event=lambda h: h[-1][1]==1
        lower,upper=gate(policy,event,2,2,((.2,.8),(.3,.7)))
        # Exact event probability is (1-p0)*p0+p0*p1; minimum .22.
        self.assertLessEqual(lower,.22+1e-12)
        self.assertGreaterEqual(upper,.22-1e-12)
    def test_true(self):
        self.assertEqual(gate(lambda h:0,lambda h:True,2,3,((0,1),(0,1))),(1.,1.))
    def test_false(self):
        self.assertEqual(gate(lambda h:0,lambda h:False,2,3,((0,1),(0,1))),(0.,0.))
    def test_early_stop(self):
        policy=lambda h:None if h else 1
        event=lambda h:bool(h[0][1])
        lower,upper=gate(policy,event,2,3,((.1,.9),(.25,.75)))
        self.assertAlmostEqual(lower,.25)
        self.assertAlmostEqual(upper,.25)
    def test_invalid(self):
        with self.assertRaises(ValueError):
            gate(lambda h:2,lambda h:True,2,1,((0,1),(0,1)))
if __name__=="__main__":
    unittest.main()
