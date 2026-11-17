"""Anytime confidence box and exact read-once robust event selection.

Assumes each edge has an independent i.i.d. Bernoulli outcome stream;
edge choices are predictable. Event probability is NOT itself a
statistical improvement certificate.
"""
import math
from itertools import product
import unittest

def anytime_intervals(counts, alpha):
    if not 0 < alpha < 1 or not counts:
        raise ValueError("invalid alpha or counts")
    E=len(counts)
    out=[]
    for wins,n in counts:
        if not isinstance(n,int) or not isinstance(wins,int) or n<0 or wins<0 or wins>n:
            raise ValueError("invalid count")
        if n==0:
            out.append((0.,1.))
            continue
        r=math.sqrt(math.log(2*E*n*(n+1)/alpha)/(2*n))
        out.append((max(0.,wins/n-r),min(1.,wins/n+r)))
    return tuple(out)

def probability(policy,p,horizon,event):
    def rec(h,remaining):
        e=policy(h) if remaining else None
        if e is None:
            return float(bool(event(h)))
        if not isinstance(e,int) or not 0<=e<len(p):
            raise ValueError("invalid edge")
        return (1-p[e])*rec(h+((e,0),),remaining-1)+p[e]*rec(h+((e,1),),remaining-1)
    return rec((),horizon)

def check_read_once(policy,edges,horizon):
    def rec(h,remaining,used):
        e=policy(h) if remaining else None
        if e is None:
            return
        if not isinstance(e,int) or not 0<=e<edges:
            raise ValueError("invalid edge")
        if e in used:
            raise ValueError("repeated edge")
        for y in (0,1):
            rec(h+((e,y),),remaining-1,used|{e})
    rec((),horizon,set())

def robust_event_plan(candidates,counts,alpha,horizon,event):
    if not candidates or not isinstance(horizon,int) or horizon<0:
        raise ValueError("invalid candidates or horizon")
    box=anytime_intervals(counts,alpha)
    values=[]
    for pi in candidates:
        check_read_once(pi,len(counts),horizon)
        values.append(min(probability(pi,p,horizon,event)
                          for p in product(*[(lo,hi) for lo,hi in box])))
    best=max(range(len(values)),key=values.__getitem__)
    return best,values[best],box

class Tests(unittest.TestCase):
    def test_zero_counts(self):
        self.assertEqual(anytime_intervals([(0,0)],.05),((0.,1.),))
    def test_center(self):
        lo,hi=anytime_intervals([(500,1000)],.05)[0]
        self.assertLess(lo,.5)
        self.assertGreater(hi,.5)
    def test_selection(self):
        p0=lambda h: 0 if not h else None
        p1=lambda h: 1 if not h else None
        ev=lambda h: bool(h and h[-1][1])
        idx,v,box=robust_event_plan([p0,p1],[(10,100),(90,100)],.05,1,ev)
        self.assertEqual(idx,1)
        self.assertAlmostEqual(v,box[1][0])
    def test_repeated_edge(self):
        with self.assertRaisesRegex(ValueError,"repeated edge"):
            robust_event_plan([lambda h:0],[(5,10)],.05,2,lambda h:True)
    def test_dense_grid(self):
        pi=lambda h: 0 if not h else (1 if h[0][1] else None)
        ev=lambda h: bool(h and h[-1][1])
        box=((.2,.6),(.1,.9))
        v=min(probability(pi,p,2,ev) for p in product(*box))
        dense=min(probability(pi,(.2+.01*i,.1+.02*j),2,ev)
                  for i in range(41) for j in range(41))
        self.assertAlmostEqual(v,dense)
    def test_bad_inputs(self):
        with self.assertRaises(ValueError): anytime_intervals([(3,2)],.05)
        with self.assertRaises(ValueError): anytime_intervals([(0,0)],1)

if __name__=="__main__":
    unittest.main()
