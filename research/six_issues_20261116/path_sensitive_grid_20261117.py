"""Path-sensitive grid bound for fixed adaptive Bernoulli query policies.
The event must be specified independently of the unknown parameter. A grid
certificate is not by itself a valid statistical certificate. No novelty claim.
"""
from functools import lru_cache
from itertools import product
import unittest

def penalty(policy, radii, horizon):
    if horizon < 0 or not radii or any(not 0 <= r <= 1 for r in radii):
        raise ValueError('invalid setup')
    @lru_cache(None)
    def go(history, left):
        if not left:
            return 0.0
        edge = policy(history)
        if edge is None:
            return 0.0
        if not isinstance(edge, int) or not 0 <= edge < len(radii):
            raise ValueError('invalid query')
        r = radii[edge]
        return r + (1-r)*max(go(history+((edge, y),), left-1) for y in (0,1))
    return go((), horizon)

def event_probability(policy, probs, horizon, event):
    @lru_cache(None)
    def go(history, left):
        if not left:
            return float(bool(event(history)))
        e = policy(history)
        if e is None:
            return float(bool(event(history)))
        return (1-probs[e])*go(history+((e,0),),left-1)+probs[e]*go(history+((e,1),),left-1)
    return go((),horizon)

def select(policies, intervals, horizon, event, steps=4):
    if not policies or not intervals or steps < 1 or horizon < 0:
        raise ValueError('invalid setup')
    if any(not 0 <= lo <= hi <= 1 for lo,hi in intervals):
        raise ValueError('invalid intervals')
    grid=list(product(*[[lo+(hi-lo)*j/steps for j in range(steps+1)] for lo,hi in intervals]))
    radii=tuple((hi-lo)/(2*steps) for lo,hi in intervals)
    scores=[min(event_probability(pi,q,horizon,event) for q in grid) for pi in policies]
    penalties=[penalty(pi,radii,horizon) for pi in policies]
    idx=max(range(len(policies)),key=lambda i:(scores[i]-penalties[i],-i))
    return {'index':idx,'grid_scores':scores,'penalties':penalties,
            'certified_lower':max(0.,scores[idx]-penalties[idx]),
            'uniform_penalty':1-(1-max(radii))**horizon,'grid_size':len(grid)}

class Tests(unittest.TestCase):
    def test_heterogeneous_path(self):
        def pi(h):return 0 if not h else (1 if h[0][1] else None)
        self.assertAlmostEqual(penalty(pi,(.02,.3),2),.02+.98*.3)
        self.assertLess(penalty(pi,(.02,.3),2),1-.7**2)
    def test_exhaustive_events(self):
        def pi(h):return 0 if not h else (1 if h[0][1] else 0)
        p=(.2,.7);q=(.3,.5);bound=penalty(pi,(.1,.2),2)
        for mask in range(16):
            def event(h):return bool(mask & (1<<(2*h[0][1]+h[1][1])))
            self.assertLessEqual(abs(event_probability(pi,p,2,event)-event_probability(pi,q,2,event)),bound+1e-12)
    def test_grid_lower_dense(self):
        policies=[lambda h:0,lambda h:1,lambda h:0 if not h else (1 if h[0][1] else None)]
        event=lambda h:bool(h and any(y for _,y in h))
        out=select(policies,[(.2,.6),(.1,.9)],2,event,steps=4)
        chosen=policies[out['index']]
        for i in range(41):
            for j in range(41):
                p=(.2+.4*i/40,.1+.8*j/40)
                self.assertGreaterEqual(event_probability(chosen,p,2,event)+1e-12,out['certified_lower'])
    def test_never_worse_than_uniform(self):
        policies=[lambda h:0,lambda h:None if h else 1]
        out=select(policies,[(.2,.4),(.1,.9)],3,lambda h:bool(h and h[0][1]),steps=2)
        self.assertTrue(all(x<=out['uniform_penalty']+1e-12 for x in out['penalties']))
    def test_invalid(self):
        with self.assertRaises(ValueError):penalty(lambda h:0,(-.1,),1)
        with self.assertRaises(ValueError):select([],[(0,1)],2,lambda h:True)

if __name__=='__main__':unittest.main()
