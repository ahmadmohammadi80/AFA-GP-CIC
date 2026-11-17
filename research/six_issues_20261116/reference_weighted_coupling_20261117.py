"""Reference-weighted adaptive coupling. Robustness only, not statistical coverage."""
from functools import lru_cache
from itertools import product
import unittest

def penalty(policy, q, radii, horizon):
    if horizon < 0 or not q or len(q)!=len(radii) or any(not 0<=v<=1 for v in (*q,*radii)):
        raise ValueError("invalid parameters")
    @lru_cache(None)
    def go(h,t):
        if not t: return 0.
        e=policy(h)
        if e is None: return 0.
        if not isinstance(e,int) or not 0<=e<len(q): raise ValueError("invalid edge")
        a=go(h+((e,0),),t-1); b=go(h+((e,1),),t-1)
        return min(1.,radii[e]+(1-q[e])*a+q[e]*b)
    return go((),horizon)

def probability(policy,p,horizon,event):
    @lru_cache(None)
    def go(h,t):
        e=policy(h) if t else None
        if e is None: return float(bool(event(h)))
        return (1-p[e])*go(h+((e,0),),t-1)+p[e]*go(h+((e,1),),t-1)
    return go((),horizon)

def certified_lower(policy,intervals,horizon,event,steps=4):
    if steps<1 or horizon<0 or not intervals or any(not 0<=a<=b<=1 for a,b in intervals):
        raise ValueError("invalid grid")
    r=tuple((b-a)/(2*steps) for a,b in intervals)
    axes=[tuple(a+(b-a)*j/steps for j in range(steps+1)) for a,b in intervals]
    return max(0.,min(probability(policy,q,horizon,event)-penalty(policy,q,r,horizon)
                      for q in product(*axes)))

class Tests(unittest.TestCase):
    def test_strict_improvement(self):
        def pi(h): return 0 if not h else (1 if h[0][1] else None)
        self.assertAlmostEqual(penalty(pi,(.1,.7),(.02,.3),2),.05)
        self.assertLess(penalty(pi,(.1,.7),(.02,.3),2),.02+.98*.3)
    def test_exhaustive_events(self):
        def pi(h): return 0 if not h else (1 if h[0][1] else 0)
        q=(.2,.7); d=penalty(pi,q,(.1,.2),2)
        for i in range(11):
            for j in range(11):
                p=(.1+.02*i,.5+.04*j)
                for mask in range(16):
                    event=lambda h,mask=mask:bool(mask & (1<<(2*h[0][1]+h[1][1])))
                    self.assertLessEqual(abs(probability(pi,p,2,event)-probability(pi,q,2,event)),d+1e-12)
    def test_grid_dense(self):
        def pi(h):return 0 if not h else (1 if h[0][1] else None)
        event=lambda h:bool(h and any(y for _,y in h))
        lower=certified_lower(pi,((.2,.6),(.1,.9)),2,event)
        for i in range(41):
            for j in range(41):
                self.assertGreaterEqual(probability(pi,(.2+.01*i,.1+.02*j),2,event)+1e-12,lower)
    def test_invalid(self):
        with self.assertRaises(ValueError):penalty(lambda h:0,(.2,),(-.1,),1)
        with self.assertRaises(ValueError):certified_lower(lambda h:0,((.5,.2),),1,lambda h:True)

if __name__=="__main__":unittest.main()
