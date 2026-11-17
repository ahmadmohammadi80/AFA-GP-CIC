from functools import lru_cache
import unittest, random

def robust_plan(counts, successes, intervals, budget, certified):
    """Rectangular robust adaptive certificate crossing; (worst-case value, first edge)."""
    counts=tuple(counts); successes=tuple(successes); intervals=tuple(tuple(x) for x in intervals)
    assert len(counts)==len(successes)==len(intervals) and len(counts)>0
    assert all(0<=s<=n for s,n in zip(successes,counts))
    assert all(0<=lo<=hi<=1 for lo,hi in intervals) and budget>=0
    @lru_cache(None)
    def solve(n,s,t):
        if certified(n,s): return (1.,None)
        if t==0: return (0.,None)
        vals=[]
        for e,(lo,hi) in enumerate(intervals):
            nn=list(n); nn[e]+=1; ss=list(s); ss[e]+=1
            v1=solve(tuple(nn),tuple(ss),t-1)[0]
            ss[e]-=1
            v0=solve(tuple(nn),tuple(ss),t-1)[0]
            p=lo if v1>=v0 else hi
            vals.append((v0+p*(v1-v0),e))
        return max(vals,key=lambda z:(z[0],-z[1]))
    return solve(counts,successes,budget)

class TestRobust(unittest.TestCase):
    def test_one_step_brute(self):
        rng=random.Random(4)
        for _ in range(200):
            k=3; n=(1,)*k;s=(0,)*k
            intervals=[tuple(sorted((rng.random(),rng.random()))) for _ in range(k)]
            target=rng.randrange(k)
            cert=lambda nn,ss: ss[target]>=1
            value,edge=robust_plan(n,s,intervals,1,cert)
            self.assertAlmostEqual(value,intervals[target][0])
            self.assertEqual(edge,target)
    def test_monotone_and_bounds(self):
        rng=random.Random(5)
        for _ in range(80):
            ints=[tuple(sorted((rng.random(),rng.random()))) for i in range(2)]
            c=lambda n,s: sum(s)>=2
            a=robust_plan((1,1),(0,0),ints,2,c)[0]
            b=robust_plan((1,1),(0,0),ints,3,c)[0]
            self.assertTrue(0<=a<=b<=1)
            mid=[((x+y)/2,(x+y)/2) for x,y in ints]
            self.assertLessEqual(a,robust_plan((1,1),(0,0),mid,2,c)[0]+1e-12)
    def test_zero_and_invalid(self):
        self.assertEqual(robust_plan((1,),(0,),((0.1,0.9),),0,lambda n,s:False),(0.,None))
        with self.assertRaises(AssertionError):robust_plan((1,),(0,),((0.9,0.1),),1,lambda n,s:False)

if __name__=='__main__': unittest.main()
