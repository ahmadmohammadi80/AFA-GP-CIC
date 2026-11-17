"""Fixed-parameter robust adaptive query planner with Pareto-frontier DP.

A single unknown Bernoulli vector p is held fixed across all queries.
A finite parameter grid G is used; the same parameter index is used in
both children. This is NOT rectangular robust Bellman optimization.
The terminal event is an externally specified certificate predicate.
No claim of literature novelty or certificate validity is made.
"""
from functools import lru_cache
from itertools import product
import unittest


def plan(intervals, horizon, certified, steps=4):
    if not intervals or not isinstance(horizon,int) or horizon<0 or not isinstance(steps,int) or steps<1:
        raise ValueError('invalid horizon, grid or intervals')
    if any(not (0<=lo<=hi<=1) for lo,hi in intervals):
        raise ValueError('invalid intervals')
    k=len(intervals)
    grid=list(product(*[[lo+(hi-lo)*j/steps for j in range(steps+1)] for lo,hi in intervals]))
    radius=max((hi-lo)/(2*steps) for lo,hi in intervals)
    explored=[0]

    @lru_cache(None)
    def frontier(n,s,left):
        if certified(n,s) or left==0:
            v=1.0 if certified(n,s) else 0.0
            return ((tuple(v for _ in grid),None),)
        candidates=[]
        for edge in range(k):
            nn=list(n);nn[edge]+=1;nn=tuple(nn)
            ss=list(s);ss[edge]+=1;ss=tuple(ss)
            zero=frontier(nn,s,left-1)
            one=frontier(nn,ss,left-1)
            for vz,az in zero:
                for vo,ao in one:
                    vec=tuple((1-p[edge])*vz[i]+p[edge]*vo[i] for i,p in enumerate(grid))
                    candidates.append((vec,(edge,az,ao)))
        # Coordinatewise dominance is preserved by convex child mixing.
        keep=[]
        for v,tree in candidates:
            if any(all(u[i]>=v[i]-1e-12 for i in range(len(grid))) for u,_ in keep):
                continue
            keep=[(u,t) for u,t in keep if not all(v[i]>=u[i]-1e-12 for i in range(len(grid)))]
            keep.append((v,tree))
        explored[0]+=len(candidates)
        return tuple(keep)
    root=frontier((0,)*k,(0,)*k,horizon)
    values,tree=max(root,key=lambda z:min(z[0]))
    score=min(values)
    eps=1-(1-radius)**horizon
    return {'grid_min':score,'guaranteed_min':max(0.,score-eps),
            'grid_optimality_gap':0.,'true_regret_bound':min(1.,2*eps),
            'radius':radius,'tree':tree,'frontier_size':len(root),
            'expanded':explored[0]}


class TestFixedParameterPareto(unittest.TestCase):
    def test_nonrectangular_gap(self):
        # Exactly one success in two draws: min_{p in [.2,.8]} 2p(1-p)=.32.
        # Rectangular nature can vary p by history and reaches .20.
        cert=lambda n,s:n[0]==2 and s[0]==1
        out=plan([(.2,.8)],2,cert,steps=6)
        self.assertAlmostEqual(out['grid_min'],.32)
        self.assertAlmostEqual(out['radius'],.05)
        self.assertGreater(out['grid_min'],.2)
    def test_adaptive_two_edge(self):
        cert=lambda n,s:sum(s)>=2
        out=plan([(.2,.7),(.1,.9)],2,cert,steps=2)
        self.assertTrue(0<=out['guaranteed_min']<=out['grid_min']<=1)
        self.assertTrue(out['frontier_size']>=1)
    def test_zero_horizon(self):
        out=plan([(.1,.9)],0,lambda n,s:False)
        self.assertEqual(out['grid_min'],0.)
        self.assertEqual(out['true_regret_bound'],0.)
    def test_invalid(self):
        for intervals in ([],[(.8,.2)]):
            with self.assertRaises(ValueError):plan(intervals,1,lambda n,s:False)


if __name__=='__main__':unittest.main()
