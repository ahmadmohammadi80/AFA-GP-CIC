"""Path-dependent coupling certificate for adaptive Bernoulli edge queries.
This is a structural refinement of standard coupling, not a novelty claim.
"""
from functools import lru_cache
import math
import unittest


def adaptive_penalty(policy, radii, budget):
    """Worst-path mismatch bound for a deterministic policy(state)->edge.

    state is tuple of (edge, observation) pairs. Policy returns None to stop.
    Radii are upper bounds on |p_e-q_e|, for fixed edge parameters.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError('invalid budget')
    if not radii or any(not isinstance(v, (int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in radii.values()):
        raise ValueError('invalid radii')

    @lru_cache(None)
    def rec(state, left):
        if left == 0:
            return 0.0
        edge = policy(state)
        if edge is None:
            return 0.0
        if edge not in radii:
            raise ValueError('unknown edge')
        r = radii[edge]
        continuation = max(rec(state + ((edge, y),), left-1) for y in (0,1))
        return r + (1-r)*continuation
    return rec((), budget)


def exact_event_probability(policy, probs, budget, event):
    def rec(state, left, mass):
        if left == 0:
            return mass * bool(event(state))
        edge = policy(state)
        if edge is None:
            return mass * bool(event(state))
        p = probs[edge]
        return rec(state+((edge,0),),left-1,mass*(1-p))+rec(state+((edge,1),),left-1,mass*p)
    return rec((),budget,1.)


class TestAdaptiveCoupling(unittest.TestCase):
    def test_repeated_edge_matches_sharp_bound(self):
        for b in range(7):
            v=adaptive_penalty(lambda s:'a', {'a':.2},b)
            self.assertAlmostEqual(v,1-.8**b)

    def test_adaptive_path_improves_global_radius(self):
        def policy(s):
            return 'a' if not s else ('b' if s[0][1] else None)
        v=adaptive_penalty(policy,{'a':.1,'b':.4},2)
        self.assertAlmostEqual(v,.1+.9*.4)
        self.assertLess(v,1-.6**2)

    def test_exhaustive_event_bound(self):
        def policy(s):
            return 'a' if not s else ('b' if s[0][1] else 'a')
        p={'a':.2,'b':.7};q={'a':.3,'b':.5}
        bound=adaptive_penalty(policy,{'a':.1,'b':.2},2)
        for mask in range(16):
            def event(s):
                bits=tuple(y for _,y in s)
                return bool(mask & (1<<(bits[0]*2+bits[1])))
            diff=abs(exact_event_probability(policy,p,2,event)-exact_event_probability(policy,q,2,event))
            self.assertLessEqual(diff,bound+1e-12)

    def test_invalid_inputs(self):
        for v in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):adaptive_penalty(lambda s:'a',{'a':v},1)
        with self.assertRaises(ValueError):adaptive_penalty(lambda s:'a',{'a':.1},-1)
        with self.assertRaises(ValueError):adaptive_penalty(lambda s:'missing',{'a':.1},1)

if __name__=='__main__': unittest.main()
