"""Exact robust certificate-event planner for read-once Bernoulli preference edges.

The guarantee concerns event probability, NOT validity of the event as a
statistical confidence certificate. All policies are finite and predeclared.
"""
from itertools import product
from functools import lru_cache
import unittest


def event_probability(policy, p, horizon, event):
    @lru_cache(None)
    def visit(history, remaining):
        edge = policy(history) if remaining else None
        if edge is None:
            return float(bool(event(history)))
        if not isinstance(edge, int) or not 0 <= edge < len(p):
            raise ValueError('invalid edge')
        return ((1-p[edge])*visit(history+((edge, 0),), remaining-1)
                +p[edge]*visit(history+((edge, 1),), remaining-1))
    return visit((), horizon)


def check_read_once(policy, num_edges, horizon):
    def visit(history, remaining, seen):
        edge = policy(history) if remaining else None
        if edge is None:
            return
        if not isinstance(edge, int) or not 0 <= edge < num_edges:
            raise ValueError('invalid edge')
        if edge in seen:
            raise ValueError('edge reused: vertex theorem inapplicable')
        for y in (0, 1):
            visit(history+((edge,y),), remaining-1, seen | {edge})
    visit((), horizon, set())


def exact_robust_value(policy, intervals, horizon, event):
    if horizon < 0 or not intervals or any(not 0 <= a <= b <= 1 for a,b in intervals):
        raise ValueError('invalid intervals or horizon')
    check_read_once(policy, len(intervals), horizon)
    vertices = product(*[(a,b) for a,b in intervals])
    return min(event_probability(policy, p, horizon, event) for p in vertices)


def select_policy(candidates, intervals, horizon, event):
    """Maximin over a finite predeclared set; ties keep earliest candidate."""
    if not candidates:
        raise ValueError('empty candidates')
    scores = [exact_robust_value(pi, intervals, horizon, event) for pi in candidates]
    index = max(range(len(scores)), key=lambda i: scores[i])
    return index, scores[index], scores


class Tests(unittest.TestCase):
    def test_adaptive_exact_vs_dense_grid(self):
        def pi(h):
            return 0 if not h else (1 if h[0][1] else None)
        event = lambda h: bool(h and h[-1][1])
        intervals = ((.2,.6),(.1,.9))
        exact = exact_robust_value(pi, intervals, 2, event)
        dense = min(event_probability(pi,(.2+.01*i,.1+.02*j),2,event)
                    for i in range(41) for j in range(41))
        self.assertAlmostEqual(exact, dense)
        self.assertAlmostEqual(exact, .2*.1)

    def test_policy_selection(self):
        p0 = lambda h: 0 if not h else None
        p1 = lambda h: 1 if not h else None
        event = lambda h: bool(h and h[0][1])
        idx, val, scores = select_policy([p0,p1],[(.1,.3),(.4,.6)],1,event)
        self.assertEqual(idx,1)
        self.assertAlmostEqual(val,.4)
        self.assertEqual(len(scores),2)

    def test_repeated_edge_rejected(self):
        with self.assertRaisesRegex(ValueError,'edge reused'):
            exact_robust_value(lambda h: 0,[(.2,.8)],2,lambda h: h[-1][1])

    def test_vertex_fails_without_read_once(self):
        pi=lambda h:0
        ev=lambda h:len(h)==2 and sum(y for _,y in h)==1
        self.assertGreater(event_probability(pi,(.5,),2,ev),
                           max(event_probability(pi,(v,),2,ev) for v in (0.,1.)))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            exact_robust_value(lambda h: None,[(.8,.2)],1,lambda h:True)

if __name__ == '__main__':
    unittest.main()
