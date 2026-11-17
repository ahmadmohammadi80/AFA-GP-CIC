"""Exact finite-horizon adaptive certificate crossing under frozen plug-in Bernoulli probabilities."""
from functools import lru_cache
from math import sqrt

def crossing(n,s,A,threshold=0.0,c=0.3):
    means=[2*x/k-1 for x,k in zip(s,n)]
    rad=[c/sqrt(k) for k in n]
    return max(min(sum(a*m-abs(a)*r for a,m,r in zip(row,means,rad)) for row in policy) for policy in A)>threshold

def solve(n,s,A,budget,threshold=0.0,c=0.3):
    """Optimal plug-in crossing probability, first query edge, myopic policy probability."""
    if not n or len(n)!=len(s) or any(k<1 or x<0 or x>k for k,x in zip(n,s)) or budget<0 or not A or any(not p or any(len(row)!=len(n) for row in p) for p in A):
        raise ValueError('invalid state')
    n=tuple(n);s=tuple(s)
    p=tuple(x/k for x,k in zip(s,n))
    @lru_cache(None)
    def value(nn,ss,t):
        if crossing(nn,ss,A,threshold,c):return 1.0,None
        if t==0:return 0.0,None
        scores=[]
        for e in range(len(nn)):
            n2=list(nn);n2[e]+=1
            s0=list(ss);s1=list(ss);s1[e]+=1
            scores.append((1-p[e])*value(tuple(n2),tuple(s0),t-1)[0]+p[e]*value(tuple(n2),tuple(s1),t-1)[0])
        edge=max(range(len(nn)),key=lambda i:scores[i])
        return scores[edge],edge
    optimal,edge=value(n,s,budget)
    @lru_cache(None)
    def greedy(nn,ss,t):
        if crossing(nn,ss,A,threshold,c):return 1.0
        if t==0:return 0.0
        immediate=[]
        for e in range(len(nn)):
            n2=list(nn);n2[e]+=1;s0=list(ss);s1=list(ss);s1[e]+=1
            immediate.append((1-p[e])*crossing(n2,s0,A,threshold,c)+p[e]*crossing(n2,s1,A,threshold,c))
        e=max(range(len(nn)),key=lambda i:immediate[i]);n2=list(nn);n2[e]+=1;s0=list(ss);s1=list(ss);s1[e]+=1
        return (1-p[e])*greedy(tuple(n2),tuple(s0),t-1)+p[e]*greedy(tuple(n2),tuple(s1),t-1)
    return optimal,edge,greedy(n,s,budget)

if __name__=='__main__':
    import unittest,random
    class Tests(unittest.TestCase):
        def test_domination(self):
            rng=random.Random(71)
            for _ in range(120):
                E=rng.randint(2,3);n=[rng.randint(1,4) for _ in range(E)];s=[rng.randint(0,k) for k in n]
                A=[[[rng.uniform(-1,1) for _ in range(E)] for _ in range(2)] for _ in range(2)]
                v,e,g=solve(n,s,A,3,0.03)
                self.assertGreaterEqual(v+1e-12,g)
                self.assertTrue(e is None or 0<=e<E)
        def test_zero_horizon(self):
            self.assertEqual(solve([2],[1],[[[1]]],0)[1],None)
        def test_invalid(self):
            with self.assertRaises(ValueError):solve([0],[0],[[[1]]],2)
    unittest.main()
