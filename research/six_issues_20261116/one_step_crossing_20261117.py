"""Exact one-step certificate crossing oracle; candidate, not established scientific novelty."""
import math
from itertools import product

def certificate(means, radii, coeffs, threshold=0.0):
    return max(min(sum(a*m-abs(a)*r for a,m,r in zip(row,means,radii))
                   for row in policy) for policy in coeffs)-threshold

def choose_edge(counts, successes, coeffs, radius, threshold=0.0):
    """Maximize plug-in Bernoulli probability of crossing next-step certificate."""
    if not counts or len(counts)!=len(successes) or any(n<=0 or s<0 or s>n for n,s in zip(counts,successes)):
        raise ValueError("invalid counts")
    E=len(counts)
    if not coeffs or any(not p or any(len(row)!=E for row in p) for p in coeffs):
        raise ValueError("invalid coefficients")
    if not 0<=threshold<math.inf:
        raise ValueError("invalid threshold")
    scores=[]
    for e in range(E):
        n2=list(counts); n2[e]+=1
        rad=[radius(n) for n in n2]
        if any(not math.isfinite(r) or r<0 for r in rad):
            raise ValueError("invalid radius")
        indicators=[]
        for y in (0,1):
            s2=list(successes); s2[e]+=y
            means=[2*s/n-1 for s,n in zip(s2,n2)]
            indicators.append(certificate(means,rad,coeffs,threshold)>0)
        p=successes[e]/counts[e]
        scores.append((1-p)*indicators[0]+p*indicators[1])
    return max(range(E),key=lambda e:scores[e]),scores

if __name__=="__main__":
    import unittest, random
    class TestCrossing(unittest.TestCase):
        def test_exactness(self):
            rng=random.Random(20261117)
            for _ in range(300):
                E=rng.randint(2,5); Q=rng.randint(1,4); H=rng.randint(1,3)
                n=[rng.randint(1,12) for _ in range(E)]
                s=[rng.randint(0,k) for k in n]
                A=[[[rng.uniform(-1,1) for _ in range(E)] for _ in range(H)] for _ in range(Q)]
                rad=lambda x:.3/math.sqrt(x)
                edge,sc=choose_edge(n,s,A,rad,.05)
                brute=[]
                for e in range(E):
                    prob=0.
                    for y in (0,1):
                        nn=n.copy(); ss=s.copy(); nn[e]+=1; ss[e]+=y
                        mm=[2*v/k-1 for v,k in zip(ss,nn)]
                        prob+=(s[e]/n[e] if y else 1-s[e]/n[e])*float(certificate(mm,[rad(k) for k in nn],A,.05)>0)
                    brute.append(prob)
                self.assertEqual(sc,brute)
                self.assertEqual(edge,max(range(E),key=lambda e:brute[e]))
        def test_validation(self):
            with self.assertRaises(ValueError): choose_edge([0],[0],[[[1]]],lambda n:1)
            with self.assertRaises(ValueError): choose_edge([1],[2],[[[1]]],lambda n:1)
            with self.assertRaises(ValueError): choose_edge([1],[0],[[[1,2]]],lambda n:1)
    unittest.main()
