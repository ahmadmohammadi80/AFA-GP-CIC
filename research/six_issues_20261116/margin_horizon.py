"""Finite-budget, margin-aware certificate planner; deterministic surrogate only."""
from math import sqrt

def certificate(scores, weights, counts, increments, scale):
    """Best worst-opponent lower certificate after a fixed allocation."""
    return max(min(scores[q][h] - sum(weights[q][h][e]*scale[e]/sqrt(counts[e]+increments[e]) for e in range(len(counts))) for h in range(len(scores[q]))) for q in range(len(scores)))

def allocations(edges, budget):
    if edges == 1:
        yield (budget,)
    else:
        for i in range(budget+1):
            for rest in allocations(edges-1,budget-i):
                yield (i,)+rest

def plan(scores, weights, counts, scale, budget):
    if not isinstance(budget,int) or budget<0 or not counts or any(n<1 for n in counts):
        raise ValueError('invalid counts/budget')
    E=len(counts)
    if len(scale)!=E or not scores or len(weights)!=len(scores):
        raise ValueError('dimensions')
    for q in range(len(scores)):
        if not scores[q] or len(weights[q])!=len(scores[q]):
            raise ValueError('opponents')
        for row in weights[q]:
            if len(row)!=E or any(v<0 for v in row):
                raise ValueError('weights')
    best=None
    for x in allocations(E,budget):
        v=certificate(scores,weights,counts,x,scale)
        if best is None or v>best[0]+1e-12:
            best=(v,x)
    return best
