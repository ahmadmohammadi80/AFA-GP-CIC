"""MACWA: exact one-step minimax allocation for certificate widths."""
from math import sqrt,log,pi
def radius(n,delta,E):
    if n<1 or E<1 or not 0<delta<1:raise ValueError("inputs")
    return min(1.,sqrt(2*log(E*pi*pi*n*n/(3*delta))/n))
def choose_minimax_edge(rows,counts,delta=.05,active=None):
    if not rows or not counts or any(len(row)!=len(counts) for row in rows):raise ValueError("shape")
    if any(n<1 for n in counts):raise ValueError("counts")
    if active is None:active=list(range(len(rows)))
    if not active or any(i<0 or i>=len(rows) for i in active):raise ValueError("active")
    E=len(counts);r=[radius(n,delta,E) for n in counts]
    widths=[sum(abs(rows[i][e])*r[e] for e in range(E)) for i in active]
    costs=[]
    for e in range(E):
        reduction=r[e]-radius(counts[e]+1,delta,E)
        costs.append(max(widths[j]-abs(rows[i][e])*reduction for j,i in enumerate(active)))
    e=min(range(E),key=lambda e:(costs[e],e))
    return e,costs[e],costs
def exhaustive_oracle(rows,counts,delta=.05,active=None):
    if active is None:active=list(range(len(rows)))
    costs=[]
    for e in range(len(counts)):
        c=list(counts);c[e]+=1
        costs.append(max(sum(abs(rows[i][j])*radius(c[j],delta,len(c)) for j in range(len(c))) for i in active))
    return min(range(len(counts)),key=lambda e:(costs[e],e)),costs
