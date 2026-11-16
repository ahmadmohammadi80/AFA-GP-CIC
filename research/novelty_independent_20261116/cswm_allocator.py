"""CSWM: certificate-width feedback with the standard EXP3-IX meta algorithm.

This is an application, NOT an established independent novelty result. Regret
compares experts on the learner's shared states, not their independent rollouts.
Unobserved counterfactual labels are never fabricated. See PROOF_REGRET.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol
import math
import numpy as np


@dataclass(frozen=True)
class BinaryObservation:
    edge: tuple[int, int]
    value: int


class GameState(Protocol):
    active_edges: tuple
    coefficients: np.ndarray  # (Q,H,E), fixed across the run
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    def copy(self): ...
    def observe(self, observation: BinaryObservation): ...
    def allowed_edges(self): ...


def certificate_uncertainty(state: GameState) -> float:
    """Normalized upper bound on the width of max_q min_h a(q,h)^T W.

    C=2 max_{q,h} ||a(q,h)||_1; U=max_{q,h} sum_e |a| width_e / C.
    W is signed and bounded in [-1,1]. This is a sensitivity envelope, not a
    probability of error. Fixed C makes U in [0,1] and differences in [-1,1].
    It also bounds every individual policy's min-h interval width / C.
    """
    a = np.asarray(state.coefficients, dtype=float)
    width = np.asarray(state.upper_bound) - np.asarray(state.lower_bound)
    if a.ndim != 3 or a.shape[-1] != len(width):
        raise ValueError('coefficients must have shape (Q,H,E)')
    if not np.all(np.isfinite(a)) or np.any(width < -1e-12) or np.any(width > 2+1e-12):
        raise ValueError('invalid signed confidence intervals')
    norm = 2 * np.max(np.sum(np.abs(a), axis=-1))
    return 0. if norm == 0 else float(np.max(np.sum(np.abs(a)*width, axis=-1))/norm)


class BaseExpert(Protocol):
    name: str
    def propose_edge(self, state: GameState) -> tuple[int, int]: ...


class CSWMAllocator:
    def __init__(self, experts: list[BaseExpert], eta: float | None, delta: float,
                 seed: int = 0):
        if not experts or not 0 < delta < 1 or (eta is not None and eta <= 0):
            raise ValueError('nonempty experts, positive eta, and delta in (0,1) required')
        self.experts = list(experts)
        self.weights = np.ones(len(experts))/len(experts)
        self.log_weights = np.zeros(len(experts))
        self.eta = eta
        self.delta = delta  # regret confidence; certificate delta lives in state
        self.rng = np.random.default_rng(seed)
        self.coverage_log = []
        self.round = 0
        self.pending = False
        self.epoch = -1

    @staticmethod
    def tuned_eta(k, horizon, delta):
        return math.sqrt((math.log(k)+math.log(2*k/delta))/(k*horizon))

    def _schedule(self):
        if self.eta is not None:
            return self.eta
        epoch = int(math.log2(self.round+1))
        if epoch != self.epoch:
            self.log_weights[:] = 0
            self.weights[:] = 1/len(self.experts)
            self.epoch = epoch
        confidence = 6*self.delta/(math.pi**2*(epoch+1)**2)
        return self.tuned_eta(len(self.experts), 2**epoch, confidence)

    def select_edge(self, state: GameState) -> tuple[int, int]:
        if self.pending:
            raise RuntimeError('update pending observation first')
        self.current_eta = self._schedule()
        # All proposals are fixed BEFORE sampling the expert or seeing a label.
        self.proposals = tuple(e.propose_edge(state) for e in self.experts)
        if any(e not in state.allowed_edges() for e in self.proposals):
            raise ValueError('expert violated the shared coverage constraint')
        self.last_expert = int(self.rng.choice(len(self.experts), p=self.weights))
        self.last_state_before = state.copy()
        self.last_prob = float(self.weights[self.last_expert])
        self.pending = True
        return self.proposals[self.last_expert]

    def estimate_counterfactual(self, expert, state, observation):
        """Only same-edge potential outcome is observed; other edges return None."""
        idx = next(i for i, e in enumerate(self.experts) if e is expert)
        if self.proposals[idx] != observation.edge:
            return None
        return certificate_uncertainty(state)-certificate_uncertainty(state.observe(observation))

    def update(self, observation: BinaryObservation, certificate_before=None,
               certificate_after=None):
        if not self.pending:
            raise RuntimeError('select_edge must precede update')
        if observation.edge != self.proposals[self.last_expert] or observation.value not in (0,1):
            raise ValueError('observation must match selected edge and be binary')
        before = certificate_uncertainty(self.last_state_before)
        after = certificate_uncertainty(self.last_state_before.observe(observation))
        reward = before-after
        # Optional certificate scalars are logged, never mistaken for uncertainty.
        self.update_bandit_reward(self.last_expert, reward, self.last_prob, self.current_eta)
        self.coverage_log.append(dict(round=self.round, expert=self.last_expert,
            edge=observation.edge, probability=self.last_prob, uncertainty_before=before,
            uncertainty_after=after, reward=reward, weights=self.weights.tolist(),
            certificate_before=certificate_before, certificate_after=certificate_after))
        self.pending = False

    def update_bandit_reward(self, idx, reward, probability, eta):
        """Selected-expert EXP3-IX update; also exposed for synthetic payoff tests."""
        if not np.isfinite(reward) or not -1-1e-12 <= reward <= 1+1e-12:
            raise ValueError('reward must be in [-1,1]')
        loss = (1-reward)/2
        self.log_weights[idx] -= eta*loss/(probability+eta/2)
        self.log_weights -= self.log_weights.max()
        self.weights = np.exp(self.log_weights)
        self.weights /= self.weights.sum()
        self.round += 1


class GateState:
    """Adapter for the frozen 49-world gate; no true p is stored in this state."""
    active_edges = ((0,1),(0,2))
    def __init__(self, gate, counts):
        self.gate = gate
        self.counts = tuple(counts)
        self.coefficients = -gate.shift*np.eye(2).reshape(2,1,2)
        bounds = [gate.intervals[self.counts[j],self.counts[j+1]] for j in (0,2)]
        self.lower_bound = np.array([x[0] for x in bounds])
        self.upper_bound = np.array([x[1] for x in bounds])
    def copy(self):
        return GateState(self.gate,self.counts)
    def allowed_edges(self):
        return tuple(self.active_edges[e] for e in self.gate.allowed(self.counts))
    def observe(self, observation):
        e=self.active_edges.index(observation.edge)
        return GateState(self.gate,self.gate.successors(self.counts,e)[observation.value])


class FiniteGameState:
    """General finite coefficient game, stationary or KNOWN bounded drift.

    Q's KL feasibility and A's construction belong to the caller (the existing
    guarded engine validates b,Q,H). No unknown true means enter this object.
    Counts, wins, sample-time sums and time are sufficient for these intervals.
    """
    def __init__(self, coefficients, active_edges, counts, wins, budget,
                 threshold=.015, delta=.05, prior=(.5,.5), period=20,
                 rho=0., sample_times=None, now=None):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'six_issues_20261116'))
        from theory_and_guarantees import clipped_intersection
        from bounded_drift import bound
        self.coefficients=np.asarray(coefficients,dtype=float).copy()
        self.active_edges=tuple(map(tuple,active_edges))
        self.n=np.asarray(counts,dtype=int).copy();self.s=np.asarray(wins,dtype=int).copy()
        self.B=int(budget);self.tau=threshold;self.delta=delta;self.prior=prior
        self.period=period;self.rho=rho;self.now=int(self.n.sum()) if now is None else now
        self.sample_times=np.zeros(len(self.n)) if sample_times is None else np.asarray(sample_times).copy()
        if len(self.n)!=len(self.active_edges) or self.coefficients.shape[-1]!=len(self.n):
            raise ValueError('edge dimensions disagree')
        if np.any(self.n<0) or np.any(self.s<0) or np.any(self.s>self.n) or self.n.sum()>self.B:
            raise ValueError('invalid counts or budget')
        if not 0<delta<1 or period<1 or rho<0 or min(prior)<=0:
            raise ValueError('invalid confidence, period, drift or prior')
        if rho and sample_times is None and self.n.sum():
            raise ValueError('drift requires actual sample-time sums')
        bounds=[]
        for e,(n,w) in enumerate(zip(self.n,self.s)):
            if rho:
                z=bound(int(w),int(n),self.sample_times[e],self.now,rho,delta,len(self.n))
            else:
                z=clipped_intersection(int(w),int(n),delta,len(self.n)) if n else (-1.,1.)
            bounds.append(z)
        self.lower_bound=np.array([x[0] for x in bounds])
        self.upper_bound=np.array([x[1] for x in bounds])
    def copy(self):
        import copy
        return copy.deepcopy(self)
    def allowed_edges(self):
        missing=np.flatnonzero(self.n==0)
        if len(missing):return (self.active_edges[missing[0]],)
        post=int(self.n.sum())-len(self.n)
        if (post+1)%self.period==0:
            return (self.active_edges[((post+1)//self.period-1)%len(self.n)],)
        return self.active_edges
    def observe(self,observation):
        if observation.value not in (0,1) or self.n.sum()>=self.B:
            raise ValueError('invalid label or exhausted budget')
        e=self.active_edges.index(observation.edge)
        n=self.n.copy();w=self.s.copy();times=self.sample_times.copy()
        n[e]+=1;w[e]+=observation.value;times[e]+=self.now+1
        return FiniteGameState(self.coefficients,self.active_edges,n,w,self.B,
            self.tau,self.delta,self.prior,self.period,self.rho,times,self.now+1)
    def lower(self):
        return np.minimum(self.coefficients*self.lower_bound,
                          self.coefficients*self.upper_bound).sum(axis=-1).min(axis=1)
    def terminal(self):return bool(self.lower().max()>self.tau)
    def greedy_edge(self):
        allowed=self.allowed_edges()
        if len(allowed)==1:return allowed[0]
        lower_h=np.minimum(self.coefficients*self.lower_bound,
                           self.coefficients*self.upper_bound).sum(axis=-1)
        q=int(np.argmax(lower_h.min(axis=1)));h=int(np.argmin(lower_h[q]))
        score=np.abs(self.coefficients[q,h])*(self.upper_bound-self.lower_bound)/(np.sqrt(self.n)+.2)
        return max(allowed,key=lambda e:score[self.active_edges.index(e)])


class GreedyExpert:
    name = 'greedy'
    def propose_edge(self, state):
        return (state.greedy_edge() if isinstance(state,FiniteGameState) else
                state.active_edges[state.gate.heuristic(state.counts,'greedy')])


class BellmanExpert:
    name = 'bellman_h2'
    def __init__(self, gate=None, horizon=2):
        if horizon not in (1,2,None):
            raise ValueError('horizon must be 1, 2, or None (full remaining budget)')
        self.gate=gate
        self.horizon=horizon
        self.name='bellman_full' if horizon is None else f'bellman_h{horizon}'
    @lru_cache(None)
    def value(self, counts, depth):
        g=self.gate
        if g.terminal(counts):
            return 1.,-1
        if depth==0 or counts[0]+counts[2]>=g.B:
            return 0.,-1
        values=[]
        greedy=g.heuristic(counts,'greedy')
        for e in g.allowed(counts):
            n,s=counts[2*e:2*e+2]
            p=(s+g.a)/(n+g.a+g.b)
            z,o=g.successors(counts,e)
            v=(1-p)*self.value(z,depth-1)[0]+p*self.value(o,depth-1)[0]
            values.append((v,e))
        # Exact finite-horizon probability; explicit greedy tie break on flat values.
        return max(values,key=lambda x:(x[0],x[1]==greedy,-x[1]))
    def generic_value(self,state,depth):
        if state.terminal():return 1.,None
        if depth==0 or state.n.sum()>=state.B:return 0.,None
        greedy=state.greedy_edge();values=[]
        for edge in state.allowed_edges():
            e=state.active_edges.index(edge)
            p=(state.s[e]+state.prior[0])/(state.n[e]+sum(state.prior))
            value=sum(prob*self.generic_value(state.observe(BinaryObservation(edge,y)),depth-1)[0]
                      for y,prob in ((0,1-p),(1,p)))
            values.append((value,edge))
        return max(values,key=lambda x:(x[0],x[1]==greedy))
    def propose_edge(self,state):
        if isinstance(state,FiniteGameState):
            depth=state.B-int(state.n.sum()) if self.horizon is None else self.horizon
            return self.generic_value(state,depth)[1]
        e=(self.gate.optimum(state.counts)[1] if self.horizon is None
           else self.value(state.counts,self.horizon)[1])
        return state.active_edges[e]


class ThompsonExpert:
    name='thompson_decision'
    def __init__(self, seed=1):
        self.rng=np.random.default_rng(seed)
    def propose_edge(self,state):
        if isinstance(state,FiniteGameState):
            allowed=state.allowed_edges()
            if len(allowed)==1:return allowed[0]
            p=self.rng.beta(state.s+state.prior[0],state.n-state.s+state.prior[1])
            scores=[];current=state.terminal()
            for edge in allowed:
                e=state.active_edges.index(edge)
                scores.append(sum(prob*(state.observe(BinaryObservation(edge,y)).terminal()!=current)
                                  for y,prob in ((0,1-p[e]),(1,p[e]))))
            if max(scores)>0:return allowed[int(np.argmax(scores))]
            # Thompson fallback: sampled best policy, its sampled worst opponent,
            # then largest sensitivity-width contribution per sqrt sample count.
            values=(state.coefficients*(2*p-1)).sum(axis=-1)
            q=int(np.argmax(values.min(axis=1)));h=int(np.argmin(values[q]))
            score=np.abs(state.coefficients[q,h])*(state.upper_bound-state.lower_bound)/(np.sqrt(state.n)+.2)
            return max(allowed,key=lambda e:score[state.active_edges.index(e)])
        allowed=state.gate.allowed(state.counts)
        if len(allowed)==1:
            return state.active_edges[allowed[0]]
        g=state.gate
        n=np.array(state.counts[::2]);s=np.array(state.counts[1::2])
        p=self.rng.beta(s+g.a,n-s+g.b)
        # Posterior-sampled probability that ONE next label changes the
        # certificate decision (accept/reject); fallback samples best true policy.
        scores=[]
        current=g.terminal(state.counts)
        for e in allowed:
            z,o=g.successors(state.counts,e)
            scores.append((1-p[e])*(g.terminal(z)!=current)+p[e]*(g.terminal(o)!=current))
        if max(scores)>0:
            e=allowed[int(np.argmax(scores))]
        else:
            e=allowed[int(np.argmin(p[list(allowed)]))]
        return state.active_edges[e]
