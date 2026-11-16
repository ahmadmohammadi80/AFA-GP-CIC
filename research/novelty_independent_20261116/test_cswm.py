import sys,json
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'core_gate_20261116'))
from exact_budget_gate import Gate
from cswm_allocator import *

@pytest.fixture(scope='module')
def gate():return Gate()

def test_uncertainty_bounds_and_normalization(gate):
    for s in ((1,0,1,1),(10,0,8,7),(12,6,12,6)):
        state=GateState(gate,s)
        u=certificate_uncertainty(state)
        assert 0<=u<=1
        assert u==pytest.approx(max(state.upper_bound-state.lower_bound)/2)
        if sum(s[::2])<24:
            for e in state.active_edges:
                for y in (0,1):
                    assert -1<=u-certificate_uncertainty(state.observe(BinaryObservation(e,y)))<=1

def test_no_unobserved_counterfactual(gate):
    class Fixed:
        def __init__(self,e):self.e=e
        def propose_edge(self,s):return self.e
    experts=[Fixed((0,1)),Fixed((0,2))]
    a=CSWMAllocator(experts,.1,.05)
    s=GateState(gate,(1,0,1,0));e=a.select_edge(s);o=BinaryObservation(e,0)
    for expert in experts:
        value=a.estimate_counterfactual(expert,s,o)
        assert (value is None)==(expert.e!=e)
    a.update(o)
    assert np.isfinite(a.weights).all() and sum(a.weights)==pytest.approx(1)
    with pytest.raises(RuntimeError):a.update(o)

def test_shared_coverage(gate):
    s=GateState(gate,(11,5,10,5))
    experts=[GreedyExpert(),BellmanExpert(gate),ThompsonExpert()]
    assert len(s.allowed_edges())==1
    assert all(e.propose_edge(s)==s.allowed_edges()[0] for e in experts)

def test_gapped_synthetic_convergence():
    # External stationary payoff diagnostic, NOT certificate-world convergence.
    a=CSWMAllocator([None]*3,.05,.05,seed=12)
    rewards=np.array([1.,-1.,-1.]);total=0
    for _ in range(5000):
        i=int(a.rng.choice(3,p=a.weights));p=a.weights[i]
        a.update_bandit_reward(i,rewards[i],p,.05);total+=rewards[i]
    assert a.weights[0]>.99
    assert 5000-total<500

def test_vector_update_matches_class():
    a=CSWMAllocator([None]*3,.2,.05)
    logw=np.zeros(3)
    for i,r in [(0,.1),(2,-.1),(1,0),(0,.2)]*20:
        p=np.exp(logw-logw.max());p/=p.sum()
        a.update_bandit_reward(i,r,p[i],.2)
        logw[i]-=.2*(1-r)/2/(p[i]+.1)
        w=np.exp(logw-logw.max());w/=w.sum()
        assert np.allclose(w,a.weights,rtol=1e-13,atol=1e-14)

def test_general_state_and_experts():
    A=np.array([[[-.12,0]],[[0,-.12]]])
    s=FiniteGameState(A,((0,1),(0,2)),[1,1],[0,1],24)
    experts=[GreedyExpert(),BellmanExpert(horizon=1),ThompsonExpert()]
    a=CSWMAllocator(experts,None,.05)
    for _ in range(6):
        e=a.select_edge(s);o=BinaryObservation(e,0);a.update(o);s=s.observe(o)
        assert 0<=certificate_uncertainty(s)<=1
    assert len(a.coverage_log)==6

def test_general_adapter_matches_frozen_gate(gate):
    for counts in ((1,0,1,1),(10,0,8,7),(11,2,10,8)):
        s=GateState(gate,counts)
        general=FiniteGameState(s.coefficients,s.active_edges,counts[::2],counts[1::2],24)
        assert certificate_uncertainty(s)==pytest.approx(certificate_uncertainty(general))
        assert np.allclose(gate.lower(counts),general.lower())
        assert GreedyExpert().propose_edge(s)==GreedyExpert().propose_edge(general)
        assert BellmanExpert(gate,2).propose_edge(s)==BellmanExpert(horizon=2).propose_edge(general)

def test_grid_integrity_and_monte_carlo_baselines():
    r=json.loads((HERE/'cswm_results.json').read_text())
    assert len(r['grid'])==49
    for row in r['grid']:
        for mode in ('greedy','joint_mix','bellman'):
            mc=row['methods'][mode];ex=row['exact_fixed'][mode]
            # Conservative fixed-test tolerance; exact references are independent.
            assert abs(mc['success_probability']-ex['success_probability'])<.04
            assert abs(mc['expected_labels']-ex['expected_labels'])<.65
        for mc in row['methods'].values():
            assert 2<=mc['expected_labels']<=24
            assert 0<=mc['success_probability']<=1
            assert sum(mc['mean_final_weights'])==pytest.approx(1)
    # A negative research gate must remain negative; software tests cannot turn it green.
    assert not r['gate_A']['cswm']['passed']

def test_drift_state_refreshes_unqueried_edges():
    A=np.array([[[-.12,0]],[[0,-.12]]])
    s=FiniteGameState(A,((0,1),(0,2)),[30,30],[0,0],100,rho=.001,
                      sample_times=[sum(range(1,31)),sum(range(31,61))],now=60)
    after=s.observe(BinaryObservation((0,1),0))
    assert after.upper_bound[1]>=s.upper_bound[1]
    assert after.now==61


def test_scalar_allocator_matches_vector_evaluator(gate):
    from run_experiments import tables,simulation
    tab,code=tables(gate)
    for seed in range(12):
        streams=np.random.default_rng(seed).random((1,2,24));p=np.array([[.2,.4]])
        pilot=(streams[0,:,0]<p[0]).astype(int)
        state=GateState(gate,(1,int(pilot[0]),1,int(pilot[1])))
        th=ThompsonExpert()
        eta=CSWMAllocator.tuned_eta(3,22,.05)
        a=CSWMAllocator([GreedyExpert(),BellmanExpert(gate,2),th],eta,.05,seed=seed+99)
        th.rng=a.rng  # same draw order as the N=1 vector evaluator
        for t in range(2,24):
            if gate.terminal(state.counts):break
            edge=a.select_edge(state);e=state.active_edges.index(edge)
            n=state.counts[2*e]
            y=int(streams[0,e,n]<p[0,e]);o=BinaryObservation(edge,y)
            a.update(o);state=state.observe(o)
        r=simulation(gate,tab,code,p,streams,'cswm',seed+99)
        assert r['successes']==int(gate.terminal(state.counts))
        assert r['expected_labels']==sum(state.counts[::2])
        assert np.allclose(r['mean_final_weights'],a.weights,rtol=1e-12,atol=1e-14)


def test_ix_concentration_one_step_mgf():
    rng=np.random.default_rng(2026)
    for _ in range(200):
        p=rng.dirichlet(np.ones(3));loss=rng.random(3);gamma=rng.uniform(.01,.8)
        alpha=rng.uniform(0,2*gamma,3)
        mgf=np.sum(p*np.exp(alpha*loss/(p+gamma)-np.sum(alpha*loss)))
        assert mgf<=1+1e-12
