"""Prespecified gate-width sensitivity of the FROZEN AFA two-query controller.
No changes to online_two_step_benchmark_20261118.py or its scientific decisions.
Uses matched seeds for all conditions; exploratory, not a tuned deployment.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from online_two_step_benchmark_20261118 import run_one

GATES=(0.0, 0.0001, 0.0005, 0.001)

def run_sensitivity(seeds=20, offset=20261116, budget=1510):
    results={}
    for gate in GATES:
        start=time.perf_counter()
        rows=[run_one(seed, 'afa_gated_two_step',gate=gate,budget=budget)
              for seed in range(offset,offset+seeds)]
        summary={'gate':gate, 'seeds':seeds,
            'certified':sum(x['certified'] for x in rows),
            'mean_labels':sum(x['labels'] for x in rows)/seeds,
            'mean_elapsed_sec':sum(x['elapsed_sec'] for x in rows)/seeds,
            'planner_calls':sum(x['planner_calls'] for x in rows),
            'positive_plans':sum(x['positive_plans'] for x in rows),
            'root_overrides':sum(x['root_overrides'] for x in rows),
            'false_certificates':sum(x['false_certificate'] for x in rows),
            'coverage_breaches':sum(x['coverage_breached'] for x in rows)}
        assert summary['false_certificates']==0
        print('SENSITIVITY',json.dumps(summary,sort_keys=True),
              'wall_seconds',round(time.perf_counter()-start,3),flush=True)
        results[str(gate)]={'summary':summary,'rows':rows}
    greedy=[run_one(seed,'greedy',budget=budget)
            for seed in range(offset,offset+seeds)]
    for x,y in zip(results['0.0']['rows'],greedy):
        assert all(x[k]==y[k] for k in (
            'certified','labels','winner','query_counts','false_certificate'))
    print('GATE_ZERO_EQUALS_GREEDY PASS',flush=True)
    return {'origin':'original online_two_step_benchmark_20261118.run_one',
      'protocol':{'offset':offset,'seeds':seeds,'budget':budget,'gate_grid':GATES,
      'study':'exploratory mechanism sensitivity, not independent holdout'},
      'gate_zero_matches_greedy':True,'results':results}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=20)
    ap.add_argument('--offset',type=int,default=20261116)
    ap.add_argument('--out',type=Path,default=Path('results/gate_sensitivity20.json'))
    args=ap.parse_args()
    result=run_sensitivity(args.seeds,args.offset)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
