"""Validate reproducible full-online AFA benchmark counts and raw JSON rows."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CASES=[
 ('online_primary40_3methods.json',40,(34,38,34),(1046.55,1115.025,1047.15)),
 ('online_holdout40_3methods.json',40,(34,37,34),(1051.9,1103.725,1049.55)),
 ('online_cyclic20_3methods.json',20,(18,20,18),(996.8,1075.3,998.45)),
 ('online_null20_3methods.json',20,(0,0,0),(1510,1510,1510)),
]
MODES=('greedy','joint_mix','afa_gated_two_step')

def validate(folder):
    for name,num,counts,means in CASES:
        d=json.loads((folder/name).read_text())
        assert d['protocol']['seeds']==num,(name,'seeds')
        for m,c,mean in zip(MODES,counts,means):
            rows=d['rows'][m]
            assert len(rows)==num,(name,m,'rows')
            assert d['summary'][m]['certified']==c,(name,m,'certificates')
            assert abs(d['summary'][m]['mean_labels']-mean)<1e-8,(name,m,'labels')
            assert sum(r['false_certificate'] for r in rows)==0,(name,m,'false certificate')
            assert d['summary'][m]['certified']==sum(r['certified'] for r in rows),(name,m,'row mismatch')
            assert abs(d['summary'][m]['mean_labels']-sum(r['labels'] for r in rows)/num)<1e-8
        print('PASS',name,'n=',num,'certifications=',counts)

if __name__=='__main__':
    validate(ROOT/'results')
