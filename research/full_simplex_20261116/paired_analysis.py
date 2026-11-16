"""Paired analysis of actual model-reference probabilities, with SIMULATED preference labels.

A confidence interval across these twelve fixed benchmark questions does not
establish external validity for human alignment.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t as t_distribution, binomtest

def main(input_path, output_path):
    data=json.loads(Path(input_path).read_text())
    rows=data['rows']
    values=np.array([r['true_gain']-r['oracle_pair_upper_gain'] for r in rows],float)
    sample_count=len(values)
    if sample_count < 2:
        raise ValueError('At least two questions are required for paired analysis')
    mean=float(np.mean(values))
    stderr=float(np.std(values,ddof=1)/np.sqrt(sample_count))
    interval=[float(x) for x in (mean-t_distribution.ppf(.975,sample_count-1)*stderr,
                                  mean+t_distribution.ppf(.975,sample_count-1)*stderr)]
    result={
      'model_id':data.get('model_id'),
      'preference_source':'seeded programmatic correct-answer oracle; NOT human',
      'question_count':sample_count,
      'mean_true_multitransfer_gain':float(np.mean([r['true_gain'] for r in rows])),
      'mean_true_one_transfer_oracle_gain':float(np.mean([r['oracle_pair_upper_gain'] for r in rows])),
      'mean_paired_difference':mean,
      'paired_t_interval_95_percent':interval,
      'positive_differences':int(np.sum(values>0)),
      'two_sided_sign_test_p':float(binomtest(int(np.sum(values>0)),sample_count,p=.5).pvalue),
      'certified_full':int(sum(r['certified_full'] for r in rows)),
      'coverage_experiments':int(sum(r['true_score_covered'] for r in rows)),
      'rows':[{'question':r['question'],'paired_gain':float(diff)}
              for r,diff in zip(rows,values)]
    }
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: paired_analysis.py real_llm.json paired_analysis.json')
    main(sys.argv[1],sys.argv[2])
