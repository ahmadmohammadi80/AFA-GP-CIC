"""Real pretrained language model pilot for AFA-GP-CIC.

The LLM supplies *reference probabilities* over A..F multiple-choice responses;
comparative feedback is an EXPLICITLY SYNTHETIC, seeded, programmatic
correctness oracle. This is not human-preference alignment evaluation.

Run on GitHub Actions or a network-enabled machine:
 pip install numpy scipy torch 'transformers>=4.47,<5' huggingface-hub
 python real_llm_pilot.py --out outputs/real_llm.json
"""
import argparse
import json
import numpy as np
from pathlib import Path
from strong_adaptive_study import kl,robust_gain,robust_full_policy,oracle_kl_policy,strongest_pair_oracle,sample_confidence

MODEL='HuggingFaceTB/SmolLM2-135M-Instruct'
QA=[
('Which is the capital of France?', ['Berlin','Paris','Madrid','Rome','Vienna','Lisbon'],1),
('Which is the largest planet in our solar system?', ['Mars','Earth','Venus','Saturn','Jupiter','Mercury'],4),
('Which gas do humans need for respiration?', ['Helium','Oxygen','Nitrogen','Argon','Neon','Hydrogen'],1),
('How many days are in a leap year?', ['360','365','366','364','367','350'],2),
('Which planet is closest to the Sun?', ['Neptune','Earth','Saturn','Mercury','Mars','Uranus'],3),
('Which mathematical operation is the inverse of multiplication?', ['Addition','Subtraction','Division','Exponentiation','Squaring','Counting'],2),
('Which of these is a mammal?', ['Salmon','Crocodile','Whale','Eagle','Frog','Trout'],2),
('Which shape has three sides?', ['Circle','Square','Pentagon','Hexagon','Triangle','Octagon'],4),
('What is the chemical symbol for gold?', ['Ag','Fe','Cu','Hg','Au','Pb'],4),
('Which ocean is the largest?', ['Atlantic','Indian','Arctic','Pacific','Southern','Mediterranean'],3),
('How many minutes are in an hour?', ['30','120','45','90','60','75'],4),
('What is the opposite of north?', ['East','West','South','Up','Down','Left'],2),
]


def true_preferences(correct,k=6):
    """Simulated pairwise feedback: correct answer dominates each wrong answer;
    add a 3-cycle over three wrong answers. All probabilities stationary.
    """
    W=np.zeros((k,k))
    for j in range(k):
        if j!=correct:W[correct,j]=.7;W[j,correct]=-.7
    wrong=[j for j in range(k) if j!=correct]
    i,j,h=wrong[:3]
    for a,b in [(i,j),(j,h),(h,i)]: W[a,b]=.25;W[b,a]=-.25
    assert np.allclose(W,-W.T)
    return W


def score_reference_probs(model_id=MODEL):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.manual_seed(20261116);torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id).eval()
    rows=[]
    for question,choices,correct in QA:
        text='Answer with one letter only.\nQuestion: '+question+'\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(choices))+'\nAnswer:'
        context=tokenizer.apply_chat_template([{'role':'user','content':text}], tokenize=False,add_generation_prompt=True)
        token_ids=tokenizer(context,return_tensors='pt',add_special_tokens=False)['input_ids']
        with torch.inference_mode(): logits=model(token_ids).logits[0,-1].float()
        lps=[]
        for letter in 'ABCDEF':
            tail=tokenizer(' '+letter,add_special_tokens=False)['input_ids']
            if len(tail)==1:
                lps.append(float(torch.log_softmax(logits,dim=-1)[tail[0]].item()))
            else:
                seq=torch.cat((token_ids,torch.tensor([tail],dtype=token_ids.dtype)),dim=-1)
                with torch.inference_mode(): l=model(seq).logits[0,token_ids.shape[-1]-1:-1].float()
                lp=torch.log_softmax(l,dim=-1)[torch.arange(len(tail)),torch.tensor(tail)].sum()
                lps.append(float(lp.item()))
        lps=np.asarray(lps)
        # Tempered restriction to the six answer continuations, with .05 uniform
        # floor. This is a *constructed* candidate distribution, not raw model P.
        u=np.exp((lps-lps.max())/1.5);u/=u.sum()
        b=.95*u+.05/len(u)
        rows.append({'question':question,'choices':choices,'correct_index':int(correct),'candidate_logprobs':lps.tolist(),'reference_policy':b.tolist()})
    return rows


def simulate(rows,n=5000,kappa=.15,gamma=.01,delta=.05):
    out=[]
    for i,item in enumerate(rows):
        b=np.asarray(item['reference_policy']); W=true_preferences(item['correct_index'])
        rng=np.random.default_rng(99119+i)
        _,lo,hi=sample_confidence(rng,W,b,n=n,delta=delta)
        q=robust_full_policy(b,lo,hi,kappa)
        score=W@b
        true=float(q@score)
        lb=robust_gain(q,b,lo,hi)
        pair_val,pair=strongest_pair_oracle(b,score,kappa)
        free=float(oracle_kl_policy(score,b,kappa)@score)
        out.append({**item,'n_per_answer':n,'queries':len(b)*n,
                    'certified_full':bool(lb>gamma),'certified_gain_lb':lb,
                    'true_gain':true,'policy_kl':kl(q,b),'oracle_pair_upper_gain':float(pair_val),
                    'oracle_full_gain':free,'true_score_covered':bool(np.all(lo<=score)&np.all(score<=hi)),
                    'q_policy':q.tolist()})
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default=MODEL);p.add_argument('--out',default='outputs/real_llm.json');p.add_argument('--n',type=int,default=5000)
    p.add_argument('--reference-json',default='');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.reference_json:
        rows=json.load(open(args.reference_json))
    elif args.dry_run:
        # Synthetic probabilities ONLY for verifying plumbing offline.
        rng=np.random.default_rng(33)
        rows=[{'question':x,'choices':v,'correct_index':int(c),'reference_policy':rng.dirichlet(np.ones(6)*2).tolist(),'candidate_logprobs':None} for x,v,c in QA]
    else:rows=score_reference_probs(args.model)
    res=simulate(rows,n=args.n)
    obj={'evaluation_type':'real_lm_reference_probabilities_and_programmatic_synthetic_preference_oracle' if not(args.dry_run or args.reference_json) else 'dry_run_no_real_llm',
         'model_id':args.model if not(args.dry_run or args.reference_json) else None,'n_questions':len(res),'n_per_response':args.n,
         'certified_full':sum(x['certified_full'] for x in res),'oracle_pair_possible':sum(x['oracle_pair_upper_gain']>.01 for x in res),
         'oracle_full_possible':sum(x['oracle_full_gain']>.01 for x in res),
         'all_covered':sum(x['true_score_covered'] for x in res),'rows':res}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))
    print(json.dumps({k:v for k,v in obj.items() if k!='rows'},indent=2))
    return obj

if __name__=='__main__':main()
