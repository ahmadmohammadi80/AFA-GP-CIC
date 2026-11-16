"""Real pretrained LM reference distribution + programmatic preference pilot.

CAUTION: LLM weights are actually loaded. Preference labels are simulated from
ground-truth multiple-choice answers, not human annotations. This does NOT test
real-human RLHF or production inference speed. Research reproducibility only.
"""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
QA=[
("What is the capital of France?",["Berlin","Paris","Madrid","Rome","Vienna","Lisbon"],1),
("Which planet is the largest in our solar system?",["Mars","Earth","Venus","Saturn","Jupiter","Mercury"],4),
("Which gas do humans need for respiration?",["Helium","Oxygen","Nitrogen","Argon","Neon","Hydrogen"],1),
("How many days are in a leap year?",["360","365","366","364","367","350"],2),
("Which planet is closest to the Sun?",["Neptune","Earth","Saturn","Mercury","Mars","Uranus"],3),
("Which mathematical operation is inverse to multiplication?",["Addition","Subtraction","Division","Exponentiation","Squaring","Counting"],2),
("Which of these is a mammal?",["Salmon","Crocodile","Whale","Eagle","Frog","Trout"],2),
("Which shape has three sides?",["Circle","Square","Pentagon","Hexagon","Triangle","Octagon"],4),
("What is the chemical symbol for gold?",["Ag","Fe","Cu","Hg","Au","Pb"],4),
("Which ocean is the largest?",["Atlantic","Indian","Arctic","Pacific","Southern","Mediterranean"],3),
("How many minutes are in one hour?",["30","120","45","90","60","75"],4),
("What is the opposite of north?",["East","West","South","Up","Down","Left"],2),
]
def kl(q,b):
    t=q>0
    return float(np.sum(q[t]*np.log(q[t]/b[t])))

def interval(rng,b,W,n=5000,delta=.05):
    k=len(b)
    radius=min(1.,np.sqrt(2*np.log(np.pi**2*k*n*n/(3*delta))/n))
    ms=[]
    for i in range(k):
        js=rng.choice(k,size=n,p=b)
        prob=(1+W[i,js])/2
        y=2*(rng.random(n)<prob).astype(float)-1
        y[js==i]=0
        ms.append(y.mean())
    m=np.array(ms)
    return np.maximum(-1,m-radius),np.minimum(1,m+radius)

def lower_gain(q,b,L,U):
    d=q-b
    return float(np.where(d>=0,d*L,d*U).sum())

def optimize_certificate(b,L,U,kappa=.15):
    mid=(L+U)/2
    def fun(q): return -lower_gain(q,b,L,U)
    def constraints():
        return [
            {'type':'eq','fun':lambda q:float(sum(q)-1.),'jac':lambda q:np.ones_like(q)},
            {'type':'ineq','fun':lambda q:float(kappa-kl(q,b)),
             'jac':lambda q:-np.log(np.maximum(q,1e-12)/b)-1.}
        ]
    x0=.97*b +.03*np.eye(len(b))[int(np.argmax(mid))]
    candidates=[]
    for start in [x0,b.copy()]:
        r=minimize(fun,start,method='SLSQP',bounds=[(1e-10,1)]*len(b),
                   constraints=constraints(),options={'maxiter':800,'ftol':1e-10})
        if abs(r.x.sum()-1)<1e-6 and kl(r.x,b)<=kappa+1e-8:
            candidates.append(r.x)
    if not candidates:raise RuntimeError('No feasible solution')
    q=min(candidates,key=fun)
    # restore KL feasibility even if SLSQP slightly overshoots due floating error
    if kl(q,b)>kappa:
        lo,hi=0.,1.
        for _ in range(60):
            m=(lo+hi)/2
            if kl(b+m*(q-b),b)<=kappa:lo=m
            else:hi=m
        q=b+lo*(q-b)
    return q

def W_oracle(correct,k=6):
    w=np.zeros((k,k))
    for j in range(k):
        if j!=correct:w[correct,j]=.7;w[j,correct]=-.7
    wrong=[j for j in range(k) if j!=correct]
    for a,b in [(wrong[0],wrong[1]),(wrong[1],wrong[2]),(wrong[2],wrong[0])]:
        w[a,b]=.25;w[b,a]=-.25
    return w

def pair_kl(b,i,j,kappa):
    lo,hi=0.,float(b[j])
    def q(t):
        p=b.copy();p[i]+=t;p[j]-=t;return p
    if kl(q(hi),b)<=kappa:return hi
    for _ in range(55):
        mid=(lo+hi)/2
        if kl(q(mid),b)<=kappa:lo=mid
        else:hi=mid
    return lo

def oracle_pair_best(b,s,kappa):
    return float(max((pair_kl(b,i,j,kappa)*(s[i]-s[j]) for i in range(len(b)) for j in range(len(b)) if i!=j),default=0.))

def model_bases(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(20261116);torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id).eval()
    for question,options,gold in QA:
        prompt='Answer with a single letter.\\nQuestion: '+question+'\\n'+'\\n'.join(f'{chr(65+i)}. {answer}' for i,answer in enumerate(options))+'\\nAnswer:'
        text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        ids=tok(text,add_special_tokens=False,return_tensors='pt')['input_ids']
        with torch.inference_mode(): logits=model(ids).logits[0,-1].float()
        logits=torch.log_softmax(logits,dim=-1)
        lps=[]
        for letter in 'ABCDEF':
            tail=tok(' '+letter,add_special_tokens=False)['input_ids']
            if len(tail)==1:val=logits[tail[0]]
            else:
                seq=torch.cat([ids,torch.tensor([tail],dtype=ids.dtype)],dim=-1)
                with torch.inference_mode():l=torch.log_softmax(model(seq).logits[0,ids.shape[1]-1:-1].float(),dim=-1)
                val=l[torch.arange(len(tail)),torch.tensor(tail)].sum()
            lps.append(float(val.item()))
        ls=np.asarray(lps)
        # Tempered and smoothed restriction to the 6 candidate completions.
        v=np.exp((ls-ls.max())/1.5);v/=v.sum()
        b=.95*v+.05/len(v)
        yield question,options,gold,b,ls

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--out',default='outputs/real_llm.json')
    pa.add_argument('--model',default=MODEL)
    pa.add_argument('--n',type=int,default=5000)
    args=pa.parse_args()
    results=[]
    for i,(question,options,gold,b,ls) in enumerate(model_bases(args.model)):
        W=W_oracle(gold)
        L,U=interval(np.random.default_rng(99119+i),b,W,n=args.n)
        q=optimize_certificate(b,L,U)
        s=W@b
        results.append({
            'question':question,'options':options,'correct_index':int(gold),
            'model_logprobabilities':ls.tolist(),'reference_policy':b.tolist(),
            'policy':q.tolist(),'kl':kl(q,b),
            'cert_lower':lower_gain(q,b,L,U),'true_gain_programmatic':float(q@s),
            'best_single_transfer_oracle_gain':oracle_pair_best(b,s,.15),
            'simultaneous_interval_covered':bool(np.all(L<=s)&np.all(s<=U)),
            'queries_programmatic':6*args.n})
    output={'model_id':args.model,'evaluation':'real_weights_programmatic_preference_oracle_NOT_human_preference',
        'n_questions':len(results),'queries_per_prompt':6*args.n,
        'certified_above_0_01':sum(r['cert_lower']>.01 for r in results),
        'single_transfer_oracle_exceeds_0_01':sum(r['best_single_transfer_oracle_gain']>.01 for r in results),
        'all_interval_covered':sum(r['simultaneous_interval_covered'] for r in results),
        'rows':results}
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
