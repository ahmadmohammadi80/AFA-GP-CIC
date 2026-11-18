"use strict";
// Run: node rectangular_two_step_audit_20261118.js
// Diagnostic only: terminal predicate must itself be a valid certificate.
console.log(JSON.stringify((function twoStepAudit(){
"use strict";
// Fixed unknown Bernoulli parameters; rectangular adversary may vary by history.
// Lower bound for static robust certificate probability; exact for rectangular adversary.
function worst(l,u,a,b){return Math.min((1-l)*a+l*b,(1-u)*a+u*b);}
function plan(box,terminal,h=[],depth=2){
 if(terminal(h))return {value:1,edge:null};
 if(!depth)return {value:0,edge:null};
 let best={value:-1,edge:null};
 for(let e=0;e<box.length;e++){
  const a=plan(box,terminal,h.concat([[e,0]]),depth-1).value;
  const b=plan(box,terminal,h.concat([[e,1]]),depth-1).value;
  const value=worst(...box[e],a,b);
  if(value>best.value)best={value,edge:e};
 }
 return best;
}
function tests(){
 let checks=0;function check(x){if(!x)throw Error("test "+checks);checks++;}
 // Terminal threshold: two positive observations on informative edge 1.
 const box=[[.2,.8],[.6,.9]], terminal=h=>h.filter(([e,y])=>e===1&&y===1).length>=2;
 const one=plan(box,terminal,[],1),two=plan(box,terminal);
 check(one.value===0&&one.edge===0);check(two.edge===1&&Math.abs(two.value-.36)<1e-12);
 // Repeated edge equality: rectangular relaxation is strictly conservative.
 const eq=h=>h.length===2&&h[0][1]===h[1][1];
 const v=plan([[.2,.8]],eq).value;
 check(Math.abs(v-.2)<1e-12);
 for(let i=0;i<=100;i++){let p=.2+.6*i/100;check(v<=p*p+(1-p)*(1-p)+1e-12);}
 // All 16 terminal events on a single edge, grid of static probabilities.
 for(let mask=0;mask<16;mask++){
  const event=h=>h.length===2&&Boolean(mask&(1<<(2*h[0][1]+h[1][1])));
  const lower=plan([[.2,.8]],event).value;
  for(let i=0;i<=100;i++){let p=.2+.6*i/100,s=0;for(let a=0;a<2;a++)for(let b=0;b<2;b++)if(event([[0,a],[0,b]]))s+=(a?p:1-p)*(b?p:1-p);check(lower<=s+1e-12);}
 }
 return {checks,one,two,equalityRectangular:v,equalityStaticMinimum:.5};
}
return tests();
})()));
