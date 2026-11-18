"use strict";
// Exact static two-query planning, deterministic trees, independent conditional Bernoulli draws.
// Run: node static_two_step_certificate_20261118.js
function value(p,e,a,b,flags){
 const x=p[e], y=p[a], z=p[b];
 return (1-x)*((1-y)*flags[0]+y*flags[1])+x*((1-z)*flags[2]+z*flags[3]);
}
function minimum(box,e,a,b,flags){
 let p=box.map(x=>x[0]), best=Infinity;
 const others=[...new Set([a,b])].filter(i=>i!==e);
 function visit(k){
  if(k<others.length){let j=others[k];for(let x of box[j]){p[j]=x;visit(k+1);}return;}
  const f=x=>{p[e]=x;return value(p,e,a,b,flags);};
  const c=f(0), v1=f(1), vh=f(.5), A=2*(v1+c-2*vh), B=v1-c-A;
  const [l,u]=box[e];best=Math.min(best,f(l),f(u));
  if(A>0){const x=-B/(2*A);if(x>l&&x<u)best=Math.min(best,f(x));}
 }
 visit(0);return best;
}
function plan(box,terminal){
 let best={value:-1};
 for(let e=0;e<box.length;e++)for(let a=0;a<box.length;a++)for(let b=0;b<box.length;b++){
  let flags=[0,1,2,3].map(k=>+terminal([[e,k>>1],[k>>1?b:a,k%2]]));
  let v=minimum(box,e,a,b,flags);
  if(v>best.value)best={value:v,e,a,b};
 }return best;
}
// Terminal is evaluated from observed history only; nested Hoeffding boxes and fixed KL-feasible candidates.
function certificate(box,s,n,weights,alpha,gamma=0){
 return h=>{
  let ss=s.slice(),nn=n.slice(),c=box.map(x=>x.slice());
  for(let [e,y] of h){ss[e]+=y;nn[e]++;
   let d=Math.sqrt(Math.log(2*n.length*nn[e]*(nn[e]+1)/alpha)/(2*nn[e]));
   c[e]=[Math.max(c[e][0],ss[e]/nn[e]-d),Math.min(c[e][1],ss[e]/nn[e]+d)];
   if(c[e][0]>c[e][1])return false;
   if(weights.some(w=>w.reduce((v,x,i)=>v+x*(c[i][x>=0?0:1]-.5),0)>gamma))return true;
  }return false;
 };
}
function tests(){
 let checks=0;const check=x=>{if(!x)throw Error("check "+checks);checks++;};
 let box=[[.2,.8],[.3,.9]];
 for(let e=0;e<2;e++)for(let a=0;a<2;a++)for(let b=0;b<2;b++)for(let mask=0;mask<16;mask++){
  let f=[0,1,2,3].map(k=>(mask>>k)&1), m=minimum(box,e,a,b,f),grid=Infinity;
  for(let i=0;i<=60;i++)for(let j=0;j<=60;j++)grid=Math.min(grid,value([.2+.01*i,.3+.01*j],e,a,b,f));
  check(m<=grid+1e-12&&grid-m<.00011);
 }
 check(Math.abs(minimum(box,0,0,0,[1,0,0,1])-.5)<1e-12);
 // Real certificate, 3 responses, uniform reference; q=(.6,.2,.2), KL < .2.
 let r=[1/3,1/3,1/3],q=[.6,.2,.2],w=[];
 for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)w.push(q[i]*r[j]-q[j]*r[i]);
 check(q.reduce((v,x,i)=>v+x*Math.log(x/r[i]),0)<.2);
 let c=[[.4,.9],[.5,.5],[.2,.8]],n=[100,100,100], hit=null;
 for(let k=50;k<=100;k++){
  let s=[k,50,50],t=certificate(c,s,n,[w],.05),p=plan(c,t);
  if(p.value>0&&p.value<1){hit={s,p};break;}
 }
 check(hit!==null);
 return {checks,hit};
}
if(typeof module!=="undefined"&&typeof require!=="undefined"&&require.main===module)console.log(JSON.stringify(tests()));
