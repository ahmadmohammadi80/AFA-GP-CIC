"use strict";
// Nested simultaneous confidence boxes for changing KL-feasible policies.
// Run: node nested_policy_certificate_20261118.js
function intersect(a,b){if(a.length!==b.length)throw Error("dimension");return a.map(([l,u],i)=>{let x=Math.max(l,b[i][0]),y=Math.min(u,b[i][1]);if(x>y)throw Error("empty confidence set: abstain");return [x,y];});}
function lower(w,b){return w.reduce((s,v,i)=>s+v*(b[i][v>=0?0:1]-.5),0);}
function weights(q,r,eps){if(q.length!==r.length||![q,r].every(a=>a.every(x=>Number.isFinite(x)&&x>=0)&&Math.abs(a.reduce((s,x)=>s+x,0)-1)<1e-10))throw Error("policy");let kl=q.reduce((s,x,i)=>s+(x===0?0:r[i]===0?Infinity:x*Math.log(x/r[i])),0);if(!(eps>=0&&kl<=eps+1e-12))throw Error("KL");let w=[];for(let i=0;i<q.length;i++)for(let j=i+1;j<q.length;j++)w.push(q[i]*r[j]-q[j]*r[i]);return w;}
function box(s,n,a){return n.map((v,i)=>{if(!v)return [0,1];let d=Math.sqrt(Math.log(2*n.length*v*(v+1)/a)/(2*v));return [Math.max(0,s[i]/v-d),Math.min(1,s[i]/v+d)];});}
function score(qs,r,eps,s,n,a,history){let b=intersect(history,box(s,n,a)),ws=qs.map(q=>weights(q,r,eps));let value=c=>Math.max(0,...ws.map(w=>lower(w,c)));let base=value(b);return n.map((_,e)=>{let z=[0,1].map(y=>{let ss=s.slice(),nn=n.slice();ss[e]+=y;nn[e]++;try{return value(intersect(b,box(ss,nn,a)));}catch(err){if(err.message.startsWith("empty confidence"))return 0;throw err;}});let [l,u]=b[e];let worst=(x,y)=>Math.min((1-l)*x+l*y,(1-u)*x+u*y);return {e,Q:worst(+(z[0]>0),+(z[1]>0)),G:worst(z[0]-base,z[1]-base)};});}
function tests(){let n=0;const check=x=>{if(!x)throw Error("test "+n);n++;};let a=[[.2,.8],[.4,.9]],b=[[.3,.9],[.1,.7]],c=intersect(a,b);
for(let i=-10;i<=10;i++)for(let j=-10;j<=10;j++){let w=[i/10,j/10];check(lower(w,c)>=Math.max(lower(w,a),lower(w,b))-1e-12);}
let r=[.5,.5],q=[.8,.2],v=[.2,.8];check(lower(weights(q,r,1),[[.7,.9]])>0);check(lower(weights(v,r,1),[[.7,.9]])<0);
let bad=false;try{intersect([[0,.1]],[[.2,1]]);}catch(e){bad=true;}check(bad);
bad=false;try{weights(q,r,.001);}catch(e){bad=true;}check(bad);
let z=score([q,v],r,1,[90],[100],.05,[[0,1]]);check(z.length===1&&z[0].Q>=0&&z[0].Q<=1);
return n;}
if(typeof require!=="undefined"&&require.main===module)console.log(JSON.stringify({passed:tests()}));
