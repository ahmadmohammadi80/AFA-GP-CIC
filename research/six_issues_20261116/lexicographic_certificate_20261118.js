"use strict";
// Certificate-preserving lexicographic acquisition. Fixed linear gain.
// Run with node. Uses the running maximum of valid lower bounds.
function box(s,n,a=.05){
 if(!n.length||s.length!==n.length||!(a>0&&a<1))throw Error("input");
 return n.map((v,i)=>{if(!Number.isInteger(v)||!Number.isInteger(s[i])||s[i]<0||s[i]>v)throw Error("counts");
 if(!v)return [0,1];let r=Math.sqrt(Math.log(2*n.length*v*(v+1)/a)/(2*v));return [Math.max(0,s[i]/v-r),Math.min(1,s[i]/v+r)];});
}
function lower(w,c,b){return c+w.reduce((a,v,i)=>a+v*b[i][v>=0?0:1],0);}
function worst(a,b,l,u){return Math.min((1-l)*a+l*b,(1-u)*a+u*b);}
function choose(w,c,s,n,a=.05,previous=-Infinity){
 if(w.length!==n.length||![...w,c].every(Number.isFinite)||Number.isNaN(previous)||previous===Infinity)throw Error("weights");
 const b=box(s,n,a),base=Math.max(previous,lower(w,c,b));
 const scores=n.map((_,e)=>{
 const z=[0,1].map(y=>{let ss=s.slice(),nn=n.slice();ss[e]+=y;nn[e]++;return Math.max(base,lower(w,c,box(ss,nn,a)));});
 return {e,Q:worst(+(z[0]>0),+(z[1]>0),...b[e]),G:worst(z[0]-base,z[1]-base,...b[e]),z};
 });
 return {best:scores.reduce((x,y)=>y.Q>x.Q||(y.Q===x.Q&&y.G>x.G)?y:x),scores,base};
}
function tests(){
 const check=v=>{if(!v)throw Error("test failed");};let count=0;
 let x=choose([0,1],-.9,[0,90],[0,100]);
 check(x.scores.every(a=>a.Q===0)&&x.best.e===1&&x.best.G>0);count++;
 for(let a of [-.2,0,.3])for(let b of [-.4,.1,.5]){
 let v=worst(a,b,.2,.8),grid=Infinity;
 for(let k=0;k<=1000;k++){let p=.2+.6*k/1000;grid=Math.min(grid,(1-p)*a+p*b);}
 check(Math.abs(v-grid)<1e-12);count++;}
 for(let i=0;i<=100;i+=5)for(let j=0;j<=100;j+=5){
 let a=choose([.4,-.6],.1,[i,j],[100,100]);let old=a.scores.reduce((x,y)=>y.Q>x.Q?y:x);
 check(a.best.Q===old.Q&&a.best.G>=old.G&&a.scores.every(s=>s.G>=0&&s.z.every(z=>z>=a.base)));count++;}
 let invalid=false;try{choose([1],0,[2],[1]);}catch(e){invalid=true;}check(invalid);count++;
 return {passed:count,example:x};
}
console.log(JSON.stringify(tests()));
