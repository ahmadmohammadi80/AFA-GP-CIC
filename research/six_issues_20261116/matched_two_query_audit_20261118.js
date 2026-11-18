"use strict";
// Equal two-query budget; fixed KL-feasible candidate q=(.6,.2,.2), uniform reference.
// Greedy first uses immediate certificate probability, then best robust continuation.
// Robust optimality over deterministic trees implies dominance in worst-case objective, not pointwise success.
function audit(){
 const alpha=.05, n=[100,100,100], w=[2/15,2/15,0];
 const interval=(s,n)=>{let d=Math.sqrt(Math.log(6*n*(n+1)/alpha)/(2*n));return [Math.max(0,s/n-d),Math.min(1,s/n+d)]};
 const low=(l,u,x,y)=>Math.min((1-l)*x+l*y,(1-u)*x+u*y);
 function run(s){let box=s.map((x,i)=>interval(x,n[i]));const gain=c=>w.reduce((v,x,i)=>v+x*(c[i][0]-.5),0);if(gain(box)>0)return null;
 function terminal(h){let c=box.map(x=>x.slice()),ss=s.slice(),nn=n.slice();for(const [e,y] of h){ss[e]+=y;nn[e]++;let d=interval(ss[e],nn[e]);c[e]=[Math.max(c[e][0],d[0]),Math.min(c[e][1],d[1])];if(c[e][0]>c[e][1])return 0;if(gain(c)>0)return 1;}return 0;}
 function min(e,a,b,f){let p=box.map(x=>x[0]),best=Infinity;for(let mask=0;mask<8;mask++){for(let i=0;i<3;i++)p[i]=box[i][(mask>>i)&1];let g=x=>{p[e]=x;let y=p[a],z=p[b];return (1-x)*((1-y)*f[0]+y*f[1])+x*((1-z)*f[2]+z*f[3])},c=g(0),v=g(1),h=g(.5),A=2*(v+c-2*h),B=v-c-A;let [l,u]=box[e];best=Math.min(best,g(l),g(u));if(A>0&&-B/(2*A)>l&&-B/(2*A)<u)best=Math.min(best,g(-B/(2*A)));}return best;}
 let trees=[];for(let e=0;e<3;e++)for(let a=0;a<3;a++)for(let b=0;b<3;b++){let f=[0,1,2,3].map(j=>terminal([[e,j>>1],[j>>1?b:a,j%2]]));let exact=min(e,a,b,f),rect=low(...box[e],low(...box[a],f[0],f[1]),low(...box[b],f[2],f[3]));if(rect>exact+1e-12)throw Error("bound");trees.push({e,a,b,exact,rect});}
 let best=trees.reduce((a,b)=>b.exact>a.exact?b:a),rb=trees.reduce((a,b)=>b.rect>a.rect?b:a);let one=Array.from({length:3},(_,e)=>low(...box[e],terminal([[e,0]]),terminal([[e,1]])));let g=one.indexOf(Math.max(...one)),greedy=trees.filter(t=>t.e===g).reduce((a,b)=>b.exact>a.exact?b:a);if(best.exact+1e-12<rb.exact||best.exact+1e-12<greedy.exact)throw Error("dominance");return {s,best,rb,greedy};}
 let states=0,positive=0,strictGreedy=0,strictRect=0,example=null;for(let a=40;a<=100;a++)for(let b=40;b<=100;b++){let x=run([a,b,50]);if(!x)continue;states++;if(x.best.exact>0)positive++;if(x.best.exact>x.greedy.exact+1e-12){strictGreedy++;example??=x;}if(x.best.exact>x.rb.exact+1e-12)strictRect++;}return {states,positive,strictGreedy,strictRect,example};
}
console.log(JSON.stringify(audit()));
