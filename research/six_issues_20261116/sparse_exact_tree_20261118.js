"use strict";
// Exact two-query static robust value: only queried edges enter the leaf probability.
// Fixed tree and fixed history imply fixed terminal certificate indicators.
// Enumerate <=4 corners for the distinct non-root queried edges, then minimize root quadratic.
// O(1) probability evaluations per tree instead of O(2^m); full search O(m^3), excluding certificate evaluation.
function minimum(box,e,a,b,f,sparse=true){
 const ids=sparse?[...new Set([a,b])].filter(i=>i!==e):box.map((_,i)=>i).filter(i=>i!==e);
 let best=Infinity;
 for(let mask=0;mask<2**ids.length;mask++){
 const p=box.map(z=>z[0]);ids.forEach((i,j)=>p[i]=box[i][(mask>>j)&1]);
 const g=x=>{p[e]=x;return (1-x)*((1-p[a])*f[0]+p[a]*f[1])+x*((1-p[b])*f[2]+p[b]*f[3]);};
 const c=g(0),v=g(1),h=g(.5),A=2*(v+c-2*h),B=v-c-A,[l,u]=box[e];
 best=Math.min(best,g(l),g(u));if(A>0){const t=-B/(2*A);if(t>l&&t<u)best=Math.min(best,g(t));}
 }return best;
}
function test(){
 let count=0;const box=Array.from({length:7},(_,i)=>[.03+i*.06,.63+i*.05]);
 for(let e=0;e<7;e++)for(let a=0;a<7;a++)for(let b=0;b<7;b++)for(let k=0;k<16;k++){
 const f=Array.from({length:4},(_,j)=>(k>>j)&1);if(Math.abs(minimum(box,e,a,b,f)-minimum(box,e,a,b,f,false))>1e-12)throw Error("mismatch");count++;
 }return count;
}
console.log(JSON.stringify({checks:test()}));
