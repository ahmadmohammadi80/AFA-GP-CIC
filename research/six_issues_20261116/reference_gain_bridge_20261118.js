// Exact bridge from pairwise preferences to reference-relative policy gain.
// Run with node. No claim of literature novelty.
function bridge(q,r,eps){
 if(q.length!==r.length||!q.length||![...q,...r].every(x=>Number.isFinite(x)&&x>=0)||!Number.isFinite(eps)||eps<0||[q,r].some(v=>Math.abs(v.reduce((a,b)=>a+b,0)-1)>1e-10))throw Error("invalid policy");
 let kl=0,offset=0,edges=[];
 for(let i=0;i<q.length;i++){
 if(q[i]>0){if(r[i]===0)throw Error("support");kl+=q[i]*Math.log(q[i]/r[i]);}
 offset+=q[i]*r[i]/2;
 for(let j=i+1;j<q.length;j++){offset+=q[j]*r[i];edges.push([i,j,q[i]*r[j]-q[j]*r[i]]);}
 }
 if(kl>eps+1e-12)throw Error("KL");
 return {offset:offset-.5,edges,kl};
}
function lower(b,box){if(box.length!==b.edges.length||box.some(([l,u])=>!(0<=l&&l<=u&&u<=1)))throw Error("box");return b.offset+b.edges.reduce((s,[i,j,w],k)=>s+w*box[k][w>=0?0:1],0);}
function tests(){
 const assert=(x)=>{if(!x)throw Error("test failed")},near=(a,b)=>assert(Math.abs(a-b)<1e-10);
 let n=0;
 for(const q of [[.2,.3,.5],[.5,.4,.1],[0,1,0]])for(const r of [[.3,.3,.4],[.2,.6,.2]]){
 const b=bridge(q,r,3),p=[.1,.7,.9],M=[[.5,p[0],p[1]],[1-p[0],.5,p[2]],[1-p[1],1-p[2],.5]];
 let direct=-.5;for(let i=0;i<3;i++)for(let j=0;j<3;j++)direct+=q[i]*r[j]*M[i][j];
 near(lower(b,p.map(x=>[x,x])),direct);n++;
 const box=[[.1,.4],[.2,.8],[.3,.9]];
 let min=Infinity;for(let mask=0;mask<8;mask++)min=Math.min(min,lower(b,box.map((v,k)=>{let x=v[(mask>>k)&1];return [x,x]})));
 near(lower(b,box),min);n++;
 }
 near(lower(bridge([.2,.8],[.2,.8],0),[[0,1]]),0);n++;
 for(const args of [[[1,0],[.5,.5],0],[[1,0],[0,1],10]]){let rejected=false;try{bridge(...args)}catch(e){rejected=true}assert(rejected);n++;}
 return n;
}
console.log("passed",tests());