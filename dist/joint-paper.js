/* Joint planar fit of metric paper patterns. Sheet translations and rotations are estimated. */
(function(root){
 const solve=(A,b)=>{const n=b.length,a=A.map((row,i)=>[...row,b[i]]);for(let k=0;k<n;k++){let p=k;for(let i=k+1;i<n;i++)if(Math.abs(a[i][k])>Math.abs(a[p][k]))p=i;if(Math.abs(a[p][k])<1e-16)return null;[a[p],a[k]]=[a[k],a[p]];const d=a[k][k];for(let j=k;j<=n;j++)a[k][j]/=d;for(let i=0;i<n;i++)if(i!==k){const f=a[i][k];for(let j=k;j<=n;j++)a[i][j]-=f*a[k][j];}}return a.map(r=>r[n]);};
 const inv=h=>{const [a,b,c,d,e,f,g,j,k]=h,D=a*(e*k-f*j)-b*(d*k-f*g)+c*(d*j-e*g);return [e*k-f*j,c*j-b*k,b*f-c*e,f*g-d*k,a*k-c*g,c*d-a*f,d*j-e*g,b*g-a*j,a*e-b*d].map(x=>x/D);};
 const project=(h,x,y)=>{const z=h[6]*x+h[7]*y+1;return [(h[0]*x+h[1]*y+h[2])/z,(h[3]*x+h[4]*y+h[5])/z];};
 function fit(cv,first,second,w,h,ruler=100){
  const S=Math.max(w,h),local=id=>A4Core.tagUV(id).map(([u,v])=>[(u-148.5)*ruler/100000,(105-v)*ruler/100000]);
  const rows=[first,second].flatMap((g,b)=>g.markers.flatMap((m,tag)=>m.corners.map((p,k)=>({b,tag,id:m.id,k,xy:local(m.id)[k],uv:[(p[0]-w/2)/S,(p[1]-h/2)/S]}))));
  const a=cv.matFromArray(first.markers.length*4,1,cv.CV_32FC2,rows.filter(r=>!r.b).flatMap(r=>r.xy)),b=cv.matFromArray(first.markers.length*4,1,cv.CV_32FC2,rows.filter(r=>!r.b).flatMap(r=>r.uv));let H;let q;
  try{H=cv.findHomography(a,b,0);if(H.empty())return null;q=Array.from(H.data64F);}finally{H?.delete();a.delete();b.delete();}
  const ih=inv(q),pairs=rows.filter(r=>r.b).map(r=>{const z=ih[6]*r.uv[0]+ih[7]*r.uv[1]+ih[8];return {s:r.xy,t:[(ih[0]*r.uv[0]+ih[1]*r.uv[1]+ih[2])/z,(ih[3]*r.uv[0]+ih[4]*r.uv[1]+ih[5])/z]};}),mean=(key,i)=>pairs.reduce((s,p)=>s+p[key][i],0)/pairs.length,sc=[mean('s',0),mean('s',1)],tc=[mean('t',0),mean('t',1)];let dot=0,cross=0;for(const p of pairs){const x=p.s[0]-sc[0],y=p.s[1]-sc[1],u=p.t[0]-tc[0],v=p.t[1]-tc[1];dot+=x*u+y*v;cross+=x*v-y*u;}const theta=Math.atan2(cross,dot),co=Math.cos(theta),si=Math.sin(theta);const start=[...q.slice(0,8),tc[0]-co*sc[0]+si*sc[1],tc[1]-si*sc[0]-co*sc[1],theta];
  const position=(p,r)=>r.b?[Math.cos(p[10])*r.xy[0]-Math.sin(p[10])*r.xy[1]+p[8],Math.sin(p[10])*r.xy[0]+Math.cos(p[10])*r.xy[1]+p[9]]:r.xy;
  const residual=(p,rs)=>rs.flatMap(r=>{const xy=position(p,r),v=project(p,...xy);return [v[0]-r.uv[0],v[1]-r.uv[1]];});
  const optimize=(initial,rs)=>{let p=initial.slice(),lambda=.001,err=residual(p,rs),cost=err.reduce((s,x)=>s+x*x,0);for(let it=0;it<70;it++){const J=[];for(let k=0;k<p.length;k++){const z=p.slice(),epsilon=1e-5*Math.max(1,Math.abs(p[k]));z[k]+=epsilon;const shifted=residual(z,rs);J.push(shifted.map((x,i)=>(x-err[i])/epsilon));}const A=J.map(u=>J.map(v=>u.reduce((s,x,i)=>s+x*v[i],0))),b=J.map(u=>-u.reduce((s,x,i)=>s+x*err[i],0));for(let k=0;k<p.length;k++)A[k][k]+=lambda*Math.max(A[k][k],1e-8);const d=solve(A,b);if(!d)break;const next=p.map((x,k)=>x+d[k]),e=residual(next,rs),c=e.reduce((s,x)=>s+x*x,0);if(Number.isFinite(c)&&c<cost){p=next;err=e;if(cost-c<1e-13){cost=c;break;}cost=c;lambda=Math.max(1e-9,lambda*.3);}else lambda=Math.min(1e10,lambda*10);}return {p,rms:Math.sqrt(cost/rs.length)*S};};
  const full=optimize(start,rows);if(full.rms>2.5||!full.p.every(Number.isFinite))return {ok:false,reason:'joint reprojection',rms:full.rms};
  const focal=p=>{const aa=p[0]*p[1]+p[3]*p[4],bb=p[6]*p[7],cc=p[0]**2+p[3]**2-p[1]**2-p[4]**2,dd=p[6]**2-p[7]**2,f2=-(aa*bb+cc*dd)/(bb*bb+dd*dd);if(!(f2>0))return null;const f=Math.sqrt(f2)*S,fov=2*Math.atan(w/(2*f))*180/Math.PI,n1=p[0]**2+p[3]**2+f2*p[6]**2,n2=p[1]**2+p[4]**2+f2*p[7]**2,residual=Math.hypot(aa+f2*bb,cc+f2*dd)/(n1+n2);return {f,fov,residual};};
  const base=focal(full.p);if(!base||base.fov<15||base.fov>130||base.residual>.025)return {ok:false,reason:'joint focal constraints',rms:full.rms,focal:base};
  const values=[];for(let bi=0;bi<2;bi++)for(let tag=0;tag<[first,second][bi].markers.length;tag++){const rs=rows.filter(r=>r.b!==bi||r.tag!==tag),estimate=focal(optimize(full.p,rs).p);if(estimate?.f>0)values.push(estimate.f);}
  const spread=values.length?(Math.max(...values)-Math.min(...values))/base.f:Infinity;if(values.length<6||spread>.16)return {ok:false,reason:'joint focal unstable',rms:full.rms,focal:base,spread};
  const f=base.f,K=[f,0,w/2,0,f,h/2,0,0,1],D=[0,0,0,0,0];
  const markers=[first,second].flatMap((g,bi)=>g.markers.map(m=>({...m,object:local(m.id).map(xy=>{const p=position(full.p,{b:bi,xy});return [p[0]*1000,p[1]*1000,0];})})));
  return {ok:true,K,D,width:w,height:h,fov:base.fov,spread,rms:full.rms,source:'a4-joint-plane',markers,secondPaperTransform:{x:full.p[8],y:full.p[9],rotation:full.p[10]},assumptions:'Selected sheets are coplanar; metric printed patterns; independent in-plane sheet translation and rotation; centered principal point; square pixels; zero distortion'};
 }
 root.JointPaper={fit};
})(self);
