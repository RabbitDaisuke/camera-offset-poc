(function(root){
 const mm=(a,b)=>Array.from({length:9},(_,i)=>[0,1,2].reduce((s,k)=>s+a[Math.floor(i/3)*3+k]*b[k*3+i%3],0));
 const mv=A4Core.mul,transpose=A4Core.transpose;
 const stats=values=>{const v=values.filter(Number.isFinite).slice().sort((a,b)=>a-b);return v.length?{count:v.length,rms:Math.sqrt(v.reduce((s,x)=>s+x*x,0)/v.length),median:v[Math.floor(v.length/2)],p95:v[Math.min(v.length-1,Math.floor(v.length*.95))],max:v.at(-1)}:null};
 async function run(cv,engine,images,calibration,pairs,rect,shifts=[]){
  const native=await StereoCalibration.module();
  const cameras=calibration.cameras;if(!cameras||!calibration.poses)throw Error('同時校正の結果が必要です');
  const detections=images.map(im=>engine.detect(im,{},true)),groups=detections.map(d=>engine.boards(d.markers));
  const R=mm(cameras[1].R,transpose(cameras[0].R)),T=cameras[1].t.map((v,i)=>v-mv(R,cameras[0].t)[i]);
  const E=mm([0,-T[2],T[1],T[2],0,-T[0],-T[1],T[0],0],R),Et=transpose(E);
  const results=pairs.map(([a,b],view)=>{
   const paperPose=calibration.poses[view],worldR=mm(transpose(cameras[0].R),paperPose.R),worldT=mv(transpose(cameras[0].R),paperPose.t.map((v,i)=>v-cameras[0].t[i]));
   const worldOf=uv=>mv(worldR,[(uv[0]-148.5)/1000,(105-uv[1])/1000,0]).map((v,k)=>v+worldT[k]);
   // Detection order changes when a paper becomes occluded. Match the same
   // physical paper by its stored pose, never by the current array index.
   const choose=ci=>groups[ci].map((g,index)=>{const distances=g.markers.flatMap(m=>A4Core.tagUV(m.id).map((uv,j)=>{const p=A4Core.project(worldOf(uv),cameras[ci],cameras[ci].K,cameras[ci].D);return p?Math.hypot(p[0]-m.corners[j][0],p[1]-m.corners[j][1]):Infinity;}));return {g,index,distance:stats(distances)?.median??Infinity};}).sort((a,b)=>a.distance-b.distance)[0];
   const selected=[choose(0),choose(1)];if(selected.some(s=>!s||s.distance>24))return {pair:[a,b],error:'同じ実物の紙を再観測できません',nearest:selected.map(s=>s&&({index:s.index,distance:s.distance}))};
   const [ga,gb]=selected.map(s=>s.g);
   const right=new Map(gb.markers.map(m=>[m.id,m])),matched=ga.markers.filter(m=>right.has(m.id));
   const p0=matched.flatMap(m=>m.corners),p1=matched.flatMap(m=>right.get(m.id).corners);
   if(!p0.length)return {pair:[a,b],error:'共通タグなし'};
   const raw=[p0,p1].map((ps,j)=>native.undistortPixels(ps,cameras[j])),corrected=[p0,p1].map((ps,j)=>ps.map(p=>p.map((v,k)=>v-(shifts[j]?.originalShift?.[k]||0)))),n0=native.undistortPixels(corrected[0],cameras[0]),n1=native.undistortPixels(corrected[1],cameras[1]);
   const points=matched.flatMap(m=>A4Core.tagUV(m.id).map(([u,v],corner)=>({id:m.id,corner,object:[(u-148.5)/1000,(105-v)/1000,0]})));
   const rows=points.map((p,i)=>{
    const X=mv(worldR,p.object).map((v,k)=>v+worldT[k]),predicted=cameras.map(c=>A4Core.project(X,c,c.K,c.D));
    const l1=mv(E,n0[i]),l0=mv(Et,n1[i]),s=n1[i].reduce((s,v,k)=>s+v*l1[k],0);
    const epi=[s/Math.hypot(l0[0],l0[1])*cameras[0].K[0],s/Math.hypot(l1[0],l1[1])*cameras[1].K[0]];
    const normalized=[n0[i],n1[i]],worldRays=normalized.map((n,j)=>mv(transpose(cameras[j].R),n)),rectified=rect?worldRays.map(ray=>{const q=mv(rect.R,ray);return [rect.f*q[0]/q[2]+rect.cx,rect.f*q[1]/q[2]+rect.cy];}):null;
    const rays=worldRays.map(r=>r.map(v=>v/Math.hypot(...r))),dot=(a,b)=>a.reduce((s,v,k)=>s+v*b[k],0),w=cameras[1].center.map((v,k)=>v-cameras[0].center[k]),c=dot(rays[0],rays[1]),wa=dot(w,rays[0]),wb=dot(w,rays[1]),alongLeft=(wa-c*wb)/(1-c*c),alongRight=(c*wa-wb)/(1-c*c),left=cameras[0].center.map((v,k)=>v+alongLeft*rays[0][k]),right=cameras[1].center.map((v,k)=>v+alongRight*rays[1][k]),triangulated=left.map((v,k)=>(v+right[k])/2);
    const rawLine=mv(E,raw[0][i]),rawEpi=dot(raw[1][i],rawLine)/Math.hypot(rawLine[0],rawLine[1])*cameras[1].K[0];
    return {...p,observed:[p0[i],p1[i]],predicted,epipolar:epi,rawEpipolar:rawEpi,rectified,verticalResidual:rectified?rectified[0][1]-rectified[1][1]:null,triangulated,rayGap:Math.hypot(...left.map((v,k)=>v-right[k])),reprojection:predicted.map((q,j)=>q?Math.hypot(q[0]-[p0,p1][j][i][0],q[1]-[p0,p1][j][i][1]):Infinity),depth:cameras.map(c=>mv(c.R,X)[2]+c.t[2])};
   });
   const edgeErrors=[];for(let i=0;i<rows.length;i+=4)for(let j=0;j<4;j++){const p=rows[i+j],q=rows[i+(j+1)%4],expected=Math.hypot(...p.object.map((v,k)=>v-q.object[k])),measured=Math.hypot(...p.triangulated.map((v,k)=>v-q.triangulated[k]));edgeErrors.push((measured-expected)*1000);}
   return {pair:[a,b],observedGroups:selected.map(s=>s.index),ids:matched.map(m=>m.id),bounds:[ga.bounds,gb.bounds],epipolar:stats(rows.flatMap(r=>r.epipolar.map(Math.abs))),rawEpipolar:stats(rows.map(r=>Math.abs(r.rawEpipolar))),rectifiedVertical:stats(rows.map(r=>Math.abs(r.verticalResidual))),tagEdgeErrorMm:stats(edgeErrors.map(Math.abs)),reprojection:stats(rows.flatMap(r=>r.reprojection)),positiveDepth:rows.every(r=>r.depth.every(z=>z>0)),rows};
  });
  return {method:'OpenCV undistortPoints; E=[T]xR from calibrated cameras; same frame translation as scan',units:'original-image pixels; rectifiedVertical in scan pixels; edge errors in mm',shifts,tags:detections.map(d=>d.markers.length),groups:groups.map(gs=>gs.map((g,i)=>({i,ids:g.markers.map(m=>m.id),bounds:g.bounds}))),baseline:Math.hypot(...T),results};
 }
 root.GeometryAudit={run};
})(self);
