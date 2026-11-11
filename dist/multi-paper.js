/* Calibrate one lens from independently posed metric paper observations. */
(function(root){
  function prepare(cv,groups){
    return groups.filter(g=>g.markers.length>=2).map(group=>{
      const points=group.markers.flatMap(m=>A4Core.tagUV(m.id).map((uv,k)=>({id:m.id,corner:k,uv,xy:m.corners[k]})));
      const a=cv.matFromArray(points.length,1,cv.CV_32FC2,points.flatMap(p=>p.uv)),b=cv.matFromArray(points.length,1,cv.CV_32FC2,points.flatMap(p=>p.xy)),mask=new cv.Mat();let H;
      try{
        H=cv.findHomography(a,b,cv.RANSAC,2,mask);
        if(H.empty())return {group,points:[],rejected:points.length};
        const kept=points.filter((_,i)=>mask.data[i]);
        return {group,points:kept,rejected:points.length-kept.length};
      }finally{H?.delete();a.delete();b.delete();mask.delete();}
    });
  }
  function fit(cv, groups, width, height, options={}) {
    // Native calibrateCamera accepts a partially visible board. Two decoded
    // squares still provide eight metric corners; do not discard a side wall
    // merely because fewer than half of its tags are inside the image.
    const prepared=prepare(cv,groups),usable=prepared.filter(g=>g.points.length>=8&&new Set(g.points.map(p=>p.id)).size>=2);
    if(usable.length<2)return {ok:false,reason:'複数の紙の観測が不足しています',views:usable.length};
    const run=(seed,omit=-1)=>{
      const resources=[],own=x=>(resources.push(x),x),objects=own(new cv.MatVector()),images=own(new cv.MatVector());
      const K=own(cv.matFromArray(3,3,cv.CV_64F,[seed,0,width/2,0,seed,height/2,0,0,1])),D=own(cv.Mat.zeros(5,1,cv.CV_64F));
      const rotations=own(new cv.MatVector()),translations=own(new cv.MatVector()),si=own(new cv.Mat()),se=own(new cv.Mat()),pe=own(new cv.Mat());
      try {
        for(const group of usable){
          const points=group.points.filter(p=>p.id!==omit);if(points.length<8||new Set(points.map(p=>p.id)).size<2)continue;
          const object=cv.matFromArray(points.length,1,cv.CV_32FC3,points.flatMap(p=>[(p.uv[0]-148.5)*(options.ruler||100)/100,(105-p.uv[1])*(options.ruler||100)/100,0]));
          const image=cv.matFromArray(points.length,1,cv.CV_32FC2,points.flatMap(p=>p.xy));
          objects.push_back(object);images.push_back(image);object.delete();image.delete();
        }
        const flags=cv.CALIB_USE_INTRINSIC_GUESS|cv.CALIB_FIX_ASPECT_RATIO|(options.principal?0:cv.CALIB_FIX_PRINCIPAL_POINT)|cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K2|cv.CALIB_FIX_K3|(options.radial?0:cv.CALIB_FIX_K1)|(options.solver==='svd'?0:cv.CALIB_USE_LU);
        const rms=cv.calibrateCameraExtended(objects,images,new cv.Size(width,height),K,D,rotations,translations,si,se,pe,flags,new cv.TermCriteria(cv.TermCriteria_COUNT+cv.TermCriteria_EPS,100,1e-9));
        const k=Array.from(K.data64F),d=Array.from(D.data64F),standardDeviation=Array.from(si.data64F),f=k[0],fov=2*Math.atan(width/(2*f))*180/Math.PI;
        const radii=[[0,0],[width,0],[0,height],[width,height]].map(([x,y])=>((x-k[2])/f)**2+((y-k[5])/f)**2);
        const radialDerivative=1+3*d[0]*Math.max(...radii),sigmaF=Math.max(standardDeviation[0]||0,standardDeviation[1]||0);
        return {K:k,D:d,rms,fov,sigmaF,relativeSigmaF:sigmaF/f,standardDeviation,viewErrors:Array.from(pe.data64F),radialDerivative,seed,solver:options.solver==='svd'?'svd':'lu',rejectedCorners:prepared.reduce((s,g)=>s+g.rejected,0),usedCorners:usable.reduce((s,g)=>s+g.points.length,0)};
      }finally{resources.reverse().forEach(x=>x.delete());}
    };
    const attempts=[];for(const ratio of [.5,.85,1.3]){try{attempts.push(run(Math.max(width,height)*ratio));}catch(e){attempts.push({error:e.message||String(e)})}}
    const valid=attempts.filter(r=>r.K?.every(Number.isFinite)&&r.D.every(Number.isFinite)&&r.K[2]>0&&r.K[2]<width&&r.K[5]>0&&r.K[5]<height&&r.fov>15&&r.fov<130&&r.radialDerivative>.1&&Number.isFinite(r.rms));
    const best=valid.sort((a,b)=>a.rms-b.rms)[0];
    if(!best)return {ok:false,reason:'物理的なレンズ解が得られません',attempts};
    const near=valid.filter(x=>x.rms<=best.rms+.05),seedSpread=(Math.max(...near.map(r=>r.K[0]))-Math.min(...near.map(r=>r.K[0])))/best.K[0];
    if(best.rms>2.5||best.relativeSigmaF>.15||seedSpread>.15)return {ok:false,reason:'レンズ値の誤差または不確かさが大きい',...best,seedSpread,attempts};
    const leaveOne=[];for(const id of new Set(usable.flatMap(g=>g.points.map(p=>p.id)))){try{const r=run(best.K[0],id);leaveOne.push({omitted:id,f:r.K[0],rms:r.rms});}catch(e){leaveOne.push({omitted:id,error:e.message||String(e)})}}
    const values=leaveOne.map(x=>x.f).filter(Number.isFinite),spread=values.length?(Math.max(...values)-Math.min(...values))/best.K[0]:Infinity;
    if(spread>.16||values.length<4)return {ok:false,reason:'一つのタグを除くと画角が不安定になります',...best,spread,seedSpread,leaveOne,attempts};
    return {ok:true,...best,width,height,spread,seedSpread,leaveOne,attempts,views:usable.length,source:'a4-multi-paper',model:options.radial?'opencv-pinhole-brown5':'opencv-pinhole',assumptions:'Each paper has its own planar pose; square pixels; principal point at image center; '+(options.radial?'radial k1 estimated; higher radial and tangential distortion fixed to zero':'distortion fixed to zero, matching the original automatic lens model')};
  }
  root.MultiPaper={fit,prepare};
})(self);
