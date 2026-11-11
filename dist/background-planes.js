'use strict';
// Background geometry only. This never adds vertices to the current scan.
// In rectified stereo a 3D plane has affine disparity d = ax + by + c.
self.BackgroundPlanes=(()=>{
 const turn=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
 function hull(points){const p=points.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]),lo=[],hi=[];for(const q of p){while(lo.length>=2&&turn(lo.at(-2),lo.at(-1),q)<=0)lo.pop();lo.push(q);}for(const q of p.slice().reverse()){while(hi.length>=2&&turn(hi.at(-2),hi.at(-1),q)<=0)hi.pop();hi.push(q);}lo.pop();hi.pop();return lo.concat(hi);}
 function fit(samples){const n=samples.length,mean=[0,0,0];for(const p of samples)for(let k=0;k<3;k++)mean[k]+=p[k]/n;let xx=0,xy=0,yy=0,xd=0,yd=0;for(const p of samples){const x=p[0]-mean[0],y=p[1]-mean[1],d=p[2]-mean[2];xx+=x*x;xy+=x*y;yy+=y*y;xd+=x*d;yd+=y*d;}const det=xx*yy-xy*xy;if(det<1e-6)return null;const a=(xd*yy-yd*xy)/det,b=(yd*xx-xd*xy)/det;return [a,b,mean[2]-a*mean[0]-b*mean[1]];}
 const value=(p,x,y)=>p[0]*x+p[1]*y+p[2];
 const inside=(h,x,y)=>h.length>=3&&h.every((a,i)=>turn(a,h[(i+1)%h.length],[x,y])>=0);
 const area=h=>Math.abs(h.reduce((s,a,i)=>{const b=h[(i+1)%h.length];return s+a[0]*b[1]-a[1]*b[0];},0))/2;
 function build(cv,reference,S,options={}){
  const started=performance.now(),samples=[],sampleStep=options.sampleStep??4,minSupport=options.minSupport??40,maxModels=options.maxModels??4,minArea=options.minArea??512,phase=Math.floor(sampleStep/2);
  for(let i=0;i<reference.uv.length;i+=4){const x=reference.uv[i],y=reference.uv[i+1],d=reference.validDisparity[y*S+x];if(x%sampleStep===phase&&y%sampleStep===phase&&Number.isFinite(d))samples.push([x,y,d]);}
  let remaining=samples;const models=[];
  for(let pass=0;pass<maxModels&&remaining.length>=minSupport;pass++){
   const src=cv.matFromArray(remaining.length,1,cv.CV_32FC2,remaining.flatMap(p=>p.slice(0,2))),dst=cv.matFromArray(remaining.length,1,cv.CV_32FC2,remaining.flatMap(p=>[p[0]-p[2],p[1]])),mask=new cv.Mat();let H,inliers;
   try{H=cv.findHomography(src,dst,cv.RANSAC,1.5,mask,1500,.995);if(H.empty())break;const accepted=mask.data;inliers=remaining.filter((p,i)=>accepted[i]);}finally{H?.delete();src.delete();dst.delete();mask.delete();}
   if(inliers.length<minSupport)break;let plane=fit(inliers);if(!plane)break;
   for(let k=0;k<3;k++){inliers=remaining.filter(p=>Math.abs(value(plane,p[0],p[1])-p[2])<=1.5);if(inliers.length<minSupport)break;plane=fit(inliers);if(!plane)break;}
   if(!plane||inliers.length<minSupport)break;const used=new Set(inliers);remaining=remaining.filter(p=>!used.has(p));
   const leftHull=hull(inliers.map(p=>p.slice(0,2))),rightHull=hull(inliers.map(p=>[p[0]-p[2],p[1]])),areas=[area(leftHull),area(rightHull)];
   if(areas.some(a=>a<minArea)||Math.abs(1-plane[0])<.1)continue;
   const rms=Math.sqrt(inliers.reduce((s,p)=>s+(value(plane,p[0],p[1])-p[2])**2,0)/inliers.length);
   models.push({plane,leftHull,rightHull,support:inliers.length,rms,areas});
  }
  // Plane propagation consumes the fitted models directly. Do not rasterize
  // two full image maps when that caller never reads them. Background depth
  // classification still requests the same support-limited maps by default.
  const maps=options.rasterize===false?[]:[new Float32Array(S*S),new Float32Array(S*S)],tolerances=maps.map(()=>new Float32Array(S*S));maps.forEach(a=>a.fill(NaN));
  for(const model of (options.rasterize===false?[]:models))for(let side=0;side<2;side++){
   const h=side?model.rightHull:model.leftHull,p=model.plane,x0=Math.max(0,Math.ceil(Math.min(...h.map(p=>p[0])))),x1=Math.min(S-1,Math.floor(Math.max(...h.map(p=>p[0])))),y0=Math.max(0,Math.ceil(Math.min(...h.map(p=>p[1])))),y1=Math.min(S-1,Math.floor(Math.max(...h.map(p=>p[1]))));
   for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){if(!inside(h,x,y))continue;const d=value(p,x,y)/(side?1-p[0]:1),k=y*S+x;if(d>0&&(!Number.isFinite(maps[side][k])||d>maps[side][k])){maps[side][k]=d;tolerances[side][k]=Math.max(3,3*model.rms);}}
  }
  return {maps,tolerances,report:{method:'OpenCV RANSAC planar background; affine disparity inside measured support hulls',samples:samples.length,models:models.map(({plane,support,rms,areas,leftHull,rightHull})=>({plane,support,rms,areas,leftHull,rightHull})),covered:maps.map(a=>a.reduce((n,d)=>n+Number.isFinite(d),0)),ms:performance.now()-started}};
 }
 return {build};
})();
