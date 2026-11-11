'use strict';
const SCAN_VERSION='20261111-scene-planes-8';
importScripts('opencv.js',...['calibration-core.js','joint-paper.js','multi-paper.js','stereo-calibration.js','geometry-audit.js','gpu-stereo.js','background-planes.js','surface-completion.js','least-squares-stereo.js','joint-depth-solver.js','piecewise-stereo.js','plane-propagation.js'].map(path=>path+'?v='+SCAN_VERSION));
let gpuPromise=null;
const getGpu=()=>gpuPromise||(gpuPromise=GpuStereo.create().catch(e=>{gpuPromise=null;throw e;}));
let CV,engine,nativeAPI,cameras=null,rect=null,bgs=null,backgroundDisparity=null,referenceMarkers=null,stabilization=[],calibrationDiagnostics=[],collected=[[],[]],fittedLenses=null;
let currentMarkers=[],backgroundSources=null,backgroundMarkers=null,backgroundRightDisparity=null,backgroundPlanes=null,exposure=[];
const ready=new Promise((resolve,reject)=>{const init=()=>{try{CV=cv;engine=new A4Core.Engine(CV);resolve({version:CV.getBuildInformation().split('\n').slice(0,4).join(' '),mog2:typeof CV.BackgroundSubtractorMOG2,sgbm:typeof CV.StereoSGBM});}catch(e){reject(e)}};if(cv.Mat)init();else cv.onRuntimeInitialized=init;});
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s),unit=a=>scale(a,1/Math.hypot(...a)),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],mv=A4Core.mul,T=A4Core.transpose;
const median=a=>{const b=a.slice().sort((a,b)=>a-b);return b[Math.floor(b.length/2)];};
const fingerprint=data=>{let h=2166136261;for(let i=0;i<data.length;i++)h=Math.imul(h^data[i],16777619);return h>>>0;};
// This WASM exposes Embind's handle clone, which aliases the original Mat.
// OpenCV copyTo explicitly allocates independent pixel storage.
const copyMat=m=>{const out=new CV.Mat();m.copyTo(out);return out;};
function lensReport(group,w,h){const cv=CV,rows=[];const calculate=ms=>{const a=cv.matFromArray(ms.length*4,1,cv.CV_32FC2,ms.flatMap(m=>A4Core.tagUV(m.id).flat())),b=cv.matFromArray(ms.length*4,1,cv.CV_32FC2,ms.flatMap(m=>m.corners.flat()));let H;try{H=cv.findHomography(a,b,0);if(H.empty())return {reason:'homography empty'};const q=H.data64F,x=[q[0]-w/2*q[6],q[3]-h/2*q[6]],y=[q[1]-w/2*q[7],q[4]-h/2*q[7]],aa=x[0]*y[0]+x[1]*y[1],bb=q[6]*q[7],cc=x[0]**2+x[1]**2-y[0]**2-y[1]**2,dd=q[6]**2-q[7]**2,den=bb*bb+dd*dd,f2=-(aa*bb+cc*dd)/den;if(!(f2>0))return {reason:'nonpositive focal squared',f2};const f=Math.sqrt(f2),n1=x[0]**2+x[1]**2+f2*q[6]**2,n2=y[0]**2+y[1]**2+f2*q[7]**2;return {f,fov:2*Math.atan(w/(2*f))*180/Math.PI,residual:Math.hypot(aa+f2*bb,cc+f2*dd)/(n1+n2),tilt:f*Math.hypot(q[6],q[7])/Math.sqrt((n1+n2)/2)};}finally{H?.delete();a.delete();b.delete();}};rows.push(calculate(group.markers));if(group.markers.length>=4)for(let i=0;i<group.markers.length;i++)rows.push(calculate(group.markers.filter((_,j)=>j!==i)));const lens=engine.focalFromPaper(group,w,h);return {accepted:!!lens,lens,estimates:rows};}
function releaseRect(){backgroundDisparity=null;backgroundSources=null;backgroundMarkers=null;if(rect){rect.maps.forEach(m=>m.forEach(x=>x.delete()));rect=null;}if(bgs){bgs.forEach(b=>{b.model.delete();b.rgb.delete();});bgs=null;}}
function calibrate(images,opts){
 const found=images.map((im,i)=>{const d=engine.detect(im,{},true);const groups=engine.boards(d.markers);const group=groups[opts.refs[i]];
  if(!group)throw Error(`カメラ${i?'B':'A'}: 基準紙${opts.refs[i]}を検出できません（タグ${d.markers.length}、紙${groups.length}）。`);
  let lens=fittedLenses?.[i]?.ok?fittedLenses[i]:engine.focalFromPaper(group,im.width,im.height),markers;const attempts=[];
  const multi=fittedLenses?.[i]||MultiPaper.fit(CV,groups,im.width,im.height,opts);attempts.push({method:'multi-paper',...multi});if(multi.ok)lens=multi;
  if(!lens){for(let gi=0;gi<groups.length;gi++){if(groups[gi]===group)continue;const j=JointPaper.fit(CV,group,groups[gi],im.width,im.height,opts.ruler);attempts.push({method:'joint-plane',other:gi,...j,markers:undefined});if(j?.ok&&(!lens||j.rms<lens.rms)){lens=j;markers=j.markers;}}}
  calibrationDiagnostics[i]={tags:d.markers.length,groups:groups.length,reference:opts.refs[i],attempts};
  if(!lens)throw Error(`カメラ${i?'B':'A'}: ${groups.length}枚・${d.markers.length}タグを検出。レンズ推定を採用できません。${multi.reason}`);
  if(!markers){
   const quality=MultiPaper.prepare(CV,[group])[0],complete=new Set(group.markers.filter(m=>quality.points.filter(p=>p.id===m.id).length===4).map(m=>m.id));
   markers=group.markers.filter(m=>complete.has(m.id)).map(m=>({...m,object:A4Core.tagUV(m.id).map(([u,v])=>[(u-148.5)*opts.ruler/100,(105-v)*opts.ruler/100,0])}));
   calibrationDiagnostics[i].poseTags=markers.map(m=>m.id);
  }
  const pose=engine.pose(markers,lens.K,lens.D,true);if(!pose)throw Error('A4の姿勢推定が成立しません');
  return {K:lens.K,D:lens.D,width:im.width,height:im.height,R:pose.R,t:pose.t.map(x=>x/1000),center:pose.center.map(x=>x/1000),rms:pose.rms,fov:lens.fov,lensSpread:lens.spread,source:lens.source,model:lens.model||'opencv-pinhole-brown5',markers:group.markers.map(m=>({id:m.id,corners:m.corners})),groupCount:groups.length,reference:opts.refs[i],assumptions:lens.assumptions};
 });
 const baseline=Math.hypot(...sub(found[0].center,found[1].center));if(baseline<.015||baseline>5)throw Error(`カメラ間隔${baseline.toFixed(3)}mは不適切です。同じ基準紙か確認してください。`);
 cameras=found;releaseRect();return {cameras,baseline,units:'m',origin:'reference paper center; X right, Y up, Z paper front',createdAt:new Date().toISOString()};
}
function importCalibration(value){const source=value.cameras||value; if(!Array.isArray(source)||source.length!==2)throw Error('2台分のcameras配列が必要です');const cs=source.map(c=>{const p=c.pose||c;const R=p.R,t=p.t;if(!Array.isArray(R)||R.length!==9||!Array.isArray(t)||t.length!==3||!Array.isArray(c.K)||c.K.length!==9||!c.width||!c.height)throw Error('K, D, width, height, R, tが必要です。tの単位はmです。');if(![...R,...t,...c.K].every(Number.isFinite))throw Error('校正に非数値があります');return {...c,R,t,D:c.D||[0,0,0,0,0],center:scale(mv(T(R),t),-1)};});cameras=cs;releaseRect();return {cameras,baseline:Math.hypot(...sub(cs[0].center,cs[1].center)),units:'m'};}
function makeRect(images,S){
 const key=images.map(im=>`${im.width}x${im.height}`).join('/')+'/'+S;if(rect?.key===key)return rect;
 releaseRect();const C0=cameras[0].center,C1=cameras[1].center,x=unit(sub(C1,C0));let z=add(mv(T(cameras[0].R),[0,0,1]),mv(T(cameras[1].R),[0,0,1]));z=unit(sub(z,scale(x,dot(z,x))));const y=unit(cross(z,x));z=unit(cross(x,y));const R=[...x,...y,...z],Rt=T(R),ranges=[];
 for(const c of cameras){const pts=[];for(let k=0;k<=8;k++)for(const p of [[c.width*k/8,0],[c.width*k/8,c.height],[0,c.height*k/8],[c.width,c.height*k/8]]){const ray=mv(R,mv(T(c.R),[(p[0]-c.K[2])/c.K[0],(p[1]-c.K[5])/c.K[4],1]));if(ray[2]>.05)pts.push([ray[0]/ray[2],ray[1]/ray[2]]);}if(pts.length<4)throw Error('共通視野を作れません');ranges.push([Math.min(...pts.map(p=>p[0])),Math.max(...pts.map(p=>p[0])),Math.min(...pts.map(p=>p[1])),Math.max(...pts.map(p=>p[1]))]);}
 const loX=Math.max(ranges[0][0],ranges[1][0]),hiX=Math.min(ranges[0][1],ranges[1][1]),loY=Math.max(ranges[0][2],ranges[1][2]),hiY=Math.min(ranges[0][3],ranges[1][3]);
 if(hiX-loX<.05||hiY-loY<.05)throw Error('共通視野が不足しています。基準紙の対応を確認してください。');
 const f=(S-10)/Math.max(hiX-loX,hiY-loY),cx=S/2-f*(hiX+loX)/2,cy=S/2-f*(hiY+loY)/2;
 const maps=cameras.map((c,i)=>{const mapX=new CV.Mat(S,S,CV.CV_32FC1),mapY=new CV.Mat(S,S,CV.CV_32FC1),sx=images[i].width/c.width,sy=images[i].height/c.height;
  for(let v=0;v<S;v++)for(let u=0;u<S;u++){const p=mv(c.R,mv(Rt,[(u-cx)/f,(v-cy)/f,1])),j=v*S+u;if(p[2]<=0){mapX.data32F[j]=-1;mapY.data32F[j]=-1;continue;}const a=p[0]/p[2],b=p[1]/p[2],r=a*a+b*b,D=c.D,rad=1+(D[0]||0)*r+(D[1]||0)*r*r+(D[4]||0)*r*r*r;mapX.data32F[j]=(c.K[0]*(a*rad+2*(D[2]||0)*a*b+(D[3]||0)*(r+2*a*a))+c.K[2])*sx;mapY.data32F[j]=(c.K[4]*(b*rad+(D[2]||0)*(r+2*b*b)+2*(D[3]||0)*a*b)+c.K[5])*sy;}return [mapX,mapY];});
 rect={key,S,f,cx,cy,R,Rt,C0,baseline:Math.hypot(...sub(C1,C0)),maps,imageSizes:images.map(im=>[im.width,im.height]),valid:[new Uint8Array(S*S),new Uint8Array(S*S)]};return rect;
}
function frameShift(im,i){
 const current=engine.detect(im,{},true).markers;currentMarkers[i]=current;
 const reference=referenceMarkers?.[i];if(!reference)return {x:0,y:0,status:'reference-unavailable'};
 const sx=cameras[i].width/im.width,sy=cameras[i].height/im.height,deltas=[],ids=[];
 for(const m of current){const candidates=reference.filter(r=>r.id===m.id).map(r=>({r,d:Math.hypot(m.corners[0][0]*sx-r.corners[0][0],m.corners[0][1]*sy-r.corners[0][1])})).sort((a,b)=>a.d-b.d);if(!candidates.length||candidates[0].d>24)continue;const ref=candidates[0].r;ids.push(m.id);m.corners.forEach((p,j)=>deltas.push([p[0]*sx-ref.corners[j][0],p[1]*sy-ref.corners[j][1]]));}
 if(new Set(ids).size<2)return {x:0,y:0,status:'insufficient-tags',corners:deltas.length};
 const x=median(deltas.map(d=>d[0])),y=median(deltas.map(d=>d[1])),errors=deltas.map(d=>Math.hypot(d[0]-x,d[1]-y)),residual=median(errors);
 if(residual>1.5)throw Error(`カメラ${i+1}: 静止タグの変化が平行移動では説明できません（${residual.toFixed(2)}px）。現在の配置を再校正してください。`);
 return {x:x/sx,y:y/sy,status:'static-tag-translation',corners:deltas.length,residual,originalShift:[x,y]};
}
function warp(im,i){const a=CV.matFromImageData(im),rgb=new CV.Mat(),out=new CV.Mat(),shift=frameShift(im,i);stabilization[i]=shift;let maps=rect.maps[i],adjusted;
 try{if(shift.x||shift.y){adjusted=maps.map(copyMat);const mx=adjusted[0].data32F,my=adjusted[1].data32F;for(let k=0;k<rect.S*rect.S;k++){mx[k]+=shift.x;my[k]+=shift.y;}maps=adjusted;}const mx=maps[0].data32F,my=maps[1].data32F;for(let k=0;k<rect.S*rect.S;k++)rect.valid[i][k]=mx[k]>=0&&my[k]>=0&&mx[k]<im.width-1&&my[k]<im.height-1?255:0;CV.cvtColor(a,rgb,CV.COLOR_RGBA2RGB);CV.remap(rgb,out,maps[0],maps[1],CV.INTER_LINEAR,CV.BORDER_CONSTANT,new CV.Scalar(0,0,0,0));return out;}finally{adjusted?.forEach(m=>m.delete());a.delete();rgb.delete();}}
function imageFromRGB(m){const width=m.cols,height=m.rows,count=width*height,src=m.data,data=new Uint8ClampedArray(count*4);for(let i=0;i<count;i++){data[4*i]=src[3*i];data[4*i+1]=src[3*i+1];data[4*i+2]=src[3*i+2];data[4*i+3]=255;}return {data,width,height};}
async function train(images,opts){
 if(!cameras)throw Error('先にカメラを校正してください');makeRect(images,opts.size);
 if(bgs)bgs.forEach(b=>{b.model.delete();b.rgb.delete();});backgroundDisparity=null;
 bgs=images.map((im,i)=>{const rgb=warp(im,i),model=new CV.BackgroundSubtractorMOG2(60,40,true),mask=new CV.Mat();try{for(let n=0;n<12;n++)model.apply(rgb,mask,n===0?1:.2);}finally{mask.delete();}return {rgb,model};});
 backgroundSources=images;backgroundMarkers=currentMarkers.map(ms=>ms.map(m=>({id:m.id,corners:m.corners.map(p=>p.slice())})));
 const noShadow={raw:new Uint8Array(rect.S*rect.S)};
 const reference=await matchSurface(bgs[0].rgb.data,bgs[1].rgb.data,noShadow,noShadow,{...opts,step:opts.method==='sgbm'?1:opts.step,subpixel:false},true);
 backgroundDisparity=reference.validDisparity;
 backgroundRightDisparity=new Float32Array(rect.S*rect.S);backgroundRightDisparity.fill(NaN);
 for(let n=0;n<reference.uv.length;n+=4){const [x,y,u]=reference.uv.subarray(n,n+3),d=backgroundDisparity[y*rect.S+x],k=y*rect.S+Math.round(u);if(!Number.isFinite(backgroundRightDisparity[k])||d>backgroundRightDisparity[k])backgroundRightDisparity[k]=d;}
 backgroundPlanes=BackgroundPlanes.build(CV,reference,rect.S);
 return {backgroundReady:true,backgroundPoints:reference.points.length/3,backgroundModel:backgroundPlanes.report,previews:bgs.map(b=>imageFromRGB(b.rgb))};
}
// MOG2 provides shadow/background rejection only. RGB correspondence is
// computed independently; its foreground class never supplies surface points.
function measureExposure(image,i){
 const before=backgroundSources[i],reference=backgroundMarkers[i],pairs=[[],[],[]],tagFits=[[],[],[]];
 const sample=(im,c,u,v)=>{const p=[0,1].map(k=>(1-v)*((1-u)*c[0][k]+u*c[1][k])+v*((1-u)*c[3][k]+u*c[2][k])),x=Math.round(p[0]),y=Math.round(p[1]),j=(y*im.width+x)*4;return Array.from(im.data.slice(j,j+3));};
 for(const m of currentMarkers[i]){
  const candidates=reference.filter(r=>r.id===m.id).map(r=>({r,d:Math.hypot(m.corners[0][0]/image.width-r.corners[0][0]/before.width,m.corners[0][1]/image.height-r.corners[0][1]/before.height)})).sort((a,b)=>a.d-b.d);if(!candidates.length||candidates[0].d>.025)continue;
  const bs=[[],[],[]],as=[[],[],[]];
  for(let y=0;y<16;y++)for(let x=0;x<16;x++){const b=sample(before,candidates[0].r.corners,(x+.5)/16,(y+.5)/16),a=sample(image,m.corners,(x+.5)/16,(y+.5)/16);for(let c=0;c<3;c++){bs[c].push(b[c]);as[c].push(a[c]);if(b[c]>8&&b[c]<247&&a[c]>8&&a[c]<247)pairs[c].push([b[c],a[c]]);}}
  for(let c=0;c<3;c++){bs[c].sort((a,b)=>a-b);as[c].sort((a,b)=>a-b);const b0=bs[c][25],b1=bs[c][230],a0=as[c][25],a1=as[c][230],gain=(a1-a0)/(b1-b0);if(b1-b0>50&&gain>.5&&gain<2)tagFits[c].push({gain,bias:a0-gain*b0});}
 }
 const channels=pairs.map((ps,c)=>{
  if(tagFits[c].length>=2){const gain=median(tagFits[c].map(t=>t.gain)),bias=median(tagFits[c].map(t=>t.bias)),sigma=1.4826*median(tagFits[c].map(t=>Math.abs(t.gain-gain)*128+Math.abs(t.bias-bias)));if(sigma<=15)return {gain,bias,sigma,ok:true,samples:ps.length,tags:tagFits[c].length,method:'paired-tag-intensity-quantiles'};}
  const fail={gain:1,bias:0,sigma:8,ok:false,samples:ps.length};ps.sort((a,b)=>a[0]-b[0]);if(ps.length<30)return fail;
  const n=Math.max(6,Math.floor(ps.length*.2)),dark=ps.slice(0,n),light=ps.slice(-n),b0=median(dark.map(p=>p[0])),b1=median(light.map(p=>p[0]));if(b1-b0<40)return fail;
  let gain=1,bias=median(ps.map(p=>p[1]-p[0]));
  for(let iteration=0;iteration<12;iteration++){
   const limit=Math.max(2,1.4826*median(ps.map(p=>Math.abs(p[1]-gain*p[0]-bias)))*1.5);let sw=0,sx=0,sy=0,sxx=0,sxy=0;
   for(const [x,y]of ps){const w=Math.min(1,limit/Math.max(1e-6,Math.abs(y-gain*x-bias)));sw+=w;sx+=w*x;sy+=w*y;sxx+=w*x*x;sxy+=w*x*y;}
   const den=sw*sxx-sx*sx;if(den<1e-6)return fail;gain=(sw*sxy-sx*sy)/den;bias=(sy-gain*sx)/sw;
  }
  const sigma=1.4826*median(ps.map(p=>Math.abs((p[1]-bias)/gain-p[0])));if(gain<.5||gain>2||sigma>15)return {...fail,reason:'unstable-photometry',estimatedGain:gain,estimatedBias:bias,estimatedSigma:sigma};
  return {gain,bias,sigma,ok:true,samples:ps.length};
 });
 return {channels,ok:channels.every(c=>c.ok)};
}
function shadowCandidates(rgb,i,image){const mask=new CV.Mat(),corrected=copyMat(rgb);exposure[i]=measureExposure(image,i);try{const dst=corrected.data,src=rgb.data;for(let k=0;k<dst.length;k++){const c=exposure[i].channels[k%3];dst[k]=Math.max(0,Math.min(255,Math.round((src[k]-c.bias)/c.gain)));}bgs[i].model.apply(corrected,mask,0);const raw=new Uint8Array(mask.data);let shadow=0;for(const value of raw)if(value===127)shadow++;return {raw,shadow};}finally{mask.delete();corrected.delete();}}
// RGB zero-mean normalized cross correlation. Each channel has its own mean.
function statistics(rgb,S,r){const n=(S+1)*(S+1),sums=[new Float64Array(n),new Float64Array(n),new Float64Array(n)],sq=new Float64Array(n);for(let y=0;y<S;y++){const rows=[0,0,0];let q=0;for(let x=0;x<S;x++){const a=(y*S+x)*3,k=(y+1)*(S+1)+x+1;for(let c=0;c<3;c++){rows[c]+=rgb[a+c];sums[c][k]=sums[c][k-S-1]+rows[c];q+=rgb[a+c]*rgb[a+c];}sq[k]=sq[k-S-1]+q;}}
 const means=[new Float32Array(S*S),new Float32Array(S*S),new Float32Array(S*S)],norms=new Float32Array(S*S),N=(2*r+1)**2;const box=(a,x,y)=>a[(y+r+1)*(S+1)+x+r+1]-a[(y-r)*(S+1)+x+r+1]-a[(y+r+1)*(S+1)+x-r]+a[(y-r)*(S+1)+x-r];
 for(let y=r;y<S-r;y++)for(let x=r;x<S-r;x++){const k=y*S+x;let v=box(sq,x,y);for(let c=0;c<3;c++){const sum=box(sums[c],x,y);means[c][k]=sum/N;v-=sum*sum/N;}norms[k]=Math.sqrt(Math.max(0,v));}return {means,norms,N,r};}
function textureEigenvalues(rgb,S,block){const src=CV.matFromArray(S,S,CV.CV_8UC3,rgb),gray=new CV.Mat(),eigen=new CV.Mat();try{CV.cvtColor(src,gray,CV.COLOR_RGB2GRAY);CV.cornerMinEigenVal(gray,eigen,block,3,CV.BORDER_DEFAULT);return new Float32Array(eigen.data32F);}finally{src.delete();gray.delete();eigen.delete();}}
async function matchSurface(rgbA,rgbB,fa,fb,opts,reference=false){const matchingStarted=performance.now(),{S,f,cx,cy,Rt,C0,baseline}=rect;
 const sceneMode=!reference&&opts.scope==='scene',foregroundOnly=!reference&&!sceneMode;
 const inspectionStage=opts.inspectRegion?new Uint8Array(S*S):null;
 rgbA=new Uint8Array(rgbA);rgbB=new Uint8Array(rgbB);
 // Stereo searches one dimension. A straight edge may still constrain its
 // disparity; a two-dimensional corner threshold is optional, not required.
 const r=opts.patchRadius??2,minimumEigenvalue=opts.minimumEigenvalue??0,eigen=reference||minimumEigenvalue<=0?null:[textureEigenvalues(rgbA,S,2*r+1),textureEigenvalues(rgbB,S,2*r+1)];
 const textures=[rgbA,rgbB,...(!reference&&bgs?bgs.map(b=>b.rgb.data):[])],st=textures.map(rgb=>statistics(rgb,S,r)),step=opts.step,minD=Math.max(1,Math.floor(f*baseline/opts.far)),maxD=Math.min(S-2*r-1,Math.ceil(f*baseline/opts.near)),disparity=new Float32Array(S*S),scores=new Float32Array(S*S),candidates=[];disparity.fill(NaN);
 if(!Number.isInteger(step)||step<1)throw Error('点間隔は1以上の整数が必要です');
 // Preserve the original two-pixel support test on each interleaved grid.
 // Increasing sampling density must not lower the evidence for an island.
 const supportStep=step===1?2:step,refined=new Uint8Array(S*S);
 const corr=(a,b,x,u,y)=>{const A=textures[a],B=textures[b],p=y*S+x,q=y*S+u,den=st[a].norms[p]*st[b].norms[q];if(den<1e-5)return -2;let v=0;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const k=((y+dy)*S+x+dx)*3,l=((y+dy)*S+u+dx)*3;v+=A[k]*B[l]+A[k+1]*B[l+1]+A[k+2]*B[l+2];}for(let c=0;c<3;c++)v-=st[a].N*st[a].means[c][p]*st[b].means[c][q];return v/den;};
 const subpixelCorr=(x,u,y)=>{
  const ix=Math.floor(u),fraction=u-ix;if(ix<r||ix+1>=S-r)return -2;
  const q=y*S+ix;if(!rect.valid[1][q]||!rect.valid[1][q+1]||(foregroundOnly&&(fb.raw[q]!==255||fb.raw[q+1]!==255)))return -2;
  const sums=[0,0,0],leftMeans=st[0].means,p=y*S+x,N=st[0].N;let crossSum=0,square=0;
  for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){
   const a=((y+dy)*S+x+dx)*3,b=((y+dy)*S+ix+dx)*3;
   for(let c=0;c<3;c++){const v=rgbB[b+c]*(1-fraction)+rgbB[b+c+3]*fraction;sums[c]+=v;square+=v*v;crossSum+=rgbA[a+c]*v;}
  }
  for(let c=0;c<3;c++){crossSum-=leftMeans[c][p]*sums[c];square-=sums[c]*sums[c]/N;}
  const norm=Math.sqrt(Math.max(0,square));return norm<Math.sqrt(N*3)*3?-2:crossSum/(st[0].norms[p]*norm);
 };
 const choose=(a,b,x,y,reverse=false)=>{const shadows=b===0?fa.raw:fb.raw,arr=[];let best=-2,bx=-1;const begin=reverse?x+minD:x-maxD,end=reverse?x+maxD:x-minD;for(let u=Math.max(r,begin);u<=Math.min(S-r-1,end);u++){const k=y*S+u;if(!rect.valid[b][k]||(!sceneMode&&shadows[k]===127)||st[b].norms[k]<Math.sqrt(st[b].N*3)*3)continue;const v=corr(a,b,x,u,y);arr.push([u,v]);if(v>best){best=v;bx=u;}}let second=-2;for(const [u,v]of arr)if(Math.abs(u-bx)>2)second=Math.max(second,v);return {x:bx,score:best,gap:best-second,candidates:arr.length};};
 const counts={sampled:0,invalidImage:0,shadow:0,texture:0,aperture:0,correlation:0,ambiguous:0,roundtrip:0,neighborhood:0,geometry:0,background:0};
 const backgroundPatch=(i,x,y)=>{if(!exposure[i]?.ok)return false;const a=textures[i],b=textures[i+2],fit=exposure[i].channels;let dot=0,norm=0;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const j=((y+dy)*S+x+dx)*3;for(let c=0;c<3;c++){const value=(a[j+c]-fit[c].bias)/fit[c].gain;dot+=value*b[j+c];norm+=b[j+c]*b[j+c];}}const alpha=dot/Math.max(1,norm);if(alpha<.25||alpha>1.08)return false;let residual=0;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const j=((y+dy)*S+x+dx)*3;for(let c=0;c<3;c++)residual+=((a[j+c]-fit[c].bias)/fit[c].gain-Math.min(1,alpha)*b[j+c])**2;}const tolerance=Math.min(12,Math.max(6,median(fit.map(c=>c.sigma))*2.5));return Math.sqrt(residual/(st[0].N*3))<tolerance;};
 const preparedAt=performance.now(),range=Math.min(S-minD-16,maxD-minD+1),dense=opts.method==='sgbm'?(opts.backend==='webgpu'?await (await getGpu()).compute(rgbA,rgbB,S,S,minD,range,nativeAPI,opts.stereoBlock||5):nativeAPI.stereoDisparity(rgbA,rgbB,S,S,minD,range)):null,stereoAt=performance.now();
 for(let y=r;y<S-r;y+=step)for(let x=r;x<S-r;x+=step){
  const k=y*S+x;counts.sampled++;if(inspectionStage)inspectionStage[k]=1;
  // Rectification adds pixels beyond the original camera image. Those black
  // border pixels are not observations, even if a patch crosses a real edge.
  if(!rect.valid[0][k]){counts.invalidImage++;continue;}
  if(!sceneMode&&fa.raw[k]===127){counts.shadow++;if(inspectionStage)inspectionStage[k]=2;continue;}if(st[0].norms[k]<Math.sqrt(st[0].N*3)*3){counts.texture++;if(inspectionStage)inspectionStage[k]=3;continue;}
  if(eigen&&eigen[0][k]<minimumEigenvalue){counts.aperture++;continue;}
  if(dense){
   const d=dense.left[k]/16,u=Math.round(x-d);
   if(d<minD||d>maxD||u<r||u>=S-r){counts.ambiguous++;if(inspectionStage)inspectionStage[k]=4;continue;}
   const q=y*S+u,back=dense.right[q]/16;
   if(!rect.valid[1][q]){counts.invalidImage++;continue;}
   if(back<=dense.rightMinimum-1||Math.abs(d+back)>1){counts.roundtrip++;if(inspectionStage)inspectionStage[k]=5;continue;}
   if(!sceneMode&&fb.raw[q]===127){counts.shadow++;if(inspectionStage)inspectionStage[k]=2;continue;}
   if(eigen&&eigen[1][q]<minimumEigenvalue){counts.aperture++;continue;}
   let score=corr(0,1,x,u,y),observedU=u;
   if(score<opts.ncc&&opts.subpixel!==false){const actualU=x-d,accurate=subpixelCorr(x,actualU,y);if(accurate>=opts.ncc){score=accurate;observedU=actualU;refined[k]=1;}}
   if(score<opts.ncc){counts.correlation++;if(inspectionStage)inspectionStage[k]=6;continue;}
   // Native SGBM already checks uniqueness of its aggregated matching cost.
   // Requiring the independent patch-only winner to agree discards the
   // surface evidence gained by that aggregation (measured in v28).
   disparity[k]=d;scores[k]=score;if(inspectionStage)inspectionStage[k]=7;candidates.push([x,y,u,observedU]);continue;
  }
  const m=choose(0,1,x,y);if(m.x<0||m.score<opts.ncc||m.candidates<3){counts.correlation++;continue;}if(m.gap<opts.unique){counts.ambiguous++;continue;}const back=choose(1,0,m.x,y,true);if(Math.abs(back.x-x)>1||back.score<opts.ncc||back.gap<opts.unique){counts.roundtrip++;continue;}let d=x-m.x;const l=corr(0,1,x,m.x-1,y),z=m.score,h=corr(0,1,x,m.x+1,y),den=l-2*z+h;if(den<-.001)d-=Math.max(-.5,Math.min(.5,.5*(l-h)/den));disparity[k]=d;scores[k]=m.score;candidates.push([x,y,m.x]);
 }
 const candidatesAt=performance.now(),points=[],colors=[],uv=[],accepted=[],depths=[],angles=[],validDisparity=new Float32Array(S*S);validDisparity.fill(NaN);
 const backgroundRGB=foregroundOnly&&bgs?[rgbA,rgbB].map(rgb=>CV.matFromArray(S,S,CV.CV_8UC3,rgb)):null;
 const backgroundCache=[new Uint8Array(S*S),new Uint8Array(S*S)];
 const localBackground=(i,x,y)=>{
  if(!backgroundRGB||x<7||y<7||x>=S-7||y>=S-7)return false;
  const k=y*S+x,cache=backgroundCache[i];if(cache[k])return cache[k]===2;
  const patch=backgroundRGB[i].roi(new CV.Rect(x-4,y-4,9,9)),search=bgs[i].rgb.roi(new CV.Rect(x-7,y-7,15,15)),response=new CV.Mat();
  try{CV.matchTemplate(search,patch,response,CV.TM_CCOEFF_NORMED);const yes=CV.minMaxLoc(response).maxVal>.98;cache[k]=yes?2:1;return yes;}
  finally{patch.delete();search.delete();response.delete();}
 };
 try{
 for(const [x,y,u,observedU=u]of candidates){
  const k=y*S+x,d=disparity[k],neighbors=[];
  if(foregroundOnly&&(fa.raw[k]===0||fb.raw[y*S+u]===0)){counts.background++;if(inspectionStage)inspectionStage[k]=8;continue;}
  for(let dy=-supportStep*2;dy<=supportStep*2;dy+=supportStep)for(let dx=-supportStep*2;dx<=supportStep*2;dx+=supportStep){if(!dx&&!dy)continue;const xx=x+dx,yy=y+dy;if(xx>=0&&xx<S&&yy>=0&&yy<S&&Number.isFinite(disparity[yy*S+xx]))neighbors.push(disparity[yy*S+xx]);}
  if(neighbors.length<2||neighbors.filter(v=>Math.abs(v-d)<Math.max(2,d*.1)).length<2){counts.neighborhood++;if(inspectionStage)inspectionStage[k]=9;continue;}
  const z=f*baseline/d,world=add(C0,mv(Rt,[(x-cx)*z/f,(y-cy)*z/f,z])),rays=cameras.map(c=>sub(world,c.center));
  const angle=Math.atan2(Math.hypot(...cross(...rays)),dot(...rays))*180/Math.PI;
  const q=mv(rect.R,rays[1]),residual=Math.hypot(f*q[0]/q[2]+cx-observedU,f*q[1]/q[2]+cy-y);
  if(z<opts.near||z>opts.far||!world.every(Number.isFinite)||cameras.some((c,i)=>mv(c.R,rays[i])[2]<=0)||angle<1||residual>1.5||(foregroundOnly&&world[2]<.03)){counts.geometry++;if(inspectionStage)inspectionStage[k]=10;continue;}
  // Static background is measured by the same stereo matcher. A shadow does
  // not become a surface merely because MOG2 labels it as foreground.
  if(foregroundOnly&&((Number.isFinite(backgroundDisparity?.[k])&&Math.abs(d-backgroundDisparity[k])<Math.max(2,d*.02))||(st.length===4&&(corr(0,2,x,x,y)>.95||corr(1,3,u,u,y)>.95||backgroundPatch(0,x,y)||backgroundPatch(1,u,y))))){counts.background++;if(inspectionStage)inspectionStage[k]=11;continue;}
  // A background edge can move slightly after camera stabilization. Compare
  // the observed RGB patch against a small background neighborhood as well.
  if(localBackground(0,x,y)||localBackground(1,u,y)){counts.background++;counts.backgroundLocal=(counts.backgroundLocal||0)+1;if(inspectionStage)inspectionStage[k]=11;continue;}
  if(inspectionStage)inspectionStage[k]=12;points.push(...world);const j=k*3;colors.push(rgbA[j],rgbA[j+1],rgbA[j+2]);uv.push(x,y,observedU,y);accepted.push(k);depths.push(z);angles.push(angle);validDisparity[k]=d;
 }
 // Reject isolated correspondence islands in measured 3D, without a person
 // silhouette, skeleton, bounding box or body-size template.
 if(!reference){
  const radius=.045,buckets=new Map(),keys=[],phases=[];for(let i=0;i<points.length/3;i++){const q=points.slice(i*3,i*3+3).map(v=>Math.floor(v/radius)),phase=step===1?((uv[i*4]-r)&1)+2*((uv[i*4+1]-r)&1):0;keys.push(q);phases.push(phase);const key=phase+':'+q.join(',');if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(i);}
  const seen=new Uint8Array(keys.length),keep=new Set();
  for(let i=0;i<keys.length;i++){if(seen[i])continue;seen[i]=1;const component=[i];for(let j=0;j<component.length;j++){const a=component[j],q=keys[a];for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++)for(const b of buckets.get(phases[a]+':'+[q[0]+dx,q[1]+dy,q[2]+dz].join(','))||[]){if(seen[b])continue;if(Math.hypot(points[a*3]-points[b*3],points[a*3+1]-points[b*3+1],points[a*3+2]-points[b*3+2])<=radius){seen[b]=1;component.push(b);}}}if(component.length>=12)component.forEach(j=>keep.add(j));}
  counts.isolated3D=keys.length-keep.size;
  for(let i=keys.length-1;i>=0;i--)if(!keep.has(i)){points.splice(i*3,3);colors.splice(i*3,3);uv.splice(i*4,4);accepted.splice(i,1);depths.splice(i,1);angles.splice(i,1);}
 }
 counts.subpixelRecovered=accepted.filter(k=>refined[k]).length;
 if(inspectionStage){for(const k of accepted)inspectionStage[k]=13;const polygon=opts.inspectRegion,inside=(x,y)=>{let yes=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){const a=polygon[i],b=polygon[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])yes=!yes;}return yes;},names=['not-sampled','other','shadow','low-texture','sgbm-invalid','left-right','rgb-correlation','candidate','mog-background','neighbors','geometry','background-match','island','accepted'],histogram={},pixels=[];for(let y=r;y<S-r;y++)for(let x=r;x<S-r;x++){if(!inside(x,y))continue;const k=y*S+x,stage=names[inspectionStage[k]];histogram[stage]=(histogram[stage]||0)+1;pixels.push([x,y,inspectionStage[k],dense?dense.left[k]/16:null,scores[k],st[0].norms[k]/Math.sqrt(st[0].N*3)]);}counts.inspection={polygon,histogram,pixels};}
 // Classification follows triangulation; every measured point remains available.
 // Missing background depth is unknown, never an invented wall or person.
 const labels=new Uint8Array(accepted.length),classification={static:0,nearer:0,unknown:0,shadowOnStatic:0,planeAssisted:0,nearerWithPlane:0},classificationSource=new Uint8Array(accepted.length);
 if(sceneMode)for(let i=0;i<accepted.length;i++){
  const k=accepted[i],d=disparity[k],u=Math.round(uv[i*4+2]),q=uv[i*4+3]*S+u,measured=[backgroundDisparity?.[k],backgroundRightDisparity?.[q]],indexes=[k,q],usedPlane=measured.some(v=>!Number.isFinite(v));
  const bg=measured.map((v,j)=>Number.isFinite(v)?v:backgroundPlanes?.maps[j][indexes[j]]),tolerances=measured.map((v,j)=>Number.isFinite(v)?Math.max(2,d*.02):Math.max(d*.02,backgroundPlanes?.tolerances[j][indexes[j]]||3));
  const [a,b]=bg,[ta,tb]=tolerances;let label=0;if(Number.isFinite(a)&&Number.isFinite(b)){if(Math.abs(d-a)<=ta&&Math.abs(d-b)<=tb)label=1;else if(d-a>ta&&d-b>tb)label=2;}
  if(label){classificationSource[i]=usedPlane?2:1;if(usedPlane){classification.planeAssisted++;if(label===2)classification.nearerWithPlane++;}}
  labels[i]=label;classification[['unknown','static','nearer'][label]]++;
  if(label===1&&(fa.raw[k]===127||fb.raw[q]===127))classification.shadowOnStatic++;
 }
 if(sceneMode)counts.backgroundModel=backgroundPlanes?.report;
 const result={method:dense?(dense.gpuTiming?'WebGPU RGB SGBM 3WAY + OpenCV filtering + RGB NCC validation':'OpenCV StereoSGBM RGB + native uniqueness + RGB NCC validation'):'RGB NCC',backend:dense?.gpuTiming?'webgpu':'cpu',gpuTiming:dense?.gpuTiming||null,patchSize:2*r+1,minimumEigenvalue,points:new Float32Array(points),colors:new Uint8Array(colors),uv:new Float32Array(uv),accepted,counts,...(sceneMode?{labels,classification,classificationSource}:{}),...(reference?{validDisparity}:{}),rayAngle:angles.length?{min:Math.min(...angles),median:median(angles)}:null,depth:depths.length?{min:Math.min(...depths),median:median(depths),max:Math.max(...depths)}:null,disparityRange:[minD,maxD],matchingTiming:{prepareMs:preparedAt-matchingStarted,stereoMs:stereoAt-preparedAt,candidateMs:candidatesAt-stereoAt,filterMs:performance.now()-candidatesAt}};
 if(opts.completion&&!reference&&(foregroundOnly||['bilateral','piecewise','alsc','photogrammetry','plane-propagation'].includes(opts.completionMethod))){
  const accept=(x,y,u,d,world)=>{
   if(x<r||y<r||u<r||x>=S-r||y>=S-r||u>=S-r)return false;
   const k=y*S+x,rays=cameras.map(c=>sub(world,c.center));
   if(cameras.some((c,i)=>mv(c.R,rays[i])[2]<=0)||Math.atan2(Math.hypot(...cross(...rays)),dot(...rays))*180/Math.PI<1)return false;
   if(!rect.valid[0][k]||!rect.valid[1][y*S+u])return false;
   if(!foregroundOnly)return true;
   if(Number.isFinite(backgroundDisparity?.[k])&&Math.abs(d-backgroundDisparity[k])<Math.max(2,d*.02))return false;
   if(st.length===4&&(corr(0,2,x,x,y)>.95||corr(1,3,u,u,y)>.95||backgroundPatch(0,x,y)||backgroundPatch(1,u,y)))return false;
   return !localBackground(0,x,y)&&!localBackground(1,u,y);
  };
  if(opts.completionMethod==='photogrammetry'||opts.completionMethod==='plane-propagation'){
   if(opts.completionMethod==='photogrammetry')await LeastSquaresStereo.complete(await getGpu(),result,rgbA,rgbB,fa,fb,rect,opts,accept);
   await PlanePropagation.complete(CV,await getGpu(),result,rgbA,rgbB,fa,fb,rect,opts,accept);
  }
  else if(opts.completionMethod==='piecewise')await PiecewiseStereo.complete(CV,await getGpu(),result,rgbA,rgbB,fa,fb,rect,opts,accept);
  else if(opts.completionMethod==='bilateral')await JointDepthSolver.complete(CV,await getGpu(),result,rgbA,rgbB,fa,fb,rect,opts,accept);
  else if(opts.completionMethod==='alsc')await LeastSquaresStereo.complete(await getGpu(),result,rgbA,rgbB,fa,fb,rect,opts,accept);
  else await SurfaceCompletion.complete(CV,await getGpu(),result,rgbA,rgbB,fa,fb,rect,opts,accept);
  // Depth classification also applies to inferred surfaces. Keep pointKinds:
  // a background/nearer label never turns an estimate into an observation.
  if(sceneMode&&result.labels&&result.labels.length<result.points.length/3){const labels=new Uint8Array(result.points.length/3),source=new Uint8Array(labels.length);labels.set(result.labels);source.set(result.classificationSource);result.classification.unknown+=labels.length-result.labels.length;result.labels=labels;result.classificationSource=source;}
  if(sceneMode&&result.labels)for(let i=accepted.length;i<result.points.length/3;i++){
   const x=result.uv[4*i],y=result.uv[4*i+1],u=result.uv[4*i+2],d=x-u,k=y*S+x,q=y*S+Math.round(u),indexes=[k,q],measured=[backgroundDisparity?.[k],backgroundRightDisparity?.[q]],usedPlane=measured.some(v=>!Number.isFinite(v));
   const bg=measured.map((v,j)=>Number.isFinite(v)?v:backgroundPlanes?.maps[j][indexes[j]]),tol=measured.map((v,j)=>Number.isFinite(v)?Math.max(2,d*.02):Math.max(d*.02,backgroundPlanes?.tolerances[j][indexes[j]]||3));
   let label=0;if(bg.every(Number.isFinite)){if(bg.every((v,j)=>Math.abs(d-v)<=tol[j]))label=1;else if(bg.every((v,j)=>d-v>tol[j]))label=2;}
   if(!label)continue;result.labels[i]=label;result.classificationSource[i]=usedPlane?2:1;result.classification.unknown--;result.classification[['unknown','static','nearer'][label]]++;if(usedPlane){result.classification.planeAssisted++;if(label===2)result.classification.nearerWithPlane++;}if(label===1&&(fa.raw[k]===127||fb.raw[q]===127))result.classification.shadowOnStatic++;
  }
 }
 return result;
 }finally{backgroundRGB?.forEach(m=>m.delete());}
}
async function analyze(images,opts){if(!cameras||!bgs)throw Error('校正と影判定用の背景が必要です');const start=performance.now();const expected=images.map(im=>`${im.width}x${im.height}`).join('/')+'/'+opts.size;if(rect.key!==expected)throw Error('処理解像度または映像の向きを変更したため、背景を学習し直してください');const rgbs=images.map((im,i)=>warp(im,i));try{const fingerprints={input:images.map(im=>fingerprint(im.data)),rectified:rgbs.map(m=>fingerprint(m.data)),background:bgs.map(b=>fingerprint(b.rgb.data))};const fs=rgbs.map((rgb,i)=>shadowCandidates(rgb,i,images[i])),maskMs=performance.now()-start,result=await matchSurface(rgbs[0].data,rgbs[1].data,fs[0],fs[1],opts);result.fingerprints={...fingerprints,afterMatch:rgbs.map(m=>fingerprint(m.data)),shadow:fs.map(m=>fingerprint(m.raw))};const previews=rgbs.map(imageFromRGB);for(let i=0;i<2;i++)for(let k=0;k<fs[i].raw.length;k++){const p=previews[i].data,j=4*k;if(fs[i].raw[k]===127){p[j]=Math.round(p[j]*.3);p[j+1]=Math.round(p[j+1]*.5);p[j+2]=160;}}
 for(let n=0;n<result.uv.length;n+=4)for(let i=0;i<2;i++){const x=Math.round(result.uv[n+2*i]),y=Math.round(result.uv[n+2*i+1]),k=y*rect.S+x;if(x>=0&&x<rect.S&&y>=0&&y<rect.S){const color=[1,2,4,5,6].includes(result.pointKinds?.[n/4])?[230,150,20]:result.labels?[[255,180,50],[60,190,255],[70,255,135]][result.labels[n/4]]:[70,255,135];for(let c=0;c<3;c++)previews[i].data[4*k+c]=color[c];}}
 return {...result,previews,shadow:fs.map(a=>({pixels:a.shadow})),stabilization,exposure,baseline:rect.baseline,coverage:result.counts.sampled?result.points.length/3/result.counts.sampled:0,ms:performance.now()-start,maskMs,rectification:{f:rect.f,cx:rect.cx,cy:rect.cy,R:rect.R,size:rect.S},cameras};}finally{rgbs.forEach(m=>m.delete());}}
self.onmessage=async e=>{
 const {id,type,images,opts,value}=e.data;
 try{
  const info=await ready;let result;
  if(type==='background'||type==='scan')nativeAPI=await StereoCalibration.module();
  if(type==='init')result={...info,scanVersion:SCAN_VERSION,stereoCalibrate:typeof CV.stereoCalibrate,stereoRectify:typeof CV.stereoRectify,Board:typeof CV.aruco_Board};
  else if(type==='resetLenses'){collected=[[],[]];fittedLenses=null;result=true;}
  else if(type==='collectLenses')result=images.map((im,i)=>{
   const groups=engine.boards(engine.detect(im,{},true).markers).filter(g=>g.markers.length>=2);
   collected[i].push(...groups);return {groups:groups.length,total:collected[i].length};
  });
  else if(type==='fitLenses'){fittedLenses=images.map((im,i)=>MultiPaper.fit(CV,collected[i],im.width,im.height,opts));result=fittedLenses;}
  else if(type==='diagnostics')result=calibrationDiagnostics;
  else if(type==='audit')result=await GeometryAudit.run(CV,engine,images,value,e.data.pairs||value.pairs,rect,images.map(frameShift));
  else if(type==='lenses')result=images.map(im=>{const d=engine.detect(im,{},true),groups=engine.boards(d.markers);return {tags:d.markers.length,groups:groups.length,...MultiPaper.fit(CV,groups,im.width,im.height,opts)};});
  else if(type==='calibrate')result=calibrate(images,opts);
  else if(type==='stereo'){
   result=await StereoCalibration.fit(CV,engine,images,e.data.pairs,opts);
   calibrationDiagnostics=[result];
   if(result.ok){cameras=result.cameras;referenceMarkers=result.referenceMarkers;releaseRect();}
  }
  else if(type==='reset'){cameras=null;referenceMarkers=null;fittedLenses=null;collected=[[],[]];releaseRect();result=true;}
  else if(type==='import'){result=importCalibration(value);referenceMarkers=value.referenceMarkers||null;}
  else if(type==='background')result=await train(images,opts);
  else if(type==='scan')result=await analyze(images,opts);
  else if(type==='detect')result=images.map(im=>{const d=engine.detect(im,{},true);return {markers:d.markers,groups:engine.boards(d.markers).map(g=>({...g,lens:lensReport(g,im.width,im.height)}))};});
  else throw Error('未知の処理');
  self.postMessage({id,result});
 }catch(e){self.postMessage({id,error:e?.message||String(e),stack:e?.stack});}
};
