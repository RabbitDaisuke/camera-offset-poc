'use strict';
// Epipolar-constrained adaptive least-squares matching and seeded region growing.
// The horizontal affine warp and per-channel radiometry are fitted to actual
// image samples. This is not GOTCHA code or an unconstrained 2D optical flow.
// Algorithm references: Gruen (1985, 2012), Andre et al. (2008), Otto/Chau (1989).
self.LeastSquaresStereo=(()=>{
 const shader=`
 struct Params {size:u32,radius:i32,iterations:u32,pad:u32,ncc:f32,minD:f32,maxD:f32,drift:f32,front:vec4<f32>,rayZ:vec4<f32>};
 @group(0) @binding(0) var<uniform> p:Params;
 @group(0) @binding(1) var<storage,read> images:array<u32>;
 @group(0) @binding(2) var<storage,read> masks:array<u32>;
 @group(0) @binding(3) var<storage,read> inputMap:array<vec4<f32>>;
 @group(0) @binding(4) var<storage,read_write> outputMap:array<vec4<f32>>;
 @group(0) @binding(5) var<storage,read_write> attemptedSeeds:array<u32>;
 fn rgb(x:i32,y:i32,side:i32)->vec3<f32>{let v=images[(side*i32(p.size)+y)*i32(p.size)+x];return vec3<f32>(f32(v&255u),f32((v>>8u)&255u),f32((v>>16u)&255u));}
 fn sample(x:f32,y:i32,side:i32)->vec3<f32>{let ix=i32(floor(x));return mix(rgb(ix,y,side),rgb(ix+1,y,side),fract(x));}
 fn invalid()->vec4<f32>{return vec4<f32>(0.,0.,0.,-1.);}
 fn inFront(x:i32,y:i32,side:i32,disp:f32)->bool{return select(p.front.y,p.front.z,side==1)+dot(p.rayZ.xyz,vec3<f32>(f32(x),f32(y),1.))*p.front.w/abs(disp)>p.front.x;}
 fn refine(x:i32,y:i32,side:i32,initial:vec3<f32>)->vec4<f32>{
  // Every accepted iterate is within p.drift of this initial disparity.
  // Bound its best possible A4-front distance before reading a whole patch.
  // This skips only fits whose entire allowed interval is behind the plane;
  // any interval crossing zero is retained, avoiding a singular depth bound.
  if((p.pad&2u)!=0u){
   let lo=initial.x-p.drift;let hi=initial.x+p.drift;
   if(lo*hi>0.){let minimum=max(p.minD,min(abs(lo),abs(hi)));let maximum=min(p.maxD,max(abs(lo),abs(hi)));
    if(minimum>maximum){return invalid();}
    let coefficient=dot(p.rayZ.xyz,vec3<f32>(f32(x),f32(y),1.))*p.front.w;
    let bestDisp=select(minimum,maximum,coefficient<0.);let bestZ=select(p.front.y,p.front.z,side==1)+coefficient/bestDisp;
    if(bestZ<p.front.x-1e-5){return invalid();}
   }
  }
  let rad=f32(p.radius);var warp=initial;var finalNcc=-1.;var finalError=1e10;var lastStep=1e10;
  var accepted=invalid();var acceptedError=1e10;
  for(var iter=0u;iter<=p.iterations;iter++){
   var sa=vec3<f32>(0);var sb=vec3<f32>(0);var aa=vec3<f32>(0);var bb=vec3<f32>(0);var ab=vec3<f32>(0);var count=0.;
   for(var dy=-p.radius;dy<=p.radius;dy++){for(var dx=-p.radius;dx<=p.radius;dx++){
    let xx=x+dx;let yy=y+dy;let basis=vec3<f32>(1.,f32(dx)/rad,f32(dy)/rad);let u=f32(xx)-dot(warp,basis);
    if(xx<0||yy<0||xx>=i32(p.size)||yy>=i32(p.size)||u<1.||u>=f32(p.size)-2.){return accepted;}
    let a=rgb(xx,yy,side);let b=sample(u,yy,1-side);sa+=a;sb+=b;aa+=a*a;bb+=b*b;ab+=a*b;count+=1.;
   }}
   let va=max(aa-sa*sa/count,vec3<f32>(0));let vb=max(bb-sb*sb/count,vec3<f32>(0));let cov=ab-sa*sb/count;
   if(dot(va,vec3<f32>(1))<27.*count||dot(vb,vec3<f32>(1))<27.*count){return accepted;}
   // Eliminate the linear RGB gain/bias parameters from the joint normal
   // equations; their coupling to the geometric warp is retained below.
   let gain=clamp(cov/max(vb,vec3<f32>(1.)),vec3<f32>(.25),vec3<f32>(4.));let bias=(sa-gain*sb)/count;
   finalNcc=dot(cov,vec3<f32>(1))/sqrt(dot(va,vec3<f32>(1))*dot(vb,vec3<f32>(1)));
   var h=mat3x3<f32>(vec3<f32>(0),vec3<f32>(0),vec3<f32>(0));var rhs=vec3<f32>(0);var residual=0.;var sumJ:array<vec3<f32>,3>;var sumJB:array<vec3<f32>,3>;var sumE=vec3<f32>(0);var sumEB=vec3<f32>(0);
   for(var dy=-p.radius;dy<=p.radius;dy++){for(var dx=-p.radius;dx<=p.radius;dx++){
    let xx=x+dx;let yy=y+dy;let basis=vec3<f32>(1.,f32(dx)/rad,f32(dy)/rad);let u=f32(xx)-dot(warp,basis);
    let a=rgb(xx,yy,side);let b=sample(u,yy,1-side);let error=a-gain*b-bias;let gradient=-gain*(sample(u+.5,yy,1-side)-sample(u-.5,yy,1-side));
    let gg=dot(gradient,gradient);h+=mat3x3<f32>(basis*basis.x*gg,basis*basis.y*gg,basis*basis.z*gg);rhs+=basis*dot(gradient,error);residual+=dot(error,error);sumE+=error;sumEB+=error*b;for(var c=0u;c<3u;c++){let j=basis*gradient[c];sumJ[c]+=j;sumJB[c]+=j*b[c];}
   }}
   finalError=residual/(count*3.);
   if(finalNcc>=p.ncc&&finalError<=144.&&lastStep<=.25&&abs(warp.x)>=p.minD&&abs(warp.x)<=p.maxD&&finalError<acceptedError&&inFront(x,y,side,warp.x)){let u=f32(x)-warp.x;let q=(u32(1-side)*p.size+u32(y))*p.size+u32(round(u));if(u>=0.&&u<f32(p.size)&&q<arrayLength(&masks)&&masks[q]==255u){accepted=vec4<f32>(warp,1.-finalNcc);acceptedError=finalError;}}
   if(iter==p.iterations||lastStep<.005){break;}
   for(var c=0u;c<3u;c++){let j=sumJ[c];h-=mat3x3<f32>(j*j.x,j*j.y,j*j.z)*(1./count);rhs-=j*sumE[c]/count;let freeGain=cov[c]/max(vb[c],1.);if(vb[c]>1.&&freeGain>.25&&freeGain<4.){let jb=sumJB[c]-j*sb[c]/count;h-=mat3x3<f32>(jb*jb.x,jb*jb.y,jb*jb.z)*(1./vb[c]);rhs-=jb*(sumEB[c]-sumE[c]*sb[c]/count)/vb[c];}}
   let damping=max(.01,(h[0].x+h[1].y+h[2].z)*.001);h[0].x+=damping;h[1].y+=damping;h[2].z+=damping;
   let determinant=dot(h[0],cross(h[1],h[2]));if(determinant<1e-6){return accepted;}
   var delta=vec3<f32>(dot(rhs,cross(h[1],h[2])),dot(h[0],cross(rhs,h[2])),dot(h[0],cross(h[1],rhs)))/determinant;
   delta=clamp(delta,vec3<f32>(-1.),vec3<f32>(1.));lastStep=length(delta);warp+=delta;
   if(abs(warp.x-initial.x)>p.drift||abs(warp.y)>rad*1.5||abs(warp.z)>rad*1.5||1.-warp.y/rad<.15){return accepted;}
  }
  return accepted;
 }
 @compute @workgroup_size(64) fn seeds(@builtin(global_invocation_id) gid:vec3<u32>){
  let k=gid.x;if(k>=2u*p.size*p.size){return;}outputMap[k]=invalid();if(inputMap[k].w<0.||masks[k]!=255u){return;}
  let x=i32(k%p.size);let y=i32(k/p.size%p.size);let side=i32(k/(p.size*p.size));outputMap[k]=refine(x,y,side,inputMap[k].xyz);
 }
 @compute @workgroup_size(64) fn grow(@builtin(global_invocation_id) gid:vec3<u32>){
  let k=gid.x;if(k>=2u*p.size*p.size){return;}outputMap[k]=inputMap[k];if(inputMap[k].w>=0.||masks[k]!=255u){return;}
  let x=i32(k%p.size);let y=i32(k/p.size%p.size);let side=i32(k/(p.size*p.size));let center=rgb(x,y,side);var candidates:array<vec4<f32>,8>;var ranks:array<f32,8>;var flags:array<u32,8>;var count=0u;
  for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
   if(dx==0&&dy==0){continue;}let flag=1u<<u32((dy+1)*3+dx+1);if((p.pad&1u)!=0u&&(attemptedSeeds[k]&flag)!=0u){continue;}let xx=x+dx;let yy=y+dy;if(xx<0||yy<0||xx>=i32(p.size)||yy>=i32(p.size)){continue;}
   let q=(u32(side)*p.size+u32(yy))*p.size+u32(xx);let other=inputMap[q];if(other.w<0.){continue;}
   let colorDistance=dot(abs(center-rgb(xx,yy,side)),vec3<f32>(1));if(colorDistance>105.){continue;}
   var candidate=other;candidate.x-=other.y*f32(dx)/f32(p.radius)+other.z*f32(dy)/f32(p.radius);candidates[count]=candidate;ranks[count]=other.w+colorDistance/765.;flags[count]=flag;count++;
  }}
  // Accepted neighbors never change in this image pair. A failed fit from
  // that exact neighbor has identical inputs in every later round. Cache only
  // that attempt, not the pixel: newly accepted neighbors are still tested.
  for(var attempt=0u;attempt<count;attempt++){var selected=0u;var best=1e10;for(var j=0u;j<count;j++){if(ranks[j]<best){best=ranks[j];selected=j;}}ranks[selected]=1e10;attemptedSeeds[k]|=flags[selected];let fitted=refine(x,y,side,candidates[selected].xyz);if(fitted.w>=0.){outputMap[k]=fitted;return;}}
 }
 `;
 class Matcher{
  constructor(device){this.device=device;this.buffers=[];}
  async init(){const module=this.device.createShaderModule({label:'Epipolar adaptive least-squares RGB matching',code:shader}),info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>m.message).join('\n'));this.seed=await this.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'seeds'}});this.grow=await this.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'grow'}});}
  async compute(result,rgbA,rgbB,fa,fb,rect,opts){
   const started=performance.now(),d=this.device,{S,R,C0,f,baseline}=rect,N=S*S,radius=opts.alscRadius||10;
   for(const b of this.buffers)b.destroy();this.buffers=[];
   const create=(size,usage,data)=>{const b=d.createBuffer({size,usage});this.buffers.push(b);if(data)d.queue.writeBuffer(b,0,data);return b;};
   const pixels=new Uint32Array(2*N),masks=new Uint32Array(2*N),seedMap=new Float32Array(2*N*4);for(let i=0;i<2*N;i++)seedMap[4*i+3]=-1;
   for(let side=0;side<2;side++){const rgb=side?rgbB:rgbA,mask=opts.scope==='scene'?rect.valid[side]:side?fb.raw:fa.raw;for(let i=0;i<N;i++){pixels[side*N+i]=rgb[3*i]|rgb[3*i+1]<<8|rgb[3*i+2]<<16;masks[side*N+i]=mask[i];}}
   for(let i=0;i<result.points.length/3;i++){const v=[0,1,2].map(c=>result.points[3*i+c]-C0[c]),z=R[6]*v[0]+R[7]*v[1]+R[8]*v[2],disp=f*baseline/z,x=result.uv[4*i],y=result.uv[4*i+1],u=Math.round(x-disp);seedMap.set([disp,0,0,0],4*(y*S+x));if(u>=0&&u<S)seedMap.set([-disp,0,0,0],4*(N+y*S+u));}
   // A measured wall pixel is not a missing pixel. Suppress new matches there
   // while retaining any front observation sharing the rounded right pixel.
   for(const [x,y,u] of result.wallConstraints||[]){masks[y*S+x]=0;const q=N+y*S+Math.round(u);if(q>=N&&q<2*N&&seedMap[q*4+3]<0)masks[q]=0;}
   // Fit an initial disparity slope only to nearby measured correspondences.
   for(let side=0;side<2;side++)for(let y=0;y<S;y++)for(let x=0;x<S;x++){
    const k=side*N+y*S+x;if(seedMap[4*k+3]<0)continue;let xx=0,xy=0,yy=0,xd=0,yd=0,n=0;
    for(let dy=-6;dy<=6;dy++)for(let dx=-6;dx<=6;dx++){const a=x+dx,b=y+dy;if(a<0||b<0||a>=S||b>=S)continue;const q=side*N+b*S+a;if(seedMap[q*4+3]<0)continue;const diff=seedMap[4*q]-seedMap[4*k];if(Math.abs(diff)>12)continue;xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;xd+=dx*diff;yd+=dy*diff;n++;}
    const det=xx*yy-xy*xy;if(n>=6&&det>1){seedMap[4*k+1]=Math.max(-radius,Math.min(radius,(xd*yy-yd*xy)/det*radius));seedMap[4*k+2]=Math.max(-radius,Math.min(radius,(yd*xx-xd*xy)/det*radius));}
   }
   const parameters=new ArrayBuffer(64);new Uint32Array(parameters).set([S,radius,20,Number(opts.alscDeduplicate!==false)|(opts.a4FrontOnly&&opts.alscFrontPrune!==false?2:0)]);new Float32Array(parameters).set([opts.ncc,1,S-1,3],4);new Float32Array(parameters).set([opts.a4FrontOnly?opts.a4FrontMeters:-1e6,C0[2],C0[2]+baseline*R[2],f*baseline,rect.Rt[6]/f,rect.Rt[7]/f,rect.Rt[8]-(rect.Rt[6]*rect.cx+rect.Rt[7]*rect.cy)/f,0],8);
   const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,uniform=create(64,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,parameters),image=create(pixels.byteLength,storage,pixels),mask=create(masks.byteLength,storage,masks),seed=create(seedMap.byteLength,storage,seedMap),a=create(seedMap.byteLength,storage),b=create(seedMap.byteLength,storage),read=create(seedMap.byteLength,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
   const attempted=create(2*N*4,storage);
   const bind=(pipeline,src,dst)=>d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[uniform,image,mask,src,dst,...(pipeline===this.grow?[attempted]:[])].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const encoder=d.createCommandEncoder(),dispatch=(pipeline,group)=>{const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(2*N/64));pass.end();};
   d.pushErrorScope('validation');dispatch(this.seed,bind(this.seed,seed,a));let src=a,dst=b;const groups=[bind(this.grow,a,b),bind(this.grow,b,a)],rounds=opts.alscRounds??32;for(let i=0;i<rounds;i++){dispatch(this.grow,groups[i%2]);[src,dst]=[dst,src];}
   encoder.copyBufferToBuffer(src,0,read,0,seedMap.byteLength);const prepared=performance.now();d.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();const error=await d.popErrorScope();if(error)throw Error(error.message);return {values,rounds,radius,prepareMs:prepared-started,computeReadbackMs:performance.now()-prepared};
  }
 }
 let matcher;
 async function complete(gpu,result,rgbA,rgbB,fa,fb,rect,opts,accept){
  const started=performance.now();if(!matcher){matcher=new Matcher(gpu.device);await matcher.init();}
  const matched=await matcher.compute(result,rgbA,rgbB,fa,fb,rect,opts),{S,f,baseline,Rt,C0,cx,cy}=rect,v=matched.values,points=Array.from(result.points),colors=Array.from(result.colors),uv=Array.from(result.uv),kinds=new Array(points.length/3).fill(0),existing=new Set(result.accepted);let added=0,lrRejected=0,filterRejected=0,matchedLeft=0,matchedRight=0;
  for(let k=0;k<S*S;k++){if(v[k*4+3]>=0)matchedLeft++;if(v[(S*S+k)*4+3]>=0)matchedRight++;}
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){
   const k=y*S+x,d=v[k*4],u=x-d;if(existing.has(k)||v[k*4+3]<0||u<0||u>=S)continue;
   const ix=Math.floor(u),q=S*S+y*S+ix,next=q+1;if(ix+1>=S||v[q*4+3]<0||v[next*4+3]<0){lrRejected++;continue;}const back=v[q*4]*(1-(u-ix))+v[next*4]*(u-ix);if(Math.abs(d+back)>1){lrRejected++;continue;}
   const z=f*baseline/d;if(z<opts.near||z>opts.far)continue;const ray=[(x-cx)*z/f,(y-cy)*z/f,z],world=[0,1,2].map(i=>C0[i]+Rt[i*3]*ray[0]+Rt[i*3+1]*ray[1]+Rt[i*3+2]*ray[2]);if((opts.scope!=='scene'&&world[2]<.03)||!accept(x,y,Math.round(u),d,world)){filterRejected++;continue;}
   points.push(...world);colors.push(rgbA[k*3],rgbA[k*3+1],rgbA[k*3+2]);uv.push(x,y,u,y);kinds.push(3);added++;
  }
  result.points=new Float32Array(points);result.colors=new Uint8Array(colors);result.uv=new Float32Array(uv);result.pointKinds=new Uint8Array(kinds);
  result.completion={method:'epipolar-joint-ALSC-region-growing',sampleStep:1,geometricRadiometricCoupling:'Schur complement',maxIterations:20,scope:opts.scope,measured:kinds.length-added,matched:added,estimated:0,matchedLeft,matchedRight,lrRejected,filterRejected,...matched,values:undefined,totalMs:performance.now()-started};return result;
 }
 return {complete};
})();
