'use strict';
// Seeded slanted-window stereo: local least-squares disparity planes, adaptive
// RGB support weights and spatial propagation (Bleyer et al., BMVC 2011).
// This is a seeded adaptation, not a complete PatchMatch Stereo port. Optional
// plane refinement searches disparity and normal, not just the seed planes.
// No temporal guesses or semantic masks are used. Added depths remain estimates.
self.PlanePropagation=(()=>{
 const shader=`
 struct Params {size:u32,pixels:u32,radius:i32,jump:i32,minD:f32,maxD:f32,gamma:f32,consistency:f32,iteration:u32,refine:u32,front:vec4<f32>,rayZ:vec4<f32>};
 struct Field {plane:vec4<f32>,bias:vec4<f32>};
 @group(0) @binding(0) var<uniform> p:Params;
 @group(0) @binding(1) var<storage,read> images:array<u32>;
 @group(0) @binding(2) var<storage,read> masks:array<u32>;
 @group(0) @binding(3) var<storage,read> seeds:array<Field>;
 @group(0) @binding(4) var<storage,read> inputMap:array<Field>;
 @group(0) @binding(5) var<storage,read_write> outputMap:array<Field>;
 fn index(x:i32,y:i32,side:i32)->u32{return u32((side*i32(p.size)+y)*i32(p.size)+x);}
 fn rgb(x:i32,y:i32,side:i32)->vec3<f32>{let v=images[index(x,y,side)];return vec3<f32>(f32(v&255u),f32((v>>8u)&255u),f32((v>>16u)&255u));}
 fn sample(x:f32,y:i32,side:i32)->vec3<f32>{let a=i32(floor(x));return mix(rgb(a,y,side),rgb(a+1,y,side),fract(x));}
 fn empty()->Field{return Field(vec4<f32>(0,0,0,1e9),vec4<f32>(0));}
 fn random(state:ptr<function,u32>)->f32{*state=1664525u*(*state)+1013904223u;return f32((*state)>>8u)/16777216.;}
 fn evaluate(x:i32,y:i32,side:i32,v:Field)->Field{
  let disp=v.plane.x;let u=f32(x)-disp;
  let worldZ=select(p.front.y,p.front.z,side==1)+dot(p.rayZ.xyz,vec3<f32>(f32(x),f32(y),1.))*p.front.w/abs(disp);
  if(worldZ<=p.front.x){return empty();}
  if(abs(disp)<p.minD||abs(disp)>p.maxD||disp*select(1.,-1.,side==1)<=0.||u<2.||u>=f32(p.size)-3.||masks[index(i32(round(u)),y,1-side)]!=255u){return empty();}
  let center=rgb(x,y,side);var weight=0.;var error=0.;var photo=0.;var anchorWeight=0.;var anchorError=0.;
  for(var dy=-p.radius;dy<=p.radius;dy++){for(var dx=-p.radius;dx<=p.radius;dx++){
   let xx=x+dx;let yy=y+dy;let d=disp+v.plane.y*f32(dx)+v.plane.z*f32(dy);let q=f32(xx)-d;
   if(xx<2||yy<2||xx>=i32(p.size)-3||yy>=i32(p.size)-3||q<2.||q>=f32(p.size)-3.){continue;}
   let k=index(xx,yy,side);if(masks[k]!=255u||masks[index(i32(round(q)),yy,1-side)]!=255u){continue;}
   let a=rgb(xx,yy,side);let b=sample(q,yy,1-side);let w=exp(-dot(abs(center-a),vec3<f32>(1))/p.gamma);
   let delta=a-b-v.bias.xyz;let ag=(rgb(xx+1,yy,side)-rgb(xx-1,yy,side))*.5;
   let bg=(sample(q+1.,yy,1-side)-sample(q-1.,yy,1-side))*.5*(1.-v.plane.y);
   error+=w*(.1*min(dot(abs(delta),vec3<f32>(1./3.)),10.)+.9*min(dot(abs(ag-bg),vec3<f32>(1./3.)),2.));
   photo+=w*dot(delta,delta)/3.;weight+=w;
   if(seeds[k].bias.w!=0.){anchorWeight+=w;anchorError+=w*min(100.,(d-seeds[k].plane.x)*(d-seeds[k].plane.x));}
  }}
  if(weight<8.||photo/weight>400.){return empty();}
  if(anchorWeight>1.&&anchorError/anchorWeight>9.){return empty();}
  let other=inputMap[index(i32(round(u)),y,1-side)];var consistency=0.;if(other.bias.w>0.){let reverse=other.plane.x+other.plane.y*(u-round(u));consistency=p.consistency*min(4.,(disp+reverse)*(disp+reverse));}
  var result=v;result.plane.w=error/weight+.1*anchorError/max(anchorWeight,1.)+consistency;return result;
 }
 @compute @workgroup_size(64) fn propagate(@builtin(global_invocation_id) gid:vec3<u32>){
  let k=gid.x;if(k>=2u*p.pixels){return;}let x=i32(k%p.size);let y=i32(k/p.size%p.size);let side=i32(k/p.pixels);
  if(masks[k]!=255u||seeds[k].bias.w<0.){outputMap[k]=empty();return;}
  var best=empty();if(inputMap[k].bias.w>0.){best=evaluate(x,y,side,inputMap[k]);}
  for(var axis=0;axis<4;axis++){
   let dx=select(0,select(-p.jump,p.jump,axis==1),axis<2);let dy=select(0,select(-p.jump,p.jump,axis==3),axis>=2);let xx=x+dx;let yy=y+dy;
   if(xx<0||yy<0||xx>=i32(p.size)||yy>=i32(p.size)){continue;}var v=inputMap[index(xx,yy,side)];if(v.bias.w<=0.){continue;}
   v.plane.x-=v.plane.y*f32(dx)+v.plane.z*f32(dy);v.bias.w+=f32(abs(dx)+abs(dy));let candidate=evaluate(x,y,side,v);
   if(candidate.plane.w<best.plane.w-1e-5||(abs(candidate.plane.w-best.plane.w)<=1e-5&&candidate.bias.w<best.bias.w)){best=candidate;}
  }
  // Evaluate the inverse slanted plane from the other view at the current
  // projected location; it must still pass this view's RGB/anchor costs.
  if(best.bias.w>0.){let u=i32(round(f32(x)-best.plane.x));if(u>=0&&u<i32(p.size)){let other=inputMap[index(u,y,1-side)];let den=1.-other.plane.y;if(other.bias.w>0.&&abs(den)>.15){let d=(other.plane.y*f32(u-x)-other.plane.x)/den;let proposal=Field(vec4<f32>(d,-other.plane.y/den,-other.plane.z/den,0),vec4<f32>(-other.bias.xyz,other.bias.w));let candidate=evaluate(x,y,side,proposal);if(candidate.plane.w<best.plane.w){best=candidate;}}}}
  // PatchMatch plane refinement (Bleyer et al. 2011, section 2.2): perturb
  // disparity and the plane normal with an exponentially shrinking range.
  // This searches new planes using RGB observations at every tested pixel.
  if(p.refine!=0u&&best.bias.w>0.&&seeds[k].bias.w<=0.){
   var state=k*747796405u+(p.iteration+1u)*2891336453u;var delta=p.maxD*.5;var normalDelta=1.;
   for(var trial=0;trial<13&&delta>=.1;trial++){
    var candidate=best;let n=normalize(vec3<f32>(-best.plane.y,-best.plane.z,1.));
    let jitter=vec3<f32>(random(&state),random(&state),random(&state))*2.-1.;let changed=normalize(n+jitter*normalDelta);
    candidate.plane.x+=(random(&state)*2.-1.)*delta;
    if(abs(changed.z)>.1){candidate.plane.y=-changed.x/changed.z;candidate.plane.z=-changed.y/changed.z;candidate=evaluate(x,y,side,candidate);if(candidate.plane.w<best.plane.w){best=candidate;}}
    delta*=.5;normalDelta*=.5;
   }
  }
  // Valid measurements constrain the plane at their own pixel, even when
  // a wide support window straddles another surface.
  if(seeds[k].bias.w>0.){best=seeds[k];best.plane.w=0.;}
  outputMap[k]=best;
 }
 `;
 class Matcher{
  constructor(device){this.device=device;this.buffers=[];}
  async init(){const module=this.device.createShaderModule({code:shader}),info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>m.message).join('\n'));this.pipeline=await this.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'propagate'}});}
  async run(cv,result,rgbs,masks,rect,opts){
   const start=performance.now(),{S,f,baseline}=rect,N=S*S,device=this.device,pixels=new Uint32Array(2*N),valid=new Uint32Array(2*N),field=new Float32Array(2*N*8),observed=new Float32Array(2*N);observed.fill(NaN);
   for(let side=0;side<2;side++)for(let k=0;k<N;k++){pixels[side*N+k]=rgbs[side][3*k]|rgbs[side][3*k+1]<<8|rgbs[side][3*k+2]<<16;valid[side*N+k]=masks[side][k];field[(side*N+k)*8+3]=1e9;}
   for(let i=0;i<result.points.length/3;i++){if(result.pointKinds&&![0,3].includes(result.pointKinds[i]))continue;const x=result.uv[4*i],y=result.uv[4*i+1],u=result.uv[4*i+2],d=x-u;observed[y*S+x]=d;if(u>=0&&u<S)observed[N+y*S+Math.round(u)]=-d;}
   const fitted=BackgroundPlanes.build(cv,{uv:result.uv,validDisparity:observed.subarray(0,N)},S,{sampleStep:2,minSupport:24,maxModels:16,minArea:64,rasterize:opts.planeRasterize===true}),models=fitted.report.models;
   const sample=(side,x,y,c)=>{const a=Math.floor(x),t=x-a;return rgbs[side][(y*S+a)*3+c]*(1-t)+rgbs[side][(y*S+a+1)*3+c]*t;};let seeds=[0,0],modelSeeds=[0,0];
   for(let side=0;side<2;side++)for(let y=2;y<S-2;y++)for(let x=2;x<S-2;x++){
    const k=side*N+y*S+x,d=observed[k];if(!Number.isFinite(d))continue;let xx=0,xy=0,yy=0,xd=0,yd=0,n=0;
    for(let dy=-6;dy<=6;dy++)for(let dx=-6;dx<=6;dx++){const a=x+dx,b=y+dy;if(a<0||b<0||a>=S||b>=S)continue;const other=observed[side*N+b*S+a];if(!Number.isFinite(other)||Math.abs(other-d)>12)continue;xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;xd+=dx*(other-d);yd+=dy*(other-d);n++;}
    const det=xx*yy-xy*xy;if(n<6||det<1)continue;let a=(xd*yy-yd*xy)/det,b=(yd*xx-xd*xy)/det,center=d,best=1.5,selected=null;
    for(const model of models){const [pa,pb,pc]=model.plane,scale=side?-1/(1-pa):1,predicted=(pa*x+pb*y+pc)*scale,e=Math.abs(predicted-d);if(e<best){best=e;selected=[predicted,pa*scale,pb*scale];}}
    if(selected){[center,a,b]=selected;modelSeeds[side]++;}if(Math.abs(a)>1.5||Math.abs(b)>1.5||1-a<.15)continue;
    const bias=[0,0,0];let count=0,squared=0;
    for(let dy=-3;dy<=3;dy++)for(let dx=-3;dx<=3;dx++){const px=x+dx,py=y+dy,u=px-center-a*dx-b*dy;if(px<0||px>=S||py<0||py>=S||u<0||u>=S-1)continue;for(let c=0;c<3;c++)bias[c]+=rgbs[side][(py*S+px)*3+c]-sample(1-side,u,py,c);count++;const q=observed[side*N+py*S+px];if(Number.isFinite(q))squared+=(q-center-a*dx-b*dy)**2;}
    if(!count||squared/count>4)continue;field.set([center,a,b,0,...bias.map(v=>v/count),1],k*8);seeds[side]++;
   }
   const anchors=new Float32Array(field);let wallAnchors=0;
   // Negative confidence denotes a measured depth excluded from output. It
   // constrains nearby plane costs but never propagates or becomes a vertex.
   for(const [x,y,u,d] of result.wallConstraints||[])for(let side=0;side<2;side++){
    const xx=side?Math.round(u):x;if(xx<0||xx>=S)continue;const k=side*N+y*S+xx;if(Number.isFinite(observed[k]))continue;
    anchors.set([side?-d:d,0,0,0,0,0,0,-1],k*8);wallAnchors++;
   }
   for(const b of this.buffers)b.destroy();this.buffers=[];const make=(bytes,usage,data)=>{const b=device.createBuffer({size:bytes,usage});this.buffers.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;};
   const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,im=make(pixels.byteLength,storage,pixels),mask=make(valid.byteLength,storage,valid),seed=make(anchors.byteLength,storage,anchors),a=make(field.byteLength,storage,field),b=make(field.byteLength,storage),read=make(field.byteLength,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
   const jumps=[1,2,4,8,16,32,16,8,4,2,1,1,2,4,8,16,8,4,2,1];let src=a,dst=b;const encoder=device.createCommandEncoder();device.pushErrorScope('validation');
   for(const [iteration,jump] of jumps.entries()){const data=new ArrayBuffer(80);new Uint32Array(data).set([S,N,opts.planeRadius??8,jump]);new Float32Array(data).set([f*baseline/opts.far,Math.min(S-1,f*baseline/opts.near),10,opts.planeConsistency??1],4);new Uint32Array(data).set([iteration,Number(!!opts.planeRefine)],8);new Float32Array(data).set([opts.a4FrontOnly?opts.a4FrontMeters:-1e6,rect.C0[2],rect.C0[2]+baseline*rect.R[2],f*baseline,rect.Rt[6]/f,rect.Rt[7]/f,rect.Rt[8]-(rect.Rt[6]*rect.cx+rect.Rt[7]*rect.cy)/f,0],12);const uniform=make(80,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,data),group=device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[uniform,im,mask,seed,src,dst].map((buffer,binding)=>({binding,resource:{buffer}}))}),pass=encoder.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(2*N/64));pass.end();[src,dst]=[dst,src];}
   encoder.copyBufferToBuffer(src,0,read,0,field.byteLength);const submitted=performance.now();device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const output=new Float32Array(read.getMappedRange().slice(0));read.unmap();const error=await device.popErrorScope();if(error)throw Error(error.message);return {output,seeds,modelSeeds,models,wallAnchors,prepareMs:submitted-start,computeReadbackMs:performance.now()-submitted,rounds:jumps.length,radius:opts.planeRadius??8};
  }
 }
 let matcher;
 async function complete(cv,gpu,result,rgbA,rgbB,fa,fb,rect,opts,accept){
  const start=performance.now();if(!matcher){matcher=new Matcher(gpu.device);await matcher.init();}const priorCompletion=result.completion,original=result.points.length/3,m=await matcher.run(cv,result,[rgbA,rgbB],opts.scope==='scene'?rect.valid:[fa.raw,fb.raw],rect,opts),v=m.output,{S,f,baseline,Rt,C0,cx,cy}=rect,N=S*S,points=Array.from(result.points),colors=Array.from(result.colors),uv=Array.from(result.uv),kinds=result.pointKinds?Array.from(result.pointKinds):new Array(original).fill(0),existing=new Set();for(let i=0;i<original;i++)existing.add(result.uv[4*i+1]*S+result.uv[4*i]);
  const counts={leftRight:0,photometric:0,geometry:0,added:0};
  for(let y=2;y<S-2;y++)for(let x=2;x<S-2;x++){const k=y*S+x,d=v[k*8],u=x-d;if(existing.has(k)||v[k*8+7]<=0||d<=0||u<2||u>=S-3)continue;const ix=Math.floor(u),q=N+y*S+ix,t=u-ix;if(v[q*8+7]<=0||v[(q+1)*8+7]<=0||Math.abs(d+v[q*8]*(1-t)+v[(q+1)*8]*t)>1.5){counts.leftRight++;continue;}if(v[k*8+3]>1.8||v[q*8+3]>1.8){counts.photometric++;continue;}const z=f*baseline/d,ray=[(x-cx)*z/f,(y-cy)*z/f,z],world=[0,1,2].map(i=>C0[i]+Rt[3*i]*ray[0]+Rt[3*i+1]*ray[1]+Rt[3*i+2]*ray[2]);if(z<opts.near||z>opts.far||!accept(x,y,Math.round(u),d,world)){counts.geometry++;continue;}points.push(...world);colors.push(rgbA[3*k],rgbA[3*k+1],rgbA[3*k+2]);uv.push(x,y,u,y);kinds.push(6);counts.added++;}
  result.points=new Float32Array(points);result.colors=new Uint8Array(colors);result.uv=new Float32Array(uv);result.pointKinds=new Uint8Array(kinds);result.completion={method:'seeded-slanted-window-plane-propagation',measured:priorCompletion?.measured??original,matched:priorCompletion?.matched||0,estimated:counts.added,counts,...m,output:undefined,priorMatching:priorCompletion,totalMs:performance.now()-start+(priorCompletion?.totalMs||0)};return result;
 }
 return {complete};
})();
