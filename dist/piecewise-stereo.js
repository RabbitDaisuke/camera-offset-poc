'use strict';
// ELAS-style piecewise disparity prior followed by real-image matching.
// The support points are existing validated stereo observations. Outputs
// influenced by this prior remain depth estimates, not independent matches.
self.PiecewiseStereo=(()=>{
 function growRegions(cv,samples,models,rgbs,masks,S,radius){
  const started=performance.now(),labels=[],distances=[];
  if(typeof cv.watershed!=='function')throw Error('OpenCV watershed is unavailable');
  const assignments=samples.map(([x,y,d])=>{let best=1.5,id=-1;models.forEach((m,i)=>{const e=Math.abs(m.plane[0]*x+m.plane[1]*y+m.plane[2]-d);if(e<best){best=e;id=i;}});return id;});
  for(let side=0;side<2;side++){
   const image=cv.matFromArray(S,S,cv.CV_8UC3,rgbs[side]),markers=new cv.Mat(S,S,cv.CV_32SC1),seedMasks=models.map(()=>new Uint8Array(S*S).fill(255));markers.data32S.fill(0);
   try{
    for(let k=0;k<S*S;k++)if(masks[side][k]!==255)markers.data32S[k]=1;
    samples.forEach(([x,y,d],i)=>{const u=Math.round(side?x-d:x);if(u<0||u>=S)return;const k=y*S+u,id=assignments[i];markers.data32S[k]=id+2;if(id>=0)seedMasks[id][k]=0;});
    cv.watershed(image,markers);labels.push(new Int32Array(markers.data32S));
    distances.push(seedMasks.map(data=>{const seeds=cv.matFromArray(S,S,cv.CV_8UC1,data),out=new cv.Mat();try{cv.distanceTransform(seeds,out,cv.DIST_L2,3);return new Float32Array(out.data32F);}finally{seeds.delete();out.delete();}}));
   }finally{image.delete();markers.delete();}
  }
  return {labels,distances,radius,ms:performance.now()-started};
 }
 function regionalPriors(cv,result,rgbs,S,rect,opts,masks){
  const validDisparity=new Float32Array(S*S);validDisparity.fill(NaN);const samples=[];
  for(let i=0;i<result.points.length/3;i++){const p=[0,1,2].map(c=>result.points[3*i+c]-rect.C0[c]),z=rect.R[6]*p[0]+rect.R[7]*p[1]+rect.R[8]*p[2],x=result.uv[4*i],y=result.uv[4*i+1],d=rect.f*rect.baseline/z;validDisparity[y*S+x]=d;samples.push([x,y,d]);}
  const fitted=BackgroundPlanes.build(cv,{uv:result.uv,validDisparity},S,opts.piecewisePlaneFit||{}),models=fitted.report.models,output=new Float32Array(2*S*S*4),neighbors=new Float32Array(2*S*S*8),best=new Float32Array(2*S*S).fill(Infinity),count=[0,0],constraints=new Map();
  const grown=opts.piecewiseGrowth?growRegions(cv,samples,models,rgbs,masks,S,opts.piecewiseGrowth):null,inside=(h,x,y)=>h.every((a,i)=>{const b=h[(i+1)%h.length];return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])>=-1e-4;}),sample=(side,x,y,c)=>{const ix=Math.floor(x),t=x-ix,k=(y*S+ix)*3;return rgbs[side][k+c]*(1-t)+rgbs[side][k+c+3]*t;};
  const supported=(side,id,h,x,y)=>{if(inside(h,x,y))return true;if(!grown)return false;const k=y*S+Math.round(x),other=grown.labels[side][k]-2;if(other<0||grown.distances[side][id][k]>grown.radius)return false;const disparity=p=>(p[0]*x+p[1]*y+p[2])/(side?1-p[0]:1);return Math.abs(disparity(models[id].plane)-disparity(models[other].plane))<=3;};
  const candidates=new Uint32Array(2*S*S),modelData=new Float32Array(Math.max(1,models.length)*8);if(opts.piecewiseGpuPriors&&models.length>32)throw Error('GPU plane priors support at most 32 planes');
  for(const [x,y,d]of samples)for(let side=0;side<2;side++){const u=side?x-d:x,key=side*S*S+Math.floor(y/4)*S+Math.floor(u/4);if(!constraints.has(key))constraints.set(key,[]);constraints.get(key).push(d);}
  for(let id=0;id<models.length;id++){const m=models[id],p=m.plane,bias=[[],[],[]];for(const [x,y,d]of samples){const u=x-d;if(Math.abs(p[0]*x+p[1]*y+p[2]-d)>1.5||u<1||u>=S-2)continue;for(let c=0;c<3;c++)bias[c].push(sample(1,u,y,c)-rgbs[0][(y*S+x)*3+c]);}const delta=bias.map(a=>{a.sort((x,y)=>x-y);return a[Math.floor(a.length/2)]||0;});
   modelData.set([...p,m.rms,...delta,Math.log(m.support)],id*8);
   for(let side=0;side<2;side++){const h=side?m.rightHull:m.leftHull,other=side?m.leftHull:m.rightHull,margin=grown?.radius||0,x0=Math.max(3,Math.ceil(Math.min(...h.map(p=>p[0])))-margin),x1=Math.min(S-4,Math.floor(Math.max(...h.map(p=>p[0])))+margin),y0=Math.max(3,Math.ceil(Math.min(...h.map(p=>p[1])))-margin),y1=Math.min(S-4,Math.floor(Math.max(...h.map(p=>p[1])))+margin);
    for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){if(!supported(side,id,h,x,y))continue;const d=(p[0]*x+p[1]*y+p[2])/(side?1-p[0]:1),u=x+(side?d:-d);if(u<3||u>=S-4||!supported(1-side,id,other,u,y))continue;const near=constraints.get(side*S*S+Math.floor(y/4)*S+Math.floor(x/4));if(near&&near.some(v=>Math.abs(v-d)>3))continue;const k=side*S*S+y*S+x;if(opts.piecewiseGpuPriors){if(!candidates[k])count[side]++;candidates[k]|=1<<id;continue;}let error=0;for(let c=0;c<3;c++){let difference=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)difference+=(sample(1-side,u+dx,y+dy,c)-sample(side,x+dx,y+dy,c))/9;if(side)difference=-difference;error+=(difference-delta[c])**2;}error=Math.sqrt(error/3);if(error>18)continue;const rank=error+m.rms-(opts.piecewisePlaneBias||0)*Math.log(m.support);if(rank>=best[k])continue;if(!output[k*4])count[side]++;best[k]=rank;output.set([d,...delta],k*4);}
   }
  }
  return {output,neighbors,candidates,modelData,gpuModels:opts.piecewiseGpuPriors?models.length:0,counts:count.map(covered=>({support:samples.length,models:models.length,...(opts.piecewiseGpuPriors?{candidatePixels:covered}:{covered}),growthRadius:grown?.radius||0,growthMs:grown?.ms||0})),models};
 }
 const plane=(p,q,r)=>{const x=q[0]-p[0],y=q[1]-p[1],u=r[0]-p[0],v=r[1]-p[1],den=x*v-y*u;if(Math.abs(den)<1e-6)return null;const a=((q[2]-p[2])*v-(r[2]-p[2])*y)/den,b=(x*(r[2]-p[2])-u*(q[2]-p[2]))/den;return [a,b,p[2]-a*p[0]-b*p[1]];};
 async function priors(result,rgb,S,rect){
  const {default:Delaunator}=await import('./vendor/delaunator.js'),samples=[];
  for(let i=0;i<result.points.length/3;i++){const p=[0,1,2].map(c=>result.points[3*i+c]-rect.C0[c]),z=rect.R[6]*p[0]+rect.R[7]*p[1]+rect.R[8]*p[2],x=result.uv[4*i],y=result.uv[4*i+1],d=rect.f*rect.baseline/z,u=x-d;if(u<1||u>=S-2)continue;const j=(y*S+Math.floor(u))*3,t=u-Math.floor(u),difference=[0,1,2].map(c=>rgb[1][j+c]*(1-t)+rgb[1][j+c+3]*t-rgb[0][(y*S+x)*3+c]);samples.push({x,y,u,d,difference});}
  const output=new Float32Array(2*S*S*4),neighbors=new Float32Array(2*S*S*8),counts=[];
  for(let side=0;side<2;side++){
   // Keep one observed point per small cell. Discontinuities are checked
   // against every original sample before a prior triangle is used.
   const cells=new Map();for(const p of samples){const x=side?p.u:p.x,key=Math.floor(x/5)+Math.floor(p.y/5)*S,rank=(x%5-2.5)**2+(p.y%5-2.5)**2;if(!cells.has(key)||rank<cells.get(key).rank)cells.set(key,{p,rank});}
   const support=[...cells.values()].map(v=>v.p),coords=new Float64Array(support.length*2);support.forEach((p,i)=>coords.set([side?p.u:p.x,p.y],i*2));if(support.length<3){counts.push({support:support.length,triangles:0,covered:0});continue;}
   const mesh=new Delaunator(coords),seedGrid=new Map();for(const p of samples){const x=side?p.u:p.x,key=Math.floor(x/20)+Math.floor(p.y/20)*S;if(!seedGrid.has(key))seedGrid.set(key,[]);seedGrid.get(key).push(p);}
   let triangles=0,covered=0;
   for(let ti=0;ti<mesh.triangles.length;ti+=3){const ps=[0,1,2].map(j=>support[mesh.triangles[ti+j]]),v=ps.map(p=>[side?p.u:p.x,p.y,p.d]),p=plane(...v);if(!p||Math.abs(p[0])>2||Math.abs(p[1])>2)continue;
    const area=(v[1][0]-v[0][0])*(v[2][1]-v[0][1])-(v[1][1]-v[0][1])*(v[2][0]-v[0][0]),bary=(x,y)=>{const b=((x-v[0][0])*(v[2][1]-v[0][1])-(y-v[0][1])*(v[2][0]-v[0][0]))/area,c=((v[1][0]-v[0][0])*(y-v[0][1])-(v[1][1]-v[0][1])*(x-v[0][0]))/area;return [1-b-c,b,c];};
    const x0=Math.max(2,Math.ceil(Math.min(...v.map(p=>p[0])))),x1=Math.min(S-3,Math.floor(Math.max(...v.map(p=>p[0])))),y0=Math.max(2,Math.ceil(Math.min(...v.map(p=>p[1])))),y1=Math.min(S-3,Math.floor(Math.max(...v.map(p=>p[1]))));
    if(x1-x0>256||y1-y0>256)continue;let conflict=false;
    for(let gy=Math.floor(y0/20);gy<=Math.floor(y1/20)&&!conflict;gy++)for(let gx=Math.floor(x0/20);gx<=Math.floor(x1/20)&&!conflict;gx++)for(const q of seedGrid.get(gx+gy*S)||[]){const x=side?q.u:q.x;if(bary(x,q.y).some(w=>w<-.001))continue;if(Math.abs(p[0]*x+p[1]*q.y+p[2]-q.d)>3){conflict=true;break;}}
    if(conflict)continue;triangles++;
    for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const w=bary(x,y);if(w.some(v=>v<-.001))continue;const k=(side*S*S+y*S+x)*4;output[k]=p[0]*x+p[1]*y+p[2];for(let c=0;c<3;c++)output[k+c+1]=w.reduce((s,v,j)=>s+v*ps[j].difference[c],0);covered++;}
   }
   const supportGrid=new Map();for(const p of support){const x=side?p.u:p.x,key=Math.floor(x/20)+Math.floor(p.y/20)*S;if(!supportGrid.has(key))supportGrid.set(key,[]);supportGrid.get(key).push(p);}
   for(let by=0;by<S;by+=20)for(let bx=0;bx<S;bx+=20){const candidates=[];for(let gy=by/20-1;gy<=by/20+1;gy++)for(let gx=bx/20-1;gx<=bx/20+1;gx++)candidates.push(...supportGrid.get(gx+gy*S)||[]);candidates.sort((a,b)=>((side?a.u:a.x)-bx-10)**2+(a.y-by-10)**2-(((side?b.u:b.x)-bx-10)**2+(b.y-by-10)**2));const values=[];for(const p of candidates){if(values.every(d=>Math.abs(d-p.d)>2))values.push(p.d);if(values.length>=8)break;}for(let y=by;y<Math.min(S,by+20);y++)for(let x=bx;x<Math.min(S,bx+20);x++)neighbors.set(values,(side*S*S+y*S+x)*8);}
   counts.push({support:support.length,triangles,covered});
  }
  return {output,neighbors,counts};
 }
 const shader=`
 struct Params {size:u32,pixels:u32,priorWeight:f32,modelCount:u32,planeBias:f32,pad1:u32,pad2:u32,pad3:u32};
 @group(0) @binding(0) var<uniform> p:Params;
 @group(0) @binding(1) var<storage,read> images:array<u32>;
 @group(0) @binding(2) var<storage,read> masks:array<u32>;
 @group(0) @binding(3) var<storage,read_write> prior:array<vec4<f32>>;
 @group(0) @binding(4) var<storage,read> neighbor:array<f32>;
 @group(0) @binding(5) var<storage,read_write> gradients:array<vec4<f32>>;
 @group(0) @binding(6) var<storage,read_write> output:array<vec4<f32>>;
 @group(0) @binding(7) var<storage,read> models:array<vec4<f32>>;
 @group(0) @binding(8) var<storage,read> candidates:array<u32>;
 fn rgb(x:i32,y:i32,side:i32)->vec3<f32>{let v=images[(side*i32(p.size)+y)*i32(p.size)+x];return vec3<f32>(f32(v&255u),f32((v>>8u)&255u),f32((v>>16u)&255u));}
 fn sample(x:f32,y:i32,side:i32)->vec3<f32>{let ix=i32(floor(x));return mix(rgb(ix,y,side),rgb(ix+1,y,side),fract(x));}
 @compute @workgroup_size(64) fn selectPrior(@builtin(global_invocation_id) gid:vec3<u32>){let k=gid.x;if(k>=2u*p.pixels){return;}let bits=candidates[k];if(bits==0u){return;}let x=i32(k%p.size);let y=i32((k/p.size)%p.size);let side=i32(k/p.pixels);var best=1e9;for(var i=0u;i<p.modelCount;i++){if((bits&(1u<<i))==0u){continue;}let plane=models[2u*i];let bias=models[2u*i+1u];let d=(plane.x*f32(x)+plane.y*f32(y)+plane.z)/select(1.,1.-plane.x,side==1);let u=f32(x)+select(-d,d,side==1);if(u<3.||u>=f32(p.size)-4.){continue;}var difference=vec3<f32>(0);for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){difference+=(sample(u+f32(dx),y+dy,1-side)-rgb(x+dx,y+dy,side))/9.;}}if(side==1){difference=-difference;}let delta=difference-bias.xyz;let error=sqrt(dot(delta,delta)/3.);if(error>18.){continue;}let rank=error+plane.w-p.planeBias*bias.w;if(rank<best){best=rank;prior[k]=vec4<f32>(d,bias.xyz);}}}
 @compute @workgroup_size(64) fn gradient(@builtin(global_invocation_id) gid:vec3<u32>){let k=gid.x;if(k>=2u*p.pixels){return;}let x=i32(k%p.size);let y=i32((k/p.size)%p.size);let side=i32(k/p.pixels);if(x<1||y<1||x>=i32(p.size)-1||y>=i32(p.size)-1){return;}let gx=(rgb(x+1,y-1,side)+2.*rgb(x+1,y,side)+rgb(x+1,y+1,side)-rgb(x-1,y-1,side)-2.*rgb(x-1,y,side)-rgb(x-1,y+1,side))/8.;let gy=(rgb(x-1,y+1,side)+2.*rgb(x,y+1,side)+rgb(x+1,y+1,side)-rgb(x-1,y-1,side)-2.*rgb(x,y-1,side)-rgb(x+1,y-1,side))/8.;gradients[2u*k]=vec4<f32>(gx,0);gradients[2u*k+1u]=vec4<f32>(gy,0);}
 fn cost(x:i32,y:i32,side:i32,d:f32,mu:f32)->f32{let u=f32(x)+select(-d,d,side==1);if(u<3.||u>=f32(p.size)-4.){return 1e9;}var error=0.;for(var dy=-2;dy<=2;dy++){for(var dx=-2;dx<=2;dx++){let a=(side*i32(p.size)+y+dy)*i32(p.size)+x+dx;let bx=i32(floor(u))+dx;let b=((1-side)*i32(p.size)+y+dy)*i32(p.size)+bx;for(var axis=0;axis<2;axis++){let g=mix(gradients[2*b+axis].xyz,gradients[2*(b+1)+axis].xyz,fract(u));error+=dot(abs(gradients[2*a+axis].xyz-g),vec3<f32>(1./3.));}}}return .03*error-p.priorWeight*log(15.+exp(-(d-mu)*(d-mu)/18.));}
 @compute @workgroup_size(64) fn correlate(@builtin(global_invocation_id) gid:vec3<u32>){let k=gid.x;if(k>=2u*p.pixels){return;}output[k]=vec4<f32>(0);let x=i32(k%p.size);let y=i32((k/p.size)%p.size);let side=i32(k/p.pixels);let mu=prior[k].x;if(x<3||y<3||x>=i32(p.size)-3||y>=i32(p.size)-3||mu<=0.||masks[k]!=255u){return;}var best=1e9;var chosen=0.;for(var j=0u;j<27u;j++){var d=mu+f32(i32(j)-9);if(j>=19u){d=neighbor[k*8u+j-19u];}if(d<=0.){continue;}let e=cost(x,y,side,d,mu);if(e<best){best=e;chosen=d;}}if(chosen<=0.){return;}let lo=cost(x,y,side,chosen-.5,mu);let hi=cost(x,y,side,chosen+.5,mu);let den=lo-2.*best+hi;if(den>1e-5){chosen+=clamp(.25*(lo-hi)/den,-.5,.5);}let u=f32(x)+select(-chosen,chosen,side==1);if(u<3.||u>=f32(p.size)-4.){return;}var difference=vec3<f32>(0);for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){difference+=sample(u+f32(dx),y+dy,1-side)-rgb(x+dx,y+dy,side);}}difference/=9.;if(side==1){difference=-difference;}let diff=difference-prior[k].yzw;let error=sqrt(dot(diff,diff)/3.);output[k]=vec4<f32>(chosen,error,best,mu);}
 `;
 class Matcher{
  constructor(device){this.device=device;this.buffers=[];}
  async init(){
   const d=this.device,module=d.createShaderModule({label:'Piecewise disparity prior RGB matching',code:shader}),info=await module.getCompilationInfo(),errors=info.messages.filter(x=>x.type==='error');
   if(errors.length)throw Error(errors.map(x=>x.message).join('\n'));
   this.layout=d.createBindGroupLayout({entries:Array.from({length:9},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':[3,5,6].includes(binding)?'storage':'read-only-storage'}}))});
   const layout=d.createPipelineLayout({bindGroupLayouts:[this.layout]});this.pipelines={};
   for(const entryPoint of ['selectPrior','gradient','correlate'])this.pipelines[entryPoint]=await d.createComputePipelineAsync({layout,compute:{module,entryPoint}});
  }
  async compute(rgb,masks,prior,S,opts){
   const start=performance.now(),d=this.device,N=S*S;for(const b of this.buffers)b.destroy();this.buffers=[];
   const buffer=(bytes,usage,data)=>{const b=d.createBuffer({size:Math.max(4,bytes),usage});this.buffers.push(b);if(data)d.queue.writeBuffer(b,0,data);return b;};
   const pixels=new Uint32Array(2*N),valid=new Uint32Array(2*N);
   for(let side=0;side<2;side++)for(let i=0;i<N;i++){pixels[side*N+i]=rgb[side][i*3]|rgb[side][i*3+1]<<8|rgb[side][i*3+2]<<16;valid[side*N+i]=masks[side][i];}
   const uniform=new Uint32Array(8),floats=new Float32Array(uniform.buffer);uniform.set([S,N,0,prior.gpuModels||0]);floats[2]=opts.piecewisePriorWeight??1;floats[4]=opts.piecewisePlaneBias||0;
   const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,models=prior.modelData||new Float32Array(8),candidates=prior.candidates||new Uint32Array(1);
   const inputs=[buffer(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,uniform),buffer(pixels.byteLength,storage,pixels),buffer(valid.byteLength,storage,valid),buffer(prior.output.byteLength,storage,prior.output),buffer(prior.neighbors.byteLength,storage,prior.neighbors),buffer(2*N*32,storage),buffer(2*N*16,storage|GPUBufferUsage.COPY_SRC),buffer(models.byteLength,storage,models),buffer(candidates.byteLength,storage,candidates)],read=buffer(2*N*16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
   d.pushErrorScope('validation');
   const group=d.createBindGroup({layout:this.layout,entries:inputs.map((buffer,binding)=>({binding,resource:{buffer}}))}),encoder=d.createCommandEncoder();
   for(const name of [...(prior.gpuModels?['selectPrior']:[]),'gradient','correlate']){const pass=encoder.beginComputePass();pass.setPipeline(this.pipelines[name]);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(2*N/64));pass.end();}
   encoder.copyBufferToBuffer(inputs[6],0,read,0,2*N*16);const submitted=performance.now();d.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();const error=await d.popErrorScope();if(error)throw Error(error.message);
   return {values,priorBackend:prior.gpuModels?'webgpu':'cpu',prepareMs:submitted-start,computeReadbackMs:performance.now()-submitted};
  }
 }
 let matcher;
 async function complete(cv,gpu,result,rgbA,rgbB,fa,fb,rect,opts,accept){const start=performance.now(),{S,f,baseline,Rt,C0,cx,cy}=rect,N=S*S,prior=opts.piecewisePlanes?regionalPriors(cv,result,[rgbA,rgbB],S,rect,opts,opts.scope==='scene'?rect.valid:[fa.raw,fb.raw]):await priors(result,[rgbA,rgbB],S,rect),built=performance.now();if(!matcher){matcher=new Matcher(gpu.device);await matcher.init();}const matched=await matcher.compute([rgbA,rgbB],opts.scope==='scene'?rect.valid:[fa.raw,fb.raw],prior,S,opts),v=matched.values,existing=new Set(result.accepted),points=Array.from(result.points),colors=Array.from(result.colors),uv=Array.from(result.uv),kinds=new Array(points.length/3).fill(0),counts={leftRight:0,photometric:0,geometry:0,background:0,added:0};
  for(let y=3;y<S-3;y++)for(let x=3;x<S-3;x++){const k=y*S+x,d=v[k*4],u=x-d;if(existing.has(k)||d<=0||u<3||u>=S-4)continue;const q=N+y*S+Math.round(u);if(v[q*4]<=0||Math.abs(d-v[q*4])>3){counts.leftRight++;continue;}if(v[k*4+1]>18||v[q*4+1]>18){counts.photometric++;continue;}const z=f*baseline/d,ray=[(x-cx)*z/f,(y-cy)*z/f,z],world=[0,1,2].map(i=>C0[i]+Rt[3*i]*ray[0]+Rt[3*i+1]*ray[1]+Rt[3*i+2]*ray[2]);if(z<opts.near||z>opts.far||(opts.scope!=='scene'&&world[2]<.03)){counts.geometry++;continue;}if(!accept(x,y,Math.round(u),d,world)){counts.background++;continue;}points.push(...world);colors.push(rgbA[3*k],rgbA[3*k+1],rgbA[3*k+2]);uv.push(x,y,u,y);kinds.push(5);counts.added++;}
  result.points=new Float32Array(points);result.colors=new Uint8Array(colors);result.uv=new Float32Array(uv);result.pointKinds=new Uint8Array(kinds);if(result.labels){const labels=new Uint8Array(kinds.length),source=new Uint8Array(kinds.length);labels.set(result.labels);source.set(result.classificationSource);result.labels=labels;result.classificationSource=source;result.classification.unknown+=counts.added;}result.completion={method:opts.piecewisePlanes?'ransac-plane-prior-rgb-gradient-stereo':'piecewise-prior-rgb-gradient-stereo',measured:kinds.length-counts.added,estimated:counts.added,counts,prior:prior.counts,models:prior.models,planeBias:opts.piecewisePlaneBias||0,priorWeight:opts.piecewisePriorWeight??1,priorMs:built-start,...matched,values:undefined,totalMs:performance.now()-start};return result;
 }
 return {complete};
})();
