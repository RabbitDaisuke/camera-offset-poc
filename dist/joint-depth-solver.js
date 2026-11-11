'use strict';
// Fast Bilateral Solver formulation from OpenCV contrib / Barron & Poole.
// See OPENCV-FBS-LICENSE.txt. Added measured cross-view links share disparity
// and radiometric differences; unobserved outputs remain inferred depths.
self.JointDepthSolver=(()=>{
 // Fit a disparity plane to observed samples; the bilateral solve operates
 // on its residual. This preserves slanted surfaces rather than favoring a
 // constant disparity. It is a depth prior, not an extra observed match.
 function planeFit(samples){
  if(samples.length<6)return null;let weights=new Float64Array(samples.length).fill(1),plane;
  for(let iteration=0;iteration<6;iteration++){
   let sw=0,mx=0,my=0,md=0;for(let i=0;i<samples.length;i++){const [x,y,d]=samples[i],w=weights[i];sw+=w;mx+=w*x;my+=w*y;md+=w*d;}mx/=sw;my/=sw;md/=sw;
   let xx=0,xy=0,yy=0,xd=0,yd=0;for(let i=0;i<samples.length;i++){const p=samples[i],x=p[0]-mx,y=p[1]-my,d=p[2]-md,w=weights[i];xx+=w*x*x;xy+=w*x*y;yy+=w*y*y;xd+=w*x*d;yd+=w*y*d;}const determinant=xx*yy-xy*xy;if(determinant<1e-5)return null;
   const a=(xd*yy-yd*xy)/determinant,b=(yd*xx-xd*xy)/determinant;plane=[a,b,md-a*mx-b*my];
   for(let i=0;i<samples.length;i++){const p=samples[i];weights[i]=Math.min(1,2/Math.max(1e-8,Math.abs(a*p[0]+b*p[1]+plane[2]-p[2])));}
  }
  return Math.abs(plane[0])<.8&&Math.abs(plane[1])<2?plane:null;
 }
 function graph(cv,result,rgbs,masks,rect,opts){
  const {S,R,C0,f,baseline}=rect,N=S*S,spatial=opts.bilateralSpatial||8,color=opts.bilateralColor||12,lambda=opts.bilateralLambda??1;
  const index=new Int32Array(2*N);index.fill(-1);const lookup=new Map(),keys=[],coords=[],mass=[],scales=[1,256,65536,16777216,4294967296];
  for(let side=0;side<2;side++){
   const source=cv.matFromArray(S,S,cv.CV_8UC3,rgbs[side]),yuv=new cv.Mat();try{cv.cvtColor(source,yuv,cv.COLOR_RGB2YCrCb);const rgb=yuv.data;
    for(let y=0;y<S;y++)for(let x=0;x<S;x++){const k=y*S+x;if(masks[side][k]!==255)continue;const c=[Math.floor(x/spatial),Math.floor(y/spatial),Math.floor(rgb[3*k]/color),Math.floor(rgb[3*k+1]/color),Math.floor(rgb[3*k+2]/color)],key=c.reduce((v,q,j)=>v+q*scales[j],side*281474976710656);let v=lookup.get(key);if(v===undefined){v=keys.length;lookup.set(key,v);keys.push(key);coords.push(c);mass.push(0);}index[side*N+k]=v;mass[v]++;}
   }finally{source.delete();yuv.delete();}
  }
  const V=keys.length,neighbors=Array.from({length:V},()=>[]),parent=Int32Array.from({length:V},(_,i)=>i),find=i=>{while(parent[i]!==i){parent[i]=parent[parent[i]];i=parent[i];}return i;},join=(a,b)=>{a=find(a);b=find(b);if(a!==b)parent[a]=b;};
  for(let i=0;i<V;i++)for(let axis=0;axis<5;axis++)for(const sign of [-1,1]){if(coords[i][axis]+sign<0||coords[i][axis]+sign>255)continue;const j=lookup.get(keys[i]+sign*scales[axis]);if(j!==undefined){neighbors[i].push(j);join(i,j);}}
  let n=new Float64Array(V).fill(1),next=new Float64Array(V);for(let iter=0;iter<10;iter++){for(let i=0;i<V;i++){let blur=10*n[i];for(const j of neighbors[i])blur+=n[j];next[i]=Math.sqrt(n[i]*mass[i]/blur);}[n,next]=[next,n];}
  const weight=new Float64Array(V),rhs=new Float32Array(V*4),links=Array.from({length:V},()=>new Map()),observations=[];
  const sample=(rgb,x,y,c)=>{const ix=Math.floor(x),t=x-ix,k=(y*S+ix)*3;return rgb[k+c]*(1-t)+rgb[k+c+3]*t;};
  for(let i=0;i<result.points.length/3;i++){
   const v=[0,1,2].map(c=>result.points[3*i+c]-C0[c]),z=R[6]*v[0]+R[7]*v[1]+R[8]*v[2],d=f*baseline/z,x=result.uv[4*i],y=result.uv[4*i+1],u=x-d;if(u<0||u>=S-1)continue;
   const a=index[y*S+x],b=index[N+y*S+Math.round(u)],target=[d/S,...[0,1,2].map(c=>(sample(rgbs[1],u,y,c)-rgbs[0][3*(y*S+x)+c])/255)];
   observations.push({a,b,x,y,u,d});
   for(const q of [a,b])if(q>=0){weight[q]++;for(let c=0;c<4;c++)rhs[4*q+c]+=target[c];}
   if(a>=0&&b>=0){links[a].set(b,(links[a].get(b)||0)+1);links[b].set(a,(links[b].get(a)||0)+1);join(a,b);}
  }
  const componentSeeds=new Float64Array(V);for(let i=0;i<V;i++)componentSeeds[find(i)]+=weight[i];
  const planes=new Map();
  if(opts.bilateralPlanes){
   const groups=new Map();for(const o of observations){if(o.a<0&&o.b<0)continue;const root=find(o.a>=0?o.a:o.b);if(!groups.has(root))groups.set(root,[]);groups.get(root).push([o.x,o.y,o.d]);}
   for(const [root,samples]of groups){const plane=planeFit(samples);if(plane)planes.set(root,plane);}
  }
  const prior=(side,x,y,vertex)=>{const p=planes.get(find(vertex));return p?(p[0]*x+p[1]*y+p[2])/(side?1-p[0]:1):0;};
  for(const o of observations){if(o.a>=0)rhs[o.a*4]-=prior(0,o.x,o.y,o.a)/S;if(o.b>=0)rhs[o.b*4]-=prior(1,o.u,o.y,o.b)/S;}
  const diagonal=new Float32Array(V),offsets=new Uint32Array(V+1),columns=[],values=[],fixed=opts.bilateralFixed!==false,observed=new Float32Array(rhs.length);
  for(let i=0;i<V;i++)if(weight[i])for(let c=0;c<4;c++)observed[4*i+c]=rhs[4*i+c]/weight[i];
  for(let i=0;i<V;i++){
   offsets[i]=columns.length;
   if(fixed&&weight[i]){diagonal[i]=1;rhs.set(observed.subarray(4*i,4*i+4),4*i);continue;}
   let blur=10*n[i];for(const j of neighbors[i])blur+=n[j];const m=n[i]*blur;
   diagonal[i]=lambda*(m-10*n[i]*n[i])+weight[i]+1e-6*mass[i];
   const edge=(j,w)=>{if(fixed&&weight[j]){for(let c=0;c<4;c++)rhs[4*i+c]-=w*observed[4*j+c];}else{columns.push(j);values.push(w);}};
   for(const j of neighbors[i])edge(j,-lambda*n[i]*n[j]);
   for(const [j,count]of links[i]){const w=8*count;diagonal[i]+=w;edge(j,-w);}
  }
  offsets[V]=columns.length;
  return {V,index,diagonal,offsets,columns:new Uint32Array(columns),values:new Float32Array(values),rhs,supported:Uint8Array.from({length:V},(_,i)=>componentSeeds[find(i)]>=6?1:0),lambda,spatial,color,fixed,prior,planes:planes.size};
 }
 const shader=`
 struct Params {vertices:u32,groups:u32,pad0:u32,pad1:u32};
 struct State {x:vec4<f32>,r:vec4<f32>,z:vec4<f32>,direction:vec4<f32>,product:vec4<f32>};
 struct Pair {rz:vec4<f32>,pq:vec4<f32>};
 struct Scalar {alpha:vec4<f32>,beta:vec4<f32>,rz:vec4<f32>,pad:vec4<f32>};
 @group(0) @binding(0) var<uniform> p:Params;
 @group(0) @binding(1) var<storage,read> diagonal:array<f32>;
 @group(0) @binding(2) var<storage,read> offsets:array<u32>;
 @group(0) @binding(3) var<storage,read> columns:array<u32>;
 @group(0) @binding(4) var<storage,read> values:array<f32>;
 @group(0) @binding(5) var<storage,read_write> states:array<State>;
 @group(0) @binding(6) var<storage,read_write> partial:array<Pair>;
 @group(0) @binding(7) var<storage,read_write> scalar:Scalar;
 var<workgroup> sumRZ:array<vec4<f32>,128>;
 var<workgroup> sumPQ:array<vec4<f32>,128>;
 @compute @workgroup_size(128) fn multiply(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p.vertices){return;}var a=diagonal[i]*states[i].direction;for(var k=offsets[i];k<offsets[i+1u];k++){a+=values[k]*states[columns[k]].direction;}states[i].product=a;}
 @compute @workgroup_size(128) fn reduce(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>){
  let i=gid.x;let t=lid.x;sumRZ[t]=vec4<f32>(0);sumPQ[t]=vec4<f32>(0);if(i<p.vertices){sumRZ[t]=states[i].r*states[i].z;sumPQ[t]=states[i].direction*states[i].product;}workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){if(t<stride){sumRZ[t]+=sumRZ[t+stride];sumPQ[t]+=sumPQ[t+stride];}workgroupBarrier();}if(t==0u){partial[group.x].rz=sumRZ[0];partial[group.x].pq=sumPQ[0];}
 }
 @compute @workgroup_size(1) fn alpha(){var rz=vec4<f32>(0);var pq=vec4<f32>(0);for(var i=0u;i<p.groups;i++){rz+=partial[i].rz;pq+=partial[i].pq;}scalar.alpha=select(vec4<f32>(0),rz/max(pq,vec4<f32>(1e-20)),pq>vec4<f32>(1e-18));scalar.rz=rz;}
 @compute @workgroup_size(128) fn update(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p.vertices){return;}states[i].x+=scalar.alpha*states[i].direction;states[i].r-=scalar.alpha*states[i].product;states[i].z=states[i].r/max(diagonal[i],1e-6);}
 @compute @workgroup_size(1) fn beta(){var rz=vec4<f32>(0);for(var i=0u;i<p.groups;i++){rz+=partial[i].rz;}scalar.beta=select(vec4<f32>(0),rz/max(scalar.rz,vec4<f32>(1e-20)),scalar.rz>vec4<f32>(1e-18));}
 @compute @workgroup_size(128) fn direction(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p.vertices){return;}states[i].direction=states[i].z+scalar.beta*states[i].direction;}
 `;
 class Solver{
  constructor(device){this.device=device;this.buffers=[];}
  async init(){const d=this.device,module=d.createShaderModule({label:'Joint bilateral least-squares depth solver',code:shader}),info=await module.getCompilationInfo(),errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(x=>x.message).join('\n'));
   const layout=d.createBindGroupLayout({entries:Array.from({length:8},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding<5?'read-only-storage':'storage'}}))});this.layout=layout;const pipelineLayout=d.createPipelineLayout({bindGroupLayouts:[layout]});this.pipelines={};for(const entryPoint of ['multiply','reduce','alpha','update','beta','direction'])this.pipelines[entryPoint]=await d.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}});
  }
  async solve(g,iterations){
   const start=performance.now(),d=this.device;for(const b of this.buffers)b.destroy();this.buffers=[];const buffer=(bytes,usage,data)=>{const b=d.createBuffer({size:Math.max(4,bytes),usage});this.buffers.push(b);if(data)d.queue.writeBuffer(b,0,data);return b;};
   const G=Math.ceil(g.V/128),state=new Float32Array(g.V*20);for(let i=0;i<g.V;i++)for(let c=0;c<4;c++){const b=g.rhs[4*i+c],z=b/Math.max(g.diagonal[i],1e-6);state[20*i+4+c]=b;state[20*i+8+c]=z;state[20*i+12+c]=z;}
   const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,inputs=[buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,new Uint32Array([g.V,G,0,0])),buffer(g.diagonal.byteLength,storage,g.diagonal),buffer(g.offsets.byteLength,storage,g.offsets),buffer(g.columns.byteLength,storage,g.columns),buffer(g.values.byteLength,storage,g.values),buffer(state.byteLength,storage|GPUBufferUsage.COPY_SRC,state),buffer(G*32,storage),buffer(64,storage)],read=buffer(state.byteLength,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),bind=d.createBindGroup({layout:this.layout,entries:inputs.map((buffer,binding)=>({binding,resource:{buffer}}))});
   d.pushErrorScope('validation');const encoder=d.createCommandEncoder();const dispatch=(name,count=G)=>{const pass=encoder.beginComputePass();pass.setPipeline(this.pipelines[name]);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(count);pass.end();};
   for(let i=0;i<iterations;i++){dispatch('multiply');dispatch('reduce');dispatch('alpha',1);dispatch('update');dispatch('reduce');dispatch('beta',1);dispatch('direction');}
   encoder.copyBufferToBuffer(inputs[5],0,read,0,state.byteLength);const submitted=performance.now();d.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const solved=new Float32Array(read.getMappedRange().slice(0));read.unmap();const error=await d.popErrorScope();if(error)throw Error(error.message);let residual=0,rhsNorm=0;for(let i=0;i<g.V;i++){residual+=solved[20*i+4]**2;rhsNorm+=g.rhs[4*i]**2;}return {solved,iterations,residual:Math.sqrt(residual/Math.max(rhsNorm,1e-20)),prepareMs:submitted-start,computeReadbackMs:performance.now()-submitted};
  }
 }
 let solver;
 async function complete(cv,gpu,result,rgbA,rgbB,fa,fb,rect,opts,accept){
  const start=performance.now(),{S,f,baseline,Rt,C0,cx,cy}=rect,N=S*S,g=graph(cv,result,[rgbA,rgbB],opts.scope==='scene'?rect.valid:[fa.raw,fb.raw],rect,opts),built=performance.now();if(!g.V)return result;if(!solver){solver=new Solver(gpu.device);await solver.init();}const answer=await solver.solve(g,opts.bilateralIterations||512),v=answer.solved;
  const map=new Float32Array(2*N*4);for(let k=0;k<2*N;k++){const i=g.index[k];if(i<0||!g.supported[i])continue;map[k*4]=v[i*20]*S+g.prior(k>=N?1:0,k%S,Math.floor(k/S)%S,i);for(let c=1;c<4;c++)map[k*4+c]=v[i*20+c]*255;}
  const existing=new Set(result.accepted),points=Array.from(result.points),colors=Array.from(result.colors),uv=Array.from(result.uv),kinds=new Array(points.length/3).fill(0),counts={supported:0,leftRight:0,photometric:0,background:0,geometry:0,added:0},sample=(rgb,x,y,c)=>{const ix=Math.floor(x),t=x-ix,k=(y*S+ix)*3;return rgb[k+c]*(1-t)+rgb[k+c+3]*t;};
  const inspection=opts.inspectRegion?{mask:{},stages:{},pixels:[]}:null,inside=(x,y)=>{const p=opts.inspectRegion;let b=false;for(let i=0,j=p.length-1;i<p.length;j=i++)if((p[i][1]>y)!==(p[j][1]>y)&&x<(p[j][0]-p[i][0])*(y-p[i][1])/(p[j][1]-p[i][1])+p[i][0])b=!b;return b;};
  const record=(x,y,stage,d,back,error)=>{if(inspection&&inside(x,y)){inspection.stages[stage]=(inspection.stages[stage]||0)+1;inspection.pixels.push([x,y,stage,d,back,error]);}};
  for(let y=2;y<S-2;y++)for(let x=2;x<S-2;x++){
   const k=y*S+x;if(inspection&&inside(x,y))inspection.mask[fa.raw[k]]=(inspection.mask[fa.raw[k]]||0)+1;if(existing.has(k)){record(x,y,'measured',map[k*4],null,null);continue;}const d=map[k*4],u=x-d;
   if(d<=0||u<2||u>=S-3){record(x,y,'no-support',d,null,null);continue;}counts.supported++;const q=N+y*S+Math.round(u),back=map[q*4];if(back<=0||Math.abs(d-back)>(opts.bilateralLR||3)){counts.leftRight++;record(x,y,'left-right',d,back,null);continue;}
   // Check RGB radiometry inferred from actual matched samples on both sides.
   // This is an inference check, not an independent correspondence measurement.
   let error=0;for(let c=0;c<3;c++){let a=0,b=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){a+=rgbA[((y+dy)*S+x+dx)*3+c]/9;b+=sample(rgbB,u+dx,y+dy,c)/9;}const expected=(map[k*4+c+1]+map[q*4+c+1])*.5;error+=(b-a-expected)**2;}
   error=Math.sqrt(error/3);if(error>18){counts.photometric++;record(x,y,'photometric',d,back,error);continue;}
   const z=f*baseline/d,ray=[(x-cx)*z/f,(y-cy)*z/f,z],world=[0,1,2].map(i=>C0[i]+Rt[i*3]*ray[0]+Rt[i*3+1]*ray[1]+Rt[i*3+2]*ray[2]);if(z<opts.near||z>opts.far||(opts.scope!=='scene'&&world[2]<.03)){counts.geometry++;record(x,y,'geometry',d,back,error);continue;}
   if(!accept(x,y,Math.round(u),d,world)){counts.background++;record(x,y,'background',d,back,error);continue;}
   points.push(...world);colors.push(rgbA[k*3],rgbA[k*3+1],rgbA[k*3+2]);uv.push(x,y,u,y);kinds.push(4);counts.added++;record(x,y,'estimated',d,back,error);
  }
  result.points=new Float32Array(points);result.colors=new Uint8Array(colors);result.uv=new Float32Array(uv);result.pointKinds=new Uint8Array(kinds);
  if(result.labels){const labels=new Uint8Array(kinds.length),sources=new Uint8Array(kinds.length);labels.set(result.labels);sources.set(result.classificationSource);result.labels=labels;result.classificationSource=sources;result.classification.unknown+=counts.added;}
  result.completion={method:'joint-fast-bilateral-least-squares',measured:kinds.length-counts.added,estimated:counts.added,vertices:g.V,edges:g.columns.length,spatial:g.spatial,color:g.color,lambda:g.lambda,fixed:g.fixed,planePriors:g.planes,counts,graphMs:built-start,...answer,solved:undefined,totalMs:performance.now()-start,inspection};return result;
 }
 return {complete};
})();
