/* WebGPU execution of OpenCV 4.13.0 RGB StereoSGBM MODE_SGBM_3WAY.
 * Cost metric, four overlapping stripes, all three paths, full disparity
 * range, uniqueness and subpixel rules are retained. Linear postprocessing
 * remains in OpenCV. See dist/native/OpenCV-LICENSE.txt for OpenCV's license.
 */
(function(root){
'use strict';
const common=`
struct Params {w:i32,h:i32,pad:i32,pw:i32,vw:i32,minx:i32,nd:i32,md:i32,ys:i32,rows:i32,side:i32,reserved:i32,radius:i32,p1:i32,p2:i32,unused:i32};
struct Feature {g:u32,c:u32,gl:u32,cl:u32,gh:u32,ch:u32};
struct Winner {disparity:i32,integer:i32,cost:i32};
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> pixels:array<u32>;
@group(0) @binding(2) var<storage,read_write> features:array<Feature>;
@group(0) @binding(3) var<storage,read_write> cost:array<i32>;
@group(0) @binding(4) var<storage,read_write> temp:array<i32>;
@group(0) @binding(5) var<storage,read_write> winners:array<Winner>;
fn sat(v:i32)->i32{return clamp(v,-32768,32767);}
fn byte(v:u32,c:u32)->i32{return i32((v>>(8u*c))&255u);}
fn packed(v:vec3<i32>)->u32{return u32(v.x)|(u32(v.y)<<8u)|(u32(v.z)<<16u);}
fn rgb(x:i32,y:i32,s:i32)->vec3<i32>{
 if(x<p.pad||x>=p.pad+p.w){return vec3<i32>(0);}
 let v=pixels[(s*p.h+clamp(y,0,p.h-1))*p.w+x-p.pad];
 return vec3<i32>(byte(v,0u),byte(v,1u),byte(v,2u));
}
fn featureAt(x:i32,y:i32,s:i32)->vec2<u32>{
 if(x<=0||x>=p.pw-1){return vec2<u32>(2039583u);}
 let g=clamp(2*(rgb(x+1,y,s)-rgb(x-1,y,s))+rgb(x+1,y-1,s)-rgb(x-1,y-1,s)+rgb(x+1,y+1,s)-rgb(x-1,y+1,s),vec3<i32>(-31),vec3<i32>(31))+vec3<i32>(31);
 return vec2<u32>(packed(g),packed(rgb(x,y,s)));
}
@compute @workgroup_size(128) fn prepare(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=i32(gid.x);if(i>=2*p.pw*p.h){return;}let x=i%p.pw;let y=(i/p.pw)%p.h;let s=i/(p.pw*p.h);
 let a=featureAt(x,y,s);let b=featureAt(max(x-1,0),y,s);let c=featureAt(min(x+1,p.pw-1),y,s);
 var lo:array<u32,2>;var hi:array<u32,2>;
 for(var j=0u;j<2u;j++){var l=vec3<i32>(0);var h=vec3<i32>(0);for(var k=0u;k<3u;k++){let v=byte(a[j],k);let vl=(v+byte(b[j],k))/2;let vr=(v+byte(c[j],k))/2;l[k]=min(v,min(vl,vr));h[k]=max(v,max(vl,vr));}lo[j]=packed(l);hi[j]=packed(h);}
 features[i]=Feature(a.x,a.y,lo[0],lo[1],hi[0],hi[1]);
}
fn bt(a:i32,al:i32,ah:i32,b:i32,bl:i32,bh:i32)->i32{return min(max(0,max(a-bh,bl-a)),max(0,max(b-ah,al-b)));}
@compute @workgroup_size(32,4) fn pixelCost(@builtin(global_invocation_id) g:vec3<u32>){
 let d=i32(g.x);let x=i32(g.y);let y=i32(g.z);if(d>=p.nd||x>=p.vw||y>=p.rows+p.radius){return;}
 let yy=min(p.ys+y,p.h-1);let xx=p.minx+x;
 let a=features[(p.side*p.h+yy)*p.pw+xx];let b=features[((1-p.side)*p.h+yy)*p.pw+xx-p.md-d];var v=0;
 for(var c=0u;c<3u;c++){v+=bt(byte(a.g,c),byte(a.gl,c),byte(a.gh,c),byte(b.g,c),byte(b.gl,c),byte(b.gh,c));v+=bt(byte(a.c,c),byte(a.cl,c),byte(a.ch,c),byte(b.c,c),byte(b.cl,c),byte(b.ch,c))>>2;}
 cost[(y*p.vw+x)*p.nd+d]=v;
}
@compute @workgroup_size(32,4) fn horizontal(@builtin(global_invocation_id) g:vec3<u32>){
 let d=i32(g.x);let x=i32(g.y);let y=i32(g.z);if(d>=p.nd||x>=p.vw||y>=p.rows+p.radius){return;}var v=0;for(var k=-p.radius;k<=p.radius;k++){v+=cost[(y*p.vw+clamp(x+k,0,p.vw-1))*p.nd+d];}temp[(y*p.vw+x)*p.nd+d]=v;
}
@compute @workgroup_size(32,4) fn vertical(@builtin(global_invocation_id) g:vec3<u32>){
 let d=i32(g.x);let x=i32(g.y);let y=i32(g.z);if(d>=p.nd||x>=p.vw||y>=p.rows){return;}var v=p.p2;for(var k=-p.radius;k<=p.radius;k++){v+=temp[(max(y+k,0)*p.vw+x)*p.nd+d];}cost[(y*p.vw+x)*p.nd+d]=v;
}
`;
const pathCode=mode=>`enable subgroups;\nrequires subgroup_id;\n${common}
const MODE=${mode};
var<workgroup> previous:array<i32,1024>;
var<workgroup> totals:array<i32,1024>;
var<workgroup> reduction:array<i32,64>;
var<workgroup> laneBest:array<atomic<u32>,8>;
var<workgroup> ambiguous:atomic<u32>;
fn groupMin(v:i32,lane:u32,sg:u32,nsg:u32)->i32{
 let m=subgroupMin(v);if(lane==0u){reduction[sg]=m;}workgroupBarrier();var best=32767;for(var i=0u;i<nsg;i++){best=min(best,reduction[i]);}workgroupBarrier();return best;
}
@compute @workgroup_size(256) fn path(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) sg:u32,@builtin(num_subgroups) nsg:u32){
 for(var j=0u;j<4u;j++){previous[tid+j*256u]=0;}workgroupBarrier();var oldMin=0;
 let length=select(p.vw,p.rows,MODE==1);
 for(var t=0;t<length;t++){
  var x=t;var y=i32(group.x);if(MODE==1){x=i32(group.x);y=t;}if(MODE==2){x=p.vw-1-t;}
  let base=(y*p.vw+x)*p.nd;var values:array<i32,4>;var localMin=32767;
  for(var j=0u;j<4u;j++){let d=i32(tid+j*256u);values[j]=32767;if(d<p.nd){var neighbor=32767;if(d>0){neighbor=min(neighbor,previous[d-1]);}if(d+1<p.nd){neighbor=min(neighbor,previous[d+1]);}let m=min(min(sat(neighbor+p.p1),previous[d]),sat(oldMin+p.p2));values[j]=sat(cost[base+d]+sat(m-sat(oldMin+p.p2)));localMin=min(localMin,values[j]);}}
  workgroupBarrier();oldMin=groupMin(localMin,lane,sg,nsg);
  var sumMin=32767;
  for(var j=0u;j<4u;j++){let d=i32(tid+j*256u);if(d<p.nd){previous[d]=values[j];if(MODE==1){temp[base+d]=values[j];}else if(MODE==0){temp[base+d]=sat(temp[base+d]+values[j]);}else{totals[d]=sat(temp[base+d]+values[j]);sumMin=min(sumMin,totals[d]);}}}
  workgroupBarrier();
  if(MODE==2){
   let bestCost=groupMin(sumMin,lane,sg,nsg);
   if(tid<8u){atomicStore(&laneBest[tid],0u);}if(tid==0u){atomicStore(&ambiguous,0u);}workgroupBarrier();
   for(var j=0u;j<4u;j++){let d=tid+j*256u;if(i32(d)<p.nd&&totals[d]==bestCost){atomicMax(&laneBest[d%8u],d+1u);}}workgroupBarrier();
   var best=32767;for(var j=0u;j<8u;j++){let v=atomicLoad(&laneBest[j]);if(v>0u){best=min(best,i32(v)-1);}}
   let threshold=(100*bestCost)/88;
   for(var j=0u;j<4u;j++){let d=i32(tid+j*256u);if(d<p.nd&&abs(d-best)>1&&totals[d]<=threshold){atomicOr(&ambiguous,1u);}}workgroupBarrier();
   if(tid==0u&&p.ys+y>=p.reserved){var disp=(p.md-1)*16;if(atomicLoad(&ambiguous)==0u){disp=best*16;if(best>0&&best<p.nd-1){let den=max(totals[best-1]+totals[best+1]-2*totals[best],1);disp+=((totals[best-1]-totals[best+1])*16+den)/(2*den);}disp+=p.md*16;}winners[((p.ys+y)*p.vw+x)]=Winner(disp,best+p.md,bestCost);}
   workgroupBarrier();
  }
 }
}`;
class GpuStereo {
 static async create(){
  if(!navigator.gpu)throw Error('WebGPUを利用できません');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('WebGPUアダプターがありません');
  if(!adapter.features.has('subgroups')||!navigator.gpu.wgslLanguageFeatures.has('subgroup_id'))throw Error('SGBMに必要なWebGPU subgroup機能がありません');
  const timing=adapter.features.has('timestamp-query');
  const device=await adapter.requestDevice({requiredFeatures:['subgroups',...(timing?['timestamp-query']:[])],requiredLimits:{maxBufferSize:Math.min(adapter.limits.maxBufferSize,1073741824),maxStorageBufferBindingSize:Math.min(adapter.limits.maxStorageBufferBindingSize,1073741824)}});
  const self=new GpuStereo(device,timing);await self.initialize();return self;
 }
 constructor(device,timing){this.device=device;this.timing=timing;this.allocations=[];this.lost=null;device.lost.then(x=>this.lost=x.message||'GPU device lost');device.addEventListener('uncapturederror',e=>{this.error=e.error.message;});}
 async initialize(){
  const d=this.device;this.layout=d.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...Array.from({length:5},(_,i)=>({binding:i+1,visibility:GPUShaderStage.COMPUTE,buffer:{type:i===0?'read-only-storage':'storage'}}))]});
  const layout=d.createPipelineLayout({bindGroupLayouts:[this.layout]});this.pipelines={};
  const sources={prepare:[common,'prepare'],pixelCost:[common,'pixelCost'],horizontal:[common,'horizontal'],vertical:[common,'vertical'],top:[pathCode(1),'path'],left:[pathCode(0),'path'],right:[pathCode(2),'path']};
  for(const [key,[code,entryPoint]]of Object.entries(sources)){const module=d.createShaderModule({label:'SGBM '+key,code});const info=await module.getCompilationInfo();const errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw Error(key+': '+errors.map(x=>`${x.lineNum}:${x.message}`).join('\n'));this.pipelines[key]=await d.createComputePipelineAsync({label:'SGBM '+key,layout,compute:{module,entryPoint}});}
 }
 allocate(w,h,minimum,disparities,block=5){
  if(![3,5,7,9].includes(block))throw Error('GPU SGBMの窓幅は3,5,7,9です');const radius=(block-1)/2,p1=24*block*block,p2=96*block*block;
  const nd=Math.ceil(disparities/16)*16,pad=minimum+nd+5,pw=w+2*pad,mdRight=-minimum-nd+1,vwLeft=pw-(minimum+nd),vwRight=pw+mdRight,stripe=Math.ceil(h/4),overlap=3+Math.ceil(.1*stripe),maxRows=stripe+overlap;
  const key=[w,h,minimum,nd,block].join('/');if(this.key===key)return this.shape;
  for(const b of this.allocations)b.destroy();this.allocations=[];
  const buffer=(size,usage)=>{const b=this.device.createBuffer({size:Math.ceil(size/4)*4,usage});this.allocations.push(b);return b;};
  const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
  const volume=Math.max(vwLeft,vwRight)*nd*(maxRows+radius)*4;
  if(nd>1024||volume>this.device.limits.maxStorageBufferBindingSize)throw Error('指定した全視差範囲のGPUバッファが上限を超えます');
  this.params=buffer(8*256,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);this.input=buffer(w*h*2*4,storage);this.features=buffer(pw*h*2*24,storage);this.cost=buffer(volume,storage);this.temp=buffer(volume,storage);this.output=buffer(Math.max(vwLeft,vwRight)*h*12,storage|GPUBufferUsage.COPY_SRC);
  this.readbacks=[vwLeft,vwRight].map(vw=>buffer(vw*h*12,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST));
  if(this.timing){this.query?.destroy();this.query=this.device.createQuerySet({type:'timestamp',count:2});this.queryResolve=buffer(16,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);this.queryRead=buffer(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);}
  this.groups=[];const uniforms=new Int32Array(8*64);
  for(let side=0;side<2;side++)for(let s=0;s<4;s++){const slot=side*4+s,md=side?mdRight:minimum,vw=side?vwRight:vwLeft,ys=Math.max(0,s*stripe-overlap),end=Math.min(h,(s+1)*stripe);uniforms.set([w,h,pad,pw,vw,Math.max(md+nd,0),nd,md,ys,end-ys,side,s*stripe,radius,p1,p2,0],slot*64);this.groups.push(this.device.createBindGroup({layout:this.layout,entries:[{binding:0,resource:{buffer:this.params,offset:slot*256,size:64}},...[this.input,this.features,this.cost,this.temp,this.output].map((buffer,i)=>({binding:i+1,resource:{buffer}}))]}));}
  this.device.queue.writeBuffer(this.params,0,uniforms);this.key=key;return this.shape={w,h,minimum,nd,pw,vwLeft,vwRight,stripe,overlap,block,radius,volumeBytes:volume*2};
 }
 async compute(left,right,w,h,minimum,disparities,native,block=5){
  if(this.lost)throw Error(this.lost);this.error=null;const start=performance.now(),s=this.allocate(w,h,minimum,disparities,block),d=this.device;
  d.pushErrorScope('validation');const packed=new Uint32Array(w*h*2);for(let side=0;side<2;side++){const rgb=side?right:left,offset=side*w*h;for(let i=0;i<w*h;i++)packed[offset+i]=rgb[3*i]|rgb[3*i+1]<<8|rgb[3*i+2]<<16;}d.queue.writeBuffer(this.input,0,packed);
  const encoder=d.createCommandEncoder(),dispatch=(name,slot,x,y=1,z=1,timestamp)=>{const pass=encoder.beginComputePass(timestamp&&this.timing?{timestampWrites:{querySet:this.query,...timestamp}}:{});pass.setPipeline(this.pipelines[name]);pass.setBindGroup(0,this.groups[slot]);pass.dispatchWorkgroups(x,y,z);pass.end();};
  dispatch('prepare',0,Math.ceil(2*s.pw*h/128),1,1,{beginningOfPassWriteIndex:0});
  for(let side=0;side<2;side++){const vw=side?s.vwRight:s.vwLeft;for(let stripe=0;stripe<4;stripe++){const slot=4*side+stripe,ys=Math.max(0,stripe*s.stripe-s.overlap),rows=Math.min(h,(stripe+1)*s.stripe)-ys;
   dispatch('pixelCost',slot,Math.ceil(s.nd/32),Math.ceil(vw/4),rows+s.radius);dispatch('horizontal',slot,Math.ceil(s.nd/32),Math.ceil(vw/4),rows+s.radius);dispatch('vertical',slot,Math.ceil(s.nd/32),Math.ceil(vw/4),rows);dispatch('top',slot,vw);dispatch('left',slot,rows);dispatch('right',slot,rows,1,1,side===1&&stripe===3?{endOfPassWriteIndex:1}:null);
  }encoder.copyBufferToBuffer(this.output,0,this.readbacks[side],0,vw*h*12);}
  if(this.timing){encoder.resolveQuerySet(this.query,0,2,this.queryResolve,0);encoder.copyBufferToBuffer(this.queryResolve,0,this.queryRead,0,16);}const submitted=performance.now();d.queue.submit([encoder.finish()]);
  await Promise.all([...this.readbacks.map(b=>b.mapAsync(GPUMapMode.READ)),...(this.timing?[this.queryRead.mapAsync(GPUMapMode.READ)]:[])]);const mapped=performance.now();
  let gpuMs=null;if(this.timing){const t=new BigUint64Array(this.queryRead.getMappedRange());gpuMs=Number(t[1]-t[0])/1e6;this.queryRead.unmap();}
  const arrays=this.readbacks.map(b=>{const copy=new Int32Array(b.getMappedRange().slice(0));b.unmap();return copy;});const error=await d.popErrorScope();if(error||this.error)throw Error(error?.message||this.error);
  const a=native.finishGpuDisparity(arrays[0],w,h,minimum,s.nd,false),b=native.finishGpuDisparity(arrays[1],w,h,minimum,s.nd,true),end=performance.now();
  return {left:a,right:b,minimum,rightMinimum:-minimum-s.nd+1,gpuTiming:{prepareMs:submitted-start,gpuMs,submitAndReadbackMs:mapped-submitted,postprocessMs:end-mapped,totalMs:end-start,volumeBytes:s.volumeBytes,backend:'WebGPU RGB SGBM 3WAY'}};
 }
}
root.GpuStereo=GpuStereo;
})(self);
