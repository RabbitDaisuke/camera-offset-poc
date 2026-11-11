'use strict';
// Seeded, image-validated local-plane depth completion. Added samples are
// estimates, not additional independent StereoSGBM measurements.
self.SurfaceCompletion=(()=>{
 const K=8;
 function fit(ps){
  const m=[0,0,0];for(const p of ps)for(let j=0;j<3;j++)m[j]+=p[j]/ps.length;
  let xx=0,xy=0,yy=0,xd=0,yd=0;
  for(const p of ps){const x=p[0]-m[0],y=p[1]-m[1],d=p[2]-m[2];xx+=x*x;xy+=x*y;yy+=y*y;xd+=x*d;yd+=y*d;}
  const det=xx*yy-xy*xy;if(det<1e-4)return null;
  const a=(xd*yy-yd*xy)/det,b=(yd*xx-xd*xy)/det;return [a,b,m[2]-a*m[0]-b*m[1]];
 }
 function models(cv,result,rect,rgbs){
  const {S,R,C0,f,baseline}=rect,samples=[];
  for(let i=0;i<result.points.length;i+=3){const v=[0,1,2].map(j=>result.points[i+j]-C0[j]),z=R[6]*v[0]+R[7]*v[1]+R[8]*v[2],k=i/3*4;samples.push([result.uv[k],result.uv[k+1],f*baseline/z]);}
  const out=[];
  for(let cy=20;cy<S;cy+=40)for(let cx=20;cx<S;cx+=40){
   let remaining=samples.filter(p=>Math.abs(p[0]-cx)<=48&&Math.abs(p[1]-cy)<=48);
   for(let pass=0;pass<2&&remaining.length>=16;pass++){
    const a=cv.matFromArray(remaining.length,1,cv.CV_32FC2,remaining.flatMap(p=>[p[0],p[1]])),b=cv.matFromArray(remaining.length,1,cv.CV_32FC2,remaining.flatMap(p=>[p[0]-p[2],p[1]])),mask=new cv.Mat();let H,ps;
    try{H=cv.findHomography(a,b,cv.RANSAC,1,mask,500,.99);if(H.empty())break;const flags=mask.data;ps=remaining.filter((_,i)=>flags[i]);}finally{H?.delete();a.delete();b.delete();mask.delete();}
    if(ps.length<16)break;let plane=fit(ps);if(!plane)break;
    for(let n=0;n<3;n++){ps=remaining.filter(p=>Math.abs(plane[0]*p[0]+plane[1]*p[1]+plane[2]-p[2])<=1);if(ps.length<16)break;plane=fit(ps);if(!plane)break;}
    if(!plane||ps.length<16)break;const used=new Set(ps);remaining=remaining.filter(p=>!used.has(p));
    const x0=Math.min(...ps.map(p=>p[0])),x1=Math.max(...ps.map(p=>p[0])),y0=Math.min(...ps.map(p=>p[1])),y1=Math.max(...ps.map(p=>p[1]));
    if(x1-x0<8||y1-y0<8||Math.abs(1-plane[0])<.15)continue;
    const rms=Math.sqrt(ps.reduce((s,p)=>s+(plane[0]*p[0]+plane[1]*p[1]+plane[2]-p[2])**2,0)/ps.length);
    // Bounded extrapolation from observed support; RGB and two-view checks
    // below still decide whether any of these pixels can receive a depth.
    const appearance=[0,1].map(side=>{const mean=[0,0,0],square=[0,0,0];for(const p of ps){const x=side?p[0]-p[2]:p[0],ix=Math.floor(x),t=x-ix,k=(p[1]*S+ix)*3;for(let c=0;c<3;c++){const v=rgbs[side][k+c]*(1-t)+rgbs[side][k+c+3]*t;mean[c]+=v/ps.length;square[c]+=v*v/ps.length;}}return {mean,std:mean.map((v,c)=>Math.max(12,Math.min(40,Math.sqrt(Math.max(0,square[c]-v*v)))))};});
    out.push({plane,support:ps.length,rms,appearance,bounds:[Math.max(cx-48,x0-24),Math.min(cx+48,x1+24),Math.max(cy-48,y0-24),Math.min(cy+48,y1+24)]});
   }
  }
  return out;
 }
 const shader=`
 struct Params {size:u32,models:u32,radius:i32,stride:i32,ncc:f32,minimum:f32,maximum:f32,propagate:f32};
 @group(0) @binding(0) var<uniform> p:Params;
 @group(0) @binding(1) var<storage,read> images:array<u32>;
 @group(0) @binding(2) var<storage,read> masks:array<u32>;
 @group(0) @binding(3) var<storage,read> planes:array<vec4<f32>>;
 @group(0) @binding(4) var<storage,read> candidates:array<u32>;
 @group(0) @binding(5) var<storage,read_write> result:array<vec4<f32>>;
 fn rgb(x:i32,y:i32,side:i32)->vec3<f32>{let k=(side*i32(p.size)+y)*i32(p.size)+x;let v=images[k];return vec3<f32>(f32(v&255u),f32((v>>8u)&255u),f32((v>>16u)&255u));}
 fn sample(x:f32,y:i32,side:i32)->vec3<f32>{let ix=i32(floor(x));return mix(rgb(ix,y,side),rgb(ix+1,y,side),fract(x));}
 fn score(x:i32,y:i32,side:i32,plane:vec3<f32>)->f32{
  let center=rgb(x,y,side);var sa=vec3<f32>(0);var sb=vec3<f32>(0);var aa=0.;var bb=0.;var ab=0.;var sw=0.;var all=0.;
  for(var dy=-p.radius;dy<=p.radius;dy+=p.stride){for(var dx=-p.radius;dx<=p.radius;dx+=p.stride){
   let xx=x+dx;let yy=y+dy;if(xx<0||yy<0||xx>=i32(p.size)||yy>=i32(p.size)){continue;}
   let a=rgb(xx,yy,side);let weight=exp(-dot(abs(a-center),vec3<f32>(1))/45.);all+=weight;
   let d=dot(plane,vec3<f32>(f32(xx),f32(yy),1));let u=f32(xx)-d;if(u<0||u>=f32(p.size-1u)){continue;}
   let b=sample(u,yy,1-side);sa+=a*weight;sb+=b*weight;aa+=dot(a,a)*weight;bb+=dot(b,b)*weight;ab+=dot(a,b)*weight;sw+=weight;
  }}
  if(sw<12.||sw<all*.9){return -2.;}let va=max(0.,aa-dot(sa,sa)/sw);let vb=max(0.,bb-dot(sb,sb)/sw);
  if(va<sw*27.||vb<sw*27.){return -2.;}return (ab-dot(sa,sb)/sw)/sqrt(va*vb);
 }
 fn appearanceScore(x:i32,y:i32,side:i32,plane:vec3<f32>,index:u32)->f32{
  let base=(index-1u)*5u;let ma=planes[base+1u+u32(side)].xyz;let mb=planes[base+2u-u32(side)].xyz;let sa=planes[base+3u+u32(side)].xyz;let sb=planes[base+4u-u32(side)].xyz;
  var a=vec3<f32>(0);var b=vec3<f32>(0);var n=0.;
  for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){let xx=x+dx;let yy=y+dy;let u=f32(xx)-dot(plane,vec3<f32>(f32(xx),f32(yy),1));if(xx<0||yy<0||xx>=i32(p.size)||yy>=i32(p.size)||u<0||u>=f32(p.size-1u)){return -2.;}a+=rgb(xx,yy,side);b+=sample(u,yy,1-side);n+=1.;}}
  a/=n;b/=n;let da=(a-ma)/sa;let db=(b-mb)/sb;if(dot(da,da)>9.||dot(db,db)>9.){return -2.;}
  let difference=(a-ma)-(b-mb);let error=dot(difference,difference)/3.;if(error>324.){return -2.;}
  // A color-supported extension of an observed plane, not a new match.
  return p.ncc+.12*exp(-error/128.);
 }
 @compute @workgroup_size(64) fn planeMatch(@builtin(global_invocation_id) gid:vec3<u32>){
  let k=gid.x;if(k>=2u*p.size*p.size){return;}result[k]=vec4<f32>(0.,-2.,0.,0.);if(masks[k]!=255u){return;}
  let x=i32(k%p.size);let y=i32((k/p.size)%p.size);let side=i32(k/(p.size*p.size));
  var best=-2.;var chosen=vec3<f32>(0);var id=0u;var selected=0.;var inferred=0.;
  for(var j=0u;j<8u;j++){let index=candidates[k*8u+j];if(index==0u){continue;}var plane=planes[(index-1u)*5u].xyz;if(side==1){plane=-plane/(1.-plane.x);}
   let d=dot(plane,vec3<f32>(f32(x),f32(y),1));if(abs(d)<p.minimum||abs(d)>p.maximum){continue;}
   let u=f32(x)-d;if(u<1.||u>=f32(p.size-1u)){continue;}let q=(u32(1-side)*p.size+u32(y))*p.size+u32(round(u));if(masks[q]!=255u){continue;}
   var s=score(x,y,side,plane);var kind=0.;if(s<p.ncc&&p.propagate>0.){s=appearanceScore(x,y,side,plane,index);kind=1.;}if(s>best){best=s;chosen=plane;id=index;selected=d;inferred=kind;}
  }
  if(best<p.ncc){return;}result[k]=vec4<f32>(selected,best,f32(id),inferred);
 }
 `;
 class Matcher{
  constructor(device){this.device=device;this.buffers=[];}
  async init(){const d=this.device,module=d.createShaderModule({label:'Seeded surface plane RGB validation',code:shader});const info=await module.getCompilationInfo();const errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(x=>x.message).join('\n'));this.pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'planeMatch'}});}
  async compute(left,right,fa,fb,ms,S,opts){
   const start=performance.now(),d=this.device;for(const b of this.buffers)b.destroy();this.buffers=[];
   const buffer=(bytes,usage,data)=>{const b=d.createBuffer({size:Math.max(4,bytes),usage});this.buffers.push(b);if(data)d.queue.writeBuffer(b,0,data);return b;};
   const pixels=new Uint32Array(2*S*S),masks=new Uint32Array(2*S*S),ids=new Uint32Array(2*S*S*K),priority=new Float32Array(ids.length);priority.fill(Infinity);
   for(let side=0;side<2;side++){const rgb=side?right:left,mask=side?fb.raw:fa.raw;for(let i=0;i<S*S;i++){pixels[side*S*S+i]=rgb[3*i]|rgb[3*i+1]<<8|rgb[3*i+2]<<16;masks[side*S*S+i]=mask[i];}}
   const planes=new Float32Array(ms.length*20);
   ms.forEach((m,id)=>{planes.set([...m.plane,m.rms,...m.appearance[0].mean,0,...m.appearance[1].mean,0,...m.appearance[0].std,0,...m.appearance[1].std,0],id*20);const [a,b,c]=m.plane,[x0,x1,y0,y1]=m.bounds;
    for(let side=0;side<2;side++){
     const corners=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]].map(([x,y])=>[side?x-(a*x+b*y+c):x,y]),loX=Math.max(0,Math.ceil(Math.min(...corners.map(p=>p[0])))),hiX=Math.min(S-1,Math.floor(Math.max(...corners.map(p=>p[0])))),loY=Math.max(0,Math.ceil(y0)),hiY=Math.min(S-1,Math.floor(y1)),cx=(loX+hiX)/2,cy=(loY+hiY)/2;
     for(let y=loY;y<=hiY;y++)for(let x=loX;x<=hiX;x++){const k=(side*S*S+y*S+x);if(masks[k]!==255)continue;const rank=(x-cx)**2+(y-cy)**2+m.rms*100;let slot=0;for(let j=1;j<K;j++)if(priority[k*K+j]>priority[k*K+slot])slot=j;if(rank<priority[k*K+slot]){ids[k*K+slot]=id+1;priority[k*K+slot]=rank;}}
    }
   });
   const params=new ArrayBuffer(32),u=new Uint32Array(params),f=new Float32Array(params);u.set([S,ms.length,opts.completionRadius||16,2]);f.set([opts.ncc,1,S-1,opts.completionPropagate===false?0:1],4);
   const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,inputs=[buffer(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,params),buffer(pixels.byteLength,storage,pixels),buffer(masks.byteLength,storage,masks),buffer(planes.byteLength,storage,planes),buffer(ids.byteLength,storage,ids)],output=buffer(2*S*S*16,storage|GPUBufferUsage.COPY_SRC),read=buffer(2*S*S*16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
   d.pushErrorScope('validation');const group=d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[...inputs,output].map((buffer,binding)=>({binding,resource:{buffer}}))}),encoder=d.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(2*S*S/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,2*S*S*16);const submitted=performance.now();d.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();const error=await d.popErrorScope();if(error)throw Error(error.message);return {values,prepareMs:submitted-start,computeReadbackMs:performance.now()-submitted};
  }
 }
 let matcher;
 async function complete(cv,gpu,result,rgbA,rgbB,fa,fb,rect,opts,accept){
  const start=performance.now(),ms=models(cv,result,rect,[rgbA,rgbB]),prepared=performance.now(),{S,f,baseline,Rt,C0,cx,cy}=rect;
  result.pointKinds=new Uint8Array(result.points.length/3);
  if(!ms.length){result.completion={measured:result.points.length/3,estimated:0,models:0};return result;}
  if(!matcher){matcher=new Matcher(gpu.device);await matcher.init();}
  const matched=await matcher.compute(rgbA,rgbB,fa,fb,ms,S,opts),v=matched.values,existing=new Set(result.accepted),points=Array.from(result.points),colors=Array.from(result.colors),uv=Array.from(result.uv),kinds=Array.from(result.pointKinds),modelIds=new Int32Array(S*S);modelIds.fill(-1);let added=0,lrRejected=0;
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){
   const k=y*S+x,d=v[k*4],score=v[k*4+1],u=x-d;if(existing.has(k)||score<opts.ncc||u<0||u>=S)continue;
   const q=S*S+y*S+Math.round(u),back=v[q*4];if(v[q*4+1]<opts.ncc||Math.abs(d+back)>1){lrRejected++;continue;}
   const z=f*baseline/d;if(z<opts.near||z>opts.far)continue;const ray=[(x-cx)*z/f,(y-cy)*z/f,z],world=[0,1,2].map(i=>C0[i]+Rt[i*3]*ray[0]+Rt[i*3+1]*ray[1]+Rt[i*3+2]*ray[2]);if(opts.scope!=='scene'&&world[2]<.03)continue;
   if(!accept(x,y,Math.round(u),d,world))continue;
   points.push(...world);colors.push(rgbA[k*3],rgbA[k*3+1],rgbA[k*3+2]);uv.push(x,y,u,y);kinds.push(v[k*4+3]?2:1);modelIds[k]=Math.round(v[k*4+2])-1;added++;
  }
  // Only retain image-connected extensions anchored by observed points.
  // This is a support check, not a claim that connected colors identify a person.
  const index=new Int32Array(S*S);index.fill(-1);for(let i=0;i<kinds.length;i++)index[uv[i*4+1]*S+uv[i*4]]=i;
  const seen=new Uint8Array(S*S),keep=new Uint8Array(kinds.length);let disconnected=0;
  for(let first=0;first<index.length;first++)if(index[first]>=0&&!seen[first]){
   const component=[first];seen[first]=1;let anchors=0;
   for(let at=0;at<component.length;at++){const k=component[at],i=index[k];if(!kinds[i])anchors++;const x=k%S,y=Math.floor(k/S);for(const q of [x>0?k-1:-1,x+1<S?k+1:-1,y>0?k-S:-1,y+1<S?k+S:-1]){if(q<0||seen[q]||index[q]<0)continue;let difference=0;for(let c=0;c<3;c++)difference+=(rgbA[k*3+c]-rgbA[q*3+c])**2;if(difference>3*35*35)continue;seen[q]=1;component.push(q);}}
   for(const k of component){const i=index[k];if(!kinds[i]||anchors>=3)keep[i]=1;else disconnected++;}
  }
  const outP=[],outC=[],outUV=[],outK=[];let colorSupported=0;for(let i=0;i<kinds.length;i++)if(keep[i]){outP.push(...points.slice(i*3,i*3+3));outC.push(...colors.slice(i*3,i*3+3));outUV.push(...uv.slice(i*4,i*4+4));outK.push(kinds[i]);if(kinds[i]===2)colorSupported++;}
  result.points=new Float32Array(outP);result.colors=new Uint8Array(outC);result.uv=new Float32Array(outUV);result.pointKinds=new Uint8Array(outK);result.completion={measured:kinds.length-added,estimated:added-disconnected,colorSupported,disconnected,models:ms.length,lrRejected,modelMs:prepared-start,...matched,values:undefined,totalMs:performance.now()-start};
  if(opts.inspectRegion)result.completion.inspection={models:ms};
  return result;
 }
 return {complete};
})();
