(function(root){
 'use strict';
 const mm=(a,b)=>Array.from({length:9},(_,i)=>[0,1,2].reduce((s,k)=>s+a[Math.floor(i/3)*3+k]*b[k*3+i%3],0));
 const mv=(a,b)=>[0,1,2].map(i=>a[i*3]*b[0]+a[i*3+1]*b[1]+a[i*3+2]*b[2]);
 const transpose=a=>[a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]];
 let native;
 async function module(){
  if(!native){importScripts('native/stereo-native.js');native=StereoNativeFactory({locateFile:name=>'native/'+name});}
  return native;
 }
 async function fit(cv,engine,images,pairs,options){
  if(!Array.isArray(pairs)||pairs.length<2)throw Error('同じ実物の紙を2組以上対応づけてください');
  if(pairs.some(p=>p.length!==2||p.some(v=>!Number.isInteger(v)||v<0)))throw Error('紙番号は0からの整数です');
  if(new Set(pairs.map(p=>p[0])).size!==pairs.length||new Set(pairs.map(p=>p[1])).size!==pairs.length)throw Error('一枚の紙を重複して指定できません');
  const groups=images.map(im=>engine.boards(engine.detect(im,{},true).markers));
  const prepared=groups.map(gs=>gs.map(g=>options.allCorners!==false?{points:g.markers.flatMap(m=>A4Core.tagUV(m.id).map((uv,corner)=>({id:m.id,corner,uv,xy:m.corners[corner]})))}:MultiPaper.prepare(cv,[g])[0]));
  const views=pairs.map(([a,b])=>{
   const A=prepared[0][a],B=prepared[1][b];
   if(!A||!B)throw Error(`紙 ${a}:${b} を検出できません`);
   const right=new Map(B.points.map(p=>[p.id+':'+p.corner,p]));
   const points=A.points.filter(p=>right.has(p.id+':'+p.corner));
   if(points.length<8)throw Error(`紙 ${a}:${b} の共通の有効な角は${points.length}個です。8個以上必要です`);
   return {ids:points.map(p=>p.id),object:points.map(p=>[(p.uv[0]-148.5)*options.ruler/100/1000,(105-p.uv[1])*options.ruler/100/1000,0]),left:points.map(p=>p.xy),right:points.map(p=>right.get(p.id+':'+p.corner).xy),pair:[a,b]};
  });
  const api=await module(),K=(i,ratio)=>{const im=images[i],f=Math.max(im.width,im.height)*ratio;return [f,0,im.width/2,0,f,im.height/2,0,0,1];};
  const run=(k0,k1,omit=-1,guess=null)=>{
   const selected=views.map(v=>{const indices=v.ids.map((id,i)=>id!==omit?i:-1).filter(i=>i>=0);return {object:indices.map(i=>v.object[i]),left:indices.map(i=>v.left[i]),right:indices.map(i=>v.right[i])};}).filter(v=>v.object.length>=8);
   if(selected.length<2)throw Error('除外後の共通観測が不足しています');
   const r=api.calibrateStereo(selected,{K0:k0,K1:k1,width:images[0].width,height:images[0].height,radial:!!options.radial,...(guess?{R:guess.R,T:guess.T,D0:guess.D0,D1:guess.D1}:{})});
   if(r.error)throw Error(r.error);
   r.baseline=Math.hypot(...r.T);return r;
  };
  const attempts=[];
  for(const [a,b]of [[.55,.55],[.8,.8],[1.2,1.2],[.55,.9],[.9,.55]])try{attempts.push(run(K(0,a),K(1,b)));}catch(e){attempts.push({error:e.message});}
  const valid=attempts.filter(r=>r.K0&&[...r.K0,...r.K1,...r.D0,...r.D1,...r.R,...r.T,r.rms].every(Number.isFinite)&&r.baseline>.015&&r.baseline<5&&[r.K0,r.K1].every((k,i)=>{const fov=2*Math.atan(images[i].width/(2*k[0]))*180/Math.PI;return fov>15&&fov<130;})).sort((a,b)=>a.rms-b.rms);
  if(!valid.length)return {ok:false,reason:'標準ステレオ校正の有効な解がありません',attempts};
  const best=valid[0],near=valid.filter(r=>r.rms<=best.rms+.05);
  const seedSpread=[0,1].map(i=>{const key='K'+i,fs=near.map(r=>r[key][0]);return (Math.max(...fs)-Math.min(...fs))/best[key][0];});
  const base={...best,pairs,cornerCounts:views.map(v=>v.ids.length),seedSpread,method:'OpenCV stereoCalibrate',attempts:attempts.map(r=>({error:r.error,rms:r.rms,f0:r.K0?.[0],f1:r.K1?.[0],baseline:r.baseline}))};
  if(best.rms>2.5||Math.max(...seedSpread)>.16)return {ok:false,reason:'ステレオ校正の誤差または初期値への依存が大きい',...base};
  const leaveOne=[];
  for(const id of new Set(views.flatMap(v=>v.ids)))try{const r=run(best.K0,best.K1,id,best);leaveOne.push({id,f0:r.K0[0],f1:r.K1[0],baseline:r.baseline,rms:r.rms});}catch(e){leaveOne.push({id,error:e.message});}
  const checks=leaveOne.filter(r=>Number.isFinite(r.f0)&&Number.isFinite(r.f1));
  const spread=['f0','f1','baseline'].map((key,i)=>{const vs=checks.map(r=>r[key]);return vs.length?(Math.max(...vs)-Math.min(...vs))/(i<2?best['K'+i][0]:best.baseline):Infinity;});
  if(checks.length<4||Math.max(...spread)>.16)return {ok:false,reason:'共通タグを一つ除くとステレオ校正が不安定になります',...base,leaveOne,spread};
  const p0=best.poses[0],p1={R:mm(best.R,p0.R),t:mv(best.R,p0.t).map((v,i)=>v+best.T[i])};
  const cameras=[p0,p1].map((p,i)=>({K:best['K'+i],D:best['D'+i],R:p.R,t:p.t,center:mv(transpose(p.R),p.t).map(v=>-v),width:images[i].width,height:images[i].height,rms:best.rms,fov:2*Math.atan(images[i].width/(2*best['K'+i][0]))*180/Math.PI,source:'a4-stereo-calibrated',model:'opencv-pinhole-brown5',reference:pairs[0][i],lensSpread:spread[i],assumptions:'Both cameras optimized jointly by OpenCV; square pixels; centered principal points; '+(options.radial?'radial k1 estimated':'zero lens distortion')}));
  return {ok:true,...base,leaveOne,spread,cameras,referenceMarkers:groups.map(gs=>gs.flatMap(g=>g.markers.map(({id,corners})=>({id,corners})))),units:'m',origin:'first paired paper center; X right, Y up, Z paper front',createdAt:new Date().toISOString()};
 }
 root.StereoCalibration={fit,module};
})(self);
