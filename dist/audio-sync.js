/* Estimate recording offset from actual decoded audio; no filename time assumptions. */
export async function envelope(file,onProgress=()=>{}){
 const meta=await A4Offline.inspect(file),audio=meta.audio;
 if(!audio||typeof AudioDecoder==='undefined')throw Error('音声による同期を利用できません');
 const support=await AudioDecoder.isConfigSupported(audio.config);if(!support.supported)throw Error('この音声形式をデコードできません');
 const rate=100,sums=new Float64Array(Math.ceil(meta.duration*rate)+2),counts=new Uint32Array(sums.length);
 let failure=null,previous=0,lastTimestamp=-Infinity;
 const decoder=new AudioDecoder({error:e=>failure=e,output:data=>{try{
  const values=new Float32Array(data.numberOfFrames);data.copyTo(values,{planeIndex:0,format:'f32-planar'});
  const start=data.timestamp/1e6+audio.timelineOffset;
  if(data.timestamp<=lastTimestamp)throw Error('音声の時刻が逆行しました');lastTimestamp=data.timestamp;
  for(let i=0;i<values.length;i++){const x=values[i],d=x-previous;previous=x;const bin=Math.floor((start+i/data.sampleRate)*rate);if(bin>=0&&bin<sums.length){sums[bin]+=d*d;counts[bin]++;}}
 }catch(e){failure=e;}finally{data.close();}}});
 decoder.configure(support.config);
 let cache=new Uint8Array(),cacheStart=-1;
 try{
  for(let i=0;i<audio.samples.length;i++){
   if(failure)throw failure;
   while(decoder.decodeQueueSize>64){await new Promise(r=>setTimeout(r,4));if(failure)throw failure;}
   const s=audio.samples[i];
   if(s.offset<cacheStart||s.offset+s.size>cacheStart+cache.byteLength){cacheStart=s.offset;cache=new Uint8Array(await file.slice(cacheStart,Math.min(file.size,cacheStart+2*1024*1024)).arrayBuffer());}
   decoder.decode(new EncodedAudioChunk({type:'key',timestamp:Math.round(s.cts*1e6/s.timescale),duration:Math.round(s.duration*1e6/s.timescale),data:cache.subarray(s.offset-cacheStart,s.offset-cacheStart+s.size)}));
   if(i%500===0)onProgress(i/audio.samples.length);
  }
  await decoder.flush();if(failure)throw failure;
 }finally{decoder.close();}
 const values=Float32Array.from(sums,(s,i)=>counts[i]?Math.log(1e-12+s/counts[i]):0);
 // Remove slow gain changes; correlate short sound events rather than exposure or level.
 const prefix=new Float64Array(values.length+1);for(let i=0;i<values.length;i++)prefix[i+1]=prefix[i]+values[i];
 const high=Float32Array.from(values,(v,i)=>{const a=Math.max(0,i-50),b=Math.min(values.length,i+51);return counts[i]?v-(prefix[b]-prefix[a])/(b-a):0;});
 onProgress(1);return {values:high,rate,duration:meta.duration,file:file.name};
}
function correlation(a,b,lag,stride,start=0,end=a.length){
 let n=0,x=0,y=0,xx=0,yy=0,xy=0;
 for(let i=Math.max(start,0,-lag);i<Math.min(end,a.length,b.length-lag);i+=stride){const u=a[i],v=b[i+lag];n++;x+=u;y+=v;xx+=u*u;yy+=v*v;xy+=u*v;}
 const den=Math.sqrt(Math.max(0,(xx-x*x/n)*(yy-y*y/n)));return n>200&&den>1e-7?(xy-x*y/n)/den:-1;
}
export function compare(a,b){
 const candidates=[],range=Math.floor(Math.min(a.duration,b.duration,120)*a.rate);
 for(let lag=-range;lag<=range;lag+=10)candidates.push({lag,score:correlation(a.values,b.values,lag,10)});
 candidates.sort((x,y)=>y.score-x.score);const coarse=candidates[0],fine=[];
 for(let lag=coarse.lag-12;lag<=coarse.lag+12;lag++)fine.push({lag,score:correlation(a.values,b.values,lag,1)});
 fine.sort((x,y)=>y.score-x.score);const best=fine[0],other=candidates.find(c=>Math.abs(c.lag-best.lag)>100);
 const windows=[];for(let i=0;i<3;i++){
  const start=Math.floor(a.values.length*i/3),end=Math.floor(a.values.length*(i+1)/3),local=[];
  for(let lag=best.lag-50;lag<=best.lag+50;lag++)local.push({lag,score:correlation(a.values,b.values,lag,2,start,end)});
  local.sort((x,y)=>y.score-x.score);windows.push({offset:local[0].lag/a.rate,score:local[0].score});
 }
 const spread=Math.max(...windows.map(w=>w.offset))-Math.min(...windows.map(w=>w.offset));
 return {offset:best.lag/a.rate,score:best.score,alternativeScore:other?.score,windows,spread,confident:best.score>.35&&best.score-(other?.score||0)>.12&&windows.every(w=>w.score>.2)&&spread<=.06,files:[a.file,b.file],convention:'camera2 time = camera1 time + offset',resolutionSeconds:1/a.rate};
}
