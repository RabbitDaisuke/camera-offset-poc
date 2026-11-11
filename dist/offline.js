/* File-time decoding with bounded backpressure. Playback time never selects frames. */
(function(root){
 'use strict';
 const pause=()=>new Promise(r=>setTimeout(r,8));
 function aborted(signal){if(signal?.aborted)throw new DOMException('解析を停止しました。','AbortError');}
 async function inspect(file,signal){
  const MP4=await import('./vendor/mp4box.all.mjs');
  const parser=MP4.createFile(false,undefined,{allowInvalidBoxTypes:true});let offset=0,metadataOffset=0,info=null,problem=null;
  parser.onError=e=>problem=Error(String(e));parser.onReady=v=>info=v;
  // Read the sample table only. Encoded frames are later read from their original offsets.
  while(offset+8<=file.size){
   aborted(signal);const head=await file.slice(offset,offset+16).arrayBuffer(),view=new DataView(head);
   let size=view.getUint32(0);const type=String.fromCharCode(...new Uint8Array(head,4,4));
   if(size===1){if(head.byteLength<16)throw Error('動画を読み直してください。');size=Number(view.getBigUint64(8));}
   if(size===0)size=file.size-offset;
   if(!Number.isSafeInteger(size)||size<8||offset+size>file.size)throw Error('動画ファイルを確認してください。');
   if(type==='ftyp'){const data=await file.slice(offset,offset+size).arrayBuffer();data.fileStart=metadataOffset;parser.appendBuffer(data);metadataOffset+=size;}
   if(type==='moov'){if(size>64*1024*1024)throw Error('短い動画を選択してください。');if(!metadataOffset){const ftyp=new Uint8Array([0,0,0,20,102,116,121,112,113,116,32,32,0,0,0,0,113,116,32,32]).buffer;ftyp.fileStart=0;parser.appendBuffer(ftyp);metadataOffset=20;}const data=await file.slice(offset,offset+size).arrayBuffer();data.fileStart=metadataOffset;parser.appendBuffer(data);break;}
   offset+=size;
  }
  if(problem)throw problem;
  const track=info?.videoTracks?.[0];if(!track)throw Error('MP4またはMOV動画を選択してください。');
  const trak=parser.getTrackById(track.id),samples=parser.getTrackSamplesInfo(track.id),entry=trak.mdia.minf.stbl.stsd.entries[0];
  if(!samples?.length)throw Error('通常のMP4またはMOVとして保存した動画を選択してください。');
  let description;for(const key of ['avcC','hvcC','vpcC','av1C'])if(entry[key]){const stream=new MP4.DataStream(undefined,0,false);entry[key].write(stream);description=new Uint8Array(stream.buffer,8);break;}
  let timelineOffset=0;const edits=trak.edts?.elst?.entries||[];
  for(const edit of edits){if(edit.media_time===-1)timelineOffset+=edit.segment_duration/info.timescale;else{timelineOffset-=edit.media_time/track.timescale;break;}}
  const duration=track.movie_duration/info.timescale||track.duration/track.timescale;
  const visible=s=>s.cts/track.timescale+timelineOffset>=-1e-6&&s.cts/track.timescale+timelineOffset<duration-1e-6;
  const matrix=trak.tkhd.matrix||[65536,0,0,0,65536,0,0,0,1073741824];
  const rotation=((Math.round(Math.atan2(matrix[1],matrix[0])*180/Math.PI/90)*90)%360+360)%360;
  const audioTrack=info.audioTracks?.[0];let audio=null;
  if(audioTrack){
   const audioTrak=parser.getTrackById(audioTrack.id),audioEntry=audioTrak.mdia.minf.stbl.stsd.entries[0];
   const descriptor=(audioEntry.esds||audioEntry.wave?.esds)?.esd;
   const findSpecific=d=>d?.tag===5?d.data:(d?.descs||[]).map(findSpecific).find(Boolean);
   let offset=0;for(const edit of audioTrak.edts?.elst?.entries||[]){if(edit.media_time===-1)offset+=edit.segment_duration/info.timescale;else{offset-=edit.media_time/audioTrack.timescale;break;}}
   audio={track:audioTrack,samples:parser.getTrackSamplesInfo(audioTrack.id),timelineOffset:offset,config:{codec:audioTrack.codec,sampleRate:audioTrack.audio.sample_rate,numberOfChannels:audioTrack.audio.channel_count,description:findSpecific(descriptor)}};
  }
  return {track,samples,description,timelineOffset,duration,rotation,audio,total:samples.filter(visible).length,
   config:{codec:track.codec,codedWidth:track.video.width,codedHeight:track.video.height,description,hardwareAcceleration:'prefer-hardware'}};
 }
 async function run(file,{signal,onFrame,onProgress=()=>{},maxFrames=Infinity}={}){
  if(typeof VideoDecoder==='undefined')throw Error('全フレーム解析は最新のChromeまたはSafariで開いてください。');
  const meta=await inspect(file,signal);aborted(signal);
  let support=await VideoDecoder.isConfigSupported(meta.config);
  if(!support.supported)support=await VideoDecoder.isConfigSupported({...meta.config,hardwareAcceleration:'no-preference'});
  if(!support.supported)throw Error('この動画をH.264のMP4として保存し、読み直してください。');
  const queue=[];let decodeError=null,consumeError=null,finished=false,processed=0,decoded=0,lastTimestamp=-Infinity;
  const decoder=new VideoDecoder({output:frame=>{decoded++;queue.push(frame)},error:e=>{decodeError=e}});
  decoder.configure(support.config);
  const started=performance.now();onProgress({processed,total:meta.total,decoded,seconds:0,elapsed:0,complete:false});
  const consume=(async()=>{try{while(!finished||queue.length){aborted(signal);if(decodeError)throw decodeError;if(!queue.length){await pause();continue;}
   const frame=queue.shift();try{const seconds=frame.timestamp/1e6+meta.timelineOffset;
    if(seconds< -1e-6||seconds>=meta.duration-1e-6)continue;
    if(frame.timestamp<=lastTimestamp)throw Error('フレーム時刻を確認できませんでした。動画を読み直してください。');
    lastTimestamp=frame.timestamp;
    await onFrame(frame,{index:processed,seconds:Math.max(0,seconds),timestampUs:frame.timestamp,total:meta.total,rotation:meta.rotation});
    processed++;onProgress({processed,total:meta.total,decoded,seconds,elapsed:(performance.now()-started)/1000,complete:false});
   }finally{frame.close();}
  }}catch(e){consumeError=e;}})();
  try{
   for(const sample of meta.samples){aborted(signal);if(consumeError)throw consumeError;if(decodeError)throw decodeError;
    while(queue.length>=2||decoder.decodeQueueSize>=4){await pause();aborted(signal);if(consumeError)throw consumeError;if(decodeError)throw decodeError;}
    if(processed>=maxFrames)throw new DOMException('解析を停止しました。','AbortError');
    const bytes=await file.slice(sample.offset,sample.offset+sample.size).arrayBuffer();
    if(bytes.byteLength!==sample.size)throw Error('動画の一部を読み取れませんでした。');
    decoder.decode(new EncodedVideoChunk({type:sample.is_sync?'key':'delta',timestamp:Math.round(sample.cts*1e6/sample.timescale),duration:Math.round(sample.duration*1e6/sample.timescale),data:bytes}));
   }
   await decoder.flush();finished=true;await consume;if(consumeError)throw consumeError;if(decodeError)throw decodeError;
   if(processed!==meta.total)throw Error(`解析したフレームは${processed} / ${meta.total}枚です。動画を読み直してください。`);
   const result={processed,total:meta.total,decoded,seconds:meta.duration,elapsed:(performance.now()-started)/1000,complete:true,codec:meta.track.codec};onProgress(result);return result;
  }finally{finished=true;if(decoder.state!=='closed')decoder.close();await consume;for(const frame of queue)frame.close();queue.length=0;}
 }
 const fileMetadata=new WeakMap();
 async function metadata(file){if(!fileMetadata.has(file))fileMetadata.set(file,inspect(file));return fileMetadata.get(file);}
 function selectFrame(meta,seconds){
  let index=-1,distance=Infinity;
  for(let i=0;i<meta.samples.length;i++){const t=meta.samples[i].cts/meta.track.timescale+meta.timelineOffset;if(t<0||t>=meta.duration)continue;const d=Math.abs(t-seconds);if(d<distance){distance=d;index=i;}}
  if(index<0)throw Error('指定時刻の映像フレームがありません');
  return {index,seconds:Math.round(meta.samples[index].cts*1e6/meta.track.timescale)/1e6+meta.timelineOffset};
 }
 async function nearestTime(file,seconds){return selectFrame(await metadata(file),seconds).seconds;}
 async function frameAt(file,seconds){
  const started=performance.now(),meta=await metadata(file),selected=selectFrame(meta,seconds).index,metadataMs=performance.now()-started;
  let first=selected;while(first>0&&!meta.samples[first].is_sync)first--;
  let end=selected+1;while(end<meta.samples.length&&!meta.samples[end].is_sync)end++;
  let chosen=null,chosenDistance=Infinity,failure=null,waiting=null,decoded=0,queueWaitMs=0;
  const decoder=new VideoDecoder({output:frame=>{decoded++;const d=Math.abs(frame.timestamp/1e6+meta.timelineOffset-seconds);if(d<chosenDistance){chosen?.close();chosen=frame;chosenDistance=d;}else frame.close();},error:e=>{failure=e;waiting?.();}});
  // Wake only when the decoder consumes input, rather than adding 8ms sleeps
  // repeatedly even when hardware decode has already completed.
  const drain=async()=>{while(decoder.decodeQueueSize>6){if(failure)throw failure;const t=performance.now();await new Promise(resolve=>{const wake=()=>{decoder.removeEventListener('dequeue',wake);waiting=null;resolve();};waiting=wake;decoder.addEventListener('dequeue',wake,{once:true});if(decoder.decodeQueueSize<=6||failure)wake();});queueWaitMs+=performance.now()-t;}if(failure)throw failure;};
  const support=await VideoDecoder.isConfigSupported(meta.config);if(!support.supported)throw Error('指定時刻のフレームデコードを利用できません');decoder.configure(support.config);
  let readMs=0,decodeMs=0;
  try{
   // Read contiguous encoded data in bounded blocks instead of one asynchronous
   // file operation for every frame. Audio interleaving is retained by offsets.
   let cache=new Uint8Array(),cacheStart=-1;const decoding=performance.now();
   for(let i=first;i<end;i++){await drain();const s=meta.samples[i];
    if(s.offset<cacheStart||s.offset+s.size>cacheStart+cache.length){const t=performance.now();cacheStart=s.offset;cache=new Uint8Array(await file.slice(cacheStart,Math.min(file.size,cacheStart+Math.max(4*1024*1024,s.size))).arrayBuffer());readMs+=performance.now()-t;}
    const bytes=cache.subarray(s.offset-cacheStart,s.offset-cacheStart+s.size);if(bytes.length!==s.size)throw Error('動画の一部を読み取れませんでした。');
    decoder.decode(new EncodedVideoChunk({type:s.is_sync?'key':'delta',timestamp:Math.round(s.cts*1e6/s.timescale),duration:Math.round(s.duration*1e6/s.timescale),data:bytes}));}
   await decoder.flush();if(failure)throw failure;if(!chosen)throw Error('指定時刻の映像をデコードできません');
   decodeMs=performance.now()-decoding;const copying=performance.now();
   const rotated=meta.rotation===90||meta.rotation===270,canvas=document.createElement('canvas');canvas.width=rotated?chosen.displayHeight:chosen.displayWidth;canvas.height=rotated?chosen.displayWidth:chosen.displayHeight;
   const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.translate(canvas.width/2,canvas.height/2);ctx.rotate(meta.rotation*Math.PI/180);ctx.drawImage(chosen,-chosen.displayWidth/2,-chosen.displayHeight/2);
   const image=ctx.getImageData(0,0,canvas.width,canvas.height);
   return {image,requested:seconds,seconds:chosen.timestamp/1e6+meta.timelineOffset,timestampUs:chosen.timestamp,errorSeconds:chosenDistance,index:selected,timing:{totalMs:performance.now()-started,metadataMs,readMs,decodeMs,queueWaitMs,copyMs:performance.now()-copying,decodedFrames:decoded,fixedWaitMs:0}};
  }finally{chosen?.close();if(decoder.state!=='closed')decoder.close();}
 }
 root.A4Offline={inspect,run,frameAt,nearestTime};
})(globalThis);

