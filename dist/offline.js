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
  return {track,samples,description,timelineOffset,duration,rotation,total:samples.filter(visible).length,
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
 root.A4Offline={inspect,run};
})(globalThis);
