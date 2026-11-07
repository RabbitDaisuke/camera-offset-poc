importScripts('core.js','vendor/opencv.js');
let engine;
function ready(){engine=new A4Core.Engine(cv);postMessage({type:'ready',patterns:Object.fromEntries(Object.keys(A4Core.spec.profiles).map(key=>[key,engine.pattern(key)]))});}
if(cv.Mat)ready();else cv.onRuntimeInitialized=ready;
self.onmessage=({data:m})=>{try{if(!engine)throw Error('準備中です');const result=m.type==='calibrate'?engine.calibrate(m.samples):engine.process(m.image,m.settings,m.flat,m.motionImage);postMessage({request:m.request,result});}catch(e){postMessage({request:m.request,error:typeof e==='number'?'画像を読み直してください。':String(e.message||e)});}};
