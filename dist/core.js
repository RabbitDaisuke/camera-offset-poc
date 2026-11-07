/* A4 cylinder geometry and OpenCV bridge. Lengths are millimetres. */
(function (root) {
  'use strict';
  const spec = {version:'a4-cylinder-apriltag36h11-v4', width:297, height:210, dictionary:'DICT_APRILTAG_36h11',profiles:{standard:{firstId:0,scale:1,count:45,columns:9,size:24},safari:{firstId:200,scale:.83,count:45,columns:9,size:24},large:{firstId:300,scale:1,count:15,columns:5,size:40},largeSafari:{firstId:400,scale:.83,count:15,columns:5,size:40},singleRow:{firstId:500,scale:1,count:4,columns:4,size:60},singleRowSafari:{firstId:550,scale:.83,count:4,columns:4,size:60},twoRows:{firstId:510,scale:1,count:8,columns:4,size:60}}};
  const profileFor=id=>Object.values(spec.profiles).find(p=>id>=p.firstId&&id<p.firstId+p.count);
  const tagUV = id => {const p=profileFor(id);if(!p)throw Error('未知の模様です');const index=id-p.firstId,c=index%p.columns,r=Math.floor(index/p.columns),x=p.size===60?12+71*c:p.size===40?15+56.75*c:13.5+30.75*c,y=p.firstId===510?25+95*r:p.size===60?75:p.size===40?23+58*r:20+36.5*r;return [[x,y],[x+p.size,y],[x+p.size,y+p.size],[x,y+p.size]].map(([u,v])=>[148.5+(u-148.5)*p.scale,105+(v-105)*p.scale]);};
  function point(u,v,settings={},flat=false) {
    const scale=(settings.ruler||100)/100;
    u=148.5+(u-148.5)*scale;v=105+(v-105)*scale;
    if(flat)return [u,v,0];
    const circumference=297-(settings.overlap||0), radius=circumference/(2*Math.PI), theta=2*Math.PI*u/circumference;
    return [radius*Math.sin(theta),210-v,radius*Math.cos(theta)];
  }
  const mul=(a,b)=>[0,1,2].map(i=>a[i*3]*b[0]+a[i*3+1]*b[1]+a[i*3+2]*b[2]);
  const transpose=a=>[a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]];
  const matmul=(a,b)=>Array.from({length:9},(_,i)=>[0,1,2].reduce((s,k)=>s+a[Math.floor(i/3)*3+k]*b[k*3+i%3],0));
  function orientation(pose){const rt=transpose(pose.R),center=mul(rt,pose.t).map(v=>-v),forward=mul(rt,[0,0,1]),up=mul(rt,[0,-1,0]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),ref=[0,1,0].map((v,i)=>v-forward[i]*forward[1]),norm=Math.hypot(...ref);let roll=null;
    if(norm>1e-6){const u=ref.map(v=>v/norm),cross=[u[1]*up[2]-u[2]*up[1],u[2]*up[0]-u[0]*up[2],u[0]*up[1]-u[1]*up[0]];roll=Math.atan2(dot(cross,forward),dot(u,up))*180/Math.PI;}
    return {center,forward,up,yaw:Math.atan2(forward[0],forward[2])*180/Math.PI,pitch:Math.asin(Math.max(-1,Math.min(1,forward[1])))*180/Math.PI,roll};}
  function intrinsics(w,h,settings={}) {
    if(settings.calibration){const c=settings.calibration, sx=w/c.width,sy=h/c.height;return {K:[c.K[0]*sx,0,c.K[2]*sx,0,c.K[4]*sy,c.K[5]*sy,0,0,1],D:c.D.slice(),source:c.source==='a4-estimated'?'a4-estimated':'calibrated',autoLens:c.source==='a4-estimated'?c:undefined};}
    const f=w/(2*Math.tan((settings.fov||65)*Math.PI/360));return {K:[f,0,w/2,0,f,h/2,0,0,1],D:[0,0,0,0,0],source:'fov'};
  }
  function project(X,pose,K,D) {
    let p=mul(pose.R,X).map((v,i)=>v+pose.t[i]);
    if(p[2]<=0.1)return null;
    const x=p[0]/p[2],y=p[1]/p[2],r=x*x+y*y,rad=1+(D[0]||0)*r+(D[1]||0)*r*r+(D[4]||0)*r*r*r;
    const xd=x*rad+2*(D[2]||0)*x*y+(D[3]||0)*(r+2*x*x),yd=y*rad+(D[2]||0)*(r+2*y*y)+2*(D[3]||0)*x*y;
    return [K[0]*xd+K[2],K[4]*yd+K[5]];
  }
  class Engine {
    constructor(cv) {
      this.cv=cv; this.dictionary=cv.getPredefinedDictionary(cv.DICT_APRILTAG_36h11);
      this.params=new cv.aruco_DetectorParameters();this.params.cornerRefinementMethod=cv.CORNER_REFINE_APRILTAG;this.params.cornerRefinementWinSize=3;this.params.aprilTagQuadDecimate=1;
      this.params.minMarkerPerimeterRate=0.005;this.params.polygonalApproxAccuracyRate=0.05;this.params.errorCorrectionRate=0.45;this.params.adaptiveThreshWinSizeMax=73;
      this.refine=new cv.aruco_RefineParameters(10,3,true);
      this.detector=new cv.aruco_ArucoDetector(this.dictionary,this.params,this.refine);
      this.fastParams=new cv.aruco_DetectorParameters();for(const key of ['cornerRefinementMethod','cornerRefinementWinSize','minMarkerPerimeterRate','polygonalApproxAccuracyRate','errorCorrectionRate','adaptiveThreshWinSizeMax'])this.fastParams[key]=this.params[key];this.fastParams.aprilTagQuadDecimate=2;this.fastDetector=new cv.aruco_ArucoDetector(this.dictionary,this.fastParams,this.refine);
      this.tracks=new Map();
    }
    pattern(profile='standard') {
      const cv=this.cv,m=new cv.Mat(), codes=[];
      const first=spec.profiles[profile].firstId;
      try{for(let id=first;id<first+spec.profiles[profile].count;id++){cv.generateImageMarker(this.dictionary,id,8,m,1);codes.push(Array.from(m.data));}}finally{m.delete();}
      return codes;
    }
    detect(image,settings={},flat=false,region=null,maxEdge=0) {
      const started=performance.now(),cv=this.cv,src=cv.matFromImageData(image),gray=new cv.Mat(),corners=new cv.MatVector(),ids=new cv.Mat(),rejected=new cv.MatVector();let view=null,small=null;
      try {
        if(region)view=src.roi(new cv.Rect(region.x,region.y,region.width,region.height));
        cv.cvtColor(view||src,gray,cv.COLOR_RGBA2GRAY);let input=gray,sx=1,sy=1;
        if(maxEdge&&Math.max(gray.cols,gray.rows)>maxEdge){const scale=maxEdge/Math.max(gray.cols,gray.rows);small=new cv.Mat();cv.resize(gray,small,new cv.Size(Math.round(gray.cols*scale),Math.round(gray.rows*scale)),0,0,cv.INTER_AREA);input=small;sx=gray.cols/small.cols;sy=gray.rows/small.rows;}
        const prepared=performance.now();(settings.fastRoi?this.fastDetector:this.detector).detectMarkers(input,corners,ids,rejected);const detected=performance.now();
        const xy=a=>[0,1,2,3].map(j=>[a[j*2]*sx+(region?.x||0),a[j*2+1]*sy+(region?.y||0)]),markers=[];
        // Repeated IDs can belong to separate printed sheets. Keep every observation.
        for(let i=0;i<ids.rows;i++){const id=ids.data32S[i];if(!profileFor(id))continue;const m=corners.get(i),a=Array.from(m.data32F);m.delete();markers.push({id,corners:xy(a),object:tagUV(id).map(([u,v])=>point(u,v,settings,flat))});}
        const rejects=[];for(let i=0;i<Math.min(80,rejected.size());i++){const m=rejected.get(i);rejects.push(xy(Array.from(m.data32F)).flat());m.delete();}
        return {markers,rejected:rejects,timing:{prepareMs:prepared-started,detectMs:detected-prepared},region,search:region?'追跡':'探索'};
      } finally {small?.delete();view?.delete();src.delete();gray.delete();corners.delete();ids.delete();rejected.delete();}
    }
    boards(markers) {
      const cv=this.cv,groups=[];
      for(const profile of Object.values(spec.profiles)){
        let remaining=markers.filter(m=>m.id>=profile.firstId&&m.id<profile.firstId+profile.count);
        while(remaining.length>=2){
          const a=cv.matFromArray(remaining.length*4,1,cv.CV_32FC2,remaining.flatMap(m=>tagUV(m.id).flat())),b=cv.matFromArray(remaining.length*4,1,cv.CV_32FC2,remaining.flatMap(m=>m.corners.flat())),mask=new cv.Mat();let H;
          try{
            H=cv.findHomography(a,b,cv.RANSAC,3,mask,1000,.995);if(H.empty())break;
            const seen=new Set(),selected=remaining.filter((m,j)=>{if(seen.has(m.id)||[0,1,2,3].filter(k=>mask.data[j*4+k]).length<3)return false;seen.add(m.id);return true;});
            if(selected.length<2)break;
            const pts=selected.flatMap(m=>m.corners),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),width=Math.max(...xs)-x,height=Math.max(...ys)-y;
            groups.push({markers:selected,profile:profile.firstId,bounds:{x,y,width,height},center:[x+width/2,y+height/2]});
            remaining=remaining.filter(m=>!selected.includes(m));
          }finally{H?.delete();a.delete();b.delete();mask.delete();}
        }
      }
      return groups.sort((a,b)=>a.center[1]-b.center[1]);
    }
    focalFromPaper(group,w,h){
      if(group.markers.length<4)return null;const cv=this.cv,values=[];
      const solve=markers=>{const a=cv.matFromArray(markers.length*4,1,cv.CV_32FC2,markers.flatMap(m=>tagUV(m.id).flat())),b=cv.matFromArray(markers.length*4,1,cv.CV_32FC2,markers.flatMap(m=>m.corners.flat()));let H;
        try{H=cv.findHomography(a,b,0);if(H.empty())return null;const q=H.data64F,x=[q[0]-w/2*q[6],q[3]-h/2*q[6]],y=[q[1]-w/2*q[7],q[4]-h/2*q[7]],aa=x[0]*y[0]+x[1]*y[1],bb=q[6]*q[7],cc=x[0]**2+x[1]**2-y[0]**2-y[1]**2,dd=q[6]**2-q[7]**2,den=bb*bb+dd*dd;
          if(den<1e-20)return null;const f2=-(aa*bb+cc*dd)/den;if(f2<=0)return null;const f=Math.sqrt(f2),fov=2*Math.atan(w/(2*f))*180/Math.PI,n1=x[0]**2+x[1]**2+f2*q[6]**2,n2=y[0]**2+y[1]**2+f2*q[7]**2,residual=Math.hypot(aa+f2*bb,cc+f2*dd)/(n1+n2);
          if(fov<15||fov>130||residual>.025||f*Math.hypot(q[6],q[7])/Math.sqrt((n1+n2)/2)<.15)return null;return f;
        }finally{H?.delete();a.delete();b.delete();}};
      const full=solve(group.markers);if(!full)return null;values.push(full);for(let i=0;i<group.markers.length;i++){const f=solve(group.markers.filter((_,j)=>i!==j));if(f)values.push(f);}if(values.length<4)return null;values.sort((a,b)=>a-b);const f=values[Math.floor(values.length/2)],spread=(values.at(-1)-values[0])/f;if(spread>.16)return null;
      const K=[f,0,w/2,0,f,h/2,0,0,1],D=[0,0,0,0,0],markers=group.markers.map(m=>({...m,object:tagUV(m.id).map(([u,v])=>[u-148.5,105-v,0])})),p=this.pose(markers,K,D,true);if(!p||p.rms>2)return null;
      return {K,D,width:w,height:h,fov:2*Math.atan(w/(2*f))*180/Math.PI,spread,rms:p.rms,source:'a4-estimated',assumptions:'square pixels; principal point at image center; zero lens distortion',model:'a4-homography-focal',createdAt:new Date().toISOString()};
    }
    region(bounds,w,h,margin=1) {
      const pad=Math.max(24,Math.min(bounds.width,bounds.height)*.45)*margin,x=Math.max(0,Math.min(w-8,Math.floor(bounds.x-pad))),y=Math.max(0,Math.min(h-8,Math.floor(bounds.y-pad)));
      return {x,y,width:Math.max(8,Math.min(w,Math.ceil(bounds.x+bounds.width+pad))-x),height:Math.max(8,Math.min(h,Math.ceil(bounds.y+bounds.height+pad))-y)};
    }
    background(image,state,w,h){
      const started=performance.now(),cv=this.cv,src=cv.matFromImageData(image),gray=new cv.Mat();cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);src.delete();
      const result={status:'uncertain',points:0,dx:0,dy:0,ms:0};
      if(state.previousGray&&state.previousGray.rows===gray.rows&&state.previousGray.cols===gray.cols){
        const shifts=[];for(let y=12;y<gray.rows-12;y+=24)for(let x=12;x<gray.cols-12;x+=24){
          if(state.boards.some(b=>x/gray.cols*w>b.bounds.x-20&&x/gray.cols*w<b.bounds.x+b.bounds.width+20&&y/gray.rows*h>b.bounds.y-20&&y/gray.rows*h<b.bounds.y+b.bounds.height+20))continue;
          const patch=state.previousGray.roi(new cv.Rect(x-4,y-4,8,8)),search=gray.roi(new cv.Rect(x-8,y-8,16,16)),scores=new cv.Mat();
          try{const pixels=[];for(let py=y-4;py<y+4;py++)for(let px=x-4;px<x+4;px++)pixels.push(state.previousGray.data[py*gray.cols+px]);const mean=pixels.reduce((s,v)=>s+v,0)/64,variance=pixels.reduce((s,v)=>s+(v-mean)**2,0)/64;if(variance<100)continue;cv.matchTemplate(search,patch,scores,cv.TM_CCOEFF_NORMED);const best=cv.minMaxLoc(scores);if(best.maxVal>.85)shifts.push([best.maxLoc.x-4,best.maxLoc.y-4]);}finally{patch.delete();search.delete();scores.delete();}
        }
        if(shifts.length>=4){const median=a=>a.sort((a,b)=>a-b)[Math.floor(a.length/2)],dx=median(shifts.map(p=>p[0])),dy=median(shifts.map(p=>p[1]));result.points=shifts.length;result.dx=dx*w/gray.cols;result.dy=dy*h/gray.rows;const moving=shifts.filter(p=>Math.hypot(...p)>=1).length/shifts.length;result.status=moving>.65?'moving':moving<.25?'steady':'uncertain';if(result.status==='moving')for(const b of state.boards){b.bounds.x+=result.dx;b.bounds.y+=result.dy;}}
      }
      state.pendingGray?.delete();state.pendingGray=gray;result.ms=performance.now()-started;return result;
    }
    appearance(image,bounds,input){
      const cv=this.cv,x=Math.max(0,Math.floor(bounds.x-(input?.x||0))),y=Math.max(0,Math.floor(bounds.y-(input?.y||0))),w=Math.min(image.width-x,Math.ceil(bounds.width)),h=Math.min(image.height-y,Math.ceil(bounds.height));if(w<8||h<8||x>=image.width||y>=image.height)return null;
      const src=cv.matFromImageData(image),roi=src.roi(new cv.Rect(x,y,w,h)),gray=new cv.Mat(),small=new cv.Mat();cv.cvtColor(roi,gray,cv.COLOR_RGBA2GRAY);cv.resize(gray,small,new cv.Size(96,64),0,0,cv.INTER_AREA);src.delete();roi.delete();gray.delete();return small;
    }
    presence(image,board,input){
      if(!board.template)return 1;const cv=this.cv,b=board.bounds,ox=input?.x||0,oy=input?.y||0,x=Math.max(0,Math.floor(b.x-ox-b.width*.35)),y=Math.max(0,Math.floor(b.y-oy-b.height*.35)),w=Math.min(image.width-x,Math.ceil(b.width*1.7)),h=Math.min(image.height-y,Math.ceil(b.height*1.7));if(w<8||h<8)return 0;
      const src=cv.matFromImageData(image),roi=src.roi(new cv.Rect(x,y,w,h)),gray=new cv.Mat(),current=new cv.Mat(),reference=board.template.roi(new cv.Rect(4,4,88,56)),scores=new cv.Mat();
      try{cv.cvtColor(roi,gray,cv.COLOR_RGBA2GRAY);const width=Math.round(w/b.width*96),height=Math.round(h/b.height*64);if(width<88||height<56)return 0;cv.resize(gray,current,new cv.Size(width,height),0,0,cv.INTER_AREA);cv.matchTemplate(current,reference,scores,cv.TM_CCOEFF_NORMED);const best=cv.minMaxLoc(scores);if(best.maxVal>.68){b.x=ox+x+(best.maxLoc.x-4)*b.width/96;b.y=oy+y+(best.maxLoc.y-4)*b.height/64;}return best.maxVal;}finally{src.delete();roi.delete();gray.delete();current.delete();reference.delete();scores.delete();}
    }
    boardPose(p,board){const R=matmul(p.R,transpose(board.R)),offset=mul(R,board.t),t=p.t.map((v,i)=>v-offset[i]);return {...p,R,t,...orientation({R,t})};}
    relocalize(groups,state,k,local){
      const observations=groups.map(g=>({g,p:this.pose(local(g),k.K,k.D,true)})).filter(o=>o.p);let best=null;
      for(const observed of observations)for(const anchor of state.boards){if(observed.g.profile!==anchor.profile)continue;const pose=this.boardPose(observed.p,anchor),used=new Set(),matches=[];let error=0;
        for(const board of state.boards){const predicted=project(board.t,pose,k.K,k.D);if(!predicted)continue;let found=null,distance=Infinity;for(const o of observations){if(used.has(o)||o.g.profile!==board.profile)continue;const center=project([0,0,0],o.p,k.K,k.D),d=Math.hypot(center[0]-predicted[0],center[1]-predicted[1]);if(d<distance){distance=d;found=o;}}
          if(found&&distance<Math.max(20,found.g.bounds.width*.2)){used.add(found);matches.push({board,observed:found});error+=distance;}}
        const ambiguous=state.boards.filter(b=>b.profile===anchor.profile).length>1;if(matches.length<(ambiguous?2:1))continue;
        if(!best||matches.length>best.matches.length||matches.length===best.matches.length&&error<best.error)best={pose,matches,error,used};
      }
      if(!best)return null;
      for(const match of best.matches)match.board.bounds=match.observed.g.bounds;
      const rt=transpose(best.pose.R);for(const o of observations)if(!best.used.has(o)){const id=Math.max(...state.boards.map(b=>b.id))+1;state.boards.push({id,profile:o.g.profile,bounds:o.g.bounds,R:matmul(rt,o.p.R),t:mul(rt,o.p.t.map((v,i)=>v-best.pose.t[i]))});}
      const chosen=best.matches.find(m=>m.board.id===state.anchor)||best.matches[0];state.active=state.boards.indexOf(chosen.board);return chosen.observed.g;
    }
    plane(image,settings,k,motionImage) {
      const input=settings.inputRegion,w=input?.sourceWidth||image.width,h=input?.sourceHeight||image.height;
      const key=settings.streamKey||'single',previous=this.tracks.get(key),valid=previous&&previous.width===w&&previous.height===h;
      const state=valid?previous:{width:w,height:h,misses:0,boards:[],active:0};
      if(settings.autoFov&&!settings.calibration&&state.autoLens)Object.assign(k,{K:state.autoLens.K,D:state.autoLens.D,source:'a4-estimated',autoLens:state.autoLens});
      if(settings.forceSearch)state.wantFullSearch=true;const searchDue=state.boards.length&&(state.wantFullSearch||state.lastMeasured&&performance.now()-state.lastMeasured>=30000&&performance.now()-(state.lastFullSearch||0)>=30000);
      if(searchDue&&input){state.wantFullSearch=true;this.tracks.set(key,state);return {markers:[],rejected:[],pose:null,requestFullSearch:true,nextBounds:null,boardCount:state.boards.length,boards:state.boards.map(b=>({id:b.id,profile:b.profile,bounds:b.bounds,R:b.R,t:b.t})),reference:state.anchor,held:!!state.lastPose&&!state.movedSincePose,holdAgeMs:performance.now()-state.lastMeasured,movedSincePose:!!state.movedSincePose,lost:true,search:'全体を確認',timing:{prepareMs:0,detectMs:0}};}
      const motion=motionImage?this.background(motionImage,state,w,h):{status:'uncertain',points:0,ms:0};if(motion.status==='moving')state.movedSincePose=true;
      const active=searchDue?null:state.boards[state.active],region=!input&&(active||state.pendingBounds)?this.region(active?.bounds||state.pendingBounds,w,h,state.misses?2:1):null;
      if(searchDue){state.lastFullSearch=performance.now();state.wantFullSearch=false;}
      const presence=active&&state.misses?this.presence(image,active,input):1;
      const skipped=active&&state.misses&&presence<.68&&motion.status!=='moving'&&state.misses%15!==0;
      let d=skipped?{markers:[],rejected:[],timing:{prepareMs:0,detectMs:0},search:'復帰待ち'}:this.detect(image,{...settings,fastRoi:!!active&&active.markerPx>=32&&!state.misses},false,region,0);
      if(input){d.markers.forEach(m=>m.corners=m.corners.map(p=>[p[0]+input.x,p[1]+input.y]));d.rejected=d.rejected.map(a=>a.map((v,i)=>v+(i%2?input.y:input.x)));d.region=input;d.search='追跡';}
      const scale=(settings.ruler||100)/100,local=g=>g.markers.map(m=>({...m,object:tagUV(m.id).map(([u,v])=>[(u-148.5)*scale,(105-v)*scale,0])}));
      let groups=this.boards(d.markers),chosen,board=active;
      if(settings.autoFov&&!settings.calibration){
        if(!state.autoLens){const candidates=groups.map(g=>this.focalFromPaper(g,w,h)).filter(Boolean).sort((a,b)=>a.spread-b.spread);if(candidates.length){const best=candidates[0],agree=candidates.every(c=>Math.abs(c.K[0]/best.K[0]-1)<.15);if(agree)state.autoLens=best;}}
        if(!state.autoLens){k.source='unresolved';state.pendingBounds=groups[0]?.bounds||state.pendingBounds;this.tracks.set(key,state);state.pendingGray?.delete();state.pendingGray=null;return {...d,pose:null,nextBounds:state.pendingBounds||null,boards:[],boardCount:0,lost:true,autoFovPending:true,retryAfterMs:1000};}
        Object.assign(k,{K:state.autoLens.K,D:state.autoLens.D,source:'a4-estimated',autoLens:state.autoLens});state.pendingBounds=null;
      }
      if(searchDue){chosen=this.relocalize(groups,state,k,local);board=state.boards[state.active];}
      else if(active){const center=[active.bounds.x+active.bounds.width/2,active.bounds.y+active.bounds.height/2];chosen=groups.filter(g=>g.profile===active.profile).sort((a,b)=>Math.hypot(a.center[0]-center[0],a.center[1]-center[1])-Math.hypot(b.center[0]-center[0],b.center[1]-center[1]))[0];if(chosen&&Math.hypot(chosen.center[0]-center[0],chosen.center[1]-center[1])>Math.max(active.bounds.width,active.bounds.height))chosen=null;}
      else if(groups.length){
        const index=Number(settings.referenceIndex)||0,anchor=groups[index],anchorPose=anchor&&this.pose(local(anchor),k.K,k.D,true);
        if(anchorPose){const rt=transpose(anchorPose.R);state.anchor=index;state.boards=groups.map((g,i)=>{const p=i===index?anchorPose:this.pose(local(g),k.K,k.D,true);return p?{id:i,profile:g.profile,bounds:g.bounds,R:matmul(rt,p.R),t:mul(rt,p.t.map((v,j)=>v-anchorPose.t[j])),template:this.appearance(image,g.bounds,input)}:null;}).filter(Boolean);state.active=state.boards.findIndex(b=>b.id===index);board=state.boards[state.active];chosen=anchor;}
      }
      let pose=null,selected=[];
      if(chosen&&board){const points=local(chosen),p=this.pose(points,k.K,k.D,true);if(p){const R=matmul(p.R,transpose(board.R)),offset=mul(R,board.t),t=p.t.map((v,i)=>v-offset[i]);selected=points.map(m=>({...m,object:m.object.map(X=>mul(board.R,X).map((v,i)=>v+board.t[i]))}));pose={...p,R,t,...orientation({R,t})};pose.projections=selected.map(m=>({id:m.id,points:m.object.map(X=>project(X,pose,k.K,k.D))}));state.lastPose=pose;state.lastMeasured=performance.now();state.movedSincePose=false;state.misses=0;
          for(const b of state.boards){const profile=profileFor(b.profile),corners=Array.from({length:profile.count},(_,i)=>tagUV(profile.firstId+i)).flat().map(([u,v])=>[(u-148.5)*scale,(105-v)*scale,0]).map(X=>mul(b.R,X).map((v,i)=>v+b.t[i])).map(X=>project(X,pose,k.K,k.D)).filter(Boolean);if(corners.length){const xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]);b.bounds={x:Math.min(...xs),y:Math.min(...ys),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys)};}}
          board.markerPx=Math.min(...chosen.markers.map(m=>Math.min(...[0,1,2,3].map(i=>Math.hypot(m.corners[i][0]-m.corners[(i+1)%4][0],m.corners[i][1]-m.corners[(i+1)%4][1])))));
          state.measuredCount=(state.measuredCount||0)+1;if(state.measuredCount%30===0){const patch=this.appearance(image,board.bounds,input);if(patch){board.template?.delete();board.template=patch;}}
        }}
      if(!pose){state.misses++;if(state.boards.length>1&&state.misses%3===0)state.active=(state.active+1)%state.boards.length;}
      if(state.pendingGray){if(pose||motion.status==='moving'){state.previousGray?.delete();state.previousGray=state.pendingGray;}else state.pendingGray.delete();state.pendingGray=null;}
      this.tracks.set(key,state);if(this.tracks.size>20){const old=this.tracks.keys().next().value,entry=this.tracks.get(old);entry.previousGray?.delete();entry.boards?.forEach(b=>b.template?.delete());this.tracks.delete(old);}
      const next=state.boards[state.active],held=!!state.lastPose&&!state.movedSincePose&&!pose;
      return {...d,markers:selected,pose,boardCount:state.boards.length,boardBounds:pose?board.bounds:null,nextBounds:next?.bounds||null,searchPadding:state.misses?2:1,boards:state.boards.map(b=>({id:b.id,profile:b.profile,bounds:b.bounds,R:b.R,t:b.t})),activeBoard:board?.id,reference:state.anchor,held,holdAgeMs:held?performance.now()-state.lastMeasured:0,movedSincePose:!!state.movedSincePose,motion,templateScore:presence,templateOnly:!!skipped,lost:!pose};
    }
    pose(markers,K,D,planar=false) {
      if(markers.length<2)return null;
      const cv=this.cv, resources=[];const own=x=>(resources.push(x),x);
      const all=markers.flatMap(m=>m.object),uv=markers.flatMap(m=>m.corners);
      const obj=own(cv.matFromArray(all.length,1,cv.CV_32FC3,all.flat())),img=own(cv.matFromArray(uv.length,1,cv.CV_32FC2,uv.flat())),km=own(cv.matFromArray(3,3,cv.CV_64F,K)),dm=own(cv.matFromArray(D.length,1,cv.CV_64F,D)),rv=own(new cv.Mat()),tv=own(new cv.Mat()),inliers=own(new cv.Mat()),rot=own(new cv.Mat());
      try {
        let ok=planar?cv.solvePnP(obj,img,km,dm,rv,tv,false,cv.SOLVEPNP_IPPE):cv.solvePnPRansac(obj,img,km,dm,rv,tv,false,180,3.5,0.995,inliers,cv.SOLVEPNP_EPNP);
        if(!ok)return null;
        const readPose=()=>{cv.Rodrigues(rv,rot);return {R:Array.from(rot.data64F),t:Array.from(tv.data64F),rvec:Array.from(rv.data64F)};};
        let pose=readPose();
        const error=m=>Math.sqrt(m.object.reduce((s,p,i)=>{const q=project(p,pose,K,D);return s+(q?Math.hypot(q[0]-m.corners[i][0],q[1]-m.corners[i][1])**2:1e10);},0)/4);
        let accepted=markers.filter(m=>error(m)<5);
        if(accepted.length<2)return null;
        const o2=own(cv.matFromArray(accepted.length*4,1,cv.CV_32FC3,accepted.flatMap(m=>m.object.flat()))),i2=own(cv.matFromArray(accepted.length*4,1,cv.CV_32FC2,accepted.flatMap(m=>m.corners.flat())));
        cv.solvePnPRefineLM(o2,i2,km,dm,rv,tv);pose=readPose();
        const errors=markers.map(m=>({id:m.id,rms:error(m),accepted:accepted.includes(m)}));
        const rms=Math.sqrt(errors.filter(e=>e.accepted).reduce((s,e)=>s+e.rms**2,0)/accepted.length);
        const rt=transpose(pose.R),center=mul(rt,pose.t).map(x=>-x),forward=mul(rt,[0,0,1]),up=mul(rt,[0,-1,0]);
        if(!center.every(Number.isFinite)||rms>5)return null;
        return {...pose,...orientation(pose),rms,errors,accepted:accepted.map(m=>m.id),projections:markers.map(m=>({id:m.id,points:m.object.map(p=>project(p,pose,K,D))}))};
      } finally {resources.reverse().forEach(x=>x.delete());}
    }
    process(image,settings,flat=false,motionImage=null) {
      const t=performance.now(),width=settings.inputRegion?.sourceWidth||image.width,height=settings.inputRegion?.sourceHeight||image.height,k=intrinsics(width,height,settings);
      if(!flat&&settings.mode==='plane'){const d=this.plane(image,settings,k,motionImage);return {...d,...k,width,height,ms:performance.now()-t,flat};}
      const d=this.detect(image,settings,flat);
      // Legacy cylindrical reconstruction expects one observation of each ID.
      d.markers=d.markers.filter((m,i,a)=>a.findIndex(n=>n.id===m.id)===i);
      let pose=null;
      if(!flat&&settings.geometry){const map=new Map();for(const p of settings.geometry){if(!p.accepted)continue;if(!map.has(p.id))map.set(p.id,[]);map.get(p.id)[p.corner]=p.point;}const mapped=d.markers.filter(m=>map.has(m.id)&&[0,1,2,3].every(i=>map.get(m.id)[i])).map(m=>({...m,object:map.get(m.id)}));pose=this.pose(mapped,k.K,k.D);}
      else if(!flat&&settings.mode==='plane'){
        const profile=spec.profiles[settings.profile||'large'],scale=(settings.ruler||100)/100;
        const planar=d.markers.filter(m=>m.id>=profile.firstId&&m.id<profile.firstId+profile.count).map(m=>({...m,object:tagUV(m.id).map(([u,v])=>[(u-148.5)*scale,(105-v)*scale,0])}));
        pose=this.pose(planar,k.K,k.D,true);
      }
      else if(!flat&&settings.mode==='cylinder')pose=this.pose(d.markers,k.K,k.D);
      return {...d,pose,...k,width:image.width,height:image.height,ms:performance.now()-t,flat};
    }
    calibrate(samples) {
      if(samples.length<6)throw Error('A4を傾けて、6枚以上撮影してください。');
      const cv=this.cv,resources=[],own=x=>(resources.push(x),x),ov=own(new cv.MatVector()),iv=own(new cv.MatVector()),K=own(cv.Mat.eye(3,3,cv.CV_64F)),D=own(cv.Mat.zeros(5,1,cv.CV_64F)),rs=own(new cv.MatVector()),ts=own(new cv.MatVector()),si=own(new cv.Mat()),se=own(new cv.Mat()),pe=own(new cv.Mat());
      try{
        for(const s of samples){const o=cv.matFromArray(s.markers.length*4,1,cv.CV_32FC3,s.markers.flatMap(m=>m.object.flat())),i=cv.matFromArray(s.markers.length*4,1,cv.CV_32FC2,s.markers.flatMap(m=>m.corners.flat()));ov.push_back(o);iv.push_back(i);o.delete();i.delete();}
        const rms=cv.calibrateCameraExtended(ov,iv,new cv.Size(samples[0].width,samples[0].height),K,D,rs,ts,si,se,pe,cv.CALIB_FIX_K3,new cv.TermCriteria(cv.TermCriteria_COUNT+cv.TermCriteria_EPS,60,1e-8));
        const k=Array.from(K.data64F),d=Array.from(D.data64F),w=samples[0].width,h=samples[0].height;
        if(!k.every(Number.isFinite)||!d.every(Number.isFinite)||!Number.isFinite(rms)||k[0]<w*.2||k[0]>w*5||k[4]<h*.2||k[4]>w*5||Math.abs(k[2]-w/2)>w*.35||Math.abs(k[5]-h/2)>h*.35||rms>5)throw Error('撮影位置と傾きを変えて撮り直してください。');
        return {width:w,height:h,K:k,D:d,rms,viewErrors:Array.from(pe.data64F),samples:samples.length,createdAt:new Date().toISOString(),model:'opencv-pinhole-brown5',dictionary:spec.dictionary};
      }finally{resources.reverse().forEach(x=>x.delete());}
    }
  }
  root.A4Core={spec,profileFor,tagUV,point,project,intrinsics,mul,transpose,Engine};
  if(typeof module!=='undefined')module.exports=root.A4Core;
})(typeof self!=='undefined'?self:globalThis);
