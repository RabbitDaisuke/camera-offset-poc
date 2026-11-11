(function(root){
 'use strict';
 const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>scale(a,1/(Math.hypot(...a)||1));
 class Scene {
  constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext('2d');this.yaw=.65;this.pitch=.42;this.zoom=1;this.cameras=[];this.settings={};let drag=null;
   canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
   canvas.addEventListener('pointermove',e=>{if(!drag)return;this.yaw-=(e.clientX-drag[0])*.007;this.pitch=Math.max(-.1,Math.min(1.55,this.pitch+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];this.draw()});
   canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
   this.viewport=canvas.closest('.scene-viewport');this.viewport.addEventListener('scroll',()=>this.draw());
   canvas.addEventListener('wheel',e=>{if(e.shiftKey||Math.abs(e.deltaX)>Math.abs(e.deltaY))return;e.preventDefault();this.zoom=Math.max(.3,Math.min(4,this.zoom*Math.exp(e.deltaY*.001)));this.draw()},{passive:false});new ResizeObserver(()=>{this.centerScroll();this.draw()}).observe(this.viewport);
  }
  set(cameras,settings){this.cameras=cameras;this.settings=settings;this.draw()}
  centerScroll(){this.viewport.scrollLeft=(this.viewport.scrollWidth-this.viewport.clientWidth)/2;}
  reset(top=false){this.yaw=top?0:.65;this.pitch=top?1.54:.42;this.zoom=1;this.centerScroll();this.draw()}
  draw(){const c=this.canvas,ctx=this.ctx,w=c.clientWidth,h=c.clientHeight;if(!w||!h)return;const dpr=Math.min(devicePixelRatio||1,2);c.width=w*dpr;c.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
   const cloud=(this.settings.points||[]).filter(p=>p.accepted),boards=this.cameras.find(c=>c.boards?.length)?.boards||[],positions=[[0,0,0],...boards.map(b=>scale(b.t,.001)),...this.cameras.map(c=>scale(c.pose.center,.001)),...cloud.map(p=>scale(p.point,.001))],target=positions.reduce((sum,p)=>add(sum,scale(p,1/positions.length)),[0,0,0]);let range=.4;for(const p of positions)range=Math.max(range,Math.hypot(...sub(p,target))*1.1);const distance=range*2.4*this.zoom,eye=add(target,[Math.sin(this.yaw)*Math.cos(this.pitch)*distance,Math.sin(this.pitch)*distance,Math.cos(this.yaw)*Math.cos(this.pitch)*distance]),forward=norm(sub(target,eye)),right=norm(cross(forward,[0,1,0])),up=cross(right,forward),f=h*1.2;
   const pan=this.viewport.scrollLeft-(this.viewport.scrollWidth-this.viewport.clientWidth)/2;
   const screen=p=>{const q=sub(p,eye),z=dot(q,forward);return z>.005?[w/2-pan+dot(q,right)*f/z,h/2-dot(q,up)*f/z,z]:null};
   const line=(a,b,color,width=1,dash=[])=>{a=screen(a);b=screen(b);if(!a||!b)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.setLineDash([])};
   const label=(p,text,color='#6f8375')=>{const q=screen(p);if(!q)return;ctx.font='10px system-ui';const tw=ctx.measureText(text).width;ctx.fillStyle='#fafbf8e8';ctx.fillRect(q[0]-tw/2-4,q[1]-12,tw+8,17);ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(text,q[0],q[1])};
   const step=range>2?.5:range>1?.2:.1,extent=Math.ceil(range/step)*step;
   for(let i=-extent;i<=extent+.0001;i+=step){line([i,0,-extent],[i,0,extent],'#e4ebe2');line([-extent,0,i],[extent,0,i],'#e4ebe2')}
   if(this.settings.mode==='plane'){const s=(this.settings.ruler||100)/100,list=boards.length?boards:[{id:0,R:[1,0,0,0,1,0,0,0,1],t:[0,0,0]}];for(const b of list){const world=p=>add(A4Core.mul(b.R,scale(p,s)),scale(b.t,.001)),points=[[-.1485,.105,0],[.1485,.105,0],[.1485,-.105,0],[-.1485,-.105,0]].map(world);for(let i=0;i<4;i++)line(points[i],points[(i+1)%4],'#258774',1.5);label(world([0,.13,0]),'A4 '+(b.id+1));}}
   if(this.settings.mode==='cylinder'){const radius=(297-(this.settings.overlap||0))/(2*Math.PI*1000);
   for(let y of [0,.0525,.105,.1575,.21])for(let i=0;i<64;i++){const a=i*Math.PI/32,b=(i+1)*Math.PI/32;line([radius*Math.sin(a),y,radius*Math.cos(a)],[radius*Math.sin(b),y,radius*Math.cos(b)],y===0||y===.21?'#258774':'#a9c6b6',y===0||y===.21?1.5:1)}
   for(let i=0;i<16;i++){const a=i*Math.PI/8;line([radius*Math.sin(a),0,radius*Math.cos(a)],[radius*Math.sin(a),.21,radius*Math.cos(a)],'#95b9a6')}
   line([0,0,radius],[0,.21,radius],'#147c67',2);label([0,.24,0],'A4');}
   for(const p of cloud){const q=screen(scale(p.point,.001));if(!q)continue;ctx.fillStyle='#258774';ctx.beginPath();ctx.arc(q[0],q[1],2.2,0,Math.PI*2);ctx.fill();const next=cloud.find(v=>v.id===p.id&&v.corner===(p.corner+1)%4);if(next)line(scale(p.point,.001),scale(next.point,.001),'#258774',1.3);}
   const axis=Math.min(.3,range*.65);line([0,0,0],[axis,0,0],'#c47769',1.5);line([0,0,0],[0,axis,0],'#699c7b',1.5);line([0,0,0],[0,0,axis],'#7596bb',1.5);label([axis,0,0],'X','#b27064');label([0,axis,0],'Y','#598b69');label([0,0,axis],'Z','#6084ae');
   const colors=['#4779bd','#bd8752','#9b6fbd','#bc6b80','#4b99a5'];
   this.cameras.forEach((cam,index)=>{const p=cam.pose,C=scale(p.center,.001),rt=A4Core.transpose(p.R),z=Math.max(.055,range*.08),fx=cam.K[0],fy=cam.K[4],sx=cam.width/(2*fx)*z,sy=cam.height/(2*fy)*z;
    const world=v=>add(C,A4Core.mul(rt,v)),corners=[[-sx,-sy,z],[sx,-sy,z],[sx,sy,z],[-sx,sy,z]].map(world),color=colors[index%colors.length];
    for(let j=0;j<4;j++){line(C,corners[j],color,1.5);line(corners[j],corners[(j+1)%4],color,1.5)}line(C,add(C,scale(p.forward,z*2)),color,1.2);line(C,[C[0],0,C[2]],color,1,[3,4]);label(add(C,[0,z,0]),cam.name,color);
   });
   ctx.textAlign='left';ctx.fillStyle='#99a69b';ctx.font='9px system-ui';ctx.fillText('グリッド '+step.toFixed(1)+' m',16,h-14);
  }
 }
 root.A4Scene=Scene;
})(window);

