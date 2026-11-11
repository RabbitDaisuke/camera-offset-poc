// Every vertex is an existing stereo correspondence. Triangle interiors are
// an estimate; long 3D edges and folds in the second image remain disconnected.
export function connectSurface(points,uv,{step=2,maxEdge=.2,method='local'}={}){
 const grid=new Map(),count=points.length/3;
 for(let i=0;i<count;i++)grid.set(`${uv[4*i]},${uv[4*i+1]}`,i);
 const edge2=(a,b)=>{let d=0;for(let k=0;k<3;k++)d+=(points[3*a+k]-points[3*b+k])**2;return d;};
 const triangles=[],limit2=maxEdge*maxEdge;let rejected=0;
 const add=(a,b,c)=>{
  if(a===undefined||b===undefined||c===undefined)return;
  if(edge2(a,b)>limit2||edge2(b,c)>limit2||edge2(c,a)>limit2){rejected++;return;}
  // Reject folds in the second observed image as well. Near-zero or reversed
  // right-image area cannot support this oriented surface cell.
  const left=(uv[4*b]-uv[4*a])*(uv[4*c+1]-uv[4*a+1])-(uv[4*b+1]-uv[4*a+1])*(uv[4*c]-uv[4*a]);
  const right=(uv[4*b+2]-uv[4*a+2])*(uv[4*c+3]-uv[4*a+3])-(uv[4*b+3]-uv[4*a+3])*(uv[4*c+2]-uv[4*a+2]);
  if(left*right<=0||Math.abs(right)<Math.abs(left)*.1){rejected++;return;}
  triangles.push(a,b,c);
 };
 if(method==='local'&&count>=3){const coords=new Float64Array(count*2);for(let i=0;i<count;i++){coords[2*i]=uv[4*i];coords[2*i+1]=uv[4*i+1];}const mesh=new Delaunator(coords);for(let i=0;i<mesh.triangles.length;i+=3)add(mesh.triangles[i],mesh.triangles[i+1],mesh.triangles[i+2]);}
 else {const cells=new Set();
 for(let i=0;i<count;i++){
  const x=uv[4*i],y=uv[4*i+1];
  for(const [dx,dy]of [[0,0],[-step,0],[0,-step],[-step,-step]])cells.add(`${x+dx},${y+dy}`);
 }
 for(const key of cells){
  const [x,y]=key.split(',').map(Number),a=grid.get(key),b=grid.get(`${x+step},${y}`),c=grid.get(`${x},${y+step}`),d=grid.get(`${x+step},${y+step}`);
  const valid=[a,b,d,c].filter(i=>i!==undefined);
  if(valid.length===3){add(...valid);continue;}
  if(valid.length!==4)continue;
  if(edge2(a,d)<=edge2(b,c)){add(a,b,d);add(a,d,c);}else{add(a,b,c);add(b,d,c);}
 }
 }
 const indices=new Uint32Array(triangles),used=new Set(indices);
 return {indices,stats:{vertices:count,connectedVertices:used.size,triangles:indices.length/3,rejected,maxEdge,step,method:method==='local'?'Delaunay in left image; bounded 3D edges; right-image fold rejection':'observed stereo grid',interior:'estimated between measured vertices',addedVertices:0}};
}
import Delaunator from './vendor/delaunator.js';
