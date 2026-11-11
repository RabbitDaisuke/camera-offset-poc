(function(root){
 'use strict';
 function pdf(codes,key){
  const profile=root.A4Core.spec.profiles[key],pt=72/25.4,n=x=>Number(x.toFixed(5)),commands=['0 g'];
  const rect=(x,y,w,h)=>commands.push(`${n(x*pt)} ${n((210-y-h)*pt)} ${n(w*pt)} ${n(h*pt)} re f`);
  const text=(x,y,size,value)=>commands.push(`BT /F1 ${n(size*pt)} Tf ${n(x*pt)} ${n((210-y)*pt)} Td (${value}) Tj ET`);
  codes.forEach((bits,index)=>{const id=profile.firstId+index,[x,y]=root.A4Core.tagUV(id)[0],cell=profile.size*profile.scale/8;for(let r=0;r<8;r++)for(let c=0;c<8;c++)if(bits[r*8+c]===0)rect(x+c*cell,y+r*cell,cell,cell);text(x+4*cell-2.5,y+8*cell+4,2.8,String(id));});
  text(13,15,2.5,'A4 / AprilTag 36h11 / PRINT 100%');text(201,198,2.5,'100 mm');
  commands.push(`${n(.25*pt)} w ${n(98.5*pt)} ${n(13*pt)} m ${n(198.5*pt)} ${n(13*pt)} l S`);
  for(const x of [98.5,198.5])commands.push(`${n(x*pt)} ${n(11*pt)} m ${n(x*pt)} ${n(15*pt)} l S`);
  const stream=commands.join('\n')+'\n',objects=[`<< /Type /Catalog /Pages 2 0 R /ViewerPreferences << /PrintScaling /None >> >>`,`<< /Type /Pages /Kids [3 0 R] /Count 1 >>`,`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${n(297*pt)} ${n(210*pt)}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>`,`<< /Length ${stream.length} >>\nstream\n${stream}endstream`,`<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>`];
  let output='%PDF-1.4\n',offsets=[0];objects.forEach((object,i)=>{offsets.push(output.length);output+=`${i+1} 0 obj\n${object}\nendobj\n`;});const xref=output.length;output+=`xref\n0 ${objects.length+1}\n0000000000 65535 f \n`;for(const offset of offsets.slice(1))output+=`${String(offset).padStart(10,'0')} 00000 n \n`;output+=`trailer\n<< /Size ${objects.length+1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;return new TextEncoder().encode(output);
 }
 root.A4Print={pdf};if(typeof module!=='undefined')module.exports=root.A4Print;
})(typeof self!=='undefined'?self:globalThis);

