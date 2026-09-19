const fs=require('fs'),vm=require('vm');
const files=process.argv.slice(2); let bad=0;
for(const f of files){const h=fs.readFileSync(f,'utf8');const re=/<script\b([^>]*)>([\s\S]*?)<\/script>/g;let m,i=0,ok=0;
 while((m=re.exec(h))){i++; if(/src=/.test(m[1])) continue; try{new vm.Script(m[2],{filename:f+'#'+i}); ok++;}catch(e){bad++;console.log('PARSE FAIL',f,'#'+i,e.message);}}
 console.log(f.split('/').pop(), 'scripts:',i,'inline parsed:',ok);}
process.exit(bad?1:0);
