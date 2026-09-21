// Every page carries the same site header, byte for byte (node docs/header-check.js).
const fs = require('fs'), path = require('path');
const dir = __dirname, seen = new Map(); let bad = 0;
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.html'))) {
  const s = fs.readFileSync(path.join(dir, f), 'utf8');
  const m = s.match(/<header class="site-header">[\s\S]*?<\/header>/);
  if (!m) { if (!/http-equiv="refresh"/.test(s)) { console.log('no header:', f); bad++; } continue; }
  seen.set(m[0], (seen.get(m[0]) || []).concat(f));
}
if (seen.size > 1) { bad++; for (const [, files] of seen) console.log(files.length + ' page(s):', files.join(' ')); }
console.log(bad ? 'header check FAILED' : `header check passed: one header on ${[...seen.values()].flat().length} pages`);
process.exit(bad ? 1 : 0);
