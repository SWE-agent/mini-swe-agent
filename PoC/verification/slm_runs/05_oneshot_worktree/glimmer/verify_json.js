const fs = require('fs');
const path = 'observatory/diagram/agent-architecture.json';
const data = JSON.parse(fs.readFileSync(path, 'utf8'));
let ok = true;
if (!data.meta || !Array.isArray(data.meta.viewBox)) { console.log('viewBox missing'); ok=false; }
else console.log('viewBox present:', data.meta.viewBox);
if (!Array.isArray(data.components)) { console.log('components missing'); ok=false; }
else {
  for (const c of data.components) {
    if (!Array.isArray(c.pos) || !Array.isArray(c.size)) {
      console.log('component missing pos/size:', c.id);
      ok=false;
    }
  }
  console.log('components checked:', data.components.length);
}
const final = data.components.find(c=>c.id==='final_output');
if (final) {
  console.log('final_output pos:', final.pos, 'size:', final.size);
  if (JSON.stringify(final.pos)!=='[400,500]') console.log('pos mismatch');
  if (JSON.stringify(final.size)!=='[200,72]') console.log('size mismatch');
} else console.log('final_output not found');
console.log('verification', ok?'PASSED':'FAILED');
