import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Presentation, PresentationFile, BUILD, metadata, tableOwners, chartOwners } from './layout.mjs';
import { member1 } from './member1.mjs';
import { member2 } from './member2.mjs';
import { member3 } from './member3.mjs';
import { appendix } from './appendix.mjs';

await fs.mkdir(`${BUILD}/renders`, { recursive: true });
const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
member1(p);
member2(p);
member3(p);
appendix(p);
assert.equal(p.slides.items.length, 32);
assert.equal(metadata.slice(0, 9).reduce((a, b) => a + b.seconds, 0), 540);
assert.equal(metadata.slice(9, 20).reduce((a, b) => a + b.seconds, 0), 720);
assert.equal(metadata.slice(20, 28).reduce((a, b) => a + b.seconds, 0), 540);
assert.equal(metadata.slice(28).reduce((a, b) => a + b.seconds, 0), 0);
await (await PresentationFile.exportPptx(p)).save(`${BUILD}/candidate.pptx`);
await fs.writeFile(`${BUILD}/presentation.json`, JSON.stringify(p.toProto()));
await fs.writeFile(`${BUILD}/requirements.json`, JSON.stringify({ metadata, tableOwners: [...new Set(tableOwners)], chartOwners: [...new Set(chartOwners)] }, null, 2));
const selected = process.argv.slice(2).map(Number);
for (const [i, s] of p.slides.items.entries()) {
  if (selected.length && !selected.includes(i + 1)) continue;
  const png = await p.export({ slide: s, format: 'png', scale: 1.5 });
  await fs.writeFile(`${BUILD}/renders/slide-${String(i + 1).padStart(2, '0')}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await s.export({ format: 'layout' });
  await fs.writeFile(`${BUILD}/renders/slide-${String(i + 1).padStart(2, '0')}.layout.json`, await layout.text());
  console.log(`Rendered ${i + 1}/32`);
}
console.log('Draft ready. Main-slide timing: 1,800 seconds.');
