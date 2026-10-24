// Node harness: replay a simulated session against the wasm module and time every step.
//
//   node web/bench.mjs [path/to/johari.wasm] [nItems=127] [nActive=100] [maxSteps=400] [seed=7]

import { stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import { shippedModule } from './client.js';
import { competitionRanks } from './johari.js';
import { loadModule, makeSim, moduleArg, numberArg, quantiles } from './sim.mjs';

const args = process.argv.slice(2);
const wasmPath = moduleArg(args, shippedModule());
const [nItemsArg, nActiveArg, stepsArg, seedArg] = args;
const maxSteps = numberArg(stepsArg, 'maxSteps', { fallback: 400, min: 1 });

const wasmFile = fileURLToPath(wasmPath);
const wasmBytes = (await stat(wasmFile)).size;
const t0 = performance.now();
const johari = await loadModule(wasmPath);
const loadMs = performance.now() - t0;

const sim = makeSim(nItemsArg, nActiveArg, seedArg);
const { nItems, nActive, truth } = sim;
using session = sim.start(johari);

const times = [];
const values = [];
let stoppedAt = null;
let ran = 0;
for (const { step, p, ms, stopping } of sim.steps(session, { maxSteps })) {
  ran = step + 1;
  times.push(ms);
  values.push(p.value);
  if (stopping) stoppedAt = step;
}
if (stoppedAt === null && ran < maxSteps) console.log(`every pair shown after ${ran} steps`);

const reviewStart = performance.now();
const review = session.review();
const reviewMs = performance.now() - reviewStart;

const q = quantiles(times);
const fmt = (x) => x.toFixed(1).padStart(7);
console.log(`wasm: ${wasmFile} (${wasmBytes} bytes)`);
console.log(`load+instantiate ${loadMs.toFixed(1)} ms; n_items=${nItems} n_active=${nActive}`);
if (times.length) {
  console.log(`propose ms: first ${fmt(times[0])}  median ${fmt(q(0.5))}  p90 ${fmt(q(0.9))}  max ${fmt(q(1))}  (n=${times.length})`);
} else {
  console.log('propose ms: no steps run');
}
console.log(`review ${reviewMs.toFixed(1)} ms`);
console.log(`values every 10th: ${values.filter((_, i) => i % 10 === 0).map((x) => x.toFixed(4)).join(' ')}`);
console.log(stoppedAt === null ? `no auto-stop within ${maxSteps} steps` : `auto-stop at pair ${stoppedAt}`);

console.log(`model top-10: ${review.order.slice(0, 10).join(' ')}`);
console.log(`true  top-10: ${truth.slice(0, 10).join(' ')}`);
const [first] = competitionRanks(review.groupSizes);
console.log(first ? `first listed: item ${review.order[0]} rank ${first.rank + 1}${first.tied ? '=' : ''} interval [${review.rankLo[0] + 1}, ${review.rankHi[0] + 1}]; ${review.groupSizes.length} tie groups` : 'nothing listed');
