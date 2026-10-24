// Node harness: time each phase of a step inside the WebAssembly module.
//
//   FEATURES=bench scripts/build-wasm.sh
//   node web/phases.mjs [path/to/johari.bench.wasm] [seeds=8] [nItems=127] [nActive=100] [checkpoints=20,60,120,180] [markNoise = 1] [careless = 0.075]

import { fileURLToPath } from 'node:url';

import { benchModule, loadModule, moduleArg, phaseNames, quantiles, simulatedLog, workload } from './sim.mjs';

const args = process.argv.slice(2);
const wasmPath = moduleArg(args, benchModule());
const w = workload(args);
const { seeds, checkpoints } = w;

const TARGET_MS = 50;

const johari = await loadModule(wasmPath);
const e = johari.exports;
if (!e.johari_bench) {
  console.error('this module was built without --features bench');
  process.exit(1);
}
const phases = phaseNames(e);

const rows = [...phases, 'review, fit cached'];

/** Milliseconds taken by `run(iters)`. */
function msTaken(run, iters) {
  const t = performance.now();
  run(iters);
  return performance.now() - t;
}

/** Milliseconds taken per iteration of `run(iters)`. */
function msPerIteration(run) {
  run(1);
  const setup = msTaken(run, 0);
  for (let iters = 1; ;) {
    const elapsed = msTaken(run, iters) - setup;
    if (elapsed >= TARGET_MS || iters >= 4096) return elapsed / iters;
    const growth = elapsed > 0 ? Math.min(100, (1.2 * TARGET_MS) / elapsed) : 100;
    iters = Math.min(4096, Math.max(iters + 1, Math.ceil(iters * growth)));
  }
}

const logs = Array.from({ length: seeds }, (_, k) => simulatedLog(johari, k + 1, w));
const ms = rows.map(() => checkpoints.map(() => []));
for (const log of logs) {
  for (const [c, answers] of checkpoints.entries()) {
    using session = johari.restore({ ...log, pairs: log.pairs.slice(0, answers) });
    const bench = (phase, iters) => e.johari_bench(session.handle, phase, iters);
    const runs = [
      ...phases.map((_, phase) => (iters) => bench(phase, iters)),
      (iters) => {
        for (let k = 0; k < iters; k++) session.review();
      },
    ];
    for (const [row, run] of runs.entries()) ms[row][c].push(msPerIteration(run));
  }
}

const { nItems, nActive, markNoise, careless } = w;
console.log(`wasm: ${fileURLToPath(wasmPath)}`);
console.log(`n_items=${nItems} n_active=${nActive} mark noise ${markNoise} careless ${careless} seeds 1..=${seeds}: ms per iteration, the median over the seeds after each number of answers`);
console.log('phase'.padEnd(32) + checkpoints.map((answers) => `@${answers}`.padStart(10)).join(''));
for (const [row, name] of rows.entries()) {
  const medians = ms[row].map((times) => quantiles(times)(0.5));
  console.log(name.padEnd(32) + medians.map((m) => m.toFixed(3).padStart(10)).join(''));
}
