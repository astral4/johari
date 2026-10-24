// Node harness: replay simulated sessions for comparing builds.
//
//   node web/replay.mjs trace [path/to/johari.wasm] [seed=7] [nItems=127] [nActive=100] [maxSteps=400] [markNoise=1] [careless=0.075]
//   node web/replay.mjs sweep [path/to/johari.wasm] [seeds=20] [nItems=127] [nActive=100] [maxSteps=400] [markNoise=1] [careless=0.075]
//   node web/replay.mjs log [path/to/johari.wasm] [seeds=8] [nItems=127] [nActive=100] [checkpoints=20,60,120,180] [markNoise=1] [careless=0.075]

import { fileURLToPath } from 'node:url';

import { Defaults } from './johari.js';
import { benchModule, fitDiagnostics, loadModule, makeSim, moduleArg, numberArg, perSeed, phaseNames, quantiles, simulatedLog, workload } from './sim.mjs';

const [mode, ...args] = process.argv.slice(2);
const wasmPath = moduleArg(args, benchModule());
const [arg1, nItemsArg, nActiveArg, stepsArg, markNoiseArg, carelessArg] = args;
const maxSteps = mode === 'log' ? null : numberArg(stepsArg, 'maxSteps', { fallback: 400, min: 1 });
const johari = await loadModule(wasmPath);

function run(seed, emit) {
  const sim = makeSim(nItemsArg, nActiveArg, seed, markNoiseArg, carelessArg);
  using s = sim.start(johari);
  let [steps, stopped] = [0, false];
  const sweeps = [];
  let unconverged = 0;
  for (const { step, p, outcome, stopping } of sim.steps(s, { maxSteps })) {
    const fit = fitDiagnostics(s);
    if (fit) {
      sweeps.push(fit.sweeps);
      unconverged += fit.converged ? 0 : 1;
    }
    emit?.(`${step} ${p.a} ${p.b} ${outcome ?? '-'} ${p.value} ${stopping ? 1 : 0} ${fit?.sweeps ?? '-'}`);
    [steps, stopped] = [stopping ? step : step + 1, stopping];
  }
  const level = Defaults.reviewLevel;
  const review = s.review(level);
  emit?.(`review order ${review.order.join(',')}`);
  emit?.(`lo ${review.rankLo.join(',')}`);
  emit?.(`hi ${review.rankHi.join(',')}`);
  emit?.(`groups ${review.groupSizes.join(',')}`);
  const score = sim.score(review, { level });
  const even = sim.score(review, { falseTie: 0.5, falseOrder: 0.5, level });
  emit?.(`score loss ${score.loss} reversed ${score.reversed} falseTie ${score.falseTie} falseOrder ${score.falseOrder} soft ${score.softLoss}`);
  emit?.(`score½ loss ${even.loss} soft ${even.softLoss}`);
  emit?.(`intervals coverage ${score.coverage} width ${score.width} score ${score.intervalScore} top ${score.intervalScoreTop}`);
  return { steps, stopped, sweeps, unconverged, ...score, even: even.loss, evenSoft: even.softLoss };
}

if (mode === 'trace') {
  run(arg1, (line) => console.log(line));
} else if (mode === 'sweep') {
  const seeds = numberArg(arg1, 'seeds', { fallback: 20, min: 1 });
  const { nItems, nActive, markNoise, careless } = makeSim(nItemsArg, nActiveArg, 1, markNoiseArg, carelessArg);
  const results = await perSeed(import.meta.url, seeds, run);
  console.log(`wasm: ${fileURLToPath(wasmPath)}; n_items=${nItems} n_active=${nActive} mark noise ${markNoise} careless ${careless}`);
  const rows = [];
  for await (const [seed, r] of results) {
    rows.push(r);
    console.log(`seed ${seed}: steps ${r.steps}${r.stopped ? '' : ' (no stop)'} loss ${r.loss.toFixed(3)} spearman ${r.spearman.toFixed(4)} top10 ${r.top10}`);
  }
  // TODO: Use `Math.sumPrecise` here, in `total`, and in the fits' total once V8 enables it by default.
  const stat = (key) => {
    const xs = rows.map((r) => r[key]);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, xs.length - 1));
    return `${mean.toFixed(3)} ± ${sd.toFixed(3)}`;
  };
  console.log(`mean: steps ${stat('steps')}  spearman ${stat('spearman')}  top10 ${stat('top10')}`);
  console.log(`      loss ${stat('loss')} (reversed ${stat('reversed')}, false tie ${stat('falseTie')}, false order ${stat('falseOrder')}), soft ${stat('softLoss')}`);
  console.log(`      at costs ½/½: loss ${stat('even')}, soft ${stat('evenSoft')}`);
  const total = (key) => rows.reduce((a, r) => a + r[key], 0);
  console.log(`      ties: ${(total('tiedPairs') / rows.length).toFixed(1)} pairs per list, ${(100 * total('tiedWithin') / total('tiedPairs')).toFixed(1)}% truly within δ, ${(100 * total('tiedWithin') / total('withinPairs')).toFixed(1)}% of the pairs within δ tied`);
  console.log(`      rank intervals: coverage ${stat('coverage')}, width ${stat('width')}, interval score ${stat('intervalScore')} (top-weighted ${stat('intervalScoreTop')})`);
  const sweeps = rows.flatMap((r) => r.sweeps);
  const at = quantiles(sweeps);
  if (sweeps.length > 0) {
    console.log(`      fits: ${sweeps.length}, sweeps median ${at(0.5)} p90 ${at(0.9)} p99 ${at(0.99)} max ${at(1)} total ${sweeps.reduce((a, b) => a + b, 0)}, ${total('unconverged')} unconverged`);
  }
} else if (mode === 'log') {
  const w = workload(args);
  const results = await perSeed(import.meta.url, w.seeds, (seed) => simulatedLog(johari, seed, w));
  const logs = await Array.fromAsync(results, ([, log]) => log);
  console.log(JSON.stringify({ wasm: fileURLToPath(wasmPath), workload: w, reviewLevel: Defaults.reviewLevel, phases: phaseNames(johari.exports), logs }));
} else {
  console.error('usage: node web/replay.mjs trace|sweep [wasm] [seed|seeds] [nItems] [nActive] [maxSteps] [markNoise] [careless]');
  console.error('       node web/replay.mjs log [wasm] [seeds] [nItems] [nActive] [checkpoints] [markNoise] [careless]');
  process.exit(2);
}
