// Synthetic respondent for `bench.mjs`, `phases.mjs`, and `replay.mjs`.

import { readFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort } from 'node:worker_threads';

import { Bucket, Defaults, Johari, Outcome, competitionRanks } from './johari.js';

/** The module for a harness. */
export function moduleArg(args, fallback) {
  return args.length > 0 && Number.isNaN(Number(args[0])) ? pathToFileURL(args.shift()) : fallback;
}

/**
 * The harness argument `arg` (string or number) as a number from `min` to `max`.
 * If `integer` is true, then `arg` must be a whole number.
 */
export function numberArg(arg, name, { fallback, min = 0, max = Infinity, integer = true } = {}) {
  if (arg === undefined) return fallback;
  const x = String(arg).trim() === '' ? NaN : Number(arg);
  if (!(integer ? Number.isSafeInteger(x) : Number.isFinite(x)) || x < min || x > max) {
    const range = max === Infinity ? `of at least ${min}` : `from ${min} to ${max}`;
    throw new Error(`${name} must be ${integer ? 'an integer' : 'a number'} ${range}, not ${JSON.stringify(arg)}`);
  }
  return x;
}

/** The module at the `file:` URL `url` from its bytes. */
export async function loadModule(url) {
  return Johari.load(await readFile(url));
}

/** The module built with the `bench` feature beside this file (`FEATURES=bench scripts/build-wasm.sh`). */
export function benchModule() {
  return new URL('./johari.bench.wasm', import.meta.url);
}

/** The first two u32 of the result record of the module whose exports are `e`. */
function resultPair(e) {
  return new Uint32Array(e.memory.buffer, e.johari_result() >>> 0, 2);
}

/** Fit diagnostics for the session's current log. Returns the number of sweeps and whether the fit converged. */
export function fitDiagnostics(session) {
  const e = session.e;
  if (!e.johari_bench_fit_diagnostics) return null;
  e.johari_bench_fit_diagnostics(session.handle);
  const [sweeps, converged] = resultPair(e);
  return { sweeps, converged: converged === 1 };
}

/** The benchmark phases by number. */
export function phaseNames(e) {
  if (!e.johari_bench_phase) return [];
  const names = [];
  for (let phase = 0; e.johari_bench_phase(phase) === 0; phase++) {
    const [ptr, len] = resultPair(e);
    names.push(new TextDecoder().decode(new Uint8Array(e.memory.buffer, ptr, len)));
  }
  return names;
}

/** The quantiles of the numbers `xs` by nearest rank. Returns a function taking p to the `p` quantile, with 1 giving the largest. */
export function quantiles(xs) {
  const sorted = xs.toSorted((x, y) => x - y);
  return (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

/**
 * Invoke `run(seed)` for seeds 1..=`seeds` on worker threads that each run the harness `script` with this process' arguments.
 * Returns `[seed, result]` in seed order.
 */
export async function perSeed(script, seeds, run) {
  if (!isMainThread) {
    parentPort.on('message', (seed) => (seed === null ? process.exit() : parentPort.postMessage(run(seed))));
    return new Promise(() => { });
  }
  const results = Array.from({ length: seeds }, () => Promise.withResolvers());
  let next = 1;
  for (let k = Math.min(seeds, availableParallelism()); k > 0; k--) {
    const worker = new Worker(new URL(script), { argv: process.argv.slice(2) });
    let seed;
    const feed = () => {
      seed = next <= seeds ? next++ : null;
      worker.postMessage(seed);
    };
    worker.on('message', (result) => {
      results[seed - 1].resolve(result);
      feed();
    });
    worker.on('error', (error) => results[seed - 1].reject(error));
    feed();
  }
  return (async function* () {
    for (const [k, result] of results.entries()) yield [k + 1, await result.promise];
  })();
}

/** PRNG for harness runs. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniformly sample from the standard normal distribution using `rand`. */
function normalFrom(rand) {
  return Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
}

const BETA = 3;
const DELTA = 1.0;
const SIGMA0 = 3.75;
const BUCKET_CUTS = [-1.5, 1.0];
const BUCKETS = Object.values(Bucket);
const MARK_NOISE = 1;
const MARK_SLIP = 0.01;
const MARK_STREAM = 0x9e3779b9;
const CARELESS = 0.075;
const CARELESS_STREAM = 0x85ebca6b;

/**
 * A simulated respondent over a roster of `nItems`, of which `nActive` get buckets and the rest get "don't know".
 * Marks have noise sd `markNoise` and answers have careless rate `careless`. Uses `seed` for reproducibility.
 */
export function makeSim(nItemsArg, nActiveArg, seedArg, markNoiseArg, carelessArg) {
  const nItems = numberArg(nItemsArg, 'nItems', { fallback: 127 });
  const nActive = numberArg(nActiveArg, 'nActive', { fallback: Math.min(100, nItems), max: nItems });
  const seed = numberArg(seedArg, 'seed', { fallback: 7, max: 0xffff_ffff });
  const markNoise = numberArg(markNoiseArg, 'markNoise', { fallback: MARK_NOISE, integer: false });
  const careless = numberArg(carelessArg, 'careless', { fallback: CARELESS, max: 1, integer: false });
  const rand = mulberry32(seed);
  const trueTheta = Array.from({ length: nItems }, () => SIGMA0 * normalFrom(rand));
  const activeSet = new Set();
  while (activeSet.size < nActive) activeSet.add(Math.floor(rand() * nItems));
  const truth = [...activeSet].sort((x, y) => trueTheta[y] - trueTheta[x] || x - y);
  const markRand = mulberry32(seed ^ MARK_STREAM);
  const marks = trueTheta.map((theta, i) => {
    const judged = theta + markNoise * normalFrom(markRand);
    const slipped = markRand() < MARK_SLIP;
    const random = BUCKETS[Math.floor(BUCKETS.length * markRand())];
    if (!activeSet.has(i)) return null;
    if (slipped) return random;
    return judged < BUCKET_CUTS[0] ? Bucket.MEH : judged < BUCKET_CUTS[1] ? Bucket.FINE : Bucket.LOVE;
  });
  const T = Math.exp(BETA * DELTA);
  // The respondent's probabilities of picking `a` and of picking `b` over the other. The rest is "same".
  const answerProbs = (a, b) => {
    const wa = Math.exp(BETA * trueTheta[a]);
    const wb = Math.exp(BETA * trueTheta[b]);
    return [wa / (wa + T * wb), wb / (wb + T * wa)];
  };
  const carelessRand = mulberry32(seed ^ CARELESS_STREAM);
  const respond = (a, b) => {
    const [pa, pb] = carelessRand() < careless ? [0.25, 0.25] : answerProbs(a, b);
    const u = rand();
    return u < pa ? Outcome.A : u < pa + pb ? Outcome.B : Outcome.TIE;
  };
  return {
    nItems,
    nActive,
    markNoise,
    careless,
    truth,
    start(johari) {
      const session = johari.newSession(nItems);
      for (let i = 0; i < nItems; i++) session.mark(i, marks[i]);
      return session;
    },
    *steps(session, { maxSteps = Infinity, stop = true } = {}) {
      for (let step = 0; step < maxSteps; step++) {
        const start = performance.now();
        const p = session.propose();
        const ms = performance.now() - start;
        if (!p) return;
        const stopping = stop && p.value < Defaults.stopThreshold;
        const outcome = stopping ? null : respond(p.a, p.b);
        yield { step, p, ms, outcome, stopping };
        if (stopping) return;
        session.answer(p.a, p.b, outcome);
      }
    },
    score(review, { falseTie = Defaults.tieCost, falseOrder = 1 - Defaults.tieCost, level = Defaults.reviewLevel } = {}) {
      const n = truth.length;
      const truePos = new Map(truth.map((item, k) => [item, k]));
      const ranks = competitionRanks(review.groupSizes);
      let tiedPairs = 0;
      let tiedWithin = 0;
      let withinPairs = 0;
      let reversed = 0;
      let tiedClear = 0;
      let orderedWithin = 0;
      let softLoss = 0;
      let d2 = 0;
      let covered = 0;
      let width = 0;
      let intervalScore = 0;
      let topScore = 0;
      let topWeight = 0;
      const alpha = 1 - level;
      review.order.forEach((upper, x) => {
        const t = truePos.get(upper);
        d2 += (x - t) ** 2;
        const [lo, hi] = [review.rankLo[x], review.rankHi[x]];
        const itemScore = hi - lo + (2 / alpha) * (Math.max(lo - t, 0) + Math.max(t - hi, 0));
        covered += lo <= t && t <= hi ? 1 : 0;
        width += hi - lo;
        intervalScore += itemScore;
        topScore += itemScore / (t + 2);
        topWeight += 1 / (t + 2);
        for (let y = x + 1; y < review.order.length; y++) {
          const lower = review.order[y];
          const w = 1 / (Math.min(t, truePos.get(lower)) + 2);
          const gap = trueTheta[upper] - trueTheta[lower];
          const tied = ranks[x].rank === ranks[y].rank;
          const within = Math.abs(gap) <= DELTA;
          tiedPairs += tied ? 1 : 0;
          tiedWithin += tied && within ? 1 : 0;
          withinPairs += within ? 1 : 0;
          if (within) {
            if (!tied) orderedWithin += w;
          } else if (tied) {
            tiedClear += w;
          } else if (gap < 0) {
            reversed += w;
          }
          const [pUpper, pLower] = answerProbs(upper, lower);
          softLoss += w * (tied ? falseTie * (pUpper + pLower) : pLower + falseOrder * (1 - pUpper - pLower));
        }
      });
      const top10 = new Set(truth.slice(0, 10));
      return {
        loss: reversed + falseTie * tiedClear + falseOrder * orderedWithin,
        reversed,
        falseTie: falseTie * tiedClear,
        falseOrder: falseOrder * orderedWithin,
        softLoss,
        tiedPairs,
        tiedWithin,
        withinPairs,
        coverage: covered / n,
        width: width / n,
        intervalScore: intervalScore / n,
        intervalScoreTop: topScore / topWeight,
        spearman: n > 1 ? 1 - (6 * d2) / (n * (n * n - 1)) : 1,
        top10: review.order.slice(0, 10).filter((i) => top10.has(i)).length,
      };
    },
  };
}

/** Simulated sessions for phases.mjs. */
export function workload([seedsArg, nItemsArg, nActiveArg, checkpointsArg, markNoiseArg, carelessArg]) {
  const seeds = numberArg(seedsArg, 'seeds', { fallback: 8, min: 1 });
  const checkpoints = (checkpointsArg ?? '20,60,120,180').split(',').map((answers) => numberArg(answers, 'a checkpoint'));
  if (checkpoints.some((answers, k) => k > 0 && answers <= checkpoints[k - 1])) {
    throw new Error(`checkpoints must increase, not ${JSON.stringify(checkpointsArg)}`);
  }
  const { nItems, nActive, markNoise, careless } = makeSim(nItemsArg, nActiveArg, 1, markNoiseArg, carelessArg);
  return { seeds, checkpoints, nItems, nActive, markNoise, careless };
}

/** The session log of the workload `w` for seed `seed`. */
export function simulatedLog(johari, seed, w) {
  const answers = Math.max(...w.checkpoints);
  const sim = makeSim(w.nItems, w.nActive, seed, w.markNoise, w.careless);
  using session = sim.start(johari);
  for (const _ of sim.steps(session, { maxSteps: answers, stop: false }));
  if (session.propose() === null) {
    throw new Error(`every pair of ${sim.nActive} active items was shown within ${answers} answers (lower the checkpoints or raise nActive)`);
  }
  return session.log();
}
