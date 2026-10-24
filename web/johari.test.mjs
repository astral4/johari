// Node tests for `johari.js`.
//
//   scripts/build-wasm.sh && node web/johari.test.mjs [path/to/module.wasm ...]

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { suite, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { shippedModule } from './client.js';
import { Code, Defaults, Johari, JohariError, Outcome, competitionRanks, sessionMethods } from './johari.js';

const paths = process.argv.length > 2 ? process.argv.slice(2).map((path) => pathToFileURL(path)) : [shippedModule()];

const throws = (code, fn, what = '') => {
  assert.throws(fn, (e) => e instanceof JohariError && e.code === code, `expected johari error ${code} ${what}`);
};
const throwsAt = (code, fn, what) => {
  assert.throws(fn, (e) => e instanceof JohariError && e.code === code && e.message.endsWith(`: ${what}`), `expected johari error ${code} at ${what}`);
};

const outcomeAt = (step) => [Outcome.A, Outcome.B, Outcome.TIE, null][step % 4];

const MARKS = [2, 1, 0, null, 2, 1, 0, null, 1].map((bucket, item) => [item, bucket]);
const TOTAL_PAIRS = (7 * 6) / 2;
const MAX_ROSTER_SIZE = 1 << 11;

function marked(johari) {
  const s = johari.newSession(9);
  for (const [item, bucket] of MARKS) s.mark(item, bucket);
  return s;
}

function played(johari) {
  const s = marked(johari);
  const first = s.propose();
  let step = 0;
  for (let p = first; p; p = s.propose()) s.answer(p.a, p.b, outcomeAt(step++));
  return { s, first };
}

for (const path of paths) {
  const bytes = await readFile(path);

  suite(fileURLToPath(path), () => {
    test("check exports' parameter counts", async () => {
      const { instance } = await WebAssembly.instantiate(bytes);
      const arities = Object.entries(instance.exports).filter(([name, value]) => typeof value === 'function' && !name.startsWith('johari_bench'));
      assert.deepEqual(Object.fromEntries(arities.map(([name, f]) => [name, f.length])), {
        johari_result: 0,
        johari_new: 2,
        johari_free_session: 1,
        johari_mark: 3,
        johari_propose: 1,
        johari_record: 4,
        johari_undo: 1,
        johari_review: 2,
      });
    });

    test('wrapper/module error code coverage', async () => {
      const e = (await Johari.load(bytes)).exports;
      const record = e.johari_result();
      assert.ok(record > 0 && record % 8 === 0, 'the result record is 8-aligned');
      for (const tieCost of [NaN, 0, -1, 1 + Number.EPSILON, Infinity]) assert.equal(e.johari_new(5, tieCost), 0, `tie cost of ${tieCost} doesn't get a session`);
      for (const nItems of [MAX_ROSTER_SIZE + 1, 0xffff_ffff]) assert.equal(e.johari_new(nItems, 0.8), 0, `roster of ${nItems} doesn't get a session`);
      const largest = e.johari_new(MAX_ROSTER_SIZE, 0.8);
      assert.ok(largest > 0, `roster of ${MAX_ROSTER_SIZE} gets a session`);
      e.johari_free_session(largest);
      const h = e.johari_new(5, 0.8);
      const steps = [
        [Code.ITEM_OUT_OF_RANGE, () => e.johari_mark(h, 5, 0)],
        [Code.BAD_BUCKET, () => e.johari_mark(h, 0, 3)],
        ...[0, 1, 2, 3].map((item) => [0, () => e.johari_mark(h, item, item % 3)]),
        [Code.ALREADY_MARKED, () => e.johari_mark(h, 0, 1)],
        [0, () => e.johari_mark(h, 4, -1)],
        [Code.PAIR_NOT_DISTINCT, () => e.johari_record(h, 1, 1, 0)],
        [Code.ITEM_NOT_ACTIVE, () => e.johari_record(h, 0, 4, 0)],
        [Code.BAD_OUTCOME, () => e.johari_record(h, 0, 1, 3)],
        [Code.BAD_ARGUMENT, () => e.johari_review(h, NaN)],
        [0, () => e.johari_record(h, 0, 1, 0)],
        [0, () => e.johari_record(h, 0, 2, -1)],
        [Code.PAIR_ALREADY_SHOWN, () => e.johari_record(h, 1, 0, 2)],
        [Code.BUCKET_STAGE_CLOSED, () => e.johari_mark(h, 0, 0)],
        [Code.BAD_POINTER, () => e.johari_undo(0)],
      ];
      for (const [k, [want, step]] of steps.entries()) assert.equal(step(), want, `step ${k}`);
      assert.equal(e.johari_result(), record, 'at one address');
      assert.deepEqual(new Set(steps.map(([want]) => want).filter((want) => want < 0)), new Set(Object.values(Code)), 'every code');
      e.johari_free_session(h);
    });

    test('the bucket stage refuses invalid calls', async () => {
      const johari = await Johari.load(bytes);
      throws(Code.BAD_ARGUMENT, () => johari.newSession(), 'a missing roster size');
      throws(Code.BAD_ARGUMENT, () => johari.newSession(-1), 'a negative roster size (not 2^32 - 1)');
      throws(Code.BAD_ARGUMENT, () => johari.newSession(MAX_ROSTER_SIZE + 1), "a roster size exceeding the maximum");
      const s = johari.newSession(9);
      throws(Code.BAD_BUCKET, () => s.mark(0, 3), 'a bucket out of range');
      throws(Code.BAD_BUCKET, () => s.mark(0), 'a missing bucket');
      throws(Code.BAD_BUCKET, () => s.mark(0, 'love'), 'a bucket by name');
      throws(Code.BAD_BUCKET, () => s.mark(0, -1), "the ABI's sentinel");
      for (const [item, bucket] of MARKS) s.mark(item, bucket);
      throws(Code.ALREADY_MARKED, () => s.mark(0, 1));
      throws(Code.ITEM_OUT_OF_RANGE, () => s.mark(9, 1));
    });

    test('the pairs stage never repeats and ends by itself', async () => {
      const s = marked(await Johari.load(bytes));
      const seen = new Set();
      const pairs = [];
      for (let step = 0; step < TOTAL_PAIRS; step++) {
        const p = s.propose();
        assert.ok(p, `a pair at step ${step}`);
        const key = Math.min(p.a, p.b) * 100 + Math.max(p.a, p.b);
        assert.ok(!seen.has(key), `step ${step} repeated ${p.a},${p.b}`);
        seen.add(key);
        assert.deepEqual(Object.keys(p), ['a', 'b', 'value'], 'a pair and its value');
        assert.ok(Number.isFinite(p.value) && p.value >= 0, 'a finite and non-negative value');
        if (step === 0) {
          throws(Code.ITEM_NOT_ACTIVE, () => s.answer(p.a, 3, Outcome.A));
          throws(Code.BAD_OUTCOME, () => s.answer(p.a, p.b, 3));
          throws(Code.BAD_OUTCOME, () => s.answer(p.a, p.b), 'a missing outcome');
          throws(Code.BAD_OUTCOME, () => s.answer(p.a, p.b, 'skip'), 'an outcome by name');
          throws(Code.BAD_OUTCOME, () => s.answer(p.a, p.b, -1), "the ABI's sentinel");
          throwsAt(Code.ITEM_OUT_OF_RANGE, () => s.answer(p.a + 2 ** 32, p.b, Outcome.A), 'a');
          throwsAt(Code.ITEM_OUT_OF_RANGE, () => s.mark(NaN, 1), 'item');
          throws(Code.ALREADY_MARKED, () => s.mark(3, 1), 'proposing alone leaves the bucket stage open');
        }
        const outcome = outcomeAt(step);
        s.answer(p.a, p.b, outcome);
        pairs.push([p.a, p.b, outcome]);
        throws(Code.PAIR_ALREADY_SHOWN, () => s.answer(p.b, p.a, null));
        if (step === 0) throws(Code.BUCKET_STAGE_CLOSED, () => s.mark(3, 1), 'the first recorded pair closes the bucket stage');
      }
      assert.equal(s.propose(), null, 'null once every pair is shown');
      assert.deepEqual(s.log(), { nItems: 9, tieCost: Defaults.tieCost, marks: MARKS, pairs }, 'the log');
      assert.deepEqual(JSON.parse(JSON.stringify(s.log())), s.log(), 'the log through JSON');
      assert.notEqual(s.log().marks, s.log().marks, 'a fresh copy for each call');
    });

    test('the review lists the bucketed items, with intervals nested by level', async () => {
      const johari = await Johari.load(bytes);
      const { s } = played(johari);
      const review = s.review();
      assert.deepEqual(review.order.toSorted((x, y) => x - y), [0, 1, 2, 4, 5, 6, 8]);
      // TODO: Use `Math.sumPrecise` once V8 enables it by default.
      assert.equal(review.groupSizes.reduce((a, b) => a + b, 0), 7);
      assert.ok(review.rankLo.every((lo, k) => Number.isInteger(lo) && 0 <= lo && lo <= review.rankHi[k] && Number.isInteger(review.rankHi[k]) && review.rankHi[k] <= 6));

      const wide = s.review(1.0);
      assert.ok(wide.rankLo.every((lo, k) => lo <= review.rankLo[k] && review.rankHi[k] <= wide.rankHi[k]), 'a higher level contains the default interval');
      assert.ok(wide.rankLo.every((lo, k) => lo === 0 && wide.rankHi[k] === 6), 'level 1 spans every rank');
      const replay = (tieCost) => {
        using t = johari.restore({ ...s.log(), tieCost });
        return t.review();
      };
      assert.equal(replay(1).groupSizes.length, 7, 'nothing ties at tie cost 1');
      assert.deepEqual(replay(Defaults.tieCost), review, 'explicitly passing the default tie cost changes nothing');
      assert.deepEqual(s.review(Defaults.reviewLevel), review, 'explicitly passing the default level changes nothing');
      assert.deepEqual(s.review(undefined), review, 'undefined takes the default');
      const median = s.review(0);
      assert.ok(median.rankLo.every((lo, k) => review.rankLo[k] <= lo && median.rankHi[k] <= review.rankHi[k]), 'level 0 nests in the default');
    });

    test('a float outside its valid range is refused', async () => {
      const johari = await Johari.load(bytes);
      const s = marked(johari);
      const show = (x) => (typeof x === 'object' ? JSON.stringify(x) : String(x));
      for (const level of [NaN, -Number.MIN_VALUE, 1 + Number.EPSILON, Infinity, '0.5', null, { level: 0.5 }]) throws(Code.BAD_ARGUMENT, () => s.review(level), `level ${show(level)}`);
      for (const tieCost of [NaN, Infinity, 0, -Number.MIN_VALUE, 1 + Number.EPSILON, null, '0.8', { tieCost: 0.5 }]) throws(Code.BAD_ARGUMENT, () => johari.newSession(9, tieCost), `tie cost ${show(tieCost)}`);
    });

    test('a restore replays a log', async () => {
      const johari = await Johari.load(bytes);
      const { s, first } = played(johari);
      const log = s.log();
      for (const bad of [null, 7, {}, { ...log, pairs: 'none' }, { ...log, tieCost: undefined }, { ...log, tieCost: 0 }, { ...log, tieCost: 2 }]) throws(Code.BAD_ARGUMENT, () => johari.restore(bad), `log ${JSON.stringify(bad)}`);
      throwsAt(Code.BAD_ARGUMENT, () => johari.restore({ ...log, marks: [[0, 2], [1]] }), 'marks[1]');
      throwsAt(Code.ITEM_OUT_OF_RANGE, () => johari.restore({ ...log, nItems: 5 }), 'marks[5]');
      throwsAt(Code.PAIR_ALREADY_SHOWN, () => johari.restore({ ...log, pairs: [...log.pairs, log.pairs[0]] }), `pairs[${TOTAL_PAIRS}]`);
      using r = johari.restore({ ...log, pairs: [] });
      assert.deepEqual(r.propose(), first, 'a restored log gives the same proposal on this module');
    });

    test('undo is LIFO', async () => {
      const { s, first } = played(await Johari.load(bytes));
      for (let k = 0; k < TOTAL_PAIRS; k++) assert.equal(s.undo(), 'pair');
      assert.deepEqual(s.log().pairs, [], 'undo takes the pairs back from the log');
      assert.deepEqual(s.log().marks, MARKS, 'undo leaves the marks');
      assert.deepEqual(s.propose(), first, 'the first proposal comes back');
      assert.equal(s.undo(), 'mark');
      assert.deepEqual(s.log().marks, MARKS.slice(0, 8), 'undo takes the last mark back');
      s.mark(8, 0);
      assert.equal(s.undo(), 'mark');
      for (let k = 0; k < 8; k++) assert.equal(s.undo(), 'mark');
      assert.equal(s.undo(), 'nothing');
      assert.deepEqual(s.log(), { nItems: 9, tieCost: Defaults.tieCost, marks: [], pairs: [] }, 'an empty log');
    });

    test('every query is total over the active set', async () => {
      const johari = await Johari.load(bytes);
      const empty = johari.newSession(0);
      assert.equal(empty.propose(), null, 'vacuously exhausted');
      assert.deepEqual(empty.review(), { order: [], groupSizes: [], rankLo: [], rankHi: [] }, 'an empty list');
      empty.free();
      throws(Code.BAD_POINTER, () => empty.review(), 'a freed empty session throws, not an empty list');
      using one = johari.newSession(2);
      one.mark(0, 1);
      assert.equal(one.propose(), null);
      const single = one.review();
      assert.deepEqual([single.order, single.rankLo, single.rankHi, single.groupSizes], [[0], [0], [0], [1]], 'a one-item list');
    });

    test('a freed session frees once and refuses every other call', async () => {
      const s = marked(await Johari.load(bytes));
      s.free();
      s.free();
      assert.deepEqual(s.log().marks, MARKS, 'the log outlives the session');
      const methods = sessionMethods();
      assert.ok(methods.includes('mark') && methods.includes('review'), 'the methods are found');
      for (const name of methods) throws(Code.BAD_POINTER, () => s[name]('junk', 'junk', 'junk', 'junk'), name);
    });

    test('`using` frees a session at the end of its scope', async () => {
      const johari = await Johari.load(bytes);
      let scoped;
      {
        using session = johari.newSession(3);
        scoped = session;
      }
      throws(Code.BAD_POINTER, () => scoped.review(), 'freed by `using`');
    });

    test('linear memory stops at 2 GiB', async () => {
      const { exports: full } = await Johari.load(bytes);
      full.memory.grow(0x8000 - full.memory.buffer.byteLength / 0x1_0000);
      assert.throws(() => full.memory.grow(1), RangeError, 'memory stops at 2 GiB');
      assert.throws(() => full.johari_new(1, 0.8), WebAssembly.RuntimeError, 'a fresh instance with no memory left to grow traps');
    });

    test('nothing runs in an instance after a trap', async () => {
      const { instance } = await WebAssembly.instantiate(bytes);
      const johari = new Johari(instance);
      assert.equal(new Johari(instance).exports, johari.exports, 'one guard per instance');
      const session = johari.newSession(3);
      session.mark(0, 1);
      const { memory } = johari.exports;
      memory.grow(0x8000 - memory.buffer.byteLength / 0x1_0000);
      const largest = { nItems: MAX_ROSTER_SIZE, tieCost: Defaults.tieCost, marks: Array.from({ length: MAX_ROSTER_SIZE }, (_, item) => [item, item % 3]), pairs: [] };
      let trap;
      assert.throws(() => johari.restore(largest).propose(), (e) => (trap = e) instanceof WebAssembly.RuntimeError, 'running out of memory traps');
      assert.throws(() => session.review(), (e) => e === trap, 'a later call throws the trap');
      assert.throws(() => new Johari(instance).newSession(3), (e) => e === trap, 'a new session through another instance traps');
      assert.equal(session.free(), undefined, 'a free does nothing');
      assert.deepEqual(session.log().marks, [[0, 1]], 'the log is still present');
    });
  });
}

test('competition ranks', () => {
  assert.deepEqual(competitionRanks([2, 1]), [{ rank: 0, tied: true }, { rank: 0, tied: true }, { rank: 2, tied: false }]);
  assert.deepEqual(competitionRanks([]), []);
});
