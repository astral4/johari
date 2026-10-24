// Node tests for `worker.js`.
//
//   scripts/build-wasm.sh && node web/worker.test.mjs

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { mock, test } from 'node:test';

import { Johari } from './johari.js';
import { serve } from './worker.js';

/** A Worker stand-in serving worker.js in-process with its own state. */
class InProcessWorker {
  constructor() {
    this.posted = [];
    this.terminated = false;
    this.scope = {
      postMessage: (data) => {
        const copy = structuredClone(data);
        setTimeout(() => {
          if (!this.terminated) this.onmessage?.({ data: copy });
        }, 0);
      },
    };
    serve(this.scope);
    InProcessWorker.last = this;
  }

  postMessage(msg) {
    const copy = structuredClone(msg);
    this.posted.push(copy.method);
    setTimeout(() => {
      if (!this.terminated) this.scope.onmessage({ data: copy });
    }, 0);
  }

  terminate() {
    this.terminated = true;
  }
}
globalThis.Worker = InProcessWorker;

const instantiate = mock.method(WebAssembly, 'instantiate');

const { Code, Defaults, JohariClient, JohariError, shippedModule } = await import('./client.js');
const bytes = await readFile(shippedModule());

async function markAll(session, n) {
  for (let item = 0; item < n; item++) await session.mark(item, item % 3);
}

const proxies = new WeakMap();
/** `target` behind a deep proxy. */
function reactive(target) {
  if (typeof target !== 'object' || target === null || !Object.isExtensible(target)) return target;
  if (!proxies.has(target)) proxies.set(target, new Proxy(target, { get: (t, key, receiver) => reactive(Reflect.get(t, key, receiver)) }));
  return proxies.get(target);
}

test('refusals preserve codes, and a trap closes the client', async () => {
  const client = new JohariClient(bytes);
  assert.equal(await client.ready, undefined);
  const worker = Worker.last;
  const { instance } = await instantiate.mock.calls.at(-1).result;
  const session = await client.newSession(6);
  await markAll(session, 6);
  const before = await session.review();

  const outOfRange = await session.mark(9, 1).catch((e) => e);
  assert.ok(outOfRange instanceof JohariError, 'a refusal');
  assert.equal(outOfRange.code, Code.ITEM_OUT_OF_RANGE, "the ABI code");
  const nan = await session.review(NaN).catch((e) => e);
  assert.equal(nan.code, Code.BAD_ARGUMENT, 'a NaN crosses the wrapper and is refused by the module');
  assert.deepEqual(await session.review(), before, 'refusals leave the session as is');
  const nanTie = await client.newSession(6, NaN).catch((e) => e);
  assert.deepEqual([nanTie instanceof JohariError, nanTie.code], [true, Code.BAD_ARGUMENT], 'a NaN tie cost is refused');
  const freed = await client.newSession(3);
  await freed.free();

  const { memory } = instance.exports;
  memory.grow(0x8000 - memory.buffer.byteLength / 0x1_0000);
  const largest = { nItems: 1 << 11, tieCost: Defaults.tieCost, marks: Array.from({ length: 1 << 11 }, (_, item) => [item, item % 3]), pairs: [] };
  const trap = await client.restore(largest).then((s) => s.propose()).catch((e) => e);
  assert.equal(trap.closed, true, 'running out of memory traps, which closes the client');
  assert.equal(worker.terminated, true, 'running out of memory terminates the worker');
  assert.equal(await session.review().catch((e) => e), trap, "all session calls are rejected with the trap");
  assert.equal((await freed.review().catch((e) => e)).code, Code.BAD_POINTER, 'a freed session is refused as freed');
  assert.equal(await session.free(), undefined, 'and freeing one does nothing');
  assert.deepEqual(session.log().marks, [0, 1, 2, 3, 4, 5].map((item) => [item, item % 3]), 'log outlives the client');
});

test('remote session matches local session', async () => {
  const local = (await Johari.load(bytes)).newSession(6);
  using client = new JohariClient(bytes);
  const remote = await client.newSession(6);
  const settle = async (f) => {
    try {
      return { value: await f() };
    } catch (e) {
      return { class: e.constructor.name, code: e.code, message: e.message };
    }
  };
  const calls = [
    ['mark', 0, 2], ['mark', 1, 0], ['mark', 2, 1], ['mark', 3, null], ['mark', 4, 1], ['mark', 5, 2], ['mark', 5, 2], ['mark', 9, 1], ['mark', 0],
    ['propose'], ['answer', 0, 1, 0], ['answer', 0, 1, 0], ['answer', 0, 3, 0], ['answer', 0, 0, 0], ['answer', 1, 2, 7],
    ['review', NaN], ['review', '0.5'], ['review', null], ['review', { level: 0 }], ['review', 1], ['review'],
    ['undo'], ['undo'], ['mark', 3, -1],
    ['free'], ['review'], ['mark', 0, 1], ['mark', 0, () => 1], ['free'],
  ];
  for (const [method, ...args] of calls) {
    const want = await settle(() => local[method](...args));
    assert.deepEqual(await settle(() => remote[method](...args)), want, `${method}(${args.map((a) => JSON.stringify(a)).join(', ')})`);
  }
  assert.deepEqual(remote.log(), local.log(), 'the same log');
  const fresh = await Johari.load(bytes);
  const restored = await client.restore(new Proxy(local.log(), {}));
  assert.deepEqual(await restored.review(), fresh.restore(local.log()).review(), 'a restored remote session is the restored local session');
  assert.deepEqual(restored.log(), local.log(), 'the same log');
  for (const bad of [null, { nItems: 6, tieCost: 0.8, marks: [[0]], pairs: [] }, { ...local.log(), nItems: 2 }]) {
    const want = await settle(() => fresh.restore(bad));
    assert.ok(want.code, `a refusal: ${JSON.stringify(want)}`);
    assert.deepEqual(await settle(() => client.restore(bad)), want, `restore(${JSON.stringify(bad)})`);
  }
  const refused = await client.restore(null).catch((e) => e);
  assert.deepEqual([refused instanceof JohariError, refused.code], [true, Code.BAD_ARGUMENT], 'a refusal');
  assert.deepEqual(await restored.review(), fresh.restore(local.log()).review(), 'the client is still open');
  const live = await client.newSession(6);
  const { class: kind, code } = await settle(() => live.mark(0, () => 1));
  assert.deepEqual([kind, code], ['JohariError', Code.BAD_ARGUMENT]);
  assert.equal(await live.mark(0, 1), undefined, 'the session is untouched');
  assert.equal((await remote.review().catch((e) => e)).code, Code.BAD_POINTER, 'freed even after later sessions are opened');
});

test('a session works through a proxy', async () => {
  using client = new JohariClient(bytes);
  const johari = await Johari.load(bytes);
  const run = async (session) => {
    await markAll(session, 6);
    const proposal = await session.propose();
    await session.answer(proposal.a, proposal.b, 0);
    await session.undo();
    await session.answer(proposal.a, proposal.b, 2);
    return { proposal, review: await session.review(), log: session.log() };
  };
  const want = await run(johari.newSession(6));
  for (const [where, open] of [['local', async () => johari.newSession(6)], ['remote', () => client.newSession(6)]]) {
    for (const [how, wrap] of [['a Proxy', (session) => new Proxy(session, {})], ['a deep proxy', reactive]]) {
      await using session = wrap(await open());
      assert.deepEqual(await run(session), want, `a ${where} session through ${how}`);
    }
  }
});

test('calls run one at a time in order', async () => {
  using client = new JohariClient(bytes);
  const outcome = ({ a, b }) => (a + b) % 3;
  const [serial, together] = [await client.newSession(30), await client.newSession(30)];
  await markAll(serial, 30);
  await Promise.all(Array.from({ length: 30 }, (_, item) => together.mark(item, item % 3)));
  let next = await together.propose();
  for (let step = 0; step < 10; step++) {
    const proposal = await serial.propose();
    await serial.answer(proposal.a, proposal.b, outcome(proposal));
    const review = await serial.review();
    const [, togetherReview, togetherNext] = await Promise.all([together.answer(next.a, next.b, outcome(next)), together.review(), together.propose()]);
    assert.deepEqual([next, togetherReview], [proposal, review], `step ${step}`);
    next = togetherNext;
  }
  const [refusal, review] = await Promise.all([together.answer(next.a, next.b, 9).catch((e) => e), together.review()]);
  assert.equal(refusal.code, Code.BAD_OUTCOME, 'a refusal');
  assert.deepEqual(review, await serial.review(), 'a later call runs on the session as is');
  assert.deepEqual(together.log(), serial.log(), 'the same log');
});

test('a malformed request is fatal', async () => {
  const replies = [];
  const scope = { postMessage: (reply) => replies.push(reply) };
  serve(scope);
  const requests = [
    { method: 'newSession', args: [3] }, // before `init`
    { method: 'review', session: 7, args: [] }, // on a session not being served
    { method: 'constructor', args: [] }, // not a call
    { method: 'review', session: 7, args: 1 }, // arguments are not a list
  ];
  for (const [id, request] of requests.entries()) {
    await scope.onmessage({ data: { id, ...request } });
    const { error, ...rest } = replies.at(-1);
    assert.deepEqual(rest, { id, code: undefined }, `${error}: fatal`);
  }
  assert.throws(() => scope.onmessageerror(), /could not be read/, 'an unreadable request');
  for (const data of [null, undefined]) assert.throws(() => scope.onmessage({ data }), TypeError, `a ${data} request`);
  assert.equal(replies.length, requests.length, 'no reply');
});

test('a module that cannot be loaded closes the client', async () => {
  const client = new JohariClient(new Response(null, { status: 404, statusText: 'Not Found' }));
  const error = await client.ready.catch((e) => e);
  assert.equal(error.closed, true, 'a failed load closes the client');
  assert.equal(await client.newSession(3).catch((e) => e), error, 'a later call is rejected with the same error');
  assert.deepEqual(Worker.last.posted, [], 'nothing is posted');
  assert.equal(Worker.last.terminated, true, 'the worker is terminated');
});

