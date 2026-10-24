// Node tests for `client.js`.
//
//   node web/client.test.mjs

import assert from 'node:assert/strict';
import { test } from 'node:test';

// A controllable stand-in for `Worker`. Each posted message is answered by `FakeWorker.handler`,
// which returns a reply payload (`{ result }` or `{ error, code }`) or `undefined` to stay silent.
class FakeWorker {
  static instances = [];
  static handler;

  constructor(url) {
    this.url = url;
    this.posted = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
    this.onmessageerror = null;
    FakeWorker.instances.push(this);
  }

  postMessage(msg) {
    const copy = structuredClone(msg);
    this.posted.push(copy);
    queueMicrotask(() => {
      if (this.terminated) return;
      const reply = FakeWorker.handler(copy, this);
      if (reply !== undefined) this.onmessage?.({ data: { id: copy.id, ...reply } });
    });
  }

  emitError(event) {
    this.onerror?.(event);
  }

  emitMessageError() {
    this.onmessageerror?.(new Event('messageerror'));
  }

  terminate() {
    this.terminated = true;
  }
}
globalThis.Worker = FakeWorker;

const { Code, JohariClient, JohariError } = await import('./client.js');

const flush = () => new Promise((r) => setTimeout(r, 0));

const defaultHandler = (msg) => {
  if (msg.method === 'init') return { result: undefined };
  if (msg.method === 'newSession') return { result: { id: 1, log: { nItems: msg.args[0], tieCost: 0.8, marks: [], pairs: [] } } };
  return { result: null };
};

/** A client on a new fake worker that answers with `handler`, loading `wasm`. */
function newClient(handler = defaultHandler, wasm = new Uint8Array(0)) {
  FakeWorker.instances = [];
  FakeWorker.handler = handler;
  return new JohariClient(wasm);
}

/** The default handler, except each method specified in `replies` gets its reply there (`undefined`: none). */
const replying = (replies) => (msg) => (Object.hasOwn(replies, msg.method) ? replies[msg.method] : defaultHandler(msg));

test('a refusal leaves the session usable', async () => {
  using client = newClient();
  const session = await client.newSession(3);
  FakeWorker.handler = replying({ mark: { error: 'item out of range', code: Code.ITEM_OUT_OF_RANGE } });
  const error = await session.mark(9, 0).catch((e) => e);
  assert.ok(error instanceof JohariError, 'a refusal');
  assert.equal(error.code, Code.ITEM_OUT_OF_RANGE, 'ABI code round-trip');
  FakeWorker.handler = defaultHandler;
  assert.equal(await session.review(), null, 'a later call still goes through');
});

test('restore reads its log when it is called', async () => {
  using client = newClient((msg) => (msg.method === 'restore' ? { result: { id: 1, log: msg.args[0] } } : defaultHandler(msg)));
  const log = { nItems: 3, tieCost: 0.8, marks: [[0, 1], [1, 2]], pairs: [] };
  const restoring = client.restore(log);
  log.pairs.push([0, 1, 0]);
  const session = await restoring;
  assert.deepEqual(session.log().pairs, [], 'the log when restore was called');
});

test('a trap closes the client', async () => {
  const client = newClient();
  const session = await client.newSession(3);
  // `review` is still in flight when the trap's reply lands.
  FakeWorker.handler = replying({ propose: { error: 'trap' }, review: undefined });
  const [trapped, inFlight] = await Promise.all([session.propose().catch((e) => e), session.review().catch((e) => e)]);
  assert.equal(trapped.closed, true, 'a trap closes the client');
  assert.equal(trapped, client.closed, 'trap is the reason for closing');
  assert.equal(inFlight, trapped, 'the call in flight is rejected with the same error');
  assert.equal(FakeWorker.instances.at(-1).terminated, true, 'the worker is terminated');
  assert.equal(await session.review().catch((e) => e), trapped, 'a later call is rejected with the same error');
  assert.equal(await client.newSession(3).catch((e) => e), trapped, 'a new session is rejected with the same error');
  assert.equal(await session.free(), undefined, 'freeing a session of the closed client resolves');
});

test('an argument refused by the clone is a refusal', async () => {
  using client = newClient();
  const session = await client.newSession(3);
  const error = await session.mark(0, () => 1).catch((e) => e);
  assert.ok(error instanceof JohariError, 'refused');
  assert.equal(error.code, Code.BAD_ARGUMENT, 'refused as a bad argument');
  assert.equal(await session.review(), null, 'a later call still goes through');
});

test('a freed session refuses before anything is sent', async () => {
  const client = newClient();
  const session = await client.newSession(3);
  assert.equal(await session.free(), undefined, 'the worker frees it');
  const worker = FakeWorker.instances.at(-1);
  const { method, session: named, args } = worker.posted.at(-1);
  assert.deepEqual([method, named, args], ['free', undefined, [1]], 'a call not specifying a session');
  const posted = worker.posted.length;
  for (const call of [() => session.review(), () => session.mark(0, () => 1)]) {
    const error = await call().catch((e) => e);
    assert.deepEqual([error instanceof JohariError, error.code], [true, Code.BAD_POINTER], 'refused as freed');
  }
  assert.equal(await session.free(), undefined, 'a second free does nothing');
  assert.equal(worker.posted.length, posted, 'nothing is sent');
  client.terminate();
  assert.equal((await session.review().catch((e) => e)).code, Code.BAD_POINTER, 'freed before closed');
});

test('using releases a session and its client', async () => {
  let client, session;
  {
    using scopedClient = newClient();
    client = scopedClient;
    {
      await using scopedSession = await client.newSession(3);
      session = scopedSession;
    }
    assert.equal(FakeWorker.instances.at(-1).posted.at(-1).method, 'free', 'the worker frees the session');
    assert.equal((await session.review().catch((e) => e)).code, Code.BAD_POINTER, 'the session is freed here');
    assert.equal(client.closed, null, 'the client is still open');
  }
  assert.equal(client.closed?.closed, true, 'the client is closed when the scope ends');
});

for (const thrown of [new TypeError('not a clone refusal'), new RangeError('not a clone refusal')]) {
  test(`a ${thrown.name} from postMessage is not a refusal`, async () => {
    using client = newClient();
    const session = await client.newSession(3);
    const worker = FakeWorker.instances.at(-1);
    const post = worker.postMessage;
    worker.postMessage = (msg) => {
      if (msg.method === 'review') throw thrown;
      post.call(worker, msg);
    };
    const error = await session.review().catch((e) => e);
    assert.equal(error, thrown, `the ${thrown.name} itself`);
    assert.ok(!(error instanceof JohariError), 'not a refusal');
    assert.equal(await session.mark(0, 1), null, 'a later call still goes through');
  });
}

test('a failed load closes the client', async () => {
  const client = newClient((msg) => (msg.method === 'init' ? { error: 'CompileError' } : undefined));
  const error = await client.ready.catch((e) => e);
  assert.equal(error.closed, true, 'a failed load closes the client');
  assert.equal(error, client.closed, '`ready` is rejected with the closing error');
  const worker = FakeWorker.instances.at(-1);
  assert.equal(worker.terminated, true, 'the worker is terminated');
  const later = await client.newSession(3).catch((e) => e);
  assert.equal(later, error, 'a later call is rejected with the closing error');
  assert.equal(await client.restore(null).catch((e) => e), error, 'a restore before its log is read is rejected with the closing error');
  await flush();
  assert.equal(worker.posted.length, 1, 'only the init was posted');
});

for (const [how, close] of [
  ['terminate()', (client) => client.terminate()],
  ['a worker error', () => FakeWorker.instances.at(-1).emitError({ message: 'no worker script' })],
]) {
  test(`a client closed by ${how} before its download fails preserves that error`, async () => {
    const client = newClient(defaultHandler, new Response(null, { status: 404 }));
    close(client);
    const error = await client.ready.catch((e) => e);
    assert.equal(error, client.closed, '`ready` is rejected with the error that closed the client, not the download error');
    assert.equal(await client.newSession(3).catch((e) => e), error, 'later calls are rejected with the error');
  });
}

test('module bytes refused by the clone fail the load', async () => {
  const client = newClient(defaultHandler, () => { });
  const worker = FakeWorker.instances.at(-1);
  const error = await client.ready.catch((e) => e);
  assert.equal(error, client.closed, 'bytes that cannot be sent close the client');
  assert.ok(!(error instanceof JohariError), 'failed load, not refusal');
  assert.equal(error.name, 'DataCloneError', "the clone's own error");
  assert.equal(worker.terminated, true, 'the worker is terminated');
  assert.equal(worker.posted.length, 0, 'nothing was posted');
});

test('the worker is worker.js beside client.js', async () => {
  using client = newClient();
  assert.equal(FakeWorker.instances.at(-1).url.href, new URL('./worker.js', import.meta.url).href, 'beside client.js');
});

test('a worker-level error closes the client', async () => {
  const client = newClient();
  await client.ready;
  // `newSession` stays unanswered, so it is in flight when `onerror` fires.
  FakeWorker.handler = () => undefined;
  const worker = FakeWorker.instances.at(-1);
  const pending = client.newSession(3).catch((e) => e);
  await flush();
  worker.emitError({ message: 'worker blew up', error: 'thrown as a string' });
  const error = await pending;
  assert.equal(error.closed, true, 'a worker error closes the client');
  assert.equal(error.message, 'worker blew up', "with the event's message");
  assert.equal(worker.terminated, true, 'the broken worker is terminated');
  assert.equal(FakeWorker.instances.length, 1, 'no worker is respawned');
  assert.equal(await client.newSession(3).catch((e) => e), error, 'later calls are rejected with the same error');
});

test('a worker that never loads closes the client', async () => {
  // `init` never replies.
  const client = newClient(() => undefined);
  await flush();
  FakeWorker.instances.at(-1).emitError(new Event('error'));
  const error = await client.ready.catch((e) => e);
  assert.equal(error.closed, true, 'the load failure closes the client');
  assert.match(error.message, /worker\.js/, 'error message specifies the worker');
  assert.equal(FakeWorker.instances.length, 1, 'no worker is respawned');
});

test('a reply that cannot be read closes the client', async () => {
  const client = newClient();
  const session = await client.newSession(3);
  FakeWorker.handler = () => undefined;
  const worker = FakeWorker.instances.at(-1);
  const pending = session.mark(0, 1).catch((e) => e);
  await flush();
  worker.emitMessageError();
  const error = await pending;
  assert.equal(error.closed, true, 'an unreadable reply closes the client');
  assert.equal(error, client.closed, 'a call in flight is rejected with the closing error');
  assert.equal(worker.terminated, true, 'the worker is terminated');
  assert.deepEqual(session.log().marks, [], 'the log is empty');
});

test('terminate() settles every call', async () => {
  const client = newClient();
  const session = await client.newSession(3);
  FakeWorker.handler = () => undefined;
  const inflight = session.propose().catch((e) => e);
  client.terminate();
  const error = await inflight;
  assert.equal(error.closed, true, 'a call in flight is rejected as closed');
  assert.equal(await session.review().catch((e) => e), error, 'a later session call is rejected with the same error');
  assert.equal(await session.free(), undefined, 'freeing a session of a closed client resolves');
  assert.equal(await client.newSession(3).catch((e) => e), error, 'a later newSession resolves');
});

test('an in-flight free when the client closes resolves', async () => {
  const client = newClient();
  const session = await client.newSession(3);
  FakeWorker.handler = () => undefined;
  const freeing = session.free();
  client.terminate();
  assert.equal(await freeing, undefined, 'the free resolves');
});

