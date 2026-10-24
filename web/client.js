// Main-thread client for worker.js.
//
// Sessions continue to work through proxies and reactive state. However, clients do not.

import { Code, JohariError, SessionLog, copyLog, sessionMethods } from './johari.js';

export { Bucket, Code, Defaults, JohariError, Outcome, competitionRanks } from './johari.js';

/** The shipped module path relative to this file. */
// This function is in `client.js` so a bundler doesn't output the module twice.
export function shippedModule() {
  return new URL('./johari.wasm', import.meta.url);
}

/** Get a module's bytes from a URL, a `Response`, or a byte array. */
async function moduleBytes(source) {
  if (source instanceof URL) source = await fetch(source);
  if (source instanceof Response) {
    if (!source.ok) throw new Error(`HTTP ${source.status} ${source.statusText} for ${source.url}`);
    return source.arrayBuffer();
  }
  return source;
}

/** A client of one worker. */
export class JohariClient {
  #worker;
  #nextId = 1;
  #pending = new Map();

  /**
   * @param {string | URL | Response | BufferSource} wasm The module to load; defaults to `shippedModule()`.
   *   Strings are parsed as URLs relative to this file.
   */
  constructor(wasm = shippedModule()) {
    const source = typeof wasm === 'string' ? new URL(wasm, import.meta.url) : wasm;
    /** The error that closed the client (once it is closed). */
    this.closed = null;
    // This is written out literally for bundlers.
    this.#worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.#worker.onmessage = ({ data }) => this.#onReply(data);
    this.#worker.onerror = (event) => this.#close(new Error(event.message || 'worker failed; make sure worker.js and its imports are served as JavaScript and allowed by the Content-Security-Policy'));
    this.#worker.onmessageerror = () => this.#close(new Error('a reply from the worker could not be read'));
    this.ready = moduleBytes(source)
      .then((bytes) => this.#send('init', [bytes]))
      .catch((error) => {
        this.#close(error);
        throw this.closed;
      });
    // We observe the result so a failed load is never an unhandled rejection. Awaiters of `ready` still see it.
    this.ready.catch(() => { });
  }

  #onReply(data) {
    const entry = this.#pending.get(data.id);
    if (!entry) return;
    this.#pending.delete(data.id);
    if (!('error' in data)) {
      entry.resolve(data.result);
      return;
    }
    const error = data.code === undefined ? new Error(data.error) : Object.assign(new JohariError(data.code), { message: data.error });
    if (data.code === undefined) this.#close(error);
    entry.reject(error);
  }

  /** Close the client. This terminates the worker and causes all pending and future calls to be rejected with `error`. */
  #close(error) {
    if (this.closed) return;
    error.closed = true;
    this.closed = error;
    this.#worker.terminate();
    for (const { reject } of this.#pending.values()) reject(error);
    this.#pending.clear();
  }

  /** Post `method` with the argument list `args` on the session `session` if one is given. */
  #send(method, args, session) {
    if (this.closed) return Promise.reject(this.closed);
    const id = this.#nextId++;
    this.#worker.postMessage({ id, method, session, args });
    return new Promise((resolve, reject) => this.#pending.set(id, { resolve, reject }));
  }

  /** Invoke `#send` on the session served as `id` by the worker. */
  async #call(method, args, id) {
    try {
      return this.#send(method, args, id);
    } catch (cause) {
      if (cause?.name !== 'DataCloneError') throw cause;
      throw new JohariError(Code.BAD_ARGUMENT, 'not transferable to the worker', { cause });
    }
  }

  /**
   * A session in the worker.
   * @returns {Promise<RemoteSession>}
   */
  async newSession(nItems, tieCost) {
    await this.ready;
    return this.#open(await this.#call('newSession', [nItems, tieCost]));
  }

  /**
   * A session in the worker rebuilt from `log`.
   * @returns {Promise<RemoteSession>}
   */
  async restore(log) {
    if (this.closed) throw this.closed;
    const copy = copyLog(log);
    await this.ready;
    return this.#open(await this.#call('restore', [copy]));
  }

  /** The remote session that the worker opened as `id`, with its log. */
  #open({ id, log }) {
    return new RemoteSession((method, args, session) => this.#call(method, args, session), id, log);
  }

  /** Close the client. An in-flight call rejects with `error.closed` set. */
  terminate() {
    this.#close(new Error('the client was terminated'));
  }

  static {
    if (Symbol.dispose) this.prototype[Symbol.dispose] = this.prototype.terminate;
  }
}

/** A `Session` in the worker. */
export class RemoteSession {
  /**
   * @param {(method: string, args: unknown[], session?: number) => Promise<unknown>} send The client's call;
   *   i.e. `method` with the argument list `args` on the `session` served by the worker if one is given.
   */
  constructor(send, id, log) {
    // These are plain properties instead of private fields so they can be reached by session proxies.
    this.send = send;
    this.id = id;
    this.history = new SessionLog(log);
  }

  /** The session's log. */
  log() {
    return this.history.copy();
  }

  /** Release the session. */
  async free() {
    const id = this.id;
    if (!id) return;
    this.id = 0;
    try {
      await this.send('free', [id]);
    } catch (error) {
      if (!error.closed) throw error;
    }
  }

  static {
    if (Symbol.asyncDispose) this.prototype[Symbol.asyncDispose] = this.prototype.free;

    for (const method of sessionMethods()) {
      Object.defineProperty(this.prototype, method, {
        async value(...args) {
          if (!this.id) throw new JohariError(Code.BAD_POINTER);
          const result = await this.send(method, args, this.id);
          this.history.note(method, args, result);
          return result;
        },
        writable: true,
        configurable: true,
      });
    }
  }
}
