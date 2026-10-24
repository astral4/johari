// Web Worker hosting the WebAssembly module.
//
// Protocol: `{ id, method, session, args }` in, where `args` is the call's argument list and `session` specifies the session
// for a `Session` method; `{ id, result }` or `{ id, error, code }` out. See `client.js`.

import { Johari, JohariError, sessionMethods } from './johari.js';

const SESSION_METHODS = new Set(sessionMethods());

/** Serve the protocol on `scope`. */
export function serve(scope) {
  let johari = null;
  const sessions = new Map();
  let nextSession = 1;

  /** Serve `s` under a new id. Returns the id and the session log. */
  function open(s) {
    const id = nextSession++;
    sessions.set(id, s);
    return { id, log: s.log() };
  }

  /** The session served as `id`. */
  function session(id) {
    const s = sessions.get(id);
    if (!s) throw new Error(`no session ${id}`);
    return s;
  }

  const host = {
    async init(bytes) {
      johari = await Johari.load(bytes);
    },
    newSession(nItems, tieCost) {
      return open(johari.newSession(nItems, tieCost));
    },
    restore(log) {
      return open(johari.restore(log));
    },
    free(id) {
      session(id).free();
      sessions.delete(id);
    },
  };

  /** The call made by a request. A malformed request throws a plain Error, which is fatal. */
  function dispatch(method, id, args) {
    if (!Array.isArray(args)) throw new Error('args is not a list');
    if (id === undefined) {
      if (!Object.hasOwn(host, method)) throw new Error(`unknown method ${method}`);
      if (!johari && method !== 'init') throw new Error(`${method} without a module`);
      return host[method](...args);
    }
    if (!SESSION_METHODS.has(method)) throw new Error(`unknown method ${method}`);
    return session(id)[method](...args);
  }

  /** Make request `id`'s call and send its reply. */
  async function reply(id, method, sessionId, args) {
    try {
      scope.postMessage({ id, result: await dispatch(method, sessionId, args) });
    } catch (error) {
      const code = error instanceof JohariError ? error.code : undefined;
      scope.postMessage({ id, error: String(error?.message ?? error), code });
    }
  }

  // This is not async so that an unreadable request throws here where it is visible to the host.
  scope.onmessage = (event) => {
    const { id, method, session: sessionId, args } = event.data;
    return reply(id, method, sessionId, args);
  };
  scope.onmessageerror = () => {
    throw new Error('a request to the worker could not be read');
  };
}

// A dedicated worker serves on its own global.
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) serve(self);
