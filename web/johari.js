// JavaScript wrapper over the johari-wasm C ABI.

/** Buckets, as passed to `Session.mark`. `null` means "don't know". */
export const Bucket = Object.freeze({ MEH: 0, FINE: 1, LOVE: 2 });

/** Answers to a shown pair, as passed to `Session.answer`. `null` skips the pair. */
export const Outcome = Object.freeze({ A: 0, B: 1, TIE: 2 });

/** The recommended policy values. */
export const Defaults = Object.freeze({
  /**
   * The proposals' `value` (the answer's expected reduction in expected loss of the final list) below which to stop asking
   * and present the final list.
   *
   * The loss is priced at the session's tie cost, so this should be rederived if the tie cost, fit, or scorer change.
   */
  stopThreshold: 0.02,
  /** Level of a review's central credible interval on rank, in [0, 1]. */
  reviewLevel: 0.8,
  /**
   * The tie cost τ. Listing u above l costs 1 if l is noticeably above u (a reversal) and 1 - τ if the two are within δ (a false order).
   * Tying them costs τ if either is noticeably above the other (a false tie). This affects proposals and tie group sizes.
   */
  tieCost: 0.8,
});

/** Each ABI error's code, name, and message. */
const ERRORS = [
  [-1, 'BAD_POINTER', 'null handle'],
  [-2, 'BAD_ARGUMENT', 'bad argument'],
  [-3, 'BAD_BUCKET', 'bad bucket'],
  [-4, 'BAD_OUTCOME', 'bad outcome'],
  [-5, 'ITEM_OUT_OF_RANGE', 'item out of range'],
  [-6, 'ALREADY_MARKED', 'item already marked'],
  [-7, 'BUCKET_STAGE_CLOSED', 'the bucket stage is closed once pairs have been shown'],
  [-8, 'PAIR_NOT_DISTINCT', 'pair is not two distinct items'],
  [-9, 'ITEM_NOT_ACTIVE', 'item has no bucket'],
  [-10, 'PAIR_ALREADY_SHOWN', 'pair already shown'],
];

/** Each ABI error's name and code. */
export const Code = Object.freeze(Object.fromEntries(ERRORS.map(([code, name, _]) => [name, code])));

/** Each ABI error's code and message. */
const MESSAGES = new Map(ERRORS.map(([code, _, message]) => [code, message]));

/**
 * A rejected call. `code` is the ABI error code. The session itself is unaffected. Anything else thrown by a call is a trap
 * (bug or device OOM), after which the instance is in an unknown state and should be discarded.
 */
export class JohariError extends Error {
  /** `what` specifies the argument refused by this layer before the call reaches the module. `options` are `Error`'s. */
  constructor(code, what, options) {
    const message = MESSAGES.get(code) ?? `johari error ${code}`;
    super(what === undefined ? message : `${message}: ${what}`, options);
    this.name = 'JohariError';
    this.code = code;
  }
}

function check(code) {
  if (code < 0) throw new JohariError(code);
  return code;
}

const refuse = (code, what) => {
  throw new JohariError(code, what);
};
const isU32 = (x) => Number.isInteger(x) && x >= 0 && x <= 0xffff_ffff;
const asItem = (x, what) => (isU32(x) ? x : refuse(Code.ITEM_OUT_OF_RANGE, what));
const asCount = (x, what) => (isU32(x) ? x : refuse(Code.BAD_ARGUMENT, what));
const asReal = (x, what) => (typeof x === 'number' ? x : refuse(Code.BAD_ARGUMENT, what));
const asCodeOrNull = (refusal, what) => (x) => (x === null ? -1 : Number.isInteger(x) && x >= 0 && x <= 0x7fff_ffff ? x : refuse(refusal, what));
const asBucket = asCodeOrNull(Code.BAD_BUCKET, 'bucket');
const asOutcome = asCodeOrNull(Code.BAD_OUTCOME, 'outcome');

/** Guarded exports for each instance. */
const guarded = new WeakMap();

/** Guard exports of `instance` so that nothing runs in the instance after a trap. */
function guard(instance) {
  let trap = null;
  const wrap = ([name, value]) => {
    if (typeof value !== 'function') return [name, value];
    const frees = name === 'johari_free_session';
    const call = (...args) => {
      if (trap) {
        if (frees) return undefined;
        throw trap;
      }
      try {
        return value(...args);
      } catch (error) {
        throw (trap = error);
      }
    };
    return [name, call];
  };
  return Object.freeze(Object.fromEntries(Object.entries(instance.exports).map(wrap)));
}

/** A loaded module. Create sessions with `newSession`. */
export class Johari {
  /**
   * @param {BufferSource} bytes The module bytes. From a compiled module, make an instance and pass it to the constructor.
   */
  static async load(bytes) {
    return new Johari((await WebAssembly.instantiate(bytes)).instance);
  }

  /** An instance of the module this file was built with. */
  constructor(instance) {
    if (!guarded.has(instance)) guarded.set(instance, guard(instance));
    this.exports = guarded.get(instance);
  }

  /**
   * A new session over a roster of `nItems` with the given `tieCost`.
   * @param {number} nItems
   * @param {number} [tieCost]
   * @returns {Session}
   * @throws {JohariError} Throws with `BAD_ARGUMENT` if `nItems` is not an integer from 0 to 2¹¹ or `tieCost` is not a number in (0, 1].
   */
  newSession(nItems, tieCost = Defaults.tieCost) {
    return new Session(this, nItems, tieCost);
  }

  /**
   * A new session from `log`. Replays the marks and answers from the session log.
   * @returns {Session}
   * @throws {JohariError} Throws if the log format is invalid or replaying triggers a call refused by the session.
   */
  restore(log) {
    const { nItems, tieCost, marks, pairs } = copyLog(log);
    const session = new Session(this, nItems, tieCost);
    const replay = (entries, what, call) => {
      for (const [k, entry] of entries.entries()) {
        try {
          call(...entry);
        } catch (error) {
          throw error instanceof JohariError ? new JohariError(error.code, `${what}[${k}]`) : error;
        }
      }
    };
    try {
      replay(marks, 'marks', (item, bucket) => session.mark(item, bucket));
      replay(pairs, 'pairs', (a, b, outcome) => session.answer(a, b, outcome));
    } catch (error) {
      session.free();
      throw error;
    }
    return session;
  }
}

/** The first `length` values of `Type` in `exports`' result record. */
function result(exports, Type, length) {
  return new Type(exports.memory.buffer, exports.johari_result() >>> 0, length);
}

/** The block of u32 that the last call left in `exports`' module. */
function block(exports) {
  const [ptr, length] = result(exports, Uint32Array, 2);
  return new Uint32Array(exports.memory.buffer, ptr, length);
}

/** One ranking session over a roster of `nItems`. */
export class Session {
  constructor(johari, nItems, tieCost) {
    nItems = asCount(nItems, 'nItems');
    tieCost = asReal(tieCost, 'tieCost');
    this.exports = johari.exports;
    this.handle = this.exports.johari_new(nItems, tieCost) || refuse(Code.BAD_ARGUMENT);
    // This is a plain property instead of a private field so it can be reached by session proxies.
    this.history = new SessionLog({ nItems, tieCost, marks: [], pairs: [] });
  }

  /** The exports. */
  get e() {
    if (!this.handle) throw new JohariError(Code.BAD_POINTER);
    return this.exports;
  }

  /** Release the session's memory. */
  free() {
    this.exports.johari_free_session(this.handle);
    this.handle = 0;
  }

  static {
    if (Symbol.dispose) this.prototype[Symbol.dispose] = this.prototype.free;
  }

  /** Record a `Bucket`, or `null` for "don't know". */
  mark(item, bucket) {
    check(this.e.johari_mark(this.handle, asItem(item, 'item'), asBucket(bucket)));
    this.history.note('mark', [item, bucket]);
  }

  /**
   * Fit the log if needed and return the next pair to show (or `null` once every pair has been shown).
   * `value` is the answer's expected reduction in expected loss of the final list.
   * @returns {{ a: number, b: number, value: number } | null}
   */
  propose() {
    const e = this.e;
    if (check(e.johari_propose(this.handle)) === 1) return null;
    const [a, b, value] = result(e, Float64Array, 3);
    return { a, b, value };
  }

  /**
   * Record the answer to the shown pair `(a, b)`: an `Outcome` (or `null` for a skip).
   */
  answer(a, b, outcome) {
    check(this.e.johari_record(this.handle, asItem(a, 'a'), asItem(b, 'b'), asOutcome(outcome)));
    this.history.note('answer', [a, b, outcome]);
  }

  /** Retract the last mark or shown pair. Returns 'nothing', 'mark', or 'pair'. */
  undo() {
    const undone = ['nothing', 'mark', 'pair'][check(this.e.johari_undo(this.handle))];
    this.history.note('undo', [], undone);
    return undone;
  }

  /**
   * The session's log.
   * @returns {{ nItems: number, tieCost: number, marks: [number, number | null][], pairs: [number, number, number | null][] }}
   */
  log() {
    return this.history.copy();
  }

  /**
   * Fit the log if needed and build the final list. The list order and tie groups are cut at the least expected loss
   * for the session's tie cost, with rank intervals at level `level`. `rankLo[k]` and `rankHi[k]` are the inclusive ends of
   * `order[k]`'s rank interval. The interval covers every rank at level 1 and is the median at level 0.
   * Use `competitionRanks(groupSizes)` to rank the order.
   * @param {number} [level]
   * @returns {{ order: number[], groupSizes: number[], rankLo: number[], rankHi: number[] }}
   * @throws {JohariError} Throws with `BAD_ARGUMENT` if `level` is not a number in [0, 1].
   * 
   */
  review(level = Defaults.reviewLevel) {
    const e = this.e;
    check(e.johari_review(this.handle, asReal(level, 'level')));
    // Layout: n, the number of tie groups g, the order as u32[n], the group sizes as u32[g], then lo and hi as u32[n] each.
    const out = block(e);
    const [n, g] = out;
    const read = (start, length) => Array.from(out.subarray(start, start + length));
    return { order: read(2, n), groupSizes: read(2 + n, g), rankLo: read(2 + n + g, n), rankHi: read(2 + 2 * n + g, n) };
  }
}

/** Names of `Session` methods (in definition order) served by `worker.js` during a session. */
export function sessionMethods() {
  return Object.entries(Object.getOwnPropertyDescriptors(Session.prototype))
    .filter(([name, { value }]) => !['constructor', 'free', 'log'].includes(name) && typeof value === 'function')
    .map(([name]) => name);
}

/**
 * A plain copy of the session log `log`.
 * @throws {JohariError} Throws with `BAD_ARGUMENT` if the log format is invalid.
 */
export function copyLog(log) {
  const entries = (list, length, what) => {
    if (!Array.isArray(list)) refuse(Code.BAD_ARGUMENT, what);
    return Array.from(list, (entry, k) => (Array.isArray(entry) && entry.length === length ? Array.from(entry) : refuse(Code.BAD_ARGUMENT, `${what}[${k}]`)));
  };
  if (typeof log !== 'object' || log === null) refuse(Code.BAD_ARGUMENT, 'log');
  return { nItems: log.nItems, tieCost: log.tieCost, marks: entries(log.marks, 2, 'marks'), pairs: entries(log.pairs, 3, 'pairs') };
}

/** A session log. */
export class SessionLog {
  /** Create a new log. */
  constructor(log) {
    Object.assign(this, copyLog(log));
  }

  /** Note the session call `method` with its argument list `args` and its `result`. */
  note(method, args, result) {
    if (method === 'mark') this.marks.push([args[0], args[1]]);
    else if (method === 'answer') this.pairs.push([args[0], args[1], args[2]]);
    else if (method === 'undo' && result !== 'nothing') (result === 'pair' ? this.pairs : this.marks).pop();
  }

  /** A plain copy. */
  copy() {
    return copyLog(this);
  }
}

/**
 * Competition ranks from a list's `groupSizes`. Ranks are 0-based.
 * @returns {{rank: number, tied: boolean}[]}
 */
export function competitionRanks(groupSizes) {
  const out = [];
  let position = 0;
  for (const size of groupSizes) {
    for (let k = 0; k < size; k++) out.push({ rank: position, tied: size > 1 });
    position += size;
  }
  return out;
}
