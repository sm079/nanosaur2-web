// Prompt tokenization matching the Nanosaur2 ComfyUI nodes (nanosaur2_support/text_encoder.py):
//  - "(text:weight)" groups are stripped to their text; every token overlapping a group gets its
//    weight (the product, for nested or overlapping groups). Other parentheses stay literal.
//  - Gemma3 SentencePiece BPE (the model ships inside the text encoder file as `spiece_model`),
//    <bos> first, no <eos>, at most 256 tokens.
// The weights don't scale embeddings: the DiT uses them as attention biases and pooling weights.
//
// The SentencePiece encoder below follows sentencepiece's bpe_model.cc for this model's settings
// (identity normalizer, spaces escaped to U+2581, no dummy prefix, byte fallback).

const MAX_LENGTH = 256;
const BOS = 2;
const SPACE = "▁";
const TYPE = { NORMAL: 1, UNKNOWN: 2, CONTROL: 3, USER_DEFINED: 4, UNUSED: 5, BYTE: 6 };

// ---------------------------------------------------------------- prompt emphasis

const WEIGHT = /^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$/;

// Port of parse_prompt_emphasis: returns the plain text (as an array of code points, matching
// Python string indexing) and [start, end, weight] spans into it.
export function parseEmphasis(caption) {
  const s = Array.from(caption);
  const spans = [];
  const out = [];
  let cursor = 0;
  let idx = 0;
  while (idx < s.length) {
    if (s[idx] !== "(") { idx++; continue; }
    let depth = 1;
    let end = idx + 1;
    while (end < s.length && depth > 0) {
      if (s[end] === "(") depth++;
      else if (s[end] === ")") depth--;
      end++;
    }
    if (depth !== 0) { idx++; continue; }
    const inner = s.slice(idx + 1, end - 1);
    let innerDepth = 0;
    let colon = -1;
    inner.forEach((ch, i) => {
      if (ch === "(") innerDepth++;
      else if (ch === ")") innerDepth--;
      else if (ch === ":" && innerDepth === 0) colon = i;
    });
    const text = inner.slice(0, colon);
    const weightText = inner.slice(colon + 1).join("").trim();
    if (colon === -1 || !text.length || !WEIGHT.test(weightText)) { idx++; continue; }
    out.push(...s.slice(cursor, idx));
    spans.push([out.length, out.length + text.length, parseFloat(weightText)]);
    out.push(...text);
    cursor = end;
    idx = end;
  }
  out.push(...s.slice(cursor));
  return { chars: out, spans };
}

// ---------------------------------------------------------------- SentencePiece model

// Minimal protobuf reader for sentencepiece's ModelProto: just the pieces (field 1:
// { piece: 1, score: 2, type: 3 }).
function readPieces(bytes) {
  const dec = new TextDecoder("utf-8", { ignoreBOM: true }); // some pieces start with U+FEFF
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 0;
  const varint = () => {
    let v = 0, mul = 1, b;
    do { b = bytes[pos++]; v += (b & 0x7f) * mul; mul *= 128; } while (b & 0x80);
    return v;
  };
  const skip = (wire) => {
    if (wire === 0) varint();
    else if (wire === 1) pos += 8;
    else if (wire === 2) pos += varint();
    else if (wire === 5) pos += 4;
    else throw new Error(`tokenizer: unsupported protobuf wire type ${wire}`);
  };
  const pieces = [];
  while (pos < bytes.length) {
    const tag = varint();
    const field = Math.floor(tag / 8), wire = tag & 7;
    if (field !== 1 || wire !== 2) { skip(wire); continue; }
    const end = varint() + pos;
    const p = { piece: "", score: 0, type: TYPE.NORMAL };
    while (pos < end) {
      const t = varint();
      const f = Math.floor(t / 8), w = t & 7;
      if (f === 1 && w === 2) { const n = varint(); p.piece = dec.decode(bytes.subarray(pos, pos + n)); pos += n; }
      else if (f === 2 && w === 5) { p.score = dv.getFloat32(pos, true); pos += 4; }
      else if (f === 3 && w === 0) p.type = varint();
      else skip(w);
    }
    pieces.push(p);
  }
  return pieces;
}

// Max-heap of merge candidates: higher score first, then the leftmost.
class Agenda {
  constructor() { this.h = []; }
  get size() { return this.h.length; }
  static before(a, b) { return a.score !== b.score ? a.score > b.score : a.left < b.left; }
  push(x) {
    const h = this.h;
    h.push(x);
    let i = h.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Agenda.before(h[i], h[p])) break;
      [h[i], h[p]] = [h[p], h[i]];
      i = p;
    }
  }
  pop() {
    const h = this.h;
    const top = h[0];
    const last = h.pop();
    if (h.length) {
      h[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < h.length && Agenda.before(h[l], h[m])) m = l;
        if (r < h.length && Agenda.before(h[r], h[m])) m = r;
        if (m === i) break;
        [h[i], h[m]] = [h[m], h[i]];
        i = m;
      }
    }
    return top;
  }
}

export class SentencePieceBPE {
  constructor(modelBytes) {
    const pieces = readPieces(modelBytes);
    this.vocab = new Map(); // mergeable pieces (normal, user-defined, unused) -> { id, score }
    this.userDefined = new Map(); // first char -> user-defined pieces, longest first
    this.bytes = new Array(256);
    this.unk = 0;
    pieces.forEach((p, id) => {
      if (p.type === TYPE.NORMAL || p.type === TYPE.USER_DEFINED || p.type === TYPE.UNUSED) {
        if (!this.vocab.has(p.piece)) this.vocab.set(p.piece, { id, score: p.score });
      }
      if (p.type === TYPE.USER_DEFINED) {
        const first = String.fromCodePoint(p.piece.codePointAt(0));
        if (!this.userDefined.has(first)) this.userDefined.set(first, []);
        this.userDefined.get(first).push(Array.from(p.piece));
      }
      if (p.type === TYPE.UNKNOWN) this.unk = id;
      if (p.type === TYPE.BYTE) this.bytes[parseInt(p.piece.slice(3, 5), 16)] = id;
    });
    for (const list of this.userDefined.values()) list.sort((a, b) => b.length - a.length);
    this.size = pieces.length;
  }

  // Length (in code points) of the longest user-defined symbol at chars[i], or 0.
  matchUserDefined(chars, i) {
    const list = this.userDefined.get(chars[i]);
    if (!list) return 0;
    for (const p of list) {
      if (p.length <= chars.length - i && p.every((c, k) => chars[i + k] === c)) return p.length;
    }
    return 0;
  }

  // chars: code points of the (unnormalized) text. Returns [{ id, begin, end }] with code point
  // offsets into chars (normalization only maps " " to U+2581, one to one).
  encodeChars(chars) {
    const norm = chars.map((c) => (c === " " ? SPACE : c));
    // initial symbols: user-defined symbols whole (frozen), everything else one character
    const sym = [];
    for (let i = 0; i < norm.length;) {
      const ud = this.matchUserDefined(norm, i);
      const n = ud || 1;
      sym.push({ text: norm.slice(i, i + n).join(""), begin: i, end: i + n, freeze: ud > 0, prev: sym.length - 1, next: -1 });
      i += n;
    }
    sym.forEach((s, k) => { s.next = k + 1 < sym.length ? k + 1 : -1; });

    const agenda = new Agenda();
    const consider = (l, r) => {
      if (l < 0 || r < 0 || sym[l].freeze || sym[r].freeze) return;
      const text = sym[l].text + sym[r].text;
      const v = this.vocab.get(text);
      if (v) agenda.push({ left: l, right: r, score: v.score, len: text.length });
    };
    for (let k = 1; k < sym.length; k++) consider(k - 1, k);
    while (agenda.size) {
      const top = agenda.pop();
      const L = sym[top.left], R = sym[top.right];
      // stale: one side already merged away or grown
      if (!L.text || !R.text || L.text.length + R.text.length !== top.len) continue;
      L.text += R.text;
      L.end = R.end;
      L.next = R.next;
      if (R.next >= 0) sym[R.next].prev = top.left;
      R.text = "";
      consider(L.prev, top.left);
      consider(top.left, L.next);
    }

    const out = [];
    for (let k = sym.length ? 0 : -1; k !== -1; k = sym[k].next) {
      const s = sym[k];
      const v = this.vocab.get(s.text);
      if (v) { out.push({ id: v.id, begin: s.begin, end: s.end }); continue; }
      // unknown piece: byte fallback, one <0xXX> token per UTF-8 byte
      for (const b of new TextEncoder().encode(s.text)) out.push({ id: this.bytes[b] ?? this.unk, begin: s.begin, end: s.end });
    }
    return out;
  }
}

export class Nanosaur2Tokenizer {
  // modelBytes: the `spiece_model` tensor of the text encoder file
  constructor(modelBytes) {
    this.spm = new SentencePieceBPE(modelBytes);
  }

  // -> { ids, weights }: <bos> + tokens, truncated to 256
  encode(text) {
    const { chars, spans } = parseEmphasis(text);
    const ids = [BOS];
    const weights = [1];
    for (const t of this.spm.encodeChars(chars)) {
      let w = 1;
      for (const [b, e, sw] of spans) if (t.begin < e && t.end > b) w *= sw;
      ids.push(t.id);
      weights.push(w);
    }
    return { ids: ids.slice(0, MAX_LENGTH), weights: weights.slice(0, MAX_LENGTH) };
  }
}
