// The panel's conversation log: every question the backend answered, with
// what was on screen when it was asked. Kept in logs/conversation/ so it
// survives restarts: one JSON line per exchange in exchanges.jsonl and the
// screenshot as a JPEG next to it. Only the newest MAX_EXCHANGES are kept.
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const MAX_EXCHANGES = 500;
const INDEX = 'exchanges.jsonl';
const SHOTS = 'shots';
const DATA_URL = /^data:image\/jpeg;base64,([A-Za-z0-9+/=]+)$/;

class ConversationLog {
  constructor(dir, { max = MAX_EXCHANGES } = {}) {
    this.dir = dir;
    this.max = max;
    this.entries = null;
  }

  _load() {
    if (this.entries) return this.entries;
    this.entries = [];
    const file = path.join(this.dir, INDEX);
    if (fs.existsSync(file)) {
      for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
        if (!line.trim()) continue;
        try {
          this.entries.push(JSON.parse(line));
        } catch {
          // a line cut short by a crash: skip it
        }
      }
    }
    return this.entries;
  }

  // Stores one `exchange` message; returns the entry as the panel shows it.
  add(msg) {
    const entries = this._load();
    fs.mkdirSync(path.join(this.dir, SHOTS), { recursive: true });
    const id = String(msg.exchange_id).replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 60);
    let image = null;
    const m = msg.screenshot ? DATA_URL.exec(msg.screenshot) : null;
    if (m) {
      image = `${SHOTS}/${Math.round(msg.asked_ts * 1000)}-${id}.jpg`;
      fs.writeFileSync(path.join(this.dir, image), Buffer.from(m[1], 'base64'));
    }
    const entry = {
      id: `${Math.round(msg.asked_ts * 1000)}-${id}`,
      askedTs: msg.asked_ts,
      answeredTs: msg.ts,
      question: msg.question,
      answer: msg.answer,
      via: msg.via,
      tools: msg.tools,
      latencyMs: msg.latency_ms,
      image,
    };
    entries.push(entry);
    if (entries.length > this.max) {
      for (const old of entries.splice(0, entries.length - this.max)) this._removeImage(old);
      this._rewrite();
    } else {
      fs.appendFileSync(path.join(this.dir, INDEX), `${JSON.stringify(entry)}\n`);
    }
    return this.view(entry);
  }

  // Oldest first, with a file:// URL for each screenshot.
  list() {
    return this._load().map((e) => this.view(e));
  }

  view(entry) {
    const imageUrl = entry.image ? pathToFileURL(path.join(this.dir, entry.image)).href : null;
    return { ...entry, imageUrl };
  }

  clear() {
    this.entries = [];
    fs.rmSync(path.join(this.dir, SHOTS), { recursive: true, force: true });
    fs.rmSync(path.join(this.dir, INDEX), { force: true });
  }

  _removeImage(entry) {
    if (entry.image) fs.rmSync(path.join(this.dir, entry.image), { force: true });
  }

  _rewrite() {
    const file = path.join(this.dir, INDEX);
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, this.entries.map((e) => `${JSON.stringify(e)}\n`).join(''));
    fs.renameSync(tmp, file);
  }
}

module.exports = { ConversationLog, MAX_EXCHANGES };
