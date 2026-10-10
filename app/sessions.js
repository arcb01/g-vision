// Sessions: one per sitting with one game. Starting a session picks the game,
// whose wiki the backend indexes and answers from (lookup). Each session keeps
// its own conversation, like the Log, in logs/sessions/<id>/: session.json
// plus exchanges.jsonl and screenshots (conversation.js).
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { ConversationLog } = require('./conversation');

const META = 'session.json';
const MAX_EXCHANGES = 5000;
const GAME_ID = /^[a-z0-9][a-z0-9-]{0,63}$/;

function slug(name) {
  return String(name).toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64);
}

// A game from the panel: a known one, or a custom name and wiki URL.
function checkGame(game) {
  if (!game || typeof game !== 'object') throw new Error('pick a game');
  const name = String(game.name || '').trim();
  if (!name) throw new Error('the game needs a name');
  let wiki = String(game.wiki || '').trim();
  if (!/^https?:\/\//i.test(wiki)) wiki = `https://${wiki}`;
  let url;
  try {
    url = new URL(wiki);
  } catch {
    throw new Error(`"${game.wiki}" is not a web address`);
  }
  if (!url.hostname.includes('.')) throw new Error(`"${game.wiki}" is not a web address`);
  const id = game.id && GAME_ID.test(game.id) ? game.id : slug(name);
  if (!GAME_ID.test(id)) throw new Error('the game name needs a letter or a number');
  return { id, name, wiki: url.href.replace(/\/$/, '') };
}

class SessionStore {
  constructor(dir) {
    this.dir = dir;
    this.logs = new Map();
  }

  _read(id) {
    try {
      return JSON.parse(fs.readFileSync(path.join(this.dir, id, META), 'utf8'));
    } catch {
      return null;
    }
  }

  _write(session) {
    const dir = path.join(this.dir, session.id);
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, META);
    fs.writeFileSync(`${file}.tmp`, JSON.stringify(session, null, 2));
    fs.renameSync(`${file}.tmp`, file);
  }

  log(id) {
    if (!this.logs.has(id)) this.logs.set(id, new ConversationLog(path.join(this.dir, id), { max: MAX_EXCHANGES }));
    return this.logs.get(id);
  }

  // Newest first, with how many questions each has.
  list() {
    if (!fs.existsSync(this.dir)) return [];
    const out = [];
    for (const id of fs.readdirSync(this.dir)) {
      const s = this._read(id);
      if (!s) continue;
      const entries = this.log(id).list();
      const last = entries[entries.length - 1];
      out.push({ ...s, count: entries.length, lastTs: last ? last.askedTs : s.startedTs, lastQuestion: last ? last.question : null });
    }
    return out.sort((a, b) => b.startedTs - a.startedTs);
  }

  // The session still going (the newest one not ended), if any.
  active() {
    return this.list().find((s) => !s.endedTs) || null;
  }

  start(game, now = Date.now()) {
    const checked = checkGame(game);
    this.end(now);
    const session = { id: `s${now}`, game: checked, startedTs: now / 1000, endedTs: null };
    this._write(session);
    return session;
  }

  end(now = Date.now()) {
    for (const s of this.list()) {
      if (!s.endedTs) this._write({ id: s.id, game: s.game, startedTs: s.startedTs, endedTs: now / 1000 });
    }
  }

  get(id) {
    const session = this._read(id);
    if (!session) return null;
    return { ...session, exchanges: this.log(id).list() };
  }

  remove(id) {
    if (!/^s\d+$/.test(id)) return;
    this.logs.delete(id);
    fs.rmSync(path.join(this.dir, id), { recursive: true, force: true });
  }

  // Adds an `exchange` message to the active session; returns { sessionId, entry } or null.
  add(msg) {
    const active = this.active();
    if (!active) return null;
    return { sessionId: active.id, entry: this.log(active.id).add(msg) };
  }
}

// The picker's list: the known games, then custom games used before, each
// with what the backend's wiki index says (<wikiDir>/<id>.json).
function gameList(known, wikiDir) {
  const index = new Map();
  if (fs.existsSync(wikiDir)) {
    for (const f of fs.readdirSync(wikiDir)) {
      if (!f.endsWith('.json')) continue;
      try {
        const meta = JSON.parse(fs.readFileSync(path.join(wikiDir, f), 'utf8'));
        if (meta && meta.game) index.set(meta.game, meta);
      } catch {
        // half-written: skip
      }
    }
  }
  const view = (g) => {
    const meta = index.get(g.id);
    return { ...g, pages: meta ? meta.pages : 0, updatedTs: meta ? meta.updated_ts : null, source: meta ? meta.source : null };
  };
  const out = known.map(view);
  for (const meta of index.values()) {
    if (!known.some((g) => g.id === meta.game)) out.push({ ...view({ id: meta.game, name: meta.name, wiki: meta.wiki }), custom: true });
  }
  return out;
}

module.exports = { SessionStore, checkGame, gameList, slug };
