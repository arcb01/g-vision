'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { SessionStore, checkGame, gameList } = require('../sessions');
const games = require('../games.json');

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-sessions-'));
const exchange = (q, ts) => ({
  v: 1, ts: ts + 1, type: 'exchange', exchange_id: `x${ts}`, asked_ts: ts, question: q, answer: 'a', via: 'voice',
  tools: ['lookup'], latency_ms: {}, steps: [], screenshot: null,
});

test('the known games are valid and unique', () => {
  const ids = new Set();
  for (const g of games) {
    assert.deepStrictEqual(checkGame(g), g);
    assert.ok(!ids.has(g.id), g.id);
    ids.add(g.id);
  }
});

test('a custom game gets an id from its name and a full wiki address', () => {
  assert.deepStrictEqual(checkGame({ name: 'Hades II', wiki: 'hades.wiki.gg/' }),
    { id: 'hades-ii', name: 'Hades II', wiki: 'https://hades.wiki.gg' });
  assert.throws(() => checkGame({ name: '', wiki: 'x.com' }), /name/);
  assert.throws(() => checkGame({ name: 'X', wiki: 'nope' }), /web address/);
});

test('a session collects the exchanges asked while it is active', () => {
  const store = new SessionStore(tmp());
  assert.strictEqual(store.add(exchange('before', 1)), null);
  const s = store.start(games[0], 1000);
  const added = store.add(exchange('what does the mason want', 2));
  assert.strictEqual(added.sessionId, s.id);
  assert.strictEqual(store.active().id, s.id);
  const s2 = store.start({ name: 'Terraria', wiki: 'https://terraria.wiki.gg' }, 2000);
  assert.strictEqual(store.active().id, s2.id);
  const [newest, oldest] = store.list();
  assert.strictEqual(newest.id, s2.id);
  assert.ok(oldest.endedTs && oldest.count === 1 && oldest.lastQuestion === 'what does the mason want');
  assert.strictEqual(store.get(s.id).exchanges[0].question, 'what does the mason want');
  store.end();
  assert.strictEqual(store.active(), null);
  store.remove(s.id);
  assert.strictEqual(store.list().length, 1);
});

test('the game list carries how fresh each wiki index is', () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'minecraft.json'), JSON.stringify({ game: 'minecraft', pages: 8214, updated_ts: 5, source: 'Minecraft Wiki' }));
  fs.writeFileSync(path.join(dir, 'hades-ii.json'), JSON.stringify({ game: 'hades-ii', name: 'Hades II', wiki: 'https://hades.wiki.gg', pages: 900, updated_ts: 6 }));
  const list = gameList(games, dir);
  const mc = list.find((g) => g.id === 'minecraft');
  assert.strictEqual(mc.pages, 8214);
  assert.strictEqual(mc.updatedTs, 5);
  assert.strictEqual(list.find((g) => g.id === 'terraria').pages, 0);
  assert.ok(list.find((g) => g.id === 'hades-ii').custom);
});
