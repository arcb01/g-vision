'use strict';

const test = require('node:test');
const assert = require('node:assert');
const examples = require('../../schema/examples.json');
const { validateMessage, makeMessage } = require('../protocol');

test('shared examples validate', () => {
  for (const ex of examples) {
    const result = validateMessage(ex);
    assert.ok(result.ok, `${ex.type}: ${result.errors}`);
  }
});

test('makeMessage builds valid messages', () => {
  assert.ok(validateMessage(makeMessage('clear', { reason: null })).ok);
  assert.ok(validateMessage(makeMessage('config_changed', { changes: { a: 1 } })).ok);
});

test('invalid messages are rejected', () => {
  const bad = [
    { v: 1, ts: 0, type: 'nope' },
    { v: 2, ts: 0, type: 'clear', reason: null },
    { v: 1, ts: 0, type: 'dim', on: true, strength: 1.5 },
    { v: 1, ts: 0, type: 'focus', refs: ['enemy'], segment_id: null },
    { v: 1, ts: 0, type: 'clear', reason: null, extra: 1 },
    { v: 1, ts: 0, type: 'clear' }, // fields with defaults are still required on the wire
  ];
  for (const msg of bad) assert.ok(!validateMessage(msg).ok, JSON.stringify(msg));
});
