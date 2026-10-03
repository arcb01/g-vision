// Shared message protocol, validated against schema/messages.schema.json,
// which is generated from python/src/gvision/protocol/messages.py.
'use strict';

const Ajv2020 = require('ajv/dist/2020');
const schema = require('../schema/messages.schema.json');

const PROTOCOL_VERSION = 1;

// strict: false lets Ajv ignore the OpenAPI-style "discriminator" keyword
// that Pydantic emits; oneOf with const "type" values still applies.
const ajv = new Ajv2020({ strict: false, allErrors: true });
const validate = ajv.compile(schema);

function validateMessage(msg) {
  if (validate(msg)) return { ok: true, msg };
  return { ok: false, errors: ajv.errorsText(validate.errors) };
}

function parseMessage(text) {
  let msg;
  try {
    msg = JSON.parse(text);
  } catch (e) {
    return { ok: false, errors: `invalid JSON: ${e.message}` };
  }
  return validateMessage(msg);
}

function makeMessage(type, fields = {}) {
  return { v: PROTOCOL_VERSION, ts: Date.now() / 1000, type, ...fields };
}

module.exports = { PROTOCOL_VERSION, schema, validateMessage, parseMessage, makeMessage };
