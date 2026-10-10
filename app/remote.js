// Two-PC mode, app side. The AI server's app runs a small control server
// (HTTP + JSON) that the gaming PC's app uses for everything that is not the
// live feed: the server's services, its settings and vision model downloads,
// which wikis it has indexed, its version and its clock. It answers even when
// the backend is down, so a bad setting can still be fixed from the gaming PC.
// The live feed itself (frames, voice, highlights) goes through the backend's
// bridge on its own port.
'use strict';

const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');

const MAX_BODY = 64 * 1024;

// The checkout's commit, so the two PCs can tell they run the same version.
function gitBuild(repoRoot) {
  try {
    const gitDir = path.join(repoRoot, '.git');
    const head = fs.readFileSync(path.join(gitDir, 'HEAD'), 'utf8').trim();
    if (!head.startsWith('ref: ')) return head.slice(0, 12);
    const ref = head.slice(5);
    const loose = path.join(gitDir, ref);
    if (fs.existsSync(loose)) return fs.readFileSync(loose, 'utf8').trim().slice(0, 12);
    const packed = fs.readFileSync(path.join(gitDir, 'packed-refs'), 'utf8');
    const line = packed.split(/\r?\n/).find((l) => l.endsWith(` ${ref}`));
    return line ? line.slice(0, 12) : null;
  } catch {
    return null;
  }
}

// This PC's LAN addresses, for the AI server's Home page.
function localAddresses(interfaces = os.networkInterfaces()) {
  const out = [];
  for (const [name, addrs] of Object.entries(interfaces)) {
    for (const a of addrs || []) {
      if (a.family === 'IPv4' && !a.internal && !a.address.startsWith('169.254.')) out.push({ name, address: a.address });
    }
  }
  return out;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(new Error('request too large'));
        req.destroy();
      } else chunks.push(c);
    });
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8');
      if (!text) return resolve({});
      try {
        return resolve(JSON.parse(text));
      } catch {
        return reject(new Error('invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

// routes: { 'GET /info': async ({ body, params, peer }) => result, 'POST /services/:name/:action': ... }
// A handler's return value is sent as JSON; a thrown error as { ok: false, error }.
function createControlServer(routes, { onRequest = null } = {}) {
  const table = Object.entries(routes).map(([key, handler]) => {
    const [method, pattern] = key.split(' ');
    const names = [];
    const re = new RegExp(`^${pattern.replace(/:(\w+)/g, (_, n) => {
      names.push(n);
      return '([^/]+)';
    })}$`);
    return { method, re, names, handler };
  });
  return http.createServer(async (req, res) => {
    const send = (status, payload) => {
      res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify(payload));
    };
    const url = new URL(req.url, 'http://x');
    const route = table.find((r) => r.method === req.method && r.re.test(url.pathname));
    if (!route) return send(404, { ok: false, error: 'not found' });
    const m = route.re.exec(url.pathname);
    const params = Object.fromEntries(route.names.map((n, i) => [n, decodeURIComponent(m[i + 1])]));
    const peer = (req.socket.remoteAddress || '').replace(/^::ffff:/, '');
    if (onRequest) onRequest(peer);
    try {
      const body = req.method === 'POST' ? await readBody(req) : {};
      return send(200, await route.handler({ body, params, peer }));
    } catch (err) {
      return send(400, { ok: false, error: err.message });
    }
  });
}

// The gaming PC's side: calls the AI server's control server.
class RemoteClient {
  constructor(baseUrl, { timeoutMs = 3000 } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeoutMs = timeoutMs;
  }

  async request(method, route, body = undefined, timeoutMs = this.timeoutMs) {
    let r;
    try {
      r = await fetch(`${this.baseUrl}${route}`, {
        method,
        headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      throw new Error(friendlyError(err, this.baseUrl));
    }
    let payload;
    try {
      payload = await r.json();
    } catch {
      throw new Error(`${this.baseUrl} answered, but not like a G-VISION AI server`);
    }
    if (!r.ok) throw new Error(payload.error || `HTTP ${r.status}`);
    return payload;
  }

  get(route, timeoutMs) {
    return this.request('GET', route, undefined, timeoutMs);
  }

  post(route, body = {}, timeoutMs) {
    return this.request('POST', route, body, timeoutMs);
  }
}

// What a failed connection most likely means, in words.
function friendlyError(err, url) {
  const code = (err.cause && err.cause.code) || err.code || '';
  const host = (() => {
    try {
      return new URL(url).host;
    } catch {
      return url;
    }
  })();
  if (err.name === 'TimeoutError' || code === 'UND_ERR_CONNECT_TIMEOUT' || code === 'ETIMEDOUT') {
    return `No answer from ${host}. Check the address, and that Windows Firewall on the AI server PC allows G-VISION.`;
  }
  if (code === 'ECONNREFUSED') {
    return `${host} is there but G-VISION isn't listening. Start G-VISION on that PC with "This PC is: AI server".`;
  }
  if (code === 'ENOTFOUND' || code === 'EAI_AGAIN') return `Can't find a PC called ${host.replace(/:\d+$/, '')} on the network.`;
  if (code === 'EHOSTUNREACH' || code === 'ENETUNREACH') return `Can't reach ${host} from this PC's network.`;
  return `Can't reach ${host}: ${code || err.message}`;
}

// Gaming PC time minus AI server time, from the round trip with the
// smallest delay among the recent ones.
class ClockSync {
  constructor(keep = 10) {
    this.keep = keep;
    this.samples = [];
  }

  add(sentMs, receivedMs, serverS) {
    const rtt = receivedMs - sentMs;
    this.samples.push({ rtt, offset: (sentMs + receivedMs) / 2000 - serverS });
    if (this.samples.length > this.keep) this.samples.shift();
  }

  get offset() {
    if (!this.samples.length) return 0;
    return this.samples.reduce((a, b) => (b.rtt < a.rtt ? b : a)).offset;
  }
}

module.exports = { ClockSync, RemoteClient, createControlServer, friendlyError, gitBuild, localAddresses };
