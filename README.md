# G-VISION

A real-time assistant that helps you perceive what's happening in a game: it
finds and highlights objects, reads on-screen text, remembers recent events and
answers spoken questions with voice and visual highlights.

The full design is in [game-vision-agent-plan.md](game-vision-agent-plan.md).
This repository is at the skeleton stage: the processes talk to each other over
the shared message schema, and a demo mode streams synthetic objects and a
scripted answer so the overlay can be built before capture, detection and Qwen
exist.

## Layout

```
python/            Perception, agent and audio processes (package: gvision)
  src/gvision/
    protocol/      Message schema: source of truth for Python <-> Electron
    bridge.py      Local WebSocket server the Electron app connects to
    demo.py        Synthetic objects and answers (stand-in for perception + agent)
    perception/    Capture, detector, tracker, text watcher, world state (todo)
    agent/         Snapshot builder, Qwen client, tools, narrator (todo)
    audio/         Push-to-talk, Nemotron ASR, Kokoro TTS (todo)
app/               Electron app: transparent overlay + control panel
  main.js          Owns the bridge connection, validates and forwards messages
  src/overlay.*    PixiJS overlay: outlines, semantic colors, spotlight dimming
  src/panel.*      Control panel shell: dashboard, answer, controls
schema/
  messages.schema.json  Generated JSON Schema (do not edit by hand)
  examples.json         One example of every message, tested from both sides
```

## Requirements

- Windows 10/11 with an NVIDIA GPU (12 GB VRAM minimum) for the real pipeline.
  The skeleton itself runs anywhere.
- Python 3.11+
- Node.js 22+

## Setup

```bash
# Python
cd python
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"

# Electron app
cd ../app
npm install
```

## Run the demo

In one terminal, start the Python bridge with synthetic data:

```bash
cd python
python -m gvision --demo
```

In another, start the app:

```bash
cd app
npm start
```

The overlay covers the primary display (transparent and click-through) and
shows three moving objects. Every 8 seconds a scripted answer plays: the screen
dims, and the spotlight moves from the barrel to the spikes to the guard while
the panel highlights the matching segment. `Ctrl+Shift+X` dismisses
highlights and dimming. The app reconnects on its own if the bridge restarts.

## Message schema

Python and Electron exchange JSON messages over `ws://127.0.0.1:8765`
(plan section 12): `objects`, `highlight`, `focus`, `dim`, `answer`,
`segment_started`, `answer_finished`, `badges`, `clear`, `status` and
`config_changed`. Every message carries `v` (protocol version), `type` and
`ts`; coordinates are normalized to 0..1; elements are addressed by reference
IDs such as `obj:22`, `text:7` or `region:top_right`.

The Pydantic models in `python/src/gvision/protocol/messages.py` are the
source of truth. After changing them, regenerate the schema:

```bash
cd python
python -m gvision.protocol ../schema/messages.schema.json
```

Both sides validate every message they receive, and both test suites check
`schema/examples.json` against the schema, so the two can't drift apart
silently.

## Tests

```bash
cd python && pytest
cd app && npm test
```

## License

AGPL-3.0, see [LICENSE](LICENSE). Model weights are downloaded at first run
under their own licenses and are never committed.
