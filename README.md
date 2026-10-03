# G-VISION

A real-time assistant that helps you perceive what's happening in a game: it
finds and highlights objects, reads on-screen text, remembers recent events and
answers spoken questions with voice and visual highlights.

The full design is in [game-vision-agent-plan.md](game-vision-agent-plan.md).
This repository is at build step 3 of the plan: the screen is captured,
YOLOE finds objects from text prompts, ByteTrack tracks them, and the overlay
glows around the ones you ask for. A demo mode also streams synthetic objects
and a scripted answer so the overlay can be worked on without a GPU.

## Layout

```
python/            Perception, agent and audio processes (package: gvision)
  src/gvision/
    protocol/      Message schema: source of truth for Python <-> Electron
    bridge.py      Local WebSocket server the Electron app connects to
    demo.py        Synthetic objects and answers (stand-in for perception + agent)
    perception/
      capture.py   dxcam screen capture (latest frame only) or a video file
      detector.py  YOLOE with text prompts, mask outlines simplified to polygons
      tracker.py   ByteTrack with tentative / confirmed / lost life cycle
      live.py      capture -> detect -> track -> overlay messages
      exclusion_check.py  proves the overlay is not in captured frames
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

For the live pipeline on Windows, install a CUDA build of PyTorch first (the
default one from PyPI is CPU-only on Windows), then the perception extras:

```bash
cd python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,perception]"
```

YOLOE weights (`yoloe-26s-seg.pt`, ~30 MB) and its text encoder download on
first run into `python/models/` and the current folder; neither is committed.

## Run the live demo (Windows)

Start the app (`npm start` in `app/`), then in another terminal:

```bash
cd python
python -m gvision --live --prompts "person,car,dog" --watch person
```

Every tracked object gets a faint white outline and its label; objects whose
label is in `--watch` get the gold "target" glow once the tracker has seen
them on 3 detector frames in a row. Add `--spotlight` to also dim the screen
around them. `Ctrl+Shift+X` dismisses the current glows. Useful options:
`--rate 15` (detector Hz, default 10), `--model yoloe-26m-seg.pt` (bigger,
slower), `--source clip.mp4` (replay a recording instead of the screen),
`--device cpu`. Games must run in borderless windowed mode.

### Check that the overlay is excluded from capture

The detector must never see the overlay's own glows. With the app running and
a still screen (desktop or a paused game):

```bash
python -m gvision --check-exclusion
```

It captures the screen, makes the overlay dim everything and draw a gold box,
captures again and prints PASS if neither shows up in the second frame (exit
code 1 on FAIL). You should see the screen dim briefly while it runs.

## Run the synthetic demo

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
