# G-VISION

A real-time assistant that helps you perceive what's happening in a game: it
finds and highlights objects, reads on-screen text, remembers recent events and
answers spoken questions with voice and visual highlights.

The full design is in [game-vision-agent-plan.md](game-vision-agent-plan.md).
This repository is at build step 5 of the plan, "find X": hold a key, ask
"where's the cow?", and Qwen3.5-2B turns the question into a `set_watch`
call, YOLOE starts looking for cows, the overlay glows around the one it
tracks and Kokoro says where it is. A demo mode also streams synthetic objects
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
    world.py       World state: tracked objects, active watches, situation; snapshot for Qwen
    memory/
      history.py   Last 60 s of the screen at 2 fps, downscaled JPEGs (~5 MB)
      events.py    What the tracker saw appear and leave
      narrator.py  Qwen's running notes on the situation, every ~25 s
      look.py      The look tool: Qwen looks at recent frames ("what just hit me?")
      shared.py    Narrator and player share llama-server, the player first
    assistant.py   Push-to-talk -> speech-to-text -> agent -> panel, spotlight, voice
    agent/
      qwen.py      Client for Qwen3.5-2B in llama-server (OpenAI-compatible API)
      tools.py     set_watch, clear_watch, query_state (with routing hints)
      agent.py     One request: snapshot + tools -> tool results -> short answer
    audio/
      ptt.py       Global push-to-talk key (pynput)
      mic.py       Microphone recording while the key is held
      asr.py       faster-whisper (default) or Nemotron speech-to-text
      tts.py       Kokoro-82M on the CPU (ONNX Runtime)
app/               Electron app: transparent overlay + control panel
  main.js          Owns the bridge connection, validates and forwards messages
  services.js      Starts and stops llama-server and the Python backend
  src/overlay.*    PixiJS overlay: outlines, semantic colors, spotlight dimming
  src/panel.*      Control panel shell: dashboard, answer, controls
G-VISION.bat       Double-click launcher (Windows)
gvision.config.example.json  Launcher paths; copy to gvision.config.json
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

## Start everything with one click (Windows)

Double-click `G-VISION.bat` in the repository folder (right-click it and pick
*Send to > Desktop (create shortcut)* to get a desktop icon). From Git Bash,
`cd app && npm start` does the same. The app starts Qwen in llama-server and
the Python backend (`python -m gvision --live --agent`) on its own, shows each
one's state in the control panel under **Startup** (with Start/Restart and Stop
buttons and the last error if one fails), and stops both when you close the
panel. Restart re-reads `gvision.config.json`, so a fixed path applies without
relaunching. Their output goes to `logs/qwen.log` and `logs/backend.log`
(**Open logs** in the panel). If a server is already running on its port (one
started by hand, or left over from an earlier run), the app uses it and leaves
it running on exit; Stop or Restart in the panel stop whatever holds the port.

Paths come from `gvision.config.json` at the repository root. Without one, the
defaults expect the llama.cpp folder at `~/Desktop/coding projects/G-vision-lab`
(with `llama/llama-server.exe` and `models/Qwen3.5-2B-Q4_K_M.gguf` plus
`models/mmproj-F16.gguf`) and use the first Python virtual environment found
among the active one (`$VIRTUAL_ENV`), `python/.venv` and `.venv` at the
repository root. To change them, copy `gvision.config.example.json` to
`gvision.config.json` and edit it; `python.exe` points at a specific Python (relative to `python/`), `python.args` sets the backend's options,
e.g. `["--live", "--agent", "--prompts", "person,cow", "--whisper-model", "small"]`.

Other ways to start: `G-VISION.bat --demo` (or `npm start -- --demo`) runs the
synthetic demo without Qwen, and `npm start -- --no-services` only opens the
app so you can start the processes by hand as described below.

## Run the live demo (Windows)

Start the app (`npm start` in `app/`), then in another terminal:

```bash
cd python
python -m gvision --live --prompts "person,car,dog" --watch person
```

Every tracked object gets a faint white contour; objects whose
label is in `--watch` get the gold "target" glow once the tracker has seen
them on 3 detector frames in a row. Add `--spotlight` to also dim the screen
around them. `Ctrl+Shift+X` dismisses the current glows. Useful options:
`--rate 15` (detector Hz, default 10), `--model yoloe-26m-seg.pt` (bigger,
slower), `--source clip.mp4` (replay a recording instead of the screen),
`--device cpu`. Games must run in borderless windowed mode.

## Find X with your voice (Windows)

Three things run side by side: the app, Qwen in llama.cpp's server, and the
Python process. `G-VISION.bat` starts all three (see above); the steps below
are the manual way.

1. Install the voice extras (after the CUDA torch and perception extras above):

   ```bash
   cd python
   pip install -e ".[dev,perception,voice]"
   ```

   This pulls faster-whisper, Kokoro (ONNX), sounddevice, pynput and the CUDA
   12 cuBLAS/cuDNN wheels that faster-whisper needs on Windows.

2. Start Qwen3.5-2B (any llama.cpp build with `--jinja`; tested with b11379):

   ```bash
   llama-server -m Qwen3.5-2B-Q4_K_M.gguf --mmproj mmproj-F16.gguf -ngl 99 -c 8192 --jinja --reasoning off --port 8080
   ```

3. Start the app (`npm start` in `app/`), then:

   ```bash
   python -m gvision --live --agent --prompts person
   ```

Hold **Alt+3**, ask "where's the cow?" and let go. While you talk, a cyan
voice wave at the bottom of the screen follows your voice; bouncing dots show
Qwen is thinking. Qwen calls `set_watch(["cow"])`, "cow" is added to YOLOE's
prompts, and once the tracker confirms one, its contour glows gold with a
see-through fill and the screen dims around it. Kokoro speaks the answer ("The
cow is in the center.") with a gold wave. No text is drawn on the overlay: what
was heard and the answer appear in the control panel. The glow and spotlight
stay on for as long as the cow is tracked; only what you asked for is
outlined (`--show-all` outlines every tracked object, for debugging). If
nothing turns up within 1.5 s the answer says so and the watch stays on, so
the glow appears as soon as one comes into view. Each new push-to-talk
replaces the previous watch; `Ctrl+Shift+X` clears everything and stops
speech. Ask "how many people are there?" for a count, or "stop highlighting"
to clear.

Useful options: `--ptt-key f8` (any key, or a combo like `ctrl+shift+space`), `--whisper-model small` (faster, less
VRAM), `--asr nemotron` (needs `transformers`; much worse on accented English
in our tests), `--voice am_michael`, `--no-mic` (type requests in the
terminal), `--no-tts`, `--qwen-url`. The first run downloads the Whisper
weights to the Hugging Face cache and Kokoro (~120 MB) into
`python/models/kokoro/`.

### Ask about text on screen

With `--agent`, a text watcher reads the screen in the background with
RapidOCR (plan 7.1), so you can ask "what does that sign say?", "what's my
quest?", "read the menu" or "what did that message say?". Qwen calls
`read_text` (optionally with a word like "quest" or a place like "top right")
or `recent_text` for text that has already gone, quotes the answer, and the
text block it quotes gets a cyan contour with a light fill while the rest of
the screen dims. Nothing is written on the overlay.

It runs on the CPU (4 threads, no GPU), about twice a second: only screen
tiles that changed are searched for text, whole-screen detection happens at
most once a second, and only new or changed lines are read. Areas where text
keeps changing (chat, notifications, subtitles) become learned zones that are
checked on every pass, and text that never changes ("HP") is ranked last.
What it learns is saved per game in `data/profiles/<game exe>.json`.
RapidOCR and its models come with the `perception` extra (or `.[ocr]` on its
own); without it the backend logs a warning and runs without reading text.
Options: `--no-text`, `--ocr-threads 2`, `--ocr-side 960` (faster detection,
misses smaller text), `--text-profiles DIR`.

### Scene memory: "what just hit me?"

With `--agent`, the backend also remembers the last minute of the screen
(640 px wide JPEGs at 2 fps, about 5 MB of RAM and nothing on the GPU) and
logs what the tracker saw appear and leave. Ask "what just hit me?", "what
was that?" or "what happened?" and Qwen calls `look`: it gets up to four
frames from the last seconds (the newest plus the ones where the screen
changed most, such as a hit flash) and answers from them in one go.

Every 25 s a narrator sends Qwen the newest frame, the recent events and its
previous notes, and gets back the situation, the player's visible state, the
objective and a short running summary of the session. These notes go into
every request, so "what's going on?" is answered without an extra look. The
narrator skips a turn when the screen hasn't changed, and a question always
goes first: a narrator call in flight is cancelled, and it waits until 5 s
after the last answer. Its readings are rough context, not exact values.

Both need llama-server started with the `--mmproj` vision file (as above).
Options: `--narrate-every 40` (or `0` to turn the narrator off),
`--history-seconds 30`, `--no-memory` (none of it).

YOLOE's text prompts work on realistic graphics. On blocky or stylized games
(Minecraft) they find little or nothing yet; visual exemplars and Qwen's
"locate it in the frame" fallback (plan 5.3 and 9.4) are what fix that.

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
`segment_started`, `answer_finished`, `badges`, `status`, `voice` (push-to-talk state and voice loudness), `clear`
and `config_changed`. Every message carries `v` (protocol version), `type` and
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
