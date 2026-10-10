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
    bridge.py      WebSocket server the Electron app connects to (and the gaming PC, on /edge)
    link.py        Two-PC mode, AI server end: frames, questions and answers from/to the gaming PC
    edge.py        Two-PC mode, gaming PC end: screen capture, push-to-talk and playback
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
    knowledge/
      wiki.py      MediaWiki API client: every article in bulk, recent edits, one section rendered
      text.py      Wikitext and HTML to plain text, split into sections
      index.py     One game's wiki in SQLite full-text search (data/wiki/<game>.sqlite)
      library.py   The session game's wiki: downloaded once, caught up in the background
      lookup.py    The lookup tool: game questions answered from the wiki, not the screen
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
  services.js      Starts and stops llama-server and the Python backend (or the edge, on a gaming PC)
  remote.js        Two-PC mode: the AI server's control server and the gaming PC's client
  settings.js      The panel's Settings tab: saved in gvision.config.json, passed as backend flags
  models.js        Vision models for Settings > Vision: download from Hugging Face, run as a second llama-server
  conversation.js  The panel's Log tab: questions, answers and screenshots in logs/conversation/
  sessions.js      The panel's Sessions tab: one per game played, in logs/sessions/
  games.json       Games offered when a session starts, with their wikis
  src/overlay.*    PixiJS overlay: outlines, semantic colors, spotlight dimming
  src/panel.*      Control panel: Home, Sessions, Log and Settings tabs
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
*Send to > Desktop (create shortcut)* to get a desktop icon). On a fresh PC
nothing needs installing by hand, not even Git, and nothing asks for admin
rights: `app/setup.ps1` downloads portable copies of Node.js, Git and Python
3.12 into `runtime/` when they are missing, turns a ZIP download into a git
checkout so it can update itself, creates `python/.venv` with the CUDA build
of PyTorch and downloads Electron. The first launch downloads a few GB; its
messages are also written to `logs/setup.log`. CI runs this first launch on a
clean Windows machine on every pull request. On every launch it pulls
the latest version with `git pull --ff-only` and reinstalls the app's npm
packages or the Python package (`pip install -e ".[dev,perception,voice]"`)
only when `app/package-lock.json` or `python/pyproject.toml` changed; if you
are offline or have local edits that conflict, it says so and starts the
version you have. **Update and restart** in the control panel does the same
without closing anything by hand. From Git Bash, `cd app && npm start` starts
the app without updating. The app starts Qwen in llama-server and
the Python backend (`python -m gvision --live --agent`) on its own, shows each
one's state on the control panel's **Home** tab (with Start/Restart and Stop
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
e.g. `["--live", "--agent", "--prompts", "person,cow"]`.

The control panel has three tabs:

- **Home**: what G-VISION is doing now (listening, thinking, speaking, what was
  heard and the answer), the services, performance and the latest questions.
- **Log**: every question as a conversation, newest first, with the time, a
  screenshot of the screen when it was asked (click it to enlarge), what was
  heard, the answer and the tools used. **How it answered** opens the steps
  behind it, each with its time: speech-to-text, Qwen choosing a tool (with
  the arguments), the detector, OCR or vision call and what it returned,
  Qwen writing the answer (or the fallback when it didn't), and the voice.
  Failed steps are red. It is kept in `logs/conversation/`
  across restarts (the newest 500 questions) and can be searched or cleared.
- **Settings**: push-to-talk key, speech-to-text model, voice, whether to speak
  answers, read on-screen text, scene memory and how often to write situation
  notes, and the dim strength. They are saved in the `settings` section of
  `gvision.config.json`. The dim strength applies at once; the rest when the
  backend restarts (the tab offers a **Restart backend** button). A flag
  written by hand in `python.args` wins over the same setting.

## Two PCs: one plays, the other thinks

With a second PC on the same network, the models can run there so they don't
share the GPU (or its VRAM) with the game. Install G-VISION on both PCs as
above; the gaming PC doesn't need the llama.cpp folder.

The first time G-VISION opens it asks **What is this PC?** (Standalone, Gaming
PC or AI server); for a gaming PC it asks for the AI server's address there
and then. The steps below do the same from Settings later.

1. On the PC that runs the models: **Settings > Network > This PC is: AI
   server**. G-VISION restarts without the overlay, and its **Home** tab shows
   the address to type on the other PC. Windows Firewall asks to allow
   `python.exe` and `electron.exe` the first time: allow them on private
   networks (ports 8765 and 8770). `G-VISION.bat --server` starts it as an AI
   server whatever Settings says, e.g. from a shortcut in `shell:startup`.
2. On the gaming PC: **This PC is: Gaming PC**, then type that address in
   **AI server address** and press **Test connection**, then **Restart
   G-VISION**.

The gaming PC then runs only the overlay, the panel and a small screen and
voice link (`python -m gvision --edge ws://<server>:8765/edge`): it captures
the screen with dxcam (so the overlay stays out of the frames), sends 10 JPEG
frames a second (about 15-30 Mbit/s at 1080p, use a cable rather than Wi-Fi),
records your question while the push-to-talk key is held and plays the
answer. Everything else runs on the AI server: YOLOE, OCR, scene memory,
speech-to-text, Qwen, the vision model, the wikis and Kokoro, which makes the
voice there and sends it back sentence by sentence.

From the gaming PC's panel you still control everything. **Home** shows both
PCs' services (restart and stop work on the AI server's too). **Settings**
tags the voice and vision settings that live on the AI server and saves them
there, and picking a vision model downloads it on the AI server. The session
picker shows which wikis the AI server has indexed. **Update and restart**
updates both PCs, and Home warns if they run different versions. The
push-to-talk key, dim strength and network settings are each PC's own.

The AI server's control server (port 8770) and bridge (port 8765) have no
password: use this on a home network you trust.

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

### How questions are answered

With scene memory on (the default), a question goes to the look tool, and the
vision model answers it in one call. Along with the screen, look gets what
perception already knows: the on-screen text the text watcher reads (exact,
so the answer can quote it; text in the part of the screen the question names
comes first), the objects the tracker confirms, and the narrator's notes. At
the same time, when the request reads like an action ("where is", "find",
"show me", "is there a", "stop"), Qwen3.5-2B checks whether to call `set_watch`
or `clear_watch`; when it calls one, the look is cancelled. Other questions
skip that check: given only the action tools, the 2B called `set_watch` for
"what does the sign say" and "how many bullets" in 7 of 8 replayed questions. This replaced a router that tried
`read_text` or `query_state` first: in 287 logged questions, 84% ended in
look anyway, and half of those had first taken a detour that found nothing.
`--route-first` brings back the old path, for comparing the two.

### Sessions: questions about the game itself

The screen can't answer "what does the mason villager want". Start a session
in the panel's Sessions tab and pick the game you are playing (or name any
other game and its wiki: Fandom, wiki.gg and most game wikis run MediaWiki).
The backend then downloads that wiki once into `data/wiki/<game>.sqlite`,
about 50 articles a request, and keeps it.

During a session, a question that doesn't point at the screen (no "this",
"that", "on screen", "left"...) goes to `lookup` first: SQLite full-text
search finds the sections whose page or heading names what was asked
("Trading > Mason"), the section is fetched rendered from the wiki once when
online (cleaner tables than the bulk copy, then kept), and the vision model
answers from that text alone, without an image. When the wiki names nothing
in the question, look answers as before.

The wiki bar above the session's chat says how fresh the copy is. It catches
up with the wiki's edits when a session starts and the copy is more than a
week old, or when you press Update wiki; a copy older than about 25 days (the
wiki's list of edits doesn't go further back) is downloaded again in full.
Each session keeps its own conversation in `logs/sessions/`, like the Log.

### Ask about text on screen

With `--agent`, a text watcher reads the screen in the background with
RapidOCR (plan 7.1), so you can ask "what does that sign say?", "what's my
quest?", "read the menu" or "what did that message say?". Qwen calls
`read_text` (optionally with a word like "quest" or a place like "top right")
or `recent_text` for text that has already gone, and quotes the answer. The
screen dims and the text blocks it quotes get a faint outline; each one glows
cyan with a light fill only while the voice is reading it, like karaoke, then
goes back to faint when the voice moves on to the next. Nothing is written on
the overlay.

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
