# Game Vision Agent: Full Plan (v1)

## 1. Goal

A real-time assistant that helps you **perceive** what's happening in a game: it finds and highlights objects, reads on-screen text, remembers recent messages and events, and answers spoken questions with voice and visual highlights, without hurting the game's performance.

## 2. Product decisions

| Decision | Choice |
|---|---|
| Platform | Windows only |
| Games | 2–3 single-player games with very different art styles |
| GPU | NVIDIA only; minimum 12 GB VRAM, recommended 24 GB |
| License | AGPL-3.0 for the whole project |
| Language | English |
| Answers | Overlay panel + mandatory text-to-speech |
| Local VLM / LLM | Qwen3.5-2B |
| Speech-to-text | Nemotron 3.5 ASR streaming 0.6B |
| Text-to-speech | Kokoro-82M |
| OCR | RapidOCR |

## 3. Core principles

1. **Three speeds, one shared memory.** Fast loop every frame, background workers periodically, slow path only when you ask. They communicate only through the world state and never wait for each other.
2. **One request path.** Every request goes to Qwen3.5-2B, which uses tools and looks at images when needed. No hand-written rules.
3. **Record continuously, answer instantly.** Text, frames and the situation are captured before you ask.
4. **Examples over training.** New objects and icons are learned from examples, not retraining.
5. **Nothing hand-configured per game.** Text zones, icons and knowledge are learned or imported automatically.
6. **Grounded answers.** Answers about the screen come from detections, OCR and frames, never from guesses. "I can't find it" is a valid answer.
7. **The game comes first.** Every component is budgeted to protect the game's frame rate.

## 4. Architecture overview

```
FAST LOOP (every frame)
  Capture ─► Detector (YOLOE + exemplars) ─► Tracker ─► Overlay (Electron + PixiJS)
                     ▲                          │
                     └──── watches ── WORLD STATE ◄── tracked objects
                                         ▲
BACKGROUND WORKERS                       │
  Text watcher (auto zones) ─────────────┤
  Frame history (60 s) ──────────────────┤
  Situation narrator (Qwen, 20–30 s) ────┘
                                         ▲
SLOW PATH (on request)                   │ tool calls
  Push-to-talk ─► Nemotron ASR ─► Qwen3.5-2B ─► Panel + Kokoro voice
```

## 5. Layer 1: the fast loop (every frame)

### 5.1 Capture engine

- Windows.Graphics.Capture / Desktop Duplication via `dxcam` or `bettercam`; frames stay on the GPU.
- Two versions of each frame: **small** (~640 px) for the detector, **full resolution** for OCR and Qwen.
- A **"latest frame" slot**, never a queue. Every frame timestamped.
- The overlay is **excluded from capture**, so the detector never sees its own effects.

### 5.2 Detector (5–15 Hz)

Two kinds of targets, handled differently.

**World objects** (3D scene: enemies, chests, doors):

1. **YOLOE text prompts** for common objects.
2. **Visual exemplars** (1–5 example crops) used as YOLOE visual prompts for game-specific objects.

**UI icons** (inventory, toolbars, map markers), by **embedding matching**:

1. Find icon-sized candidate regions (slots, grid cells).
2. Embed each with SigLIP2 or DINOv2.
3. Match against the **icon library**; the best match above a threshold gives the label.

**SAM 3** is used in the **Quality preset (24 GB)** for precise masks and hard cases, and offline for auto-labeling.

**Fine-tuning** a small game-specific detector is a later, optional step for the one or two games demoed most: record gameplay → auto-label with SAM 3 / Qwen / exemplars → review a sample by hand → train.

### 5.3 Exemplar library

Grows automatically:

- **Wiki bootstrap (optional, first launch of a game):** item icons, names and short descriptions from the game's Fandom wiki through the MediaWiki API. Downloaded on the user's machine only, never bundled. Rate-limited and cached. Icons composited onto typical slot backgrounds at several sizes before embedding.
- **VLM fallback:** when Qwen finds an object the detector missed, the crop is saved.
- **One-click teaching** in interaction mode.
- **Self-correction:** confirmed matches add real in-game crops, gradually replacing wiki images.

### 5.4 Tracker (every frame)

ByteTrack or BoT-SORT:

1. **Predict** motion (Kalman filter).
2. **Match** detections to tracks by overlap and appearance.
3. **Life cycle:** *tentative → confirmed* (allowed to glow) *→ lost* (last position kept for edge arrows) *→ removed*.
4. **Re-identify** objects after brief occlusion, using appearance embeddings.

### 5.5 Overlay: Electron + PixiJS

- Transparent, click-through, always-on-top window, excluded from capture with `setContentProtection(true)`.
- Receives **data only** (reference IDs, outline polygons, styles, velocities) over a local WebSocket at ~60 Hz.
- **Rendering layers**, bottom to top: dim layer → glows and outlines → labels and edge arrows → answer panel → badges.
- **Focus manager:** drives dimming, spotlight movement and glow states from speech events and active watches.
- **Position extrapolation** using tracker velocity hides display delay.
- **Hysteresis:** only confirmed objects glow; fade in and out gently.
- Requires **borderless windowed** mode in games.

## 6. Visual effects

### 6.1 Spotlight dimming

- A **semi-transparent dark layer** over the screen with **feathered cut-outs** around highlighted elements, following their outlines and moving with them every frame.
- Implementation: dark rectangle in a render texture, highlight shapes erased with an erase blend mode after blurring for soft edges.
- **Fades** in and out over ~200 ms.
- The overlay can only draw *on top* of the game, so it **darkens** but never blurs or desaturates the background (that would require re-drawing a lagging captured copy).
- **Never in the way:**
  - temporary: lasts while an answer is spoken plus a short hold (~2 s);
  - never fully dark: default ~40–50%, adjustable;
  - lifts immediately on a new push-to-talk or the dismiss hotkey;
  - lighter, or skipped, during intense moments such as combat (configurable).

### 6.2 Highlight switching while speaking

- Qwen answers in **segments with references** (constrained JSON):

```json
{"segments": [
  {"text": "There's an explosive barrel on your left,", "refs": ["obj:22"]},
  {"text": "spikes just ahead,", "refs": ["obj:31"]},
  {"text": "and a guard behind the crates.", "refs": ["obj:21"]}
]}
```

- References can be **objects** (`obj:`), **text blocks** (`text:`), or **regions** (`region:`).
- Kokoro speaks segment by segment and emits `segment_started` events.
- On each event, the overlay:
  - gives the **focused** element the full spotlight and strongest glow;
  - keeps **already-mentioned** elements subtly outlined;
  - **animates** the spotlight to the next element (~200 ms, eased).
- The panel highlights the segment being spoken, karaoke-style.
- **Edge cases:** off-screen elements get a pulsing edge arrow; disappearing elements fade at their last position; a new push-to-talk cancels everything; after the answer, mentioned elements stay softly outlined for a few seconds.
- **Fallback:** if Qwen's references are missing or invalid, highlight everything at once without switching.

### 6.3 High contrast and glow

- **Layered outline, visible on any background:** thin dark outer edge + crisp bright core line (2–3 px) + soft colored bloom.
- **Adaptive color:** background brightness and hue around each object are sampled from the small frame, and the glow adjusts for contrast.
- **Semantic colors:**

| Meaning | Color | Extra signal |
|---|---|---|
| Target you asked for | Gold | — |
| Danger | Red | Warning icon |
| Information / text | Cyan | — |
| Uncertain | Any | Dashed outline |

- Color is never the only signal; a **colorblind-safe palette** is available.
- **Motion and accessibility:** slow, low-intensity pulse (~1 per second), never flashing (well below ~3 flashes per second); a **reduce-motion** option disables pulses and transitions.
- Implementation: PixiJS glow filter plus a custom shader for the layered outline. Mask outlines are simplified into polygons in Python before sending.

## 7. Layer 2: background workers

### 7.1 Text watcher with automatic zone discovery (2–4 Hz)

1. **Detect text everywhere cheaply:** RapidOCR's detection stage only, on the full frame at 1–2 Hz, skipping unchanged screen tiles.
2. **Track text boxes** like objects, each with an ID.
3. **Recognize only new or changed boxes** (RapidOCR recognition on full-resolution crops).
4. **Separate UI text from world text:** text fixed on screen while the camera moves is UI; text moving with the scene belongs to the world.
5. **Group lines into blocks** by proximity, size and alignment.
6. **Learn static text:** blocks that never change ("HP", "AMMO") are flagged and deprioritized.
7. **Learn zones from history:** areas where text often appears and changes (chat, notifications, subtitles) become **learned zones**, watched with priority.
8. **Save a generated game profile** so later sessions start with learned zones. The user can correct it in the GUI, but never has to write it.
9. **Store** each block once: zone, position, text, confidence, size, static flag, appear/disappear times, crop.

### 7.2 Frame history

- ~60 s at ~2 fps, compressed (~25 MB), for "what was that?" questions.
- Qwen3.5 accepts multiple frames or short video, so it can look across several frames at once.

### 7.3 Situation narrator (Qwen3.5-2B, every ~20–30 s during gameplay)

- **Input:** current frame, new text blocks, recent events, previous summary.
- **Output:** current situation, player state (as far as visible), current objective, and the condensed session summary.
- **Pauses** during menus, loading and cutscenes, and when the scene hasn't changed.
- **User requests always pre-empt it.**
- Its readings (e.g. health) are approximate context, not precise values.

## 8. The world state

| Section | Contents | Written by | Persisted |
|---|---|---|---|
| **Session context** | Game, screen size, mode (gameplay/menu/cutscene/loading) | Context worker | No |
| **Game profile** | Learned text zones, static text, user corrections | Text watcher, GUI | Per game |
| **Tracked objects** | Reference ID, label, confidence, box, mask outline, velocity, status, timestamps, embedding | Tracker | No |
| **Active watches** | Target, style, color role, expiry, status, search start | Qwen + fallback logic | No |
| **Situation** | Current situation, player state, objective, session summary | Narrator | No |
| **Label knowledge cache** | Per-type facts | Qwen, wiki import, user | Per game |
| **Exemplar library** | World-object exemplars and icon library with embeddings | Wiki import, fallback, user | Per game |
| **Text log** | Blocks with reference ID, zone, position, text, size, static flag, times, crop | Text watcher (+ Qwen corrections) | No |
| **Frame history** | ~120 compressed frames | Frame history worker | No |
| **Event stream** | Append-only log of what happened | All components | No |

**Implementation rules:** one writer per section; versioned snapshots for the slow path; large data (frames, masks, crops) stored separately and referenced by ID; normalized 0–1 coordinates; reference IDs (`obj:`, `text:`, `region:`) usable in answers.

**Snapshot sent to Qwen:** compact JSON with situation, objective, objects (with reference IDs and positions in words), watches and recent text. A few hundred tokens, with counts precomputed.

## 9. Layer 3: the slow path (only when you ask)

### 9.1 Input and speech-to-text

- **Push-to-talk**, which also prevents the agent from hearing its own voice.
- **Nemotron ASR streaming:** transcription is nearly ready when the key is released; chunk sizes from 80 ms to 1.12 s trade latency for accuracy. The English-only sibling, `nemotron-speech-streaming-en-0.6b`, is the card's recommendation for English-only use.

### 9.2 Qwen3.5-2B: LLM and vision in one model

Handles tool calls, image questions, OCR fallback, locating missed objects, and the narrator.

**Configuration:**

- non-thinking mode (default; avoids the 2B model's thinking loops);
- context capped at 8–16k tokens (the 262k default wastes memory);
- constrained JSON output for tool calls and segmented answers;
- quantized GGUF through a llama.cpp server on Windows;
- crops instead of full frames whenever possible;
- optional larger Qwen3.5 size in the Quality preset on 24 GB GPUs.

**Tools (kept short for the small model):**

| Tool | What it does |
|---|---|
| `set_watch(targets, style, color_role)` / `clear_watch(target)` | Controls highlighting |
| `query_state()` | Current objects, counts, positions |
| `learn_label(label, facts)` | Fills the knowledge cache |
| `read_region(region)` | Reads text with block selection |
| `get_recent_text(since, zone)` | Reads the text log |
| `look(question, region or frames)` | Qwen examines images directly |

**Answer format:** short, speakable **segments with references** (see 6.2).

### 9.3 Answer: panel + Kokoro (mandatory)

- Every answer is shown in the panel **and** spoken.
- Kokoro runs on the **CPU** (ONNX Runtime), using no GPU memory.
- Synthesized per segment; speech starts as soon as the first segment is ready.
- Emits `segment_started` and `answer_finished` events for visual sync.
- Volume, voice and speed adjustable; "stop speaking" hotkey; optional ducking of game audio.

### 9.4 Automatic fallbacks

- Watched target not found within ~2 s → Qwen looks at the frame and locates it → saved as an exemplar.
- Low-confidence OCR → Qwen re-reads the saved crop.
- Nothing matching in the text log → Qwen searches the frame history.

### 9.5 Reading text: block selection

1. **Group** lines into blocks.
2. **Describe** each: position in words, size, age, static flag.
3. **Rank:** location requested, recency, salience, static text last, intent ("the sign" vs "the chat").
4. **Answer:** read the clear winner and mention others, or show **numbered badges** and ask when ambiguous.

### 9.6 Optional later: fast intent classifier

Only if frequent commands feel slow: a tiny classifier for 4–5 fixed intents (find, clear, read region, read recent messages), with example phrases in a data file. Everything else goes to Qwen.

## 10. Request walkthroughs

| Request | Path | Time |
|---|---|---|
| "Where is the TV?" | ASR → Qwen → `set_watch` → detection → gold glow + spotlight + spoken confirmation | ~1–2 s |
| "Where is the TV?" (unrecognized) | + ~2 s fallback → Qwen locates → exemplar → tracked | ~3–5 s first time |
| "Where's my wood?" | Qwen → `set_watch("wood")` → icon library match → glow | ~1–2 s |
| "What can hurt me here?" | Qwen + snapshot + cache → segmented answer → dim → spotlight moves barrel → spikes → guard → red outlines with warning icons → fade | ~1–3 s |
| "What does it say top right?" | Qwen → `read_region` → RapidOCR → block ranking → cyan highlight + panel + voice | ~1–2 s |
| "What did the recent messages say?" | Qwen → `get_recent_text` → spoken summary, each message highlighted if still visible | ~1–2 s |
| "What was that?" | Qwen → `look(frames: last 10 s)` | ~2–5 s |
| "What's going on?" | Qwen answers from situation and session summary | ~1–2 s |

Times include speech-to-text; voice starts shortly after the first segment is ready.

## 11. Performance

### 11.1 VRAM budget (approximate)

| Component | 12 GB GPU | 24 GB GPU |
|---|---|---|
| Game | 4–8 GB | 4–12 GB |
| YOLOE + tracker embeddings | ~1 GB | ~1 GB |
| RapidOCR | ~0.3 GB (or CPU) | ~0.3 GB |
| Nemotron ASR | ~1–1.5 GB | ~1–1.5 GB |
| Qwen3.5-2B (quantized, capped context) | ~2–3 GB | ~2–3 GB, or a larger Qwen3.5 |
| SAM 3 | Off | Optional (Quality) |
| Kokoro TTS | CPU | CPU |

On 12 GB, demanding games need the Performance preset. Measure real numbers on the target games early.

### 11.2 Rates

| Component | Rate |
|---|---|
| Capture, tracker, overlay data, visual effects | Every frame |
| Detector | 5–15 Hz |
| Text detection / recognition | 1–2 Hz / on change |
| Frame history | ~2 fps |
| Narrator | Every 20–30 s, pre-empted by requests |
| Qwen requests | On demand |

### 11.3 Graceful degradation

If game FPS drops: lower the detector rate first, then the narrator and text watcher, then glow quality. Never the tracker, overlay positions or capture.

## 12. Processes, runtimes and communication

| Process | Contains | Runtime |
|---|---|---|
| **Perception** (Python) | Capture, detector, tracker, text watcher, frame history, world state | YOLOE exported to ONNX Runtime / TensorRT; RapidOCR on ONNX Runtime |
| **Model server** | Qwen3.5-2B | llama.cpp server (GGUF), OpenAI-compatible API |
| **Agent** (Python) | Snapshot builder, Qwen client, tools, narrator scheduling, fallbacks | Python 3.11+ |
| **Audio** (Python) | Nemotron ASR (streaming), Kokoro TTS | PyTorch/Transformers for ASR (GPU); ONNX Runtime for Kokoro (CPU) |
| **App** (Electron) | Overlay window + control panel | Electron + React + PixiJS |

- Frames move between Python processes through **shared memory**.
- Python ↔ Electron use a **local WebSocket** with a defined schema (MessagePack or JSON):

| Message | Purpose |
|---|---|
| `objects(...)` | Per-frame object outlines, positions, velocities |
| `highlight(ref, style, color_role)` | Set an element's glow style and meaning |
| `focus(refs, segment_id)` | Move the spotlight |
| `dim(on/off, strength)` | Control the dim layer |
| `answer(segments)` | Show the answer in the panel |
| `segment_started` / `answer_finished` | Sync visuals with speech |
| `badges(blocks)` | Numbered choices |
| `clear()` | Cancel on interrupt or dismiss |
| `status(...)` | FPS, VRAM, latency, active components for the dashboard |
| `config_changed(...)` | Settings updates |

- **GPU priority:** game → perception → audio → Qwen.
- **Electron launches and supervises** the Python processes and the model server, restarting them on crash.

## 13. GUI strategy

### 13.1 Structure

- **One Electron app, two windows:** the transparent overlay and a normal control panel (React), sharing code and the WebSocket connection.
- **System tray icon:** start/pause, open panel, current preset, quit.

### 13.2 First-run wizard

1. **Hardware check:** GPU and VRAM → suggested preset; warning below 12 GB.
2. **Model downloads** with progress bars (weights never bundled).
3. **Microphone test** and push-to-talk key selection.
4. **Voice selection** for Kokoro, with preview.
5. **Notices:** single-player only (anti-cheat), privacy defaults, borderless windowed mode.

### 13.3 Control panel sections

| Section | Contents |
|---|---|
| **Dashboard** | Running/paused, detected game, game FPS, VRAM, response times, active components |
| **Presets & settings** | Performance / Balanced / Quality / Custom + advanced toggles with estimated VRAM, speed, and what stops working when off |
| **Visual effects** | Dim strength and hold; dimming on/off; lighter dimming in combat; highlight switching; glow style; adaptive contrast; color palette (default / colorblind-safe); pulse; reduce motion |
| **Game profiles** | Learned text zones (viewable on a screenshot, correctable), exemplar and icon library browser, wiki import, knowledge cache viewer |
| **History** | Question/answer transcript, text log browser, replay of spoken answers |
| **Overlay appearance** | Panel position and size, label visibility |
| **Voice & input** | Push-to-talk and hotkeys, ASR latency, TTS voice, speed and volume |
| **Privacy** | Exclude chat from logs, clear history, clear game data |
| **Developer** (hidden) | Debug overlay (raw detections, track IDs, text boxes, reference IDs), logs, test bench runner |

### 13.4 In-game interaction

- **Push-to-talk** for all questions.
- **Interaction mode hotkey:** temporarily makes the overlay clickable, to pick a numbered badge, teach an exemplar by drawing a box, or dismiss the panel.
- **Quick menu hotkey:** switch presets or pause without alt-tabbing.
- **Dismiss hotkey:** stops speech and clears dimming and highlights.

### 13.5 Presets and toggles

| | Performance | Balanced | Quality |
|---|---|---|---|
| Detector rate | 5 Hz | 10 Hz | 15 Hz |
| SAM 3 masks | Off | Off | On (24 GB) |
| Narrator | Every 60 s | Every 20–30 s | Every 20 s |
| Frame history | 15 s | 60 s | 60 s+ |
| Glow | Simple outline | Layered + light glow | Full bloom |
| Adaptive contrast | Off | On | On |
| Dimming cut-outs | Box-shaped | Mask-shaped, feathered | Mask-shaped, feathered |
| Spotlight transitions | Instant | Animated | Animated |
| Qwen size | 2B | 2B | Larger Qwen3.5 option |

- **Auto mode** suggests a preset at first run and steps down if game FPS drops.
- **Dependencies:** dependent settings grey out automatically, and the agent says when a feature is off instead of failing silently.
- **Not toggleable:** tracker, capture exclusion, latest-frame capture, privacy defaults, text-to-speech (only voice, speed and volume adjustable).

### 13.6 Settings architecture

One **config file with a schema**, read by every component. The GUI only edits this config; components reload changes live where possible, and the GUI says when a restart is needed. Presets are named config bundles.

## 14. Tech stack and licenses

| Role | Choice | License |
|---|---|---|
| Capture | `dxcam` / `bettercam` | Check each library |
| Detection | YOLOE-26 | AGPL-3.0 |
| Segmentation (Quality) | SAM 3 / 3.1 | SAM License (download separately, credit Meta) |
| Tracker | ByteTrack / BoT-SORT | Permissive |
| Embeddings | SigLIP2 or DINOv2 | Apache 2.0 |
| OCR | RapidOCR | Apache 2.0 |
| LLM + vision | Qwen3.5-2B | Apache 2.0 |
| Model server | llama.cpp | MIT |
| Speech-to-text | Nemotron 3.5 ASR streaming 0.6B | OpenMDW-1.1 |
| Text-to-speech | Kokoro-82M (ONNX) | Apache 2.0 |
| App + overlay | Electron, React, PixiJS | MIT |

Permissive components can be used inside an AGPL-3.0 project. Model weights are downloaded at first run under their own licenses, not included in the repository. Wiki content and game datasets are never redistributed. (Not legal advice; review each license before publishing.)

## 15. Testing and data

- **Test bench:** replay recorded clips (your own, plus selected sessions from `markov-ai/gaming-500-hours`, kept private) through the pipeline, reporting results per preset.
- **Success metrics, defined up front:**
  - glow appears within ~2 s of a request;
  - under 1 flicker per minute;
  - spotlight alignment error within a few pixels on moving objects;
  - detection and text-reading accuracy above target thresholds on the test clips;
  - game FPS loss below a set limit per preset.
- **Training data:** frames auto-labeled by SAM 3 / Qwen / exemplars, spot-checked by hand.
- **Always confirm live** at 60 fps; recorded 30 fps footage behaves differently.

## 16. Main risks and challenges

| Risk | Mitigation |
|---|---|
| **Nemotron ASR officially supports Linux only** | Verify on Windows first (Transformers path); keep Whisper as fallback |
| **Tool-call reliability with a 2B model** | Constrained JSON, short tool list, larger Qwen option |
| **GPU sharing with the game**, especially on 12 GB | Presets, graceful degradation, CPU for TTS |
| **Stylized graphics, icons and fonts** | Exemplars, icon library, Qwen fallbacks |
| **Zone discovery needs a few minutes of play** | Saved game profiles; Qwen fallback meanwhile |
| **Flicker and overlay delay** | Tracker life cycles, hysteresis, extrapolation |
| **Spotlight misalignment on fast objects** | Extrapolation, feathered edges |
| **Dimming obstructing gameplay** | Temporary, quick release, lighter in combat |
| **Speech–visual sync errors** | Fallback to highlighting everything at once |
| **Narrator accuracy and up to ~30 s lag** | Treated as context only |
| **Exclusive fullscreen games** | Borderless windowed mode required |
| **Anti-cheat and terms of service** | Single-player only; warning in the wizard |
| **Privacy (chat, usernames)** | Local only; chat exclusion; blur in videos |

## 17. Deliberately excluded from v1

Hand-written rules router, anomaly scan, HUD reader, per-game config files, Jev, Turnstone, NitroGen, V-JEPA 2.

## 18. Future versions

**Game knowledge base (grounded game knowledge).** A per-game searchable library so the agent can answer "how does the game work" questions from real sources instead of model memory:

- **Sources, by priority:** text the game shows (tooltips, tutorials, quests, dialogue, already collected by the text watcher), full wiki page text, official sources and patch notes, optional live web search for brand-new games.
- **New pieces:** knowledge-builder worker, local embedding model plus vector index (keyword + embedding search), `search_knowledge` and optional `web_search` tools, knowledge browser in the GUI.
- **Rules:** game-knowledge answers must cite a retrieved source; otherwise "I don't know yet."

Other candidates: a fast intent classifier, a fine-tuned detector per demo game, multi-language support.

## 19. Build order

1. **Verify the risky pieces** on Windows next to a running game: Nemotron ASR, and Qwen3.5-2B via llama.cpp (tool calling, image input, constrained JSON).
2. **Electron app skeleton:** overlay with a test glow + control panel shell; capture exclusion verified.
3. **Capture + YOLOE + tracker:** real objects tracked.
4. **Layered high-contrast glow** with semantic colors and adaptive contrast.
5. **World state + push-to-talk + Qwen with `set_watch` + Kokoro voice:** "find X" works end to end.
6. **Segmented answers + spotlight dimming + synchronized highlight switching.**
7. **`read_region` with block selection.**
8. **Text watcher with automatic zone discovery + text log.**
9. **Exemplar library + icon matching**, then **wiki import.**
10. **Situation narrator.**
11. **Qwen fallbacks + teaching in interaction mode.**
12. **Frame history.**
13. **First-run wizard, presets and full control panel** (including Visual effects settings).
14. **Test bench + optional fine-tuning** for the main demo game.

Each step produces something that works and is worth showing in a video.

## References

- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Nemotron 3.5 ASR streaming 0.6B model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Kokoro-82M (fork with model card)](https://github.com/zboyles/Kokoro-82M)
- [YOLOE tutorial (LearnOpenCV)](https://learnopencv.com/yoloe-tutorial-real-time-open-vocabulary-detection/)
- [SAM 3 (Roboflow)](https://blog.roboflow.com/what-is-sam3/)
- [Ultralytics AGPL guidance](https://docs.ultralytics.com/help/contributing)
- [markov-ai/gaming-500-hours](https://huggingface.co/datasets/markov-ai/gaming-500-hours/blob/main/README.md)
