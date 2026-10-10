"""Entry point.

    python -m gvision --demo                    synthetic objects and answers
    python -m gvision --live --watch person     capture + YOLOE + ByteTrack
    python -m gvision --live --agent            + push-to-talk, Qwen, Kokoro: "find X",
                                                  reading on-screen text (RapidOCR) and
                                                  scene memory ("what just hit me?")
    python -m gvision --check-exclusion         is the overlay kept out of capture?

Two PCs (the gaming PC plays, the AI server thinks):

    python -m gvision --live --agent --source edge --host 0.0.0.0     on the AI server
    python -m gvision --edge ws://<server>:8765/edge                    on the gaming PC
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys

from gvision.bridge import DEFAULT_HOST, DEFAULT_PORT, Bridge
from gvision.protocol import Message

log = logging.getLogger("gvision")


async def _amain(args: argparse.Namespace) -> int:
    if args.edge:
        return await _edge(args)
    bridge = Bridge(args.host, args.port)
    stop = asyncio.Event()

    async def log_incoming(msg: Message) -> None:
        log.info("from app: %s", msg.type)

    bridge.on_message(log_incoming)

    tasks = [bridge.run(stop)]
    if args.demo:
        from gvision.demo import run_demo

        tasks.append(run_demo(bridge, stop))
    elif args.live:
        link = None
        if args.source == "edge":
            from gvision.link import EdgeLink

            link = EdgeLink()
            bridge.edge = link.serve
            log.info("two-PC mode: waiting for the gaming PC on ws://%s:%d/edge", args.host, args.port)
        tasks.append(_live(args, bridge, stop, link))
    elif args.check_exclusion:
        from gvision.perception.capture import open_source
        from gvision.perception.exclusion_check import run_exclusion_check

        tasks.append(run_exclusion_check(bridge, stop, open_source(args.source)))
    else:
        log.info("nothing to run: pass --demo, --live or --check-exclusion")
        return 2
    results = await asyncio.gather(*tasks)
    return 1 if False in results else 0


async def _edge(args: argparse.Namespace) -> int:
    """The gaming PC in two-PC mode: capture, push-to-talk and playback only."""
    from gvision.edge import Edge, Player
    from gvision.perception.capture import open_source

    recorder = player = None
    if not args.no_mic:
        from gvision.audio.mic import Recorder

        recorder = Recorder()
    if not args.no_tts:
        player = Player()
    source = open_source("screen" if args.source == "edge" else args.source)
    edge = Edge(args.edge, source, args.ptt_key, fps=args.stream_fps, quality=args.jpeg_quality,
                recorder=recorder, player=player)
    await edge.run(asyncio.Event())
    return 0


async def _live(args: argparse.Namespace, bridge: Bridge, stop: asyncio.Event, link=None) -> None:
    from gvision.perception.capture import open_source
    from gvision.perception.detector import YoloeDetector
    from gvision.perception.live import LivePipeline, LiveSettings

    prompts = _split(args.prompts)
    watch = _split(args.watch if args.watch is not None else ("" if args.agent else "person"))
    prompts += [w for w in watch if w not in prompts]
    source = link.source if link else open_source(args.source)
    detector = await asyncio.to_thread(YoloeDetector, prompts, args.model, device=args.device)
    # With the agent, the spotlight follows every watch for as long as the
    # target is tracked, and only watched objects are drawn.
    settings = LiveSettings(
        rate_hz=args.rate, watch=set(watch), spotlight=args.spotlight or args.agent,
        show_all=args.show_all or not args.agent,
    )
    log.info("live: detecting %s at %.0f Hz, glowing %s", prompts, args.rate, watch or "nothing")
    world = memory = screen = None
    jobs = []
    if args.agent:
        from gvision.snapshot import LatestFrame

        screen = LatestFrame()
        from gvision.world import WorldState

        world = WorldState()
        text = await _text_watcher(args, bridge, source)
        if text:
            jobs.append(text.run(stop))
        if not args.no_memory:
            from gvision.memory import EventLog, FrameHistory

            history = FrameHistory(args.history_seconds)
            events = EventLog(world)
            history.listeners.append(lambda f: events.update(f.ts))
            memory = (history, events)
        jobs.append(_agent(args, bridge, world, stop, text, memory, screen, link))
    pipeline = LivePipeline(bridge, source, detector, settings, world=world)
    if screen:
        pipeline.frame_listeners.append(screen.offer)
    if memory:
        pipeline.frame_listeners.append(memory[0].offer)
    await asyncio.gather(pipeline.run(stop), *jobs)


async def _text_watcher(args: argparse.Namespace, bridge: Bridge, source):
    """The background text watcher, or None when it is off or RapidOCR is missing."""
    if args.no_text:
        return None
    from gvision.perception.ocr import RapidOcrEngine
    from gvision.perception.text_watcher import TextWatcher

    try:
        engine = await asyncio.to_thread(RapidOcrEngine, args.ocr_side, args.ocr_threads)
    except RuntimeError as e:
        log.warning("not reading on-screen text: %s", e)
        return None
    log.info("text watcher: RapidOCR on the CPU, profiles in %s", args.text_profiles)
    return TextWatcher(bridge, source, engine, profiles_dir=args.text_profiles)


async def _agent(args: argparse.Namespace, bridge: Bridge, world, stop: asyncio.Event, text=None, memory=None,
                 screen=None, link=None) -> None:
    from gvision.agent.agent import Agent
    from gvision.agent.qwen import QwenClient
    from gvision.agent.tools import ToolExecutor
    from gvision.assistant import Assistant

    qwen = QwenClient(args.qwen_url)
    tools = ToolExecutor(world, text=text)
    narrator = look = None
    if memory:
        from gvision.memory import SITUATION_HINT, LookTool, Narrator, SharedQwen
        from gvision.memory.look import HINT, SCHEMA

        history, events = memory
        # Thinking takes longer than the 30 s default allows for.
        vision = QwenClient(args.vision_url, timeout=90.0 if args.vision_reasoning else 30.0) if args.vision_url else None
        qwen = SharedQwen(qwen, vision=vision)
        side = 0 if args.look_size == "640" else 1 << 16 if args.look_size == "full" else int(args.look_size)
        from gvision.memory.evidence import Evidence

        evidence = None if args.route_first else Evidence(world, text)
        look = LookTool(qwen.looking, history, events, screen, screen_side=side,
                        reasoning=args.vision_reasoning and bool(args.vision_url), evidence=evidence)
        tools.register(SCHEMA, look, HINT)
        tools.vision_model = args.vision_model
        if args.narrate_every > 0:
            tools.hints.append(SITUATION_HINT)
            narrator = asyncio.create_task(Narrator(qwen, history, events, world, args.narrate_every).run(stop))
    knowledge = await _knowledge(args, bridge, tools, qwen.looking if memory else qwen)
    if not await qwen.health():
        log.warning("no llama-server at %s yet; start it (see README) and requests will work", args.qwen_url)
    if memory and args.vision_url:
        log.info("vision model: look and situation notes use %s", args.vision_url)
    tts = asr = recorder = None
    if not args.no_tts:
        from gvision.audio.tts import KokoroTTS

        tts = await asyncio.to_thread(KokoroTTS, args.voice, device=args.tts_device)
        if link:  # synthesized here, played on the gaming PC
            from gvision.link import RemoteVoice

            tts = RemoteVoice(tts, link)
    if link and not args.no_mic:
        from gvision.audio.asr import load_asr

        asr = await asyncio.to_thread(load_asr, args.asr, args.asr_device, args.whisper_model)
        recorder = link.recorder
    elif not args.no_mic:
        from gvision.audio.asr import load_asr
        from gvision.audio.mic import Recorder

        asr = await asyncio.to_thread(load_asr, args.asr, args.asr_device, args.whisper_model)
        recorder = Recorder()
    assistant = Assistant(bridge, world, Agent(qwen, world, tools, vision_only=args.vision_only,
                                                    route_first=args.route_first), asr=asr, tts=tts, recorder=recorder,
                          screen=screen, on_listen=look.warm if look else None)
    try:
        if args.no_mic:
            await assistant.read_stdin(stop)
            return
        if link:  # the key is held on the gaming PC
            link.on_ptt_down, link.on_ptt_up = assistant.ptt_down, assistant.ptt_up
            await stop.wait()
            return
        from gvision.audio.ptt import PushToTalk

        ptt = PushToTalk(args.ptt_key, assistant.ptt_down, assistant.ptt_up)
        ptt.start(asyncio.get_running_loop())
        try:
            await stop.wait()
        finally:
            ptt.stop()
    finally:
        if narrator:
            narrator.cancel()
        await knowledge.close()
        await qwen.close()


async def _knowledge(args: argparse.Namespace, bridge: Bridge, tools, qwen):
    """The session game's wiki and the lookup tool that answers from it. The
    panel sends the game when a session starts (and again on reconnect)."""
    from gvision.knowledge import Knowledge
    from gvision.knowledge.lookup import HINT, SCHEMA, LookupTool
    from gvision.protocol import SessionMsg, WikiUpdateMsg

    knowledge = Knowledge(args.wiki_dir, bridge.send)
    tools.register(SCHEMA, LookupTool(qwen, knowledge), HINT)

    async def on_message(msg: Message) -> None:
        if isinstance(msg, SessionMsg):
            log.info("session %s: %s", msg.session_id, msg.game.name if msg.game else "ended")
            await knowledge.set_game(msg.game)
        elif isinstance(msg, WikiUpdateMsg):
            await knowledge.update(full=msg.full)

    bridge.on_message(on_message)
    return knowledge


def _split(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="gvision")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="stream synthetic objects and answers")
    mode.add_argument("--live", action="store_true", help="capture, detect and track real objects")
    mode.add_argument("--check-exclusion", action="store_true", help="verify the overlay is not captured")
    mode.add_argument("--edge", metavar="URL", default=None,
                      help="two-PC mode, gaming PC: stream the screen and voice to the AI server at "
                           "ws://<server>:8765/edge")
    live = parser.add_argument_group("live")
    live.add_argument("--source", default="screen",
                      help="'screen', 'screen:<n>', a video/image file, or 'edge': frames from the gaming PC")
    live.add_argument("--prompts", default="person", help="comma-separated YOLOE text prompts")
    live.add_argument("--watch", default=None, help="comma-separated labels that always glow gold (default: person, none with --agent)")
    live.add_argument("--spotlight", action="store_true", help="also dim the screen around watched objects")
    live.add_argument("--show-all", action="store_true", help="outline every tracked object, not only watched ones")
    live.add_argument("--model", default="yoloe-26s-seg.pt", help="YOLOE weights, downloaded to models/")
    live.add_argument("--device", default=None, help="torch device, e.g. cuda:0 or cpu")
    live.add_argument("--rate", type=float, default=10.0, help="detector rate in Hz (plan: 5-15)")
    agent = parser.add_argument_group("agent (with --live)")
    agent.add_argument("--agent", action="store_true", help="push-to-talk questions answered by Qwen with glow + voice")
    agent.add_argument("--qwen-url", default="http://127.0.0.1:8080", help="llama-server running Qwen3.5-2B")
    agent.add_argument("--vision-url", default=None,
                       help="llama-server with a separate vision model for look and situation notes")
    agent.add_argument("--vision-model", default=None,
                       help="name of the model behind look, shown in the panel's step log")
    agent.add_argument("--vision-only", action="store_true",
                       help="testing: skip routing and send every question to look (needs scene memory)")
    agent.add_argument("--route-first", action="store_true",
                       help="old path: Qwen 2B picks any tool first and look is the fallback (default: look answers "
                            "questions with OCR text and tracked objects attached; Qwen only picks actions)")
    agent.add_argument("--look-size", choices=["640", "1024", "1280", "1600", "1920", "full"], default="1600",
                       help="longest side of the screen look sends for right-now questions; 640 = the history frame")
    agent.add_argument("--vision-reasoning", action="store_true",
                       help="let the vision model think before it answers a look question (slower)")
    agent.add_argument("--ptt-key", default="alt+3", help="push-to-talk hotkey: alt+3, f8, ctrl+shift+space...")
    agent.add_argument("--asr", choices=["whisper", "nemotron"], default="whisper", help="speech-to-text model")
    agent.add_argument("--whisper-model", default="medium", help="faster-whisper size: small, medium...")
    agent.add_argument("--asr-device", default="cuda", help="device for speech-to-text")
    agent.add_argument("--voice", default="af_heart", help="Kokoro voice")
    agent.add_argument("--tts-device", choices=["cuda", "cpu"], default="cuda",
                       help="where Kokoro runs (falls back to the CPU without onnxruntime-gpu)")
    agent.add_argument("--no-mic", action="store_true", help="type requests in the terminal instead of speaking")
    agent.add_argument("--no-tts", action="store_true", help="show answers without speaking them")
    agent.add_argument("--no-text", action="store_true", help="don't read on-screen text")
    agent.add_argument("--text-profiles", default="data/profiles", help="where learned text zones are saved per game")
    agent.add_argument("--ocr-side", type=int, default=1280, help="text detection resolution (long side, px)")
    agent.add_argument("--ocr-threads", type=int, default=4, help="CPU threads for RapidOCR")
    agent.add_argument("--wiki-dir", default="data/wiki", help="where each session game's wiki index is kept")
    agent.add_argument("--no-memory", action="store_true", help="no frame history, look tool or narrator")
    agent.add_argument("--history-seconds", type=float, default=60.0, help="how much of the screen to remember")
    agent.add_argument("--narrate-every", type=float, default=25.0,
                       help="seconds between situation summaries by Qwen (0: off)")
    edge = parser.add_argument_group("edge (with --edge)")
    edge.add_argument("--stream-fps", type=float, default=10.0, help="frames a second sent to the AI server")
    edge.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality of the streamed frames")
    parser.add_argument("--host", default=DEFAULT_HOST, help="0.0.0.0 to take a gaming PC on the network")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
