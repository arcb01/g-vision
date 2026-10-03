"""Entry point.

    python -m gvision --demo                    synthetic objects and answers
    python -m gvision --live --watch person     capture + YOLOE + ByteTrack
    python -m gvision --live --agent            + push-to-talk, Qwen, Kokoro: "find X",
                                                  scene memory ("what just hit me?")
    python -m gvision --check-exclusion         is the overlay kept out of capture?
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
        tasks.append(_live(args, bridge, stop))
    elif args.check_exclusion:
        from gvision.perception.capture import open_source
        from gvision.perception.exclusion_check import run_exclusion_check

        tasks.append(run_exclusion_check(bridge, stop, open_source(args.source)))
    else:
        log.info("nothing to run: pass --demo, --live or --check-exclusion")
        return 2
    results = await asyncio.gather(*tasks)
    return 1 if False in results else 0


async def _live(args: argparse.Namespace, bridge: Bridge, stop: asyncio.Event) -> None:
    from gvision.perception.capture import open_source
    from gvision.perception.detector import YoloeDetector
    from gvision.perception.live import LivePipeline, LiveSettings

    prompts = _split(args.prompts)
    watch = _split(args.watch if args.watch is not None else ("" if args.agent else "person"))
    prompts += [w for w in watch if w not in prompts]
    source = open_source(args.source)
    detector = await asyncio.to_thread(YoloeDetector, prompts, args.model, device=args.device)
    # With the agent, the spotlight follows every watch for as long as the
    # target is tracked, and only watched objects are drawn.
    settings = LiveSettings(
        rate_hz=args.rate, watch=set(watch), spotlight=args.spotlight or args.agent,
        show_all=args.show_all or not args.agent,
    )
    log.info("live: detecting %s at %.0f Hz, glowing %s", prompts, args.rate, watch or "nothing")
    world = memory = None
    jobs = []
    if args.agent:
        from gvision.world import WorldState

        world = WorldState()
        if not args.no_memory:
            from gvision.memory import EventLog, FrameHistory

            history = FrameHistory(args.history_seconds)
            events = EventLog(world)
            history.listeners.append(lambda f: events.update(f.ts))
            memory = (history, events)
        jobs.append(_agent(args, bridge, world, stop, memory))
    pipeline = LivePipeline(bridge, source, detector, settings, world=world)
    if memory:
        pipeline.frame_listeners.append(memory[0].offer)
    await asyncio.gather(pipeline.run(stop), *jobs)


async def _agent(args: argparse.Namespace, bridge: Bridge, world, stop: asyncio.Event, memory=None) -> None:
    from gvision.agent.agent import Agent
    from gvision.agent.qwen import QwenClient
    from gvision.agent.tools import ToolExecutor
    from gvision.assistant import Assistant

    qwen = QwenClient(args.qwen_url)
    tools = ToolExecutor(world)
    narrator = None
    if memory:
        from gvision.memory import SITUATION_HINT, LookTool, Narrator, SharedQwen
        from gvision.memory.look import HINT, SCHEMA

        history, events = memory
        qwen = SharedQwen(qwen)
        tools.register(SCHEMA, LookTool(qwen, history, events), HINT)
        if args.narrate_every > 0:
            tools.hints.append(SITUATION_HINT)
            narrator = asyncio.create_task(Narrator(qwen, history, events, world, args.narrate_every).run(stop))
    if not await qwen.health():
        log.warning("no llama-server at %s yet; start it (see README) and requests will work", args.qwen_url)
    tts = asr = recorder = None
    if not args.no_tts:
        from gvision.audio.tts import KokoroTTS

        tts = await asyncio.to_thread(KokoroTTS, args.voice)
    if not args.no_mic:
        from gvision.audio.asr import load_asr
        from gvision.audio.mic import Recorder

        asr = await asyncio.to_thread(load_asr, args.asr, args.asr_device, args.whisper_model)
        recorder = Recorder()
    assistant = Assistant(bridge, world, Agent(qwen, world, tools), asr=asr, tts=tts, recorder=recorder)
    try:
        if args.no_mic:
            await assistant.read_stdin(stop)
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
        await qwen.close()


def _split(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="gvision")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="stream synthetic objects and answers")
    mode.add_argument("--live", action="store_true", help="capture, detect and track real objects")
    mode.add_argument("--check-exclusion", action="store_true", help="verify the overlay is not captured")
    live = parser.add_argument_group("live")
    live.add_argument("--source", default="screen", help="'screen', 'screen:<n>' or a video/image file")
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
    agent.add_argument("--ptt-key", default="alt+3", help="push-to-talk hotkey: alt+3, f8, ctrl+shift+space...")
    agent.add_argument("--asr", choices=["whisper", "nemotron"], default="whisper", help="speech-to-text model")
    agent.add_argument("--whisper-model", default="medium", help="faster-whisper size: small, medium...")
    agent.add_argument("--asr-device", default="cuda", help="device for speech-to-text")
    agent.add_argument("--voice", default="af_heart", help="Kokoro voice")
    agent.add_argument("--no-mic", action="store_true", help="type requests in the terminal instead of speaking")
    agent.add_argument("--no-tts", action="store_true", help="show answers without speaking them")
    agent.add_argument("--no-memory", action="store_true", help="no frame history, look tool or narrator")
    agent.add_argument("--history-seconds", type=float, default=60.0, help="how much of the screen to remember")
    agent.add_argument("--narrate-every", type=float, default=25.0,
                       help="seconds between situation summaries by Qwen (0: off)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
