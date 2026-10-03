"""Entry point.

    python -m gvision --demo                    synthetic objects and answers
    python -m gvision --live --watch person     capture + YOLOE + ByteTrack
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
    watch = _split(args.watch)
    prompts += [w for w in watch if w not in prompts]
    source = open_source(args.source)
    detector = await asyncio.to_thread(YoloeDetector, prompts, args.model, device=args.device)
    settings = LiveSettings(rate_hz=args.rate, watch=set(watch), spotlight=args.spotlight)
    log.info("live: detecting %s at %.0f Hz, glowing %s", prompts, args.rate, watch or "nothing")
    await LivePipeline(bridge, source, detector, settings).run(stop)


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
    live.add_argument("--watch", default="person", help="comma-separated labels that get the gold glow")
    live.add_argument("--spotlight", action="store_true", help="also dim the screen around watched objects")
    live.add_argument("--model", default="yoloe-26s-seg.pt", help="YOLOE weights, downloaded to models/")
    live.add_argument("--device", default=None, help="torch device, e.g. cuda:0 or cpu")
    live.add_argument("--rate", type=float, default=10.0, help="detector rate in Hz (plan: 5-15)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
