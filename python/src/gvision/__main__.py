"""Entry point: ``python -m gvision [--demo]``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging

from gvision.bridge import DEFAULT_HOST, DEFAULT_PORT, Bridge
from gvision.protocol import Message

log = logging.getLogger("gvision")


async def _amain(args: argparse.Namespace) -> None:
    bridge = Bridge(args.host, args.port)
    stop = asyncio.Event()

    async def log_incoming(msg: Message) -> None:
        log.info("from app: %s", msg.type)

    bridge.on_message(log_incoming)

    tasks = [bridge.run(stop)]
    if args.demo:
        from gvision.demo import run_demo

        tasks.append(run_demo(bridge, stop))
    else:
        log.info("perception is not implemented yet; run with --demo for synthetic data")
    await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(prog="gvision")
    parser.add_argument("--demo", action="store_true", help="stream synthetic objects and answers")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
