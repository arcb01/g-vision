"""Synthetic perception and agent output, so the overlay can be developed
before capture, detection and Qwen exist.

Moves a few fake objects around the screen and plays a scripted, segmented
answer with spotlight focus every few seconds.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time

from gvision.bridge import Bridge
from gvision.protocol import (
    AnswerFinishedMsg,
    AnswerMsg,
    Box,
    ClearMsg,
    DimMsg,
    FocusMsg,
    HighlightMsg,
    ObjectsMsg,
    Segment,
    SegmentStartedMsg,
    StatusMsg,
    TrackedObject,
)

OBJECTS_HZ = 30
SEGMENT_SECONDS = 1.6
ANSWER_EVERY_SECONDS = 8.0

# (ref, label, size, Lissajous parameters)
_FAKES = [
    ("obj:21", "guard", (0.06, 0.16), (0.30, 0.35, 0.23, 0.31)),
    ("obj:22", "explosive barrel", (0.05, 0.08), (0.20, 0.70, 0.17, 0.13)),
    ("obj:31", "spikes", (0.12, 0.05), (0.65, 0.75, 0.11, 0.07)),
]


def fake_objects(t: float) -> list[TrackedObject]:
    out = []
    for i, (ref, label, (w, h), (cx, cy, fx, fy)) in enumerate(_FAKES):
        ax, ay = 0.12, 0.08
        x = cx + ax * math.sin(2 * math.pi * fx * t + i)
        y = cy + ay * math.sin(2 * math.pi * fy * t + 2 * i)
        vx = ax * 2 * math.pi * fx * math.cos(2 * math.pi * fx * t + i)
        vy = ay * 2 * math.pi * fy * math.cos(2 * math.pi * fy * t + 2 * i)
        out.append(
            TrackedObject(
                ref=ref,
                label=label,
                confidence=0.9,
                box=Box(x=x - w / 2, y=y - h / 2, w=w, h=h),
                velocity=(vx, vy),
                status="confirmed",
            )
        )
    return out


async def _stream_objects(bridge: Bridge, stop: asyncio.Event) -> None:
    t0 = time.monotonic()
    while not stop.is_set():
        now = time.time()
        bridge.send(ObjectsMsg(frame_ts=now, objects=fake_objects(time.monotonic() - t0)))
        await asyncio.sleep(1 / OBJECTS_HZ)


async def _stream_status(bridge: Bridge, stop: asyncio.Event) -> None:
    while not stop.is_set():
        bridge.send(
            StatusMsg(
                perception_fps=OBJECTS_HZ,
                latency_ms={"detector": 0.0},
                components={
                    "capture": "off",
                    "detector": "off",
                    "tracker": "off",
                    "demo": "running",
                },
            )
        )
        await asyncio.sleep(1.0)


async def _play_answers(bridge: Bridge, stop: asyncio.Event) -> None:
    script = [
        Segment(id=0, text="There's an explosive barrel on your left,", refs=["obj:22"]),
        Segment(id=1, text="spikes just ahead,", refs=["obj:31"]),
        Segment(id=2, text="and a guard behind the crates.", refs=["obj:21"]),
    ]
    for n in itertools.count(1):
        await asyncio.sleep(ANSWER_EVERY_SECONDS)
        if stop.is_set():
            return
        if not bridge.connected:
            continue
        answer_id = f"demo-{n}"
        bridge.send(AnswerMsg(answer_id=answer_id, segments=script))
        for seg in script:
            for ref in seg.refs:
                bridge.send(HighlightMsg(ref=ref, color_role="danger"))
        bridge.send(DimMsg(on=True))
        for seg in script:
            bridge.send(SegmentStartedMsg(answer_id=answer_id, segment_id=seg.id))
            bridge.send(FocusMsg(refs=seg.refs, segment_id=seg.id))
            await asyncio.sleep(SEGMENT_SECONDS)
        bridge.send(AnswerFinishedMsg(answer_id=answer_id))
        await asyncio.sleep(2.0)  # hold after speaking (plan 6.1)
        bridge.send(DimMsg(on=False))
        bridge.send(ClearMsg(reason="demo answer finished"))


async def run_demo(bridge: Bridge, stop: asyncio.Event) -> None:
    await asyncio.gather(
        _stream_objects(bridge, stop),
        _stream_status(bridge, stop),
        _play_answers(bridge, stop),
    )
