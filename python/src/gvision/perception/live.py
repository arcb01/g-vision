"""First live pipeline: capture -> YOLOE -> ByteTrack -> overlay (plan step 3).

Streams tracked objects to the app and gives every confirmed track whose
label is in the watch list a gold "target" glow. With a ``WorldState``,
the watch list also follows the agent's ``set_watch`` calls, and new watch
targets are added to YOLOE's text prompts on the fly. With ``spotlight`` on,
the screen also dims around the watched objects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Protocol

import numpy as np

from gvision.bridge import Bridge
from gvision.perception.capture import Frame, FrameSource
from gvision.perception.detector import Detection
from gvision.perception.tracker import ByteTracker, Track
from gvision.protocol import (
    Box,
    ColorRole,
    ClearMsg,
    ConfigChangedMsg,
    DimMsg,
    FocusMsg,
    HighlightMsg,
    Message,
    ObjectsMsg,
    StatusMsg,
    TrackedObject,
)
from gvision.world import WorldState

log = logging.getLogger(__name__)


class Detector(Protocol):
    """Anything with YoloeDetector's ``detect``; lets tests fake it."""

    classes: list[str]

    def detect(self, image: np.ndarray) -> list[Detection]: ...

    def set_classes(self, classes: list[str]) -> None: ...


@dataclass
class LiveSettings:
    rate_hz: float = 10.0
    watch: set[str] = field(default_factory=set)
    spotlight: bool = False
    dim_strength: float = DimMsg.model_fields["strength"].default
    show_all: bool = True
    """Send every track to the overlay (faint debug outlines), not only watched ones."""


def to_message(tr: Track) -> TrackedObject:
    x0, y0, x1, y1 = (float(v) for v in tr.box)
    return TrackedObject(
        ref=tr.ref,
        label=tr.label,
        confidence=min(max(tr.confidence, 0.0), 1.0),
        box=Box(x=x0, y=y0, w=max(x1 - x0, 0.0), h=max(y1 - y0, 0.0)),
        outline=tr.outline,
        velocity=tr.velocity,
        status=tr.status,
    )


class LivePipeline:
    def __init__(
        self, bridge: Bridge, source: FrameSource, detector: Detector, settings: LiveSettings,
        tracker: ByteTracker | None = None, world: WorldState | None = None,
    ) -> None:
        self.bridge = bridge
        self.source = source
        self.detector = detector
        self.tracker = tracker or ByteTracker()
        self.settings = settings
        self.world = world
        self.base_prompts = list(getattr(detector, "classes", []) or [])
        self._watch_version = world.watch_version if world else 0
        self.dismissed: set[str] = set()
        self._latency: dict[str, deque[float]] = {k: deque(maxlen=30) for k in ("capture", "detector", "tracker")}
        self._frame_times: deque[float] = deque(maxlen=30)
        self._dimmed = False
        self.frame_listeners: list[Callable[[Frame], None]] = []
        """Called with every captured frame; must return quickly (scene memory uses it)."""
        bridge.on_message(self._on_message)

    async def _on_message(self, msg: Message) -> None:
        if isinstance(msg, ClearMsg):
            # Dismiss: drop the current glows; tracks that appear later still glow.
            self.dismissed |= {tr.ref for tr in self.tracker.tracks}
            self._dimmed = False
        elif isinstance(msg, ConfigChangedMsg):
            strength = msg.changes.get("visual_effects.dim_strength")
            if isinstance(strength, (int, float)) and 0 <= strength <= 1:
                self.settings.dim_strength = float(strength)

    def watch_roles(self) -> dict[str, ColorRole]:
        """Label -> glow color for everything currently watched."""
        roles: dict[str, ColorRole] = {label: "target" for label in self.settings.watch}
        if self.world:
            roles.update({w.target: w.color_role for w in self.world.watches.values()})
        return roles

    def prompts(self) -> list[str]:
        """YOLOE prompts: the startup ones plus every watched target."""
        return self.base_prompts + [t for t in self.watch_roles() if t not in self.base_prompts]

    def watched(self, tracks: list[Track], statuses: tuple[str, ...] = ("confirmed",)) -> list[Track]:
        roles = self.watch_roles()
        return [
            tr for tr in tracks
            if tr.status in statuses and tr.label in roles and tr.ref not in self.dismissed
        ]

    def step(self, frame_ts: float, detections: list[Detection]) -> list[Message]:
        """Track one detector result and return the messages to send."""
        t0 = time.perf_counter()
        tracks = self.tracker.update(detections, frame_ts)
        self._latency["tracker"].append((time.perf_counter() - t0) * 1000)
        live_refs = {tr.ref for tr in tracks}
        self.dismissed &= live_refs
        objects = [to_message(tr) for tr in tracks]
        out: list[Message] = []
        if self.world:
            self.world.set_objects(frame_ts, objects)
            if self.world.watch_version != self._watch_version:
                # A new request replaces the old glows: the overlay only drops
                # highlights on clear, so clear first and resend what remains.
                self._watch_version = self.world.watch_version
                self.dismissed.clear()
                out.append(ClearMsg(reason="watches changed"))
        roles = self.watch_roles()
        if not self.settings.show_all:
            # Only what was asked for reaches the overlay; Qwen still sees everything.
            objects = [o for o in objects if o.label in roles]
        out.append(ObjectsMsg(frame_ts=frame_ts, objects=objects))
        targets = self.watched(tracks)
        # Highlights, focus and dim are idempotent in the overlay, so they are
        # resent every step: an app that connects late still gets them.
        out += [HighlightMsg(ref=tr.ref, color_role=roles[tr.label]) for tr in targets]
        if self.settings.spotlight:
            # Lost tracks keep the spotlight for their short grace period, so
            # a brief occlusion doesn't flash the dimming off and on.
            spot = self.watched(tracks, ("confirmed", "lost"))
            if spot:
                out.append(FocusMsg(refs=[tr.ref for tr in spot]))
                out.append(DimMsg(on=True, strength=self.settings.dim_strength))
                self._dimmed = True
            elif self._dimmed:
                out.append(DimMsg(on=False, strength=self.settings.dim_strength))
                self._dimmed = False
        return out

    def status(self) -> StatusMsg:
        fps = None
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / span if span > 0 else None
        latency = {k: sum(v) / len(v) for k, v in self._latency.items() if v}
        return StatusMsg(
            perception_fps=fps,
            latency_ms=latency,
            components={"capture": "running", "detector": "running", "tracker": "running"},
        )

    async def run(self, stop: asyncio.Event) -> None:
        period = 1.0 / self.settings.rate_hz
        last_status = 0.0
        while not stop.is_set():
            started = time.monotonic()
            t0 = time.perf_counter()
            frame = await asyncio.to_thread(self.source.latest)
            self._latency["capture"].append((time.perf_counter() - t0) * 1000)
            if frame is None:
                await asyncio.sleep(period)
                continue
            for listen in self.frame_listeners:
                listen(frame)
            prompts = self.prompts()
            if prompts != list(self.detector.classes):
                log.info("YOLOE prompts: %s", ", ".join(prompts))
                await asyncio.to_thread(self.detector.set_classes, prompts)
            t0 = time.perf_counter()
            detections = await asyncio.to_thread(self.detector.detect, frame.image)
            self._latency["detector"].append((time.perf_counter() - t0) * 1000)
            for msg in self.step(frame.ts, detections):
                self.bridge.send(msg)
            self._frame_times.append(time.monotonic())
            if started - last_status >= 1.0:
                self.bridge.send(self.status())
                last_status = started
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))
        self.source.close()
