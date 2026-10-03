"""Messages exchanged between the Python processes and the Electron app.

This module is the single source of truth for the WebSocket protocol
(plan section 12). ``schema/messages.schema.json`` is generated from it with
``python -m gvision.protocol`` and consumed by the Electron app.

Conventions (plan section 8):
- every message carries ``v`` (protocol version), ``type`` and ``ts``
  (sender's Unix time in seconds);
- screen coordinates are normalized to 0..1, origin top-left;
- elements are addressed by reference IDs: ``obj:<n>``, ``text:<n>``,
  ``region:<name>``.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PROTOCOL_VERSION = 1

Ref = Annotated[str, Field(pattern=r"^(obj|text|region):[A-Za-z0-9_.-]+$")]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[float, float]
"""Normalized (x, y). May fall outside 0..1 for objects partly off-screen."""

ColorRole = Literal["target", "danger", "info"]
"""Semantic glow colors: target = gold, danger = red, info = cyan (plan 6.3)."""

HighlightStyle = Literal["glow", "outline", "subtle"]
TrackStatus = Literal["tentative", "confirmed", "lost"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class _Message(_Model):
    v: Literal[1] = PROTOCOL_VERSION
    ts: float = Field(default_factory=time.time)


# --- Payload parts ---------------------------------------------------------


class Box(_Model):
    x: float
    y: float
    w: Annotated[float, Field(ge=0.0)]
    h: Annotated[float, Field(ge=0.0)]


class TrackedObject(_Model):
    ref: Ref
    label: str
    confidence: Unit
    box: Box
    outline: list[Point] | None = None
    """Simplified mask polygon; ``None`` means use the box."""
    velocity: Point = (0.0, 0.0)
    """Normalized units per second, for overlay extrapolation."""
    status: TrackStatus


class Segment(_Model):
    id: Annotated[int, Field(ge=0)]
    text: str
    refs: list[Ref] = []


class Badge(_Model):
    number: Annotated[int, Field(ge=1)]
    ref: Ref


# --- Python -> app ---------------------------------------------------------


class ObjectsMsg(_Message):
    """Tracked objects for one frame."""

    type: Literal["objects"] = "objects"
    frame_ts: float
    objects: list[TrackedObject]


class HighlightMsg(_Message):
    """Set an element's glow style and meaning."""

    type: Literal["highlight"] = "highlight"
    ref: Ref
    style: HighlightStyle = "glow"
    color_role: ColorRole
    uncertain: bool = False
    """Rendered as a dashed outline."""
    box: Box | None = None
    """Where the element is, for ``text:`` and ``region:`` refs that don't
    appear in ``objects`` messages. ``obj:`` refs follow their track instead."""


class FocusMsg(_Message):
    """Move the spotlight to these elements."""

    type: Literal["focus"] = "focus"
    refs: list[Ref]
    segment_id: int | None = None


class DimMsg(_Message):
    """Control the spotlight dim layer."""

    type: Literal["dim"] = "dim"
    on: bool
    strength: Unit = 0.6
    """Opacity of the dark layer; 0 is no dimming, 1 is black."""


class AnswerMsg(_Message):
    """Show an answer in the panel, split into speakable segments."""

    type: Literal["answer"] = "answer"
    answer_id: str
    segments: list[Segment]


class SegmentStartedMsg(_Message):
    type: Literal["segment_started"] = "segment_started"
    answer_id: str
    segment_id: int


class AnswerFinishedMsg(_Message):
    type: Literal["answer_finished"] = "answer_finished"
    answer_id: str


class BadgesMsg(_Message):
    """Numbered choices when a request is ambiguous."""

    type: Literal["badges"] = "badges"
    badges: list[Badge]


class StatusMsg(_Message):
    """Dashboard metrics."""

    type: Literal["status"] = "status"
    perception_fps: float | None = None
    game_fps: float | None = None
    vram_used_mb: float | None = None
    latency_ms: dict[str, float] = {}
    components: dict[str, Literal["running", "paused", "off", "error"]] = {}


class VoiceMsg(_Message):
    """Push-to-talk state for the panel: what was heard and what happens now."""

    type: Literal["voice"] = "voice"
    state: Literal["listening", "thinking", "speaking", "idle"]
    transcript: str | None = None
    """What speech-to-text heard, once the key is released."""
    level: Unit | None = None
    """Loudness of the microphone (listening) or of the answer's voice
    (speaking), sent ~20 times a second to drive the overlay's voice waves."""


# --- Either direction ------------------------------------------------------


class ClearMsg(_Message):
    """Cancel speech, highlights and dimming (interrupt or dismiss)."""

    type: Literal["clear"] = "clear"
    reason: str | None = None


# --- App -> Python ---------------------------------------------------------


class ConfigChangedMsg(_Message):
    """Settings edited in the control panel."""

    type: Literal["config_changed"] = "config_changed"
    changes: dict[str, Any]


Message = Annotated[
    Union[
        ObjectsMsg,
        HighlightMsg,
        FocusMsg,
        DimMsg,
        AnswerMsg,
        SegmentStartedMsg,
        AnswerFinishedMsg,
        BadgesMsg,
        StatusMsg,
        VoiceMsg,
        ClearMsg,
        ConfigChangedMsg,
    ],
    Field(discriminator="type"),
]

message_adapter: TypeAdapter[Message] = TypeAdapter(Message)


def parse(data: str | bytes) -> Message:
    """Parse and validate one JSON message."""
    return message_adapter.validate_json(data)


def dump(msg: Message) -> str:
    """Serialize one message to JSON."""
    return msg.model_dump_json()


def json_schema() -> dict[str, Any]:
    # Serialization mode marks defaulted fields (v, type, ts, ...) as
    # required: a message on the wire always carries them.
    schema = message_adapter.json_schema(mode="serialization")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/arcb01/g-vision/schema/messages.schema.json",
        "title": "G-VISION message",
        "description": "Generated from python/src/gvision/protocol/messages.py. Do not edit by hand.",
        **schema,
    }
