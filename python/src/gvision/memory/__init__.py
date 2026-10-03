"""Scene memory (plan 7.2, 7.3): what happened in the last minute.

- ``FrameHistory``: ~60 s of downscaled JPEG frames at ~2 fps;
- ``EventLog``: what the tracker saw appear and leave;
- ``Narrator``: Qwen's running notes on the situation, every ~25 s;
- ``LookTool``: the agent's ``look`` tool for "what just hit me?";
- ``SharedQwen``: lets the narrator share llama-server, the player first.
"""

from gvision.memory.events import Event, EventLog
from gvision.memory.history import FrameHistory, StoredFrame
from gvision.memory.look import LookTool
from gvision.memory.narrator import Narrator
from gvision.memory.shared import Preempted, SharedQwen

SITUATION_HINT = "- For 'what's going on' or 'what am I doing' questions, answer from the situation in the state."

__all__ = [
    "Event", "EventLog", "FrameHistory", "LookTool", "Narrator", "Preempted", "SITUATION_HINT", "SharedQwen",
    "StoredFrame",
]
