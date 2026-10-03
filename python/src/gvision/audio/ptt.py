"""Global push-to-talk key (plan 9.1), heard even while the game has focus."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

DEFAULT_KEY = "f8"


def parse_key(name: str):
    """'f8', 'caps_lock', 'ctrl_r' or a single character, as a pynput key."""
    from pynput import keyboard

    name = name.strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    try:
        return keyboard.Key[name]
    except KeyError:
        raise ValueError(f"unknown push-to-talk key {name!r}; try f8, caps_lock or a letter") from None


class PushToTalk:
    """Calls ``on_press`` once when the key goes down and ``on_release`` when it
    comes up, on the asyncio loop. Key auto-repeat is ignored."""

    def __init__(self, key: str, on_press: Callable[[], None], on_release: Callable[[], None]) -> None:
        self.key = parse_key(key)
        self.on_press = on_press
        self.on_release = on_release
        self._down = False
        self._listener = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        from pynput import keyboard

        def press(key) -> None:
            if key == self.key and not self._down:
                self._down = True
                loop.call_soon_threadsafe(self.on_press)

        def release(key) -> None:
            if key == self.key and self._down:
                self._down = False
                loop.call_soon_threadsafe(self.on_release)

        self._listener = keyboard.Listener(on_press=press, on_release=release)
        self._listener.start()
        log.info("push-to-talk: hold %s", self.key)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
