"""Global push-to-talk hotkey (plan 9.1), heard even while the game has focus.

The hotkey is a key with optional modifiers, e.g. ``alt+3``, ``f8`` or
``ctrl+shift+space``. Talking starts when the whole combination is down and
stops as soon as any part of it is released.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

DEFAULT_KEY = "alt+3"
MODIFIERS = {"alt", "ctrl", "shift"}


def parse_hotkey(text: str) -> tuple[frozenset[str], str]:
    """'Alt+3' -> ({'alt'}, '3'). Key names follow pynput: f8, space, caps_lock..."""
    parts = [p.strip().lower() for p in text.split("+") if p.strip()]
    if not parts or parts[-1] in MODIFIERS:
        raise ValueError(f"push-to-talk hotkey {text!r} needs a key, e.g. alt+3 or f8")
    mods = frozenset(parts[:-1])
    if not mods <= MODIFIERS:
        raise ValueError(f"unknown modifier in {text!r}; use alt, ctrl or shift")
    return mods, parts[-1]


def key_name(key) -> str | None:
    """Name of a pynput key event: 'alt' for either Alt (AltGr included),
    '3' for the 3 key whatever character the layout and modifiers give it."""
    name = getattr(key, "name", None)
    if name:  # pynput.keyboard.Key
        base = name.split("_")[0]
        return base if base in MODIFIERS else name
    vk = getattr(key, "vk", None)
    if vk is not None and (0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A):
        return chr(vk).lower()  # Windows virtual-key codes for 0-9 and A-Z
    char = getattr(key, "char", None)
    return char.lower() if char else None


class Hotkey:
    """Tracks key state; ``press``/``release`` return True when the hotkey
    becomes held or stops being held."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.mods, self.key = parse_hotkey(text)
        self.down: set[str] = set()
        self.held = False

    def press(self, name: str | None) -> bool:
        if name is None:
            return False
        self.down.add(name)
        if not self.held and self.key in self.down and self.mods <= self.down:
            self.held = True
            return True
        return False

    def release(self, name: str | None) -> bool:
        self.down.discard(name)
        if self.held and (name == self.key or name in self.mods):
            self.held = False
            return True
        return False


class PushToTalk:
    """Calls ``on_press`` once when the hotkey goes down and ``on_release``
    when it comes up, on the asyncio loop. Key auto-repeat is ignored."""

    def __init__(self, key: str, on_press: Callable[[], None], on_release: Callable[[], None]) -> None:
        self.hotkey = Hotkey(key)
        self.on_press = on_press
        self.on_release = on_release
        self._listener = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        from pynput import keyboard

        def press(key) -> None:
            if self.hotkey.press(key_name(key)):
                loop.call_soon_threadsafe(self.on_press)

        def release(key) -> None:
            if self.hotkey.release(key_name(key)):
                loop.call_soon_threadsafe(self.on_release)

        self._listener = keyboard.Listener(on_press=press, on_release=release)
        self._listener.start()
        log.info("push-to-talk: hold %s", self.hotkey.text)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
