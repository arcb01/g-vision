"""Kokoro-82M text-to-speech on the CPU via ONNX Runtime (plan 9.3).

The model files (~90 MB int8 + ~28 MB voices) download on first use into
``models/kokoro/``; they are not committed.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path("models") / "kokoro"
_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"
DEFAULT_VOICE = "af_heart"


def _fetch(name: str) -> Path:
    path = MODELS_DIR / name
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", name)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(_RELEASE + name, tmp)
        tmp.replace(path)
    return path


class KokoroTTS:
    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = 1.1) -> None:
        from kokoro_onnx import Kokoro

        t0 = time.perf_counter()
        self.kokoro = Kokoro(str(_fetch(MODEL_FILE)), str(_fetch(VOICES_FILE)))
        self.voice = voice
        self.speed = speed
        self._stop = threading.Event()
        log.info("Kokoro ready in %.1f s (voice %s)", time.perf_counter() - t0, voice)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self.kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Blocking playback; returns early when ``stop`` is called."""
        import sounddevice as sd

        self._stop.clear()
        sd.play(samples, sample_rate)
        end = time.monotonic() + len(samples) / sample_rate
        while time.monotonic() < end and not self._stop.wait(0.02):
            pass
        sd.stop()

    def stop(self) -> None:
        self._stop.set()
