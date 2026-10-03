"""Microphone recording while push-to-talk is held."""

from __future__ import annotations

import threading

import numpy as np

SAMPLE_RATE = 16_000
"""What both Whisper and Nemotron expect."""


class Recorder:
    def __init__(self, device: int | str | None = None) -> None:
        self.device = device
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=self.device, callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time, status) -> None:  # noqa: ARG002
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        """Stop and return the mono float32 clip at 16 kHz."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
            self._chunks = []
        return audio
