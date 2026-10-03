"""Microphone recording while push-to-talk is held."""

from __future__ import annotations

import threading

import numpy as np

SAMPLE_RATE = 16_000
"""What both Whisper and Nemotron expect."""


def loudness(samples: np.ndarray) -> float:
    """RMS mapped to 0..1 so normal speech sits around the middle."""
    if len(samples) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return min(1.0, (rms / 0.15) ** 0.6)


def envelope(samples: np.ndarray, sample_rate: int, hz: float) -> list[float]:
    """Loudness every 1/hz seconds, to animate a voice while it plays."""
    step = max(1, int(sample_rate / hz))
    return [loudness(samples[i : i + step]) for i in range(0, len(samples), step)]


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

    def level(self) -> float:
        """Loudness of the latest audio block, 0..1, for the voice waves."""
        with self._lock:
            last = self._chunks[-1] if self._chunks else None
        return loudness(last) if last is not None else 0.0

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
