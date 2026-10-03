"""Kokoro-82M text-to-speech via ONNX Runtime (plan 9.3).

On the GPU when onnxruntime-gpu can reach CUDA, else on the CPU. On Arnau's
PC the CPU int8 model ran slower than real time (12.4 s for 10.3 s of
speech) with the game and the text watcher sharing the cores. The GPU uses
the fp32 model, since int8's quantized ops fall back to the CPU there.

The model files (~90 MB int8 or ~310 MB fp32, + ~28 MB voices) download on
first use into ``models/kokoro/``; they are not committed.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path("models") / "kokoro"
_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
CPU_MODEL_FILE = "kokoro-v1.0.int8.onnx"
GPU_MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
DEFAULT_VOICE = "af_heart"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCE_CHARS = 12
"""Shorter pieces ("Yes.") are spoken together with a neighbouring sentence."""


def _fetch(name: str) -> Path:
    path = MODELS_DIR / name
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", name)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(_RELEASE + name, tmp)
        tmp.replace(path)
    return path


def split_sentences(text: str) -> list[str]:
    """Pieces to synthesize one at a time, so speech starts after the first."""
    pieces: list[str] = []
    for part in _SENTENCE_END.split(text.strip()):
        if pieces and len(pieces[-1]) < MIN_SENTENCE_CHARS:
            pieces[-1] = f"{pieces[-1]} {part}"
        elif part:
            pieces.append(part)
    if len(pieces) > 1 and len(pieces[-1]) < MIN_SENTENCE_CHARS:
        pieces[-2:] = [f"{pieces[-2]} {pieces[-1]}"]
    return pieces


def _cuda_session(model: Path):
    """An onnxruntime session on CUDA, or None when this install can't."""
    from gvision.audio.asr import add_cuda_dll_dirs

    add_cuda_dll_dirs()
    try:
        import torch  # noqa: F401  # its CUDA and cuDNN DLLs serve onnxruntime too
    except ImportError:
        pass
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # missing DLLs show up as no CUDA provider below
            log.debug("onnxruntime preload_dlls: %s", e)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        log.warning("Kokoro on the CPU: onnxruntime has no CUDA (install onnxruntime-gpu)")
        return None
    # DEFAULT, not the exhaustive cuDNN search: every new sentence length is a
    # new input shape, and searching each one costs more than it saves.
    cuda = ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
    try:
        session = ort.InferenceSession(str(model), providers=[cuda, "CPUExecutionProvider"])
    except Exception as e:
        log.warning("Kokoro on the CPU: CUDA session failed: %s", e)
        return None
    if session.get_providers()[0] != "CUDAExecutionProvider":
        log.warning("Kokoro on the CPU: CUDA was not usable")
        return None
    return session


class KokoroTTS:
    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = 1.1, device: str = "cuda") -> None:
        from kokoro_onnx import Kokoro

        t0 = time.perf_counter()
        voices = str(_fetch(VOICES_FILE))
        session = _cuda_session(_fetch(GPU_MODEL_FILE)) if device == "cuda" else None
        if session is not None:
            self.kokoro = Kokoro.from_session(session, voices)
            self.device = "GPU"
        else:
            # Kokoro(path) would pick every provider onnxruntime-gpu offers,
            # including CUDA without the DLLs set up, and fail on first use.
            import onnxruntime as ort

            cpu = ort.InferenceSession(str(_fetch(CPU_MODEL_FILE)), providers=["CPUExecutionProvider"])
            self.kokoro = Kokoro.from_session(cpu, voices)
            self.device = "CPU"
        self.voice = voice
        self.speed = speed
        self._stop = threading.Event()
        if self.device == "GPU":
            self.synthesize("Ready.")  # first CUDA run builds kernels; keep that off the first answer
        log.info("Kokoro on the %s ready in %.1f s (voice %s)", self.device, time.perf_counter() - t0, voice)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self.kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Blocking playback; returns early when ``stop`` is called."""
        import sounddevice as sd

        if self._stop.is_set():
            return
        sd.play(samples, sample_rate)
        end = time.monotonic() + len(samples) / sample_rate
        while time.monotonic() < end and not self._stop.wait(0.02):
            pass
        sd.stop()

    def start(self) -> None:
        """A new answer: playback is allowed again after ``stop``."""
        self._stop.clear()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()
