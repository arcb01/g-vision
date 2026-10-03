"""Speech-to-text for push-to-talk clips (plan 9.1).

faster-whisper is the default: on Arnau's PC it got 5% word errors on
Spanish-accented English against 69% for Nemotron, at ~200-450 ms per clip
for ``medium``. Nemotron stays available with ``--asr nemotron``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from gvision.audio.mic import SAMPLE_RATE

log = logging.getLogger(__name__)

NEMOTRON_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"


class SpeechToText(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


def add_cuda_dll_dirs() -> None:
    """faster-whisper (CTranslate2) on Windows needs cuBLAS and cuDNN 9 DLLs.
    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels install them, but not
    on the DLL search path."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        log.warning("nvidia-cublas-cu12 / nvidia-cudnn-cu12 not installed; faster-whisper may not find CUDA")
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


class WhisperASR:
    def __init__(self, model: str = "medium", device: str = "cuda", language: str = "en") -> None:
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        compute_type = "float16" if device.startswith("cuda") else "int8"
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        log.info("faster-whisper %s on %s ready in %.1f s", model, device, time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language=self.language, beam_size=5, vad_filter=False, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class NemotronASR:
    """Whole-clip transcription through Transformers, as tested on Windows
    (no NeMo). Streaming chunks are not used yet."""

    def __init__(self, model: str = NEMOTRON_MODEL, device: str = "cuda") -> None:
        import torch
        from transformers import pipeline

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device, torch_dtype=dtype)

    def transcribe(self, audio: np.ndarray) -> str:
        return self.pipe({"raw": audio, "sampling_rate": SAMPLE_RATE})["text"].strip()


def load_asr(name: str, device: str = "cuda", whisper_model: str = "medium") -> SpeechToText:
    if name == "whisper":
        return WhisperASR(whisper_model, device)
    if name == "nemotron":
        return NemotronASR(device=device)
    raise ValueError(f"unknown ASR {name!r}: use whisper or nemotron")
