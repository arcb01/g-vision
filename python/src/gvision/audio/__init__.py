"""Audio process (plan sections 9.1 and 9.3).

Push-to-talk recording, speech-to-text (faster-whisper by default, Nemotron
as an option) and Kokoro text-to-speech on the CPU. Heavy libraries are
imported lazily so the package works without them.
"""
