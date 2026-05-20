"""
cinemateca.models.audio.clap_hf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HF-transformers CLAP audio embedder backend.

This is a stub. Full implementation in Task 4.
"""

from __future__ import annotations


class ClapHFEmbedder:
    def __init__(self, cfg=None, device=None):
        self._cfg = cfg
        self._device = device

    def encode_audio(self, wav_paths):
        raise NotImplementedError

    def encode_text(self, text):
        raise NotImplementedError

    def encode_audio_single(self, wav_path):
        raise NotImplementedError
