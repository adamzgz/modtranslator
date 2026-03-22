"""Hybrid translation backend: NLLB for long strings, DeepL for short strings.

NLLB produces good translations on long texts (4+ words) but hallucinates
on short inputs (1-3 words). DeepL handles short strings reliably.
This hybrid gets the best of both: offline NLLB quality on long text,
DeepL precision on short text.
"""

from __future__ import annotations

from modtranslator.backends.base import TranslationBackend

# Strings with fewer words than this go to DeepL; the rest go to NLLB.
DEFAULT_WORD_THRESHOLD = 4

# NLLB chunk size to bound GPU memory (see hybrid.py for rationale).
_CHUNK_BATCH_RATIO = 15


class HybridDeepLBackend(TranslationBackend):
    """Hybrid backend: DeepL for short strings, NLLB for long strings."""

    def __init__(
        self,
        api_key: str,
        device: str = "auto",
        word_threshold: int = DEFAULT_WORD_THRESHOLD,
        nllb_model_size: str = "1.3B",
    ) -> None:
        from modtranslator.backends.deepl import DeepLBackend
        from modtranslator.backends.nllb import NLLBBackend

        self._word_threshold = word_threshold
        self._deepl = DeepLBackend(api_key=api_key)
        self._nllb = NLLBBackend(device=device, model_size=nllb_model_size)
        self._nllb_chunk_size = self._nllb._gpu_batch_size * _CHUNK_BATCH_RATIO

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        if not texts:
            return []

        short_indices: list[int] = []
        short_texts: list[str] = []
        long_indices: list[int] = []
        long_texts: list[str] = []

        for i, text in enumerate(texts):
            if len(text.split()) < self._word_threshold:
                short_indices.append(i)
                short_texts.append(text)
            else:
                long_indices.append(i)
                long_texts.append(text)

        short_results: list[str] = []
        long_results: list[str] = []

        if short_texts:
            short_results = self._deepl.translate_batch(short_texts, target_lang, source_lang)

        if long_texts:
            for start in range(0, len(long_texts), self._nllb_chunk_size):
                chunk = long_texts[start : start + self._nllb_chunk_size]
                long_results.extend(
                    self._nllb.translate_batch(chunk, target_lang, source_lang)
                )

        results = [""] * len(texts)
        for i, idx in enumerate(short_indices):
            results[idx] = short_results[i]
        for i, idx in enumerate(long_indices):
            results[idx] = long_results[i]

        return results
