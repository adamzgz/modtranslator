"""Hybrid translation backend: tc-big for short strings, NLLB for long strings.

Combines Opus-MT tc-big (fast, reliable on short texts) with NLLB 1.3B
(better quality on longer sentences) to get the best of both models.

Analysis on 9,708 strings (4 ESMs) showed:
- NLLB produces better translations on 80.4% of long strings (4+ words)
- NLLB hallucinates/duplicates on short strings (95 hallucinations avoided)
- Hybrid approach: 0 regressions vs tc-big-only, significant quality gain on long text
"""

from __future__ import annotations

from modtranslator.backends.base import TranslationBackend

# Strings with fewer words than this go to tc-big; the rest go to NLLB.
# Threshold of 4 was determined by analysis: <=3 words is where NLLB
# hallucinates ("Scar" -> "es una cicatriz.") and duplicates ("Sofá Sofá").
DEFAULT_WORD_THRESHOLD = 4

# NLLB accumulates tokenized texts in GPU memory. Sending 20K+ texts at once
# can exhaust VRAM (8GB). Chunk to keep GPU memory bounded.
_NLLB_CHUNK_SIZE = 500


class HybridBackend(TranslationBackend):
    """Hybrid backend: Opus-MT tc-big for short strings, NLLB 1.3B for long strings."""

    def __init__(
        self,
        device: str = "auto",
        word_threshold: int = DEFAULT_WORD_THRESHOLD,
        nllb_model_size: str = "1.3B",
    ) -> None:
        from modtranslator.backends.nllb import NLLBBackend
        from modtranslator.backends.opus_mt import OpusMTBackend

        self._word_threshold = word_threshold
        self._tc_big = OpusMTBackend(device=device, model_variant="tc-big")
        self._nllb = NLLBBackend(device=device, model_size=nllb_model_size)

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        if not texts:
            return []

        # Split texts by word count into short (tc-big) and long (NLLB) groups
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

        # Translate each group with its backend
        short_results: list[str] = []
        long_results: list[str] = []

        if short_texts:
            short_results = self._tc_big.translate_batch(
                short_texts, target_lang, source_lang
            )
        if long_texts:
            # Chunk NLLB calls to avoid GPU memory exhaustion on large files
            for start in range(0, len(long_texts), _NLLB_CHUNK_SIZE):
                chunk = long_texts[start : start + _NLLB_CHUNK_SIZE]
                long_results.extend(
                    self._nllb.translate_batch(chunk, target_lang, source_lang)
                )

        # Reassemble in original order
        results = [""] * len(texts)
        for i, idx in enumerate(short_indices):
            results[idx] = short_results[i]
        for i, idx in enumerate(long_indices):
            results[idx] = long_results[i]

        return results
