"""Hybrid translation backend: Opus-MT for short strings, NLLB for long strings.

Combines Opus-MT (fast, reliable on short texts) with NLLB 1.3B
(better quality on longer sentences) to get the best of both models.

For each language pair, the Opus-MT backend is selected automatically:
  1. Try opus-mt-tc-big-{src}-{tgt} (best quality)
  2. Fall back to opus-mt-{src}-{tgt} (base, smaller but reliable)
  3. If neither exists, use NLLB for all strings (NLLB-only mode)

Analysis on 9,708 strings (4 ESMs) showed:
- NLLB produces better translations on 80.4% of long strings (4+ words)
- NLLB hallucinates/duplicates on short strings (95 hallucinations avoided)
- Hybrid approach: 0 regressions vs tc-big-only, significant quality gain on long text
"""

from __future__ import annotations

import warnings

from modtranslator.backends.base import TranslationBackend

# Strings with fewer words than this go to Opus-MT; the rest go to NLLB.
# Threshold of 4 was determined by analysis: <=3 words is where NLLB
# hallucinates ("Scar" -> "es una cicatriz.") and duplicates ("Sofá Sofá").
DEFAULT_WORD_THRESHOLD = 4

# NLLB accumulates tokenized texts in GPU memory. Sending 20K+ texts at once
# can exhaust VRAM (8GB). Chunk to keep GPU memory bounded.
_NLLB_CHUNK_SIZE = 500


class HybridBackend(TranslationBackend):
    """Hybrid backend: Opus-MT for short strings, NLLB 1.3B for long strings.

    The Opus-MT variant (tc-big or base) is selected automatically on the
    first translate_batch call based on model availability for the language pair.
    """

    def __init__(
        self,
        device: str = "auto",
        word_threshold: int = DEFAULT_WORD_THRESHOLD,
        nllb_model_size: str = "1.3B",
    ) -> None:
        from modtranslator.backends.nllb import NLLBBackend

        self._word_threshold = word_threshold
        self._device = device
        self._nllb = NLLBBackend(device=device, model_size=nllb_model_size)

        # Opus-MT backend, resolved lazily on first translate_batch call.
        # None means NLLB-only mode (no Opus-MT available for the language pair).
        self._opus: object | None = None
        self._opus_initialized: bool = False  # True once _init_opus has run
        self._opus_variant: str | None = None  # "tc-big" or "base" if loaded

    @property
    def mode(self) -> str:
        """Active mode after first translate_batch call.

        Returns "opus-mt-{variant}+nllb" or "nllb-only".
        Before initialization, returns "pending".
        """
        if not self._opus_initialized:
            return "pending"
        if self._opus is None:
            return "nllb-only"
        return f"opus-mt-{self._opus_variant}+nllb"

    def _init_opus(self, target_lang: str, source_lang: str | None) -> None:
        """Detect best available Opus-MT variant for the language pair.

        Tries tc-big first (better quality), then base (smaller but no hallucinations).
        If neither is available, self._opus stays None and NLLB handles all strings.
        """
        from modtranslator.backends.opus_mt import _LANG_CODES, OpusMTBackend

        self._opus_initialized = True
        src_key = (source_lang or "EN").upper()
        src = _LANG_CODES.get(src_key, src_key.lower())
        tgt_key = target_lang.upper()
        tgt = _LANG_CODES.get(tgt_key, tgt_key.lower())

        errors: list[str] = []
        for variant in ("tc-big", "base"):
            try:
                backend = OpusMTBackend(device=self._device, model_variant=variant)
                backend._ensure_model(src, tgt)
                self._opus = backend
                self._opus_variant = variant
                return
            except Exception as exc:
                errors.append(f"opus-mt-{variant}: {exc}")
                continue
        # No Opus-MT available → NLLB-only (self._opus stays None)
        warnings.warn(
            f"Opus-MT not available for {src}→{tgt}, using NLLB-only mode. "
            f"Tried: {'; '.join(errors)}",
            stacklevel=2,
        )

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        if not texts:
            return []

        if not self._opus_initialized:
            self._init_opus(target_lang, source_lang)

        if self._opus is None:
            # NLLB-only mode: send all strings to NLLB in chunks
            results: list[str] = []
            for start in range(0, len(texts), _NLLB_CHUNK_SIZE):
                chunk = texts[start : start + _NLLB_CHUNK_SIZE]
                results.extend(self._nllb.translate_batch(chunk, target_lang, source_lang))
            return results

        # Split texts by word count into short (Opus-MT) and long (NLLB) groups
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
            short_results = self._opus.translate_batch(  # type: ignore[union-attr]
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
