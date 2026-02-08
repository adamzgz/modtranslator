"""Abstract base class for translation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TranslationBackend(ABC):
    """Interface for translation backends."""

    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        """Translate a batch of texts.

        Args:
            texts: List of strings to translate.
            target_lang: Target language code (e.g. "ES" for Spanish).
            source_lang: Source language code, or None for auto-detect.

        Returns:
            List of translated strings, same length as input.
        """
        ...

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
    ) -> str:
        """Translate a single text. Default implementation uses translate_batch."""
        results = self.translate_batch([text], target_lang, source_lang)
        return results[0]
