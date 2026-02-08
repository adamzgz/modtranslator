"""Dummy translation backend for testing — prefixes strings with [XX] tag."""

from __future__ import annotations

from modtranslator.backends.base import TranslationBackend


class DummyBackend(TranslationBackend):
    """Test backend that prefixes each string with the target language tag.

    Example: "Iron Sword" → "[ES] Iron Sword"
    """

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        tag = f"[{target_lang.upper()}]"
        return [f"{tag} {text}" for text in texts]
