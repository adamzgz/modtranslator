"""Protect Spanish words in mixed-language strings before translation.

Thin wrapper around target_protect.py for backwards compatibility.
All logic is now in the generic target_protect module.
"""

from __future__ import annotations

from modtranslator.translation.target_protect import (
    _normalize_placeholders as _normalize_es_placeholders,
)
from modtranslator.translation.target_protect import (
    protect_target_batch,
    protect_target_words,
    restore_target_batch,
    restore_target_words,
)


def protect_spanish_words(text: str) -> tuple[str, dict[str, str]]:
    """Replace Spanish words in text with Cx0, Cx1... placeholders."""
    return protect_target_words(text, "ES")


def protect_spanish_batch(
    texts: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Protect Spanish words in multiple texts."""
    return protect_target_batch(texts, "ES")


def restore_spanish_words(text: str, mapping: dict[str, str]) -> str:
    """Restore Cx0 placeholders with original Spanish words."""
    return restore_target_words(text, mapping)


def restore_spanish_batch(
    texts: list[str], mappings: list[dict[str, str]]
) -> list[str]:
    """Restore Spanish word placeholders in multiple texts."""
    return restore_target_batch(texts, mappings)


__all__ = [
    "_normalize_es_placeholders",
    "protect_spanish_batch",
    "protect_spanish_words",
    "restore_spanish_batch",
    "restore_spanish_words",
]
