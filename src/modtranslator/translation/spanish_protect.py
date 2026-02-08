"""Protect Spanish words in mixed-language strings before translation.

When a string contains both English and Spanish words (e.g. already-translated
terms mixed with English text), the translation backend may mangle the Spanish
parts. This module replaces detected Spanish words with placeholders Cx0, Cx1...
before translation, then restores them after.

Follows the same protect/restore pattern as glossary.py but uses Cx{i}
placeholders (distinct from glossary's Gx{i}).
"""

from __future__ import annotations

import re

from modtranslator.translation.lang_detect import get_spanish_words_in_text

# Pattern to detect existing glossary placeholders — we must not touch these
_GLOSSARY_PH_RE = re.compile(r"Gx\d+")


def _normalize_es_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Normalize mangled ``Cx0`` placeholders after neural MT translation.

    Same strategy as glossary: recover placeholders that the MT backend may
    have mangled (inserted spaces, changed case, etc.).
    """
    if not mapping:
        return text

    result = text
    for placeholder in mapping:
        if placeholder in result:
            continue
        m = re.match(r"([A-Z]x)(\d+)", placeholder)
        if not m:
            continue
        prefix, num = m.group(1), m.group(2)
        mangled_re = re.compile(
            rf"(?<!\w){prefix}\s*{num}(?!\d)",
            re.IGNORECASE,
        )
        result = mangled_re.sub(placeholder, result)

    return result

# Pattern to detect our own ES placeholders (for restore)
_ES_PH_RE = re.compile(r"Cx\d+")


def protect_spanish_words(text: str) -> tuple[str, dict[str, str]]:
    """Replace Spanish words in text with Cx0, Cx1... placeholders.

    Returns (protected_text, mapping) where mapping is {placeholder: original_word}.
    Existing glossary placeholders Gx0, Gx1... are left untouched.
    Replacements are done right-to-left to preserve character offsets.
    """
    spanish_words = get_spanish_words_in_text(text)
    if not spanish_words:
        return text, {}

    # Find positions occupied by glossary placeholders — skip overlaps
    glossary_spans: set[int] = set()
    for m in _GLOSSARY_PH_RE.finditer(text):
        for i in range(m.start(), m.end()):
            glossary_spans.add(i)

    # Filter out words that overlap with glossary placeholders
    filtered: list[tuple[str, int, int]] = []
    for word, start, end in spanish_words:
        if any(i in glossary_spans for i in range(start, end)):
            continue
        filtered.append((word, start, end))

    if not filtered:
        return text, {}

    # Replace right-to-left to preserve offsets
    mapping: dict[str, str] = {}
    result = text
    for idx, (word, start, end) in enumerate(reversed(filtered)):
        placeholder = f"Cx{idx}"
        mapping[placeholder] = word
        result = result[:start] + placeholder + result[end:]

    return result, mapping


def protect_spanish_batch(
    texts: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Protect Spanish words in multiple texts. Each text gets independent mappings.

    Returns (protected_texts, mappings) where each mapping is per-text.
    """
    protected_texts: list[str] = []
    mappings: list[dict[str, str]] = []
    for t in texts:
        protected, mapping = protect_spanish_words(t)
        protected_texts.append(protected)
        mappings.append(mapping)
    return protected_texts, mappings


def restore_spanish_words(text: str, mapping: dict[str, str]) -> str:
    """Restore Cx0 placeholders with original Spanish words.

    Normalizes mangled placeholders before restoring (handles MT models
    that insert spaces, change case, etc.).
    """
    result = _normalize_es_placeholders(text, mapping)
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    return result


def restore_spanish_batch(
    texts: list[str], mappings: list[dict[str, str]]
) -> list[str]:
    """Restore Spanish word placeholders in multiple texts."""
    return [restore_spanish_words(t, m) for t, m in zip(texts, mappings, strict=False)]
