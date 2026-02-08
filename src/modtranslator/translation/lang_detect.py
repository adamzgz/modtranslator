"""Heuristic-based language detection for filtering already-translated strings."""

from __future__ import annotations

import re
from pathlib import Path

# Spanish-specific accented characters (lowercase)
_SPANISH_CHARS = set("áéíóúñü")

# Common Spanish words — short, frequent, unlikely in English
_SPANISH_WORDS: set[str] = {
    "del", "los", "las", "una", "uno", "para", "que", "con", "por",
    "pero", "como", "más", "este", "esta", "esto", "estos", "estas",
    "ese", "esa", "esos", "esas", "aquel", "aquella",
    "también", "puede", "desde", "hasta", "sobre", "entre",
    "después", "antes", "durante", "hacia", "según",
    "tiene", "están", "será", "sido", "hacer",
    "todos", "todas", "cada", "otro", "otra", "otros", "otras",
    "muy", "sin", "cuando", "donde", "mientras", "aunque",
    "porque", "además", "entonces",
    # articles/prepositions — weighted lower (ambiguous with English)
    "el", "la", "de", "y",
}

# Short Spanish words that are too ambiguous to count as core hits
_WEAK_SPANISH_WORDS: set[str] = {"el", "la", "de", "y"}

# Words that are very strong Spanish indicators (rarely appear in English text)
_STRONG_SPANISH_WORDS: set[str] = {
    "también", "después", "además", "según", "está", "están",
    "será", "aquí", "así", "más", "todavía", "ningún", "ninguna",
    "puede", "pueden", "tiene", "tienen", "hace", "hacia",
}

# Lazy-loaded Spanish dictionary (~2000 words from data/spanish_words.txt)
_spanish_dictionary: frozenset[str] | None = None


def _load_spanish_dictionary() -> frozenset[str]:
    """Load the curated Spanish dictionary from data/spanish_words.txt.

    Uses importlib.resources with fallback to __file__-relative path.
    Loaded once, cached as frozenset for O(1) lookups.
    """
    global _spanish_dictionary
    if _spanish_dictionary is not None:
        return _spanish_dictionary

    # Try importlib.resources first (works with installed packages)
    try:
        from importlib.resources import files

        data_path = files("modtranslator.data").joinpath("spanish_words.txt")
        text = data_path.read_text(encoding="utf-8")
    except (ImportError, FileNotFoundError, TypeError):
        # Fallback: resolve relative to this file
        fallback = Path(__file__).resolve().parent.parent / "data" / "spanish_words.txt"
        text = fallback.read_text(encoding="utf-8")

    words: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            words.add(stripped.lower())

    _spanish_dictionary = frozenset(words)
    return _spanish_dictionary

# Common English words (general vocabulary)
_ENGLISH_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "not", "no", "yes", "and", "or", "but", "if", "then",
    "that", "this", "these", "those", "it", "its",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "up", "out", "off", "over", "into", "through", "about",
    "after", "before", "between", "under", "above",
    "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "any", "many", "much", "such",
    "only", "also", "very", "just", "than", "too",
    "here", "there", "where", "when", "how", "what", "which", "who",
    "you", "your", "he", "she", "they", "we", "my", "his", "her",
    "find", "get", "give", "go", "keep", "let", "make", "say",
    "take", "come", "see", "look", "want", "use", "work",
}

# Strong English indicators — words that almost never appear in Spanish text
_STRONG_ENGLISH_WORDS: set[str] = {
    "the", "with", "that", "have", "this", "from", "they",
    "been", "would", "could", "should", "which", "their",
    "through", "between", "those", "these", "before", "after",
    "where", "there", "does", "were", "your", "himself",
    "herself", "itself", "ourselves", "themselves",
    "something", "nothing", "everything", "anything",
    "however", "although", "because", "whether", "without",
}

# Minimum text length to attempt detection (2 = skip only empty/single-char strings)
_MIN_LENGTH = 2

# Word boundary regex for splitting
_WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)


def is_spanish(text: str) -> bool:
    """Determine if a text string is likely Spanish.

    Uses a scoring heuristic based on:
    - Presence of Spanish-specific characters (á, é, í, ó, ú, ñ, ü)
    - Frequency of common Spanish words
    - Strong indicator words that almost never appear in English
    - For 1-2 word strings: check against the curated dictionary with lower threshold

    Returns True if the text scores above the detection threshold.
    Short strings (< 4 chars) always return False (not enough signal).
    """
    if len(text) < _MIN_LENGTH:
        return False

    lower = text.lower()
    words = _WORD_RE.findall(lower)

    if not words:
        return False

    score = 0.0
    word_count = len(words)
    dictionary = _load_spanish_dictionary()

    # Check for Spanish-specific characters
    spanish_char_count = sum(1 for c in lower if c in _SPANISH_CHARS)
    if spanish_char_count > 0:
        score += min(spanish_char_count * 0.3, 1.5)

    # For short strings (1-2 words), check against dictionary
    if word_count <= 2:
        for w in words:
            if w in dictionary:
                score += 0.6
            if w in _STRONG_SPANISH_WORDS:
                score += 0.8
            elif w in _SPANISH_WORDS:
                score += 0.3
        # Lower threshold for short strings — 0.6 means a single match suffices
        return score >= 0.6

    # Check for common Spanish words (3+ words)
    spanish_word_hits = 0
    core_hits = 0  # hits from _SPANISH_WORDS / _STRONG_SPANISH_WORDS (higher confidence)
    for w in words:
        if w in _STRONG_SPANISH_WORDS:
            score += 0.8
            spanish_word_hits += 1
            core_hits += 1
        elif w in _SPANISH_WORDS:
            # Short function words are weak signals — could appear in English
            if w in _WEAK_SPANISH_WORDS:
                score += 0.1
                spanish_word_hits += 1
                # Don't count as core_hit — too ambiguous alone
            else:
                score += 0.4
                spanish_word_hits += 1
                core_hits += 1
        elif w in dictionary:
            # Dictionary match as fallback — lower weight than core words
            score += 0.2
            spanish_word_hits += 1

    # Ratio of Spanish words to total words
    if word_count > 2:
        ratio = spanish_word_hits / word_count
        if core_hits > 0 and ratio > 0.3:
            # At least one core word confirms the ratio signal
            score += 1.0
        elif core_hits > 0 and ratio > 0.15:
            score += 0.5
        elif ratio > 0.5:
            # Dictionary-only (+ weak words): strict majority of words are Spanish
            score += 0.8
        elif ratio > 0.3:
            # Dictionary-only, moderate ratio
            score += 0.4

    # Threshold: need a score of at least 1.0 to classify as Spanish
    return score >= 1.0


def is_english(text: str) -> bool:
    """Determine if a text string is likely English.

    Mirrors is_spanish() logic but using English word lists.
    Returns True if the text has strong English signals.
    Short strings (< 4 chars) always return False.
    """
    if len(text) < _MIN_LENGTH:
        return False

    lower = text.lower()
    words = _WORD_RE.findall(lower)

    if not words:
        return False

    score = 0.0
    word_count = len(words)

    # Presence of Spanish characters is negative evidence for English
    spanish_char_count = sum(1 for c in lower if c in _SPANISH_CHARS)
    if spanish_char_count > 0:
        score -= min(spanish_char_count * 0.5, 2.0)

    english_word_hits = 0
    for w in words:
        if w in _STRONG_ENGLISH_WORDS:
            score += 0.8
            english_word_hits += 1
        elif w in _ENGLISH_WORDS:
            score += 0.3
            english_word_hits += 1

    # Ratio boost for longer texts
    if word_count > 2:
        ratio = english_word_hits / word_count
        if ratio > 0.3:
            score += 1.0
        elif ratio > 0.15:
            score += 0.5

    # For short strings (1-2 words), lower threshold
    if word_count <= 2:
        return score >= 0.3

    return score >= 1.0


def get_spanish_words_in_text(text: str) -> list[tuple[str, int, int]]:
    """Find Spanish words in a text string.

    Returns a list of (word, start, end) tuples for each word that is
    either in the Spanish dictionary or contains Spanish-specific characters
    (ñ, á, é, í, ó, ú, ü). Only words of 4+ characters are considered.

    Useful for protecting Spanish words in mixed-language strings before
    sending to a translation backend.
    """
    dictionary = _load_spanish_dictionary()
    results: list[tuple[str, int, int]] = []

    for match in _WORD_RE.finditer(text):
        word = match.group()
        if len(word) < 4:
            continue
        lower = word.lower()
        # In dictionary or contains Spanish-specific characters
        has_spanish_chars = any(c in _SPANISH_CHARS for c in lower)
        if lower in dictionary or has_spanish_chars:
            results.append((word, match.start(), match.end()))

    return results


def should_translate(
    text: str,
    target_lang: str,
    glossary_terms: set[str] | None = None,
    glossary_source_terms: set[str] | None = None,
) -> bool:
    """Decide whether a string should be translated.

    Combines multiple layers to minimize false translations:
    0. Glossary source check — if text matches a source term, always translate
    1. Glossary target check — if text matches a target term exactly, skip it
    2. is_spanish() — heuristic with dictionary and short-string support
    3. is_english() — positive English signal as final confirmation

    Returns True if the string should be translated (likely English).
    Returns False if it should be skipped (likely already in target language).

    Philosophy: only translate when confident it's English. In doubt, skip.
    """
    stripped = text.strip()

    # Layer 0: Glossary source terms — always translate (glossary protect/restore
    # will handle the substitution). This bypasses _MIN_LENGTH so short terms
    # like "Dad" → "Papá" are never skipped.
    if glossary_source_terms and stripped.lower() in glossary_source_terms:
        return True

    # Very short strings — not enough signal, skip
    if len(stripped) < _MIN_LENGTH:
        return False

    # Layer 1: Glossary exact match — if the whole text is a known term, skip
    if glossary_terms:
        lower = stripped.lower()
        if lower in glossary_terms:
            return False

    # Only ES detection implemented for now
    if target_lang.upper() != "ES":
        # For non-ES targets, fall back to simple "not short" check
        return True

    # Layer 2: is_spanish() — heuristic (handles short strings too)
    if is_spanish(stripped):
        return False

    # Layer 3: is_english() — positive confirmation for English
    if is_english(stripped):
        return True

    # Ambiguous — for short strings (1-2 words), only skip if there was some
    # Spanish signal. If neither is_spanish nor is_english fired, the word is
    # likely an English noun/name (e.g. "Iron Sword") that should be translated.
    # True Spanish words are already caught by _SPANISH_SINGLE_WORDS in is_spanish().
    return True
