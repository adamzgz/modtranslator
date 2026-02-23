"""Heuristic-based language detection for filtering already-translated strings."""

from __future__ import annotations

import re
from pathlib import Path

# Non-English accented/special characters across all target languages.
# Used by is_english() as negative evidence — their presence suggests non-English text.
# ES: áéíóúñü  FR: àâæçéèêëîïôœùûü  DE: äöüß  IT: àèéìíòóùú  PT: ãõàáâéêíóôú  PL: ąćęłńóśźż
_NON_ENGLISH_CHARS = set("áéíóúñüàâæçèêëîïôœùûäößãõąćęłńśźż")

# Per-language common words (short, frequent, unlikely in English)
_LANG_COMMON_WORDS: dict[str, set[str]] = {
    "ES": {
        "del", "los", "las", "una", "uno", "para", "que", "con", "por",
        "pero", "como", "más", "este", "esta", "esto", "estos", "estas",
        "ese", "esa", "esos", "esas", "aquel", "aquella",
        "también", "puede", "desde", "hasta", "sobre", "entre",
        "después", "antes", "durante", "hacia", "según",
        "tiene", "están", "será", "sido", "hacer",
        "todos", "todas", "cada", "otro", "otra", "otros", "otras",
        "muy", "sin", "cuando", "donde", "mientras", "aunque",
        "porque", "además", "entonces",
        "el", "la", "de", "y",
    },
    "FR": {
        "le", "la", "les", "un", "une", "des", "du", "au", "aux",
        "dans", "pour", "avec", "sur", "sous", "vers", "chez", "entre",
        "mais", "donc", "puis", "ou", "et", "ni", "car",
        "est", "sont", "être", "avoir", "fait", "peut", "très",
        "qui", "que", "quoi", "dont", "où",
        "ce", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur",
        "tout", "tous", "toute", "toutes", "autre", "autres",
        "pas", "plus", "bien", "peu", "ici", "comme",
    },
    "DE": {
        "der", "die", "das", "ein", "eine", "einem", "einen", "einer",
        "mit", "für", "auf", "aus", "bei", "nach", "von", "zu", "zum", "zur",
        "und", "oder", "aber", "wenn", "dass", "weil",
        "ist", "sind", "hat", "wird", "kann", "muss",
        "ich", "du", "er", "sie", "wir", "ihr",
        "nicht", "noch", "schon", "nur", "sehr", "hier",
        "dem", "den", "des", "sich", "doch", "dann",
    },
    "IT": {
        "il", "lo", "la", "gli", "le", "un", "una", "uno",
        "con", "per", "tra", "fra", "sul", "dal", "nel", "del", "dei", "della",
        "ma", "poi", "che", "chi", "cui", "dove",
        "è", "sono", "ha", "può", "deve", "sta",
        "questo", "questa", "quello", "quella", "ogni",
        "non", "più", "qui", "come", "tutto", "tutti",
        "io", "tu", "lui", "lei", "noi", "loro",
    },
    "PT": {
        "o", "a", "os", "as", "um", "uma", "uns", "umas",
        "com", "para", "por", "sem", "sob", "sobre", "entre",
        "mas", "pois", "que", "quem", "onde", "como",
        "é", "são", "tem", "pode", "deve", "está",
        "este", "esta", "esse", "essa", "aquele", "aquela",
        "não", "mais", "bem", "aqui", "todo", "todos", "toda",
        "eu", "ele", "ela", "nós", "eles", "elas",
    },
    "RU": {
        "и", "в", "на", "не", "что", "это", "он", "она", "как", "но",
        "для", "все", "так", "его", "уже", "мы", "вы", "они",
        "был", "была", "были", "быть", "есть", "нет",
        "от", "по", "за", "до", "из", "при", "без",
        "тот", "этот", "эта", "эти", "мой", "твой", "свой",
        "где", "кто", "чем", "чего", "или", "если", "когда",
    },
    "PL": {
        "i", "w", "na", "nie", "to", "jest", "jak", "ale", "dla", "ten", "ta",
        "się", "od", "do", "po", "za", "bez", "przy",
        "co", "kto", "lub", "czy", "gdy", "że",
        "są", "ma", "może", "musi", "był", "była",
        "jego", "jej", "ich", "mój", "twój", "nasz",
        "tu", "tam", "tak", "już", "tylko", "bardzo",
        "tego", "tej", "tym", "tych",
    },
}

# Per-language strong indicator words (rarely appear in English text)
_LANG_STRONG_WORDS: dict[str, set[str]] = {
    "ES": {
        "también", "después", "además", "según", "está", "están",
        "será", "aquí", "así", "más", "todavía", "ningún", "ninguna",
        "puede", "pueden", "tiene", "tienen", "hace", "hacia",
    },
    "FR": {
        "aussi", "après", "très", "toujours", "jamais", "peut-être",
        "aujourd'hui", "beaucoup", "maintenant", "cependant", "quelque",
        "plusieurs", "seulement", "pendant", "encore", "depuis",
        "dehors", "dedans", "ailleurs", "parfois", "souvent",
    },
    "DE": {
        "auch", "nicht", "noch", "schon", "immer", "vielleicht", "zwischen",
        "allerdings", "bereits", "eigentlich", "trotzdem", "außerdem",
        "jetzt", "niemals", "manchmal", "natürlich", "wahrscheinlich",
        "deshalb", "darum", "nämlich",
    },
    "IT": {
        "anche", "ancora", "sempre", "dopo", "prima", "molto",
        "però", "quindi", "adesso", "comunque", "davvero",
        "tuttavia", "soltanto", "qualcosa", "qualcuno", "nessuno",
        "finalmente", "spesso", "insieme", "durante",
    },
    "PT": {
        "também", "ainda", "sempre", "depois", "antes", "muito",
        "porém", "então", "agora", "contudo", "realmente",
        "todavia", "somente", "alguém", "ninguém", "alguma",
        "finalmente", "durante", "apenas", "bastante",
    },
    "RU": {
        "также", "после", "перед", "между", "однако", "потому",
        "поэтому", "никогда", "всегда", "сейчас", "теперь",
        "наконец", "обычно", "иногда", "конечно", "действительно",
        "например", "возможно", "несколько", "достаточно",
    },
    "PL": {
        "także", "również", "bardzo", "zawsze", "nigdy", "między",
        "jednak", "dlatego", "teraz", "ponieważ", "właśnie",
        "naprawdę", "wreszcie", "czasami", "oczywiście",
        "prawdopodobnie", "natychmiast", "zazwyczaj",
    },
}

# Per-language weak words (ambiguous with English — too short or shared)
_LANG_WEAK_WORDS: dict[str, set[str]] = {
    "ES": {"el", "la", "de", "y"},
    "FR": {"a", "on", "or", "me", "an"},
    "DE": {"in", "an", "so", "man"},
    "IT": {"a", "e", "me"},
    "PT": {"a", "no", "e", "me"},
    "RU": set(),  # Cyrillic words are never ambiguous with English
    "PL": {"to", "go"},
}

# Lazy-loaded dictionaries: one frozenset per language code (e.g. "ES", "FR")
_dictionaries: dict[str, frozenset[str]] = {}

# Map language codes to dictionary filenames
_LANG_DICT_FILES: dict[str, str] = {
    "ES": "spanish_words.txt",
    "FR": "french_words.txt",
    "DE": "german_words.txt",
    "IT": "italian_words.txt",
    "PT": "portuguese_words.txt",
    "RU": "russian_words.txt",
    "PL": "polish_words.txt",
}

# Per-language special characters (used by get_target_words_in_text for accent detection)
_LANG_SPECIAL_CHARS: dict[str, set[str]] = {
    "ES": set("áéíóúñü"),
    "FR": set("àâæçéèêëïîôœùûüÿ"),
    "DE": set("äöüßẞ"),
    "IT": set("àèéìíîòóùú"),
    "PT": set("ãõàáâéêíóôúç"),
    "RU": set(),  # Cyrillic detected via word regex, not individual chars
    "PL": set("ąćęłńóśźż"),
}

# Word-matching regex per language (most use Latin + accented chars; RU uses Cyrillic)
_LANG_WORD_RE: dict[str, re.Pattern[str]] = {
    "RU": re.compile(r"[а-яёА-ЯЁ]+", re.IGNORECASE),
}


def _load_dictionary(lang: str) -> frozenset[str]:
    """Load a curated word dictionary for the given language.

    Uses importlib.resources with fallback to __file__-relative path.
    Loaded once per language, cached as frozenset for O(1) lookups.
    """
    lang_upper = lang.upper()
    if lang_upper in _dictionaries:
        return _dictionaries[lang_upper]

    filename = _LANG_DICT_FILES.get(lang_upper)
    if filename is None:
        _dictionaries[lang_upper] = frozenset()
        return _dictionaries[lang_upper]

    # Try importlib.resources first (works with installed packages)
    try:
        from importlib.resources import files

        data_path = files("modtranslator.data").joinpath(filename)
        text = data_path.read_text(encoding="utf-8")
    except (ImportError, FileNotFoundError, TypeError):
        # Fallback: resolve relative to this file
        fallback = Path(__file__).resolve().parent.parent / "data" / filename
        if not fallback.exists():
            _dictionaries[lang_upper] = frozenset()
            return _dictionaries[lang_upper]
        text = fallback.read_text(encoding="utf-8")

    words: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            words.add(stripped.lower())

    _dictionaries[lang_upper] = frozenset(words)
    return _dictionaries[lang_upper]


def _load_spanish_dictionary() -> frozenset[str]:
    """Load the curated Spanish dictionary. Convenience wrapper for backwards compat."""
    return _load_dictionary("ES")

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

# Word boundary regex for splitting (Latin script — covers ES, FR, DE, IT, PT, PL)
_WORD_RE = re.compile(
    r"[a-záéíóúñüàâæçèêëïîôœùûÿäößẞìíîòãõąćęłńśźż]+",
    re.IGNORECASE,
)

# Characters unique to specific target languages — used to detect already-translated text.
# Each pattern only matches characters that are unambiguous for that language.
#   RU → Cyrillic block (also covers other Slavic Cyrillic scripts)
#   DE → ß/ẞ (only in German among all supported languages)
#   PT → ã/õ (nasal vowels exclusive to Portuguese in Latin script)
#   PL → Polish diacritics not shared with other supported languages
_LANG_UNIQUE_CHARS: dict[str, re.Pattern[str]] = {
    "RU": re.compile(r"[\u0400-\u04FF]"),
    "DE": re.compile(r"[ßẞ]"),
    "PT": re.compile(r"[ãõÃÕ]"),
    "PL": re.compile(r"[ąęśćźżńłĄĘŚĆŹŻŃŁ]"),
}


def is_target_language(text: str, lang: str) -> bool:
    """Determine if a text string is likely in the given target language.

    Generic version of is_spanish() that works for all supported languages.
    Uses a scoring heuristic based on:
    - Unique characters (Cyrillic, ß, ã/õ, Polish diacritics) — fast strong signal
    - Language-specific special characters (accents)
    - Common/strong/weak word lists per language
    - Curated dictionary fallback
    - Short string handling (1-2 words)

    Returns True if the text scores above the detection threshold.
    Short strings (< 2 chars) always return False.
    """
    if len(text) < _MIN_LENGTH:
        return False

    lang_upper = lang.upper()
    lower = text.lower()

    # Fast path: unique characters for RU/DE/PT/PL
    char_pattern = _LANG_UNIQUE_CHARS.get(lang_upper)
    if char_pattern and char_pattern.search(text):
        return True

    # Select word regex (Cyrillic for RU, Latin for others)
    word_re = _LANG_WORD_RE.get(lang_upper, _WORD_RE)
    words = word_re.findall(lower)

    if not words:
        return False

    score = 0.0
    word_count = len(words)
    dictionary = _load_dictionary(lang_upper)

    # Get per-language data
    special_chars = _LANG_SPECIAL_CHARS.get(lang_upper, set())
    common_words = _LANG_COMMON_WORDS.get(lang_upper, set())
    strong_words = _LANG_STRONG_WORDS.get(lang_upper, set())
    weak_words = _LANG_WEAK_WORDS.get(lang_upper, set())

    # Check for language-specific special characters
    char_count = sum(1 for c in lower if c in special_chars) if special_chars else 0
    if char_count > 0:
        score += min(char_count * 0.3, 1.5)

    # For short strings (1-2 words), check against dictionary + word lists
    if word_count <= 2:
        for w in words:
            if w in dictionary:
                score += 0.6
            if w in strong_words:
                score += 0.8
            elif w in common_words:
                score += 0.3
        return score >= 0.6

    # Check for target language words (3+ words)
    lang_word_hits = 0
    core_hits = 0
    for w in words:
        if w in strong_words:
            score += 0.8
            lang_word_hits += 1
            core_hits += 1
        elif w in common_words:
            if w in weak_words:
                score += 0.1
                lang_word_hits += 1
            else:
                score += 0.4
                lang_word_hits += 1
                core_hits += 1
        elif w in dictionary:
            score += 0.2
            lang_word_hits += 1

    # Ratio of target language words to total words
    if word_count > 2:
        ratio = lang_word_hits / word_count
        if core_hits > 0 and ratio > 0.3:
            score += 1.0
        elif core_hits > 0 and ratio > 0.15:
            score += 0.5
        elif ratio > 0.5:
            score += 0.8
        elif ratio > 0.3:
            score += 0.4

    return score >= 1.0


def is_spanish(text: str) -> bool:
    """Determine if a text string is likely Spanish.

    Convenience wrapper around is_target_language for backwards compat.
    """
    return is_target_language(text, "ES")


def is_english(text: str) -> bool:
    """Determine if a text string is likely English.

    Mirrors is_spanish() logic but using English word lists.
    Returns True if the text has strong English signals.
    Short strings (< 2 chars) always return False.
    """
    if len(text) < _MIN_LENGTH:
        return False

    lower = text.lower()
    words = _WORD_RE.findall(lower)

    if not words:
        return False

    score = 0.0
    word_count = len(words)

    # Presence of non-English accented characters is negative evidence for English
    accent_count = sum(1 for c in lower if c in _NON_ENGLISH_CHARS)
    if accent_count > 0:
        score -= min(accent_count * 0.5, 2.0)

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


def get_target_words_in_text(text: str, lang: str) -> list[tuple[str, int, int]]:
    """Find words of the target language in a text string.

    Returns a list of (word, start, end) tuples for each word that is
    either in the target language dictionary or contains language-specific
    special characters. Only words of 4+ characters are considered.

    Works for all supported languages: ES, FR, DE, IT, PT, RU, PL.
    """
    lang_upper = lang.upper()
    dictionary = _load_dictionary(lang_upper)
    special_chars = _LANG_SPECIAL_CHARS.get(lang_upper, set())
    word_re = _LANG_WORD_RE.get(lang_upper, _WORD_RE)
    results: list[tuple[str, int, int]] = []

    for match in word_re.finditer(text):
        word = match.group()
        if len(word) < 4:
            continue
        lower = word.lower()
        has_special = bool(special_chars) and any(c in special_chars for c in lower)
        if lower in dictionary or has_special:
            results.append((word, match.start(), match.end()))

    return results


def get_spanish_words_in_text(text: str) -> list[tuple[str, int, int]]:
    """Find Spanish words in a text string.

    Convenience wrapper around get_target_words_in_text for backwards compat.
    """
    return get_target_words_in_text(text, "ES")


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
    2. is_target_language() — heuristic with dictionary and short-string support
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

    # Layer 2: is_target_language() — heuristic (handles short strings too)
    if is_target_language(stripped, target_lang):
        return False

    # Layer 3: is_english() — positive confirmation for English
    if is_english(stripped):
        return True

    # Ambiguous — for short strings (1-2 words), only skip if there was some
    # target language signal. If neither is_target_language nor is_english fired,
    # the word is likely an English noun/name (e.g. "Iron Sword") that should
    # be translated.
    return True
