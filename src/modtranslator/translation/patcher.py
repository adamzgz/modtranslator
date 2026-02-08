"""Apply translations back to the plugin's subrecord data."""

from __future__ import annotations

from modtranslator.core.constants import DEFAULT_ENCODING
from modtranslator.translation.extractor import TranslatableString


def apply_translations(
    strings: list[TranslatableString],
    translations: dict[str, str],
    encoding: str = DEFAULT_ENCODING,
) -> int:
    """Mutate subrecord data with translated text.

    Args:
        strings: List of TranslatableString from the extractor.
        translations: Mapping of TranslatableString.key → translated text.
        encoding: Target encoding for the string data.

    Returns:
        Number of strings patched.
    """
    patched = 0

    for ts in strings:
        translated = translations.get(ts.key)
        if translated is None:
            continue

        # Skip if translation is identical to original
        if translated == ts.original_text:
            continue

        # Encode and write to the subrecord's mutable bytearray
        ts.subrecord.encode_string(translated, encoding)
        patched += 1

    return patched
