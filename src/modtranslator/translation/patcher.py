"""Apply translations back to the plugin's subrecord data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modtranslator.core.constants import DEFAULT_ENCODING
from modtranslator.translation.extractor import TranslatableString

if TYPE_CHECKING:
    from modtranslator.core.string_table import StringTableSet


def apply_translations(
    strings: list[TranslatableString],
    translations: dict[str, str],
    encoding: str = DEFAULT_ENCODING,
    string_tables: StringTableSet | None = None,
) -> int:
    """Mutate subrecord data or string table entries with translated text.

    Args:
        strings: List of TranslatableString from the extractor.
        translations: Mapping of TranslatableString.key → translated text.
        encoding: Target encoding for inline string data.
        string_tables: If provided, localized strings are patched here instead of subrecords.

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

        if ts.string_id is not None and string_tables is not None:
            # Localized: update the string table entry
            for table in [string_tables.strings, string_tables.dlstrings, string_tables.ilstrings]:
                if ts.string_id in table.entries:
                    table.set(ts.string_id, translated)
                    break
            patched += 1
        else:
            # Inline: mutate the subrecord's bytearray
            ts.subrecord.encode_string(translated, encoding)
            patched += 1

    return patched
