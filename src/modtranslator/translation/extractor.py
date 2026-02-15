"""Extract translatable strings from a parsed plugin file."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from modtranslator.core.constants import DEFAULT_ENCODING
from modtranslator.core.records import GroupRecord, PluginFile, Record, Subrecord
from modtranslator.translation.registry import is_translatable

_RE_HEX_PREFIX = re.compile(r"^\d{2}[A-Za-z]")
_RE_CAMEL_CASE = re.compile(r"[a-z][A-Z]")


@dataclass
class TranslatableString:
    """A translatable string with a direct reference to its source subrecord.

    Mutating `subrecord.data` is how translations are applied.
    """

    record_type: bytes  # Parent record type, e.g. b"WEAP"
    subrecord_type: bytes  # e.g. b"FULL"
    form_id: int  # FormID of the parent record
    original_text: str  # Decoded original string
    subrecord: Subrecord  # Direct reference for mutation by patcher
    editor_id: str = ""  # EDID of parent record, for context
    sub_index: int = 0  # Occurrence index within parent record (for duplicate sub types)
    source_file: str = ""  # Source filename (stem) to disambiguate formIDs across mods
    string_id: int | None = None  # StringID for localized plugins (None = inline)

    @property
    def key(self) -> str:
        """Unique key for caching: source file + formID + subrecord type + index.

        FormIDs are relative to a plugin's master list, so two mods with the
        same single master (e.g. Fallout3.esm) can have overlapping formIDs
        (both start at 01xxxxxx).  Including the source filename prevents
        cache collisions between different mods.
        """
        prefix = f"{self.source_file}:" if self.source_file else ""
        return f"{prefix}{self.form_id:08X}:{self.subrecord_type.decode('ascii')}:{self.sub_index}"


def _looks_like_editor_id(text: str) -> bool:
    """Heuristic: return True if *text* looks like an internal editor ID.

    Editor IDs are CamelCase identifiers without spaces (e.g.
    ``00MSAOlevRescueTopic03A``, ``DrNylusDeathRay``, ``Player_Fallback``).
    Translating them breaks scripts, quests and causes CTDs.

    False-positives (``RadAway``, ``StealthBoy``) are acceptable — those
    are proper nouns that the official localisation keeps untranslated.
    """
    if " " in text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if _RE_HEX_PREFIX.match(stripped):
        return True
    if _RE_CAMEL_CASE.search(stripped):
        return True
    if "_" in stripped:
        return True
    return bool(len(stripped) > 8 and stripped.isalnum() and not stripped.isalpha())


def extract_strings(plugin: PluginFile) -> list[TranslatableString]:
    """Walk the plugin tree and extract all translatable strings."""
    results: list[TranslatableString] = []
    merged = plugin.string_tables.merged if plugin.string_tables is not None else None

    # TES4 header may have translatable strings too (FULL for master name)
    _extract_from_record(plugin.header, results, merged)

    for group in plugin.groups:
        _extract_from_group(group, results, merged)

    return results


def _extract_from_group(
    group: GroupRecord,
    results: list[TranslatableString],
    merged: dict[int, str] | None = None,
) -> None:
    for child in group.children:
        if isinstance(child, GroupRecord):
            _extract_from_group(child, results, merged)
        else:
            _extract_from_record(child, results, merged)


def _extract_from_record(
    record: Record,
    results: list[TranslatableString],
    merged: dict[int, str] | None = None,
) -> None:
    # Get EDID for context
    editor_id = ""
    for sub in record.subrecords:
        if sub.type == b"EDID":
            editor_id = sub.decode_string(DEFAULT_ENCODING)
            break

    # Track occurrence index per subrecord type within this record.
    # Records like MESG (multiple ITXT buttons), QUST (multiple CNAM/NNAM
    # stages/objectives), and INFO (multiple NAM1 response pages) can have
    # several translatable subrecords of the same type.  Without an index
    # they'd all share the same cache/translation key, causing the last
    # translation to overwrite all previous ones.
    sub_type_counter: dict[bytes, int] = {}

    for sub in record.subrecords:
        if not is_translatable(record.type, sub.type):
            continue

        # Skip empty or binary-only subrecords
        if sub.size == 0:
            continue

        string_id: int | None = None

        # Localized plugin: subrecords of exactly 4 bytes contain a uint32 StringID
        if merged is not None and sub.size == 4:
            string_id = struct.unpack_from("<I", sub.data, 0)[0]
            if string_id == 0:
                continue  # StringID 0 = no string
            text = merged.get(string_id)
            if text is None:
                continue  # ID not found in string tables
        else:
            # Inline: decode from the subrecord (FO3 behavior, or non-localized Skyrim)
            text = sub.decode_string(DEFAULT_ENCODING)

        # Skip strings that are empty or look like binary garbage
        if not text or not text.strip():
            continue

        # Skip binary data masquerading as text.  Two layers:
        # 1) Pure control chars: no printable character at all.
        if not any(c >= " " for c in text):
            continue
        # 2) Short strings (≤3 chars) with control characters — these are
        #    almost certainly 4-byte FormID references (e.g. TNAM in NOTE,
        #    RNAM in TERM) where the trailing null was stripped, leaving 3
        #    bytes that partially decode to printable cp1252 glyphs.
        #    Real 1-3 character text (e.g. "Hi", "No") never has ctrl chars.
        if len(text) <= 3 and any(ord(c) < 0x20 for c in text):
            continue

        if _looks_like_editor_id(text):
            continue

        idx = sub_type_counter.get(sub.type, 0)
        sub_type_counter[sub.type] = idx + 1

        results.append(
            TranslatableString(
                record_type=record.type,
                subrecord_type=sub.type,
                form_id=record.form_id,
                original_text=text,
                subrecord=sub,
                editor_id=editor_id,
                sub_index=idx,
                string_id=string_id,
            )
        )
