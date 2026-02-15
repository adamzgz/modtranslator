"""Parse, manipulate, and serialize Skyrim string table files (.STRINGS, .DLSTRINGS, .ILSTRINGS).

String table format (all little-endian):
  Header:    [count: u32] [data_size: u32]
  Directory: [string_id: u32] [offset: u32] × count   (offsets relative to data block start)
  Data:
    STRINGS:     null-terminated UTF-8 strings
    IL/DLSTRINGS: [length: u32 (includes null)] [string: null-terminated]
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class StringTableType(Enum):
    """The three string table file types."""
    STRINGS = "STRINGS"
    DLSTRINGS = "DLSTRINGS"
    ILSTRINGS = "ILSTRINGS"


@dataclass
class StringTable:
    """A single string table (one of the three types)."""
    table_type: StringTableType
    entries: dict[int, str] = field(default_factory=dict)

    def set(self, string_id: int, text: str) -> None:
        self.entries[string_id] = text


@dataclass
class StringTableSet:
    """The three string tables for a localized plugin, plus a merged lookup dict."""
    strings: StringTable = field(default_factory=lambda: StringTable(StringTableType.STRINGS))
    dlstrings: StringTable = field(default_factory=lambda: StringTable(StringTableType.DLSTRINGS))
    ilstrings: StringTable = field(default_factory=lambda: StringTable(StringTableType.ILSTRINGS))
    merged: dict[int, str] = field(default_factory=dict)

    def build_merged(self) -> None:
        """Rebuild the merged lookup dict from all three tables."""
        self.merged = {}
        self.merged.update(self.strings.entries)
        self.merged.update(self.dlstrings.entries)
        self.merged.update(self.ilstrings.entries)


def _decode_string(raw: bytes) -> str:
    """Decode a string from a string table. UTF-8 primary, cp1252 fallback."""
    raw = raw.rstrip(b"\x00")
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def parse_string_table(data: bytes, table_type: StringTableType) -> StringTable:
    """Parse a binary string table into a StringTable.

    Args:
        data: Raw bytes of the .STRINGS/.DLSTRINGS/.ILSTRINGS file.
        table_type: Which type of table this is.

    Returns:
        Parsed StringTable with string_id → text entries.
    """
    if len(data) < 8:
        return StringTable(table_type=table_type)

    count, data_size = struct.unpack_from("<II", data, 0)

    # Directory starts at offset 8, each entry is 8 bytes (id + offset)
    dir_size = count * 8
    if len(data) < 8 + dir_size:
        return StringTable(table_type=table_type)

    # Data block starts after header + directory
    data_block_start = 8 + dir_size

    entries: dict[int, str] = {}

    for i in range(count):
        dir_offset = 8 + i * 8
        string_id, str_offset = struct.unpack_from("<II", data, dir_offset)

        abs_offset = data_block_start + str_offset

        if table_type == StringTableType.STRINGS:
            # Null-terminated string
            end = data.find(b"\x00", abs_offset)
            if end == -1:
                end = len(data)
            raw = data[abs_offset:end]
        else:
            # DLSTRINGS / ILSTRINGS: length-prefixed
            if abs_offset + 4 > len(data):
                continue
            length = struct.unpack_from("<I", data, abs_offset)[0]
            raw_start = abs_offset + 4
            raw_end = raw_start + length
            if raw_end > len(data):
                raw_end = len(data)
            raw = data[raw_start:raw_end]

        entries[string_id] = _decode_string(raw)

    return StringTable(table_type=table_type, entries=entries)


def serialize_string_table(table: StringTable) -> bytes:
    """Serialize a StringTable back to binary format.

    Entries are written in ascending string_id order for determinism.
    """
    sorted_ids = sorted(table.entries.keys())
    count = len(sorted_ids)

    if count == 0:
        return struct.pack("<II", 0, 0)

    # Build data block and directory simultaneously
    data_parts: list[bytes] = []
    directory: list[tuple[int, int]] = []  # (string_id, offset)
    current_offset = 0

    for sid in sorted_ids:
        text = table.entries[sid]
        encoded = text.encode("utf-8") + b"\x00"

        directory.append((sid, current_offset))

        if table.table_type == StringTableType.STRINGS:
            data_parts.append(encoded)
            current_offset += len(encoded)
        else:
            # DLSTRINGS / ILSTRINGS: length prefix (includes null)
            length = len(encoded)
            data_parts.append(struct.pack("<I", length) + encoded)
            current_offset += 4 + length

    data_block = b"".join(data_parts)

    # Header
    header = struct.pack("<II", count, len(data_block))

    # Directory
    dir_bytes = b"".join(struct.pack("<II", sid, off) for sid, off in directory)

    return header + dir_bytes + data_block


def load_string_tables(
    plugin_path: str | Path,
    language: str = "English",
) -> StringTableSet:
    """Load the three string table files for a localized plugin.

    Looks for {stem}_{language}.STRINGS, .DLSTRINGS, .ILSTRINGS
    next to the plugin file.

    Args:
        plugin_path: Path to the ESP/ESM file.
        language: Language suffix (default "English").

    Returns:
        StringTableSet with all found tables loaded and merged dict built.
    """
    plugin_path = Path(plugin_path)
    stem = plugin_path.stem
    parent = plugin_path.parent

    # Bethesda convention: string files go in a "strings" subdirectory
    # but some mods put them next to the plugin. Check both.
    search_dirs = [parent / "strings", parent / "Strings", parent]

    table_set = StringTableSet()

    type_map = {
        StringTableType.STRINGS: "STRINGS",
        StringTableType.DLSTRINGS: "DLSTRINGS",
        StringTableType.ILSTRINGS: "ILSTRINGS",
    }
    # Also try lowercase extensions (some mods/tests use them)
    ext_variants = {
        "STRINGS": ["STRINGS", "strings"],
        "DLSTRINGS": ["DLSTRINGS", "dlstrings"],
        "ILSTRINGS": ["ILSTRINGS", "ilstrings"],
    }
    attr_map = {
        StringTableType.STRINGS: "strings",
        StringTableType.DLSTRINGS: "dlstrings",
        StringTableType.ILSTRINGS: "ilstrings",
    }

    for tt, attr_name in attr_map.items():
        ext = type_map[tt]
        variants = ext_variants[ext]
        found = False
        for search_dir in search_dirs:
            for ext_v in variants:
                filepath = search_dir / f"{stem}_{language}.{ext_v}"
                if filepath.exists():
                    raw = filepath.read_bytes()
                    table = parse_string_table(raw, tt)
                    setattr(table_set, attr_name, table)
                    found = True
                    break
            if found:
                break
        if not found:
            logger.warning("String table file not found: %s_%s.%s", stem, language, ext)

    table_set.build_merged()
    return table_set


def save_string_tables(
    tables: StringTableSet,
    plugin_path: str | Path,
    language: str = "Spanish",
) -> list[Path]:
    """Write the three string table files next to the output plugin.

    Args:
        tables: The StringTableSet to write.
        plugin_path: Path to the output ESP/ESM file.
        language: Language suffix for output filenames.

    Returns:
        List of paths written.
    """
    plugin_path = Path(plugin_path)
    stem = plugin_path.stem
    parent = plugin_path.parent

    # Write to "strings" subdirectory if it exists, otherwise next to plugin
    strings_dir = parent / "strings"
    if not strings_dir.exists():
        strings_dir = parent / "Strings"
    if not strings_dir.exists():
        strings_dir = parent
    strings_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    type_map = {
        "strings": (tables.strings, "STRINGS"),
        "dlstrings": (tables.dlstrings, "DLSTRINGS"),
        "ilstrings": (tables.ilstrings, "ILSTRINGS"),
    }

    for _, (table, ext) in type_map.items():
        if not table.entries:
            continue
        filename = f"{stem}_{language}.{ext}"
        filepath = strings_dir / filename
        filepath.write_bytes(serialize_string_table(table))
        written.append(filepath)

    return written
