"""Tests for string table parsing, serialization, and loading."""

from __future__ import annotations

import struct
from pathlib import Path

from modtranslator.core.string_table import (
    StringTable,
    StringTableSet,
    StringTableType,
    load_string_tables,
    parse_string_table,
    save_string_tables,
    serialize_string_table,
)


def _build_strings_binary(entries: list[tuple[int, str]]) -> bytes:
    """Build a raw STRINGS binary (null-terminated, no length prefix)."""
    count = len(entries)
    data_parts: list[bytes] = []
    directory: list[tuple[int, int]] = []
    offset = 0
    for sid, text in entries:
        encoded = text.encode("utf-8") + b"\x00"
        directory.append((sid, offset))
        data_parts.append(encoded)
        offset += len(encoded)

    data_block = b"".join(data_parts)
    header = struct.pack("<II", count, len(data_block))
    dir_bytes = b"".join(struct.pack("<II", sid, off) for sid, off in directory)
    return header + dir_bytes + data_block


def _build_dlstrings_binary(entries: list[tuple[int, str]]) -> bytes:
    """Build a raw DLSTRINGS/ILSTRINGS binary (length-prefixed)."""
    count = len(entries)
    data_parts: list[bytes] = []
    directory: list[tuple[int, int]] = []
    offset = 0
    for sid, text in entries:
        encoded = text.encode("utf-8") + b"\x00"
        length = len(encoded)
        part = struct.pack("<I", length) + encoded
        directory.append((sid, offset))
        data_parts.append(part)
        offset += len(part)

    data_block = b"".join(data_parts)
    header = struct.pack("<II", count, len(data_block))
    dir_bytes = b"".join(struct.pack("<II", sid, off) for sid, off in directory)
    return header + dir_bytes + data_block


class TestParseStringTable:
    def test_parse_strings_basic(self):
        raw = _build_strings_binary([(1, "Hello"), (2, "World")])
        table = parse_string_table(raw, StringTableType.STRINGS)
        assert table.entries[1] == "Hello"
        assert table.entries[2] == "World"
        assert table.table_type == StringTableType.STRINGS

    def test_parse_dlstrings_basic(self):
        raw = _build_dlstrings_binary([(10, "Iron Sword"), (20, "Steel Armor")])
        table = parse_string_table(raw, StringTableType.DLSTRINGS)
        assert table.entries[10] == "Iron Sword"
        assert table.entries[20] == "Steel Armor"

    def test_parse_ilstrings_basic(self):
        raw = _build_dlstrings_binary([(100, "Description text")])
        table = parse_string_table(raw, StringTableType.ILSTRINGS)
        assert table.entries[100] == "Description text"

    def test_parse_empty_table(self):
        raw = struct.pack("<II", 0, 0)
        table = parse_string_table(raw, StringTableType.STRINGS)
        assert len(table.entries) == 0

    def test_parse_utf8_characters(self):
        raw = _build_strings_binary([(1, "Espada de Hierro"), (2, "café ñoño")])
        table = parse_string_table(raw, StringTableType.STRINGS)
        assert table.entries[1] == "Espada de Hierro"
        assert table.entries[2] == "café ñoño"

    def test_parse_too_short_data(self):
        table = parse_string_table(b"\x00\x00", StringTableType.STRINGS)
        assert len(table.entries) == 0


class TestSerializeStringTable:
    def test_serialize_roundtrip_strings(self):
        original = StringTable(
            table_type=StringTableType.STRINGS,
            entries={1: "Hello", 5: "World", 3: "Test"},
        )
        data = serialize_string_table(original)
        parsed = parse_string_table(data, StringTableType.STRINGS)
        assert parsed.entries == original.entries

    def test_serialize_roundtrip_dlstrings(self):
        original = StringTable(
            table_type=StringTableType.DLSTRINGS,
            entries={10: "Iron Sword", 20: "A long description of the sword."},
        )
        data = serialize_string_table(original)
        parsed = parse_string_table(data, StringTableType.DLSTRINGS)
        assert parsed.entries == original.entries

    def test_serialize_roundtrip_ilstrings(self):
        original = StringTable(
            table_type=StringTableType.ILSTRINGS,
            entries={100: "Info line one", 200: "Info line two"},
        )
        data = serialize_string_table(original)
        parsed = parse_string_table(data, StringTableType.ILSTRINGS)
        assert parsed.entries == original.entries

    def test_serialize_empty(self):
        table = StringTable(table_type=StringTableType.STRINGS)
        data = serialize_string_table(table)
        assert data == struct.pack("<II", 0, 0)

    def test_serialize_utf8(self):
        original = StringTable(
            table_type=StringTableType.STRINGS,
            entries={1: "café", 2: "ñoño"},
        )
        data = serialize_string_table(original)
        parsed = parse_string_table(data, StringTableType.STRINGS)
        assert parsed.entries == original.entries

    def test_entries_sorted_by_id(self):
        """Serialized entries should be in ascending string_id order."""
        table = StringTable(
            table_type=StringTableType.STRINGS,
            entries={99: "Z", 1: "A", 50: "M"},
        )
        data = serialize_string_table(table)
        # Parse and check order via the directory
        count = struct.unpack_from("<I", data, 0)[0]
        assert count == 3
        ids = [struct.unpack_from("<I", data, 8 + i * 8)[0] for i in range(count)]
        assert ids == [1, 50, 99]


class TestStringTableSet:
    def test_build_merged(self):
        sts = StringTableSet()
        sts.strings.entries = {1: "Name1", 2: "Name2"}
        sts.dlstrings.entries = {10: "Desc1"}
        sts.ilstrings.entries = {100: "Info1"}
        sts.build_merged()
        assert sts.merged == {1: "Name1", 2: "Name2", 10: "Desc1", 100: "Info1"}


class TestLoadSaveStringTables:
    def test_load_from_strings_dir(self, tmp_path: Path):
        """String tables found in a 'strings' subdirectory."""
        plugin_path = tmp_path / "MyMod.esp"
        plugin_path.write_bytes(b"")  # dummy

        strings_dir = tmp_path / "strings"
        strings_dir.mkdir()

        raw_strings = _build_strings_binary([(1, "Iron Sword")])
        (strings_dir / "MyMod_English.strings").write_bytes(raw_strings)

        raw_dl = _build_dlstrings_binary([(10, "A fine sword.")])
        (strings_dir / "MyMod_English.dlstrings").write_bytes(raw_dl)

        raw_il = _build_dlstrings_binary([(100, "Info text")])
        (strings_dir / "MyMod_English.ilstrings").write_bytes(raw_il)

        ts = load_string_tables(plugin_path)
        assert ts.strings.entries[1] == "Iron Sword"
        assert ts.dlstrings.entries[10] == "A fine sword."
        assert ts.ilstrings.entries[100] == "Info text"
        assert ts.merged[1] == "Iron Sword"
        assert ts.merged[10] == "A fine sword."

    def test_load_next_to_plugin(self, tmp_path: Path):
        """String tables found next to the plugin file."""
        plugin_path = tmp_path / "MyMod.esp"
        plugin_path.write_bytes(b"")

        raw_strings = _build_strings_binary([(1, "Hello")])
        (tmp_path / "MyMod_English.strings").write_bytes(raw_strings)

        ts = load_string_tables(plugin_path)
        assert ts.strings.entries[1] == "Hello"
        assert 1 in ts.merged

    def test_load_missing_file_warns(self, tmp_path: Path, caplog):
        """Missing string table files produce warnings, not errors."""
        plugin_path = tmp_path / "MyMod.esp"
        plugin_path.write_bytes(b"")

        import logging
        with caplog.at_level(logging.WARNING):
            ts = load_string_tables(plugin_path)

        assert len(ts.merged) == 0
        assert any("not found" in r.message for r in caplog.records)

    def test_save_and_reload(self, tmp_path: Path):
        """Save string tables and reload them."""
        sts = StringTableSet()
        sts.strings = StringTable(StringTableType.STRINGS, {1: "Espada", 2: "Escudo"})
        sts.dlstrings = StringTable(StringTableType.DLSTRINGS, {10: "Una espada fina."})
        sts.ilstrings = StringTable(StringTableType.ILSTRINGS, {100: "Información"})

        plugin_path = tmp_path / "MyMod.esp"
        plugin_path.write_bytes(b"")

        written = save_string_tables(sts, plugin_path, language="Spanish")
        assert len(written) == 3

        # Reload
        reloaded = load_string_tables(plugin_path, language="Spanish")
        assert reloaded.strings.entries[1] == "Espada"
        assert reloaded.dlstrings.entries[10] == "Una espada fina."
        assert reloaded.ilstrings.entries[100] == "Información"
