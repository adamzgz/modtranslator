"""Tests for the plugin facade (load_plugin, save_plugin, plugin_to_bytes, plugin_from_bytes)."""

from __future__ import annotations

import io

from modtranslator.core.plugin import load_plugin, plugin_from_bytes, plugin_to_bytes, save_plugin
from modtranslator.core.writer import write_plugin
from tests.conftest import make_plugin, make_subrecord


class TestPluginToBytes:
    def test_roundtrip_bytes(self):
        plugin = make_plugin([
            ("WEAP", 0x1000, [make_subrecord("FULL", "Iron Sword")]),
        ])
        data = plugin_to_bytes(plugin)
        assert data[:4] == b"TES4"
        assert len(data) > 0

    def test_bytes_match_stream_writer(self):
        plugin = make_plugin([
            ("WEAP", 0x1000, [make_subrecord("FULL", "Iron Sword")]),
        ])
        data = plugin_to_bytes(plugin)
        buf = io.BytesIO()
        write_plugin(plugin, buf)
        assert data == buf.getvalue()


class TestPluginFromBytes:
    def test_parse_from_bytes(self):
        plugin = make_plugin([
            ("ARMO", 0x2000, [
                make_subrecord("EDID", "TestArmor"),
                make_subrecord("FULL", "Leather Armor"),
            ]),
        ])
        data = plugin_to_bytes(plugin)
        reparsed = plugin_from_bytes(data)
        assert reparsed.header.type == b"TES4"
        assert len(reparsed.groups) == 1
        rec = reparsed.groups[0].children[0]
        assert rec.type == b"ARMO"

    def test_roundtrip_preserves_strings(self):
        plugin = make_plugin([
            ("BOOK", 0x3000, [
                make_subrecord("FULL", "Wasteland Guide"),
                make_subrecord("DESC", "A helpful book."),
            ]),
        ])
        data = plugin_to_bytes(plugin)
        reparsed = plugin_from_bytes(data)
        subs = reparsed.groups[0].children[0].subrecords
        assert subs[0].decode_string() == "Wasteland Guide"
        assert subs[1].decode_string() == "A helpful book."


class TestLoadSavePlugin:
    def test_load_and_save(self, tmp_path):
        # Write a binary plugin to disk
        plugin = make_plugin([
            ("WEAP", 0x1000, [make_subrecord("FULL", "Laser Rifle")]),
        ])
        source = tmp_path / "test.esp"
        with open(source, "wb") as f:
            write_plugin(plugin, f)

        # Load it back
        loaded = load_plugin(source)
        assert loaded.header.type == b"TES4"
        rec = loaded.groups[0].children[0]
        assert rec.subrecords[0].decode_string() == "Laser Rifle"

        # Save it
        output = tmp_path / "output.esp"
        save_plugin(loaded, output)
        assert output.exists()

        # Verify the saved file is parseable
        reloaded = load_plugin(output)
        assert reloaded.groups[0].children[0].subrecords[0].decode_string() == "Laser Rifle"

    def test_save_plugin_with_string_path(self, tmp_path):
        plugin = make_plugin([
            ("WEAP", 0x1000, [make_subrecord("FULL", "Sword")]),
        ])
        source = tmp_path / "test.esp"
        with open(source, "wb") as f:
            write_plugin(plugin, f)

        loaded = load_plugin(str(source))
        output = str(tmp_path / "output.esp")
        save_plugin(loaded, output)

    def test_save_localized_plugin_writes_string_tables(self, tmp_path):
        """When a plugin has string_tables set, save_plugin writes them too."""
        from modtranslator.core.string_table import StringTable, StringTableSet, StringTableType

        plugin = make_plugin([
            ("WEAP", 0x1000, [make_subrecord("FULL", "Sword")]),
        ])
        st = StringTableSet()
        st.strings = StringTable(StringTableType.STRINGS, entries={1: "Hello"})
        st.dlstrings = StringTable(StringTableType.DLSTRINGS, entries={2: "World"})
        st.ilstrings = StringTable(StringTableType.ILSTRINGS, entries={})
        plugin.string_tables = st

        source = tmp_path / "Test.esp"
        with open(source, "wb") as f:
            write_plugin(plugin, f)

        save_plugin(plugin, source, output_language="Spanish")
        # Verify at least one string table file was created
        st_files = list(tmp_path.glob("Test_*.*STRINGS"))
        assert len(st_files) > 0
