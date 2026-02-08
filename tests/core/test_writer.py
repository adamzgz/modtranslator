"""Tests for the binary writer."""

import io

from modtranslator.core.parser import parse_plugin
from modtranslator.core.writer import write_plugin
from tests.conftest import make_plugin, make_subrecord


class TestWriter:
    def test_write_minimal_plugin(self):
        plugin = make_plugin([
            ("WEAP", 0x00001000, [
                make_subrecord("EDID", "TestWeapon"),
                make_subrecord("FULL", "Iron Sword"),
            ]),
        ])

        buf = io.BytesIO()
        write_plugin(plugin, buf)
        data = buf.getvalue()

        assert data[:4] == b"TES4"
        assert len(data) > 0

    def test_written_plugin_is_parseable(self):
        plugin = make_plugin([
            ("WEAP", 0x00001000, [
                make_subrecord("EDID", "TestWeapon"),
                make_subrecord("FULL", "Iron Sword"),
            ]),
        ])

        buf = io.BytesIO()
        write_plugin(plugin, buf)

        buf.seek(0)
        reparsed = parse_plugin(buf)

        assert reparsed.header.type == b"TES4"
        assert len(reparsed.groups) == 1
        assert len(reparsed.groups[0].children) == 1
        rec = reparsed.groups[0].children[0]
        assert rec.type == b"WEAP"

    def test_subrecord_size_updated_after_mutation(self):
        sub = make_subrecord("FULL", "Short")
        original_size = sub.size

        sub.encode_string("A much longer string that should increase size")
        assert sub.size > original_size

    def test_writer_uses_updated_sizes(self):
        plugin = make_plugin([
            ("WEAP", 0x00001000, [
                make_subrecord("FULL", "Short"),
            ]),
        ])

        # Mutate
        plugin.groups[0].children[0].subrecords[0].encode_string("Much longer text here")

        buf = io.BytesIO()
        write_plugin(plugin, buf)

        buf.seek(0)
        reparsed = parse_plugin(buf)

        full_sub = reparsed.groups[0].children[0].subrecords[0]
        assert full_sub.decode_string() == "Much longer text here"
