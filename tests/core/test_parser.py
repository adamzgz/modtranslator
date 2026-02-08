"""Tests for the binary parser."""

from pathlib import Path

from modtranslator.core.constants import Game
from modtranslator.core.parser import parse_plugin

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestParserMinimalFO3:
    def test_parse_tes4_header(self):
        with open(FIXTURES / "minimal_fo3.esp", "rb") as f:
            plugin = parse_plugin(f)

        assert plugin.header.type == b"TES4"
        assert plugin.header.form_id == 0

    def test_detect_game_fo3(self):
        with open(FIXTURES / "minimal_fo3.esp", "rb") as f:
            plugin = parse_plugin(f)

        assert plugin.game == Game.FALLOUT3

    def test_parse_groups(self):
        with open(FIXTURES / "minimal_fo3.esp", "rb") as f:
            plugin = parse_plugin(f)

        assert len(plugin.groups) == 1
        group = plugin.groups[0]
        assert group.label == b"WEAP"

    def test_parse_record_in_group(self):
        with open(FIXTURES / "minimal_fo3.esp", "rb") as f:
            plugin = parse_plugin(f)

        group = plugin.groups[0]
        assert len(group.children) == 1
        record = group.children[0]
        assert record.type == b"WEAP"
        assert record.form_id == 0x00001000

    def test_parse_subrecords(self):
        with open(FIXTURES / "minimal_fo3.esp", "rb") as f:
            plugin = parse_plugin(f)

        record = plugin.groups[0].children[0]
        types = [s.type for s in record.subrecords]
        assert b"EDID" in types
        assert b"FULL" in types

    def test_subrecord_string_decode(self):
        with open(FIXTURES / "minimal_fo3.esp", "rb") as f:
            plugin = parse_plugin(f)

        record = plugin.groups[0].children[0]
        full_sub = next(s for s in record.subrecords if s.type == b"FULL")
        assert full_sub.decode_string() == "Iron Sword"


class TestParserMultiRecord:
    def test_parse_multiple_groups(self):
        with open(FIXTURES / "multi_record.esp", "rb") as f:
            plugin = parse_plugin(f)

        assert len(plugin.groups) == 3
        labels = [g.label for g in plugin.groups]
        assert b"WEAP" in labels
        assert b"ARMO" in labels
        assert b"BOOK" in labels

    def test_all_records_parsed(self):
        with open(FIXTURES / "multi_record.esp", "rb") as f:
            plugin = parse_plugin(f)

        all_records = []
        for group in plugin.groups:
            all_records.extend(group.children)

        assert len(all_records) == 3
        types = {r.type for r in all_records}
        assert types == {b"WEAP", b"ARMO", b"BOOK"}

    def test_desc_subrecord(self):
        with open(FIXTURES / "multi_record.esp", "rb") as f:
            plugin = parse_plugin(f)

        # Find the ARMO record
        armo = None
        for group in plugin.groups:
            for child in group.children:
                if child.type == b"ARMO":
                    armo = child
                    break

        assert armo is not None
        desc = next(s for s in armo.subrecords if s.type == b"DESC")
        assert desc.decode_string() == "A sturdy set of leather armor."


class TestParserCompressed:
    def test_parse_compressed_record(self):
        with open(FIXTURES / "compressed.esp", "rb") as f:
            plugin = parse_plugin(f)

        assert len(plugin.groups) == 1
        record = plugin.groups[0].children[0]
        assert record.type == b"WEAP"
        assert record.is_compressed

    def test_compressed_subrecords_accessible(self):
        with open(FIXTURES / "compressed.esp", "rb") as f:
            plugin = parse_plugin(f)

        record = plugin.groups[0].children[0]
        full_sub = next(s for s in record.subrecords if s.type == b"FULL")
        assert full_sub.decode_string() == "Plasma Rifle"


