"""Roundtrip tests: parse → write → parse should produce identical results."""

import io
from pathlib import Path

from modtranslator.core.parser import parse_plugin
from modtranslator.core.string_table import (
    StringTable,
    StringTableSet,
    StringTableType,
    serialize_string_table,
)
from modtranslator.core.writer import write_plugin
from modtranslator.translation.extractor import extract_strings
from modtranslator.translation.patcher import apply_translations

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _roundtrip_file(filepath: Path) -> None:
    """Parse a file, write it, reparse, and compare the two plugin structures."""
    with open(filepath, "rb") as f:
        plugin1 = parse_plugin(f)

    buf = io.BytesIO()
    write_plugin(plugin1, buf)

    buf.seek(0)
    plugin2 = parse_plugin(buf)

    # Compare headers
    assert plugin1.header.type == plugin2.header.type
    assert plugin1.header.flags == plugin2.header.flags
    assert plugin1.header.form_id == plugin2.header.form_id
    assert len(plugin1.header.subrecords) == len(plugin2.header.subrecords)

    for s1, s2 in zip(plugin1.header.subrecords, plugin2.header.subrecords, strict=False):
        assert s1.type == s2.type
        assert bytes(s1.data) == bytes(s2.data)

    # Compare groups
    assert len(plugin1.groups) == len(plugin2.groups)

    for g1, g2 in zip(plugin1.groups, plugin2.groups, strict=False):
        _compare_groups(g1, g2)


def _compare_groups(g1, g2) -> None:
    assert g1.label == g2.label
    assert g1.group_type == g2.group_type
    assert len(g1.children) == len(g2.children)

    for c1, c2 in zip(g1.children, g2.children, strict=False):
        if hasattr(c1, "label"):  # GroupRecord
            _compare_groups(c1, c2)
        else:
            _compare_records(c1, c2)


def _compare_records(r1, r2) -> None:
    assert r1.type == r2.type
    assert r1.form_id == r2.form_id
    assert r1.flags == r2.flags
    assert len(r1.subrecords) == len(r2.subrecords)

    for s1, s2 in zip(r1.subrecords, r2.subrecords, strict=False):
        assert s1.type == s2.type
        assert bytes(s1.data) == bytes(s2.data)


class TestRoundtrip:
    def test_minimal_fo3(self):
        _roundtrip_file(FIXTURES / "minimal_fo3.esp")

    def test_multi_record(self):
        _roundtrip_file(FIXTURES / "multi_record.esp")

    def test_compressed(self):
        """Compressed records should roundtrip with identical subrecord data."""
        _roundtrip_file(FIXTURES / "compressed.esp")


class TestByteRoundtrip:
    """Byte-level roundtrip for uncompressed files."""

    def test_minimal_fo3_bytes(self):
        original = (FIXTURES / "minimal_fo3.esp").read_bytes()

        with open(FIXTURES / "minimal_fo3.esp", "rb") as f:
            plugin = parse_plugin(f)

        buf = io.BytesIO()
        write_plugin(plugin, buf)
        rewritten = buf.getvalue()

        assert rewritten == original

    def test_multi_record_bytes(self):
        original = (FIXTURES / "multi_record.esp").read_bytes()

        with open(FIXTURES / "multi_record.esp", "rb") as f:
            plugin = parse_plugin(f)

        buf = io.BytesIO()
        write_plugin(plugin, buf)
        rewritten = buf.getvalue()

        assert rewritten == original


class TestSkyrimRoundtrip:
    def test_skyrim_inline_roundtrip(self):
        """Skyrim inline plugin roundtrips correctly."""
        filepath = FIXTURES / "skyrim_inline.esp"
        if not filepath.exists():
            import pytest
            pytest.skip("Fixture not generated yet")
        _roundtrip_file(filepath)

    def test_skyrim_inline_bytes(self):
        """Skyrim inline plugin is byte-identical after roundtrip."""
        filepath = FIXTURES / "skyrim_inline.esp"
        if not filepath.exists():
            import pytest
            pytest.skip("Fixture not generated yet")
        original = filepath.read_bytes()
        with open(filepath, "rb") as f:
            plugin = parse_plugin(f)
        buf = io.BytesIO()
        write_plugin(plugin, buf)
        assert buf.getvalue() == original

    def test_localized_extract_patch_verify(self):
        """End-to-end: localized plugin → extract → translate → patch → verify string tables."""
        from tests.conftest import (
            make_skyrim_plugin,
            make_string_id_subrecord,
            make_subrecord,
        )

        sts = StringTableSet()
        sts.strings = StringTable(StringTableType.STRINGS, {42: "Iron Sword", 43: "Steel Mace"})
        sts.dlstrings = StringTable(StringTableType.DLSTRINGS, {100: "A fine weapon."})
        sts.build_merged()

        plugin = make_skyrim_plugin(
            records=[
                ("WEAP", 0x100, [
                    make_subrecord("EDID", "Sword"),
                    make_string_id_subrecord("FULL", 42),
                ]),
                ("WEAP", 0x101, [
                    make_subrecord("EDID", "Mace"),
                    make_string_id_subrecord("FULL", 43),
                    make_string_id_subrecord("DESC", 100),
                ]),
            ],
            localized=True,
            string_tables=sts,
        )

        # Extract
        strings = extract_strings(plugin)
        assert len(strings) == 3

        # Translate
        translations = {s.key: f"[ES] {s.original_text}" for s in strings}
        patched = apply_translations(strings, translations, string_tables=sts)
        assert patched == 3

        # Verify string tables updated
        assert sts.strings.entries[42] == "[ES] Iron Sword"
        assert sts.strings.entries[43] == "[ES] Steel Mace"
        assert sts.dlstrings.entries[100] == "[ES] A fine weapon."

        # Verify string tables serialize and re-parse correctly
        raw = serialize_string_table(sts.strings)
        from modtranslator.core.string_table import parse_string_table
        reparsed = parse_string_table(raw, StringTableType.STRINGS)
        assert reparsed.entries[42] == "[ES] Iron Sword"
