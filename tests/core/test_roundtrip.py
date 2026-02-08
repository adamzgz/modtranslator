"""Roundtrip tests: parse → write → parse should produce identical results."""

import io
from pathlib import Path

from modtranslator.core.parser import parse_plugin
from modtranslator.core.writer import write_plugin

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
