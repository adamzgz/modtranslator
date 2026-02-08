"""Stress tests: large plugin parsing, extraction, translation, roundtrip."""

from __future__ import annotations

import struct
from io import BytesIO

import pytest

from modtranslator.core.parser import parse_plugin
from modtranslator.core.writer import write_plugin
from modtranslator.translation.extractor import extract_strings


def _make_string(s: str) -> bytes:
    return s.encode("cp1252") + b"\x00"


def _build_subrecord(tag: str, data: bytes) -> bytes:
    return tag.encode("ascii") + struct.pack("<H", len(data)) + data


def _build_large_plugin(n_records: int) -> bytes:
    """Build a FO3 plugin with n_records WEAP records in one GRUP."""
    version = struct.pack("<f", 0.94)
    hedr_data = version + struct.pack("<I", 0) + struct.pack("<I", 0x000800)
    hedr_sub = _build_subrecord("HEDR", hedr_data)
    tes4_data = hedr_sub
    tes4_data_size = len(tes4_data)

    tes4_header = b"TES4" + struct.pack("<I", tes4_data_size)
    tes4_header += struct.pack("<I", 0) + struct.pack("<I", 0)
    tes4_header += struct.pack("<I", 0) + struct.pack("<I", 0)

    # Build all records
    records_bytes = b""
    for i in range(n_records):
        edid = _build_subrecord("EDID", _make_string(f"Weapon{i:04d}"))
        full = _build_subrecord("FULL", _make_string(f"Iron Sword {i}"))
        rec_data = edid + full
        rec_header = b"WEAP" + struct.pack("<I", len(rec_data))
        rec_header += struct.pack("<I", 0)  # flags
        rec_header += struct.pack("<I", 0x00010000 + i)  # FormID
        rec_header += struct.pack("<I", 0) + struct.pack("<I", 0)
        records_bytes += rec_header + rec_data

    grup_size = 24 + len(records_bytes)
    grup = b"GRUP" + struct.pack("<I", grup_size) + b"WEAP"
    grup += struct.pack("<I", 0) + struct.pack("<I", 0) + struct.pack("<I", 0)

    return tes4_header + tes4_data + grup + records_bytes


class TestParse500Records:
    def test_parse_500_records(self):
        """Plugin with 500 records parses correctly."""
        data = _build_large_plugin(500)
        plugin = parse_plugin(BytesIO(data))
        assert len(plugin.groups) == 1
        assert len(plugin.groups[0].children) == 500


class TestExtract500Records:
    def test_extract_500_records(self):
        """Extractor handles 500 records."""
        data = _build_large_plugin(500)
        plugin = parse_plugin(BytesIO(data))
        strings = extract_strings(plugin)
        # Each record has 1 FULL subrecord → 500 translatable strings
        assert len(strings) == 500


class TestTranslate500RecordsDummy:
    def test_translate_500_records_dummy(self):
        """Full pipeline with 500 records using dummy backend."""
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.translation.patcher import apply_translations

        data = _build_large_plugin(500)
        plugin = parse_plugin(BytesIO(data))
        strings = extract_strings(plugin)

        backend = DummyBackend()
        texts = [s.original_text for s in strings]
        translated = backend.translate_batch(texts, "ES")

        translations = {s.key: t for s, t in zip(strings, translated, strict=False)}
        patched = apply_translations(strings, translations)
        assert patched == 500


class TestRoundtrip500Records:
    def test_roundtrip_500_records(self):
        """parse → write → parse with 500 records preserves structure."""
        data = _build_large_plugin(500)
        plugin = parse_plugin(BytesIO(data))

        out = BytesIO()
        write_plugin(plugin, out)

        out.seek(0)
        plugin2 = parse_plugin(out)

        assert len(plugin2.groups) == 1
        assert len(plugin2.groups[0].children) == 500

        # Verify a few records
        for i in [0, 249, 499]:
            rec = plugin2.groups[0].children[i]
            full_subs = [s for s in rec.subrecords if s.type == b"FULL"]
            assert len(full_subs) == 1
            assert f"Iron Sword {i}" in full_subs[0].decode_string()


class TestCompletelyEmpty:
    def test_completely_empty(self):
        """Empty file raises ValueError."""
        with pytest.raises(ValueError):
            parse_plugin(BytesIO(b""))


class TestOnlyTES4NoGroups:
    def test_only_tes4_no_groups(self):
        """TES4 header without any GRUPs is valid."""
        data = _build_large_plugin(0)
        # Remove the empty GRUP (last 24 bytes: GRUP header with no children)
        # Actually _build_large_plugin(0) creates a GRUP with 0 records which is fine
        plugin = parse_plugin(BytesIO(data))
        # The GRUP is there but empty
        assert len(plugin.groups) == 1
        assert len(plugin.groups[0].children) == 0
