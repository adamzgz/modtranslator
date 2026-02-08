"""Tests for parser robustness against corrupted/truncated ESP files."""

from __future__ import annotations

import struct
import warnings
from io import BytesIO

import pytest

from modtranslator.core.parser import parse_plugin


def _make_tes4_header(data_size: int = 0, game_fo3: bool = True) -> bytes:
    """Build a minimal TES4 record header with HEDR subrecord."""
    version = struct.pack("<f", 0.94 if game_fo3 else 1.0)
    hedr_data = version + struct.pack("<I", 0) + struct.pack("<I", 0x000800)
    hedr_sub = b"HEDR" + struct.pack("<H", len(hedr_data)) + hedr_data
    total_data = len(hedr_sub)

    header = b"TES4" + struct.pack("<I", total_data) + struct.pack("<I", 0)
    header += struct.pack("<I", 0) + struct.pack("<I", 0)
    if game_fo3:
        header += struct.pack("<I", 0)  # vcs2

    return header + hedr_sub


class TestEmptyFile:
    def test_empty_file(self):
        """Parsing an empty stream raises ValueError."""
        with pytest.raises(ValueError):
            parse_plugin(BytesIO(b""))


class TestTruncatedTES4Header:
    def test_truncated_tes4_header(self):
        """A truncated TES4 header raises ValueError."""
        # Only first 10 bytes of a header
        with pytest.raises(ValueError):
            parse_plugin(BytesIO(b"TES4" + b"\x00" * 6))


class TestTruncatedGroupHeader:
    def test_truncated_group_header(self):
        """A GRUP with incomplete header raises ValueError."""
        tes4 = _make_tes4_header()
        # Append start of GRUP but truncated
        truncated_grup = b"GRUP" + b"\x00" * 4  # only 8 of 24 bytes
        with pytest.raises(ValueError):
            parse_plugin(BytesIO(tes4 + truncated_grup))


class TestZeroByteSubrecord:
    def test_zero_byte_subrecord(self):
        """A subrecord with size=0 should parse without crash."""
        version = struct.pack("<f", 0.94)
        hedr_data = version + struct.pack("<I", 0) + struct.pack("<I", 0x000800)
        hedr_sub = b"HEDR" + struct.pack("<H", len(hedr_data)) + hedr_data
        # Add a FULL subrecord with size=0
        full_sub = b"FULL" + struct.pack("<H", 0)
        tes4_data = hedr_sub + full_sub
        tes4_data_size = len(tes4_data)

        header = b"TES4" + struct.pack("<I", tes4_data_size)
        header += struct.pack("<I", 0) + struct.pack("<I", 0)
        header += struct.pack("<I", 0) + struct.pack("<I", 0)

        plugin = parse_plugin(BytesIO(header + tes4_data))
        # Should parse successfully — the zero-size subrecord is there
        full_subs = [s for s in plugin.header.subrecords if s.type == b"FULL"]
        assert len(full_subs) == 1
        assert full_subs[0].size == 0


class TestMalformedCompressedData:
    def test_malformed_compressed_data(self):
        """Compressed flag with garbage data emits warning, empty subrecords."""
        tes4 = _make_tes4_header()

        # Build a record with COMPRESSED flag and garbage compressed payload
        rec_type = b"WEAP"
        garbage_payload = struct.pack("<I", 100) + b"\xDE\xAD\xBE\xEF"
        data_size = len(garbage_payload)

        grup_size = 24 + 24 + data_size
        grup = b"GRUP" + struct.pack("<I", grup_size) + b"WEAP"
        grup += struct.pack("<I", 0) + struct.pack("<I", 0) + struct.pack("<I", 0)

        rec_header = rec_type + struct.pack("<I", data_size)
        rec_header += struct.pack("<I", 0x00040000)  # COMPRESSED flag
        rec_header += struct.pack("<I", 0x00002000)  # FormID
        rec_header += struct.pack("<I", 0) + struct.pack("<I", 0)

        data = tes4 + grup + rec_header + garbage_payload

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plugin = parse_plugin(BytesIO(data))

        # Should have emitted a decompression warning
        decomp_warnings = [x for x in w if "Decompression failed" in str(x.message)]
        assert len(decomp_warnings) == 1

        # The record should exist but with empty subrecords
        assert len(plugin.groups) == 1
        rec = plugin.groups[0].children[0]
        assert rec.subrecords == []
