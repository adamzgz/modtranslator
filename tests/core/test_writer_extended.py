"""Extended writer tests: compressed records, XXXX, OFST, nested groups."""

from __future__ import annotations

import io
import struct
import zlib

from modtranslator.core.constants import Game, RecordFlag
from modtranslator.core.parser import parse_plugin
from modtranslator.core.records import GroupRecord, PluginFile, Record, Subrecord
from modtranslator.core.writer import write_plugin
from tests.conftest import make_group, make_plugin, make_record, make_subrecord, make_tes4_header


class TestCompressedRecords:
    def _make_compressed_record(self, data: bytes, modified: bool = False) -> Record:
        """Build a compressed record with original compressed data preserved."""
        compressed = zlib.compress(data)
        rec = Record(
            type=b"NAVM",
            flags=RecordFlag.COMPRESSED,
            form_id=0x100,
            vcs1=0,
            vcs2=0,
            subrecords=[Subrecord(type=b"NVNM", data=bytearray(data))],
            _compressed_data=compressed,
            _decompressed_size=len(data),
        )
        if modified:
            # Change subrecord data so it no longer matches original
            rec.subrecords[0].data = bytearray(data + b"\xFF")
            rec._compressed_data = compressed  # still points to old data
        return rec

    def test_unchanged_compressed_preserves_original_bytes(self):
        """Unmodified compressed record reuses original compressed payload when
        the decompressed data matches the serialized subrecords."""
        from modtranslator.core.writer import _serialize_subrecords

        # Build a record, then set _compressed_data from the actual serialized subrecords
        data = b"Hello World" * 100
        rec = Record(
            type=b"NAVM",
            flags=RecordFlag.COMPRESSED,
            form_id=0x100,
            vcs1=0,
            vcs2=0,
            subrecords=[Subrecord(type=b"NVNM", data=bytearray(data))],
        )
        serialized = _serialize_subrecords(rec)
        compressed = zlib.compress(serialized)
        rec._compressed_data = compressed
        rec._decompressed_size = len(serialized)

        plugin = PluginFile(
            header=make_tes4_header(),
            groups=[make_group("NAVM", [rec])],
            game=Game.FALLOUT3,
        )

        buf = io.BytesIO()
        write_plugin(plugin, buf)
        raw = buf.getvalue()

        # The original compressed data should appear verbatim in the output
        assert compressed in raw

    def test_modified_compressed_recompresses(self):
        """Modified compressed record gets recompressed."""
        original_data = b"Hello World" * 100
        rec = self._make_compressed_record(original_data, modified=True)

        plugin = PluginFile(
            header=make_tes4_header(),
            groups=[make_group("NAVM", [rec])],
            game=Game.FALLOUT3,
        )

        buf = io.BytesIO()
        write_plugin(plugin, buf)
        raw = buf.getvalue()

        # Should still be valid — parseable
        buf.seek(0)
        reparsed = parse_plugin(buf)
        assert len(reparsed.groups) == 1

    def test_corrupted_compressed_data_recompresses(self):
        """If original compressed data is corrupted, fall through to recompress."""
        data = b"test data" * 50
        rec = Record(
            type=b"NAVM",
            flags=RecordFlag.COMPRESSED,
            form_id=0x200,
            vcs1=0,
            vcs2=0,
            subrecords=[Subrecord(type=b"NVNM", data=bytearray(data))],
            _compressed_data=b"\x00\x01\x02\x03",  # corrupted
            _decompressed_size=len(data),
        )

        plugin = PluginFile(
            header=make_tes4_header(),
            groups=[make_group("NAVM", [rec])],
            game=Game.FALLOUT3,
        )

        buf = io.BytesIO()
        write_plugin(plugin, buf)

        buf.seek(0)
        reparsed = parse_plugin(buf)
        assert len(reparsed.groups) == 1


class TestNestedGroups:
    def test_nested_group_write_roundtrip(self):
        """Groups nested inside groups serialize and reparse correctly."""
        inner_rec = make_record("CELL", 0x300, [make_subrecord("FULL", "Test Cell")])
        inner_group = make_group("CELL", [inner_rec])
        outer_group = make_group("WRLD", [inner_group])

        plugin = PluginFile(
            header=make_tes4_header(),
            groups=[outer_group],
            game=Game.FALLOUT3,
        )

        buf = io.BytesIO()
        write_plugin(plugin, buf)

        buf.seek(0)
        reparsed = parse_plugin(buf)
        assert len(reparsed.groups) == 1
        outer = reparsed.groups[0]
        assert len(outer.children) == 1
        assert isinstance(outer.children[0], GroupRecord)
        inner = outer.children[0]
        assert len(inner.children) == 1
        assert inner.children[0].subrecords[0].decode_string() == "Test Cell"


class TestOFSTStripping:
    def test_wrld_ofst_stripped(self):
        """OFST subrecords in WRLD records are stripped during write."""
        rec = make_record("WRLD", 0x400, [
            make_subrecord("EDID", "TestWorld"),
            Subrecord(type=b"OFST", data=bytearray(b"\x00" * 20)),
            make_subrecord("FULL", "My World"),
        ])
        plugin = PluginFile(
            header=make_tes4_header(),
            groups=[make_group("WRLD", [rec])],
            game=Game.FALLOUT3,
        )

        buf = io.BytesIO()
        write_plugin(plugin, buf)

        buf.seek(0)
        reparsed = parse_plugin(buf)
        wrld_rec = reparsed.groups[0].children[0]
        sub_types = [s.type for s in wrld_rec.subrecords]
        assert b"OFST" not in sub_types
        assert b"EDID" in sub_types
        assert b"FULL" in sub_types

    def test_non_wrld_ofst_preserved(self):
        """OFST in non-WRLD records is NOT stripped."""
        rec = make_record("CELL", 0x500, [
            make_subrecord("EDID", "TestCell"),
            Subrecord(type=b"OFST", data=bytearray(b"\x01\x02\x03\x04")),
        ])
        plugin = PluginFile(
            header=make_tes4_header(),
            groups=[make_group("CELL", [rec])],
            game=Game.FALLOUT3,
        )

        buf = io.BytesIO()
        write_plugin(plugin, buf)

        buf.seek(0)
        reparsed = parse_plugin(buf)
        cell_rec = reparsed.groups[0].children[0]
        sub_types = [s.type for s in cell_rec.subrecords]
        assert b"OFST" in sub_types


class TestXXXXExtendedSize:
    def test_large_subrecord_uses_xxxx(self):
        """Subrecords >65535 bytes use the XXXX extended-size mechanism."""
        large_data = bytearray(b"\x42" * 70000)
        rec = make_record("NAVM", 0x600, [
            Subrecord(type=b"NVNM", data=large_data),
        ])
        plugin = PluginFile(
            header=make_tes4_header(),
            groups=[make_group("NAVM", [rec])],
            game=Game.FALLOUT3,
        )

        buf = io.BytesIO()
        write_plugin(plugin, buf)
        raw = buf.getvalue()

        # XXXX marker should be present
        assert b"XXXX" in raw

        # Parse back — data should be intact
        buf.seek(0)
        reparsed = parse_plugin(buf)
        navm_rec = reparsed.groups[0].children[0]
        assert len(navm_rec.subrecords[0].data) == 70000
