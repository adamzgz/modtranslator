"""Tests for zlib compression/decompression."""

import struct
import zlib

from modtranslator.core.compression import compress_record_data, decompress_record_data


class TestDecompress:
    def test_basic_decompress(self):
        original = b"Hello, World! This is test data."
        compressed = struct.pack("<I", len(original)) + zlib.compress(original)

        result, size = decompress_record_data(compressed)
        assert result == original
        assert size == len(original)

    def test_empty_raises(self):
        import pytest
        with pytest.raises(ValueError, match="too short"):
            decompress_record_data(b"\x00")

    def test_size_mismatch_raises(self):
        import pytest
        original = b"Hello"
        # Lie about decompressed size
        bad_data = struct.pack("<I", 999) + zlib.compress(original)
        with pytest.raises(ValueError, match="mismatch"):
            decompress_record_data(bad_data)


class TestCompress:
    def test_basic_compress(self):
        original = b"Hello, World! This is test data."
        compressed = compress_record_data(original)

        # Should start with 4-byte decompressed size
        size = struct.unpack("<I", compressed[:4])[0]
        assert size == len(original)

        # Should be decompressible
        result = zlib.decompress(compressed[4:])
        assert result == original

    def test_roundtrip(self):
        original = b"Test data for roundtrip" * 10
        compressed = compress_record_data(original)
        decompressed, size = decompress_record_data(compressed)
        assert decompressed == original
        assert size == len(original)
