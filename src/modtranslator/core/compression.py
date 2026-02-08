"""zlib compression/decompression for TES4 compressed records."""

from __future__ import annotations

import struct
import zlib


def decompress_record_data(raw: bytes) -> tuple[bytes, int]:
    """Decompress a compressed record's data payload.

    Args:
        raw: The full data payload (starts with 4-byte decompressed size,
             followed by zlib-compressed data).

    Returns:
        Tuple of (decompressed_bytes, original_decompressed_size).
    """
    if len(raw) < 4:
        raise ValueError("Compressed data too short: missing decompressed size field")
    decompressed_size = struct.unpack("<I", raw[:4])[0]
    compressed = raw[4:]
    decompressed = zlib.decompress(compressed)
    if len(decompressed) != decompressed_size:
        raise ValueError(
            f"Decompressed size mismatch: expected {decompressed_size}, got {len(decompressed)}"
        )
    return decompressed, decompressed_size


def compress_record_data(data: bytes) -> bytes:
    """Compress data for a compressed record.

    Returns:
        4-byte decompressed size (little-endian) + zlib compressed data.
    """
    compressed = zlib.compress(data)
    return struct.pack("<I", len(data)) + compressed
