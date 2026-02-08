"""Binary writer: Record tree → bytes for TES4 plugin files."""

from __future__ import annotations

import struct
import zlib
from typing import BinaryIO

from modtranslator.core.compression import compress_record_data
from modtranslator.core.constants import (
    GRUP_HEADER_SIZE,
    GRUP_TYPE,
    Game,
)
from modtranslator.core.records import GroupRecord, PluginFile, Record


def write_plugin(plugin: PluginFile, stream: BinaryIO) -> None:
    """Serialize a PluginFile to a binary stream."""
    _write_record(plugin.header, stream, plugin.game)
    for group in plugin.groups:
        _write_group(group, stream, plugin.game)


def _serialize_subrecords(record: Record) -> bytes:
    """Serialize all subrecords of a record to bytes.

    Handles the XXXX extended-size mechanism: if a subrecord's data exceeds
    65535 bytes, a preceding XXXX subrecord is emitted with the real uint32
    size and the actual subrecord's uint16 size field is set to 0.

    OFST subrecords in WRLD records are stripped because they contain
    file-level offsets that become stale whenever record sizes change
    (e.g. after translation).  The game engine rebuilds this cache at load
    time, so removing it is safe and is standard practice in modding tools.
    """
    is_wrld = record.type == b"WRLD"
    parts: list[bytes] = []
    for sub in record.subrecords:
        # Strip OFST from WRLD records — contains stale file offsets
        if is_wrld and sub.type == b"OFST":
            continue
        if sub.size > 0xFFFF:
            # Emit XXXX extended-size marker
            parts.append(b"XXXX")
            parts.append(struct.pack("<H", 4))
            parts.append(struct.pack("<I", sub.size))
            # Actual subrecord with uint16 size = 0
            parts.append(sub.type)
            parts.append(struct.pack("<H", 0))
            parts.append(bytes(sub.data))
        else:
            parts.append(sub.type)
            parts.append(struct.pack("<H", sub.size))
            parts.append(bytes(sub.data))
    return b"".join(parts)


def _compress_or_preserve(record: Record, subrecord_data: bytes) -> bytes:
    """For compressed records, preserve original compressed bytes if data is unchanged.

    Recompressing unmodified records with Python's zlib can produce different
    (often larger) output than the original tool. This changes group sizes
    throughout the file. Preserving original bytes for untouched records keeps
    the output byte-identical to the source for all unmodified data.
    """
    if record._compressed_data is not None:
        try:
            original_decompressed = zlib.decompress(record._compressed_data)
            if original_decompressed == subrecord_data:
                # Unchanged — reuse original compressed data verbatim
                return (
                    struct.pack("<I", record._decompressed_size)
                    + record._compressed_data
                )
        except zlib.error:
            pass  # corrupted original — fall through to recompress
    return compress_record_data(subrecord_data)


def _write_record(record: Record, stream: BinaryIO, game: Game) -> None:
    """Write a single record to the stream, recalculating sizes."""
    subrecord_data = _serialize_subrecords(record)

    if record.is_compressed:
        payload = _compress_or_preserve(record, subrecord_data)
        data_size = len(payload)
    else:
        data_size = len(subrecord_data)
        payload = subrecord_data

    # Write header
    stream.write(record.type)
    stream.write(struct.pack("<I", data_size))
    stream.write(struct.pack("<I", record.flags))
    stream.write(struct.pack("<I", record.form_id))
    stream.write(struct.pack("<I", record.vcs1))
    stream.write(struct.pack("<I", record.vcs2))

    # Write payload
    stream.write(payload)


def _write_group(group: GroupRecord, stream: BinaryIO, game: Game) -> None:
    """Write a GRUP and all children, recalculating group_size."""
    # First, serialize all children to a buffer to compute size
    children_data = bytearray()
    for child in group.children:
        if isinstance(child, GroupRecord):
            import io

            buf = io.BytesIO()
            _write_group(child, buf, game)
            children_data.extend(buf.getvalue())
        else:
            import io

            buf = io.BytesIO()
            _write_record(child, buf, game)
            children_data.extend(buf.getvalue())

    group_size = GRUP_HEADER_SIZE + len(children_data)

    # Write GRUP header
    stream.write(GRUP_TYPE)
    stream.write(struct.pack("<I", group_size))
    stream.write(group.label)
    stream.write(struct.pack("<I", group.group_type))
    stream.write(struct.pack("<I", group.stamp))
    stream.write(struct.pack("<I", group.vcs))

    # Write children
    stream.write(bytes(children_data))
