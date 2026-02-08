"""Binary reader: bytes → Record tree for TES4 plugin files."""

from __future__ import annotations

import struct
import warnings
import zlib
from typing import BinaryIO

from modtranslator.core.compression import decompress_record_data
from modtranslator.core.constants import (
    GRUP_HEADER_SIZE,
    GRUP_TYPE,
    RECORD_HEADER_SIZE_FO3,
    SUBRECORD_HEADER_SIZE,
    Game,
    RecordFlag,
)
from modtranslator.core.records import GroupRecord, PluginFile, Record, Subrecord


def parse_plugin(stream: BinaryIO) -> PluginFile:
    """Parse a full ESP/ESM file from a binary stream.

    Returns a PluginFile with the TES4 header and all top-level GRUPs.
    """
    initial_game = Game.FALLOUT3

    header = _parse_record(stream, initial_game)
    if header.type != b"TES4":
        raise ValueError(f"Expected TES4 header, got {header.type!r}")

    plugin = PluginFile(header=header)
    plugin.game = plugin.detect_game()
    if plugin.game == Game.UNKNOWN:
        plugin.game = initial_game

    # Parse remaining top-level groups
    while True:
        peek = stream.read(4)
        if len(peek) < 4:
            break  # EOF
        stream.seek(-4, 1)

        try:
            if peek == GRUP_TYPE:
                group = _parse_group(stream, plugin.game)
                plugin.groups.append(group)
            else:
                # Shouldn't happen at top level, but handle gracefully.
                # Valid plugins should only have GRUPs after TES4.
                _parse_record(stream, plugin.game)
                break
        except struct.error as e:
            raise ValueError(f"Corrupted or truncated ESP file: {e}") from e

    return plugin


def _parse_record(stream: BinaryIO, game: Game) -> Record:
    """Parse a single record (not a GRUP)."""
    header_size = RECORD_HEADER_SIZE_FO3
    header_data = stream.read(header_size)
    if len(header_data) < header_size:
        raise ValueError("Unexpected end of file reading record header")

    rec_type = header_data[0:4]
    data_size = struct.unpack_from("<I", header_data, 4)[0]
    flags = struct.unpack_from("<I", header_data, 8)[0]
    form_id = struct.unpack_from("<I", header_data, 12)[0]
    vcs1 = struct.unpack_from("<I", header_data, 16)[0]
    vcs2 = struct.unpack_from("<I", header_data, 20)[0]

    raw_data = stream.read(data_size)
    if len(raw_data) < data_size:
        raise ValueError(f"Unexpected end of file reading record data ({rec_type!r})")

    record = Record(
        type=rec_type,
        flags=flags,
        form_id=form_id,
        vcs1=vcs1,
        vcs2=vcs2,
    )

    is_compressed = bool(flags & RecordFlag.COMPRESSED)

    if is_compressed:
        # Keep original compressed data for roundtrip
        record._compressed_data = raw_data[4:]
        record._decompressed_size = struct.unpack("<I", raw_data[:4])[0]
        try:
            decompressed, _ = decompress_record_data(raw_data)
            record.subrecords = _parse_subrecords(decompressed)
        except (zlib.error, struct.error, ValueError) as e:
            warnings.warn(
                f"Decompression failed for record {rec_type!r} "
                f"FormID 0x{form_id:08X}: {e}",
                stacklevel=2,
            )
            record._compressed_data = raw_data[4:]
            record.subrecords = []
    else:
        record.subrecords = _parse_subrecords(raw_data)

    return record


def _parse_group(stream: BinaryIO, game: Game) -> GroupRecord:
    """Parse a GRUP and all its children recursively."""
    header_data = stream.read(GRUP_HEADER_SIZE)
    if len(header_data) < GRUP_HEADER_SIZE:
        raise ValueError("Unexpected end of file reading GRUP header")

    tag = header_data[0:4]
    if tag != GRUP_TYPE:
        raise ValueError(f"Expected GRUP tag, got {tag!r}")

    group_size = struct.unpack_from("<I", header_data, 4)[0]
    label = header_data[8:12]
    group_type = struct.unpack_from("<I", header_data, 12)[0]
    stamp = struct.unpack_from("<I", header_data, 16)[0]
    vcs = struct.unpack_from("<I", header_data, 20)[0]

    group = GroupRecord(
        label=label,
        group_type=group_type,
        stamp=stamp,
        vcs=vcs,
    )

    # Read children within the group's data area
    bytes_remaining = group_size - GRUP_HEADER_SIZE
    start_pos = stream.tell()

    while (stream.tell() - start_pos) < bytes_remaining:
        peek = stream.read(4)
        if len(peek) < 4:
            break
        stream.seek(-4, 1)

        try:
            if peek == GRUP_TYPE:
                child = _parse_group(stream, game)
                group.children.append(child)
            else:
                child_rec = _parse_record(stream, game)
                group.children.append(child_rec)
        except struct.error as e:
            raise ValueError(f"Corrupted or truncated ESP file: {e}") from e

    return group


def _parse_subrecords(data: bytes) -> list[Subrecord]:
    """Parse subrecords from a flat data buffer.

    Handles the XXXX extended-size mechanism: when a subrecord's data exceeds
    65535 bytes, the file places a preceding XXXX subrecord (size=4) whose
    uint32 payload carries the real size.  The actual subrecord follows with
    its uint16 size field set to 0.
    """
    subrecords: list[Subrecord] = []
    offset = 0
    xxxx_size: int | None = None

    while offset < len(data):
        if offset + SUBRECORD_HEADER_SIZE > len(data):
            break
        sub_type = data[offset : offset + 4]
        sub_size = struct.unpack_from("<H", data, offset + 4)[0]
        offset += SUBRECORD_HEADER_SIZE

        if sub_type == b"XXXX":
            # Extended-size marker: data is a uint32 real size for the NEXT sub
            if sub_size >= 4:
                xxxx_size = struct.unpack_from("<I", data, offset)[0]
            offset += sub_size
            # Don't emit XXXX as a subrecord — it's metadata consumed by writer
            continue

        # Apply extended size from preceding XXXX
        if xxxx_size is not None:
            sub_size = xxxx_size
            xxxx_size = None

        sub_data = bytearray(data[offset : offset + sub_size])
        offset += sub_size
        subrecords.append(Subrecord(type=sub_type, data=sub_data))
    return subrecords
