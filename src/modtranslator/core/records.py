"""Data classes representing the TES4 plugin record tree."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modtranslator.core.constants import (
    GRUP_HEADER_SIZE,
    RECORD_HEADER_SIZE_FO3,
    SUBRECORD_HEADER_SIZE,
    Game,
    RecordFlag,
)

if TYPE_CHECKING:
    from modtranslator.core.string_table import StringTableSet


@dataclass
class Subrecord:
    """A single subrecord: Type(4) + Size(2) + Data(N).

    data is a mutable bytearray so the patcher can modify it in-place.
    size is always computed from len(data).
    """

    type: bytes  # 4-byte ASCII tag, e.g. b"FULL"
    data: bytearray

    @property
    def size(self) -> int:
        return len(self.data)

    def decode_string(self, encoding: str = "cp1252") -> str:
        """Decode null-terminated string data.

        Strips *all* trailing null bytes — TES4 strings have a single null
        terminator but some records contain padding nulls or are entirely
        null (e.g. empty terminal-menu prompts stored as 4 zero bytes).
        Stripping only one would leave ghost null characters that the
        extractor wrongly treats as translatable text.

        Fallback chain: UTF-8 → cp1252 → latin-1.
        latin-1 (iso-8859-1) accepts all byte values 0x00-0xFF so it never fails.
        """
        raw = bytes(self.data)
        raw = raw.rstrip(b"\x00")
        if not raw:
            return ""
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                return raw.decode("latin-1")

    def encode_string(self, text: str, encoding: str = "cp1252") -> None:
        """Encode a string and update data (null-terminated)."""
        try:
            encoded = text.encode("utf-8")
            # Only use UTF-8 if it roundtrips through cp1252 or is ASCII.
            # TES4 games expect cp1252 so prefer that unless characters require UTF-8.
            text.encode(encoding)
            encoded = text.encode(encoding)
        except UnicodeEncodeError:
            # Characters outside cp1252 range — keep UTF-8
            encoded = text.encode("utf-8")
        self.data = bytearray(encoded + b"\x00")


@dataclass
class Record:
    """A TES4 record with header fields and a list of subrecords.

    The data_size used for serialization is computed from subrecords.
    """

    type: bytes  # 4-byte ASCII tag, e.g. b"WEAP"
    flags: int
    form_id: int
    vcs1: int
    vcs2: int
    subrecords: list[Subrecord] = field(default_factory=list)

    # Preserved raw compressed payload (set only when flag is compressed and
    # we keep the original compressed data for roundtrip fidelity).
    _compressed_data: bytes | None = field(default=None, repr=False)
    _decompressed_size: int = field(default=0, repr=False)

    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & RecordFlag.COMPRESSED)

    def computed_data_size(self, game: Game = Game.FALLOUT3) -> int:
        """Compute the data payload size for serialization."""
        if self.is_compressed and self._compressed_data is not None:
            # 4 bytes decompressed size + compressed data
            return 4 + len(self._compressed_data)
        total = 0
        for sub in self.subrecords:
            total += SUBRECORD_HEADER_SIZE + sub.size
        return total

    def header_size(self, game: Game = Game.FALLOUT3) -> int:
        return RECORD_HEADER_SIZE_FO3


@dataclass
class GroupRecord:
    """A GRUP container holding records and nested groups.

    group_size includes the 24-byte header.
    """

    label: bytes  # 4 raw bytes (meaning depends on group_type)
    group_type: int
    stamp: int
    vcs: int
    children: list[Record | GroupRecord] = field(default_factory=list)

    def computed_group_size(self, game: Game = Game.FALLOUT3) -> int:
        """Compute total group size including the 24-byte header."""
        size = GRUP_HEADER_SIZE
        for child in self.children:
            if isinstance(child, GroupRecord):
                size += child.computed_group_size(game)
            else:
                size += child.header_size(game) + child.computed_data_size(game)
        return size


@dataclass
class PluginFile:
    """Top-level representation of an ESP/ESM file.

    Contains the TES4 header record followed by top-level GRUPs.
    """

    header: Record  # The TES4 record
    groups: list[GroupRecord] = field(default_factory=list)
    game: Game = Game.FALLOUT3
    string_tables: StringTableSet | None = field(default=None, repr=False)

    @property
    def is_localized(self) -> bool:
        """Check if this plugin uses external string tables (LOCALIZED flag)."""
        return bool(self.header.flags & RecordFlag.LOCALIZED)

    def detect_game(self) -> Game:
        """Detect game from the HEDR subrecord version float."""
        for sub in self.header.subrecords:
            if sub.type == b"HEDR" and len(sub.data) >= 4:
                version = struct.unpack("<f", bytes(sub.data[:4]))[0]
                if abs(version - 1.70) < 0.02:
                    return Game.SKYRIM
                if abs(version - 0.94) < 0.01:
                    return Game.FALLOUT3
        return Game.UNKNOWN
