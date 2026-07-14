"""Tests for the read-only Bethesda archive reader (bsa_archive)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from modtranslator.core.bsa_archive import (
    ArchiveError,
    BethesdaArchive,
    iter_translatable_assets,
)

# BSA archive flags
_DIRECTORY_STRINGS = 0x1
_FILE_STRINGS = 0x2
_COMPRESSED = 0x4
_COMPRESSION_TOGGLE = 0x40000000


def _bzstring(text: str) -> bytes:
    """Byte-length-prefixed, null-terminated string (BSA folder name)."""
    raw = text.encode("cp1252") + b"\x00"
    return bytes([len(raw)]) + raw


def _bsa_data_block(content: bytes, compressed: bool) -> bytes:
    if not compressed:
        return content
    return struct.pack("<I", len(content)) + zlib.compress(content)


def build_bsa(
    version: int,
    files: list[tuple[str, str, bytes, bool]],
) -> bytes:
    """Build a minimal BSA.

    files: list of (folder, filename, content, compressed).
    Groups files by folder preserving first-seen folder order.
    """
    # Group by folder.
    folders: dict[str, list[tuple[str, bytes, bool]]] = {}
    for folder, filename, content, comp in files:
        folders.setdefault(folder, []).append((filename, content, comp))

    flags = _DIRECTORY_STRINGS | _FILE_STRINGS
    folder_rec_size = 24 if version == 105 else 16

    # File-name block (flat order: folder order, then file order).
    name_block = b""
    for folder in folders:
        for filename, _content, _comp in folders[folder]:
            name_block += filename.encode("cp1252") + b"\x00"

    # Size of sections that precede the file data.
    header_size = 36
    folder_records_size = len(folders) * folder_rec_size
    folder_blocks_size = 0
    for folder, entries in folders.items():
        folder_blocks_size += len(_bzstring(folder)) + len(entries) * 16
    data_start = header_size + folder_records_size + folder_blocks_size + len(name_block)

    # Assign data offsets sequentially and build data blob.
    data_blob = b""
    offsets: dict[tuple[str, str], tuple[int, int, bool]] = {}
    for folder, entries in folders.items():
        for filename, content, comp in entries:
            block = _bsa_data_block(content, comp)
            offsets[(folder, filename)] = (data_start + len(data_blob), len(block), comp)
            data_blob += block

    file_count = sum(len(v) for v in folders.values())
    header = b"BSA\x00"
    header += struct.pack(
        "<8I",
        version,
        header_size,
        flags,
        len(folders),
        file_count,
        sum(len(_bzstring(f)) - 1 for f in folders),  # folder names length (approx)
        len(name_block),
        0,
    )

    # Folder records (offset field unused by the parser; set to 0).
    folder_records = b""
    for entries in folders.values():
        if version == 105:
            folder_records += struct.pack("<QIIQ", 0, len(entries), 0, 0)
        else:
            folder_records += struct.pack("<QII", 0, len(entries), 0)

    # Folder file-blocks.
    folder_blocks = b""
    for folder, entries in folders.items():
        folder_blocks += _bzstring(folder)
        for filename, _content, comp in entries:
            offset, block_size, _ = offsets[(folder, filename)]
            size_field = block_size | (_COMPRESSION_TOGGLE if comp else 0)
            folder_blocks += struct.pack("<QII", 0, size_field, offset)

    return header + folder_records + folder_blocks + name_block + data_blob


def build_ba2(
    files: list[tuple[str, bytes, bool]],
    type_tag: bytes = b"GNRL",
) -> bytes:
    """Build a minimal BA2. files: list of (name, content, compressed)."""
    header_size = 24
    records_size = len(files) * 36
    data_start = header_size + records_size

    data_blob = b""
    records = []
    for name, content, comp in files:
        if comp:
            packed = zlib.compress(content)
            offset = data_start + len(data_blob)
            data_blob += packed
            records.append((name, offset, len(packed), len(content)))
        else:
            offset = data_start + len(data_blob)
            data_blob += content
            records.append((name, offset, 0, len(content)))

    name_table_offset = data_start + len(data_blob)

    header = b"BTDX" + struct.pack("<I4sIQ", 1, type_tag, len(files), name_table_offset)

    rec_bytes = b""
    for name, offset, packed, unpacked in records:
        ext = name.rsplit(".", 1)[-1].encode("cp1252")[:4].ljust(4, b"\x00")
        rec_bytes += struct.pack(
            "<I4sIIQIII", 0, ext, 0, 0, offset, packed, unpacked, 0xBAADF00D
        )

    name_table = b""
    for name, _o, _p, _u in records:
        raw = name.encode("cp1252")
        name_table += struct.pack("<H", len(raw)) + raw

    return header + rec_bytes + data_blob + name_table


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# -- BSA v104 -----------------------------------------------------------------


def test_bsa104_list_and_read(tmp_path: Path) -> None:
    data = build_bsa(
        104,
        [
            ("scripts", "plain.pex", b"hello world", False),
            ("scripts", "packed.pex", b"compressed payload here" * 4, True),
        ],
    )
    path = _write(tmp_path, "test.bsa", data)
    with BethesdaArchive(path) as arc:
        names = sorted(e.name for e in arc.entries())
        assert names == ["scripts/packed.pex", "scripts/plain.pex"]
        assert arc.read("scripts/plain.pex") == b"hello world"
        assert arc.read("scripts/packed.pex") == b"compressed payload here" * 4


def test_bsa104_compressed_flag_and_size(tmp_path: Path) -> None:
    content = b"a longer compressible body " * 8
    data = build_bsa(104, [("scripts", "packed.pex", content, True)])
    path = _write(tmp_path, "c.bsa", data)
    with BethesdaArchive(path) as arc:
        entry = arc.entries()[0]
        assert entry.compressed is True
        assert entry.size == len(content)
        assert len(arc.read(entry.name)) == len(content)


def test_bsa_case_insensitive_read(tmp_path: Path) -> None:
    data = build_bsa(104, [("Scripts", "MixedCase.pex", b"data", False)])
    path = _write(tmp_path, "m.bsa", data)
    with BethesdaArchive(path) as arc:
        assert arc.read("SCRIPTS/MIXEDCASE.PEX") == b"data"


# -- BSA v105 -----------------------------------------------------------------


def test_bsa105_uncompressed(tmp_path: Path) -> None:
    """v105 with uncompressed files (24-byte folder records, no LZ4 needed)."""
    data = build_bsa(105, [("scripts", "loose.pex", b"skyrim se data", False)])
    path = _write(tmp_path, "se.bsa", data)
    with BethesdaArchive(path) as arc:
        assert [e.name for e in arc.entries()] == ["scripts/loose.pex"]
        assert arc.read("scripts/loose.pex") == b"skyrim se data"


def test_bsa105_lz4_read(tmp_path: Path) -> None:
    """v105 compressed files use LZ4 frame; requires the optional lz4 package."""
    lz4 = pytest.importorskip("lz4.frame")
    content = b"lz4 compressed skyrim se payload " * 4
    # Build a v105 with an LZ4-frame data block manually.
    packed = lz4.compress(content)

    header_size = 36
    folder_rec = struct.pack("<QIIQ", 0, 1, 0, 0)  # 24 bytes
    folder_name = _bzstring("scripts")
    name_block = b"lz4.pex\x00"
    block = struct.pack("<I", len(content)) + packed
    data_start = header_size + len(folder_rec) + len(folder_name) + 16 + len(name_block)
    size_field = len(block) | _COMPRESSION_TOGGLE
    file_rec = struct.pack("<QII", 0, size_field, data_start)

    header = b"BSA\x00" + struct.pack(
        "<8I", 105, header_size, _DIRECTORY_STRINGS | _FILE_STRINGS, 1, 1, 8, len(name_block), 0
    )
    data = header + folder_rec + folder_name + file_rec + name_block + block
    path = _write(tmp_path, "lz4.bsa", data)
    with BethesdaArchive(path) as arc:
        assert arc.read("scripts/lz4.pex") == content


# -- BA2 GNRL -----------------------------------------------------------------


def test_ba2_gnrl_list_and_read(tmp_path: Path) -> None:
    data = build_ba2(
        [
            (r"scripts\plain.pex", b"flat ba2 file", False),
            (r"scripts\packed.pex", b"zlib ba2 payload " * 4, True),
        ]
    )
    path = _write(tmp_path, "test.ba2", data)
    with BethesdaArchive(path) as arc:
        entries = {e.name: e for e in arc.entries()}
        assert set(entries) == {"scripts/plain.pex", "scripts/packed.pex"}
        assert entries["scripts/plain.pex"].compressed is False
        assert entries["scripts/packed.pex"].compressed is True
        assert arc.read("scripts/plain.pex") == b"flat ba2 file"
        assert arc.read("scripts/packed.pex") == b"zlib ba2 payload " * 4


def test_ba2_dx10_ignored(tmp_path: Path) -> None:
    data = build_ba2([(r"textures\foo.dds", b"texture bytes", False)], type_tag=b"DX10")
    path = _write(tmp_path, "tex.ba2", data)
    with BethesdaArchive(path) as arc:
        assert arc.entries() == []


# -- iter_translatable_assets -------------------------------------------------


def test_iter_translatable_assets_filters(tmp_path: Path) -> None:
    data = build_bsa(
        104,
        [
            ("scripts", "keep.pex", b"pex body", False),
            ("interface\\translations", "mod_english.txt", b"key\tvalue", False),
            ("textures", "skip.dds", b"nope", False),
            ("meshes", "skip.nif", b"nope", False),
        ],
    )
    path = _write(tmp_path, "assets.bsa", data)
    found = dict(iter_translatable_assets(path))
    assert set(found) == {
        "scripts/keep.pex",
        "interface/translations/mod_english.txt",
    }
    assert found["scripts/keep.pex"] == b"pex body"
    assert found["interface/translations/mod_english.txt"] == b"key\tvalue"


def test_iter_translatable_assets_ba2(tmp_path: Path) -> None:
    data = build_ba2(
        [
            (r"Scripts\Quest.pex", b"quest script", False),
            (r"Interface\Translations\Quest_en.txt", b"$Quest\tQuest", True),
            (r"Textures\ui.dds", b"pixels", False),
        ]
    )
    path = _write(tmp_path, "mod.ba2", data)
    found = dict(iter_translatable_assets(path))
    assert set(found) == {
        "scripts/quest.pex",
        "interface/translations/quest_en.txt",
    }


# -- robustness ---------------------------------------------------------------


def test_unknown_magic_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.bsa", b"NOPE" + b"\x00" * 40)
    with pytest.raises(ArchiveError):
        BethesdaArchive(path)


def test_truncated_bsa_raises(tmp_path: Path) -> None:
    full = build_bsa(104, [("scripts", "a.pex", b"data", False)])
    path = _write(tmp_path, "trunc.bsa", full[:20])
    with pytest.raises(ArchiveError):
        BethesdaArchive(path)


def test_unsupported_bsa_version_raises(tmp_path: Path) -> None:
    header = b"BSA\x00" + struct.pack("<8I", 99, 36, 3, 0, 0, 0, 0, 0)
    path = _write(tmp_path, "old.bsa", header)
    with pytest.raises(ArchiveError):
        BethesdaArchive(path)


def test_read_missing_name_raises_keyerror(tmp_path: Path) -> None:
    data = build_bsa(104, [("scripts", "a.pex", b"data", False)])
    path = _write(tmp_path, "k.bsa", data)
    with BethesdaArchive(path) as arc, pytest.raises(KeyError):
        arc.read("scripts/missing.pex")
