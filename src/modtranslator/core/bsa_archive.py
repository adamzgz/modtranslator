"""Read-only reader for Bethesda packed archives (.bsa / .ba2).

Extracts loose assets (compiled scripts, MCM translation files) from packed
mod archives so the pipeline can translate them and write the result back as a
*loose* file (which the games load with priority over archived content). Nothing
is ever re-injected into the archive.

Supported formats:
  * BSA (TES4) versions 103 (Oblivion), 104 (FO3/FNV/Skyrim LE) and 105
    (Skyrim SE). Compression is zlib for 103/104 and LZ4 frame for 105.
  * BA2 (Fallout 4) GNRL (general) archives with per-file zlib. The DX10
    (texture) variant is recognised but exposes no entries.

LZ4 (needed only by BSA v105 compressed files) is an optional dependency: install
the ``bsa`` extra (``pip install -e ".[bsa]"``). If a v105 file needs LZ4 and the
package is missing, :meth:`BethesdaArchive.read` raises :class:`ArchiveError`.

Format references:
  * BSA: https://en.uesp.net/wiki/Skyrim_Mod:Archive_File_Format
  * BA2: https://docs.rs/ba2/latest/ba2/fo4/index.html
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

try:
    import lz4.frame as _lz4_frame  # type: ignore[import-not-found]
except ImportError:
    _lz4_frame = None

BSA_MAGIC = b"BSA\x00"
BA2_MAGIC = b"BTDX"

# BSA archive flags
_FLAG_DIRECTORY_STRINGS = 0x1
_FLAG_FILE_STRINGS = 0x2
_FLAG_COMPRESSED = 0x4
_FLAG_EMBEDDED_NAMES = 0x100

# BSA file-record size field: bit 30 toggles compression vs. the archive default.
_FILE_COMPRESSION_TOGGLE = 0x40000000
_FILE_SIZE_MASK = 0x3FFFFFFF


class ArchiveError(Exception):
    """Raised for unsupported, truncated or otherwise unreadable archives."""


@dataclass
class ArchiveEntry:
    """A single file inside an archive."""

    name: str  # normalised internal path: lowercase, '/' separators
    size: int  # uncompressed size in bytes
    compressed: bool


@dataclass
class _FileRef:
    """Internal per-file read information."""

    name: str
    offset: int
    stored_size: int  # bytes of the on-disk data block (excludes nothing)
    uncompressed_size: int
    compressed: bool
    embedded_name: bool
    size_prefixed: bool  # BSA compressed blocks start with a uint32 original size


def _normalise(path: str) -> str:
    return path.replace("\\", "/").strip("/").lower()


class BethesdaArchive:
    """Read-only accessor for a BSA or BA2 archive.

    Detects the format from the file magic. Use as a context manager or call
    :meth:`close` when done.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._refs: dict[str, _FileRef] = {}
        self._lz4 = False  # True when compressed data uses LZ4 frame (BSA v105)
        try:
            self._fh = path.open("rb")
        except OSError as exc:  # pragma: no cover - trivial passthrough
            raise ArchiveError(f"cannot open archive: {exc}") from exc
        try:
            magic = self._fh.read(4)
            if magic == BSA_MAGIC:
                self._parse_bsa()
            elif magic == BA2_MAGIC:
                self._parse_ba2()
            else:
                raise ArchiveError(f"unknown archive magic: {magic!r}")
        except struct.error as exc:
            self._fh.close()
            raise ArchiveError(f"truncated or corrupt archive: {exc}") from exc
        except Exception:
            self._fh.close()
            raise

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> BethesdaArchive:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    # -- public API --------------------------------------------------------

    def entries(self) -> list[ArchiveEntry]:
        return [
            ArchiveEntry(name=r.name, size=r.uncompressed_size, compressed=r.compressed)
            for r in self._refs.values()
        ]

    def read(self, name: str) -> bytes:
        """Return the (decompressed) bytes of *name* (case-insensitive)."""
        ref = self._refs.get(_normalise(name))
        if ref is None:
            raise KeyError(name)
        try:
            return self._read_ref(ref)
        except struct.error as exc:
            raise ArchiveError(f"truncated file data for {name!r}: {exc}") from exc

    # -- reading -----------------------------------------------------------

    def _read_ref(self, ref: _FileRef) -> bytes:
        fh = self._fh
        fh.seek(ref.offset)
        consumed = 0
        if ref.embedded_name:
            length = fh.read(1)[0]
            fh.read(length)
            consumed = 1 + length
        if not ref.compressed:
            return fh.read(ref.stored_size - consumed)
        if ref.size_prefixed:
            original_size = int(struct.unpack("<I", fh.read(4))[0])
            consumed += 4
        else:
            original_size = ref.uncompressed_size
        blob = fh.read(ref.stored_size - consumed)
        if self._lz4:
            if _lz4_frame is None:
                raise ArchiveError(
                    "this archive uses LZ4 compression; install the 'bsa' extra "
                    "(pip install modtranslator[bsa])"
                )
            data: bytes = _lz4_frame.decompress(blob)
        else:
            data = zlib.decompress(blob)
        if len(data) != original_size:
            raise ArchiveError(
                f"decompressed size mismatch for {ref.name!r}: "
                f"{len(data)} != {original_size}"
            )
        return data

    # -- BSA parsing -------------------------------------------------------

    def _parse_bsa(self) -> None:
        fh = self._fh
        # Header is 36 bytes; first 4 (magic) already consumed.
        (
            version,
            header_size,
            flags,
            folder_count,
            file_count,
            _folder_names_len,
            file_names_len,
            _file_flags,
        ) = struct.unpack("<8I", fh.read(32))

        if version not in (103, 104, 105):
            raise ArchiveError(f"unsupported BSA version: {version}")
        self._lz4 = version == 105
        default_compressed = bool(flags & _FLAG_COMPRESSED)
        embedded_names = bool(flags & _FLAG_EMBEDDED_NAMES)

        fh.seek(header_size)
        # Folder records: v103/104 = 16 bytes, v105 = 24 bytes.
        folder_file_counts: list[int] = []
        for _ in range(folder_count):
            if version == 105:
                _hash, count, _pad, _offset = struct.unpack("<QIIQ", fh.read(24))
            else:
                _hash, count, _offset = struct.unpack("<QII", fh.read(16))
            folder_file_counts.append(count)

        # Folder file-blocks follow contiguously in folder order. Each block is
        # an optional bzstring folder name followed by the folder's file records.
        raw_files: list[tuple[str, int, int]] = []  # (folder_name, size_field, offset)
        for count in folder_file_counts:
            folder_name = ""
            if flags & _FLAG_DIRECTORY_STRINGS:
                length = fh.read(1)[0]
                folder_name = fh.read(length).split(b"\x00", 1)[0].decode("cp1252")
            for _ in range(count):
                _hash, size_field, offset = struct.unpack("<QII", fh.read(16))
                raw_files.append((folder_name, size_field, offset))

        # File names block: null-terminated names in the same flat order.
        file_names: list[str] = []
        if flags & _FLAG_FILE_STRINGS:
            block = fh.read(file_names_len)
            file_names = [n.decode("cp1252") for n in block.split(b"\x00") if n]

        if file_count != len(raw_files):  # defensive; header vs. actual mismatch
            raise ArchiveError("BSA file count mismatch")

        for i, (folder_name, size_field, offset) in enumerate(raw_files):
            file_name = file_names[i] if i < len(file_names) else f"file{i:05d}"
            full = f"{folder_name}/{file_name}" if folder_name else file_name
            toggled = bool(size_field & _FILE_COMPRESSION_TOGGLE)
            stored_size = size_field & _FILE_SIZE_MASK
            compressed = default_compressed ^ toggled
            uncompressed = self._bsa_uncompressed_size(
                offset, stored_size, compressed, embedded_names
            )
            ref = _FileRef(
                name=_normalise(full),
                offset=offset,
                stored_size=stored_size,
                uncompressed_size=uncompressed,
                compressed=compressed,
                embedded_name=embedded_names,
                size_prefixed=True,
            )
            self._refs[ref.name] = ref

    def _bsa_uncompressed_size(
        self, offset: int, stored_size: int, compressed: bool, embedded_names: bool
    ) -> int:
        """Determine a file's logical size (reads the size prefix if compressed)."""
        consumed = 0
        if embedded_names:
            self._fh.seek(offset)
            consumed = 1 + self._fh.read(1)[0]
        if not compressed:
            return stored_size - consumed
        self._fh.seek(offset + consumed)
        return int(struct.unpack("<I", self._fh.read(4))[0])

    # -- BA2 parsing -------------------------------------------------------

    def _parse_ba2(self) -> None:
        fh = self._fh
        # Header: magic(4, consumed) + version(4) + type(4) + count(4) + name_offset(8)
        _version, type_tag, file_count, name_table_offset = struct.unpack(
            "<I4sIQ", fh.read(20)
        )
        if type_tag != b"GNRL":
            # DX10 (textures) and any other variant: nothing translatable here.
            return

        records: list[tuple[int, int, int]] = []  # (offset, packed, unpacked)
        for _ in range(file_count):
            # nameHash(4) ext(4) dirHash(4) flags(4) offset(8) packed(4) unpacked(4) align(4)
            (_nh, _ext, _dh, _fl, offset, packed, unpacked, _align) = struct.unpack(
                "<I4sIIQIII", fh.read(36)
            )
            records.append((offset, packed, unpacked))

        fh.seek(name_table_offset)
        names: list[str] = []
        for _ in range(file_count):
            (length,) = struct.unpack("<H", fh.read(2))
            names.append(fh.read(length).decode("cp1252"))

        for i, (offset, packed, unpacked) in enumerate(records):
            name = names[i] if i < len(names) else f"file{i:05d}"
            compressed = packed != 0
            ref = _FileRef(
                name=_normalise(name),
                offset=offset,
                stored_size=(packed if compressed else unpacked),
                uncompressed_size=unpacked,
                compressed=compressed,
                embedded_name=False,
                size_prefixed=False,
            )
            self._refs[ref.name] = ref


def iter_translatable_assets(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield ``(internal_path, data)`` for translatable assets in an archive.

    Selects ``.pex`` files under ``scripts/`` and ``.txt`` files under
    ``interface/translations/``. Paths are normalised (lowercase, '/').
    """
    with BethesdaArchive(path) as archive:
        for entry in archive.entries():
            name = entry.name
            is_pex = name.endswith(".pex") and "scripts/" in name
            is_mcm = name.endswith(".txt") and "interface/translations/" in name
            if is_pex or is_mcm:
                yield name, archive.read(name)
