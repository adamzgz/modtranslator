"""Script to generate minimal ESP fixture files for testing.

Run with: python -m tests.fixtures.create_fixtures
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def write_uint32(f, value: int) -> None:
    f.write(struct.pack("<I", value))


def write_uint16(f, value: int) -> None:
    f.write(struct.pack("<H", value))


def write_subrecord(f, tag: str, data: bytes) -> None:
    f.write(tag.encode("ascii"))
    write_uint16(f, len(data))
    f.write(data)


def make_string(s: str) -> bytes:
    return s.encode("cp1252") + b"\x00"


def create_minimal_fo3_esp() -> None:
    """Create a minimal Fallout 3 ESP with a TES4 header + 1 GRUP + 1 WEAP record."""
    path = FIXTURES_DIR / "minimal_fo3.esp"

    # Build subrecord data
    hedr_data = struct.pack("<f", 0.94) + struct.pack("<I", 0) + struct.pack("<I", 0x000800)
    cnam_data = make_string("TestAuthor")
    edid_data = make_string("TestWeapon")
    full_data = make_string("Iron Sword")

    # Calculate TES4 data size
    tes4_data_size = (6 + len(hedr_data)) + (6 + len(cnam_data))

    # Calculate WEAP record
    weap_data_size = (6 + len(edid_data)) + (6 + len(full_data))
    weap_header_size = 24  # FO3 record header

    # GRUP size = 24 (header) + weap_header_size + weap_data_size
    grup_size = 24 + weap_header_size + weap_data_size

    with open(path, "wb") as f:
        # TES4 header (24 bytes)
        f.write(b"TES4")
        write_uint32(f, tes4_data_size)
        write_uint32(f, 0)  # flags
        write_uint32(f, 0)  # formID
        write_uint32(f, 0)  # vcs1
        write_uint32(f, 0)  # vcs2

        # TES4 subrecords
        write_subrecord(f, "HEDR", hedr_data)
        write_subrecord(f, "CNAM", cnam_data)

        # GRUP header
        f.write(b"GRUP")
        write_uint32(f, grup_size)
        f.write(b"WEAP")  # label
        write_uint32(f, 0)  # group type (top)
        write_uint32(f, 0)  # stamp
        write_uint32(f, 0)  # vcs

        # WEAP record header (24 bytes)
        f.write(b"WEAP")
        write_uint32(f, weap_data_size)
        write_uint32(f, 0)  # flags
        write_uint32(f, 0x00001000)  # formID
        write_uint32(f, 0)  # vcs1
        write_uint32(f, 0)  # vcs2

        # WEAP subrecords
        write_subrecord(f, "EDID", edid_data)
        write_subrecord(f, "FULL", full_data)

    print(f"Created {path} ({path.stat().st_size} bytes)")


def create_multi_record_esp() -> None:
    """Create an ESP with multiple record types in separate GRUPs."""
    path = FIXTURES_DIR / "multi_record.esp"

    hedr_data = struct.pack("<f", 0.94) + struct.pack("<I", 0) + struct.pack("<I", 0x000800)
    tes4_data_size = 6 + len(hedr_data)

    # Build records
    records = [
        ("WEAP", 0x00001000, [
            ("EDID", make_string("TestWeapon")),
            ("FULL", make_string("Iron Sword")),
        ]),
        ("ARMO", 0x00001001, [
            ("EDID", make_string("TestArmor")),
            ("FULL", make_string("Leather Armor")),
            ("DESC", make_string("A sturdy set of leather armor.")),
        ]),
        ("BOOK", 0x00001002, [
            ("EDID", make_string("TestBook")),
            ("FULL", make_string("Wasteland Survival Guide")),
            ("DESC", make_string("A guide to surviving the wasteland.")),
        ]),
    ]

    with open(path, "wb") as f:
        # TES4 header
        f.write(b"TES4")
        write_uint32(f, tes4_data_size)
        write_uint32(f, 0)
        write_uint32(f, 0)
        write_uint32(f, 0)
        write_uint32(f, 0)
        write_subrecord(f, "HEDR", hedr_data)

        # One GRUP per record type
        for rec_type, form_id, subs in records:
            rec_data_size = sum(6 + len(d) for _, d in subs)
            grup_size = 24 + 24 + rec_data_size

            # GRUP header
            f.write(b"GRUP")
            write_uint32(f, grup_size)
            f.write(rec_type.encode("ascii"))
            write_uint32(f, 0)
            write_uint32(f, 0)
            write_uint32(f, 0)

            # Record header (24 bytes FO3)
            f.write(rec_type.encode("ascii"))
            write_uint32(f, rec_data_size)
            write_uint32(f, 0)
            write_uint32(f, form_id)
            write_uint32(f, 0)
            write_uint32(f, 0)

            # Subrecords
            for tag, data in subs:
                write_subrecord(f, tag, data)

    print(f"Created {path} ({path.stat().st_size} bytes)")


def create_compressed_record_esp() -> None:
    """Create an ESP with a compressed record (flag 0x00040000)."""
    path = FIXTURES_DIR / "compressed.esp"

    hedr_data = struct.pack("<f", 0.94) + struct.pack("<I", 0) + struct.pack("<I", 0x000800)
    tes4_data_size = 6 + len(hedr_data)

    # Build the WEAP subrecords as raw bytes (uncompressed)
    edid_data = make_string("CompressedWeapon")
    full_data = make_string("Plasma Rifle")

    raw_subrecords = b""
    raw_subrecords += b"EDID" + struct.pack("<H", len(edid_data)) + edid_data
    raw_subrecords += b"FULL" + struct.pack("<H", len(full_data)) + full_data

    # Compress
    compressed = zlib.compress(raw_subrecords)
    # Record data = 4 bytes decompressed size + compressed data
    record_data = struct.pack("<I", len(raw_subrecords)) + compressed
    record_data_size = len(record_data)

    grup_size = 24 + 24 + record_data_size

    with open(path, "wb") as f:
        # TES4 header
        f.write(b"TES4")
        write_uint32(f, tes4_data_size)
        write_uint32(f, 0)
        write_uint32(f, 0)
        write_uint32(f, 0)
        write_uint32(f, 0)
        write_subrecord(f, "HEDR", hedr_data)

        # GRUP
        f.write(b"GRUP")
        write_uint32(f, grup_size)
        f.write(b"WEAP")
        write_uint32(f, 0)
        write_uint32(f, 0)
        write_uint32(f, 0)

        # WEAP record with COMPRESSED flag
        f.write(b"WEAP")
        write_uint32(f, record_data_size)
        write_uint32(f, 0x00040000)  # COMPRESSED flag
        write_uint32(f, 0x00002000)
        write_uint32(f, 0)
        write_uint32(f, 0)
        f.write(record_data)

    print(f"Created {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_minimal_fo3_esp()
    create_multi_record_esp()
    create_compressed_record_esp()
    print("All fixtures created.")
