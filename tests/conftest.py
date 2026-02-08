"""Shared test fixtures for modtranslator tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from modtranslator.core.constants import Game
from modtranslator.core.records import GroupRecord, PluginFile, Record, Subrecord

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_subrecord(type_tag: str, data: bytes | str) -> Subrecord:
    """Create a Subrecord from a string type and data."""
    if isinstance(data, str):
        data = data.encode("cp1252") + b"\x00"
    return Subrecord(type=type_tag.encode("ascii"), data=bytearray(data))


def make_tes4_header(version: float = 0.94) -> Record:
    """Create a minimal TES4 header record."""
    hedr_data = struct.pack("<f", version) + struct.pack("<I", 0) + struct.pack("<I", 0)
    return Record(
        type=b"TES4",
        flags=0,
        form_id=0,
        vcs1=0,
        vcs2=0,
        subrecords=[
            Subrecord(type=b"HEDR", data=bytearray(hedr_data)),
        ],
    )


def make_record(
    type_tag: str,
    form_id: int,
    subrecords: list[Subrecord] | None = None,
    flags: int = 0,
) -> Record:
    """Create a Record with the given type, form_id, and subrecords."""
    return Record(
        type=type_tag.encode("ascii"),
        flags=flags,
        form_id=form_id,
        vcs1=0,
        vcs2=0,
        subrecords=subrecords or [],
    )


def make_group(label: str, children: list[Record | GroupRecord] | None = None) -> GroupRecord:
    """Create a GroupRecord with the given label."""
    return GroupRecord(
        label=label.encode("ascii").ljust(4, b"\x00")[:4],
        group_type=0,
        stamp=0,
        vcs=0,
        children=children or [],
    )


def make_plugin(
    records: list[tuple[str, int, list[Subrecord]]] | None = None,
    version: float = 0.94,
) -> PluginFile:
    """Build a minimal valid PluginFile.

    Args:
        records: List of (record_type, form_id, subrecords) tuples.
                 They will all be placed in a single GRUP.
        version: HEDR version float.
    """
    header = make_tes4_header(version)
    plugin = PluginFile(header=header, groups=[], game=Game.FALLOUT3)

    if records:
        children = []
        for rec_type, form_id, subs in records:
            children.append(make_record(rec_type, form_id, subs))
        group = make_group(records[0][0][:4], children)
        plugin.groups.append(group)

    return plugin


@pytest.fixture
def simple_plugin() -> PluginFile:
    """A minimal plugin with a WEAP record containing EDID and FULL."""
    return make_plugin([
        ("WEAP", 0x00001000, [
            make_subrecord("EDID", "TestWeapon"),
            make_subrecord("FULL", "Iron Sword"),
        ]),
    ])


@pytest.fixture
def multi_record_plugin() -> PluginFile:
    """A plugin with multiple record types for translation testing."""
    return make_plugin([
        ("WEAP", 0x00001000, [
            make_subrecord("EDID", "TestWeapon"),
            make_subrecord("FULL", "Iron Sword"),
        ]),
        ("ARMO", 0x00001001, [
            make_subrecord("EDID", "TestArmor"),
            make_subrecord("FULL", "Leather Armor"),
            make_subrecord("DESC", "A sturdy set of leather armor."),
        ]),
        ("BOOK", 0x00001002, [
            make_subrecord("EDID", "TestBook"),
            make_subrecord("FULL", "Wasteland Survival Guide"),
            make_subrecord("DESC", "A guide to surviving the wasteland."),
        ]),
    ])


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def tmp_cache(tmp_path: Path):
    """Create a temporary translation cache."""
    from modtranslator.translation.cache import TranslationCache
    cache = TranslationCache(db_path=tmp_path / "test_cache.db")
    yield cache
    cache.close()
