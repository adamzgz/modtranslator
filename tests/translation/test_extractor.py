"""Tests for string extraction."""

from modtranslator.core.string_table import StringTable, StringTableSet, StringTableType
from modtranslator.translation.extractor import extract_strings
from tests.conftest import (
    make_plugin,
    make_skyrim_plugin,
    make_string_id_subrecord,
    make_subrecord,
)


class TestExtractor:
    def test_extract_full_subrecord(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        assert len(strings) == 1
        assert strings[0].original_text == "Iron Sword"
        assert strings[0].subrecord_type == b"FULL"

    def test_extract_multiple_records(self, multi_record_plugin):
        strings = extract_strings(multi_record_plugin)
        texts = {s.original_text for s in strings}
        assert "Iron Sword" in texts
        assert "Leather Armor" in texts
        assert "A sturdy set of leather armor." in texts
        assert "Wasteland Survival Guide" in texts

    def test_skips_edid(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        sub_types = {s.subrecord_type for s in strings}
        assert b"EDID" not in sub_types

    def test_desc_extracted_from_book(self):
        plugin = make_plugin([
            ("BOOK", 0x100, [
                make_subrecord("EDID", "MyBook"),
                make_subrecord("FULL", "Book Title"),
                make_subrecord("DESC", "Book description text."),
            ]),
        ])
        strings = extract_strings(plugin)
        texts = {s.original_text for s in strings}
        assert "Book Title" in texts
        assert "Book description text." in texts

    def test_desc_not_extracted_from_non_allowed_record(self):
        """DESC should not be extracted from record types not in the registry."""
        plugin = make_plugin([
            ("STAT", 0x100, [
                make_subrecord("EDID", "MyStatic"),
                make_subrecord("DESC", "This should not be extracted."),
            ]),
        ])
        strings = extract_strings(plugin)
        assert len(strings) == 0

    def test_form_id_preserved(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        assert strings[0].form_id == 0x00001000

    def test_editor_id_captured(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        assert strings[0].editor_id == "TestWeapon"

    def test_key_format(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        assert strings[0].key == "00001000:FULL:0"

    def test_key_format_with_source_file(self, simple_plugin):
        """Cache key includes source filename to prevent cross-mod collisions."""
        strings = extract_strings(simple_plugin)
        strings[0].source_file = "MyMod"
        assert strings[0].key == "MyMod:00001000:FULL:0"

    def test_subrecord_reference_is_direct(self, simple_plugin):
        """The TranslatableString holds a reference to the actual Subrecord."""
        strings = extract_strings(simple_plugin)
        ts = strings[0]
        # Mutate via the reference
        ts.subrecord.encode_string("Modified")
        # Verify the plugin's subrecord was actually changed
        rec = simple_plugin.groups[0].children[0]
        full_sub = next(s for s in rec.subrecords if s.type == b"FULL")
        assert full_sub.decode_string() == "Modified"

    def test_full_not_extracted_from_internal_record_types(self):
        """FULL should not be extracted from record types that are internal/
        never visible to the player (explosions, projectiles, etc.)."""
        internal_types = ["EXPL", "PROJ", "WATR", "ARMA"]
        for rec_type in internal_types:
            plugin = make_plugin([
                (rec_type, 0x100, [
                    make_subrecord("EDID", f"Test{rec_type}"),
                    make_subrecord("FULL", "Some Internal Name"),
                ]),
            ])
            strings = extract_strings(plugin)
            assert len(strings) == 0, (
                f"FULL in {rec_type} should not be extracted (internal record type)"
            )

    def test_full_extracted_from_visible_record_types(self):
        """FULL should be extracted from record types visible to the player."""
        visible_types = ["WEAP", "ARMO", "ALCH", "NPC_", "CONT", "DOOR", "ACTI",
                         "HDPT", "MSTT", "REFR"]
        for rec_type in visible_types:
            plugin = make_plugin([
                (rec_type, 0x100, [
                    make_subrecord("EDID", f"Test{rec_type}"),
                    make_subrecord("FULL", "Visible Name"),
                ]),
            ])
            strings = extract_strings(plugin)
            assert len(strings) == 1, (
                f"FULL in {rec_type} should be extracted (player-visible)"
            )
            assert strings[0].original_text == "Visible Name"

    def test_desc_extracted_from_terminal(self):
        """DESC in TERM records is the terminal header text, visible to player."""
        plugin = make_plugin([
            ("TERM", 0x100, [
                make_subrecord("EDID", "MyTerminal"),
                make_subrecord("FULL", "Terminal Name"),
                make_subrecord("DESC", "Vault-Tec Corporate Headquarters"),
            ]),
        ])
        strings = extract_strings(plugin)
        texts = {s.original_text for s in strings}
        assert "Terminal Name" in texts
        assert "Vault-Tec Corporate Headquarters" in texts

    def test_desc_extracted_from_avif(self):
        """DESC in AVIF records is the SPECIAL/skill description, visible in Pip-Boy."""
        plugin = make_plugin([
            ("AVIF", 0x100, [
                make_subrecord("EDID", "AVEndurance"),
                make_subrecord("FULL", "Endurance"),
                make_subrecord("DESC", "Endurance is a measure of your overall fitness."),
            ]),
        ])
        strings = extract_strings(plugin)
        texts = {s.original_text for s in strings}
        assert "Endurance" in texts
        assert "Endurance is a measure of your overall fitness." in texts

    def test_refr_map_marker_extracted(self):
        """FULL in REFR records contains map marker names visible on Pip-Boy."""
        plugin = make_plugin([
            ("REFR", 0x100, [
                make_subrecord("FULL", "Vault-Tec Hideout"),
            ]),
        ])
        strings = extract_strings(plugin)
        assert len(strings) == 1
        assert strings[0].original_text == "Vault-Tec Hideout"

    def test_hdpt_extracted(self):
        """FULL in HDPT records contains beard/hair names in character creation."""
        plugin = make_plugin([
            ("HDPT", 0x100, [
                make_subrecord("EDID", "BeardPirate"),
                make_subrecord("FULL", "The Pirate"),
            ]),
        ])
        strings = extract_strings(plugin)
        assert len(strings) == 1
        assert strings[0].original_text == "The Pirate"

    def test_empty_string_skipped(self):
        plugin = make_plugin([
            ("WEAP", 0x100, [
                make_subrecord("FULL", b"\x00"),  # empty null-terminated
            ]),
        ])
        strings = extract_strings(plugin)
        assert len(strings) == 0


class TestSkyrimExtractor:
    def test_skyrim_inline_extraction(self):
        """Skyrim plugin without LOCALIZED flag extracts strings inline (like FO3)."""
        plugin = make_skyrim_plugin(
            records=[("WEAP", 0x100, [
                make_subrecord("EDID", "TestSword"),
                make_subrecord("FULL", "Iron Sword"),
            ])],
            localized=False,
        )
        strings = extract_strings(plugin)
        assert len(strings) == 1
        assert strings[0].original_text == "Iron Sword"
        assert strings[0].string_id is None

    def test_localized_extraction(self):
        """Localized plugin reads text from string tables via StringID."""
        sts = StringTableSet()
        sts.strings = StringTable(StringTableType.STRINGS, {42: "Iron Sword"})
        sts.build_merged()

        plugin = make_skyrim_plugin(
            records=[("WEAP", 0x100, [
                make_subrecord("EDID", "TestSword"),
                make_string_id_subrecord("FULL", 42),
            ])],
            localized=True,
            string_tables=sts,
        )
        strings = extract_strings(plugin)
        assert len(strings) == 1
        assert strings[0].original_text == "Iron Sword"
        assert strings[0].string_id == 42

    def test_localized_string_id_zero_skipped(self):
        """StringID 0 means no string — should be skipped."""
        sts = StringTableSet()
        sts.build_merged()

        plugin = make_skyrim_plugin(
            records=[("WEAP", 0x100, [
                make_string_id_subrecord("FULL", 0),
            ])],
            localized=True,
            string_tables=sts,
        )
        strings = extract_strings(plugin)
        assert len(strings) == 0

    def test_localized_missing_string_id_skipped(self):
        """StringID not found in string tables should be skipped."""
        sts = StringTableSet()
        sts.strings = StringTable(StringTableType.STRINGS, {1: "Hello"})
        sts.build_merged()

        plugin = make_skyrim_plugin(
            records=[("WEAP", 0x100, [
                make_string_id_subrecord("FULL", 999),  # not in tables
            ])],
            localized=True,
            string_tables=sts,
        )
        strings = extract_strings(plugin)
        assert len(strings) == 0

    def test_skyrim_flora_extracted(self):
        """FLOR record FULL should be extracted (Skyrim-specific)."""
        plugin = make_skyrim_plugin(
            records=[("FLOR", 0x100, [
                make_subrecord("EDID", "TestFlora"),
                make_subrecord("FULL", "Mountain Flower"),
            ])],
        )
        strings = extract_strings(plugin)
        assert len(strings) == 1
        assert strings[0].original_text == "Mountain Flower"

    def test_skyrim_woop_dnam_extracted(self):
        """DNAM in WOOP records should be extracted (Word of Power text)."""
        plugin = make_skyrim_plugin(
            records=[("WOOP", 0x100, [
                make_subrecord("EDID", "TestWord"),
                make_subrecord("FULL", "Force"),
                make_subrecord("DNAM", "Fus"),
            ])],
        )
        strings = extract_strings(plugin)
        texts = {s.original_text for s in strings}
        assert "Force" in texts
        assert "Fus" in texts
