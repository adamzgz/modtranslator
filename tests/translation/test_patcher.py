"""Tests for the translation patcher."""

from modtranslator.core.string_table import StringTable, StringTableSet, StringTableType
from modtranslator.translation.extractor import extract_strings
from modtranslator.translation.patcher import apply_translations
from tests.conftest import (
    make_plugin,
    make_skyrim_plugin,
    make_string_id_subrecord,
    make_subrecord,
)


class TestPatcher:
    def test_apply_single_translation(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        translations = {strings[0].key: "Espada de Hierro"}

        patched = apply_translations(strings, translations)
        assert patched == 1

        # Verify the subrecord was mutated
        rec = simple_plugin.groups[0].children[0]
        full_sub = next(s for s in rec.subrecords if s.type == b"FULL")
        assert full_sub.decode_string() == "Espada de Hierro"

    def test_apply_multiple_translations(self, multi_record_plugin):
        strings = extract_strings(multi_record_plugin)
        translations = {s.key: f"[ES] {s.original_text}" for s in strings}

        patched = apply_translations(strings, translations)
        assert patched == len(strings)

    def test_skip_missing_translations(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        # Empty translations dict → nothing patched
        patched = apply_translations(strings, {})
        assert patched == 0

    def test_skip_identical_translation(self, simple_plugin):
        strings = extract_strings(simple_plugin)
        # Same text → no patch
        translations = {strings[0].key: "Iron Sword"}
        patched = apply_translations(strings, translations)
        assert patched == 0

    def test_subrecord_size_changes(self):
        plugin = make_plugin([
            ("WEAP", 0x100, [
                make_subrecord("FULL", "Hi"),
            ]),
        ])
        strings = extract_strings(plugin)
        original_size = strings[0].subrecord.size

        apply_translations(strings, {strings[0].key: "A much longer translated string"})
        new_size = strings[0].subrecord.size
        assert new_size > original_size


class TestSkyrimPatcher:
    def test_patch_localized_updates_string_table(self):
        """Patching a localized string updates the string table, not the subrecord."""
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

        translations = {strings[0].key: "Espada de Hierro"}
        patched = apply_translations(strings, translations, string_tables=sts)

        assert patched == 1
        assert sts.strings.entries[42] == "Espada de Hierro"
        # Subrecord should still contain the original StringID (4 bytes)
        assert strings[0].subrecord.size == 4

    def test_patch_skyrim_inline_mutates_subrecord(self):
        """Non-localized Skyrim plugin patches subrecords (like FO3)."""
        plugin = make_skyrim_plugin(
            records=[("WEAP", 0x100, [
                make_subrecord("EDID", "TestSword"),
                make_subrecord("FULL", "Iron Sword"),
            ])],
            localized=False,
        )

        strings = extract_strings(plugin)
        translations = {strings[0].key: "Espada de Hierro"}
        patched = apply_translations(strings, translations)

        assert patched == 1
        rec = plugin.groups[0].children[0]
        full_sub = next(s for s in rec.subrecords if s.type == b"FULL")
        assert full_sub.decode_string() == "Espada de Hierro"

    def test_patch_localized_dlstrings(self):
        """Patching works for strings in DLSTRINGS table too."""
        sts = StringTableSet()
        sts.dlstrings = StringTable(StringTableType.DLSTRINGS, {10: "A fine weapon."})
        sts.build_merged()

        plugin = make_skyrim_plugin(
            records=[("WEAP", 0x100, [
                make_subrecord("EDID", "TestSword"),
                make_string_id_subrecord("DESC", 10),
            ])],
            localized=True,
            string_tables=sts,
        )

        strings = extract_strings(plugin)
        assert len(strings) == 1

        translations = {strings[0].key: "Un arma fina."}
        patched = apply_translations(strings, translations, string_tables=sts)

        assert patched == 1
        assert sts.dlstrings.entries[10] == "Un arma fina."
