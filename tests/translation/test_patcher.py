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


class TestPatcherEdgeCases:
    def test_empty_strings_list(self):
        """apply_translations with empty strings list returns 0."""
        patched = apply_translations([], {"some_key": "some_value"})
        assert patched == 0

    def test_localized_string_id_without_string_tables(self):
        """Localized string (string_id set) but string_tables=None falls to inline."""
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
        assert strings[0].string_id == 42

        # Pass string_tables=None — the patcher should still patch (inline path)
        translations = {strings[0].key: "Espada de Hierro"}
        patched = apply_translations(strings, translations, string_tables=None)
        # Falls to inline encode_string path since string_tables is None
        assert patched == 1

    def test_localized_string_id_not_in_any_table(self):
        """Localized string where string_id doesn't exist in any table → patched count = 0."""
        from modtranslator.translation.extractor import TranslatableString

        sts = StringTableSet()
        sts.strings = StringTable(StringTableType.STRINGS, {99: "Other"})
        sts.build_merged()

        # Construct a TranslatableString with string_id=42 (not in any table)
        sub = make_string_id_subrecord("FULL", 42)
        ts = TranslatableString(
            record_type=b"WEAP",
            subrecord_type=b"FULL",
            form_id=0x100,
            original_text="Missing Text",
            subrecord=sub,
            string_id=42,
        )

        translations = {ts.key: "Espada"}
        patched = apply_translations([ts], translations, string_tables=sts)
        assert patched == 0

    def test_encode_string_fallback_non_cp1252(self):
        """Text with non-cp1252 chars (e.g. emoji) triggers UTF-8 fallback in encode_string."""
        plugin = make_plugin([
            ("WEAP", 0x100, [make_subrecord("FULL", "Sword")]),
        ])
        strings = extract_strings(plugin)
        translations = {strings[0].key: "Espada \U0001f5e1"}  # 🗡 emoji
        patched = apply_translations(strings, translations)
        assert patched == 1
        # Should have fallen back to UTF-8 encoding
        sub = strings[0].subrecord
        raw = bytes(sub.data).rstrip(b"\x00")
        decoded = raw.decode("utf-8")
        assert "\U0001f5e1" in decoded


class TestFo4Patcher:
    def test_patch_fo4_localized_updates_string_table(self):
        """Patching a FO4 localized string updates the string table."""
        from tests.conftest import make_fo4_plugin
        sts = StringTableSet()
        sts.strings = StringTable(StringTableType.STRINGS, {55: "10mm Pistol"})
        sts.build_merged()

        plugin = make_fo4_plugin(
            records=[("WEAP", 0x100, [
                make_string_id_subrecord("FULL", 55),
            ])],
            localized=True,
            string_tables=sts,
        )

        strings = extract_strings(plugin)
        assert len(strings) == 1

        translations = {strings[0].key: "Pistola 10mm"}
        patched = apply_translations(strings, translations, string_tables=sts)

        assert patched == 1
        assert sts.strings.entries[55] == "Pistola 10mm"
        # Subrecord still holds the 4-byte StringID
        assert strings[0].subrecord.size == 4

    def test_patch_fo4_inline_mutates_subrecord(self):
        """Non-localized FO4 plugin patches subrecords directly."""
        from tests.conftest import make_fo4_plugin
        plugin = make_fo4_plugin(
            records=[("OMOD", 0x100, [
                make_subrecord("FULL", "Rifled Barrel"),
            ])],
            localized=False,
        )

        strings = extract_strings(plugin)
        assert len(strings) == 1

        translations = {strings[0].key: "Canon Estriado"}
        patched = apply_translations(strings, translations)

        assert patched == 1
        text_in_data = strings[0].subrecord.data[:-1].decode("cp1252")
        assert text_in_data == "Canon Estriado"

    def test_patch_fo4_dlstrings(self):
        """FO4 DESC (long description) stored in DLSTRINGS is patched correctly."""
        from tests.conftest import make_fo4_plugin
        sts = StringTableSet()
        sts.dlstrings = StringTable(StringTableType.DLSTRINGS, {20: "A powerful pistol."})
        sts.build_merged()

        plugin = make_fo4_plugin(
            records=[("WEAP", 0x100, [
                make_string_id_subrecord("DESC", 20),
            ])],
            localized=True,
            string_tables=sts,
        )

        strings = extract_strings(plugin)
        assert len(strings) == 1

        translations = {strings[0].key: "Una potente pistola."}
        patched = apply_translations(strings, translations, string_tables=sts)

        assert patched == 1
        assert sts.dlstrings.entries[20] == "Una potente pistola."
