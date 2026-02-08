"""Tests for the editor-ID heuristic filter in the extractor."""

import pytest

from modtranslator.translation.extractor import _looks_like_editor_id, extract_strings
from tests.conftest import make_plugin, make_subrecord


class TestLooksLikeEditorId:
    """Unit tests for _looks_like_editor_id()."""

    # --- Should be detected as editor IDs ---

    @pytest.mark.parametrize("text", [
        "00MSAOlevRescueTopic03A",
        "00MSDrNylusDeathRayDamocles02",
        "01DLC01VaultSuit",
    ])
    def test_hex_prefix(self, text):
        assert _looks_like_editor_id(text) is True

    @pytest.mark.parametrize("text", [
        "DrNylusDeathRay",
        "RadAway",
        "StealthBoy",
        "MegaCityQuest",
        "vDialogueMoira",
    ])
    def test_camel_case(self, text):
        assert _looks_like_editor_id(text) is True

    @pytest.mark.parametrize("text", [
        "Player_Fallback",
        "NPC_Greeting_01",
        "Quest_Stage_20",
    ])
    def test_underscores(self, text):
        assert _looks_like_editor_id(text) is True

    @pytest.mark.parametrize("text", [
        "YOURSERVICESMAAM01",
        "YOURSERVICES02",
        "YOURSERVICESMAAM123",
    ])
    def test_long_alphanumeric(self, text):
        assert _looks_like_editor_id(text) is True

    # --- Should NOT be detected (real translatable text) ---

    @pytest.mark.parametrize("text", [
        "Iron Sword",
        "A sturdy set of leather armor.",
        "Do you have any work for me?",
        "Welcome to Megaton!",
        "You need to find three items.",
    ])
    def test_real_text_with_spaces(self, text):
        assert _looks_like_editor_id(text) is False

    @pytest.mark.parametrize("text", [
        "DANGER",
        "EXIT",
        "WARNING",
        "Stimpak",
        "Nuka-Cola",
    ])
    def test_short_labels(self, text):
        assert _looks_like_editor_id(text) is False

    def test_empty_string(self):
        assert _looks_like_editor_id("") is False

    def test_whitespace_only(self):
        assert _looks_like_editor_id("   ") is False


class TestEditorIdFilterIntegration:
    """Integration tests: editor-ID strings are skipped during extraction."""

    def test_editor_id_as_full_not_extracted(self):
        """DIAL with an editor-ID-like FULL should NOT be extracted."""
        plugin = make_plugin([
            ("DIAL", 0x200, [
                make_subrecord("EDID", "SomeDialogue"),
                make_subrecord("FULL", "00MSAOlevRescueTopic03A"),
            ]),
        ])
        strings = extract_strings(plugin)
        texts = {s.original_text for s in strings}
        assert "00MSAOlevRescueTopic03A" not in texts

    def test_real_text_full_extracted(self):
        """DIAL with real text as FULL should be extracted."""
        plugin = make_plugin([
            ("DIAL", 0x201, [
                make_subrecord("EDID", "SomeDialogue"),
                make_subrecord("FULL", "Do you need anything?"),
            ]),
        ])
        strings = extract_strings(plugin)
        texts = {s.original_text for s in strings}
        assert "Do you need anything?" in texts

    def test_weapon_name_still_extracted(self):
        """Normal weapon names pass the filter."""
        plugin = make_plugin([
            ("WEAP", 0x300, [
                make_subrecord("EDID", "SomeWeapon"),
                make_subrecord("FULL", "Plasma Rifle"),
            ]),
        ])
        strings = extract_strings(plugin)
        assert len(strings) == 1
        assert strings[0].original_text == "Plasma Rifle"

    def test_camel_case_id_filtered(self):
        """CamelCase IDs appearing as FULL are filtered out."""
        plugin = make_plugin([
            ("WEAP", 0x400, [
                make_subrecord("EDID", "InternalWeapon"),
                make_subrecord("FULL", "DrNylusDeathRay"),
            ]),
        ])
        strings = extract_strings(plugin)
        assert len(strings) == 0

    def test_underscore_id_filtered(self):
        """Underscore-style IDs appearing as FULL are filtered out."""
        plugin = make_plugin([
            ("WEAP", 0x500, [
                make_subrecord("EDID", "Fallback"),
                make_subrecord("FULL", "Player_Fallback"),
            ]),
        ])
        strings = extract_strings(plugin)
        assert len(strings) == 0

    def test_mixed_records_filter_selective(self):
        """Only editor-ID-like strings are filtered; real text passes."""
        plugin = make_plugin([
            ("WEAP", 0x600, [
                make_subrecord("EDID", "RealWeapon"),
                make_subrecord("FULL", "Hunting Rifle"),
            ]),
            ("WEAP", 0x601, [
                make_subrecord("EDID", "FakeWeapon"),
                make_subrecord("FULL", "vDialogueWeaponTest"),
            ]),
        ])
        strings = extract_strings(plugin)
        texts = {s.original_text for s in strings}
        assert "Hunting Rifle" in texts
        assert "vDialogueWeaponTest" not in texts
