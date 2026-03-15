"""Tests for the translation registry."""

from modtranslator.translation.registry import get_translatable_subrecord_types, is_translatable


class TestIsTranslatable:
    def test_never_translate_edid(self):
        """EDID is always excluded regardless of record type."""
        assert not is_translatable(b"WEAP", b"EDID")
        assert not is_translatable(b"NPC_", b"EDID")

    def test_never_translate_modl(self):
        assert not is_translatable(b"ARMO", b"MODL")

    def test_never_translate_icon(self):
        assert not is_translatable(b"ALCH", b"ICON")

    def test_unknown_subrecord_type(self):
        """Unknown subrecord type returns False."""
        assert not is_translatable(b"WEAP", b"XXXX")
        assert not is_translatable(b"WEAP", b"ZZZZ")
        assert not is_translatable(b"WEAP", b"DATA")

    def test_full_in_allowed_record(self):
        assert is_translatable(b"WEAP", b"FULL")
        assert is_translatable(b"ARMO", b"FULL")
        assert is_translatable(b"NPC_", b"FULL")

    def test_full_in_non_allowed_record(self):
        """FULL in a record type not in _FULL_ALLOWED_RECORDS returns False."""
        assert not is_translatable(b"ARMA", b"FULL")
        assert not is_translatable(b"PROJ", b"FULL")

    def test_desc_in_allowed_record(self):
        assert is_translatable(b"BOOK", b"DESC")
        assert is_translatable(b"WEAP", b"DESC")

    def test_desc_in_non_allowed_record(self):
        assert not is_translatable(b"DOOR", b"DESC")

    def test_dual_purpose_fact_rnam_false(self):
        """FACT+RNAM is uint32, not text — must return False."""
        assert not is_translatable(b"FACT", b"RNAM")

    def test_rnam_in_info(self):
        """INFO+RNAM is a dialog prompt — translatable."""
        assert is_translatable(b"INFO", b"RNAM")

    def test_rnam_in_term(self):
        assert is_translatable(b"TERM", b"RNAM")

    def test_rnam_in_acti(self):
        assert is_translatable(b"ACTI", b"RNAM")


class TestSkyrimSpecificCombos:
    def test_woop_dnam(self):
        assert is_translatable(b"WOOP", b"DNAM")

    def test_shou_shrt(self):
        assert is_translatable(b"SHOU", b"SHRT")

    def test_npc_shrt(self):
        assert is_translatable(b"NPC_", b"SHRT")

    def test_acti_rnam(self):
        assert is_translatable(b"ACTI", b"RNAM")

    def test_woop_full(self):
        assert is_translatable(b"WOOP", b"FULL")

    def test_shou_full(self):
        assert is_translatable(b"SHOU", b"FULL")

    def test_lctn_full(self):
        assert is_translatable(b"LCTN", b"FULL")


class TestFo4SpecificCombos:
    def test_omod_full(self):
        assert is_translatable(b"OMOD", b"FULL")

    def test_cmpo_full(self):
        assert is_translatable(b"CMPO", b"FULL")

    def test_innr_full(self):
        assert is_translatable(b"INNR", b"FULL")

    def test_dmgt_full(self):
        assert is_translatable(b"DMGT", b"FULL")


class TestGetTranslatableSubrecordTypes:
    def test_returns_expected_types(self):
        types = get_translatable_subrecord_types()
        assert b"FULL" in types
        assert b"DESC" in types
        assert b"NAM1" in types
        assert b"RNAM" in types
        assert b"DNAM" in types
        assert b"SHRT" in types

    def test_never_translate_not_in_list(self):
        types = get_translatable_subrecord_types()
        assert b"EDID" not in types
        assert b"MODL" not in types
        assert b"ICON" not in types
