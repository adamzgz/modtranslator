"""Tests for generic target-language word protection in mixed-language strings."""

import pytest

from modtranslator.translation.target_protect import (
    _normalize_placeholders,
    protect_target_batch,
    protect_target_words,
    restore_target_batch,
    restore_target_words,
)

# === Parametrized tests across languages ===

# (lang, word_in_dictionary, sample_text_with_target_word)
_LANG_SAMPLES = [
    ("ES", "espada", "Find the espada in the cave"),
    ("FR", "épée", "Find the épée in the cave"),
    ("DE", "schwert", "Find the Schwert in the cave"),
    ("IT", "spada", "Find the spada in the cave"),
    ("PT", "espada", "Find the espada in the cave"),
    ("RU", "найти", "Go and найти the key"),
    ("PL", "miecz", "Find the miecz in the cave"),
]


class TestProtectTargetWords:
    def test_pure_english_no_protection(self):
        for lang in ["ES", "FR", "DE", "IT", "PT", "RU", "PL"]:
            text = "Iron Sword of the Wasteland"
            protected, mapping = protect_target_words(text, lang)
            assert mapping == {}, f"Unexpected protection for lang={lang}"
            assert protected == text

    @pytest.mark.parametrize("lang,word,text", _LANG_SAMPLES)
    def test_target_word_gets_placeholder(self, lang: str, word: str, text: str):
        protected, mapping = protect_target_words(text, lang)
        assert word.lower() not in protected.lower(), f"{word} should be protected for lang={lang}"
        assert "Cx" in protected
        assert any(v.lower() == word.lower() for v in mapping.values())

    @pytest.mark.parametrize("lang,word,text", _LANG_SAMPLES)
    def test_roundtrip(self, lang: str, word: str, text: str):
        protected, mapping = protect_target_words(text, lang)
        restored = restore_target_words(protected, mapping)
        assert restored == text

    def test_short_words_ignored(self):
        """Words shorter than 4 characters are not protected."""
        text = "el sol está aquí"
        protected, mapping = protect_target_words(text, "ES")
        # "sol" has 3 chars -> not protected
        assert "sol" in protected

    def test_glossary_placeholders_not_touched(self):
        text = "Take the Gx0 and espada"
        protected, mapping = protect_target_words(text, "ES")
        assert "Gx0" in protected
        assert "espada" not in protected

    def test_empty_string(self):
        protected, mapping = protect_target_words("", "FR")
        assert protected == ""
        assert mapping == {}

    def test_placeholder_format(self):
        text = "Take the espada"
        protected, mapping = protect_target_words(text, "ES")
        assert any(ph.startswith("Cx") for ph in mapping)


class TestRestoreTargetWords:
    def test_restore_simple(self):
        text = "Take the Cx0 from the chest"
        mapping = {"Cx0": "espada"}
        restored = restore_target_words(text, mapping)
        assert restored == "Take the espada from the chest"

    def test_restore_multiple(self):
        mapping = {"Cx0": "cuchillo", "Cx1": "escudo"}
        text = "Use Cx1 and Cx0"
        restored = restore_target_words(text, mapping)
        assert "cuchillo" in restored
        assert "escudo" in restored

    def test_restore_empty_mapping(self):
        text = "No placeholders here"
        restored = restore_target_words(text, {})
        assert restored == text


class TestProtectTargetBatch:
    @pytest.mark.parametrize("lang", ["ES", "FR", "DE", "IT", "PT", "RU", "PL"])
    def test_batch_empty(self, lang: str):
        protected, mappings = protect_target_batch([], lang)
        assert protected == []
        assert mappings == []

    def test_batch_roundtrip_es(self):
        texts = [
            "Take the espada and cuchillo",
            "Pure English sentence here",
            "Find the refugio near town",
        ]
        protected, mappings = protect_target_batch(texts, "ES")
        restored = restore_target_batch(protected, mappings)
        assert restored == texts

    def test_batch_independent_mappings(self):
        texts = [
            "Take the espada",
            "Find the escudo",
            "No Spanish here",
        ]
        protected, mappings = protect_target_batch(texts, "ES")
        assert len(protected) == 3
        assert len(mappings) == 3
        assert "espada" not in protected[0]
        assert "escudo" not in protected[1]
        assert mappings[2] == {}


class TestNormalizePlaceholders:
    def test_space_inserted(self):
        mapping = {"Cx0": "espada"}
        result = _normalize_placeholders("Take the Cx 0", mapping)
        assert "Cx0" in result

    def test_case_change(self):
        mapping = {"Cx0": "espada"}
        result = _normalize_placeholders("Take the cx0", mapping)
        assert "Cx0" in result

    def test_already_correct(self):
        mapping = {"Cx0": "espada"}
        text = "Take the Cx0"
        result = _normalize_placeholders(text, mapping)
        assert result == text

    def test_restore_with_mangled(self):
        mapping = {"Cx0": "espada"}
        restored = restore_target_words("Take the cx0", mapping)
        assert restored == "Take the espada"
