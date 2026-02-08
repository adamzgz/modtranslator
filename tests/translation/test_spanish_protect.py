"""Tests for Spanish word protection in mixed-language strings."""

from modtranslator.translation.spanish_protect import (
    _normalize_es_placeholders,
    protect_spanish_batch,
    protect_spanish_words,
    restore_spanish_batch,
    restore_spanish_words,
)


class TestProtectSpanishWords:
    def test_pure_english_no_protection(self):
        text = "Iron Sword of the Wasteland"
        protected, mapping = protect_spanish_words(text)
        assert mapping == {}
        assert protected == text

    def test_spanish_word_gets_placeholder(self):
        text = "Find the espada in the cave"
        protected, mapping = protect_spanish_words(text)
        assert "espada" not in protected
        assert "Cx" in protected
        assert any(v == "espada" for v in mapping.values())

    def test_multiple_spanish_words(self):
        text = "Take the escudo and cuchillo from the chest"
        protected, mapping = protect_spanish_words(text)
        assert "escudo" not in protected
        assert "cuchillo" not in protected
        assert len(mapping) == 2

    def test_short_words_ignored(self):
        """Words shorter than 4 characters are not protected."""
        text = "el sol est\u00e1 aqu\u00ed"
        protected, mapping = protect_spanish_words(text)
        # "sol" has 3 chars -> not protected; "esta" and "aqui" have 4+ -> protected
        assert "sol" in protected

    def test_accented_word_detected(self):
        """Words with Spanish accents are detected even if not in dictionary."""
        text = "Theci\u00f3n was rare"
        protected, mapping = protect_spanish_words(text)
        assert "ci\u00f3n" not in protected
        assert len(mapping) >= 1

    def test_glossary_placeholders_not_touched(self):
        text = "Take the Gx0 and espada"
        protected, mapping = protect_spanish_words(text)
        assert "Gx0" in protected
        assert "espada" not in protected

    def test_empty_string(self):
        protected, mapping = protect_spanish_words("")
        assert protected == ""
        assert mapping == {}

    def test_no_spanish_words(self):
        text = "Find the hidden key"
        protected, mapping = protect_spanish_words(text)
        assert protected == text
        assert mapping == {}

    def test_placeholder_format(self):
        text = "Take the espada"
        protected, mapping = protect_spanish_words(text)
        # Should have Cx0 format
        assert any(ph.startswith("Cx") for ph in mapping)

    def test_cumpleanos_detected(self):
        """Word with \u00f1 is detected as Spanish."""
        text = "Happy cumplea\u00f1os"
        protected, mapping = protect_spanish_words(text)
        assert "cumplea\u00f1os" not in protected
        assert any(v == "cumplea\u00f1os" for v in mapping.values())


class TestRestoreSpanishWords:
    def test_restore_simple(self):
        text = "Take the Cx0 from the chest"
        mapping = {"Cx0": "espada"}
        restored = restore_spanish_words(text, mapping)
        assert restored == "Take the espada from the chest"

    def test_restore_multiple(self):
        mapping = {"Cx0": "cuchillo", "Cx1": "escudo"}
        text = "Use Cx1 and Cx0"
        restored = restore_spanish_words(text, mapping)
        assert "cuchillo" in restored
        assert "escudo" in restored

    def test_restore_empty_mapping(self):
        text = "No placeholders here"
        restored = restore_spanish_words(text, {})
        assert restored == text

    def test_roundtrip(self):
        """protect -> restore produces original text."""
        original = "Take the espada and escudo from the cave"
        protected, mapping = protect_spanish_words(original)
        restored = restore_spanish_words(protected, mapping)
        assert restored == original


class TestProtectBatch:
    def test_batch_independent_mappings(self):
        texts = [
            "Take the espada",
            "Find the escudo",
            "No Spanish here",
        ]
        protected, mappings = protect_spanish_batch(texts)
        assert len(protected) == 3
        assert len(mappings) == 3
        assert "espada" not in protected[0]
        assert "escudo" not in protected[1]
        assert mappings[2] == {}

    def test_batch_roundtrip(self):
        texts = [
            "Take the espada and cuchillo",
            "Pure English sentence here",
            "Find the refugio near town",
        ]
        protected, mappings = protect_spanish_batch(texts)
        restored = restore_spanish_batch(protected, mappings)
        assert restored == texts

    def test_batch_with_glossary_placeholders(self):
        """Glossary placeholders in text are preserved through protect/restore."""
        texts = [
            "Take the Gx0 and espada",
            "Use Gx1 near refugio",
        ]
        protected, mappings = protect_spanish_batch(texts)
        # Glossary placeholders still present
        assert "Gx0" in protected[0]
        assert "Gx1" in protected[1]
        # Spanish words replaced
        assert "espada" not in protected[0]
        assert "refugio" not in protected[1]

    def test_empty_batch(self):
        protected, mappings = protect_spanish_batch([])
        assert protected == []
        assert mappings == []


class TestNormalizeEsPlaceholders:
    """Test recovery of mangled Cx placeholders from neural MT."""

    def test_space_inserted(self):
        """Cx 0 -> Cx0 (space between prefix and number)."""
        mapping = {"Cx0": "espada"}
        result = _normalize_es_placeholders("Take the Cx 0", mapping)
        assert "Cx0" in result

    def test_case_change(self):
        """cx0 -> Cx0 (lowercased by MT)."""
        mapping = {"Cx0": "espada"}
        result = _normalize_es_placeholders("Take the cx0", mapping)
        assert "Cx0" in result

    def test_space_and_case_change(self):
        """cx 0 -> Cx0 (both mangling types)."""
        mapping = {"Cx0": "espada"}
        result = _normalize_es_placeholders("Take the cx 0", mapping)
        assert "Cx0" in result

    def test_false_positive_extra_digit(self):
        """Cx00 should NOT match Cx0 (extra digit)."""
        mapping = {"Cx0": "espada"}
        result = _normalize_es_placeholders("Code Cx00 here", mapping)
        assert result == "Code Cx00 here"

    def test_false_positive_word_char_before(self):
        """aCx0 should NOT be normalized (word char before)."""
        mapping = {"Cx0": "espada"}
        result = _normalize_es_placeholders("Code aCx0 here", mapping)
        assert result == "Code aCx0 here"

    def test_already_correct_no_change(self):
        """Cx0 stays unchanged."""
        mapping = {"Cx0": "espada"}
        text = "Take the Cx0"
        result = _normalize_es_placeholders(text, mapping)
        assert result == text

    def test_restore_with_mangled(self):
        """Full restore with mangled Cx placeholder."""
        mapping = {"Cx0": "espada"}
        restored = restore_spanish_words("Take the cx0", mapping)
        assert restored == "Take the espada"

    def test_restore_batch_with_mangled(self):
        mappings = [
            {"Cx0": "espada"},
            {"Cx0": "escudo"},
        ]
        texts = ["Take the cx0", "Use cx 0"]
        restored = restore_spanish_batch(texts, mappings)
        assert restored[0] == "Take the espada"
        assert restored[1] == "Use escudo"
