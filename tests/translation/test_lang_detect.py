"""Tests for the language detection heuristic."""

from modtranslator.translation.lang_detect import (
    _load_spanish_dictionary,
    get_spanish_words_in_text,
    is_english,
    is_spanish,
    should_translate,
)


class TestIsSpanish:
    def test_english_simple(self):
        assert not is_spanish("Iron Sword")

    def test_english_description(self):
        assert not is_spanish("A sturdy set of leather armor.")

    def test_english_quest(self):
        assert not is_spanish("Find the hidden key in the wasteland.")

    def test_spanish_with_accents(self):
        assert is_spanish("Espada de hierro con oxidación única")

    def test_spanish_sentence(self):
        assert is_spanish("Encuentra la llave escondida para abrir la puerta.")

    def test_spanish_description(self):
        assert is_spanish("Una armadura resistente de cuero para protección.")

    def test_spanish_dialog(self):
        assert is_spanish("¿Qué estás haciendo aquí? No deberías estar aquí.")

    def test_short_string_returns_false(self):
        assert not is_spanish("Hi")
        assert not is_spanish("Sí")

    def test_empty_string(self):
        assert not is_spanish("")

    def test_single_word_english(self):
        assert not is_spanish("Sword")

    def test_proper_nouns_not_falsely_spanish(self):
        assert not is_spanish("Megaton")
        assert not is_spanish("Vault 101")

    def test_spanish_with_strong_words(self):
        assert is_spanish("También puede encontrar más objetos aquí.")

    def test_mixed_but_mostly_spanish(self):
        assert is_spanish("El Vault-Tec también tiene una base secreta.")

    def test_english_with_el(self):
        # "el" alone should not trigger false positive
        assert not is_spanish("The el dorado mine")

    def test_numbers_and_codes(self):
        assert not is_spanish("NPC_01_Combat")
        assert not is_spanish("0x00FF00")

    # New tests for short Spanish strings (the core bug)
    def test_chico_detected_as_spanish(self):
        assert is_spanish("Chico")

    def test_chica_detected_as_spanish(self):
        assert is_spanish("Chica")

    def test_refugio_detected_as_spanish(self):
        assert is_spanish("Refugio")

    def test_arma_detected_as_spanish(self):
        assert is_spanish("Arma")

    def test_espada_detected_as_spanish(self):
        assert is_spanish("Espada")

    def test_agua_detected_as_spanish(self):
        assert is_spanish("Agua")

    def test_muerte_detected_as_spanish(self):
        assert is_spanish("Muerte")

    def test_two_spanish_words(self):
        assert is_spanish("Arma rota")
        assert is_spanish("Espada vieja")

    def test_short_with_accent(self):
        assert is_spanish("Poción")


class TestIsEnglish:
    def test_english_sentence(self):
        assert is_english("Find the hidden key in the wasteland.")

    def test_english_description(self):
        assert is_english("A sturdy set of leather armor.")

    def test_english_with_common_words(self):
        assert is_english("You should take this with you.")

    def test_spanish_not_english(self):
        assert not is_english("Encuentra la llave escondida para abrir la puerta.")

    def test_spanish_accents_penalize(self):
        assert not is_english("Espada de hierro con oxidación única")

    def test_short_english_word(self):
        # "the" is correctly detected as English even at 3 chars
        assert is_english("the")

    def test_short_returns_false(self):
        assert not is_english("Hi")
        assert not is_english("")

    def test_english_two_words_no_stop_words(self):
        # "Iron Sword" — neither word is in _ENGLISH_WORDS, so no signal
        assert not is_english("Iron Sword")

    def test_english_two_words_with_stop_word(self):
        assert is_english("the sword")

    def test_english_game_text(self):
        assert is_english("You have discovered a new location.")

    def test_ambiguous_single_word(self):
        # "Sword" isn't in _ENGLISH_WORDS, so it won't score high
        assert not is_english("Sword")


class TestShouldTranslate:
    def test_chico_should_not_translate(self):
        """Core bug: 'Chico' is Spanish and should NOT be translated."""
        assert not should_translate("Chico", "ES")

    def test_iron_sword_should_translate(self):
        # "Iron Sword" — no Spanish signal, so it passes through to translate
        assert should_translate("Iron Sword", "ES")

    def test_english_sentence_should_translate(self):
        assert should_translate("Find the hidden key in the wasteland.", "ES")

    def test_spanish_sentence_should_not_translate(self):
        assert not should_translate("Encuentra la llave escondida para abrir la puerta.", "ES")

    def test_short_string_skipped(self):
        # Single char and empty strings are skipped (not enough signal)
        assert not should_translate("", "ES")
        assert not should_translate("A", "ES")

    def test_short_english_word_translates(self):
        # Short English words like "Hi" should be translated
        assert should_translate("Hi", "ES")

    def test_glossary_exact_match_skips(self):
        terms = {"chico", "refugio", "arma"}
        assert not should_translate("Chico", "ES", glossary_terms=terms)
        assert not should_translate("Refugio", "ES", glossary_terms=terms)

    def test_glossary_does_not_block_english(self):
        terms = {"chico"}
        assert should_translate("Find the sword", "ES", glossary_terms=terms)

    def test_non_es_target_always_translates(self):
        # For non-ES targets, should_translate just checks length
        assert should_translate("Chico", "FR")
        assert should_translate("Iron Sword", "FR")

    def test_agua_should_not_translate(self):
        assert not should_translate("Agua", "ES")

    def test_muerte_should_not_translate(self):
        assert not should_translate("Muerte", "ES")

    def test_refugio_should_not_translate(self):
        assert not should_translate("Refugio", "ES")

    def test_english_description_should_translate(self):
        assert should_translate("A sturdy set of leather armor.", "ES")

    def test_ambiguous_short_string_translates(self):
        # Unknown single word with no Spanish signal — translate
        # (likely an English noun like item names)
        assert should_translate("Xyzzy", "ES")

    def test_longer_ambiguous_translates(self):
        # Longer text without clear signals — translate (likely English game content)
        assert should_translate("Abandoned factory north section level three", "ES")

    def test_glossary_source_term_bypasses_min_length(self):
        """Short glossary source terms like 'Dad' should always be translated."""
        source_terms = {"dad"}
        assert should_translate("Dad", "ES", glossary_source_terms=source_terms)

    def test_glossary_source_term_normal_length(self):
        """Glossary source terms of normal length also force-translate."""
        source_terms = {"sweetroll", "stimpak"}
        assert should_translate("Sweetroll", "ES", glossary_source_terms=source_terms)

    def test_short_words_now_translate(self):
        """Short English words like 'Dad' are translated with MIN_LENGTH=2."""
        assert should_translate("Dad", "ES")
        assert should_translate("Mom", "ES")
        assert should_translate("Gun", "ES")

    def test_glossary_source_overrides_spanish_detection(self):
        """Source terms that look Spanish (e.g. 'Perk') still get translated."""
        source_terms = {"perk"}
        assert should_translate("Perk", "ES", glossary_source_terms=source_terms)

    def test_self_protection_not_in_source_terms(self):
        """Self-protection terms (source==target) should be excluded when building the set."""
        # This tests the CLI logic: only k.lower() != v.lower() should be in source_terms
        terms = {"Dad": "Papá", "Jet": "Jet", "Oro": "Oro"}
        source_terms = {k.lower() for k, v in terms.items() if k.lower() != v.lower()}
        assert "dad" in source_terms
        assert "jet" not in source_terms
        assert "oro" not in source_terms


class TestSpanishDictionary:
    def test_dictionary_loads(self):
        dictionary = _load_spanish_dictionary()
        assert isinstance(dictionary, frozenset)
        assert len(dictionary) > 1500

    def test_dictionary_contains_original_single_words(self):
        """All words from the old _SPANISH_SINGLE_WORDS must be in dictionary."""
        dictionary = _load_spanish_dictionary()
        for word in [
            "chico", "chica", "arma", "refugio", "espada",
            "agua", "fuego", "hierro", "muerte", "vida",
            "trampa", "chatarra", "basura", "comida",
        ]:
            assert word in dictionary, f"{word} missing from dictionary"

    def test_dictionary_excludes_cognates(self):
        """Common cognates that are identical in English should not be in dict."""
        dictionary = _load_spanish_dictionary()
        # These are intentionally excluded to avoid false positives
        for word in ["animal", "hotel", "hospital"]:
            assert word not in dictionary, f"cognate {word} should not be in dict"

    def test_cumpleaños_detected_by_dictionary(self):
        """Words like cumpleaños are now in the dictionary."""
        dictionary = _load_spanish_dictionary()
        assert "cumpleaños" in dictionary

    def test_is_spanish_uses_dictionary_for_short_strings(self):
        """Dictionary words (not in old _SPANISH_SINGLE_WORDS) are now detected."""
        # 'aldea' was not in _SPANISH_SINGLE_WORDS but is in the dictionary
        assert is_spanish("Aldea")

    def test_is_spanish_dictionary_fallback_long_text(self):
        """For long texts, dictionary words contribute to score as fallback."""
        # Text with many dictionary words — should be detected as Spanish
        assert is_spanish("La aldea estaba llena de campesinos y mercaderes")

    def test_is_spanish_english_with_dorado_not_false_positive(self):
        """'The el dorado mine' is English despite 'el' and 'dorado' being Spanish."""
        assert not is_spanish("The el dorado mine")


class TestGetSpanishWordsInText:
    def test_finds_dictionary_words(self):
        result = get_spanish_words_in_text("Take the espada from the cave")
        words = [w for w, _, _ in result]
        assert "espada" in words

    def test_finds_accented_words(self):
        result = get_spanish_words_in_text("The poción was strong")
        words = [w for w, _, _ in result]
        assert "poción" in words

    def test_ignores_short_words(self):
        result = get_spanish_words_in_text("el sol")
        # "el" (2 chars) and "sol" (3 chars) are < 4 chars
        assert result == []

    def test_returns_positions(self):
        text = "Take the espada"
        result = get_spanish_words_in_text(text)
        assert len(result) == 1
        word, start, end = result[0]
        assert word == "espada"
        assert text[start:end] == "espada"

    def test_pure_english_no_results(self):
        result = get_spanish_words_in_text("Find the hidden key in the wasteland")
        # Most English words won't be in dictionary
        assert all(w not in ["find", "hidden", "key", "wasteland"] for w, _, _ in result)

    def test_empty_string(self):
        assert get_spanish_words_in_text("") == []

    def test_multiple_words_found(self):
        result = get_spanish_words_in_text("Find the espada and escudo")
        words = [w for w, _, _ in result]
        assert "espada" in words
        assert "escudo" in words
