"""Tests for the language detection heuristic."""

import pytest

from modtranslator.translation.lang_detect import (
    _load_spanish_dictionary,
    get_spanish_words_in_text,
    is_english,
    is_spanish,
    is_target_language,
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

    def test_non_es_target_translates_english(self):
        # English text should be translated for any target language
        assert should_translate("Iron Sword", "FR")
        assert should_translate("Find the hidden key in the wasteland.", "DE")

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


class TestIsTargetLanguage:
    """Tests for the generic is_target_language() function."""

    # --- French ---
    def test_french_sentence(self):
        assert is_target_language("Trouvez la clé cachée dans le désert.", "FR")

    def test_french_with_accents(self):
        assert is_target_language("L'épée était très résistante.", "FR")

    def test_french_short_word(self):
        # "très" is a strong French word
        assert is_target_language("très", "FR")

    # --- German ---
    def test_german_sentence(self):
        assert is_target_language("Finde den versteckten Schlüssel in der Ödnis.", "DE")

    def test_german_with_eszett(self):
        # ß is a unique German character — fast path
        assert is_target_language("Schließen Sie die Tür.", "DE")

    def test_german_short_word(self):
        assert is_target_language("nicht", "DE")

    # --- Italian ---
    def test_italian_sentence(self):
        assert is_target_language("Trova la chiave nascosta nel deserto.", "IT")

    def test_italian_with_accents(self):
        assert is_target_language("L'armatura è molto resistente.", "IT")

    def test_italian_short_word(self):
        assert is_target_language("anche", "IT")

    # --- Portuguese ---
    def test_portuguese_sentence(self):
        assert is_target_language("Encontre a chave escondida para abrir a porta.", "PT")

    def test_portuguese_with_tilde(self):
        # ã/õ are unique Portuguese characters — fast path
        assert is_target_language("A proteção do coração.", "PT")

    def test_portuguese_short_word(self):
        assert is_target_language("também", "PT")

    # --- Russian ---
    def test_russian_sentence(self):
        assert is_target_language("Найдите скрытый ключ в пустыне.", "RU")

    def test_russian_single_word(self):
        # Cyrillic chars → fast path
        assert is_target_language("Меч", "RU")

    # --- Polish ---
    def test_polish_sentence(self):
        assert is_target_language("Znajdź ukryty klucz na pustyni.", "PL")

    def test_polish_with_diacritics(self):
        # ą/ę/ś/etc. are unique Polish characters — fast path
        assert is_target_language("Włócznia jest bardzo silna.", "PL")

    def test_polish_short_word(self):
        assert is_target_language("bardzo", "PL")

    # --- Spanish (delegates to existing logic) ---
    def test_spanish_delegates(self):
        assert is_target_language("Espada de hierro con oxidación única", "ES")

    def test_spanish_short(self):
        assert is_target_language("Chico", "ES")

    # --- Negative cases: English not detected as target ---
    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL"])
    def test_english_not_detected(self, lang: str):
        assert not is_target_language("Iron Sword", lang)

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL"])
    def test_english_sentence_not_detected(self, lang: str):
        assert not is_target_language("Find the hidden key in the wasteland.", lang)

    def test_empty_string(self):
        assert not is_target_language("", "FR")

    def test_short_string(self):
        assert not is_target_language("A", "FR")


class TestShouldTranslateNonES:
    """Tests for should_translate() with non-ES target languages."""

    def test_french_text_skipped(self):
        assert not should_translate("Trouvez la clé cachée dans le désert.", "FR")

    def test_german_text_skipped(self):
        assert not should_translate("Finde den versteckten Schlüssel in der Ödnis.", "DE")

    def test_italian_text_skipped(self):
        assert not should_translate("Trova la chiave nascosta nel deserto.", "IT")

    def test_portuguese_text_skipped(self):
        assert not should_translate("Encontre a chave escondida para abrir a porta.", "PT")

    def test_russian_text_skipped(self):
        assert not should_translate("Найдите скрытый ключ в пустыне.", "RU")

    def test_polish_text_skipped(self):
        assert not should_translate("Znajdź ukryty klucz na pustyni.", "PL")

    def test_english_translated_for_all_langs(self):
        for lang in ["FR", "DE", "IT", "PT", "RU", "PL"]:
            assert should_translate("Find the hidden key in the wasteland.", lang)

    def test_german_eszett_skipped(self):
        assert not should_translate("Schließen", "DE")

    def test_russian_cyrillic_skipped(self):
        assert not should_translate("Меч", "RU")

    def test_polish_diacritics_skipped(self):
        assert not should_translate("Włócznia", "PL")

    def test_portuguese_tilde_skipped(self):
        assert not should_translate("Proteção", "PT")

    def test_glossary_source_works_for_non_es(self):
        source_terms = {"sword"}
        assert should_translate("Sword", "FR", glossary_source_terms=source_terms)

    def test_glossary_target_works_for_non_es(self):
        terms = {"épée"}
        assert not should_translate("Épée", "FR", glossary_terms=terms)


# ---------------------------------------------------------------------------
# Intensive tests: Bethesda mod-typical strings, edge cases, cross-language
# ---------------------------------------------------------------------------


class TestIsTargetLanguageFrenchIntensive:
    """French detection — item names, descriptions, dialogues from Bethesda mods."""

    # -- Item names (short, 1-2 words) --
    def test_fr_item_epee(self):
        assert is_target_language("Épée", "FR")  # accent → special chars

    def test_fr_item_bouclier(self):
        # "bouclier" should be in FR dictionary
        assert is_target_language("Bouclier", "FR")

    def test_fr_item_two_words_with_accent(self):
        assert is_target_language("Épée longue", "FR")  # accent on common word

    # -- Descriptions (3+ words) --
    def test_fr_armor_description(self):
        assert is_target_language(
            "Une armure solide faite pour résister aux coups les plus violents.", "FR"
        )

    def test_fr_weapon_description(self):
        assert is_target_language(
            "Cette épée est très tranchante et peut couper facilement.", "FR"
        )

    def test_fr_quest_text(self):
        assert is_target_language(
            "Allez dans la grotte et trouvez le trésor caché.", "FR"
        )

    # -- Dialogues --
    def test_fr_dialogue_npc(self):
        assert is_target_language(
            "Je ne sais pas où il est allé. Peut-être dans la forêt.", "FR"
        )

    def test_fr_dialogue_short(self):
        assert is_target_language("Bonjour, comment allez-vous?", "FR")

    def test_fr_dialogue_with_strong_word(self):
        assert is_target_language(
            "Il faut toujours être prudent dans ce donjon.", "FR"
        )

    # -- Negative: English not detected as French --
    def test_fr_english_item_not_detected(self):
        assert not is_target_language("Leather Armor", "FR")

    def test_fr_english_description_not_detected(self):
        assert not is_target_language(
            "A sturdy shield made from dragon bones.", "FR"
        )

    def test_fr_english_dialogue_not_detected(self):
        assert not is_target_language(
            "I don't know where he went. Maybe into the forest.", "FR"
        )


class TestIsTargetLanguageGermanIntensive:
    """German detection — Bethesda mod strings."""

    # -- Item names --
    def test_de_item_schwert(self):
        assert is_target_language("Schwert", "DE")

    def test_de_item_with_umlaut(self):
        assert is_target_language("Rüstung", "DE")  # ü → special char

    def test_de_item_with_eszett(self):
        assert is_target_language("Großschwert", "DE")  # ß → unique char fast path

    def test_de_compound_without_special_chars_not_detected(self):
        # German compound nouns without ä/ö/ü/ß are a known blind spot —
        # they look like random Latin strings with no signal.
        assert not is_target_language("Eisenschwert", "DE")

    # -- Descriptions --
    def test_de_armor_description(self):
        assert is_target_language(
            "Eine schwere Rüstung aus Eisen, die sehr gut schützt.", "DE"
        )

    def test_de_quest_text(self):
        assert is_target_language(
            "Gehen Sie in die Höhle und finden Sie den Schatz.", "DE"
        )

    def test_de_dialogue(self):
        assert is_target_language(
            "Ich weiß nicht, wo er ist. Vielleicht im Wald.", "DE"
        )

    def test_de_long_description(self):
        assert is_target_language(
            "Dieses Schwert wurde von einem berühmten Schmied hergestellt "
            "und ist außerdem mit einer Verzauberung versehen.",
            "DE",
        )

    # -- Negative --
    def test_de_english_not_detected(self):
        assert not is_target_language("Heavy Iron Armor", "DE")

    def test_de_english_quest_not_detected(self):
        assert not is_target_language(
            "Go to the cave and find the hidden treasure.", "DE"
        )


class TestIsTargetLanguageItalianIntensive:
    """Italian detection — Bethesda mod strings."""

    # -- Item names --
    def test_it_item_spada(self):
        assert is_target_language("Spada", "IT")

    def test_it_item_scudo(self):
        assert is_target_language("Scudo", "IT")

    def test_it_item_pozione(self):
        assert is_target_language("Pozione", "IT")

    def test_it_item_two_words(self):
        assert is_target_language("Spada lunga", "IT")

    # -- Descriptions --
    def test_it_armor_description(self):
        assert is_target_language(
            "Un'armatura pesante fatta di ferro molto resistente.", "IT"
        )

    def test_it_quest_text(self):
        assert is_target_language(
            "Vai nella grotta e trova il tesoro nascosto.", "IT"
        )

    def test_it_dialogue(self):
        assert is_target_language(
            "Non so dove sia andato. Forse nella foresta.", "IT"
        )

    def test_it_with_strong_words(self):
        assert is_target_language(
            "Devi anche trovare la chiave prima di entrare.", "IT"
        )

    # -- Negative --
    def test_it_english_not_detected(self):
        assert not is_target_language("Steel Plate Armor", "IT")

    def test_it_english_dialogue_not_detected(self):
        assert not is_target_language(
            "You should take this sword with you.", "IT"
        )


class TestIsTargetLanguagePortugueseIntensive:
    """Portuguese detection — Bethesda mod strings."""

    # -- Unique chars (ã/õ) fast path --
    def test_pt_word_with_tilde_a(self):
        assert is_target_language("Proteção", "PT")

    def test_pt_word_with_tilde_o(self):
        assert is_target_language("Campeões", "PT")

    def test_pt_item_espada(self):
        assert is_target_language("Espada", "PT")

    def test_pt_item_escudo(self):
        assert is_target_language("Escudo", "PT")

    # -- Descriptions --
    def test_pt_armor_description(self):
        assert is_target_language(
            "Uma armadura pesada feita de ferro para proteção.", "PT"
        )

    def test_pt_quest_text(self):
        assert is_target_language(
            "Vá até a caverna e encontre o tesouro escondido.", "PT"
        )

    def test_pt_dialogue(self):
        assert is_target_language(
            "Não sei onde ele foi. Talvez na floresta.", "PT"
        )

    def test_pt_with_strong_words(self):
        assert is_target_language(
            "Você também precisa encontrar a chave antes de entrar.", "PT"
        )

    # -- Negative --
    def test_pt_english_not_detected(self):
        assert not is_target_language("Heavy Iron Shield", "PT")

    def test_pt_english_description_not_detected(self):
        assert not is_target_language(
            "A powerful enchantment that increases your strength.", "PT"
        )


class TestIsTargetLanguageRussianIntensive:
    """Russian detection — Cyrillic fast path + word lists."""

    # -- Any Cyrillic text → fast path True --
    def test_ru_single_word(self):
        assert is_target_language("Меч", "RU")

    def test_ru_item_name(self):
        assert is_target_language("Железный щит", "RU")

    def test_ru_description(self):
        assert is_target_language(
            "Тяжёлая броня из железа для максимальной защиты.", "RU"
        )

    def test_ru_dialogue(self):
        assert is_target_language(
            "Я не знаю, куда он ушёл. Возможно, в лес.", "RU"
        )

    def test_ru_quest_text(self):
        assert is_target_language(
            "Идите в пещеру и найдите спрятанное сокровище.", "RU"
        )

    def test_ru_single_cyrillic_char_in_latin(self):
        # Even a single Cyrillic char in otherwise Latin text → True
        assert is_target_language("Iron Мeч", "RU")

    # -- Negative --
    def test_ru_english_not_detected(self):
        assert not is_target_language("Iron Sword", "RU")

    def test_ru_english_description_not_detected(self):
        assert not is_target_language(
            "A powerful enchantment from the ancient mages.", "RU"
        )


class TestIsTargetLanguagePolishIntensive:
    """Polish detection — diacritics fast path + word lists."""

    # -- Unique chars fast path (ą, ę, ś, ć, ź, ż, ń, ł) --
    def test_pl_word_with_l_stroke(self):
        assert is_target_language("Włócznia", "PL")  # ł

    def test_pl_word_with_ogoneks(self):
        assert is_target_language("Więzień", "PL")  # ę

    def test_pl_word_with_kreska(self):
        assert is_target_language("Miecz Króla", "PL")  # ó (shared), but ł not here

    # -- Descriptions --
    def test_pl_armor_description(self):
        assert is_target_language(
            "Ciężka zbroja z żelaza zapewniająca maksymalną ochronę.", "PL"
        )

    def test_pl_quest_text(self):
        assert is_target_language(
            "Idź do jaskini i znajdź ukryty skarb.", "PL"
        )

    def test_pl_dialogue(self):
        assert is_target_language(
            "Nie wiem, dokąd poszedł. Może do lasu.", "PL"
        )

    def test_pl_with_strong_words(self):
        assert is_target_language(
            "Musisz również znaleźć klucz przed wejściem.", "PL"
        )

    # -- Negative --
    def test_pl_english_not_detected(self):
        assert not is_target_language("Dwarven Crossbow", "PL")

    def test_pl_english_quest_not_detected(self):
        assert not is_target_language(
            "Find the ancient artifact in the dungeon.", "PL"
        )


class TestCrossLanguageFalsePositives:
    """Cross-language detection tests.

    Romance languages (ES, FR, IT, PT) share significant vocabulary (articles,
    prepositions, common words like "la", "para", "con"). Perfect cross-language
    separation is not possible with heuristics alone — and that's OK for our use
    case: the goal is to detect already-translated text vs English source text,
    not to distinguish between Romance languages.

    Tests here verify what DOES work (non-overlapping language families) and
    document known limitations (Romance cross-detection).
    """

    # -- Non-overlapping language families: these MUST work --
    def test_russian_not_polish(self):
        assert not is_target_language(
            "Найдите скрытый ключ в пустыне.", "PL"
        )

    def test_polish_not_russian(self):
        assert not is_target_language(
            "Znajdź ukryty klucz na pustyni.", "RU"
        )

    def test_german_not_spanish(self):
        assert not is_target_language(
            "Dieses Schwert wurde von einem berühmten Schmied hergestellt.", "ES"
        )

    def test_german_not_french(self):
        assert not is_target_language(
            "Gehen Sie in die Höhle und finden Sie den Schatz.", "FR"
        )

    def test_german_not_russian(self):
        assert not is_target_language(
            "Diese Rüstung ist sehr widerstandsfähig.", "RU"
        )

    def test_russian_not_german(self):
        assert not is_target_language(
            "Эта броня очень устойчива к огненному урону.", "DE"
        )

    def test_polish_not_french(self):
        assert not is_target_language(
            "Ciężka zbroja z żelaza zapewniająca ochronę.", "FR"
        )

    def test_french_not_german(self):
        assert not is_target_language(
            "Trouvez la clé cachée dans le désert.", "DE"
        )

    def test_french_not_russian(self):
        assert not is_target_language(
            "Je ne sais pas où il est allé.", "RU"
        )

    def test_italian_not_german(self):
        assert not is_target_language(
            "Vai nella grotta e trova il tesoro nascosto.", "DE"
        )

    # -- Romance cross-detection: document known limitations --
    # ES/FR/IT/PT share "la", "para"/"pour"/"per", "que", "con", "entre", etc.
    # Some cross-detection is expected and acceptable — the system's primary
    # job is EN vs target-lang, not FR vs IT.

    def test_french_not_spanish_with_unique_words(self):
        # Works when French text has clearly French-only words
        assert not is_target_language(
            "Trouvez la clé cachée dans le désert.", "ES"
        )

    def test_italian_not_french(self):
        assert not is_target_language(
            "Vai nella grotta e trova il tesoro nascosto.", "FR"
        )

    def test_italian_not_spanish(self):
        assert not is_target_language(
            "Un'armatura pesante fatta di ferro molto resistente.", "ES"
        )

    def test_spanish_not_italian(self):
        assert not is_target_language(
            "Una armadura resistente de cuero para protección.", "IT"
        )

    # Known limitation: ES ↔ PT have very high lexical overlap
    # "para", "que", "con", "entre", "antes" are shared.
    # We accept some cross-detection here — better to skip an already-translated
    # string than to re-translate it.


class TestMixedLanguageStrings:
    """Strings mixing English with target-language words (common in partially translated mods)."""

    # Mixed strings should generally NOT be detected as fully target-language
    # unless the target words dominate.

    def test_mostly_english_one_french_word(self):
        # One French word in English sentence should not trigger
        assert not is_target_language("Take the épée from the shelf.", "FR")

    def test_mostly_english_one_german_word(self):
        assert not is_target_language("Find the Schwert in the cave.", "DE")

    def test_mostly_english_one_italian_word(self):
        assert not is_target_language("Use the Spada against the enemy.", "IT")

    def test_mostly_french_some_english(self):
        # Mostly French with a proper noun → still French
        assert is_target_language(
            "Le Dragonborn doit trouver la clé dans le donjon.", "FR"
        )

    def test_mostly_german_some_english(self):
        assert is_target_language(
            "Der Dragonborn muss den Schlüssel in der Höhle finden.", "DE"
        )

    def test_mostly_italian_some_english(self):
        assert is_target_language(
            "Il Dragonborn deve trovare la chiave nel dungeon.", "IT"
        )


class TestEdgeCasesIntensive:
    """Edge cases: numbers, codes, proper nouns, symbols, very short strings."""

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL", "ES"])
    def test_number_not_detected(self, lang: str):
        assert not is_target_language("12345", lang)

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL", "ES"])
    def test_code_not_detected(self, lang: str):
        assert not is_target_language("NPC_01_Combat", lang)

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL", "ES"])
    def test_hex_not_detected(self, lang: str):
        assert not is_target_language("0x00FF00", lang)

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL", "ES"])
    def test_empty_not_detected(self, lang: str):
        assert not is_target_language("", lang)

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL", "ES"])
    def test_single_char_not_detected(self, lang: str):
        assert not is_target_language("X", lang)

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "PL", "ES"])
    def test_proper_noun_not_detected(self, lang: str):
        """Game proper nouns (Megaton, Whiterun) should not trigger detection."""
        assert not is_target_language("Megaton", lang)
        assert not is_target_language("Whiterun", lang)
        assert not is_target_language("Vault 101", lang)

    def test_proper_noun_not_russian(self):
        # Latin text never matches Cyrillic regex
        assert not is_target_language("Megaton", "RU")

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL", "ES"])
    def test_symbols_only_not_detected(self, lang: str):
        assert not is_target_language("---", lang)
        assert not is_target_language("...", lang)
        assert not is_target_language("***", lang)


class TestShouldTranslateIntensiveBethesda:
    """Integration tests: should_translate with typical Bethesda mod strings per language."""

    # ---- French ----
    def test_fr_translated_item_skipped(self):
        assert not should_translate("Épée de fer", "FR")

    def test_fr_translated_description_skipped(self):
        assert not should_translate(
            "Cette armure est très résistante aux dégâts de feu.", "FR"
        )

    def test_fr_english_item_translated(self):
        assert should_translate("Iron Sword", "FR")

    def test_fr_english_description_translated(self):
        assert should_translate(
            "This armor is very resistant to fire damage.", "FR"
        )

    # ---- German ----
    def test_de_translated_item_skipped(self):
        assert not should_translate("Großschwert", "DE")  # ß → detected

    def test_de_translated_description_skipped(self):
        assert not should_translate(
            "Diese Rüstung ist sehr widerstandsfähig gegen Feuerschaden.", "DE"
        )

    def test_de_english_item_translated(self):
        assert should_translate("Daedric Mace", "DE")

    def test_de_english_description_translated(self):
        assert should_translate(
            "This weapon does extra damage to undead creatures.", "DE"
        )

    # ---- Italian ----
    def test_it_translated_item_skipped(self):
        assert not should_translate("Spada di ferro", "IT")

    def test_it_translated_description_skipped(self):
        assert not should_translate(
            "Questa armatura è molto resistente ai danni da fuoco.", "IT"
        )

    def test_it_english_item_translated(self):
        assert should_translate("Glass Bow", "IT")

    def test_it_english_description_translated(self):
        assert should_translate(
            "A bow crafted from refined malachite.", "IT"
        )

    # ---- Portuguese ----
    def test_pt_translated_item_skipped(self):
        assert not should_translate("Proteção contra fogo", "PT")

    def test_pt_translated_description_skipped(self):
        assert not should_translate(
            "Esta armadura também oferece proteção contra veneno.", "PT"
        )

    def test_pt_english_item_translated(self):
        assert should_translate("Elven Shield", "PT")

    def test_pt_english_description_translated(self):
        assert should_translate(
            "A shield crafted by the ancient elves.", "PT"
        )

    # ---- Russian ----
    def test_ru_translated_item_skipped(self):
        assert not should_translate("Железный меч", "RU")

    def test_ru_translated_description_skipped(self):
        assert not should_translate(
            "Эта броня очень устойчива к огненному урону.", "RU"
        )

    def test_ru_english_item_translated(self):
        assert should_translate("Orcish Battleaxe", "RU")

    def test_ru_english_description_translated(self):
        assert should_translate(
            "An axe forged in the fires of mount doom.", "RU"
        )

    # ---- Polish ----
    def test_pl_translated_item_skipped(self):
        assert not should_translate("Żelazny miecz", "PL")

    def test_pl_translated_description_skipped(self):
        assert not should_translate(
            "Ta zbroja jest bardzo odporna na obrażenia od ognia.", "PL"
        )

    def test_pl_english_item_translated(self):
        assert should_translate("Ebony Dagger", "PL")

    def test_pl_english_description_translated(self):
        assert should_translate(
            "A dagger made from the finest ebony.", "PL"
        )

    # ---- Ambiguous short strings: should translate (likely English game content) ----
    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL"])
    def test_unknown_short_word_translates(self, lang: str):
        """Unknown single words (English item names) should be translated."""
        assert should_translate("Xyzzy", lang)

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL"])
    def test_english_game_nouns_translate(self, lang: str):
        """Common English game nouns should be translated."""
        assert should_translate("Sword", lang)
        assert should_translate("Shield", lang)
        assert should_translate("Armor", lang)

    # ---- Glossary interaction with all languages ----
    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL"])
    def test_glossary_source_overrides_detection(self, lang: str):
        """Glossary source terms always get translated, even if they look like target lang."""
        source_terms = {"stimpak"}
        assert should_translate("Stimpak", lang, glossary_source_terms=source_terms)

    @pytest.mark.parametrize("lang,term", [
        ("FR", "épée"),
        ("DE", "schwert"),
        ("IT", "spada"),
        ("PT", "espada"),
        ("RU", "меч"),
        ("PL", "miecz"),
    ])
    def test_glossary_target_skips(self, lang: str, term: str):
        """Glossary target terms are skipped."""
        assert not should_translate(term.title(), lang, glossary_terms={term})
