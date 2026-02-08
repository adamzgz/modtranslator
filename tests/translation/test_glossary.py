"""Tests for glossary term protection."""

from modtranslator.translation.glossary import Glossary, _normalize_placeholders


class TestGlossary:
    def test_protect_and_restore(self):
        g = Glossary(terms={"Wasteland": "Yermo", "Stimpak": "Estimulante"})

        protected = g.protect("Welcome to the Wasteland, use a Stimpak.")
        assert "Wasteland" not in protected
        assert "Stimpak" not in protected
        assert "Gx" in protected

        restored = g.restore(protected)
        assert "Yermo" in restored
        assert "Estimulante" in restored

    def test_protect_case_insensitive(self):
        g = Glossary(terms={"Pip-Boy": "Pip-Boy"})

        protected = g.protect("Check your pip-boy")
        assert "pip-boy" not in protected.lower() or "Gx" in protected

    def test_protect_word_boundary_no_substring(self):
        """'Dad' must not match inside 'habilidades' (word boundary fix)."""
        g = Glossary(terms={"Dad": "Papá", "Vault": "Refugio"})

        # Substring: must NOT match
        text, mapping = g.protect_with_mapping("Editar habilidades destacadas")
        assert "Papá" not in text
        assert "Gx" not in text
        assert text == "Editar habilidades destacadas"

        # Whole word: must match
        text2, mapping2 = g.protect_with_mapping("Talk to Dad")
        assert "Gx" in text2
        restored = g.restore(text2, mapping2)
        assert restored == "Talk to Papá"

        # "Vault" must not match in "Vaulting"
        text3, _ = g.protect_with_mapping("Vaulting over")
        assert text3 == "Vaulting over"

    def test_no_terms_no_change(self):
        g = Glossary(terms={})
        text = "Hello World"
        assert g.protect(text) == text
        assert g.restore(text) == text

    def test_protect_batch(self):
        g = Glossary(terms={"Vault": "Refugio"})
        texts = ["Enter the Vault", "Vault door", "No match here"]

        protected, mappings = g.protect_batch(texts)
        assert all("Vault" not in p or "Gx" in p for p in protected[:2])
        assert len(mappings) == 3

    def test_from_toml(self, tmp_path):
        toml_content = """
[terms]
Wasteland = "Yermo"
"Pip-Boy" = "Pip-Boy"
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content, encoding="utf-8")

        g = Glossary.from_toml(toml_file)
        assert g.terms["Wasteland"] == "Yermo"
        assert g.terms["Pip-Boy"] == "Pip-Boy"

    def test_placeholder_format(self):
        g = Glossary(terms={"Nuka-Cola": "Nuka-Cola"})
        protected = g.protect("Drink Nuka-Cola")
        # Should contain a placeholder like Gx0
        assert "Gx0" in protected


class TestGlossaryBatchMappings:
    def test_protect_batch_independent_mappings(self):
        """Each text in a batch gets its own independent mapping."""
        g = Glossary(terms={"Vault": "Refugio", "Wasteland": "Yermo"})
        texts = [
            "Enter the Vault",
            "Cross the Wasteland",
            "No glossary terms here",
        ]
        protected, mappings = g.protect_batch(texts)

        # First text only has Vault
        assert "Gx" in protected[0]
        assert any("Refugio" in m.values() for m in [mappings[0]])
        # Second text only has Wasteland
        assert "Gx" in protected[1]
        assert any("Yermo" in m.values() for m in [mappings[1]])
        # Third text has no matches
        assert mappings[2] == {}

    def test_restore_batch_with_mappings(self):
        """restore_batch with explicit mappings restores correctly."""
        g = Glossary(terms={"Vault": "Refugio", "Wasteland": "Yermo"})
        texts = ["Enter the Vault", "Cross the Wasteland"]
        protected, mappings = g.protect_batch(texts)

        # Simulate "translation" by keeping placeholders unchanged
        restored = g.restore_batch(protected, mappings)

        assert "Refugio" in restored[0]
        assert "Yermo" in restored[1]

    def test_protect_with_mapping_returns_tuple(self):
        """protect_with_mapping returns (protected_text, mapping_dict)."""
        g = Glossary(terms={"Stimpak": "Estimulante"})
        result = g.protect_with_mapping("Use a Stimpak")
        assert isinstance(result, tuple)
        assert len(result) == 2
        protected, mapping = result
        assert isinstance(protected, str)
        assert isinstance(mapping, dict)
        assert "Gx0" in protected
        assert mapping["Gx0"] == "Estimulante"

    def test_restore_with_explicit_placeholders(self):
        """restore() accepts an explicit placeholders dict."""
        g = Glossary(terms={"Vault": "Refugio"})
        explicit = {"Gx0": "Refugio"}
        restored = g.restore("Enter the Gx0", explicit)
        assert restored == "Enter the Refugio"

    def test_backward_compat_protect_restore_single(self):
        """Original protect/restore API still works (stateful)."""
        g = Glossary(terms={"Wasteland": "Yermo"})
        protected = g.protect("The Wasteland awaits")
        assert "Gx" in protected

        # restore without args uses internal state
        restored = g.restore(protected)
        assert "Yermo" in restored
        assert "Gx" not in restored


class TestGlossaryMerge:
    def test_merge_adds_terms(self):
        g1 = Glossary(terms={"Vault": "Refugio"})
        g2 = Glossary(terms={"Wasteland": "Yermo"})
        g1.merge(g2)
        assert g1.terms["Vault"] == "Refugio"
        assert g1.terms["Wasteland"] == "Yermo"

    def test_merge_overrides_on_conflict(self):
        g1 = Glossary(terms={"Stimpak": "Estimulante"})
        g2 = Glossary(terms={"Stimpak": "Stimpak override"})
        g1.merge(g2)
        assert g1.terms["Stimpak"] == "Stimpak override"

    def test_merge_preserves_existing(self):
        g1 = Glossary(terms={"Vault": "Refugio", "Karma": "Karma"})
        g2 = Glossary(terms={"Wasteland": "Yermo"})
        g1.merge(g2)
        assert g1.terms["Vault"] == "Refugio"
        assert g1.terms["Karma"] == "Karma"
        assert g1.terms["Wasteland"] == "Yermo"

    def test_from_multiple_toml_empty(self):
        g = Glossary.from_multiple_toml([])
        assert g.terms == {}

    def test_from_multiple_toml_single(self, tmp_path):
        toml = tmp_path / "base.toml"
        toml.write_text('[terms]\nVault = "Refugio"\n', encoding="utf-8")
        g = Glossary.from_multiple_toml([toml])
        assert g.terms == {"Vault": "Refugio"}

    def test_from_multiple_toml_merge_order(self, tmp_path):
        base = tmp_path / "base.toml"
        base.write_text('[terms]\nStimpak = "Base"\nVault = "Refugio"\n', encoding="utf-8")
        specific = tmp_path / "specific.toml"
        specific.write_text(
            '[terms]\nStimpak = "Override"\nMegaton = "Megat\u00f3n"\n', encoding="utf-8",
        )

        g = Glossary.from_multiple_toml([base, specific])
        # Later file overrides
        assert g.terms["Stimpak"] == "Override"
        # Base terms preserved
        assert g.terms["Vault"] == "Refugio"
        # Specific terms added
        assert g.terms["Megaton"] == "Megat\u00f3n"

    def test_merged_protects_all_terms(self):
        g1 = Glossary(terms={"Vault": "Refugio"})
        g2 = Glossary(terms={"Wasteland": "Yermo"})
        g1.merge(g2)

        protected, mappings = g1.protect_batch(["Enter the Vault in the Wasteland"])
        assert "Gx" in protected[0]
        restored = g1.restore_batch(protected, mappings)
        assert "Refugio" in restored[0]
        assert "Yermo" in restored[0]


class TestNormalizePlaceholders:
    """Test recovery of mangled placeholders from neural MT backends."""

    def test_space_inserted(self):
        """Gx 48 -> Gx48 (space between prefix and number)."""
        mapping = {"Gx48": "Yermo"}
        result = _normalize_placeholders("Enter the Gx 48", mapping)
        assert "Gx48" in result

    def test_case_change(self):
        """gx48 -> Gx48 (lowercased by MT)."""
        mapping = {"Gx48": "Yermo"}
        result = _normalize_placeholders("Enter the gx48", mapping)
        assert "Gx48" in result

    def test_space_and_case_change(self):
        """gx 48 -> Gx48 (both mangling types)."""
        mapping = {"Gx48": "Yermo"}
        result = _normalize_placeholders("Enter the gx 48", mapping)
        assert "Gx48" in result

    def test_false_positive_extra_digit(self):
        """Gx480 should NOT match Gx48 (extra digit)."""
        mapping = {"Gx48": "Yermo"}
        result = _normalize_placeholders("Code Gx480 here", mapping)
        assert "Gx48" not in result or "Gx480" in result

    def test_false_positive_word_char_before(self):
        """aGx48 should NOT be normalized (word char before)."""
        mapping = {"Gx48": "Yermo"}
        result = _normalize_placeholders("Code aGx48 here", mapping)
        assert result == "Code aGx48 here"

    def test_already_correct_no_change(self):
        """Gx48 stays unchanged."""
        mapping = {"Gx48": "Yermo"}
        text = "Enter the Gx48"
        result = _normalize_placeholders(text, mapping)
        assert result == text

    def test_restore_roundtrip_with_mangled(self):
        """Full roundtrip: mangled placeholder restores to glossary term."""
        g = Glossary(terms={"Wasteland": "Yermo"})
        mapping = {"Gx0": "Yermo"}
        # Simulate MT mangling: Gx0 -> gx0 (lowercased)
        mangled = "Enter the gx0"
        restored = g.restore(mangled, mapping)
        assert restored == "Enter the Yermo"

    def test_restore_batch_with_mangled(self):
        """Batch restore handles mangled placeholders."""
        g = Glossary(terms={"Vault": "Refugio", "Wasteland": "Yermo"})
        mappings = [
            {"Gx0": "Refugio"},
            {"Gx0": "Yermo"},
        ]
        mangled_texts = [
            "Enter the gx0",
            "Cross the gx 0",
        ]
        restored = g.restore_batch(mangled_texts, mappings)
        assert restored[0] == "Enter the Refugio"
        assert restored[1] == "Cross the Yermo"

    def test_empty_mapping(self):
        result = _normalize_placeholders("Just text", {})
        assert result == "Just text"

    def test_multiple_mangled_in_one_text(self):
        """Multiple different mangled placeholders in same text."""
        mapping = {"Gx0": "Refugio", "Gx1": "Yermo"}
        text = "Enter the gx0 in the gx 1"
        result = _normalize_placeholders(text, mapping)
        assert "Gx0" in result
        assert "Gx1" in result
