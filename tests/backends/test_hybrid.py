"""Tests for the hybrid (tc-big + NLLB) translation backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modtranslator.backends.hybrid import DEFAULT_WORD_THRESHOLD, HybridBackend

# Patch targets: the imports happen inside __init__, so patch at source modules
_PATCH_OPUS = "modtranslator.backends.opus_mt.OpusMTBackend"
_PATCH_NLLB = "modtranslator.backends.nllb.NLLBBackend"


@pytest.fixture
def mock_hybrid():
    """Create a HybridBackend with mocked sub-backends."""
    with (
        patch(_PATCH_OPUS) as mock_opus_cls,
        patch(_PATCH_NLLB) as mock_nllb_cls,
    ):
        mock_tc_big = MagicMock()
        mock_nllb = MagicMock()
        mock_opus_cls.return_value = mock_tc_big
        mock_nllb_cls.return_value = mock_nllb

        backend = HybridBackend(device="cpu")

        yield backend, mock_tc_big, mock_nllb, mock_opus_cls, mock_nllb_cls


class TestHybridInit:
    def test_creates_tc_big_and_nllb(self, mock_hybrid):
        _backend, _tc, _nllb, mock_opus_cls, mock_nllb_cls = mock_hybrid
        mock_opus_cls.assert_called_once_with(device="cpu", model_variant="tc-big")
        mock_nllb_cls.assert_called_once_with(device="cpu", model_size="1.3B")

    def test_custom_nllb_model_size(self):
        with patch(_PATCH_OPUS), patch(_PATCH_NLLB) as mock_nllb_cls:
            HybridBackend(device="cpu", nllb_model_size="600M")
            mock_nllb_cls.assert_called_once_with(device="cpu", model_size="600M")

    def test_custom_word_threshold(self):
        with patch(_PATCH_OPUS), patch(_PATCH_NLLB):
            backend = HybridBackend(device="cpu", word_threshold=6)
            assert backend._word_threshold == 6

    def test_default_word_threshold(self, mock_hybrid):
        backend = mock_hybrid[0]
        assert backend._word_threshold == DEFAULT_WORD_THRESHOLD


class TestHybridTranslateBatch:
    def test_empty_batch(self, mock_hybrid):
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        result = backend.translate_batch([], "ES")
        assert result == []
        mock_tc.translate_batch.assert_not_called()
        mock_nllb.translate_batch.assert_not_called()

    def test_all_short_strings(self, mock_hybrid):
        """Strings with <4 words should all go to tc-big."""
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        texts = ["Stimpak", "Power Armor", "Nuka-Cola Quantum"]
        mock_tc.translate_batch.return_value = ["Estimulante", "Servoarmadura", "Nuka-Cola Quantum"]

        result = backend.translate_batch(texts, "ES", "EN")

        mock_tc.translate_batch.assert_called_once_with(texts, "ES", "EN")
        mock_nllb.translate_batch.assert_not_called()
        assert result == ["Estimulante", "Servoarmadura", "Nuka-Cola Quantum"]

    def test_all_long_strings(self, mock_hybrid):
        """Strings with >=4 words should all go to NLLB."""
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        texts = [
            "Are you looking for the Mechanist?",
            "I think I have too much blood in my Alcohol system!",
        ]
        mock_nllb.translate_batch.return_value = [
            "Estas buscando al Mecanista?",
            "Creo que tengo demasiada sangre en mi sistema de alcohol!",
        ]

        result = backend.translate_batch(texts, "ES")

        mock_nllb.translate_batch.assert_called_once_with(texts, "ES", None)
        mock_tc.translate_batch.assert_not_called()
        assert len(result) == 2

    def test_mixed_short_and_long(self, mock_hybrid):
        """Mixed batch: short strings go to tc-big, long to NLLB."""
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        texts = [
            "Stimpak",                              # 1 word -> tc-big [0]
            "Are you looking for the Mechanist?",   # 7 words -> NLLB [1]
            "Power Armor",                          # 2 words -> tc-big [2]
            "Can you spare a liver?",               # 5 words -> NLLB [3]
        ]
        mock_tc.translate_batch.return_value = ["Estimulante", "Servoarmadura"]
        mock_nllb.translate_batch.return_value = [
            "Estas buscando al Mecanista?",
            "Puedes darme un higado?",
        ]

        result = backend.translate_batch(texts, "ES", "EN")

        # tc-big gets short strings
        mock_tc.translate_batch.assert_called_once_with(
            ["Stimpak", "Power Armor"], "ES", "EN"
        )
        # NLLB gets long strings
        mock_nllb.translate_batch.assert_called_once_with(
            ["Are you looking for the Mechanist?", "Can you spare a liver?"],
            "ES", "EN",
        )
        # Results reassembled in original order
        assert result == [
            "Estimulante",
            "Estas buscando al Mecanista?",
            "Servoarmadura",
            "Puedes darme un higado?",
        ]

    def test_preserves_original_order(self, mock_hybrid):
        """Verify order is preserved even with interleaved short/long strings."""
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        texts = [
            "This is a long sentence with many words in it",  # long [0]
            "Short",                                           # short [1]
            "Another very long sentence to test ordering",     # long [2]
            "Two words",                                       # short [3]
            "Yet another sentence for NLLB backend",           # long [4]
        ]
        mock_tc.translate_batch.return_value = ["Corto", "Dos palabras"]
        mock_nllb.translate_batch.return_value = ["Larga1", "Larga2", "Larga3"]

        result = backend.translate_batch(texts, "ES")

        assert result == ["Larga1", "Corto", "Larga2", "Dos palabras", "Larga3"]

    def test_threshold_boundary(self, mock_hybrid):
        """Strings with exactly threshold words go to NLLB."""
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        # threshold=4: "one two three" (3 words) -> tc-big, "one two three four" (4 words) -> NLLB
        texts = ["one two three", "one two three four"]
        mock_tc.translate_batch.return_value = ["uno dos tres"]
        mock_nllb.translate_batch.return_value = ["uno dos tres cuatro"]

        result = backend.translate_batch(texts, "ES")

        mock_tc.translate_batch.assert_called_once_with(["one two three"], "ES", None)
        mock_nllb.translate_batch.assert_called_once_with(
            ["one two three four"], "ES", None
        )
        assert result == ["uno dos tres", "uno dos tres cuatro"]

    def test_single_string_short(self, mock_hybrid):
        """Single short string uses translate inherited from base."""
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        mock_tc.translate_batch.return_value = ["Estimulante"]

        result = backend.translate("Stimpak", "ES")

        mock_tc.translate_batch.assert_called_once()
        assert result == "Estimulante"

    def test_single_string_long(self, mock_hybrid):
        """Single long string uses translate inherited from base."""
        backend, mock_tc, mock_nllb, _, _ = mock_hybrid
        mock_nllb.translate_batch.return_value = ["Estas buscando al Mecanista?"]

        result = backend.translate("Are you looking for the Mechanist?", "ES")

        mock_nllb.translate_batch.assert_called_once()
        assert result == "Estas buscando al Mecanista?"


class TestHybridCLI:
    def test_cli_creates_hybrid_backend(self):
        """Verify _create_backend handles 'hybrid' backend name."""
        with patch(_PATCH_OPUS), patch(_PATCH_NLLB):
            from modtranslator.cli import _create_backend

            backend, label = _create_backend("hybrid", device="cpu")

            assert isinstance(backend, HybridBackend)
            assert label == "hybrid:tc-big+nllb"
