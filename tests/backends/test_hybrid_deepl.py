"""Tests for the hybrid-deepl (NLLB + DeepL) translation backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modtranslator.backends.hybrid_deepl import DEFAULT_WORD_THRESHOLD, HybridDeepLBackend

_PATCH_DEEPL = "modtranslator.backends.deepl.DeepLBackend"
_PATCH_NLLB = "modtranslator.backends.nllb.NLLBBackend"


@pytest.fixture
def mock_hybrid_deepl():
    """Create a HybridDeepLBackend with mocked sub-backends."""
    with (
        patch(_PATCH_DEEPL) as mock_deepl_cls,
        patch(_PATCH_NLLB) as mock_nllb_cls,
    ):
        mock_deepl = MagicMock()
        mock_nllb = MagicMock()
        mock_nllb._gpu_batch_size = 32
        mock_deepl_cls.return_value = mock_deepl
        mock_nllb_cls.return_value = mock_nllb

        backend = HybridDeepLBackend(api_key="test-key", device="cpu")

        yield backend, mock_deepl, mock_nllb


class TestHybridDeepLBasic:
    def test_empty_batch(self, mock_hybrid_deepl):
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        assert backend.translate_batch([], "ES") == []
        mock_deepl.translate_batch.assert_not_called()
        mock_nllb.translate_batch.assert_not_called()

    def test_default_word_threshold(self, mock_hybrid_deepl):
        backend = mock_hybrid_deepl[0]
        assert backend._word_threshold == DEFAULT_WORD_THRESHOLD

    def test_all_short_to_deepl(self, mock_hybrid_deepl):
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        texts = ["Stimpak", "Power Armor", "Nuka-Cola"]
        mock_deepl.translate_batch.return_value = ["Estimulante", "Servoarmadura", "Nuka-Cola"]

        result = backend.translate_batch(texts, "ES")

        mock_deepl.translate_batch.assert_called_once_with(texts, "ES", None)
        mock_nllb.translate_batch.assert_not_called()
        assert result == ["Estimulante", "Servoarmadura", "Nuka-Cola"]

    def test_all_long_to_nllb(self, mock_hybrid_deepl):
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        texts = [
            "Are you looking for the Mechanist?",
            "I think I have too much blood",
        ]
        mock_nllb.translate_batch.return_value = ["Buscas al Mecanista?", "Creo que tengo mucha sangre"]

        result = backend.translate_batch(texts, "ES")

        mock_nllb.translate_batch.assert_called_once()
        mock_deepl.translate_batch.assert_not_called()
        assert len(result) == 2

    def test_mixed_short_and_long(self, mock_hybrid_deepl):
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        texts = [
            "Stimpak",                              # short -> DeepL
            "Are you looking for the Mechanist?",   # long -> NLLB
            "Power Armor",                          # short -> DeepL
            "Can you spare a liver?",               # long -> NLLB
        ]
        mock_deepl.translate_batch.return_value = ["Estimulante", "Servoarmadura"]
        mock_nllb.translate_batch.return_value = ["Buscas al Mecanista?", "Puedes darme un higado?"]

        result = backend.translate_batch(texts, "ES")

        assert result == [
            "Estimulante",
            "Buscas al Mecanista?",
            "Servoarmadura",
            "Puedes darme un higado?",
        ]


class TestHybridDeepLFallback:
    """Error handling: one backend failing should not block the other."""

    def test_deepl_fails_long_texts_still_translated(self, mock_hybrid_deepl):
        """DeepL failure keeps originals for short but NLLB still translates long."""
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        texts = [
            "Stimpak",                              # short -> DeepL (fails)
            "Are you looking for the Mechanist?",   # long -> NLLB (works)
        ]
        mock_deepl.translate_batch.side_effect = ConnectionError("DeepL timeout")
        mock_nllb.translate_batch.return_value = ["Buscas al Mecanista?"]

        result = backend.translate_batch(texts, "ES")

        assert result[0] == "Stimpak"                # original kept, NOT sent to NLLB
        assert result[1] == "Buscas al Mecanista?"    # long translated normally
        # NLLB only called once (for long texts), never for short fallback
        mock_nllb.translate_batch.assert_called_once()

    def test_deepl_fails_only_short_texts_raises(self, mock_hybrid_deepl):
        """If only short texts and DeepL fails, exception propagates."""
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        texts = ["Stimpak", "Power Armor"]
        mock_deepl.translate_batch.side_effect = ConnectionError("DeepL timeout")

        with pytest.raises(ConnectionError, match="DeepL timeout"):
            backend.translate_batch(texts, "ES")

    def test_nllb_fails_falls_back_to_deepl_for_long(self, mock_hybrid_deepl):
        """NLLB failure sends long texts to DeepL as fallback."""
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        texts = [
            "Stimpak",                              # short -> DeepL
            "Are you looking for the Mechanist?",   # long -> NLLB (fails) -> DeepL
        ]
        mock_deepl.translate_batch.side_effect = [
            ["Estimulante"],                          # normal for short
            ["Buscas al Mecanista?"],                 # fallback for long
        ]
        mock_nllb.translate_batch.side_effect = RuntimeError("CUDA OOM")

        result = backend.translate_batch(texts, "ES")

        assert result[0] == "Estimulante"
        assert result[1] == "Buscas al Mecanista?"

    def test_nllb_fails_deepl_fallback_also_fails(self, mock_hybrid_deepl):
        """Both fail for long texts -> originals preserved."""
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        texts = [
            "Stimpak",                              # short
            "Are you looking for the Mechanist?",   # long
        ]
        mock_deepl.translate_batch.side_effect = [
            ["Estimulante"],                          # short OK
            ConnectionError("DeepL also down"),       # long fallback fails
        ]
        mock_nllb.translate_batch.side_effect = RuntimeError("CUDA OOM")

        result = backend.translate_batch(texts, "ES")

        assert result[0] == "Estimulante"
        assert result[1] == "Are you looking for the Mechanist?"

    def test_nllb_partial_chunk_failure(self, mock_hybrid_deepl):
        """NLLB fails mid-chunk: already-translated kept, rest goes to DeepL."""
        backend, mock_deepl, mock_nllb = mock_hybrid_deepl
        backend._nllb_chunk_size = 1  # force 1 text per chunk
        texts = [
            "First long string with many words",
            "Second long string with many words",
        ]
        mock_nllb.translate_batch.side_effect = [
            ["Primera cadena larga"],           # chunk 1 OK
            RuntimeError("OOM on chunk 2"),     # chunk 2 fails
        ]
        mock_deepl.translate_batch.return_value = ["Segunda cadena larga"]

        result = backend.translate_batch(texts, "ES")

        assert result[0] == "Primera cadena larga"
        assert result[1] == "Segunda cadena larga"


class TestHybridDeepLCLI:
    def test_cli_creates_hybrid_deepl_backend(self):
        with patch(_PATCH_DEEPL), patch(_PATCH_NLLB):
            from modtranslator.pipeline import create_backend

            backend, label = create_backend("hybrid-deepl", api_key="test-key", device="cpu")

            assert isinstance(backend, HybridDeepLBackend)
            assert label == "hybrid:nllb+deepl"

    def test_cli_hybrid_deepl_requires_api_key(self):
        from modtranslator.pipeline import create_backend

        with pytest.raises(ValueError, match="API key required"):
            create_backend("hybrid-deepl")
