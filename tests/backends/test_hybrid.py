"""Tests for the hybrid (Opus-MT + NLLB) translation backend."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, call, patch

import pytest

from modtranslator.backends.hybrid import DEFAULT_WORD_THRESHOLD, HybridBackend

# Patch targets: the imports happen inside __init__, so patch at source modules
_PATCH_OPUS = "modtranslator.backends.opus_mt.OpusMTBackend"
_PATCH_NLLB = "modtranslator.backends.nllb.NLLBBackend"


@pytest.fixture
def mock_hybrid():
    """Create a HybridBackend with mocked sub-backends.

    Opus-MT is initialized lazily on first translate_batch call.
    mock_tc_big = OpusMTBackend.return_value (the instance _init_opus creates).
    """
    with (
        patch(_PATCH_OPUS) as mock_opus_cls,
        patch(_PATCH_NLLB) as mock_nllb_cls,
    ):
        mock_tc_big = MagicMock()
        mock_nllb = MagicMock()
        mock_nllb._gpu_batch_size = 32  # Default for 8GB VRAM
        mock_opus_cls.return_value = mock_tc_big
        mock_nllb_cls.return_value = mock_nllb

        backend = HybridBackend(device="cpu")

        yield backend, mock_tc_big, mock_nllb, mock_opus_cls, mock_nllb_cls


class TestHybridInit:
    def test_creates_nllb_immediately_opus_is_lazy(self, mock_hybrid):
        """NLLB is created in __init__; Opus-MT is lazy (created on first translate_batch)."""
        _backend, _tc, _nllb, mock_opus_cls, mock_nllb_cls = mock_hybrid
        # NLLB created eagerly
        mock_nllb_cls.assert_called_once_with(device="cpu", model_size="1.3B")
        # Opus NOT created yet (lazy)
        mock_opus_cls.assert_not_called()

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


class TestHybridFallback:
    """Tests for the tc-big → base → NLLB-only fallback chain."""

    def test_falls_back_to_base_when_no_tc_big(self):
        """When tc-big is not available, _init_opus uses base Opus-MT."""
        with (
            patch(_PATCH_OPUS) as mock_opus_cls,
            patch(_PATCH_NLLB) as mock_nllb_cls,
        ):
            tc_big_inst = MagicMock()
            tc_big_inst._ensure_model.side_effect = RuntimeError("model not found")
            base_inst = MagicMock()
            base_inst._ensure_model.return_value = None
            base_inst.translate_batch.return_value = ["Hallo"]

            mock_opus_cls.side_effect = [tc_big_inst, base_inst]
            mock_nllb_cls.return_value = MagicMock(_gpu_batch_size=32)

            backend = HybridBackend(device="cpu")
            result = backend.translate_batch(["Hello"], "DE")

            assert backend._opus is base_inst
            assert result == ["Hallo"]
            # tc-big was tried first, then base
            assert mock_opus_cls.call_args_list == [
                call(device="cpu", model_variant="tc-big"),
                call(device="cpu", model_variant="base"),
            ]

    def test_falls_back_to_nllb_only_when_no_opus(self):
        """When no Opus-MT is available, all strings go to NLLB with a warning."""
        with (
            patch(_PATCH_OPUS) as mock_opus_cls,
            patch(_PATCH_NLLB) as mock_nllb_cls,
        ):
            failing_inst = MagicMock()
            failing_inst._ensure_model.side_effect = RuntimeError("no model")
            mock_opus_cls.return_value = failing_inst

            mock_nllb = MagicMock(_gpu_batch_size=32)
            mock_nllb.translate_batch.return_value = ["Привет"]
            mock_nllb_cls.return_value = mock_nllb

            backend = HybridBackend(device="cpu")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = backend.translate_batch(["Hello"], "RU")
                assert len(w) == 1
                assert "Opus-MT not available" in str(w[0].message)
                assert "NLLB-only" in str(w[0].message)

            assert backend._opus is None
            assert backend.mode == "nllb-only"
            mock_nllb.translate_batch.assert_called_once()
            assert result == ["Привет"]

    def test_nllb_only_routes_all_strings_regardless_of_length(self):
        """In NLLB-only mode, short AND long strings go to NLLB."""
        with (
            patch(_PATCH_OPUS) as mock_opus_cls,
            patch(_PATCH_NLLB) as mock_nllb_cls,
        ):
            failing_inst = MagicMock()
            failing_inst._ensure_model.side_effect = RuntimeError("no model")
            mock_opus_cls.return_value = failing_inst

            mock_nllb = MagicMock(_gpu_batch_size=32)
            mock_nllb.translate_batch.return_value = ["A", "B", "C"]
            mock_nllb_cls.return_value = mock_nllb

            backend = HybridBackend(device="cpu")
            texts = ["Short", "Short two", "A much longer string that exceeds threshold"]
            result = backend.translate_batch(texts, "RU")

            # All texts go to NLLB in one chunk
            mock_nllb.translate_batch.assert_called_once_with(texts, "RU", None)
            assert result == ["A", "B", "C"]

    def test_opus_init_only_happens_once(self):
        """_init_opus is called only on the first non-empty translate_batch."""
        with (
            patch(_PATCH_OPUS) as mock_opus_cls,
            patch(_PATCH_NLLB) as mock_nllb_cls,
        ):
            mock_opus_inst = MagicMock()
            mock_opus_inst._ensure_model.return_value = None
            mock_opus_inst.translate_batch.return_value = ["X"]
            mock_opus_cls.return_value = mock_opus_inst
            mock_nllb_cls.return_value = MagicMock(_gpu_batch_size=32)

            backend = HybridBackend(device="cpu")
            backend.translate_batch(["Hi"], "FR")
            backend.translate_batch(["Bye"], "FR")

            # OpusMTBackend() called exactly twice (tc-big attempt, which succeeds)
            assert mock_opus_cls.call_count == 1


class TestHybridMode:
    """Tests for the mode property."""

    def test_mode_pending_before_init(self, mock_hybrid):
        backend = mock_hybrid[0]
        assert backend.mode == "pending"

    def test_mode_tc_big_after_init(self, mock_hybrid):
        backend, mock_tc, _, _, _ = mock_hybrid
        mock_tc.translate_batch.return_value = ["Hola"]
        backend.translate_batch(["Hello"], "ES", "EN")
        assert backend.mode == "opus-mt-tc-big+nllb"

    def test_mode_base_after_fallback(self):
        with (
            patch(_PATCH_OPUS) as mock_opus_cls,
            patch(_PATCH_NLLB) as mock_nllb_cls,
        ):
            tc_big_inst = MagicMock()
            tc_big_inst._ensure_model.side_effect = RuntimeError("not found")
            base_inst = MagicMock()
            base_inst._ensure_model.return_value = None
            base_inst.translate_batch.return_value = ["Hola"]
            mock_opus_cls.side_effect = [tc_big_inst, base_inst]
            mock_nllb_cls.return_value = MagicMock(_gpu_batch_size=32)

            backend = HybridBackend(device="cpu")
            backend.translate_batch(["Hello"], "ES")
            assert backend.mode == "opus-mt-base+nllb"

    def test_mode_nllb_only(self):
        with (
            patch(_PATCH_OPUS) as mock_opus_cls,
            patch(_PATCH_NLLB) as mock_nllb_cls,
        ):
            failing = MagicMock()
            failing._ensure_model.side_effect = RuntimeError("no model")
            mock_opus_cls.return_value = failing
            mock_nllb = MagicMock(_gpu_batch_size=32)
            mock_nllb.translate_batch.return_value = ["Hola"]
            mock_nllb_cls.return_value = mock_nllb

            backend = HybridBackend(device="cpu")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                backend.translate_batch(["Hello"], "ES")
            assert backend.mode == "nllb-only"


class TestHybridChunkScaling:
    """Tests for NLLB chunk size scaling based on GPU batch size."""

    def test_chunk_size_scales_with_gpu_batch_size(self):
        """Chunk size = gpu_batch_size * 15."""
        with patch(_PATCH_OPUS), patch(_PATCH_NLLB) as mock_nllb_cls:
            mock_nllb = MagicMock()
            mock_nllb._gpu_batch_size = 64  # e.g. 12GB VRAM
            mock_nllb_cls.return_value = mock_nllb

            backend = HybridBackend(device="cpu")
            assert backend._nllb_chunk_size == 64 * 15  # 960

    def test_chunk_size_default_for_8gb(self):
        """Default batch_size=32 → chunk=480 (close to old 500)."""
        with patch(_PATCH_OPUS), patch(_PATCH_NLLB) as mock_nllb_cls:
            mock_nllb = MagicMock()
            mock_nllb._gpu_batch_size = 32
            mock_nllb_cls.return_value = mock_nllb

            backend = HybridBackend(device="cpu")
            assert backend._nllb_chunk_size == 32 * 15  # 480

    def test_chunk_size_128_batch(self):
        """Max batch_size=128 → chunk=1920."""
        with patch(_PATCH_OPUS), patch(_PATCH_NLLB) as mock_nllb_cls:
            mock_nllb = MagicMock()
            mock_nllb._gpu_batch_size = 128
            mock_nllb_cls.return_value = mock_nllb

            backend = HybridBackend(device="cpu")
            assert backend._nllb_chunk_size == 128 * 15  # 1920


class TestHybridCLI:
    def test_cli_creates_hybrid_backend(self):
        """Verify _create_backend handles 'hybrid' backend name."""
        with patch(_PATCH_OPUS), patch(_PATCH_NLLB):
            from modtranslator.pipeline import create_backend

            backend, label = create_backend("hybrid", device="cpu")

            assert isinstance(backend, HybridBackend)
            assert label == "hybrid:tc-big+nllb"
