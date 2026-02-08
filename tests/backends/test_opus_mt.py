"""Tests for the Opus-MT CTranslate2 backend (all deps mocked)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build mock ctranslate2 / transformers modules
# ---------------------------------------------------------------------------

def _make_mock_modules():
    """Return mock ctranslate2 and transformers modules."""
    ct2 = ModuleType("ctranslate2")
    ct2.get_supported_compute_types = MagicMock(side_effect=RuntimeError("no CUDA"))

    # Translator mock: translate_batch returns hypothesis objects
    mock_translator_cls = MagicMock()
    ct2.Translator = mock_translator_cls

    transformers = ModuleType("transformers")
    mock_auto_tokenizer = MagicMock()
    transformers.AutoTokenizer = mock_auto_tokenizer

    # Converter submodule (for subprocess import check)
    ct2_converters = ModuleType("ctranslate2.converters")
    ct2_transformers = ModuleType("ctranslate2.converters.transformers")

    return ct2, transformers, ct2_converters, ct2_transformers


def _make_hypothesis(tokens: list[str]):
    """Create a mock translation result with hypotheses."""
    hyp = SimpleNamespace(hypotheses=[tokens])
    return hyp


def _make_empty_hypothesis():
    return SimpleNamespace(hypotheses=[])


@pytest.fixture()
def mock_opus_env(tmp_path):
    """Fixture that patches sys.modules and provides a working OpusMTBackend."""
    ct2, transformers_mod, ct2_conv, ct2_conv_t = _make_mock_modules()

    patches = {
        "ctranslate2": ct2,
        "ctranslate2.converters": ct2_conv,
        "ctranslate2.converters.transformers": ct2_conv_t,
        "transformers": transformers_mod,
    }

    with patch.dict(sys.modules, patches):
        # Force reimport so the backend picks up our mocks
        if "modtranslator.backends.opus_mt" in sys.modules:
            del sys.modules["modtranslator.backends.opus_mt"]

        from modtranslator.backends.opus_mt import OpusMTBackend

        # Set up a fake model dir with model.bin already present.
        # Name must match _ct2_model_dir format: opus-mt-{src}-{tgt}-ct2-{compute_type}
        model_dir = tmp_path / "models" / "opus-mt-en-es-ct2-int8"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"fake")

        # Configure the mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [f"tok{i}" for i in ids]
        tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        tokenizer.decode.side_effect = lambda ids, **kw: "translated text"
        tokenizer.batch_decode.side_effect = lambda ids_list, **kw: [
            "translated text" for _ in ids_list
        ]
        transformers_mod.AutoTokenizer.from_pretrained.return_value = tokenizer

        # Configure mock translator
        def _translate_batch(tokenized, beam_size=2, max_batch_size=256,
                             repetition_penalty=1.0, no_repeat_ngram_size=0):
            return [_make_hypothesis([f"out{i}" for i in range(3)]) for _ in tokenized]

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = _translate_batch
        ct2.Translator.return_value = mock_translator

        yield SimpleNamespace(
            OpusMTBackend=OpusMTBackend,
            ct2=ct2,
            transformers=transformers_mod,
            tokenizer=tokenizer,
            translator=mock_translator,
            models_dir=tmp_path / "models",
        )


@pytest.fixture()
def mock_opus_tc_big_env(tmp_path):
    """Fixture with tc-big model directory pre-created."""
    ct2, transformers_mod, ct2_conv, ct2_conv_t = _make_mock_modules()

    patches = {
        "ctranslate2": ct2,
        "ctranslate2.converters": ct2_conv,
        "ctranslate2.converters.transformers": ct2_conv_t,
        "transformers": transformers_mod,
    }

    with patch.dict(sys.modules, patches):
        if "modtranslator.backends.opus_mt" in sys.modules:
            del sys.modules["modtranslator.backends.opus_mt"]

        from modtranslator.backends.opus_mt import OpusMTBackend

        # tc-big model dir: opus-mt-tc-big-en-es-ct2-int8
        model_dir = tmp_path / "models" / "opus-mt-tc-big-en-es-ct2-int8"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"fake")

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [f"tok{i}" for i in ids]
        tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        tokenizer.decode.side_effect = lambda ids, **kw: "translated text"
        tokenizer.batch_decode.side_effect = lambda ids_list, **kw: [
            "translated text" for _ in ids_list
        ]
        transformers_mod.AutoTokenizer.from_pretrained.return_value = tokenizer

        def _translate_batch(tokenized, beam_size=4, max_batch_size=256,
                             repetition_penalty=1.1, no_repeat_ngram_size=3):
            return [_make_hypothesis([f"out{i}" for i in range(3)]) for _ in tokenized]

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = _translate_batch
        ct2.Translator.return_value = mock_translator

        yield SimpleNamespace(
            OpusMTBackend=OpusMTBackend,
            ct2=ct2,
            transformers=transformers_mod,
            tokenizer=tokenizer,
            translator=mock_translator,
            models_dir=tmp_path / "models",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTranslateBatch:
    def test_translate_batch_simple(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        result = backend.translate_batch(["Hello", "World"], "ES")
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_translate_empty_batch(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        result = backend.translate_batch([], "ES")
        assert result == []

    def test_translate_single(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        result = backend.translate("Hello", "ES")
        assert isinstance(result, str)

    def test_translate_batch_large(self, mock_opus_env):
        """CTranslate2 handles sub-batching internally via max_batch_size."""
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        texts = [f"text {i}" for i in range(300)]
        result = backend.translate_batch(texts, "ES")
        assert len(result) == 300

        # Single call to CT2 — it handles sub-batching via max_batch_size=256
        calls = mock_opus_env.translator.translate_batch.call_args_list
        assert len(calls) == 1
        assert len(calls[0][0][0]) == 300
        assert calls[0][1]["max_batch_size"] == 256


class TestDefaultSourceLang:
    def test_default_source_lang_is_en(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        backend.translate_batch(["test"], "ES")

        # Translator should have been created for en→es
        mock_opus_env.ct2.Translator.assert_called_once()
        call_args = mock_opus_env.ct2.Translator.call_args
        model_path = call_args[0][0]
        assert "opus-mt-en-es" in model_path


class TestModelName:
    def test_model_name_construction(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        assert backend._model_name("en", "es") == "Helsinki-NLP/opus-mt-en-es"
        assert backend._model_name("en", "fr") == "Helsinki-NLP/opus-mt-en-fr"


class TestDeviceResolution:
    def test_device_auto_cpu_fallback(self, mock_opus_env):
        """Without CUDA, auto should fall back to cpu."""
        mock_opus_env.ct2.get_supported_compute_types.side_effect = RuntimeError("no CUDA")
        backend = mock_opus_env.OpusMTBackend(
            device="auto", models_dir=mock_opus_env.models_dir
        )
        assert backend._device == "cpu"
        assert backend._compute_type == "int8"

    def test_device_explicit_cpu(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        assert backend._device == "cpu"
        assert backend._compute_type == "int8"

    def test_device_explicit_cuda(self, mock_opus_env):
        # When device is explicitly "cuda", get_supported_compute_types must
        # return valid types so _resolve_compute_type picks the best one.
        mock_opus_env.ct2.get_supported_compute_types.side_effect = None
        mock_opus_env.ct2.get_supported_compute_types.return_value = {
            "int8_float16", "int8", "float32",
        }
        backend = mock_opus_env.OpusMTBackend(
            device="cuda", models_dir=mock_opus_env.models_dir
        )
        assert backend._device == "cuda"
        assert backend._compute_type == "int8_float16"

    def test_compute_type_mapping(self, mock_opus_env):
        cpu_backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        assert cpu_backend._compute_type == "int8"

        mock_opus_env.ct2.get_supported_compute_types.side_effect = None
        mock_opus_env.ct2.get_supported_compute_types.return_value = {
            "int8_float16", "int8", "float32",
        }
        cuda_backend = mock_opus_env.OpusMTBackend(
            device="cuda", models_dir=mock_opus_env.models_dir
        )
        assert cuda_backend._compute_type == "int8_float16"


class TestFallbackOnEmptyHypothesis:
    def test_fallback_on_empty_hypothesis(self, mock_opus_env):
        """When translator returns empty hypotheses, return original text."""

        def _translate_batch_empty(tokenized, beam_size=2, max_batch_size=256,
                                   repetition_penalty=1.0, no_repeat_ngram_size=0):
            return [_make_empty_hypothesis() for _ in tokenized]

        mock_opus_env.translator.translate_batch.side_effect = _translate_batch_empty

        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        result = backend.translate_batch(["Keep this"], "ES")
        assert result == ["Keep this"]


class TestImportError:
    def test_import_error_ctranslate2(self):
        """ImportError is raised with a clear message when ctranslate2 is missing."""
        mods = {k: v for k, v in sys.modules.items()}
        mods["ctranslate2"] = None  # type: ignore[assignment]

        with patch.dict(sys.modules, mods):
            if "modtranslator.backends.opus_mt" in sys.modules:
                del sys.modules["modtranslator.backends.opus_mt"]

            with pytest.raises(ImportError, match="ctranslate2"):
                from modtranslator.backends.opus_mt import OpusMTBackend

                OpusMTBackend(device="cpu")


class TestCharHeuristicSkipsEncode:
    def test_short_texts_skip_first_encode(self, mock_opus_env):
        """Texts below CHAR_HEURISTIC_THRESHOLD should not trigger the Phase 2 encode."""
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        # Short texts (well under 1200 chars) — only Phase 3 tokenize should call encode
        texts = ["Hello world", "Short text"]
        mock_opus_env.tokenizer.encode.reset_mock()

        backend.translate_batch(texts, "ES")

        # Phase 3 encodes each segment once (via ThreadPool). Phase 2 never runs.
        # Total encode calls = len(texts) (one per segment in Phase 3)
        assert mock_opus_env.tokenizer.encode.call_count == len(texts)

    def test_long_text_triggers_phase2_encode(self, mock_opus_env):
        """Texts at or above CHAR_HEURISTIC_THRESHOLD get an extra encode in Phase 2."""
        from modtranslator.backends.opus_mt import CHAR_HEURISTIC_THRESHOLD

        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        long_text = "word " * (CHAR_HEURISTIC_THRESHOLD // 5 + 1)  # exceeds threshold
        short_text = "Hello"
        texts = [short_text, long_text]
        mock_opus_env.tokenizer.encode.reset_mock()

        backend.translate_batch(texts, "ES")

        # Phase 2 encodes the long text once. Phase 3 encodes both segments.
        # Total = 1 (Phase 2 for long) + 2 (Phase 3 for both) = 3
        assert mock_opus_env.tokenizer.encode.call_count == 3


class TestBatchDecode:
    def test_uses_batch_decode(self, mock_opus_env):
        """batch_decode should be called instead of individual decode calls."""
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        texts = ["Hello", "World", "Test"]
        mock_opus_env.tokenizer.batch_decode.reset_mock()
        mock_opus_env.tokenizer.decode.reset_mock()

        backend.translate_batch(texts, "ES")

        # batch_decode called exactly once
        assert mock_opus_env.tokenizer.batch_decode.call_count == 1
        # individual decode never called
        assert mock_opus_env.tokenizer.decode.call_count == 0

    def test_batch_decode_receives_all_ids(self, mock_opus_env):
        """batch_decode should receive one list of ids per segment."""
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        texts = ["one two", "three four five"]

        backend.translate_batch(texts, "ES")

        call_args = mock_opus_env.tokenizer.batch_decode.call_args
        ids_list = call_args[0][0]
        assert len(ids_list) == 2  # one per input text


class TestThreadPoolUsage:
    def test_threadpool_is_used(self, mock_opus_env):
        """Verify ThreadPoolExecutor is actually invoked for tokenization."""
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )

        with patch(
            "modtranslator.backends.opus_mt.ThreadPoolExecutor"
        ) as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool.__enter__ = MagicMock(return_value=mock_pool)
            mock_pool.__exit__ = MagicMock(return_value=False)
            # Make pool.map return tokenized results
            mock_pool.map.side_effect = lambda fn, segs: [fn(s) for s in segs]
            mock_pool_cls.return_value = mock_pool

            backend.translate_batch(["Hello", "World"], "ES")

            mock_pool_cls.assert_called_once_with(max_workers=4)
            mock_pool.map.assert_called_once()


class TestModelConversion:
    def test_ensure_model_skips_existing(self, mock_opus_env):
        """If model.bin exists, no conversion happens."""
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )

        with patch("subprocess.run") as mock_run:
            backend._ensure_model("en", "es")
            mock_run.assert_not_called()

    def test_convert_cleanup_on_failure(self, mock_opus_env, tmp_path):
        """Failed conversion cleans up partial directory."""
        import subprocess

        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )

        # Point to a new language pair so model.bin doesn't exist
        new_dir = mock_opus_env.models_dir / "opus-mt-en-fr-ct2"
        assert not new_dir.exists()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ct2", stderr="conversion failed"
            )
            with pytest.raises(RuntimeError, match="Failed to convert"):
                backend._convert_model("Helsinki-NLP/opus-mt-en-fr", new_dir)

            # Directory should have been cleaned up
            assert not new_dir.exists()


class TestModelVariants:
    def test_default_variant_is_base(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir
        )
        assert backend._model_variant == "base"
        assert backend._model_name("en", "es") == "Helsinki-NLP/opus-mt-en-es"

    def test_tc_big_variant_model_name(self, mock_opus_tc_big_env):
        backend = mock_opus_tc_big_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_tc_big_env.models_dir,
            model_variant="tc-big",
        )
        assert backend._model_variant == "tc-big"
        assert backend._model_name("en", "es") == "Helsinki-NLP/opus-mt-tc-big-en-es"
        assert backend._model_name("en", "fr") == "Helsinki-NLP/opus-mt-tc-big-en-fr"

    def test_tc_big_default_beam_size(self, mock_opus_tc_big_env):
        backend = mock_opus_tc_big_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_tc_big_env.models_dir,
            model_variant="tc-big",
        )
        assert backend._beam_size == 4

    def test_base_default_beam_size(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir,
        )
        assert backend._beam_size == 2

    def test_explicit_beam_overrides_variant_default(self, mock_opus_tc_big_env):
        backend = mock_opus_tc_big_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_tc_big_env.models_dir,
            model_variant="tc-big", beam_size=2,
        )
        assert backend._beam_size == 2

    def test_tc_big_passes_repetition_penalty(self, mock_opus_tc_big_env):
        backend = mock_opus_tc_big_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_tc_big_env.models_dir,
            model_variant="tc-big",
        )
        backend.translate_batch(["Hello world"], "ES")

        call_kwargs = mock_opus_tc_big_env.translator.translate_batch.call_args[1]
        assert call_kwargs["repetition_penalty"] == 1.1
        assert call_kwargs["no_repeat_ngram_size"] == 3
        assert call_kwargs["beam_size"] == 4

    def test_base_passes_default_penalties(self, mock_opus_env):
        backend = mock_opus_env.OpusMTBackend(
            device="cpu", models_dir=mock_opus_env.models_dir,
        )
        backend.translate_batch(["Hello world"], "ES")

        call_kwargs = mock_opus_env.translator.translate_batch.call_args[1]
        assert call_kwargs["repetition_penalty"] == 1.0
        assert call_kwargs["no_repeat_ngram_size"] == 0
        assert call_kwargs["beam_size"] == 2

    def test_invalid_variant_raises(self, mock_opus_env):
        with pytest.raises(ValueError, match="Unknown Opus-MT model variant"):
            mock_opus_env.OpusMTBackend(
                device="cpu", models_dir=mock_opus_env.models_dir,
                model_variant="nonexistent",
            )
