"""Tests for the NLLB CTranslate2 backend (all deps mocked)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_modules():
    """Return mock ctranslate2 and transformers modules."""
    ct2 = ModuleType("ctranslate2")
    ct2.get_supported_compute_types = MagicMock(side_effect=RuntimeError("no CUDA"))

    mock_translator_cls = MagicMock()
    ct2.Translator = mock_translator_cls

    transformers = ModuleType("transformers")
    mock_auto_tokenizer = MagicMock()
    transformers.AutoTokenizer = mock_auto_tokenizer

    ct2_converters = ModuleType("ctranslate2.converters")
    ct2_transformers = ModuleType("ctranslate2.converters.transformers")

    return ct2, transformers, ct2_converters, ct2_transformers


def _make_hypothesis(tokens: list[str]):
    return SimpleNamespace(hypotheses=[tokens])


def _make_empty_hypothesis():
    return SimpleNamespace(hypotheses=[])


@pytest.fixture()
def mock_nllb_env(tmp_path):
    """Fixture that patches sys.modules and provides a working NLLBBackend."""
    ct2, transformers_mod, ct2_conv, ct2_conv_t = _make_mock_modules()

    patches = {
        "ctranslate2": ct2,
        "ctranslate2.converters": ct2_conv,
        "ctranslate2.converters.transformers": ct2_conv_t,
        "transformers": transformers_mod,
    }

    with patch.dict(sys.modules, patches):
        if "modtranslator.backends.nllb" in sys.modules:
            del sys.modules["modtranslator.backends.nllb"]

        from modtranslator.backends.nllb import NLLBBackend

        # Fake model dir for 1.3B variant
        model_dir = tmp_path / "models" / "nllb-200-distilled-1.3B-ct2-int8"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"fake")

        # Also create dir for 600M variant
        vdir = tmp_path / "models" / "nllb-200-distilled-600M-ct2-int8"
        vdir.mkdir(parents=True)
        (vdir / "model.bin").write_bytes(b"fake")

        # Configure mock tokenizer
        tokenizer = MagicMock()
        tokenizer.src_lang = "eng_Latn"
        tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [f"tok{i}" for i in ids]
        tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        tokenizer.decode.side_effect = lambda ids, **kw: "translated text"
        tokenizer.batch_decode.side_effect = lambda ids_list, **kw: [
            "translated text" for _ in ids_list
        ]
        transformers_mod.AutoTokenizer.from_pretrained.return_value = tokenizer

        # Configure mock translator
        def _translate_batch(tokenized, target_prefix=None, beam_size=2,
                             max_batch_size=256, **kwargs):
            return [_make_hypothesis([f"out{i}" for i in range(3)]) for _ in tokenized]

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = _translate_batch
        ct2.Translator.return_value = mock_translator

        yield SimpleNamespace(
            NLLBBackend=NLLBBackend,
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
    def test_translate_batch_simple(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        result = backend.translate_batch(["Hello", "World"], "ES")
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_translate_empty_batch(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        result = backend.translate_batch([], "ES")
        assert result == []

    def test_translate_single(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        result = backend.translate("Hello", "ES")
        assert isinstance(result, str)

    def test_translate_batch_large(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        texts = [f"text {i}" for i in range(300)]
        result = backend.translate_batch(texts, "ES")
        assert len(result) == 300


class TestFloresCodes:
    def test_target_prefix_passed(self, mock_nllb_env):
        """translate_batch passes target_prefix with FLORES-200 code."""
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        backend.translate_batch(["Hello"], "ES")

        call_kwargs = mock_nllb_env.translator.translate_batch.call_args[1]
        assert call_kwargs["target_prefix"] == [["spa_Latn"]]

    def test_target_prefix_french(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        backend.translate_batch(["Hello"], "FR")

        call_kwargs = mock_nllb_env.translator.translate_batch.call_args[1]
        assert call_kwargs["target_prefix"] == [["fra_Latn"]]

    def test_source_lang_set_on_tokenizer(self, mock_nllb_env):
        """Tokenizer src_lang is set to the FLORES-200 source code."""
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        # First call with default EN source
        backend.translate_batch(["Hello"], "ES")

        # Tokenizer was loaded with src_lang="eng_Latn"
        load_call = mock_nllb_env.transformers.AutoTokenizer.from_pretrained
        call_kwargs = load_call.call_args[1]
        assert call_kwargs["src_lang"] == "eng_Latn"

    def test_unsupported_target_lang_raises(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        with pytest.raises(ValueError, match="Unsupported target language"):
            backend.translate_batch(["Hello"], "XX")

    def test_unsupported_source_lang_raises(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        with pytest.raises(ValueError, match="Unsupported source language"):
            backend.translate_batch(["Hello"], "ES", source_lang="XX")

    def test_multiple_texts_target_prefix(self, mock_nllb_env):
        """Each text in the batch gets its own target_prefix."""
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        backend.translate_batch(["Hello", "World", "Test"], "ES")

        call_kwargs = mock_nllb_env.translator.translate_batch.call_args[1]
        assert call_kwargs["target_prefix"] == [["spa_Latn"]] * 3


class TestModelVariants:
    def test_default_variant_is_1_3b(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        assert backend._model_size == "1.3B"
        assert "nllb-200-distilled-1.3B" in backend._hf_model_name

    def test_variant_600m(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir, model_size="600M"
        )
        assert backend._model_size == "600M"
        assert "nllb-200-distilled-600M" in backend._hf_model_name

    def test_invalid_variant_raises(self, mock_nllb_env):
        with pytest.raises(ValueError, match="Unknown NLLB model size"):
            mock_nllb_env.NLLBBackend(
                device="cpu", models_dir=mock_nllb_env.models_dir, model_size="99B"
            )


class TestDeviceResolution:
    def test_device_auto_cpu_fallback(self, mock_nllb_env):
        mock_nllb_env.ct2.get_supported_compute_types.side_effect = RuntimeError("no CUDA")
        backend = mock_nllb_env.NLLBBackend(
            device="auto", models_dir=mock_nllb_env.models_dir
        )
        assert backend._device == "cpu"
        assert backend._compute_type == "int8"

    def test_device_explicit_cpu(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        assert backend._device == "cpu"
        assert backend._compute_type == "int8"

    def test_device_explicit_cuda(self, mock_nllb_env):
        mock_nllb_env.ct2.get_supported_compute_types.side_effect = None
        mock_nllb_env.ct2.get_supported_compute_types.return_value = {
            "int8_float16", "int8", "float32",
        }
        backend = mock_nllb_env.NLLBBackend(
            device="cuda", models_dir=mock_nllb_env.models_dir
        )
        assert backend._device == "cuda"
        assert backend._compute_type == "int8_float16"


class TestFallbackOnEmptyHypothesis:
    def test_fallback_on_empty_hypothesis(self, mock_nllb_env):
        def _translate_batch_empty(tokenized, target_prefix=None, beam_size=2,
                                   max_batch_size=256, **kwargs):
            return [_make_empty_hypothesis() for _ in tokenized]

        mock_nllb_env.translator.translate_batch.side_effect = _translate_batch_empty

        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        result = backend.translate_batch(["Keep this"], "ES")
        assert result == ["Keep this"]


class TestImportError:
    def test_import_error_ctranslate2(self):
        mods = {k: v for k, v in sys.modules.items()}
        mods["ctranslate2"] = None  # type: ignore[assignment]

        with patch.dict(sys.modules, mods):
            if "modtranslator.backends.nllb" in sys.modules:
                del sys.modules["modtranslator.backends.nllb"]

            with pytest.raises(ImportError, match="ctranslate2"):
                from modtranslator.backends.nllb import NLLBBackend

                NLLBBackend(device="cpu")


class TestModelConversion:
    def test_ensure_model_skips_existing(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )

        with patch("subprocess.run") as mock_run:
            backend._ensure_model()
            mock_run.assert_not_called()

    def test_convert_cleanup_on_failure(self, mock_nllb_env):
        import subprocess

        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )

        # Use a different compute type so the dir doesn't exist yet
        backend._compute_type = "float32"
        new_dir = backend._ct2_model_dir()
        assert not new_dir.exists()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ct2", stderr="conversion failed"
            )
            with pytest.raises(RuntimeError, match="Failed to convert"):
                backend._convert_model(new_dir)

            assert not new_dir.exists()


class TestBatchDecode:
    def test_uses_batch_decode(self, mock_nllb_env):
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        texts = ["Hello", "World", "Test"]
        mock_nllb_env.tokenizer.batch_decode.reset_mock()
        mock_nllb_env.tokenizer.decode.reset_mock()

        backend.translate_batch(texts, "ES")

        assert mock_nllb_env.tokenizer.batch_decode.call_count == 1
        assert mock_nllb_env.tokenizer.decode.call_count == 0


class TestAntiRepetition:
    def test_repetition_penalty_passed(self, mock_nllb_env):
        """translate_batch passes repetition_penalty to CT2."""
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )
        backend.translate_batch(["Hello"], "ES")

        call_kwargs = mock_nllb_env.translator.translate_batch.call_args[1]
        assert call_kwargs["repetition_penalty"] == 1.2


class TestTrimHallucinatedFiller:
    """Tests for _trim_hallucinated_filler() post-processing."""

    def _get_cls(self, mock_nllb_env):
        return mock_nllb_env.NLLBBackend

    def test_trims_es_el(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._trim_hallucinated_filler(
            "Bolsa de sangre es el valor de la moneda.", "Blood Pack"
        ) == "Bolsa de sangre"

    def test_trims_es_una(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._trim_hallucinated_filler(
            "La ketamina es una sustancia", "Ketamine"
        ) == "La ketamina"

    def test_trims_es_un(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._trim_hallucinated_filler(
            "Budweiser es un tipo de cerveza", "Budweiser"
        ) == "Budweiser"

    def test_trims_tambien(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._trim_hallucinated_filler(
            "Methsoundfx también conocido como MethsoundFX", "Methsoundfx"
        ) == "Methsoundfx"

    def test_trims_fue(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._trim_hallucinated_filler(
            "Smirnoff fue elegido por el Consejo", "Smirnoff"
        ) == "Smirnoff"

    def test_no_trim_on_clean_translation(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._trim_hallucinated_filler(
            "Bolsa de sangre", "Blood Pack"
        ) == "Bolsa de sangre"

    def test_no_trim_on_long_input(self, mock_nllb_env):
        """Long inputs are not subject to filler trimming."""
        cls = self._get_cls(mock_nllb_env)
        text = "El arma es un rifle de asalto"
        original = "The weapon is an assault rifle"
        assert cls._trim_hallucinated_filler(text, original) == text

    def test_trims_trailing_punctuation(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._trim_hallucinated_filler(
            "Morfina, es el nombre de un producto.", "Morphine"
        ) == "Morfina"

    def test_does_not_return_empty(self, mock_nllb_env):
        """If trimming would produce empty, return original."""
        cls = self._get_cls(mock_nllb_env)
        result = cls._trim_hallucinated_filler(
            " es el mejor", "Best"
        )
        assert result == " es el mejor"


class TestDeduplicateShort:
    """Tests for _deduplicate_short() post-processing safety net."""

    def _get_backend_class(self, mock_nllb_env):
        return mock_nllb_env.NLLBBackend

    def test_single_word_duplication(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Estimulante Estimulante", "Stimpak") == "Estimulante"

    def test_multi_word_duplication(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short(
            "Bolsa de sangre Bolsa de sangre", "Blood Pack"
        ) == "Bolsa de sangre"

    def test_triple_repetition(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Morfina Morfina Morfina", "Morphine") == "Morfina"

    def test_name_with_hyphen(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Nuka-Cola Nuka-Cola", "Nuka-Cola") == "Nuka-Cola"

    def test_case_insensitive_duplication(self, mock_nllb_env):
        """Catches 'Abierto abierto' as a case-insensitive dupe."""
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Abierto abierto", "Open") == "Abierto"

    def test_no_duplication_passthrough(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Estimulante", "Stimpak") == "Estimulante"

    def test_long_input_skipped(self, mock_nllb_env):
        """Inputs with >5 words are not subject to dedup (avoids false positives)."""
        cls = self._get_backend_class(mock_nllb_env)
        text = "La bolsa La bolsa"
        original = "The blood pack is very useful indeed"
        assert cls._deduplicate_short(text, original) == text

    def test_no_false_positive_different_halves(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short(
            "Bolsa de sangre fresca", "Fresh Blood Pack"
        ) == "Bolsa de sangre fresca"

    def test_empty_string(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("", "") == ""

    def test_single_word_no_repeat(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Morfina", "Morphine") == "Morfina"

    def test_bookend_piedra_de_piedra(self, mock_nllb_env):
        """'Stone' → 'Piedra de piedra' → 'Piedra' (bookend: first==last)."""
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Piedra de piedra", "Stone") == "Piedra"

    def test_bookend_cama_de_cama(self, mock_nllb_env):
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Cama de cama", "Bed") == "Cama"

    def test_bookend_case_insensitive(self, mock_nllb_env):
        """Bookend check is case-insensitive."""
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Casa de la Casa", "House") == "Casa"

    def test_bookend_not_triggered_for_multiword_input(self, mock_nllb_env):
        """Bookend only applies to single-word inputs."""
        cls = self._get_backend_class(mock_nllb_env)
        text = "Piedra de piedra"
        assert cls._deduplicate_short(text, "Big Stone") == text

    def test_bookend_not_triggered_for_different_words(self, mock_nllb_env):
        """No false positive when first != last."""
        cls = self._get_backend_class(mock_nllb_env)
        assert cls._deduplicate_short("Casilla de correos", "Mailbox") == "Casilla de correos"


class TestCapSingleWordOutput:
    """Tests for _cap_single_word_output() post-processing."""

    def _get_cls(self, mock_nllb_env):
        return mock_nllb_env.NLLBBackend

    def test_caps_hallucinated_filler(self, mock_nllb_env):
        """'Light' → 'Luz de la luz' should be capped to 'Luz'."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("Luz de la luz", "Light") == "Luz"

    def test_allows_article_plus_noun(self, mock_nllb_env):
        """'Ketamine' → 'La ketamina' is valid (article + noun)."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("La ketamina", "Ketamine") == "La ketamina"

    def test_single_word_passthrough(self, mock_nllb_env):
        """Clean 1-word output passes through."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("Morfina", "Morphine") == "Morfina"

    def test_two_word_input_not_capped(self, mock_nllb_env):
        """Multi-word inputs are never capped."""
        cls = self._get_cls(mock_nllb_env)
        text = "Bolsa de sangre de la tienda"
        assert cls._cap_single_word_output(text, "Blood Pack") == text

    def test_caps_without_article(self, mock_nllb_env):
        """'Booth' → 'Booth de la tienda' → 'Booth' (first word, no article)."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("Booth de la tienda", "Booth") == "Booth"

    def test_article_el(self, mock_nllb_env):
        """'el' article is recognized."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("El refugio de los muertos", "Vault") == "El refugio"

    def test_article_un(self, mock_nllb_env):
        """'un' article is recognized."""
        cls = self._get_cls(mock_nllb_env)
        result = cls._cap_single_word_output("Un estimulante de combate", "Stimpak")
        assert result == "Un estimulante"

    def test_three_word_compound_passthrough(self, mock_nllb_env):
        """Legitimate 3-word compound passes through (cap=3)."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("Casilla de correos", "Mailbox") == "Casilla de correos"

    def test_three_word_phrase_passthrough(self, mock_nllb_env):
        """Legitimate 3-word phrase passes through."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("No hay nada", "Nothing") == "No hay nada"

    def test_two_word_output_passthrough(self, mock_nllb_env):
        """Exactly 2 words output is within limit."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("Bolsa sangre", "Bloodpack") == "Bolsa sangre"

    def test_empty_string(self, mock_nllb_env):
        cls = self._get_cls(mock_nllb_env)
        assert cls._cap_single_word_output("", "") == ""


class TestStripEcho:
    """Tests for _strip_echo() double-space echo removal."""

    def _get_cls(self, mock_nllb_env):
        return mock_nllb_env.NLLBBackend

    def test_echo_en_at_end(self, mock_nllb_env):
        """EN echoed after Spanish translation."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._strip_echo(
            "Switch de energia  Power switch", "Power Switch"
        ) == "Switch de energia"

    def test_echo_en_at_start(self, mock_nllb_env):
        """EN echoed before Spanish translation."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._strip_echo(
            "Art Loft  Arte en el altillo", "Art Loft"
        ) == "Arte en el altillo"

    def test_double_translation(self, mock_nllb_env):
        """Two Spanish translations: pick part with more non-original words."""
        cls = self._get_cls(mock_nllb_env)
        result = cls._strip_echo(
            "Hotel Fairfax  El hotel", "Fairfax Hotel"
        )
        # "Hotel Fairfax" has 100% overlap with original (reordered), score=0
        # "El hotel" has 1 new word ("el"), score=1 → wins
        assert result == "El hotel"

    def test_no_echo_passthrough(self, mock_nllb_env):
        """Clean translation without double space passes through."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._strip_echo(
            "Bolsa de sangre", "Blood Pack"
        ) == "Bolsa de sangre"

    def test_long_input_skipped(self, mock_nllb_env):
        """Long inputs are not subject to echo stripping."""
        cls = self._get_cls(mock_nllb_env)
        text = "Primera parte  Segunda parte"
        original = "First part of the longer sentence here"
        assert cls._strip_echo(text, original) == text

    def test_single_space_not_split(self, mock_nllb_env):
        """Normal single spaces in translation are preserved."""
        cls = self._get_cls(mock_nllb_env)
        text = "Casilla de correos"
        assert cls._strip_echo(text, "Mailbox") == text

    def test_echo_with_identical_part(self, mock_nllb_env):
        """If one part is identical to original, keep the other."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._strip_echo(
            "Play Cell  Juega con la celda", "Play Cell"
        ) == "Juega con la celda"

    def test_corona_real_echo(self, mock_nllb_env):
        """Real case: Crown Royal — no exact EN substring, passthrough."""
        cls = self._get_cls(mock_nllb_env)
        result = cls._strip_echo(
            "Corona Real de la corona real", "Crown Royal"
        )
        # Original "Crown Royal" not found as substring, passthrough
        assert result == "Corona Real de la corona real"

    def test_en_echo_at_start_no_double_space(self, mock_nllb_env):
        """EN stuck at start without double space separator."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._strip_echo(
            "Play Cell Juega con la celda", "Play Cell"
        ) == "Juega con la celda"

    def test_en_echo_at_end_no_double_space(self, mock_nllb_env):
        """EN stuck at end without double space separator."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._strip_echo(
            "La busqueda no ha logrado Quest Failed", "Quest Failed"
        ) == "La busqueda no ha logrado"

    def test_en_echo_case_insensitive(self, mock_nllb_env):
        """EN echo detection is case-insensitive."""
        cls = self._get_cls(mock_nllb_env)
        assert cls._strip_echo(
            "Ant Vision La vision de la hormiga", "Ant Vision"
        ) == "La vision de la hormiga"

    def test_no_false_positive_short_overlap(self, mock_nllb_env):
        """Don't strip if original is just a common short word."""
        cls = self._get_cls(mock_nllb_env)
        # "Red" appears in "Redes" but not as echo
        assert cls._strip_echo(
            "Vino tinto", "Red Wine"
        ) == "Vino tinto"
