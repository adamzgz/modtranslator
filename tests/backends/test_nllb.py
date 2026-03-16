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


class TestMultilangPostProcessing:
    """Tests for language-aware post-processing in filler trim and cap."""

    def test_trim_filler_french_est_le(self):
        from modtranslator.backends.nllb import NLLBBackend
        assert NLLBBackend._trim_hallucinated_filler(
            "Sac de sang est le seul remède", "Blood Pack", "FR"
        ) == "Sac de sang"

    def test_trim_filler_french_aussi(self):
        from modtranslator.backends.nllb import NLLBBackend
        assert NLLBBackend._trim_hallucinated_filler(
            "Kétamine aussi connue comme", "Ketamine", "FR"
        ) == "Kétamine"

    def test_trim_filler_german_ist_der(self):
        from modtranslator.backends.nllb import NLLBBackend
        assert NLLBBackend._trim_hallucinated_filler(
            "Blutbeutel ist der Name eines Produkts", "Blood Pack", "DE"
        ) == "Blutbeutel"

    def test_trim_filler_italian_è_il(self):
        from modtranslator.backends.nllb import NLLBBackend
        assert NLLBBackend._trim_hallucinated_filler(
            "Sacca di sangue è il prodotto", "Blood Pack", "IT"
        ) == "Sacca di sangue"

    def test_trim_filler_unknown_lang_passthrough(self):
        """Languages without a filler pattern return translated unchanged."""
        from modtranslator.backends.nllb import NLLBBackend
        text = "Кровяной пакет это продукт"
        assert NLLBBackend._trim_hallucinated_filler(text, "Blood Pack", "RU") == text

    def test_cap_single_word_french_article(self):
        """French article 'le' is recognized, allowing article + noun."""
        from modtranslator.backends.nllb import NLLBBackend
        assert NLLBBackend._cap_single_word_output(
            "Le médicament de la douleur", "Stimpak", "FR"
        ) == "Le médicament"

    def test_cap_single_word_german_article(self):
        """German article 'der' is recognized."""
        from modtranslator.backends.nllb import NLLBBackend
        assert NLLBBackend._cap_single_word_output(
            "Der Stimulant der Energie", "Stimpak", "DE"
        ) == "Der Stimulant"

    def test_cap_single_word_unknown_lang_no_article(self):
        """Unknown language: no article recognized, returns first word when >3 words output."""
        from modtranslator.backends.nllb import NLLBBackend
        # 4-word output for 1-word input, no RU articles recognized → first word
        assert NLLBBackend._cap_single_word_output(
            "Стимулятор боевой единицы армии", "Stimpak", "RU"
        ) == "Стимулятор"


class TestDetectVram:
    """Tests for _detect_vram_mb() VRAM detection."""

    def test_detect_vram_success(self):
        from modtranslator.backends.nllb import _detect_vram_mb

        with patch("modtranslator.backends.nllb.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="8192\n")
            assert _detect_vram_mb() == 8192

    def test_detect_vram_multi_gpu_takes_first(self):
        from modtranslator.backends.nllb import _detect_vram_mb

        with patch("modtranslator.backends.nllb.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="24576\n12288\n")
            assert _detect_vram_mb() == 24576

    def test_detect_vram_nvidia_smi_not_found(self):
        from modtranslator.backends.nllb import _detect_vram_mb

        with patch("modtranslator.backends.nllb.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("not found")
            assert _detect_vram_mb() is None

    def test_detect_vram_nonzero_returncode(self):
        from modtranslator.backends.nllb import _detect_vram_mb

        with patch("modtranslator.backends.nllb.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert _detect_vram_mb() is None

    def test_detect_vram_timeout(self):
        import subprocess as sp

        from modtranslator.backends.nllb import _detect_vram_mb

        with patch("modtranslator.backends.nllb.subprocess.run") as mock_run:
            mock_run.side_effect = sp.TimeoutExpired("nvidia-smi", 5)
            assert _detect_vram_mb() is None


class TestComputeGpuBatchSize:
    """Tests for _compute_gpu_batch_size() scaling function."""

    def test_6gb_vram(self):
        from modtranslator.backends.nllb import _compute_gpu_batch_size

        result = _compute_gpu_batch_size(6144)
        assert result == 44  # (6144 - 2600) // 80 = 44

    def test_8gb_vram(self):
        from modtranslator.backends.nllb import _compute_gpu_batch_size

        # 8192 MB → (8192 - 2600) // 80 = 69
        result = _compute_gpu_batch_size(8192)
        assert result == 69

    def test_12gb_vram(self):
        from modtranslator.backends.nllb import _compute_gpu_batch_size

        result = _compute_gpu_batch_size(12288)
        assert result == 121

    def test_24gb_vram_capped_at_128(self):
        from modtranslator.backends.nllb import _compute_gpu_batch_size

        result = _compute_gpu_batch_size(24576)
        assert result == 128

    def test_4gb_vram_floor_at_16(self):
        from modtranslator.backends.nllb import _compute_gpu_batch_size

        result = _compute_gpu_batch_size(4096)
        assert result == 18  # (4096 - 2600) // 80 = 18

    def test_very_low_vram_floor_at_16(self):
        from modtranslator.backends.nllb import _compute_gpu_batch_size

        result = _compute_gpu_batch_size(3000)
        assert result == 16  # clamped to min


class TestOomRetry:
    """Tests for OOM retry logic in translate_batch."""

    def test_oom_retries_with_half_batch(self, mock_nllb_env):
        """On CUDA OOM, batch size is halved and translation retried."""
        backend = mock_nllb_env.NLLBBackend(
            device="cuda", models_dir=mock_nllb_env.models_dir
        )
        backend._gpu_batch_size = 64

        call_count = 0

        def _translate_batch_oom(tokenized, target_prefix=None, beam_size=2,
                                 max_batch_size=256, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return [_make_hypothesis([f"out{i}" for i in range(3)]) for _ in tokenized]

        mock_nllb_env.translator.translate_batch.side_effect = _translate_batch_oom

        result = backend.translate_batch(["Hello"], "ES")
        assert len(result) == 1
        assert backend._gpu_batch_size == 32  # halved from 64

    def test_oom_falls_to_cpu_after_retry_fails(self, mock_nllb_env):
        """If retry also OOMs, falls back to CPU."""
        backend = mock_nllb_env.NLLBBackend(
            device="cuda", models_dir=mock_nllb_env.models_dir
        )
        backend._gpu_batch_size = 64

        def _always_oom(tokenized, target_prefix=None, beam_size=2,
                        max_batch_size=256, **kwargs):
            raise RuntimeError("CUDA out of memory")

        mock_nllb_env.translator.translate_batch.side_effect = _always_oom

        # After both GPU attempts fail, _fallback_to_cpu is called.
        # The CPU translator mock returns normal results.
        cpu_translator = MagicMock()
        cpu_translator.translate_batch.side_effect = lambda tokenized, **kw: [
            _make_hypothesis(["ok"]) for _ in tokenized
        ]

        with patch.object(backend, "_fallback_to_cpu", return_value=cpu_translator):
            result = backend.translate_batch(["Hello"], "ES")
            assert len(result) == 1
            assert backend._gpu_batch_size == 32  # halved

    def test_non_oom_runtime_error_falls_to_cpu_directly(self, mock_nllb_env):
        """Non-OOM RuntimeError skips retry and goes straight to CPU fallback."""
        backend = mock_nllb_env.NLLBBackend(
            device="cuda", models_dir=mock_nllb_env.models_dir
        )
        backend._gpu_batch_size = 64

        def _other_error(tokenized, target_prefix=None, beam_size=2,
                         max_batch_size=256, **kwargs):
            raise RuntimeError("some other CUDA error")

        mock_nllb_env.translator.translate_batch.side_effect = _other_error

        cpu_translator = MagicMock()
        cpu_translator.translate_batch.side_effect = lambda tokenized, **kw: [
            _make_hypothesis(["ok"]) for _ in tokenized
        ]

        with patch.object(backend, "_fallback_to_cpu", return_value=cpu_translator):
            result = backend.translate_batch(["Hello"], "ES")
            assert len(result) == 1
            assert backend._gpu_batch_size == 64  # unchanged, no OOM retry


class TestNLLBEdgeCases:
    def test_unsupported_source_language(self, mock_nllb_env):
        """Unsupported source language raises ValueError."""
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir,
        )

        with pytest.raises(ValueError, match="Unsupported source language"):
            backend.translate_batch(["Hello"], "ES", source_lang="XX")

    def test_unsupported_target_language(self, mock_nllb_env):
        """Unsupported target language raises ValueError."""
        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir,
        )

        with pytest.raises(ValueError, match="Unsupported target language"):
            backend.translate_batch(["Hello"], "XX")

    def test_compute_gpu_batch_size_zero_vram(self):
        """_compute_gpu_batch_size with VRAM=0 clamps to 16."""
        from modtranslator.backends.nllb import _compute_gpu_batch_size

        result = _compute_gpu_batch_size(0)
        assert result == 16

    def test_compute_gpu_batch_size_at_model_size(self):
        """_compute_gpu_batch_size with VRAM exactly at model size clamps to 16."""
        from modtranslator.backends.nllb import _NLLB_MODEL_VRAM_MB, _compute_gpu_batch_size

        result = _compute_gpu_batch_size(_NLLB_MODEL_VRAM_MB)
        assert result == 16


class TestSplitLongTextNewlineJoiner:
    """Tests for _split_long_text preserving newline separators."""

    def test_split_by_newlines_returns_newline_joiner(self, mock_nllb_env):
        """When text has no sentence boundaries and splits by \\n, joiner is \\n."""
        cls = mock_nllb_env.NLLBBackend
        tokenizer = mock_nllb_env.tokenizer

        # Each word = 1 token in our mock. MAX_TOKENS=900 for NLLB.
        # Build 2 lines that together exceed MAX_TOKENS.
        line1 = " ".join(f"word{i}" for i in range(500))
        line2 = " ".join(f"word{i}" for i in range(500, 1000))
        text = f"{line1}\n{line2}"

        segments, joiner = cls._split_long_text(text, tokenizer)
        assert joiner == "\n"
        assert len(segments) >= 2

    def test_split_by_sentences_returns_space_joiner(self, mock_nllb_env):
        """When text splits by sentence boundaries, joiner is space."""
        cls = mock_nllb_env.NLLBBackend
        tokenizer = mock_nllb_env.tokenizer

        line1 = " ".join(f"word{i}" for i in range(500))
        line2 = " ".join(f"word{i}" for i in range(500, 1000))
        text = f"{line1}. {line2}."

        segments, joiner = cls._split_long_text(text, tokenizer)
        assert joiner == " "
        assert len(segments) >= 2

    def test_no_split_needed_returns_original(self, mock_nllb_env):
        """Short text that doesn't need splitting returns as-is."""
        cls = mock_nllb_env.NLLBBackend
        tokenizer = mock_nllb_env.tokenizer

        text = "Short text here"
        segments, joiner = cls._split_long_text(text, tokenizer)
        assert segments == ["Short text here"]

    def test_end_to_end_newline_preserved(self, mock_nllb_env):
        """Full translate_batch preserves \\n when reassembling split segments."""
        from modtranslator.backends.nllb import CHAR_HEURISTIC_THRESHOLD

        backend = mock_nllb_env.NLLBBackend(
            device="cpu", models_dir=mock_nllb_env.models_dir
        )

        # Build text that exceeds CHAR_HEURISTIC_THRESHOLD and MAX_TOKENS
        # Each word = 1 token. We need >900 tokens and >2250 chars.
        line1 = " ".join(f"longword{i:04d}" for i in range(500))
        line2 = " ".join(f"longword{i:04d}" for i in range(500, 1000))
        long_text = f"{line1}\n{line2}"

        assert len(long_text) >= CHAR_HEURISTIC_THRESHOLD

        # Mock batch_decode to return distinguishable translations per segment
        call_count = [0]

        def _batch_decode(ids_list, **kw):
            results = []
            for _ in ids_list:
                call_count[0] += 1
                results.append(f"translated_segment_{call_count[0]}")
            return results

        mock_nllb_env.tokenizer.batch_decode.side_effect = _batch_decode

        result = backend.translate_batch([long_text], "ES")
        assert len(result) == 1
        # The segments should be joined with \n, not space
        assert "\n" in result[0]
