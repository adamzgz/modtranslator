"""Tests for GUI model manager — dynamic Opus-MT model resolution per language."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

from modtranslator.gui.model_manager import (
    _opus_display_name,
    _opus_model_id,
    check_backend_ready,
    get_missing_model_ids,
    get_model_status,
)


def _ensure_fake_ct2() -> None:
    """Ensure ctranslate2 and sentencepiece are importable (stub if missing)."""
    for mod_name in ("ctranslate2", "sentencepiece"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

# ---------------------------------------------------------------------------
# _opus_model_id
# ---------------------------------------------------------------------------

class TestOpusModelId:
    """Test dynamic Opus-MT model ID resolution."""

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_tc_big_languages_prefer_tc_big(self, mock_exists: object) -> None:
        """ES/FR/IT/PT should default to tc-big when nothing is downloaded."""
        for lang in ("ES", "FR", "IT", "PT"):
            model_id = _opus_model_id(lang)
            assert model_id is not None
            assert "tc-big" in model_id
            assert model_id.endswith(f"-en-{lang.lower()}")

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_base_only_languages(self, mock_exists: object) -> None:
        """DE/RU only have base models (PL has no Opus-MT on HuggingFace)."""
        for lang in ("DE", "RU"):
            model_id = _opus_model_id(lang)
            assert model_id is not None
            assert "tc-big" not in model_id
            assert model_id == f"Helsinki-NLP/opus-mt-en-{lang.lower()}"

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_unknown_language_returns_none(self, mock_exists: object) -> None:
        """Language not in _OPUS_VARIANTS should return None."""
        assert _opus_model_id("JA") is None
        assert _opus_model_id("ZH") is None

    def test_prefers_downloaded_tc_big(self) -> None:
        """If tc-big is downloaded, return it even for langs that also have base."""
        def fake_exists(model_id: str) -> bool:
            return "tc-big" in model_id

        with patch("modtranslator.gui.model_manager._check_model_exists", side_effect=fake_exists):
            model_id = _opus_model_id("ES")
            assert model_id is not None
            assert "tc-big" in model_id

    def test_falls_back_to_base_if_tc_big_not_downloaded(self) -> None:
        """If tc-big is NOT downloaded but base IS, return base."""
        def fake_exists(model_id: str) -> bool:
            return "tc-big" not in model_id and "opus-mt-en-" in model_id

        with patch("modtranslator.gui.model_manager._check_model_exists", side_effect=fake_exists):
            model_id = _opus_model_id("FR")
            assert model_id is not None
            assert "tc-big" not in model_id
            assert model_id == "Helsinki-NLP/opus-mt-en-fr"

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_case_insensitive(self, mock_exists: object) -> None:
        """Should work with lowercase input too."""
        assert _opus_model_id("es") is not None
        assert _opus_model_id("Es") is not None


# ---------------------------------------------------------------------------
# _opus_display_name
# ---------------------------------------------------------------------------

class TestOpusDisplayName:
    def test_tc_big_name(self) -> None:
        name = _opus_display_name("Helsinki-NLP/opus-mt-tc-big-en-es")
        assert name == "Opus-MT tc-big-en-es"

    def test_base_name(self) -> None:
        name = _opus_display_name("Helsinki-NLP/opus-mt-en-de")
        assert name == "Opus-MT en-de"


# ---------------------------------------------------------------------------
# get_model_status
# ---------------------------------------------------------------------------

class TestGetModelStatus:
    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_includes_opus_for_supported_lang(self, mock_exists: object) -> None:
        """Model list should include Opus-MT for supported languages."""
        models = get_model_status(lang="FR")
        names = [m.name for m in models]
        assert any("Opus-MT" in n for n in names)
        assert any("NLLB" in n for n in names)
        # Opus-MT model should be FR, not ES
        opus = [m for m in models if "Opus-MT" in m.name][0]
        assert "en-fr" in opus.description

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_no_opus_for_unsupported_lang(self, mock_exists: object) -> None:
        """Languages without Opus-MT should only show NLLB."""
        models = get_model_status(lang="JA")
        names = [m.name for m in models]
        assert not any("Opus-MT" in n for n in names)
        assert any("NLLB" in n for n in names)

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_each_lang_gets_correct_model(self, mock_exists: object) -> None:
        """Every supported language should get its own Opus-MT model ID."""
        for lang in ("ES", "FR", "DE", "IT", "PT", "RU"):
            models = get_model_status(lang=lang)
            opus = [m for m in models if "Opus-MT" in m.name]
            assert len(opus) == 1, f"Expected 1 Opus-MT model for {lang}"
            assert f"en-{lang.lower()}" in opus[0].description


# ---------------------------------------------------------------------------
# check_backend_ready
# ---------------------------------------------------------------------------

class TestCheckBackendReady:
    def test_dummy_always_ready(self) -> None:
        ready, msg = check_backend_ready("dummy", lang="FR")
        assert ready is True

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=True)
    def test_hybrid_ready_with_correct_lang(self, mock_exists: object) -> None:
        _ensure_fake_ct2()
        ready, msg = check_backend_ready("hybrid", lang="DE")
        assert ready is True
        assert msg == "Ready"

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_hybrid_not_ready_when_models_missing(self, mock_exists: object) -> None:
        _ensure_fake_ct2()
        ready, msg = check_backend_ready("hybrid", lang="IT")
        assert ready is False
        assert "not downloaded" in msg.lower()

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_opus_no_model_for_unsupported_lang(self, mock_exists: object) -> None:
        """Opus-MT for an unsupported language should fail gracefully."""
        _ensure_fake_ct2()
        ready, msg = check_backend_ready("opus-mt", lang="JA")
        assert ready is False
        assert "JA" in msg

    def test_unknown_backend(self) -> None:
        ready, msg = check_backend_ready("nonexistent", lang="ES")
        assert ready is False
        assert "unknown" in msg.lower()


# ---------------------------------------------------------------------------
# get_missing_model_ids
# ---------------------------------------------------------------------------

class TestGetMissingModelIds:
    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=True)
    def test_nothing_missing_when_all_downloaded(self, mock_exists: object) -> None:
        for lang in ("ES", "FR", "DE"):
            missing = get_missing_model_ids("hybrid", lang=lang)
            assert missing == []

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_both_missing_for_hybrid(self, mock_exists: object) -> None:
        missing = get_missing_model_ids("hybrid", lang="FR")
        assert len(missing) == 2
        names = [m[0] for m in missing]
        assert any("Opus-MT" in n for n in names)
        assert any("NLLB" in n for n in names)
        # Opus entry should be FR
        opus = [m for m in missing if "Opus-MT" in m[0]][0]
        assert "en-fr" in opus[1]

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_opus_missing_correct_model_per_lang(self, mock_exists: object) -> None:
        """Each lang should report its own Opus-MT model as missing."""
        for lang, expected_suffix in [("ES", "tc-big-en-es"), ("DE", "en-de"), ("RU", "en-ru")]:
            missing = get_missing_model_ids("opus-mt", lang=lang)
            assert len(missing) == 1
            assert expected_suffix in missing[0][1]

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_nllb_backend_ignores_lang(self, mock_exists: object) -> None:
        """NLLB backend is language-agnostic (single multilingual model)."""
        missing_es = get_missing_model_ids("nllb", lang="ES")
        missing_fr = get_missing_model_ids("nllb", lang="FR")
        assert len(missing_es) == 1
        assert len(missing_fr) == 1
        assert missing_es[0][1] == missing_fr[0][1]

    @patch("modtranslator.gui.model_manager._check_model_exists", return_value=False)
    def test_unsupported_lang_no_opus_in_hybrid(self, mock_exists: object) -> None:
        """Hybrid for unsupported Opus lang should only list NLLB as missing."""
        missing = get_missing_model_ids("hybrid", lang="JA")
        assert len(missing) == 1
        assert "NLLB" in missing[0][0]
