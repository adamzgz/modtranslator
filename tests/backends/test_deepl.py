"""Tests for the DeepL backend (mocked)."""

import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_deepl():
    """Create a mock deepl module with required exception classes."""
    mock_mod = ModuleType("deepl")
    mock_mod.Translator = MagicMock()  # type: ignore[attr-defined]
    mock_mod.QuotaExceededException = type("QuotaExceededException", (Exception,), {})  # type: ignore[attr-defined]
    mock_mod.DeepLException = type("DeepLException", (Exception,), {})  # type: ignore[attr-defined]
    return mock_mod


class TestDeepLBackend:
    def test_translate_batch(self):
        mock_deepl = _make_mock_deepl()
        mock_translator = MagicMock()
        mock_deepl.Translator.return_value = mock_translator

        mock_result1 = MagicMock()
        mock_result1.text = "Hola"
        mock_result2 = MagicMock()
        mock_result2.text = "Mundo"
        mock_translator.translate_text.return_value = [mock_result1, mock_result2]

        with patch.dict("sys.modules", {"deepl": mock_deepl}):
            from modtranslator.backends.deepl import DeepLBackend
            backend = DeepLBackend(api_key="fake-key")
            results = backend.translate_batch(["Hello", "World"], "ES")

        assert results == ["Hola", "Mundo"]
        mock_translator.translate_text.assert_called_once()

    def test_empty_batch(self):
        mock_deepl = _make_mock_deepl()
        mock_translator = MagicMock()
        mock_deepl.Translator.return_value = mock_translator

        with patch.dict("sys.modules", {"deepl": mock_deepl}):
            from modtranslator.backends.deepl import DeepLBackend
            backend = DeepLBackend(api_key="fake-key")
            results = backend.translate_batch([], "ES")

        assert results == []
        mock_translator.translate_text.assert_not_called()

    def test_large_batch_is_chunked(self):
        mock_deepl = _make_mock_deepl()
        mock_translator = MagicMock()
        mock_deepl.Translator.return_value = mock_translator

        def make_result(texts, **kwargs):
            return [MagicMock(text=f"translated_{t}") for t in texts]

        mock_translator.translate_text.side_effect = make_result

        with patch.dict("sys.modules", {"deepl": mock_deepl}):
            from modtranslator.backends.deepl import DeepLBackend
            backend = DeepLBackend(api_key="fake-key")

            texts = [f"text_{i}" for i in range(75)]
            results = backend.translate_batch(texts, "ES")

        assert len(results) == 75
        assert mock_translator.translate_text.call_count == 2


class TestDeepLErrorCases:
    def test_quota_exceeded_propagates_immediately(self):
        """QuotaExceededException is re-raised immediately without retry."""
        mock_deepl = _make_mock_deepl()
        mock_translator = MagicMock()
        mock_deepl.Translator.return_value = mock_translator
        mock_translator.translate_text.side_effect = mock_deepl.QuotaExceededException(
            "quota exceeded"
        )

        with patch.dict("sys.modules", {"deepl": mock_deepl}):
            from modtranslator.backends.deepl import DeepLBackend
            backend = DeepLBackend(api_key="fake-key")
            with pytest.raises(Exception, match="quota exceeded"):
                backend.translate_batch(["Hello"], "ES")

        # Should have been called only once (no retries)
        assert mock_translator.translate_text.call_count == 1

    def test_retry_exhaustion_raises(self):
        """After MAX_RETRIES, the last exception is raised."""
        mock_deepl = _make_mock_deepl()
        mock_translator = MagicMock()
        mock_deepl.Translator.return_value = mock_translator
        mock_translator.translate_text.side_effect = mock_deepl.DeepLException("server error")

        with (
            patch.dict("sys.modules", {"deepl": mock_deepl}),
            patch.object(time, "sleep"),
        ):
            from modtranslator.backends.deepl import DeepLBackend, MAX_RETRIES  # noqa: I001
            backend = DeepLBackend(api_key="fake-key")
            with pytest.raises(Exception, match="server error"):
                backend.translate_batch(["Hello"], "ES")

        assert mock_translator.translate_text.call_count == MAX_RETRIES

    def test_import_error_without_deepl_package(self):
        """ImportError when deepl package is not installed."""
        import importlib

        with (
            patch.dict("sys.modules", {"deepl": None}),
            pytest.raises(ImportError, match="deepl"),
        ):
            from modtranslator.backends import deepl as deepl_mod
            importlib.reload(deepl_mod)
            deepl_mod.DeepLBackend(api_key="fake-key")
