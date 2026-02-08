"""Tests for the DeepL backend (mocked)."""

from types import ModuleType
from unittest.mock import MagicMock, patch


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
