"""Tests for the dummy translation backend."""

from modtranslator.backends.dummy import DummyBackend


class TestDummyBackend:
    def test_translate_single(self):
        backend = DummyBackend()
        result = backend.translate("Hello", "ES")
        assert result == "[ES] Hello"

    def test_translate_batch(self):
        backend = DummyBackend()
        results = backend.translate_batch(["Hello", "World"], "ES")
        assert results == ["[ES] Hello", "[ES] World"]

    def test_translate_empty_batch(self):
        backend = DummyBackend()
        assert backend.translate_batch([], "ES") == []

    def test_language_tag_uppercase(self):
        backend = DummyBackend()
        result = backend.translate("Test", "fr")
        assert result == "[FR] Test"
