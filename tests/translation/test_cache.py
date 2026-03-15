"""Tests for the translation cache."""

import sqlite3

import pytest

from modtranslator.translation.cache import TranslationCache


class TestTranslationCache:
    def test_put_and_get(self, tmp_cache):
        tmp_cache.put("00001000:FULL:0", "ES", "Iron Sword", "Espada de Hierro")
        result = tmp_cache.get("00001000:FULL:0", "ES")
        assert result == "Espada de Hierro"

    def test_get_missing_returns_none(self, tmp_cache):
        assert tmp_cache.get("nonexistent", "ES") is None

    def test_different_languages_independent(self, tmp_cache):
        tmp_cache.put("key1", "ES", "Hello", "Hola")
        tmp_cache.put("key1", "FR", "Hello", "Bonjour")

        assert tmp_cache.get("key1", "ES") == "Hola"
        assert tmp_cache.get("key1", "FR") == "Bonjour"

    def test_get_batch(self, tmp_cache):
        tmp_cache.put("k1", "ES", "One", "Uno")
        tmp_cache.put("k2", "ES", "Two", "Dos")
        tmp_cache.put("k3", "FR", "Three", "Trois")

        result = tmp_cache.get_batch(["k1", "k2", "k3", "k4"], "ES")
        assert result == {"k1": "Uno", "k2": "Dos"}

    def test_put_batch(self, tmp_cache):
        entries = [
            ("k1", "ES", "One", "Uno"),
            ("k2", "ES", "Two", "Dos"),
        ]
        tmp_cache.put_batch(entries)

        assert tmp_cache.get("k1", "ES") == "Uno"
        assert tmp_cache.get("k2", "ES") == "Dos"

    def test_count(self, tmp_cache):
        assert tmp_cache.count() == 0
        tmp_cache.put("k1", "ES", "a", "b")
        assert tmp_cache.count() == 1

    def test_clear(self, tmp_cache):
        tmp_cache.put("k1", "ES", "a", "b")
        tmp_cache.put("k2", "ES", "c", "d")
        deleted = tmp_cache.clear()
        assert deleted == 2
        assert tmp_cache.count() == 0

    def test_upsert_on_duplicate(self, tmp_cache):
        tmp_cache.put("k1", "ES", "Hello", "Hola")
        tmp_cache.put("k1", "ES", "Hello", "Hola Updated")
        assert tmp_cache.get("k1", "ES") == "Hola Updated"
        assert tmp_cache.count() == 1


class TestCacheEdgeCases:
    def test_get_batch_large_chunking(self, tmp_cache):
        """get_batch with >900 keys uses chunking correctly."""
        # Insert 1000 entries
        entries = [(f"k{i}", "ES", f"orig{i}", f"trans{i}") for i in range(1000)]
        tmp_cache.put_batch(entries)

        keys = [f"k{i}" for i in range(1000)]
        result = tmp_cache.get_batch(keys, "ES")
        assert len(result) == 1000
        assert result["k0"] == "trans0"
        assert result["k999"] == "trans999"

    def test_get_batch_empty_keys(self, tmp_cache):
        """get_batch with empty keys returns empty dict."""
        result = tmp_cache.get_batch([], "ES")
        assert result == {}

    def test_put_batch_empty(self, tmp_cache):
        """put_batch with empty entries is a no-op."""
        tmp_cache.put_batch([])
        assert tmp_cache.count() == 0

    def test_corrupt_database(self, tmp_path):
        """Opening a corrupt database file raises an error."""
        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        with pytest.raises(sqlite3.DatabaseError):
            cache = TranslationCache(db_path=db_path)
            # Force a query to trigger the error
            cache.get("key", "ES")
