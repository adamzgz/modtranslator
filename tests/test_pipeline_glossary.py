"""Tests for resolve_glossary_paths() multi-language support."""

from __future__ import annotations

from pathlib import Path

import pytest

from modtranslator.core.constants import Game
from modtranslator.pipeline import GameChoice, resolve_glossary_paths

# Real glossaries dir (relative to this file: tests/ → project root → glossaries/)
_GLOSSARIES_DIR = Path(__file__).resolve().parent.parent / "glossaries"


class TestResolveGlossaryPathsES:
    """Existing Spanish behavior is unchanged."""

    def test_fo3_es_returns_existing_glossaries(self):
        paths = resolve_glossary_paths(None, "ES", GameChoice.fo3, Game.FALLOUT3)
        assert len(paths) > 0
        assert all(p.exists() for p in paths)
        assert any("fallout_base_es" in p.name for p in paths)

    def test_skyrim_es_returns_existing_glossary(self):
        paths = resolve_glossary_paths(None, "ES", GameChoice.skyrim, Game.SKYRIM)
        assert len(paths) > 0
        assert any("skyrim_base_es" in p.name for p in paths)

    def test_fo4_es_returns_existing_glossaries(self):
        paths = resolve_glossary_paths(None, "ES", GameChoice.fo4, Game.FALLOUT4)
        assert len(paths) > 0
        assert any("fallout_base_es" in p.name for p in paths)


class TestResolveGlossaryPathsMultilang:
    """Non-ES languages look for *_{lang}.toml; return empty if not present."""

    def test_fr_fo3_looks_for_fr_toml(self):
        """For FR, candidates use _fr.toml suffix; empty if not yet created."""
        paths = resolve_glossary_paths(None, "FR", GameChoice.fo3, Game.FALLOUT3)
        # If FR glossaries don't exist yet, result is empty (not an error)
        assert isinstance(paths, list)
        assert all(p.exists() for p in paths)
        # If any path returned, it must have _fr suffix
        for p in paths:
            assert "_fr" in p.name

    def test_de_fo3_returns_de_glossaries(self):
        """DE glossaries exist — returned paths have _de suffix."""
        paths = resolve_glossary_paths(None, "DE", GameChoice.fo3, Game.FALLOUT3)
        assert isinstance(paths, list)
        assert all(p.exists() for p in paths)
        for p in paths:
            assert "_de" in p.name

    def test_ru_skyrim_returns_ru_glossary(self):
        paths = resolve_glossary_paths(None, "RU", GameChoice.skyrim, Game.SKYRIM)
        assert isinstance(paths, list)
        assert all(p.exists() for p in paths)
        for p in paths:
            assert "_ru" in p.name

    def test_pl_fnv_returns_pl_glossaries(self):
        paths = resolve_glossary_paths(None, "PL", GameChoice.fnv, Game.FALLOUT3)
        assert isinstance(paths, list)
        assert all(p.exists() for p in paths)
        for p in paths:
            assert "_pl" in p.name

    def test_it_fo4_returns_it_glossaries(self):
        paths = resolve_glossary_paths(None, "IT", GameChoice.fo4, Game.FALLOUT4)
        assert isinstance(paths, list)
        assert all(p.exists() for p in paths)
        for p in paths:
            assert "_it" in p.name

    def test_explicit_glossary_bypasses_lang(self, tmp_path):
        """Explicit --glossary always wins regardless of lang."""
        custom = tmp_path / "custom.toml"
        custom.write_text("")
        paths = resolve_glossary_paths(custom, "RU", GameChoice.auto, Game.FALLOUT3)
        assert paths == [custom]

    def test_nonexistent_explicit_glossary_returns_empty(self, tmp_path):
        custom = tmp_path / "nonexistent.toml"
        paths = resolve_glossary_paths(custom, "ES", GameChoice.auto, Game.FALLOUT3)
        assert paths == []
