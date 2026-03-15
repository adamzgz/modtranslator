"""Tests for Minecraft lang file parser."""

from __future__ import annotations

import json

import pytest

from modtranslator.core.mc_lang_parser import (
    MC_LANG_MAP,
    detect_indent,
    dump_json_lang,
    dump_legacy_lang,
    filter_translatable,
    merge_translations,
    parse_json_lang,
    parse_legacy_lang,
)

# ── detect_indent ──


class TestDetectIndent:
    def test_two_spaces(self):
        raw = '{\n  "key": "value"\n}'
        assert detect_indent(raw) == "  "

    def test_four_spaces(self):
        raw = '{\n    "key": "value"\n}'
        assert detect_indent(raw) == "    "

    def test_tabs(self):
        raw = '{\n\t"key": "value"\n}'
        assert detect_indent(raw) == "\t"

    def test_minified(self):
        raw = '{"key":"value"}'
        assert detect_indent(raw) == ""

    def test_empty(self):
        assert detect_indent("{}") == ""


# ── parse_json_lang ──


class TestParseJsonLang:
    def test_basic(self):
        text = '{"item.sword": "Iron Sword", "item.bow": "Bow"}'
        result = parse_json_lang(text)
        assert result == {"item.sword": "Iron Sword", "item.bow": "Bow"}

    def test_skip_non_string(self):
        text = '{"key": "value", "list_key": [1, 2], "dict_key": {"a": 1}}'
        result = parse_json_lang(text)
        assert result == {"key": "value"}

    def test_empty(self):
        assert parse_json_lang("{}") == {}

    def test_not_dict(self):
        assert parse_json_lang("[]") == {}

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_lang("not json")


# ── parse_legacy_lang ──


class TestParseLegacyLang:
    def test_basic(self):
        text = "item.sword=Iron Sword\nitem.bow=Bow"
        result = parse_legacy_lang(text)
        assert result == {"item.sword": "Iron Sword", "item.bow": "Bow"}

    def test_comments(self):
        text = "# This is a comment\nitem.sword=Iron Sword"
        result = parse_legacy_lang(text)
        assert result == {"item.sword": "Iron Sword"}

    def test_empty_lines(self):
        text = "\n\nitem.sword=Iron Sword\n\n"
        result = parse_legacy_lang(text)
        assert result == {"item.sword": "Iron Sword"}

    def test_value_with_equals(self):
        text = "key=value=with=equals"
        result = parse_legacy_lang(text)
        assert result == {"key": "value=with=equals"}

    def test_empty(self):
        assert parse_legacy_lang("") == {}


# ── filter_translatable ──


class TestFilterTranslatable:
    def test_basic(self):
        entries = {"key1": "Hello", "key2": "World"}
        result = filter_translatable(entries)
        assert result == entries

    def test_skip_empty(self):
        entries = {"key1": "Hello", "key2": "", "key3": "  "}
        result = filter_translatable(entries)
        assert result == {"key1": "Hello"}

    def test_skip_non_alpha(self):
        entries = {"key1": "Hello", "key2": "123", "key3": "---"}
        result = filter_translatable(entries)
        assert result == {"key1": "Hello"}

    def test_skip_comment_keys(self):
        entries = {"_comment": "ignore", "comment_id": "1", "key1": "Hello"}
        result = filter_translatable(entries)
        assert result == {"key1": "Hello"}

    def test_skip_existing(self):
        entries = {"key1": "Hello", "key2": "World"}
        existing = {"key1": "Hola"}
        result = filter_translatable(entries, existing)
        assert result == {"key2": "World"}


# ── merge_translations ──


class TestMergeTranslations:
    def test_basic(self):
        en_us = {"key1": "Hello", "key2": "World"}
        new = {"key1": "Hola", "key2": "Mundo"}
        result = merge_translations(en_us, {}, new)
        assert result == {"key1": "Hola", "key2": "Mundo"}

    def test_existing_takes_precedence(self):
        en_us = {"key1": "Hello", "key2": "World"}
        existing = {"key1": "Hola existente"}
        new = {"key2": "Mundo"}
        result = merge_translations(en_us, existing, new)
        assert result == {"key1": "Hola existente", "key2": "Mundo"}

    def test_extra_keys_from_existing(self):
        en_us = {"key1": "Hello"}
        existing = {"key1": "Hola", "old_key": "Viejo"}
        result = merge_translations(en_us, existing, {})
        assert list(result.keys()) == ["key1", "old_key"]
        assert result["old_key"] == "Viejo"

    def test_en_us_order_preserved(self):
        en_us = {"c": "C", "a": "A", "b": "B"}
        result = merge_translations(en_us, {}, {"c": "Cc", "a": "Aa", "b": "Bb"})
        assert list(result.keys()) == ["c", "a", "b"]

    def test_fallback_to_en_us(self):
        en_us = {"key1": "Hello"}
        result = merge_translations(en_us, {}, {})
        assert result == {"key1": "Hello"}


# ── dump_json_lang ──


class TestDumpJsonLang:
    def test_two_spaces(self):
        entries = {"key": "value"}
        result = dump_json_lang(entries, "  ")
        parsed = json.loads(result)
        assert parsed == entries
        assert "\n  " in result

    def test_four_spaces(self):
        result = dump_json_lang({"key": "val"}, "    ")
        assert "\n    " in result

    def test_tabs(self):
        result = dump_json_lang({"key": "val"}, "\t")
        assert "\n\t" in result

    def test_minified(self):
        result = dump_json_lang({"key": "val"}, "")
        assert "\n" not in result


# ── dump_legacy_lang ──


class TestDumpLegacyLang:
    def test_basic(self):
        entries = {"key1": "Hello", "key2": "World"}
        result = dump_legacy_lang(entries)
        assert result == "key1=Hello\nkey2=World\n"


# ── MC_LANG_MAP ──


class TestLangMap:
    def test_all_languages_present(self):
        for lang in ("ES", "FR", "DE", "IT", "PT", "RU", "PL"):
            assert lang in MC_LANG_MAP
