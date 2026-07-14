"""Tests for the FTB Quests lang (SNBT subset) parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from modtranslator.core.ftbq_lang import (
    SnbtError,
    dump_snbt_lang,
    find_lang_files,
    parse_snbt_lang,
    target_path,
)

# A realistic ATM10-style sample: simple strings, colour codes with escaped
# quotes, a multiline list of paragraphs, a chapter.* key, and a \n inside a
# string.
SAMPLE = """{
    quest.04833C4BB0CCCBB7.quest_desc: [
        "Same as Stone Weapons just the &7Sword &r has extra reach"
        "Second paragraph with a \\"quote\\" inside"
    ]
    quest.04833C4BB0CCCBB7.title: "&7Holystone&r Weapons"
    chapter.79E723AC0392F166.title: "The Aether"
    quest.ABC123.subtitle: "Line one\\nLine two"
    quest.ABC123.literal: "Keep &7 and \\\\n and //n literal"
}
"""


class TestParseSnbtLang:
    def test_basic_string_and_list(self):
        data = parse_snbt_lang(SAMPLE)
        assert data["quest.04833C4BB0CCCBB7.title"] == "&7Holystone&r Weapons"
        assert data["chapter.79E723AC0392F166.title"] == "The Aether"
        assert data["quest.04833C4BB0CCCBB7.quest_desc"] == [
            "Same as Stone Weapons just the &7Sword &r has extra reach",
            'Second paragraph with a "quote" inside',
        ]

    def test_newline_escape_decoded(self):
        data = parse_snbt_lang(SAMPLE)
        assert data["quest.ABC123.subtitle"] == "Line one\nLine two"

    def test_literal_sequences_preserved(self):
        data = parse_snbt_lang(SAMPLE)
        # \\n in the source is a literal backslash + n; //n and &7 are plain text.
        assert data["quest.ABC123.literal"] == "Keep &7 and \\n and //n literal"

    def test_order_preserved(self):
        data = parse_snbt_lang(SAMPLE)
        assert list(data.keys()) == [
            "quest.04833C4BB0CCCBB7.quest_desc",
            "quest.04833C4BB0CCCBB7.title",
            "chapter.79E723AC0392F166.title",
            "quest.ABC123.subtitle",
            "quest.ABC123.literal",
        ]

    def test_quoted_key(self):
        data = parse_snbt_lang('{"quest.with space.title": "Hi"}')
        assert data == {"quest.with space.title": "Hi"}

    def test_unquoted_key(self):
        data = parse_snbt_lang("{ chapter.ABC.title: \"Hi\" }")
        assert data == {"chapter.ABC.title": "Hi"}

    def test_unicode_values(self):
        text = '{title: "áñ 日本語", desc: ["línea uno", "ふたつ"]}'
        data = parse_snbt_lang(text)
        assert data["title"] == "áñ 日本語"
        assert data["desc"] == ["línea uno", "ふたつ"]

    def test_empty_compound(self):
        assert parse_snbt_lang("{}") == {}

    def test_skips_unsupported_value_types(self):
        # Nested compound and a bare number must be skipped, not crash.
        text = "{ a: \"keep\", nested: {x: \"y\"}, num: 42b, b: \"also\" }"
        data = parse_snbt_lang(text)
        assert data == {"a": "keep", "b": "also"}

    def test_inline_and_multiline_list_equivalent(self):
        inline = parse_snbt_lang('{k: ["a", "b"]}')
        multiline = parse_snbt_lang('{k: [\n    "a"\n    "b"\n]}')
        assert inline == multiline == {"k": ["a", "b"]}


class TestParseErrors:
    def test_missing_opening_brace(self):
        with pytest.raises(SnbtError):
            parse_snbt_lang('k: "v"')

    def test_unterminated_string(self):
        with pytest.raises(SnbtError):
            parse_snbt_lang('{k: "no end}')

    def test_unterminated_compound(self):
        with pytest.raises(SnbtError):
            parse_snbt_lang('{k: "v"')

    def test_missing_colon(self):
        with pytest.raises(SnbtError):
            parse_snbt_lang('{k "v"}')

    def test_trailing_content(self):
        with pytest.raises(SnbtError):
            parse_snbt_lang('{k: "v"} garbage')


class TestDumpSnbtLang:
    def test_style(self):
        data: dict[str, str | list[str]] = {
            "chapter.ABC.title": "Hello",
            "quest.X.desc": ["one", "two"],
        }
        out = dump_snbt_lang(data)
        assert out == (
            "{\n"
            '    chapter.ABC.title: "Hello"\n'
            "    quest.X.desc: [\n"
            '        "one"\n'
            '        "two"\n'
            "    ]\n"
            "}\n"
        )

    def test_quoted_key_when_needed(self):
        out = dump_snbt_lang({"has space": "v"})
        assert '    "has space": "v"' in out

    def test_empty_list_and_empty_string(self):
        out = dump_snbt_lang({"empty_list": [], "empty_str": ""})
        assert "    empty_list: []" in out
        assert '    empty_str: ""' in out

    def test_escapes_encoded(self):
        out = dump_snbt_lang({"k": 'a "q" \\ n\nnext'})
        assert '    k: "a \\"q\\" \\\\ n\\nnext"' in out


class TestRoundTrip:
    def test_sample_round_trip(self):
        parsed = parse_snbt_lang(SAMPLE)
        assert parse_snbt_lang(dump_snbt_lang(parsed)) == parsed

    def test_unicode_round_trip(self):
        parsed = parse_snbt_lang('{title: "áñ 日本語", desc: ["línea", "ふたつ"]}')
        assert parse_snbt_lang(dump_snbt_lang(parsed)) == parsed

    def test_empty_values_round_trip(self):
        data: dict[str, str | list[str]] = {"a": "", "b": [], "c": ["x"]}
        assert parse_snbt_lang(dump_snbt_lang(data)) == data

    def test_quoted_key_round_trip(self):
        data: dict[str, str | list[str]] = {"has space": "v", "normal.key": "w"}
        assert parse_snbt_lang(dump_snbt_lang(data)) == data


class TestFindLangFiles:
    def _build_tree(self, root: Path) -> None:
        base = root / "lang" / "en_us"
        (base / "chapters").mkdir(parents=True)
        (base / "reward_tables").mkdir(parents=True)
        (base / "file.snbt").write_text("{}", encoding="utf-8")
        (base / "chapter_group.snbt").write_text("{}", encoding="utf-8")
        (base / "chapters" / "a.snbt").write_text("{}", encoding="utf-8")
        (base / "chapters" / "b.snbt").write_text("{}", encoding="utf-8")
        (base / "reward_tables" / "r.snbt").write_text("{}", encoding="utf-8")
        (base / "notes.txt").write_text("ignore", encoding="utf-8")

    def test_finds_all_snbt(self, tmp_path: Path):
        self._build_tree(tmp_path)
        files = find_lang_files(tmp_path)
        names = {p.name for p in files}
        assert names == {"file.snbt", "chapter_group.snbt", "a.snbt", "b.snbt", "r.snbt"}

    def test_returns_sorted(self, tmp_path: Path):
        self._build_tree(tmp_path)
        files = find_lang_files(tmp_path)
        assert files == sorted(files)

    def test_missing_dir(self, tmp_path: Path):
        assert find_lang_files(tmp_path) == []

    def test_custom_locale(self, tmp_path: Path):
        base = tmp_path / "lang" / "fr_fr"
        base.mkdir(parents=True)
        (base / "x.snbt").write_text("{}", encoding="utf-8")
        assert find_lang_files(tmp_path, source_locale="fr_fr") == [base / "x.snbt"]


class TestTargetPath:
    def test_simple(self):
        src = Path("config/ftbquests/quests/lang/en_us/chapters/x.snbt")
        out = target_path(src, "en_us", "es_es")
        assert out == Path("config/ftbquests/quests/lang/es_es/chapters/x.snbt")

    def test_top_level_file(self):
        src = Path("lang/en_us/file.snbt")
        out = target_path(src, "en_us", "fr_fr")
        assert out == Path("lang/fr_fr/file.snbt")

    def test_absolute_path(self, tmp_path: Path):
        src = tmp_path / "lang" / "en_us" / "chapters" / "a.snbt"
        out = target_path(src, "en_us", "es_es")
        assert out == tmp_path / "lang" / "es_es" / "chapters" / "a.snbt"

    def test_locale_not_present(self):
        with pytest.raises(ValueError):
            target_path(Path("lang/de_de/x.snbt"), "en_us", "es_es")
