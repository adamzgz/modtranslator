"""Tests for reporting formatters (JSON, Markdown, CSV, save_report)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from modtranslator.reporting.formatters import save_report, to_csv, to_json, to_markdown
from modtranslator.reporting.report import TranslationReport


def _make_report(**overrides) -> TranslationReport:
    """Build a TranslationReport with sensible defaults."""
    now = datetime(2025, 1, 1, 12, 0, 0)
    defaults = dict(
        source_file="test.esp",
        output_file="test_ES.esp",
        target_lang="ES",
        game_detected="Fallout 3",
        backend="dummy",
        total_strings_found=100,
        strings_from_cache=10,
        strings_translated=80,
        strings_patched=80,
        strings_failed=0,
        dry_run=False,
        started_at=now,
        finished_at=now + timedelta(seconds=12.5),
        errors=[],
    )
    defaults.update(overrides)
    return TranslationReport(**defaults)


class TestToJson:
    def test_basic_json(self):
        report = _make_report()
        result = to_json(report)
        data = json.loads(result)
        assert data["source_file"] == "test.esp"
        assert data["strings_translated"] == 80
        assert data["duration_seconds"] == 12.5

    def test_json_with_errors(self):
        report = _make_report(errors=["Error 1", "Error 2"])
        data = json.loads(to_json(report))
        assert data["errors"] == ["Error 1", "Error 2"]

    def test_json_custom_indent(self):
        report = _make_report()
        result = to_json(report, indent=4)
        assert "    " in result


class TestToMarkdown:
    def test_basic_markdown(self):
        report = _make_report()
        md = to_markdown(report)
        assert "# Translation Report" in md
        assert "test.esp" in md
        assert "12.5s" in md
        assert "## Glossary" not in md
        assert "## Errors" not in md

    def test_markdown_with_glossary(self):
        report = _make_report(glossary_file="glossary.toml", glossary_terms=42)
        md = to_markdown(report)
        assert "## Glossary" in md
        assert "glossary.toml" in md
        assert "42" in md

    def test_markdown_with_errors(self):
        report = _make_report(errors=["Bad record", "Missing data"])
        md = to_markdown(report)
        assert "## Errors" in md
        assert "- Bad record" in md
        assert "- Missing data" in md

    def test_markdown_ends_with_newline(self):
        report = _make_report()
        assert to_markdown(report).endswith("\n")


class TestToCsv:
    def test_basic_csv(self):
        report = _make_report()
        csv_str = to_csv(report)
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "source_file" in lines[0]
        assert "test.esp" in lines[1]

    def test_csv_flattens_errors(self):
        report = _make_report(errors=["Err A", "Err B"])
        csv_str = to_csv(report)
        assert "Err A; Err B" in csv_str


class TestSaveReport:
    def test_save_json(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.json"
        save_report(report, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["source_file"] == "test.esp"

    def test_save_markdown(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.md"
        save_report(report, path)
        content = path.read_text(encoding="utf-8")
        assert "# Translation Report" in content

    def test_save_markdown_extension(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.markdown"
        save_report(report, path)
        content = path.read_text(encoding="utf-8")
        assert "# Translation Report" in content

    def test_save_csv(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.csv"
        save_report(report, path)
        content = path.read_text(encoding="utf-8")
        assert "source_file" in content

    def test_save_unknown_extension_defaults_to_json(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.txt"
        save_report(report, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["source_file"] == "test.esp"

    def test_save_with_path_string(self, tmp_path):
        report = _make_report()
        path = str(tmp_path / "report.json")
        save_report(report, path)
