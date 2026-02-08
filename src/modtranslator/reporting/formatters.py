"""Output formatters for translation reports."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from modtranslator.reporting.report import TranslationReport


def to_json(report: TranslationReport, indent: int = 2) -> str:
    """Format report as JSON string."""
    return json.dumps(report.to_dict(), indent=indent, default=str)


def to_markdown(report: TranslationReport) -> str:
    """Format report as Markdown."""
    lines = [
        "# Translation Report",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Source | `{report.source_file}` |",
        f"| Output | `{report.output_file}` |",
        f"| Target language | {report.target_lang} |",
        f"| Game detected | {report.game_detected} |",
        f"| Backend | {report.backend} |",
        f"| Dry run | {report.dry_run} |",
        "",
        "## Statistics",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Strings found | {report.total_strings_found} |",
        f"| From cache | {report.strings_from_cache} |",
        f"| Translated | {report.strings_translated} |",
        f"| Patched | {report.strings_patched} |",
        f"| Failed | {report.strings_failed} |",
        f"| Duration | {report.duration_seconds:.1f}s |",
    ]

    if report.glossary_file:
        lines.extend([
            "",
            "## Glossary",
            "",
            f"- File: `{report.glossary_file}`",
            f"- Terms: {report.glossary_terms}",
        ])

    if report.errors:
        lines.extend([
            "",
            "## Errors",
            "",
        ])
        for err in report.errors:
            lines.append(f"- {err}")

    return "\n".join(lines) + "\n"


def to_csv(report: TranslationReport) -> str:
    """Format report as a single-row CSV."""
    output = io.StringIO()
    data = report.to_dict()
    # Flatten errors list
    data["errors"] = "; ".join(data["errors"])
    writer = csv.DictWriter(output, fieldnames=data.keys())
    writer.writeheader()
    writer.writerow(data)
    return output.getvalue()


def save_report(report: TranslationReport, path: str | Path) -> None:
    """Save report to file, auto-detecting format from extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        content = to_json(report)
    elif suffix in (".md", ".markdown"):
        content = to_markdown(report)
    elif suffix == ".csv":
        content = to_csv(report)
    else:
        content = to_json(report)

    path.write_text(content, encoding="utf-8")
