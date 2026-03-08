"""Minecraft lang file parser: JSON (.json) and legacy (.lang) formats.

Handles:
- JSON lang files (Minecraft 1.13+): flat {"key": "value"} dicts
- Legacy .lang files (pre-1.13): key=value lines with # comments
- Indentation detection and preservation for JSON output
- BOM handling (utf-8-sig)
- Skips non-string values (lists, dicts), empty values, comment keys
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Keys to skip (comments, metadata)
_SKIP_KEY_PREFIXES = ("_comment", "comment_id")


@dataclass
class LangFile:
    """Parsed Minecraft lang file."""

    path: str
    mod_id: str
    locale: str
    format: str  # "json" or "lang"
    entries: dict[str, str] = field(default_factory=dict)
    indent: str = "  "  # detected indentation style


def detect_indent(raw: str) -> str:
    """Detect indentation style from JSON text.

    Returns the indent string: "\\t", "  ", "    ", or "" (minified).
    """
    for line in raw.split("\n")[1:20]:  # skip first line (opening brace)
        stripped = line.lstrip()
        if not stripped or stripped.startswith("}"):
            continue
        leading = line[: len(line) - len(stripped)]
        if leading.startswith("\t"):
            return "\t"
        if leading.startswith("    "):
            return "    "
        if leading.startswith("  "):
            return "  "
    return ""  # minified


def parse_json_lang(text: str) -> dict[str, str]:
    """Parse a JSON lang file, returning only string values.

    Uses strict=False to tolerate control characters (tabs, newlines)
    inside string values — common in community-made translations.
    """
    data = json.loads(text, strict=False)
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def parse_legacy_lang(text: str) -> dict[str, str]:
    """Parse a legacy .lang file (key=value per line, # comments)."""
    entries: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eq = line.find("=")
        if eq < 0:
            continue
        key = line[:eq]
        value = line[eq + 1 :]
        entries[key] = value
    return entries


def filter_translatable(
    entries: dict[str, str],
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    """Filter entries to only translatable ones.

    - Skips keys starting with _comment or comment_id
    - Skips empty values or values without alphabetic characters
    - If existing is provided, skips keys already in existing
    """
    result: dict[str, str] = {}
    for key, value in entries.items():
        # Skip comment keys
        if any(key.startswith(prefix) for prefix in _SKIP_KEY_PREFIXES):
            continue
        # Skip empty or non-alpha values
        if not value or not value.strip():
            continue
        if not any(c.isalpha() for c in value):
            continue
        # Skip keys already translated
        if existing is not None and key in existing:
            continue
        result[key] = value
    return result


def merge_translations(
    en_us: dict[str, str],
    existing_target: dict[str, str],
    new_translations: dict[str, str],
) -> dict[str, str]:
    """Merge translations following en_us key order.

    - Keys from en_us come first (in en_us order)
    - For each key: use existing_target if present, else new_translations, else en_us
    - Extra keys in existing_target (not in en_us) go at the end
    """
    result: dict[str, str] = {}

    # Keys in en_us order
    for key in en_us:
        if key in existing_target:
            result[key] = existing_target[key]
        elif key in new_translations:
            result[key] = new_translations[key]
        else:
            result[key] = en_us[key]

    # Extra keys from existing_target
    for key, value in existing_target.items():
        if key not in result:
            result[key] = value

    return result


def dump_json_lang(entries: dict[str, str], indent: str = "  ") -> str:
    """Serialize a lang dict to JSON string preserving indent style."""
    if not indent:
        # Minified
        return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    # json.dumps indent parameter: int for spaces, or string not supported directly
    # Use custom approach for tabs
    if indent == "\t":
        raw = json.dumps(entries, ensure_ascii=False, indent=2)
        # Replace leading spaces with tabs (2 spaces → 1 tab)
        lines = raw.split("\n")
        result_lines: list[str] = []
        for line in lines:
            stripped = line.lstrip(" ")
            n_spaces = len(line) - len(stripped)
            n_tabs = n_spaces // 2
            result_lines.append("\t" * n_tabs + stripped)
        return "\n".join(result_lines)
    # Standard space indent
    n_spaces = len(indent)
    return json.dumps(entries, ensure_ascii=False, indent=n_spaces)


def dump_legacy_lang(entries: dict[str, str]) -> str:
    """Serialize a dict to legacy .lang format."""
    lines = [f"{key}={value}" for key, value in entries.items()]
    return "\n".join(lines) + "\n"


# ── Minecraft locale code mapping ──

MC_LANG_MAP: dict[str, str] = {
    "ES": "es_es",
    "FR": "fr_fr",
    "DE": "de_de",
    "IT": "it_it",
    "PT": "pt_br",
    "RU": "ru_ru",
    "PL": "pl_pl",
}

# Regex for format specifiers: %s, %1$s, %d, %2$s, %%
FORMAT_SPEC_RE = re.compile(r"%(?:\d+\$)?[sdfc]|%%")

# Regex for Minecraft section sign codes:
# §0-9, §a-f (colors), §k-o (formatting), §r (reset), §x/y/z (hex color prefix)
# Case-insensitive to catch §A, §C, etc.
SECTION_SIGN_RE = re.compile(r"§[0-9a-fk-orxyz]", re.IGNORECASE)
