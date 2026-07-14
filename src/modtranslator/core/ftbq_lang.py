"""FTB Quests lang file parser (SNBT subset) for MC 1.21+ modpacks.

FTB Quests stores quest-book text as a tree of ``.snbt`` files under
``config/ftbquests/quests/lang/<locale>/`` (e.g. ``file.snbt``,
``chapter_group.snbt``, ``chapters/*.snbt``, ``reward_tables/*.snbt``).
Each file is a flat SNBT compound whose values are either a quoted
string or a list of quoted strings (paragraphs). Keys are dotted/hex
identifiers, usually unquoted.

This module parses that subset and serialises it back in FTB Quests
style (4-space indent, unquoted keys where possible, one list item per
line). Round-trip stable at the data level:
``parse_snbt_lang(dump_snbt_lang(parse_snbt_lang(x)))`` equals
``parse_snbt_lang(x)``.

Recognised string escapes are ``\\"``, ``\\\\`` and ``\\n``. Literal
sequences such as ``&7``, ``§``, ``\\\\n`` and ``//n`` are preserved
as-is. Well-formed but unsupported value types (nested compounds,
numbers, byte arrays) are skipped instead of crashing; structurally
malformed input raises :class:`SnbtError`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NoReturn

INDENT = "    "

_BARE_KEY_RE = re.compile(r"[A-Za-z0-9._-]+")


class SnbtError(Exception):
    """Raised when SNBT lang content is malformed."""


class _Scanner:
    """Single-pass recursive-descent scanner for the SNBT lang subset."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.len = len(text)

    def error(self, msg: str) -> NoReturn:
        raise SnbtError(f"{msg} at position {self.pos}")

    def peek(self) -> str:
        if self.pos >= self.len:
            return ""
        return self.text[self.pos]

    def skip_ws(self) -> None:
        while self.pos < self.len and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def skip_ws_commas(self) -> None:
        while self.pos < self.len and self.text[self.pos] in " \t\r\n,":
            self.pos += 1

    def parse_compound(self) -> dict[str, str | list[str]]:
        self.skip_ws()
        if self.peek() != "{":
            self.error("expected '{'")
        self.pos += 1
        result: dict[str, str | list[str]] = {}
        while True:
            self.skip_ws_commas()
            c = self.peek()
            if c == "":
                self.error("unterminated compound")
            if c == "}":
                self.pos += 1
                break
            key = self.parse_key()
            self.skip_ws()
            if self.peek() != ":":
                self.error("expected ':'")
            self.pos += 1
            self.skip_ws()
            value = self.parse_value()
            if value is not None:
                result[key] = value
        return result

    def parse_key(self) -> str:
        if self.peek() == '"':
            return self.parse_string()
        m = _BARE_KEY_RE.match(self.text, self.pos)
        if m is None:
            self.error("expected key")
        self.pos = m.end()
        return m.group()

    def parse_value(self) -> str | list[str] | None:
        c = self.peek()
        if c == "":
            self.error("unexpected end of input")
        if c == '"':
            return self.parse_string()
        if c == "[":
            return self.parse_list()
        if c == "{":
            self._skip_balanced("{", "}")
            return None
        self.skip_scalar()
        return None

    def parse_string(self) -> str:
        # Assumes the current char is the opening double quote.
        self.pos += 1
        chars: list[str] = []
        while True:
            if self.pos >= self.len:
                self.error("unterminated string")
            ch = self.text[self.pos]
            if ch == "\\":
                self.pos += 1
                if self.pos >= self.len:
                    self.error("unterminated escape")
                esc = self.text[self.pos]
                if esc == '"':
                    chars.append('"')
                elif esc == "\\":
                    chars.append("\\")
                elif esc == "n":
                    chars.append("\n")
                else:
                    # Unknown escape: preserve both chars literally.
                    chars.append("\\")
                    chars.append(esc)
                self.pos += 1
            elif ch == '"':
                self.pos += 1
                break
            else:
                chars.append(ch)
                self.pos += 1
        return "".join(chars)

    def parse_list(self) -> list[str]:
        # Assumes the current char is '['.
        self.pos += 1
        items: list[str] = []
        while True:
            self.skip_ws_commas()
            c = self.peek()
            if c == "":
                self.error("unterminated list")
            if c == "]":
                self.pos += 1
                break
            if c == '"':
                items.append(self.parse_string())
            elif c == "[":
                self._skip_balanced("[", "]")
            elif c == "{":
                self._skip_balanced("{", "}")
            else:
                self.skip_scalar()
        return items

    def skip_scalar(self) -> None:
        start = self.pos
        while self.pos < self.len and self.text[self.pos] not in " \t\r\n,{}[]\"":
            self.pos += 1
        if self.pos == start:
            self.error("unexpected character")

    def _skip_balanced(self, open_c: str, close_c: str) -> None:
        # Assumes the current char is open_c. Consumes a balanced pair,
        # respecting quoted strings so brackets inside them do not count.
        depth = 0
        while self.pos < self.len:
            ch = self.text[self.pos]
            if ch == '"':
                self.parse_string()
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    return
            self.pos += 1
        self.error(f"unterminated '{open_c}'")


def parse_snbt_lang(text: str) -> dict[str, str | list[str]]:
    """Parse a flat FTB Quests lang SNBT compound.

    Raises :class:`SnbtError` on malformed input.
    """
    scanner = _Scanner(text)
    result = scanner.parse_compound()
    scanner.skip_ws()
    if scanner.pos != scanner.len:
        raise SnbtError(f"unexpected trailing content at position {scanner.pos}")
    return result


def _encode_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _format_key(key: str) -> str:
    if key and _BARE_KEY_RE.fullmatch(key):
        return key
    return _encode_string(key)


def dump_snbt_lang(data: dict[str, str | list[str]]) -> str:
    """Serialize back in FTB Quests style.

    4-space indent, unquoted keys (when they match ``[A-Za-z0-9._-]+``),
    and one list item per line.
    """
    lines: list[str] = ["{"]
    for key, value in data.items():
        formatted_key = _format_key(key)
        if isinstance(value, list):
            if not value:
                lines.append(f"{INDENT}{formatted_key}: []")
            else:
                lines.append(f"{INDENT}{formatted_key}: [")
                for item in value:
                    lines.append(f"{INDENT}{INDENT}{_encode_string(item)}")
                lines.append(f"{INDENT}]")
        else:
            lines.append(f"{INDENT}{formatted_key}: {_encode_string(value)}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def find_lang_files(quests_dir: Path, source_locale: str = "en_us") -> list[Path]:
    """All ``.snbt`` files under ``<quests_dir>/lang/<source_locale>/`` (recursive)."""
    base = quests_dir / "lang" / source_locale
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.snbt") if p.is_file())


def target_path(source_file: Path, source_locale: str, target_locale: str) -> Path:
    """Map ``lang/en_us/chapters/x.snbt`` -> ``lang/es_es/chapters/x.snbt``."""
    parts = list(source_file.parts)
    idx = None
    for i in range(1, len(parts)):
        if parts[i] == source_locale and parts[i - 1] == "lang":
            idx = i
            break
    if idx is None:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == source_locale:
                idx = i
                break
    if idx is None:
        raise ValueError(f"source locale {source_locale!r} not found in {source_file}")
    parts[idx] = target_locale
    return Path(*parts)
