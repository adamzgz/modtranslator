"""Glossary support: protect specific terms from being translated."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


def _normalize_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Normalize mangled placeholders after neural MT translation.

    Canonical form is ``Gx48``.  Translation backends (especially Opus-MT)
    may insert spaces or change case.  This function recovers known
    placeholder IDs from the mapping.
    """
    if not mapping:
        return text

    result = text
    for placeholder in mapping:
        if placeholder in result:
            continue
        m = re.match(r"([A-Z]x)(\d+)", placeholder)
        if not m:
            continue
        prefix, num = m.group(1), m.group(2)
        # Match mangled versions: inserted spaces, case changes
        mangled_re = re.compile(
            rf"(?<!\w){prefix}\s*{num}(?!\d)",
            re.IGNORECASE,
        )
        result = mangled_re.sub(placeholder, result)

    return result


@dataclass
class Glossary:
    """Manages term protection using placeholder substitution.

    Terms listed in the glossary are replaced with placeholders before
    translation, then restored with their target-language equivalents after.
    """

    # source_term → target_term
    terms: dict[str, str] = field(default_factory=dict)

    # Internal mapping for active protection (used by single protect/restore)
    _placeholders: dict[str, str] = field(default_factory=dict, repr=False)
    _reverse: dict[str, str] = field(default_factory=dict, repr=False)

    @staticmethod
    def _make_pattern(source: str) -> re.Pattern[str]:
        """Build a case-insensitive regex with word boundaries.

        Adds \\b at the start/end of the pattern when the source term
        starts/ends with a word character, preventing substring matches
        (e.g., "Dad" must not match inside "habilidades").
        """
        escaped = re.escape(source)
        prefix = r"\b" if re.match(r"\w", source) else ""
        suffix = r"\b" if re.search(r"\w$", source) else ""
        return re.compile(prefix + escaped + suffix, re.IGNORECASE)

    def merge(self, other: Glossary) -> None:
        """Merge another glossary. Other's terms override on conflict."""
        self.terms.update(other.terms)

    @classmethod
    def from_multiple_toml(cls, paths: list[Path]) -> Glossary:
        """Load and merge multiple TOML files. Later files override earlier ones."""
        if not paths:
            return cls()
        result = cls.from_toml(paths[0])
        for p in paths[1:]:
            result.merge(cls.from_toml(p))
        return result

    @classmethod
    def from_toml(cls, path: str | Path) -> Glossary:
        """Load a glossary from a TOML file.

        Expected format:
            [terms]
            Wasteland = "Yermo"
            Stimpak = "Estimulante"
            "Pip-Boy" = "Pip-Boy"
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        terms = data.get("terms", {})
        return cls(terms=terms)

    def _sorted_terms(self) -> list[tuple[int, str, str]]:
        """Return terms sorted by source length descending.

        Longer terms are replaced first so that multi-word terms like
        "Capital Wasteland" match before single-word terms like "Wasteland".
        """
        indexed = [(i, src, tgt) for i, (src, tgt) in enumerate(self.terms.items())]
        indexed.sort(key=lambda t: len(t[1]), reverse=True)
        return indexed

    def protect(self, text: str) -> str:
        """Replace glossary terms with placeholders before translation."""
        self._placeholders.clear()
        self._reverse.clear()

        protected = text
        for i, source, target in self._sorted_terms():
            placeholder = f"Gx{i}"
            pattern = self._make_pattern(source)
            if pattern.search(protected):
                self._placeholders[placeholder] = target
                self._reverse[source.lower()] = placeholder
                protected = pattern.sub(placeholder, protected)

        return protected

    def protect_with_mapping(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace glossary terms with placeholders. Stateless — returns mapping."""
        placeholders: dict[str, str] = {}
        protected = text
        for i, source, target in self._sorted_terms():
            placeholder = f"Gx{i}"
            pattern = self._make_pattern(source)
            if pattern.search(protected):
                placeholders[placeholder] = target
                protected = pattern.sub(placeholder, protected)

        return protected, placeholders

    def restore(self, text: str, placeholders: dict[str, str] | None = None) -> str:
        """Replace placeholders with target-language terms after translation.

        If placeholders dict is given, uses that instead of internal state.
        Normalizes mangled placeholders before restoring (handles MT models
        that strip angle brackets, add spaces, etc.).
        """
        mapping = placeholders if placeholders is not None else self._placeholders
        restored = _normalize_placeholders(text, mapping)
        for placeholder, target in mapping.items():
            restored = restored.replace(placeholder, target)
        return restored

    def protect_batch(self, texts: list[str]) -> tuple[list[str], list[dict[str, str]]]:
        """Protect terms in multiple texts. Each text gets independent placeholders.

        Returns (protected_texts, mappings) where each mapping is per-text.
        """
        protected_texts: list[str] = []
        mappings: list[dict[str, str]] = []
        for t in texts:
            protected, mapping = self.protect_with_mapping(t)
            protected_texts.append(protected)
            mappings.append(mapping)
        return protected_texts, mappings

    def restore_batch(
        self, texts: list[str], mappings: list[dict[str, str]] | None = None
    ) -> list[str]:
        """Restore placeholders in multiple texts.

        If mappings is given, uses per-text mappings. Otherwise falls back to
        internal state (backward-compatible).
        """
        if mappings is not None:
            return [self.restore(t, m) for t, m in zip(texts, mappings, strict=False)]
        return [self.restore(t) for t in texts]
