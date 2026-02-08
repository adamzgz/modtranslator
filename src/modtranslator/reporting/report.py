"""Translation report data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TranslationReport:
    """Collects statistics about a translation run."""

    source_file: str = ""
    output_file: str = ""
    target_lang: str = ""
    game_detected: str = ""
    backend: str = ""

    total_records: int = 0
    total_groups: int = 0
    total_strings_found: int = 0
    strings_from_cache: int = 0
    strings_translated: int = 0
    strings_patched: int = 0
    strings_failed: int = 0

    glossary_file: str | None = None
    glossary_terms: int = 0

    dry_run: bool = False
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None

    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def finish(self) -> None:
        self.finished_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "output_file": self.output_file,
            "target_lang": self.target_lang,
            "game_detected": self.game_detected,
            "backend": self.backend,
            "total_records": self.total_records,
            "total_groups": self.total_groups,
            "total_strings_found": self.total_strings_found,
            "strings_from_cache": self.strings_from_cache,
            "strings_translated": self.strings_translated,
            "strings_patched": self.strings_patched,
            "strings_failed": self.strings_failed,
            "glossary_file": self.glossary_file,
            "glossary_terms": self.glossary_terms,
            "dry_run": self.dry_run,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }
