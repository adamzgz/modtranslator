"""Pipeline types, constants, and enums (no internal dependencies).

Extracted from pipeline.py for maintainability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event

# Files larger than this are parsed twice to avoid holding both the plugin
# and the translation model in memory simultaneously.
_LARGE_FILE_THRESHOLD = 150 * 1024 * 1024  # 150 MB


class GameChoice(str, Enum):
    """User-facing game selection."""
    auto = "auto"
    fo3 = "fo3"
    fnv = "fnv"
    fo4 = "fo4"
    skyrim = "skyrim"


# Pipeline progress messages per language
_PIPELINE_MESSAGES: dict[str, dict[str, str]] = {
    "EN": {
        "files_scanned": "Files scanned",
        "found": "Found",
        "no_files": "No files",
        "translating_esp": "Translating {n} ESP/ESM files...",
        "translating_pex": "Translating {n} PEX files...",
        "translating_mcm": "Translating MCM files...",
    },
    "ES": {
        "files_scanned": "Archivos escaneados",
        "found": "Encontrado",
        "no_files": "Sin archivos",
        "translating_esp": "Traduciendo {n} archivos ESP/ESM...",
        "translating_pex": "Traduciendo {n} archivos PEX...",
        "translating_mcm": "Traduciendo archivos MCM...",
    },
    "FR": {
        "files_scanned": "Fichiers analysés",
        "found": "Trouvé",
        "no_files": "Aucun fichier",
        "translating_esp": "Traduction de {n} fichiers ESP/ESM...",
        "translating_pex": "Traduction de {n} fichiers PEX...",
        "translating_mcm": "Traduction des fichiers MCM...",
    },
    "DE": {
        "files_scanned": "Dateien gescannt",
        "found": "Gefunden",
        "no_files": "Keine Dateien",
        "translating_esp": "Übersetze {n} ESP/ESM-Dateien...",
        "translating_pex": "Übersetze {n} PEX-Dateien...",
        "translating_mcm": "Übersetze MCM-Dateien...",
    },
    "IT": {
        "files_scanned": "File analizzati",
        "found": "Trovato",
        "no_files": "Nessun file",
        "translating_esp": "Traduzione di {n} file ESP/ESM...",
        "translating_pex": "Traduzione di {n} file PEX...",
        "translating_mcm": "Traduzione dei file MCM...",
    },
    "PT": {
        "files_scanned": "Arquivos analisados",
        "found": "Encontrado",
        "no_files": "Sem arquivos",
        "translating_esp": "Traduzindo {n} arquivos ESP/ESM...",
        "translating_pex": "Traduzindo {n} arquivos PEX...",
        "translating_mcm": "Traduzindo arquivos MCM...",
    },
    "RU": {
        "files_scanned": "Файлы просканированы",
        "found": "Найдено",
        "no_files": "Нет файлов",
        "translating_esp": "Перевод {n} файлов ESP/ESM...",
        "translating_pex": "Перевод {n} файлов PEX...",
        "translating_mcm": "Перевод файлов MCM...",
    },
    "PL": {
        "files_scanned": "Pliki przeskanowane",
        "found": "Znaleziono",
        "no_files": "Brak plików",
        "translating_esp": "Tłumaczenie {n} plików ESP/ESM...",
        "translating_pex": "Tłumaczenie {n} plików PEX...",
        "translating_mcm": "Tłumaczenie plików MCM...",
    },
}


def _msg(lang: str, key: str, **kwargs: object) -> str:
    """Get a localized pipeline message. Falls back to English."""
    messages = _PIPELINE_MESSAGES.get(lang.upper(), _PIPELINE_MESSAGES["EN"])
    template = messages.get(key, _PIPELINE_MESSAGES["EN"].get(key, key))
    return template.format(**kwargs) if kwargs else template


# Type alias for progress callback: (phase, current, total, message)
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class BatchResult:
    """Result of a batch translation operation."""
    success_count: int = 0
    error_count: int = 0
    skip_count: int = 0
    total_strings: int = 0
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)


class CancelledError(Exception):
    """Raised when the user cancels the operation."""


def _check_cancel(cancel_event: Event | None) -> None:
    """Raise CancelledError if the cancel event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Operation cancelled by user")


@dataclass
class _FileContext:
    """Per-file state carried through the 3-phase pipeline."""

    file_path: Path
    output_path: Path | None
    plugin: object | None = None
    all_strings: list = field(default_factory=list)
    to_translate: list = field(default_factory=list)
    to_translate_keys: list[str] = field(default_factory=list)
    to_translate_originals: list[str] = field(default_factory=list)
    cached: dict[str, str] = field(default_factory=dict)
    protected_texts: list[str] = field(default_factory=list)
    gloss_mappings: list[dict[str, str]] | None = None
    lang_mappings: list[dict[str, str]] | None = None
    dedup_indices: list[int] = field(default_factory=list)
    translations: dict[str, str] = field(default_factory=dict)
    patched_count: int = 0
    status: str = "pending"
    error_message: str = ""


@dataclass
class ScanResult:
    """What was found in a directory."""
    esp_files: list[Path] = field(default_factory=list)
    pex_files: list[Path] = field(default_factory=list)
    has_mcm: bool = False
    mcm_directory: Path | None = None


@dataclass
class BatchAllResult:
    """Combined result from translating all file types."""
    esp_result: BatchResult | None = None
    pex_result: BatchResult | None = None
    mcm_result: BatchResult | None = None
    total_success: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0
