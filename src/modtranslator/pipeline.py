"""Shared pipeline logic for ESP/ESM, PEX, and MCM batch translation.

Used by both the CLI (cli.py) and the GUI (gui/). Extracts the 3-phase
pipeline (Prepare → Translate → Write) from cli.py into reusable functions
with callback-based progress reporting and cancellation support.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re as _re
import sys
import time as _time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event

from modtranslator.backends.base import TranslationBackend
from modtranslator.core.constants import Game

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


# ── Glossary resolution ──


def resolve_glossary_paths(
    glossary: Path | None,
    lang: str,
    game: GameChoice,
    detected_game: Game,
) -> list[Path]:
    """Resolve which glossary files to load based on --glossary, --game, and detected game."""
    if getattr(sys, "frozen", False):
        glossaries_dir = Path(sys._MEIPASS) / "glossaries"  # type: ignore[attr-defined]
    else:
        glossaries_dir = Path(__file__).resolve().parent.parent.parent / "glossaries"

    if glossary is not None:
        return [glossary] if glossary.exists() else []

    lang_lower = lang.lower()

    if game != GameChoice.auto:
        effective = game
    elif detected_game == Game.SKYRIM:
        effective = GameChoice.skyrim
    elif detected_game == Game.FALLOUT4:
        effective = GameChoice.fo4
    else:
        effective = GameChoice.fo3

    if effective == GameChoice.fo3:
        candidates = [
            glossaries_dir / "fallout" / "base" / f"fallout_base_{lang_lower}.toml",
            glossaries_dir / "fallout" / "fo3" / f"fallout3_{lang_lower}.toml",
        ]
    elif effective == GameChoice.fnv:
        candidates = [
            glossaries_dir / "fallout" / "base" / f"fallout_base_{lang_lower}.toml",
            glossaries_dir / "fallout" / "fnv" / f"falloutnv_{lang_lower}.toml",
        ]
    elif effective == GameChoice.fo4:
        candidates = [
            glossaries_dir / "fallout" / "base" / f"fallout_base_{lang_lower}.toml",
            glossaries_dir / "fallout" / "fo4" / f"fallout4_{lang_lower}.toml",
        ]
    elif effective == GameChoice.skyrim:
        candidates = [
            glossaries_dir / "skyrim" / f"skyrim_base_{lang_lower}.toml",
            glossaries_dir / "skyrim" / f"skyrim_{lang_lower}.toml",
        ]
    else:
        return []

    return [p for p in candidates if p.exists()]


# ── Backend creation ──


def create_backend(
    backend_name: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    device: str | None = None,
) -> tuple[TranslationBackend, str]:
    """Create a translation backend instance.

    Returns:
        Tuple of (backend_instance, backend_label_for_report).

    Raises:
        ValueError: If DeepL is selected without an API key.
    """
    from modtranslator.backends.dummy import DummyBackend

    if backend_name == "dummy":
        return DummyBackend(), "dummy"
    elif backend_name == "opus-mt":
        from modtranslator.backends.opus_mt import OpusMTBackend
        effective_device = device or "auto"
        variant = model or "base"
        return OpusMTBackend(device=effective_device, model_variant=variant), f"opus-mt:{variant}"
    elif backend_name == "nllb":
        from modtranslator.backends.nllb import NLLBBackend
        effective_device = device or "auto"
        nllb_size = model or "1.3B"
        return NLLBBackend(device=effective_device, model_size=nllb_size), f"nllb:{nllb_size}"
    elif backend_name == "hybrid":
        from modtranslator.backends.hybrid import HybridBackend
        effective_device = device or "auto"
        return HybridBackend(device=effective_device), "hybrid:tc-big+nllb"
    else:
        # Default: deepl
        if not api_key:
            raise ValueError(
                "DeepL API key required. Use --api-key or set DEEPL_API_KEY."
            )
        from modtranslator.backends.deepl import DeepLBackend
        return DeepLBackend(api_key), "deepl"


# ── Internal per-file context ──


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


def _prepare_file(
    file_path: Path,
    output_path: Path | None,
    lang: str,
    gloss: object | None,
    glossary_terms: set[str] | None,
    glossary_source_terms: set[str] | None,
    skip_translated: bool,
) -> _FileContext:
    """Phase 1: parse, extract, filter, protect."""
    from modtranslator.core.plugin import load_plugin
    from modtranslator.translation.extractor import extract_strings

    ctx = _FileContext(file_path=file_path, output_path=output_path)

    try:
        plugin = load_plugin(file_path)
        ctx.plugin = plugin

        strings = extract_strings(plugin)
        file_stem = file_path.stem
        for s in strings:
            s.source_file = file_stem
        ctx.all_strings = strings

        if not strings:
            ctx.status = "skipped"
            return ctx

        if skip_translated:
            from modtranslator.translation.lang_detect import should_translate

            strings = [
                s for s in strings
                if should_translate(s.original_text, lang, glossary_terms, glossary_source_terms)
            ]

        if not strings:
            ctx.status = "skipped"
            return ctx

        ctx.all_strings = strings
        ctx.to_translate = list(strings)
        texts = [s.original_text for s in ctx.to_translate]

        if gloss and texts:
            texts, ctx.gloss_mappings = gloss.protect_batch(texts)  # type: ignore[union-attr]

        if texts:
            from modtranslator.translation.target_protect import protect_target_batch

            texts, ctx.lang_mappings = protect_target_batch(texts, lang)

        ctx.protected_texts = texts
        ctx.status = "prepared"

    except Exception as e:
        ctx.status = "error"
        ctx.error_message = str(e)

    return ctx


def _build_dedup_map(
    contexts: list[_FileContext],
) -> tuple[list[str], dict[str, int]]:
    """Build deduplicated text list from all prepared contexts."""
    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []

    for ctx in contexts:
        if ctx.status != "prepared" or not ctx.protected_texts:
            continue

        indices: list[int] = []
        for text in ctx.protected_texts:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices.append(text_to_index[text])
        ctx.dedup_indices = indices

    return unique_texts, text_to_index


def _writeback_file(
    ctx: _FileContext,
    translated_unique: list[str],
    gloss: object | None,
    lang: str,
    backend_label: str,
) -> _FileContext:
    """Phase 3: restore placeholders, patch, save."""
    from modtranslator.core.plugin import load_plugin, save_plugin
    from modtranslator.translation.extractor import extract_strings
    from modtranslator.translation.patcher import apply_translations

    try:
        if ctx.dedup_indices and translated_unique:
            translated = [translated_unique[i] for i in ctx.dedup_indices]
        else:
            translated = []

        if ctx.lang_mappings is not None and translated:
            from modtranslator.translation.target_protect import restore_target_batch

            translated = restore_target_batch(translated, ctx.lang_mappings)

        if gloss and translated:
            translated = gloss.restore_batch(translated, ctx.gloss_mappings)  # type: ignore[union-attr]

        translations: dict[str, str] = dict(ctx.cached)
        for key, t in zip(ctx.to_translate_keys, translated, strict=True):
            translations[key] = t

        ctx.translations = translations

        if ctx.plugin is None:
            ctx.plugin = load_plugin(ctx.file_path)
            ctx.all_strings = extract_strings(ctx.plugin)
            file_stem = ctx.file_path.stem
            for s in ctx.all_strings:
                s.source_file = file_stem

        from modtranslator.core.constants import encoding_for_lang
        st = getattr(ctx.plugin, "string_tables", None)
        patched = apply_translations(
            ctx.all_strings, translations,
            encoding=encoding_for_lang(lang),
            string_tables=st,
        )
        ctx.patched_count = patched

        from modtranslator.core.string_table import ISO_TO_FULL_LANGUAGE
        output_language = ISO_TO_FULL_LANGUAGE.get(lang.upper(), lang)

        if ctx.output_path is not None:
            save_plugin(ctx.plugin, ctx.output_path, output_language=output_language)  # type: ignore[arg-type]
        else:
            out = ctx.file_path.with_stem(f"{ctx.file_path.stem}_{lang}")
            save_plugin(ctx.plugin, out, output_language=output_language)  # type: ignore[arg-type]

        ctx.status = "written"

    except Exception as e:
        ctx.status = "error"
        ctx.error_message = str(e)

    return ctx


# ── Shared translate-in-chunks helper ──


def _translate_chunks(
    texts: list[str],
    backend: TranslationBackend,
    lang: str,
    chunk_size: int = 500,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
    phase_name: str = "translate",
) -> tuple[list[str], list[str]]:
    """Translate texts in chunks with progress and cancellation.

    Returns (translated_texts, chunk_errors).
    """
    translated: list[str] = []
    chunk_errors: list[str] = []
    total = len(texts)
    for chunk_start in range(0, total, chunk_size):
        _check_cancel(cancel_event)
        chunk = texts[chunk_start:chunk_start + chunk_size]
        try:
            translated.extend(backend.translate_batch(chunk, lang))
        except CancelledError:
            raise
        except Exception as e:
            chunk_errors.append(
                f"Chunk {chunk_start // chunk_size + 1}: {e}"
            )
            translated.extend(chunk)  # fallback: original text
        if on_progress:
            on_progress(phase_name, len(translated), total, "")
    return translated, chunk_errors


# ── Setup glossary helper ──


def _setup_glossary(
    glossary: Path | None,
    lang: str,
    game: GameChoice,
    detected_game: Game,
) -> tuple[object | None, set[str] | None, set[str] | None]:
    """Load glossary and build term sets. Returns (gloss, glossary_terms, glossary_source_terms)."""
    from modtranslator.translation.glossary import Glossary

    gloss = None
    glossary_terms: set[str] | None = None
    glossary_source_terms: set[str] | None = None

    gloss_paths = resolve_glossary_paths(glossary, lang, game, detected_game)
    if gloss_paths:
        gloss = Glossary.from_multiple_toml(gloss_paths)
        glossary_terms = {v.lower() for v in gloss.terms.values()}
        glossary_source_terms = {
            k.lower() for k, v in gloss.terms.items()
            if k.lower() != v.lower()
        }

    return gloss, glossary_terms, glossary_source_terms


# ═══════════════════════════════════════════════════════════════════
# Public API: ESP/ESM batch translation
# ═══════════════════════════════════════════════════════════════════


def batch_translate_esp(
    files: list[Path],
    *,
    lang: str = "ES",
    backend: TranslationBackend,
    backend_label: str,
    game: GameChoice = GameChoice.auto,
    glossary: Path | None = None,
    skip_translated: bool = True,
    output_dir: Path | None = None,
    no_cache: bool = False,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate a list of ESP/ESM files using the 3-phase pipeline."""
    from modtranslator.translation.cache import TranslationCache
    from modtranslator.translation.lang_detect import _load_dictionary

    result = BatchResult()
    t0 = _time.monotonic()

    try:
        _load_dictionary(lang)
        cache = None if no_cache else TranslationCache()

        # Detect game from first file if auto
        if game == GameChoice.auto:
            from modtranslator.core.plugin import load_plugin
            try:
                first_plugin = load_plugin(files[0])
                detected_game = first_plugin.game
            except Exception:
                detected_game = Game.FALLOUT3
        elif game == GameChoice.skyrim:
            detected_game = Game.SKYRIM
        elif game == GameChoice.fo4:
            detected_game = Game.FALLOUT4
        else:
            detected_game = Game.FALLOUT3

        gloss, glossary_terms, glossary_source_terms = _setup_glossary(
            glossary, lang, game, detected_game,
        )

        # Build file pairs
        file_pairs: list[tuple[Path, Path | None]] = []
        for f in files:
            out = (output_dir / f.name) if output_dir is not None else None
            file_pairs.append((f, out))

        # ── Phase 1: Prepare ──
        contexts: list[_FileContext] = []
        for i, (fp, out) in enumerate(file_pairs):
            _check_cancel(cancel_event)
            ctx = _prepare_file(fp, out, lang, gloss, glossary_terms,
                                glossary_source_terms, skip_translated)
            contexts.append(ctx)
            if on_progress:
                on_progress("prepare", i + 1, len(files), fp.name)

        # Cache check (main thread — SQLite not thread-safe)
        if cache is not None:
            for ctx in contexts:
                if ctx.status != "prepared":
                    continue
                keys = [s.key for s in ctx.all_strings]
                ctx.cached = cache.get_batch(keys, lang)
                if ctx.cached:
                    ctx.to_translate = [s for s in ctx.all_strings if s.key not in ctx.cached]
                    texts = [s.original_text for s in ctx.to_translate]
                    if gloss and texts:
                        texts, ctx.gloss_mappings = gloss.protect_batch(texts)  # type: ignore[union-attr]
                    else:
                        ctx.gloss_mappings = None
                    if texts:
                        from modtranslator.translation.target_protect import protect_target_batch
                        texts, ctx.lang_mappings = protect_target_batch(texts, lang)
                    else:
                        ctx.lang_mappings = None
                    ctx.protected_texts = texts

        # Save keys/originals
        for ctx in contexts:
            if ctx.status == "prepared":
                ctx.to_translate_keys = [s.key for s in ctx.to_translate]
                ctx.to_translate_originals = [s.original_text for s in ctx.to_translate]

        # Deduplication
        unique_texts, _ = _build_dedup_map(contexts)

        # Free large plugins
        for ctx in contexts:
            if ctx.status != "prepared" or not ctx.protected_texts:
                continue
            if ctx.file_path.stat().st_size > _LARGE_FILE_THRESHOLD:
                ctx.plugin = None
                ctx.all_strings = []
                ctx.to_translate = []
                gc.collect()

        result.total_strings = len(unique_texts)

        # ── Phase 2: Translate ──
        translated_unique: list[str] = []
        if unique_texts:
            _check_cancel(cancel_event)
            translated_unique, chunk_errors = _translate_chunks(
                unique_texts, backend, lang,
                on_progress=on_progress, cancel_event=cancel_event,
            )
            for err in chunk_errors:
                result.errors.append(("batch", err))

        # ── Phase 3: Write ──
        writable = [c for c in contexts if c.status == "prepared"]
        for i, ctx in enumerate(writable):
            _check_cancel(cancel_event)
            _writeback_file(ctx, translated_unique, gloss, lang, backend_label)
            if on_progress:
                on_progress("write", i + 1, len(writable), ctx.file_path.name)

        # Roundtrip skipped files when output_dir is set
        skipped_with_output = [
            c for c in contexts
            if c.status == "skipped" and c.output_path is not None
        ]
        if skipped_with_output:
            from modtranslator.core.plugin import load_plugin, save_plugin
            from modtranslator.core.string_table import ISO_TO_FULL_LANGUAGE

            _out_lang = ISO_TO_FULL_LANGUAGE.get(lang.upper(), lang)
            for ctx in skipped_with_output:
                try:
                    plugin = load_plugin(ctx.file_path)
                    save_plugin(plugin, ctx.output_path, output_language=_out_lang)  # type: ignore[arg-type]
                    ctx.status = "written"
                except Exception as e:
                    ctx.status = "error"
                    ctx.error_message = str(e)

        # Cache write-back
        if cache is not None:
            try:
                cache_entries: list[tuple[str, str, str, str]] = []
                for ctx in contexts:
                    if ctx.status == "written":
                        for key, orig in zip(
                            ctx.to_translate_keys,
                            ctx.to_translate_originals,
                            strict=True,
                        ):
                            t_text = ctx.translations.get(key, "")
                            if t_text:
                                cache_entries.append((key, lang, orig, t_text))
                if cache_entries:
                    cache.put_batch(cache_entries, backend=backend_label)
            except Exception as exc:
                warnings.warn(
                    f"Failed to write translation cache: {exc}",
                    stacklevel=1,
                )
            finally:
                cache.close()

        # Collect results
        for ctx in contexts:
            if ctx.status == "written":
                result.success_count += 1
            elif ctx.status == "skipped":
                result.skip_count += 1
            elif ctx.status == "error":
                result.error_count += 1
                result.errors.append((ctx.file_path.name, ctx.error_message))

    except CancelledError:
        result.errors.append(("", "Cancelled by user"))

    result.elapsed_seconds = _time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════
# Public API: PEX batch translation
# ═══════════════════════════════════════════════════════════════════


def batch_translate_pex(
    files: list[Path],
    *,
    lang: str = "ES",
    backend: TranslationBackend,
    backend_label: str,
    game: GameChoice = GameChoice.skyrim,
    glossary: Path | None = None,
    skip_translated: bool = True,
    output_dir: Path | None = None,
    no_cache: bool = False,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate a list of .pex Papyrus script files."""
    from modtranslator.core.pex_parser import load_pex, save_pex
    from modtranslator.translation.cache import TranslationCache

    result = BatchResult()
    t0 = _time.monotonic()

    try:
        if game == GameChoice.skyrim:
            detected_game = Game.SKYRIM
        elif game == GameChoice.fo4:
            detected_game = Game.FALLOUT4
        else:
            detected_game = Game.FALLOUT3
        gloss, glossary_terms, glossary_source_terms = _setup_glossary(
            glossary, lang, game, detected_game,
        )

        cache = None if no_cache else TranslationCache()

        # Phase 1: Scan .pex files
        all_entries: list[tuple[Path, dict[int, str]]] = []
        unique_texts: list[str] = []
        text_to_index: dict[str, int] = {}
        parse_error_count = 0

        for i, pex_path in enumerate(files):
            _check_cancel(cancel_event)
            try:
                pex = load_pex(pex_path)
                translatable = pex.get_translatable_strings()

                if skip_translated and translatable:
                    from modtranslator.translation.lang_detect import should_translate
                    translatable = {
                        idx: text for idx, text in translatable.items()
                        if should_translate(text, lang, glossary_terms, glossary_source_terms)
                    }

                if translatable:
                    all_entries.append((pex_path, translatable))
                    for text in translatable.values():
                        if text not in text_to_index:
                            text_to_index[text] = len(unique_texts)
                            unique_texts.append(text)
            except Exception as e:
                parse_error_count += 1
                result.error_count += 1
                result.errors.append((pex_path.name, str(e)))
            if on_progress:
                on_progress("prepare", i + 1, len(files), pex_path.name)

        result.total_strings = len(unique_texts)

        if not unique_texts:
            result.skip_count = len(files)
            result.elapsed_seconds = _time.monotonic() - t0
            return result

        # Cache check
        cached: dict[str, str] = {}
        if cache is not None:
            cached = cache.get_batch(unique_texts, lang)

        to_translate = [t for t in unique_texts if t not in cached]

        # Protect glossary terms
        gloss_mappings: list[dict[str, str]] | None = None
        if gloss and to_translate:
            to_translate, gloss_mappings = gloss.protect_batch(to_translate)  # type: ignore[union-attr]

        # Protect target-language words
        lang_mappings: list[dict[str, str]] | None = None
        if to_translate:
            from modtranslator.translation.target_protect import protect_target_batch
            to_translate, lang_mappings = protect_target_batch(to_translate, lang)

        # Phase 2: Translate
        translated: list[str] = []
        if to_translate:
            _check_cancel(cancel_event)
            translated, chunk_errors = _translate_chunks(
                to_translate, backend, lang,
                on_progress=on_progress, cancel_event=cancel_event,
            )
            for err in chunk_errors:
                result.errors.append(("batch", err))

            if lang_mappings is not None:
                from modtranslator.translation.target_protect import restore_target_batch
                translated = restore_target_batch(translated, lang_mappings)

            if gloss and gloss_mappings:
                translated = gloss.restore_batch(translated, gloss_mappings)  # type: ignore[union-attr]

        # Build translation map
        translations: dict[str, str] = dict(cached)
        uncached_originals = [t for t in unique_texts if t not in cached]
        for orig, trans in zip(uncached_originals, translated, strict=True):
            translations[orig] = trans

        # Cache new translations
        if cache is not None:
            try:
                if translated:
                    cache_entries = [
                        (orig, lang, orig, translations[orig])
                        for orig in uncached_originals
                        if orig in translations
                    ]
                    if cache_entries:
                        cache.put_batch(cache_entries, backend=backend_label)
            except Exception as exc:
                warnings.warn(
                    f"Failed to write translation cache: {exc}",
                    stacklevel=1,
                )
            finally:
                cache.close()

        # Phase 3: Write
        files_with_text = len(all_entries)
        for i, (pex_path, translatable) in enumerate(all_entries):
            _check_cancel(cancel_event)
            try:
                pex = load_pex(pex_path)
                modified = False
                for idx, original_text in translatable.items():
                    if original_text in translations:
                        pex.string_table[idx] = translations[original_text]
                        modified = True

                if modified:
                    out_path = (output_dir / pex_path.name) if output_dir else pex_path
                    save_pex(pex, out_path)
                    result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append((pex_path.name, str(e)))
            if on_progress:
                on_progress("write", i + 1, files_with_text, pex_path.name)

        result.skip_count = len(files) - files_with_text - parse_error_count

    except CancelledError:
        result.errors.append(("", "Cancelled by user"))

    result.elapsed_seconds = _time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════
# Public API: MCM batch translation
# ═══════════════════════════════════════════════════════════════════


def batch_translate_mcm(
    directory: Path,
    *,
    lang: str = "ES",
    backend: TranslationBackend,
    backend_label: str,
    game: GameChoice = GameChoice.skyrim,
    glossary: Path | None = None,
    skip_translated: bool = True,
    output_dir: Path | None = None,
    no_cache: bool = False,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate MCM Interface/translations/ files and MCM Recorder JSON."""
    from modtranslator.translation.cache import TranslationCache

    lang_names = {
        "ES": "spanish", "FR": "french", "DE": "german", "IT": "italian",
        "PT": "portuguese", "RU": "russian", "PL": "polish", "CS": "czech",
        "JA": "japanese", "ZH": "chinese",
    }
    # Short codes used by FO4 MCM files (e.g. _en.txt, _es.txt)
    lang_short = {
        "ES": "es", "FR": "fr", "DE": "de", "IT": "it",
        "PT": "ptbr", "RU": "ru", "PL": "pl", "CS": "cs",
        "JA": "ja", "ZH": "cn",
    }
    target_lang_name = lang_names.get(lang.upper(), lang.lower())
    target_lang_short = lang_short.get(lang.upper(), lang.lower())

    translations_dir = directory / "Interface" / "translations"
    mcm_recorder_dir = directory / "McmRecorder"

    if not translations_dir.is_dir() and not mcm_recorder_dir.is_dir():
        raise FileNotFoundError(
            f"No Interface/translations/ or McmRecorder/ in {directory}"
        )

    result = BatchResult()
    t0 = _time.monotonic()

    try:
        if game == GameChoice.skyrim:
            detected_game = Game.SKYRIM
        elif game == GameChoice.fo4:
            detected_game = Game.FALLOUT4
        else:
            detected_game = Game.FALLOUT3
        gloss, glossary_terms, glossary_source_terms = _setup_glossary(
            glossary, lang, game, detected_game,
        )

        cache = None if no_cache else TranslationCache()

        # Phase 1: Collect translatable strings
        mcm_files: list[tuple[Path, Path, list[str], list[int], list[str]]] = []
        mcm_jsons: list[tuple[Path, Path, dict, list[str], list[str]]] = []

        if translations_dir.is_dir():
            all_txt = sorted(translations_dir.iterdir())
            eng_files = [
                f for f in all_txt
                if _re.search(r'_(?:english|en)\.txt$', f.name, _re.IGNORECASE)
            ]

            for eng_path in eng_files:
                _check_cancel(cancel_event)
                stem = _re.sub(r'(?i)_(?:english|en)\.txt$', '', eng_path.name)

                # Check for existing target file (both long and short naming)
                target_name_variants = [
                    f"{stem}_{target_lang_name}.txt",
                    f"{stem}_{target_lang_name.upper()}.txt",
                    f"{stem}_{target_lang_name.capitalize()}.txt",
                    f"{stem}_{target_lang_short}.txt",
                    f"{stem}_{target_lang_short.upper()}.txt",
                ]
                existing_spa = None
                for variant in target_name_variants:
                    candidate = translations_dir / variant
                    if candidate.exists():
                        existing_spa = candidate
                        break

                if existing_spa is not None:
                    eng_hash = hashlib.md5(eng_path.read_bytes()).hexdigest()
                    spa_hash = hashlib.md5(existing_spa.read_bytes()).hexdigest()
                    if eng_hash != spa_hash:
                        continue

                # Determine output filename using same convention as source
                uses_short = _re.search(r'_en\.txt$', eng_path.name, _re.IGNORECASE)
                if existing_spa is not None:
                    out_name = existing_spa.name
                elif uses_short:
                    out_name = f"{stem}_{target_lang_short}.txt"
                elif "_ENGLISH." in eng_path.name:
                    out_name = f"{stem}_{target_lang_name.upper()}.txt"
                else:
                    out_name = f"{stem}_{target_lang_name}.txt"

                if output_dir is not None:
                    out_path = output_dir / out_name
                else:
                    out_path = translations_dir / out_name

                try:
                    raw = eng_path.read_bytes()
                    if raw[:2] == b'\xff\xfe':
                        text = raw.decode('utf-16-le')
                        if text and text[0] == '\ufeff':
                            text = text[1:]
                    else:
                        text = raw.decode('utf-8', errors='replace')

                    lines = text.splitlines(keepends=True)
                    translatable_indices: list[int] = []
                    translatable_texts: list[str] = []

                    for i, line in enumerate(lines):
                        stripped = line.rstrip('\r\n')
                        if not stripped or stripped.startswith(';'):
                            continue
                        if '\t' in stripped and stripped.startswith('$'):
                            parts = stripped.split('\t', 1)
                            if len(parts) == 2:
                                value = parts[1].strip()
                                if value and any(c.isalpha() for c in value):
                                    translatable_indices.append(i)
                                    translatable_texts.append(value)

                    if translatable_texts:
                        mcm_files.append(
                            (eng_path, out_path, lines, translatable_indices, translatable_texts)
                        )
                except Exception:
                    pass

        if mcm_recorder_dir.is_dir():
            json_translatable_keys = ["welcome", "complete"]
            for json_path in sorted(mcm_recorder_dir.glob("*.json")):
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    keys_found: list[str] = []
                    texts_found: list[str] = []
                    for k in json_translatable_keys:
                        if k in data and isinstance(data[k], str) and data[k].strip():
                            keys_found.append(k)
                            texts_found.append(data[k])
                    if texts_found:
                        out_json = (output_dir / json_path.name) if output_dir else json_path
                        mcm_jsons.append((json_path, out_json, data, keys_found, texts_found))
                except Exception:
                    pass

        total_files = len(mcm_files) + len(mcm_jsons)
        if total_files == 0:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

        if on_progress:
            on_progress("prepare", total_files, total_files, _msg(lang, "files_scanned"))

        # Collect all unique texts
        unique_texts: list[str] = []
        text_to_index: dict[str, int] = {}

        for _, _, _, _, texts in mcm_files:
            for t in texts:
                if t not in text_to_index:
                    text_to_index[t] = len(unique_texts)
                    unique_texts.append(t)
        for _, _, _, _, texts in mcm_jsons:
            for t in texts:
                if t not in text_to_index:
                    text_to_index[t] = len(unique_texts)
                    unique_texts.append(t)

        result.total_strings = len(unique_texts)

        # Language filter
        if skip_translated:
            from modtranslator.translation.lang_detect import should_translate

            to_translate_set = {
                t for t in unique_texts
                if should_translate(t, lang, glossary_terms, glossary_source_terms)
            }
            texts_for_translation = [t for t in unique_texts if t in to_translate_set]
        else:
            texts_for_translation = list(unique_texts)

        # Cache check
        cached: dict[str, str] = {}
        if cache is not None:
            cached = cache.get_batch(texts_for_translation, lang)

        to_translate = [t for t in texts_for_translation if t not in cached]

        # Protect HTML tags (e.g. <font color='#33dd33'>text</font>)
        html_mappings: list[dict[str, str]] | None = None

        def _protect_html(texts: list[str]) -> tuple[list[str], list[dict[str, str]]]:
            mappings: list[dict[str, str]] = []
            result: list[str] = []
            for text in texts:
                tags = _re.findall(r'<[^>]+>', text)
                mapping: dict[str, str] = {}
                protected = text
                for i, tag in enumerate(tags):
                    placeholder = f"HTx{i}"
                    mapping[placeholder] = tag
                    protected = protected.replace(tag, placeholder, 1)
                mappings.append(mapping)
                result.append(protected)
            return result, mappings

        def _restore_html(texts: list[str], mappings: list[dict[str, str]]) -> list[str]:
            result: list[str] = []
            for text, mapping in zip(texts, mappings, strict=True):
                for placeholder, tag in mapping.items():
                    text = text.replace(placeholder, tag)
                result.append(text)
            return result

        # Protect glossary terms
        gloss_mappings: list[dict[str, str]] | None = None
        if gloss and to_translate:
            to_translate, gloss_mappings = gloss.protect_batch(to_translate)  # type: ignore[union-attr]

        # Protect HTML tags
        if to_translate and any('<' in t for t in to_translate):
            to_translate, html_mappings = _protect_html(to_translate)

        # Protect target-language words
        lang_mappings: list[dict[str, str]] | None = None
        if to_translate:
            from modtranslator.translation.target_protect import protect_target_batch
            to_translate, lang_mappings = protect_target_batch(to_translate, lang)

        # Phase 2: Translate
        translated: list[str] = []
        if to_translate:
            _check_cancel(cancel_event)
            translated, chunk_errors = _translate_chunks(
                to_translate, backend, lang,
                on_progress=on_progress, cancel_event=cancel_event,
            )
            for err in chunk_errors:
                result.errors.append(("batch", err))

            if lang_mappings is not None:
                from modtranslator.translation.target_protect import restore_target_batch
                translated = restore_target_batch(translated, lang_mappings)

            if html_mappings is not None:
                translated = _restore_html(translated, html_mappings)

            if gloss and gloss_mappings:
                translated = gloss.restore_batch(translated, gloss_mappings)  # type: ignore[union-attr]

        # Build translation map
        translations: dict[str, str] = dict(cached)
        uncached_originals = [t for t in texts_for_translation if t not in cached]
        for orig, trans in zip(uncached_originals, translated, strict=True):
            translations[orig] = trans

        # Cache new translations
        if cache is not None:
            try:
                if translated:
                    cache_entries = [
                        (orig, lang, orig, translations[orig])
                        for orig in uncached_originals
                        if orig in translations
                    ]
                    if cache_entries:
                        cache.put_batch(cache_entries, backend=backend_label)
            except Exception as exc:
                warnings.warn(
                    f"Failed to write translation cache: {exc}",
                    stacklevel=1,
                )
            finally:
                cache.close()

        # Phase 3: Write
        write_idx = 0
        for eng_path, out_path, lines, indices, orig_texts in mcm_files:
            _check_cancel(cancel_event)
            try:
                new_lines = list(lines)
                for idx, orig_text in zip(indices, orig_texts, strict=True):
                    if orig_text in translations:
                        line = new_lines[idx]
                        stripped = line.rstrip('\r\n')
                        ending = line[len(stripped):]
                        parts = stripped.split('\t', 1)
                        new_lines[idx] = f"{parts[0]}\t{translations[orig_text]}{ending}"

                out_path.parent.mkdir(parents=True, exist_ok=True)
                content = ''.join(new_lines)
                out_path.write_bytes(b'\xff\xfe' + content.encode('utf-16-le'))
                result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append((eng_path.name, str(e)))
            write_idx += 1
            if on_progress:
                on_progress("write", write_idx, total_files, eng_path.name)

        for json_path, out_json, data, keys, orig_texts in mcm_jsons:
            _check_cancel(cancel_event)
            try:
                for key, orig_text in zip(keys, orig_texts, strict=True):
                    if orig_text in translations:
                        data[key] = translations[orig_text]
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append((json_path.name, str(e)))
            write_idx += 1
            if on_progress:
                on_progress("write", write_idx, total_files, json_path.name)

    except CancelledError:
        result.errors.append(("", "Cancelled by user"))

    result.elapsed_seconds = _time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════
# Public API: Utilities
# ═══════════════════════════════════════════════════════════════════


def scan_file(file: Path) -> list:
    """Parse an ESP/ESM file and return its translatable strings."""
    from modtranslator.core.plugin import load_plugin
    from modtranslator.translation.extractor import extract_strings

    plugin = load_plugin(file)
    strings = extract_strings(plugin)
    for s in strings:
        s.source_file = file.stem
    return strings


def get_cache_info() -> dict[str, object]:
    """Return cache statistics."""
    from modtranslator.translation.cache import TranslationCache

    cache = TranslationCache()
    try:
        info = {
            "count": cache.count(),
            "path": str(cache._db_path),
        }
    finally:
        cache.close()
    return info


def clear_cache() -> int:
    """Clear the translation cache. Returns number of entries deleted."""
    from modtranslator.translation.cache import TranslationCache

    cache = TranslationCache()
    try:
        deleted = cache.clear()
    finally:
        cache.close()
    return deleted


# ═══════════════════════════════════════════════════════════════════
# Public API: Auto-detect and translate everything
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ScanResult:
    """What was found in a directory."""
    esp_files: list[Path] = field(default_factory=list)
    pex_files: list[Path] = field(default_factory=list)
    has_mcm: bool = False
    mcm_directory: Path | None = None


def scan_directory(directory: Path) -> ScanResult:
    """Scan a directory for translatable content (ESP/ESM, PEX, MCM)."""
    result = ScanResult()

    # ESP/ESM files (direct children)
    result.esp_files = sorted(
        list(directory.glob("*.esp"))
        + list(directory.glob("*.esm"))
        + list(directory.glob("*.esl"))
    )

    # PEX files (direct children or Scripts/ subdirectory)
    pex_files = list(directory.glob("*.pex"))
    scripts_dir = directory / "Scripts"
    if scripts_dir.is_dir():
        pex_files.extend(scripts_dir.glob("*.pex"))
    result.pex_files = sorted(pex_files)

    # MCM: check for Interface/translations/ or McmRecorder/
    translations_dir = directory / "Interface" / "translations"
    mcm_recorder_dir = directory / "McmRecorder"
    if translations_dir.is_dir() or mcm_recorder_dir.is_dir():
        result.has_mcm = True
        result.mcm_directory = directory

    return result


@dataclass
class BatchAllResult:
    """Combined result from translating all file types."""
    esp_result: BatchResult | None = None
    pex_result: BatchResult | None = None
    mcm_result: BatchResult | None = None
    total_success: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0


def batch_translate_all(
    directory: Path,
    *,
    output_dir: Path,
    lang: str = "ES",
    backend: TranslationBackend,
    backend_label: str,
    game: GameChoice = GameChoice.auto,
    glossary: Path | None = None,
    skip_translated: bool = True,
    no_cache: bool = False,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchAllResult:
    """Auto-detect and translate all content in a directory.

    Scans for ESP/ESM, PEX, and MCM files and translates them sequentially.
    """
    t0 = _time.monotonic()
    result = BatchAllResult()

    scan = scan_directory(directory)

    # Notify what was found
    if on_progress:
        found_parts = []
        if scan.esp_files:
            found_parts.append(f"{len(scan.esp_files)} ESP/ESM")
        if scan.pex_files:
            found_parts.append(f"{len(scan.pex_files)} PEX")
        if scan.has_mcm:
            found_parts.append("MCM")
        if found_parts:
            msg = _msg(lang, "found") + ": " + ", ".join(found_parts)
        else:
            msg = _msg(lang, "no_files")
        on_progress("scan", 0, 0, msg)

    # ESP/ESM
    if scan.esp_files:
        if on_progress:
            on_progress("scan", 0, 0, _msg(lang, "translating_esp", n=len(scan.esp_files)))
        try:
            _check_cancel(cancel_event)
            result.esp_result = batch_translate_esp(
                scan.esp_files,
                lang=lang, backend=backend, backend_label=backend_label,
                game=game, glossary=glossary, skip_translated=skip_translated,
                output_dir=output_dir, no_cache=no_cache,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # PEX
    if scan.pex_files:
        if on_progress:
            on_progress("scan", 0, 0, _msg(lang, "translating_pex", n=len(scan.pex_files)))
        try:
            _check_cancel(cancel_event)
            result.pex_result = batch_translate_pex(
                scan.pex_files,
                lang=lang, backend=backend, backend_label=backend_label,
                game=game, glossary=glossary, skip_translated=skip_translated,
                output_dir=output_dir, no_cache=no_cache,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # MCM
    if scan.has_mcm and scan.mcm_directory:
        if on_progress:
            on_progress("scan", 0, 0, _msg(lang, "translating_mcm"))
        try:
            _check_cancel(cancel_event)
            result.mcm_result = batch_translate_mcm(
                scan.mcm_directory,
                lang=lang, backend=backend, backend_label=backend_label,
                game=game, glossary=glossary, skip_translated=skip_translated,
                output_dir=output_dir, no_cache=no_cache,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # Aggregate
    for r in (result.esp_result, result.pex_result, result.mcm_result):
        if r:
            result.total_success += r.success_count
            result.total_errors += r.error_count

    result.elapsed_seconds = _time.monotonic() - t0
    return result
