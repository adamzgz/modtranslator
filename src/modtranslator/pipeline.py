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
import time as _time
import warnings
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modtranslator.core.mc_jar import JarScanResult
    from modtranslator.core.pex_parser import PexFile

# ── Re-exports from _pipeline_helpers ──
from modtranslator._pipeline_helpers import (  # noqa: F401
    _build_dedup_map,
    _prepare_file,
    _setup_glossary,
    _translate_chunks,
    _writeback_file,
    clear_cache,
    create_backend,
    get_cache_info,
    resolve_glossary_paths,
    scan_directory,
    scan_file,
)

# ── Re-exports from _pipeline_types ──
from modtranslator._pipeline_types import (  # noqa: F401
    _LARGE_FILE_THRESHOLD,
    _PIPELINE_MESSAGES,
    BatchAllResult,
    BatchResult,
    CancelledError,
    GameChoice,
    ProgressCallback,
    ScanResult,
    _check_cancel,
    _FileContext,
    _msg,
)
from modtranslator.backends.base import TranslationBackend
from modtranslator.core.constants import Game
from modtranslator.core.io_utils import atomic_write

# Type aliases for extracted MCM/MC phase helpers
_McmTxtEntry = tuple[Path, Path, list[str], list[int], list[str]]
_McmJsonEntry = tuple[Path, Path, dict[str, object], list[str], list[str]]
# (config_path, out_path, data, json-paths, texts, indent)
_McmConfigEntry = tuple[
    Path, Path, dict[str, object], list[list[object]], list[str], str
]
_JarModEntry = tuple[str, str, dict[str, str], dict[str, str], dict[str, str], str, str]
# (en_us_path, target_path, en_entries, existing, to_translate, indent)
_KubejsEntry = tuple[Path, Path, dict[str, str], dict[str, str], dict[str, str], str]


# ═══════════════════════════════════════════════════════════════════
# Extracted phase helpers (MCM / Minecraft)
# ═══════════════════════════════════════════════════════════════════


# MCM Interface/translations language naming (long + FO4 short codes).
_MCM_LANG_NAMES = {
    "ES": "spanish", "FR": "french", "DE": "german", "IT": "italian",
    "PT": "portuguese", "RU": "russian", "PL": "polish", "CS": "czech",
    "JA": "japanese", "ZH": "chinese",
}
_MCM_LANG_SHORT = {
    "ES": "es", "FR": "fr", "DE": "de", "IT": "it",
    "PT": "ptbr", "RU": "ru", "PL": "pl", "CS": "cs",
    "JA": "ja", "ZH": "cn",
}

# FO4 MCM config.json keys whose string values are visible UI text (translatable).
_MCM_CONFIG_TEXT_KEYS = frozenset(
    {"text", "help", "displayName", "pageDisplayName"}
)


def _parse_mcm_english_txt(raw: bytes) -> tuple[list[str], list[int], list[str]]:
    """Parse an MCM Interface/translations english txt: return (lines, indices, texts)."""
    if raw[:2] == b'\xff\xfe':
        text = raw.decode('utf-16-le')
        if text and text[0] == '﻿':
            text = text[1:]
    else:
        text = raw.decode('utf-8', errors='replace')

    lines = text.splitlines(keepends=True)
    indices: list[int] = []
    texts: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip('\r\n')
        if not stripped or stripped.startswith(';'):
            continue
        if '\t' in stripped and stripped.startswith('$'):
            parts = stripped.split('\t', 1)
            if len(parts) == 2:
                value = parts[1].strip()
                if value and any(c.isalpha() for c in value):
                    indices.append(i)
                    texts.append(value)
    return lines, indices, texts


def _walk_mcm_config(
    node: object,
    path: list[object],
    out_paths: list[list[object]],
    out_texts: list[str],
) -> None:
    """Recursively collect visible-text json-paths from an MCM config.json tree.

    Only string values under _MCM_CONFIG_TEXT_KEYS are collected; $-prefixed
    values (translation keys) and empty strings are skipped.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _MCM_CONFIG_TEXT_KEYS and isinstance(value, str):
                if value.strip() and not value.startswith("$"):
                    out_paths.append(path + [key])
                    out_texts.append(value)
            else:
                _walk_mcm_config(value, path + [key], out_paths, out_texts)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk_mcm_config(item, path + [i], out_paths, out_texts)


def mcm_config_pointer(path: list[object]) -> str:
    """Render a json-path list as an abbreviated pointer, e.g. pages[0].content[1].help."""
    parts: list[str] = []
    for elem in path:
        if isinstance(elem, int):
            parts.append(f"[{elem}]")
        else:
            parts.append(f".{elem}" if parts else str(elem))
    return "".join(parts)


def _collect_mcm_config_targets(
    directory: Path,
    output_dir: Path | None,
) -> list[_McmConfigEntry]:
    """MCM phase 1: discover MCM/Config/*/config.json translatable text."""
    config_root = directory / "MCM" / "Config"
    if not config_root.is_dir():
        return []

    from modtranslator.core.mc_lang_parser import detect_indent

    entries: list[_McmConfigEntry] = []
    for config_path in sorted(config_root.glob("*/config.json")):
        try:
            raw = config_path.read_text(encoding="utf-8-sig")
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        paths: list[list[object]] = []
        texts: list[str] = []
        _walk_mcm_config(data, [], paths, texts)
        if not texts:
            continue
        indent = detect_indent(raw) or "\t"
        if output_dir is not None:
            try:
                out_path = output_dir / config_path.relative_to(directory)
            except ValueError:
                out_path = output_dir / config_path.name
        else:
            out_path = config_path
        entries.append((config_path, out_path, data, paths, texts, indent))
    return entries


def _write_mcm_config(
    out_path: Path,
    data: dict[str, object],
    paths: list[list[object]],
    orig_texts: list[str],
    translations: dict[str, str],
    indent: str,
) -> None:
    """Write a translated MCM config.json (UTF-8, preserving indentation)."""
    for path, orig_text in zip(paths, orig_texts, strict=True):
        if orig_text not in translations:
            continue
        node: object = data
        for elem in path[:-1]:
            node = node[elem]  # type: ignore[index]
        node[path[-1]] = translations[orig_text]  # type: ignore[index]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_indent: str | None = indent if indent else None
    json_bytes = (
        json.dumps(data, indent=json_indent, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    atomic_write(out_path, json_bytes)


def _collect_mcm_archive_txts(
    archives: list[Path],
    directory: Path,
    lang: str,
    output_dir: Path | None,
) -> tuple[list[_McmTxtEntry], list[str], list[tuple[str, str]]]:
    """Extract Interface/translations english txt from .bsa/.ba2 → loose txt targets.

    Loose files win: an archived txt is skipped when the matching loose english
    source or the loose target already exists. Returns (entries, source_ids,
    archive_errors); source_ids[i] is the ``<archive>::<internal>`` origin of
    entries[i] (for the export/apply exchange format).
    """
    from modtranslator.core.bsa_archive import ArchiveError, iter_translatable_assets

    target_name = _MCM_LANG_NAMES.get(lang.upper(), lang.lower())
    target_short = _MCM_LANG_SHORT.get(lang.upper(), lang.lower())
    translations_dir = directory / "Interface" / "translations"
    out_base = output_dir if output_dir is not None else translations_dir

    entries: list[_McmTxtEntry] = []
    source_ids: list[str] = []
    errors: list[tuple[str, str]] = []
    for arc in archives:
        try:
            for internal, data in iter_translatable_assets(arc):
                if not internal.endswith(".txt"):
                    continue
                base = internal.rsplit("/", 1)[-1]
                if not _re.search(r'_(?:english|en)\.txt$', base, _re.IGNORECASE):
                    continue
                uses_short = bool(_re.search(r'_en\.txt$', base, _re.IGNORECASE))
                stem = _re.sub(r'(?i)_(?:english|en)\.txt$', '', base)
                out_name = (
                    f"{stem}_{target_short}.txt" if uses_short
                    else f"{stem}_{target_name}.txt"
                )
                if (translations_dir / base).exists() or (
                    translations_dir / out_name
                ).exists():
                    continue  # loose wins
                lines, indices, texts = _parse_mcm_english_txt(data)
                if texts:
                    entries.append((arc, out_base / out_name, lines, indices, texts))
                    source_ids.append(f"{arc}::{internal}")
        except ArchiveError as exc:
            errors.append((arc.name, str(exc)))
    return entries, source_ids, errors


def _collect_pex_archive_entries(
    archives: list[Path],
    directory: Path,
    output_dir: Path | None,
) -> tuple[list[tuple[Path, str, PexFile, dict[int, str]]], list[tuple[str, str]]]:
    """Extract scripts/*.pex from .bsa/.ba2 → parsed loose pex targets.

    Returns (entries, archive_errors) where each entry is
    (loose_target_path, source_id, PexFile, translatable). Loose files win:
    an archived pex is skipped when the matching loose pex already exists.
    """
    from modtranslator.core.bsa_archive import ArchiveError, iter_translatable_assets
    from modtranslator.core.pex_parser import parse_pex

    out_base = output_dir if output_dir is not None else directory
    entries: list[tuple[Path, str, PexFile, dict[int, str]]] = []
    errors: list[tuple[str, str]] = []
    for arc in archives:
        try:
            for internal, data in iter_translatable_assets(arc):
                if not internal.endswith(".pex"):
                    continue
                sub = internal.split("scripts/", 1)[-1]
                if (directory / "Scripts" / sub).exists():
                    continue  # loose wins
                try:
                    pex = parse_pex(data)
                    translatable = pex.get_translatable_strings()
                except Exception:
                    continue
                if translatable:
                    entries.append((
                        out_base / "Scripts" / sub,
                        f"{arc}::{internal}",
                        pex,
                        translatable,
                    ))
        except ArchiveError as exc:
            errors.append((arc.name, str(exc)))
    return entries, errors


def _collect_mcm_targets(
    directory: Path,
    lang: str,
    output_dir: Path | None,
    cancel_event: Event | None = None,
) -> tuple[list[_McmTxtEntry], list[_McmJsonEntry], list[_McmConfigEntry]]:
    """MCM phase 1: discover translations txt + McmRecorder json + config.json targets."""
    target_lang_name = _MCM_LANG_NAMES.get(lang.upper(), lang.lower())
    target_lang_short = _MCM_LANG_SHORT.get(lang.upper(), lang.lower())

    translations_dir = directory / "Interface" / "translations"
    mcm_recorder_dir = directory / "McmRecorder"

    mcm_files: list[_McmTxtEntry] = []
    mcm_jsons: list[_McmJsonEntry] = []

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
                lines, translatable_indices, translatable_texts = (
                    _parse_mcm_english_txt(eng_path.read_bytes())
                )
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

    mcm_configs = _collect_mcm_config_targets(directory, output_dir)

    return mcm_files, mcm_jsons, mcm_configs


def _write_mcm_txt(
    out_path: Path,
    lines: list[str],
    indices: list[int],
    orig_texts: list[str],
    translations: dict[str, str],
) -> None:
    """Write a translated MCM txt file (UTF-16-LE with BOM)."""
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
    atomic_write(out_path, b'\xff\xfe' + content.encode('utf-16-le'))


def _write_mcm_json(
    out_path: Path,
    data: dict[str, object],
    keys: list[str],
    orig_texts: list[str],
    translations: dict[str, str],
) -> None:
    """Write a translated McmRecorder json file (UTF-8)."""
    for key, orig_text in zip(keys, orig_texts, strict=True):
        if orig_text in translations:
            data[key] = translations[orig_text]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write(out_path, json_bytes)


def _collect_jar_mod_entries(
    scan: JarScanResult, target_locale: str
) -> list[_JarModEntry]:
    """MC phase 1 per-JAR: parse lang entries + filter_translatable (no lang-detect)."""
    from modtranslator.core.mc_lang_parser import (
        filter_translatable,
        parse_json_lang,
        parse_legacy_lang,
    )

    mod_entries: list[_JarModEntry] = []
    for entry in scan.lang_entries:
        # Parse en_us
        if entry.format == "json":
            en_entries = parse_json_lang(entry.en_us_content)
        else:
            en_entries = parse_legacy_lang(entry.en_us_content)

        # Parse existing target
        existing: dict[str, str] = {}
        if entry.target_content is not None:
            if entry.format == "json":
                existing = parse_json_lang(entry.target_content)
            else:
                existing = parse_legacy_lang(entry.target_content)

        # Filter to translatable keys
        to_translate = filter_translatable(en_entries, existing)

        # Determine target zip path
        lang_dir = "/".join(entry.en_us_path.split("/")[:-1])
        if entry.target_path is not None:
            target_zip = entry.target_path
        elif entry.format == "json":
            target_zip = f"{lang_dir}/{target_locale}.json"
        else:
            target_zip = f"{lang_dir}/{target_locale}.lang"

        mod_entries.append((
            entry.en_us_path,
            target_zip,
            en_entries,
            existing,
            to_translate,
            entry.format,
            entry.indent,
        ))
    return mod_entries


def _build_mod_lang_files(
    mod_entries: list[_JarModEntry],
    translations: dict[str, str],
) -> dict[str, bytes]:
    """Build {internal_zip_path: content_bytes} lang files for a JAR's mod entries."""
    from modtranslator.core.mc_lang_parser import (
        dump_json_lang,
        dump_legacy_lang,
        merge_translations,
    )

    new_files: dict[str, bytes] = {}

    for (
        _en_us_path, target_zip, en_entries, existing,
        to_translate_keys, fmt, indent,
    ) in mod_entries:
        # Build translations for this mod's keys
        mod_translations = {
            k: translations[v]
            for k, v in to_translate_keys.items()
            if v in translations
        }

        # Merge with existing translations
        merged = merge_translations(en_entries, existing, mod_translations)

        # Serialize
        content = (
            dump_json_lang(merged, indent) if fmt == "json"
            else dump_legacy_lang(merged)
        )

        new_files[target_zip] = content.encode("utf-8")

    return new_files


def _write_jar_translations(
    jar_path: Path,
    out_path: Path | None,
    mod_entries: list[_JarModEntry],
    translations: dict[str, str],
) -> bool:
    """MC phase 3 per-JAR: conservative merge + rebuild. Returns True if a JAR was written."""
    from modtranslator.core.mc_jar import rebuild_jar_with_lang

    new_files = _build_mod_lang_files(mod_entries, translations)

    if new_files:
        rebuild_jar_with_lang(jar_path, new_files, output_path=out_path)
        return True
    return False


def _write_resource_pack(
    resource_pack: Path,
    pack_format: int,
    lang: str,
    lang_files: dict[str, bytes],
) -> None:
    """Write a resource pack zip: pack.mcmeta + assets/<modid>/lang/<locale>.json overrides."""
    import os
    import zipfile

    mcmeta = json.dumps(
        {
            "pack": {
                "pack_format": pack_format,
                "description": f"modtranslator {lang} overlay",
            }
        },
        indent=2,
    )

    resource_pack.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resource_pack.with_suffix(resource_pack.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pack.mcmeta", mcmeta)
            for zip_path, content in lang_files.items():
                zf.writestr(zip_path, content)
        os.replace(tmp_path, resource_pack)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _collect_kubejs_entries(
    files: list[Path], target_locale: str
) -> list[_KubejsEntry]:
    """Parse loose KubeJS en_us lang files + filter_translatable (no lang-detect)."""
    from modtranslator.core.mc_lang_parser import (
        detect_indent,
        filter_translatable,
        parse_json_lang,
    )

    entries: list[_KubejsEntry] = []
    for en_path in files:
        try:
            content = en_path.read_text(encoding="utf-8-sig")
            en_entries = parse_json_lang(content)
        except Exception:
            continue
        indent = detect_indent(content)
        target = en_path.parent / f"{target_locale}.json"
        existing: dict[str, str] = {}
        if target.exists():
            try:
                existing = parse_json_lang(target.read_text(encoding="utf-8-sig"))
            except Exception:
                existing = {}
        to_translate = filter_translatable(en_entries, existing)
        entries.append((en_path, target, en_entries, existing, to_translate, indent))
    return entries


def _write_kubejs_translations(
    entries: list[_KubejsEntry],
    translations: dict[str, str],
) -> int:
    """Write loose KubeJS target lang files (conservative merge). Returns files written."""
    from modtranslator.core.mc_lang_parser import dump_json_lang, merge_translations

    written = 0
    for _en_path, target, en_entries, existing, to_translate_keys, indent in entries:
        mod_translations = {
            k: translations[v]
            for k, v in to_translate_keys.items()
            if v in translations
        }
        merged = merge_translations(en_entries, existing, mod_translations)
        content = dump_json_lang(merged, indent)
        atomic_write(target, content.encode("utf-8"))
        written += 1
    return written


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
                    # Protect [bracket] game-mechanic tags first
                    from modtranslator._pipeline_helpers import _protect_brackets_batch
                    texts, ctx.bracket_mappings = _protect_brackets_batch(texts)
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
    directory: Path | None = None,
    archives: list[Path] | None = None,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate a list of .pex Papyrus script files.

    When *archives* (.bsa/.ba2) and *directory* are given, translatable .pex
    inside the archives are extracted, translated and written as *loose* files
    under ``<directory>/Scripts/`` (or ``<output_dir>/Scripts/``). A loose .pex
    already present in the source wins and its archived copy is skipped.
    """
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

        # Archive (.bsa/.ba2) pex → loose targets
        archive_entries: list[tuple[Path, PexFile, dict[int, str]]] = []
        if archives and directory is not None:
            entries, arc_errors = _collect_pex_archive_entries(
                archives, directory, output_dir
            )
            for arc_name, msg in arc_errors:
                result.error_count += 1
                result.errors.append((arc_name, msg))
            for target, _source_id, pex_obj, translatable in entries:
                if skip_translated and translatable:
                    from modtranslator.translation.lang_detect import should_translate
                    translatable = {
                        idx: text for idx, text in translatable.items()
                        if should_translate(
                            text, lang, glossary_terms, glossary_source_terms
                        )
                    }
                if not translatable:
                    continue
                archive_entries.append((target, pex_obj, translatable))
                for text in translatable.values():
                    if text not in text_to_index:
                        text_to_index[text] = len(unique_texts)
                        unique_texts.append(text)

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

        # Write archive-extracted pex as loose files
        for target, pex_obj, translatable in archive_entries:
            _check_cancel(cancel_event)
            try:
                modified = False
                for idx, original_text in translatable.items():
                    if original_text in translations:
                        pex_obj.string_table[idx] = translations[original_text]
                        modified = True
                if modified:
                    save_pex(pex_obj, target)
                    result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append((target.name, str(e)))

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
    archives: list[Path] | None = None,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate MCM translations txt, MCM Recorder JSON and FO4 config.json.

    When *archives* (.bsa/.ba2) are given, Interface/translations english txt
    inside them are extracted and written as *loose* translated files (a loose
    file already present in the source wins).
    """
    from modtranslator.translation.cache import TranslationCache

    translations_dir = directory / "Interface" / "translations"
    mcm_recorder_dir = directory / "McmRecorder"
    mcm_config_dir = directory / "MCM" / "Config"

    if (
        not translations_dir.is_dir()
        and not mcm_recorder_dir.is_dir()
        and not mcm_config_dir.is_dir()
        and not archives
    ):
        raise FileNotFoundError(
            f"No Interface/translations/, McmRecorder/ or MCM/Config/ in {directory}"
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
        mcm_files, mcm_jsons, mcm_configs = _collect_mcm_targets(
            directory, lang, output_dir, cancel_event
        )

        # Archive (.bsa/.ba2) translations txt → loose targets
        if archives:
            arc_txt_entries, _arc_ids, arc_errors = _collect_mcm_archive_txts(
                archives, directory, lang, output_dir
            )
            mcm_files.extend(arc_txt_entries)
            for arc_name, msg in arc_errors:
                result.error_count += 1
                result.errors.append((arc_name, msg))

        total_files = len(mcm_files) + len(mcm_jsons) + len(mcm_configs)
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
        for _cfg in mcm_configs:
            for t in _cfg[4]:
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
                _write_mcm_txt(out_path, lines, indices, orig_texts, translations)
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
                _write_mcm_json(out_json, data, keys, orig_texts, translations)
                result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append((json_path.name, str(e)))
            write_idx += 1
            if on_progress:
                on_progress("write", write_idx, total_files, json_path.name)

        for config_path, out_cfg, data, paths, orig_texts, indent in mcm_configs:
            _check_cancel(cancel_event)
            try:
                _write_mcm_config(out_cfg, data, paths, orig_texts, translations, indent)
                result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append((config_path.name, str(e)))
            write_idx += 1
            if on_progress:
                on_progress("write", write_idx, total_files, config_path.name)

    except CancelledError:
        result.errors.append(("", "Cancelled by user"))

    result.elapsed_seconds = _time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════
# Public API: Minecraft JAR batch translation
# ═══════════════════════════════════════════════════════════════════


def batch_translate_mc(
    files: list[Path],
    *,
    lang: str = "ES",
    backend: TranslationBackend,
    backend_label: str,
    glossary: Path | None = None,
    skip_translated: bool = True,
    output_dir: Path | None = None,
    no_cache: bool = False,
    kubejs_files: list[Path] | None = None,
    resource_pack: Path | None = None,
    pack_format: int = 34,
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate Minecraft mod JAR files (and loose KubeJS lang files).

    For each JAR:
    1. Scan for en_us.json (or en_US.lang) inside assets/*/lang/
    2. Load existing target translation if present
    3. Calculate missing keys (en_us - existing_target)
    4. Filter: only strings, non-empty, alphabetic, not comment keys
    5. Protect format specifiers (%s, %1$s) and color codes (§a)
    6. Translate with deduplication across all JARs
    7. Restore protections
    8. Merge: existing_target + new translations
    9. Rebuild JAR with updated/new lang file

    When *resource_pack* is set, JARs are NOT rewritten: all lang overrides are
    accumulated into a single resource pack zip (signed JARs are included too).
    Loose KubeJS lang files are always written next to their en_us.json.
    """
    from modtranslator.core.mc_jar import scan_jar
    from modtranslator.core.mc_lang_parser import (
        FORMAT_SPEC_RE,
        MC_LANG_MAP,
        SECTION_SIGN_RE,
    )
    from modtranslator.translation.cache import TranslationCache

    result = BatchResult()
    t0 = _time.monotonic()

    target_locale = MC_LANG_MAP.get(lang.upper(), "es_es")
    use_resource_pack = resource_pack is not None

    try:
        gloss, glossary_terms, glossary_source_terms = _setup_glossary(
            glossary, lang, GameChoice.minecraft, Game.FALLOUT3,  # Game doesn't matter
        )

        cache = None if no_cache else TranslationCache()

        # ── Phase 1: Scan JARs and collect translatable strings ──

        # Per-JAR state for Phase 3: (jar_path, output_path, per-mod entries)
        jar_entries: list[tuple[Path, Path | None, list[_JarModEntry]]] = []

        unique_texts: list[str] = []
        text_to_index: dict[str, int] = {}
        signed_count = 0

        for i, jar_path in enumerate(files):
            _check_cancel(cancel_event)

            scan = scan_jar(jar_path, target_locale, include_signed=use_resource_pack)

            if scan.error:
                result.error_count += 1
                result.errors.append((jar_path.name, scan.error))
                if on_progress:
                    on_progress("prepare", i + 1, len(files), jar_path.name)
                continue

            if scan.is_signed and not use_resource_pack:
                signed_count += 1
                result.skip_count += 1
                if on_progress:
                    on_progress("prepare", i + 1, len(files), f"{jar_path.name} (signed)")
                continue

            if not scan.lang_entries:
                result.skip_count += 1
                if on_progress:
                    on_progress("prepare", i + 1, len(files), jar_path.name)
                continue

            out_path = (output_dir / jar_path.name) if output_dir else None
            mod_entries: list[_JarModEntry] = []

            for (
                en_us_path, target_zip, en_entries, existing,
                to_translate, fmt, indent,
            ) in _collect_jar_mod_entries(scan, target_locale):
                # Skip translated strings (language detection)
                if skip_translated and to_translate:
                    from modtranslator.translation.lang_detect import should_translate
                    to_translate = {
                        k: v for k, v in to_translate.items()
                        if should_translate(v, lang, glossary_terms, glossary_source_terms)
                    }

                if not to_translate:
                    continue

                # Collect unique texts for deduplication
                for text in to_translate.values():
                    if text not in text_to_index:
                        text_to_index[text] = len(unique_texts)
                        unique_texts.append(text)

                mod_entries.append((
                    en_us_path,
                    target_zip,
                    en_entries,
                    existing,
                    to_translate,
                    fmt,
                    indent,
                ))

            if mod_entries:
                jar_entries.append((jar_path, out_path, mod_entries))

            if on_progress:
                on_progress("prepare", i + 1, len(files), jar_path.name)

        # Loose KubeJS lang files (always written to their kubejs/assets/ location)
        kubejs_entries: list[_KubejsEntry] = []
        for (
            en_path, target, en_entries, existing, to_translate, indent
        ) in _collect_kubejs_entries(kubejs_files or [], target_locale):
            if skip_translated and to_translate:
                from modtranslator.translation.lang_detect import should_translate
                to_translate = {
                    k: v for k, v in to_translate.items()
                    if should_translate(v, lang, glossary_terms, glossary_source_terms)
                }
            if not to_translate:
                continue
            for text in to_translate.values():
                if text not in text_to_index:
                    text_to_index[text] = len(unique_texts)
                    unique_texts.append(text)
            kubejs_entries.append(
                (en_path, target, en_entries, existing, to_translate, indent)
            )

        result.total_strings = len(unique_texts)

        if not unique_texts:
            result.skip_count = len(files) - result.error_count
            result.elapsed_seconds = _time.monotonic() - t0
            return result

        # Cache check
        cached: dict[str, str] = {}
        if cache is not None:
            cached = cache.get_batch(unique_texts, lang)

        to_translate_texts = [t for t in unique_texts if t not in cached]

        # ── Protect format specifiers and color codes ──

        fmt_mappings: list[dict[str, str]] = []
        protected_texts: list[str] = []

        for text in to_translate_texts:
            mapping: dict[str, str] = {}
            protected = text

            # Protect format specifiers (%s, %1$s, %d, etc.)
            fmt_matches = FORMAT_SPEC_RE.findall(protected)
            for j, match in enumerate(fmt_matches):
                placeholder = f"FMx{j}"
                mapping[placeholder] = match
                protected = protected.replace(match, placeholder, 1)

            # Protect section sign codes (§a, §6, etc.)
            sc_matches = SECTION_SIGN_RE.findall(protected)
            for j, match in enumerate(sc_matches):
                placeholder = f"SSx{j}"
                mapping[placeholder] = match
                protected = protected.replace(match, placeholder, 1)

            fmt_mappings.append(mapping)
            protected_texts.append(protected)

        # Protect glossary terms
        gloss_mappings: list[dict[str, str]] | None = None
        if gloss and protected_texts:
            protected_texts, gloss_mappings = gloss.protect_batch(protected_texts)  # type: ignore[union-attr]

        # Protect target-language words
        lang_mappings: list[dict[str, str]] | None = None
        if protected_texts:
            from modtranslator.translation.target_protect import protect_target_batch
            protected_texts, lang_mappings = protect_target_batch(protected_texts, lang)

        # ── Phase 2: Translate ──

        translated: list[str] = []
        if protected_texts:
            _check_cancel(cancel_event)
            translated, chunk_errors = _translate_chunks(
                protected_texts, backend, lang,
                on_progress=on_progress, cancel_event=cancel_event,
            )
            for err in chunk_errors:
                result.errors.append(("batch", err))

            # Restore target-language words
            if lang_mappings is not None:
                from modtranslator.translation.target_protect import restore_target_batch
                translated = restore_target_batch(translated, lang_mappings)

            # Restore glossary terms
            if gloss and gloss_mappings:
                translated = gloss.restore_batch(translated, gloss_mappings)  # type: ignore[union-attr]

        # Restore format specifiers and color codes
        restored: list[str] = []
        for text, mapping in zip(translated, fmt_mappings, strict=True):
            for placeholder, original in mapping.items():
                text = text.replace(placeholder, original)
            restored.append(text)
        translated = restored

        # Validate: if format specifier count doesn't match, keep original
        validated: list[str] = []
        for orig, trans in zip(to_translate_texts, translated, strict=True):
            orig_fmts = FORMAT_SPEC_RE.findall(orig)
            trans_fmts = FORMAT_SPEC_RE.findall(trans)
            if len(orig_fmts) != len(trans_fmts):
                validated.append(orig)  # fallback to original
            else:
                validated.append(trans)
        translated = validated

        # Build translation map
        translations: dict[str, str] = dict(cached)
        for orig, trans in zip(to_translate_texts, translated, strict=True):
            translations[orig] = trans

        # Cache new translations
        if cache is not None:
            try:
                if translated:
                    cache_entries = [
                        (orig, lang, orig, translations[orig])
                        for orig in to_translate_texts
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

        # ── Phase 3: Write ──

        total_jars = len(jar_entries)
        resource_pack_files: dict[str, bytes] = {}
        for i, (jar_path, out_path, mod_entries) in enumerate(jar_entries):
            _check_cancel(cancel_event)
            try:
                if use_resource_pack:
                    resource_pack_files.update(
                        _build_mod_lang_files(mod_entries, translations)
                    )
                    result.success_count += 1
                elif _write_jar_translations(
                    jar_path, out_path, mod_entries, translations
                ):
                    result.success_count += 1

            except PermissionError:
                result.error_count += 1
                result.errors.append((
                    jar_path.name,
                    "Permission denied — close Minecraft/launcher and retry",
                ))
            except Exception as e:
                result.error_count += 1
                result.errors.append((jar_path.name, str(e)))

            if on_progress:
                on_progress("write", i + 1, total_jars, jar_path.name)

        # Write the resource pack zip (JARs are left untouched)
        if use_resource_pack and resource_pack_files:
            _write_resource_pack(
                resource_pack, pack_format, lang, resource_pack_files  # type: ignore[arg-type]
            )

        # Write loose KubeJS lang files (always to filesystem, never the pack)
        for entry in kubejs_entries:
            _check_cancel(cancel_event)
            try:
                if _write_kubejs_translations([entry], translations):
                    result.success_count += 1
            except Exception as e:
                result.error_count += 1
                result.errors.append((entry[0].name, str(e)))

    except CancelledError:
        result.errors.append(("", "Cancelled by user"))

    result.elapsed_seconds = _time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════
# Public API: FTB Quests (SNBT lang) batch translation
# ═══════════════════════════════════════════════════════════════════


# Regex for FTB Quests ampersand colour/format codes: &7, &r, &l, ...
_FTBQ_AMP_RE = _re.compile(r"&[0-9a-fk-or]", _re.IGNORECASE)


def batch_translate_ftbq(
    quests_dir: Path,
    *,
    lang: str = "ES",
    backend: TranslationBackend,
    backend_label: str,
    glossary: Path | None = None,
    skip_translated: bool = True,
    no_cache: bool = False,
    source_locale: str = "en_us",
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate FTB Quests SNBT lang files (single-file or folder layout)."""
    from modtranslator._pipeline_helpers import ftbq_source_files, ftbq_target_file
    from modtranslator.core.ftbq_lang import dump_snbt_lang, parse_snbt_lang
    from modtranslator.core.mc_lang_parser import (
        FORMAT_SPEC_RE,
        MC_LANG_MAP,
        SECTION_SIGN_RE,
    )
    from modtranslator.translation.cache import TranslationCache

    result = BatchResult()
    t0 = _time.monotonic()

    target_locale = MC_LANG_MAP.get(lang.upper(), "es_es")

    try:
        gloss, glossary_terms, glossary_source_terms = _setup_glossary(
            glossary, lang, GameChoice.minecraft, Game.FALLOUT3,  # Game doesn't matter
        )

        cache = None if no_cache else TranslationCache()

        source_files = ftbq_source_files(quests_dir, source_locale)

        # ── Phase 1: Parse + collect ──
        # Per-file state: (source_file, parsed_data)
        file_states: list[tuple[Path, dict[str, str | list[str]]]] = []
        unique_texts: list[str] = []
        text_to_index: dict[str, int] = {}

        for sf in source_files:
            _check_cancel(cancel_event)
            try:
                data = parse_snbt_lang(sf.read_text(encoding="utf-8"))
            except Exception as e:
                result.error_count += 1
                result.errors.append((sf.name, str(e)))
                continue
            file_states.append((sf, data))
            for value in data.values():
                items = value if isinstance(value, list) else [value]
                for text in items:
                    if skip_translated:
                        from modtranslator.translation.lang_detect import should_translate
                        if not should_translate(
                            text, lang, glossary_terms, glossary_source_terms
                        ):
                            continue
                    if text not in text_to_index:
                        text_to_index[text] = len(unique_texts)
                        unique_texts.append(text)

        result.total_strings = len(unique_texts)

        if not unique_texts:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

        # Cache check
        cached: dict[str, str] = {}
        if cache is not None:
            cached = cache.get_batch(unique_texts, lang)

        to_translate_texts = [t for t in unique_texts if t not in cached]

        # ── Protect format specifiers, section-sign codes and &-codes ──
        fmt_mappings: list[dict[str, str]] = []
        protected_texts: list[str] = []
        for text in to_translate_texts:
            mapping: dict[str, str] = {}
            protected = text
            for j, match in enumerate(FORMAT_SPEC_RE.findall(protected)):
                placeholder = f"FMx{j}"
                mapping[placeholder] = match
                protected = protected.replace(match, placeholder, 1)
            for j, match in enumerate(SECTION_SIGN_RE.findall(protected)):
                placeholder = f"SSx{j}"
                mapping[placeholder] = match
                protected = protected.replace(match, placeholder, 1)
            for j, match in enumerate(_FTBQ_AMP_RE.findall(protected)):
                placeholder = f"AMx{j}"
                mapping[placeholder] = match
                protected = protected.replace(match, placeholder, 1)
            fmt_mappings.append(mapping)
            protected_texts.append(protected)

        # Protect glossary terms
        gloss_mappings: list[dict[str, str]] | None = None
        if gloss and protected_texts:
            protected_texts, gloss_mappings = gloss.protect_batch(protected_texts)  # type: ignore[attr-defined]

        # Protect target-language words
        lang_mappings: list[dict[str, str]] | None = None
        if protected_texts:
            from modtranslator.translation.target_protect import protect_target_batch
            protected_texts, lang_mappings = protect_target_batch(protected_texts, lang)

        # ── Phase 2: Translate ──
        translated: list[str] = []
        if protected_texts:
            _check_cancel(cancel_event)
            translated, chunk_errors = _translate_chunks(
                protected_texts, backend, lang,
                on_progress=on_progress, cancel_event=cancel_event,
            )
            for err in chunk_errors:
                result.errors.append(("batch", err))

            if lang_mappings is not None:
                from modtranslator.translation.target_protect import restore_target_batch
                translated = restore_target_batch(translated, lang_mappings)

            if gloss and gloss_mappings:
                translated = gloss.restore_batch(translated, gloss_mappings)  # type: ignore[attr-defined]

        # Restore format/section/&-codes
        restored: list[str] = []
        for text, mapping in zip(translated, fmt_mappings, strict=True):
            for placeholder, original in mapping.items():
                text = text.replace(placeholder, original)
            restored.append(text)
        translated = restored

        # Validate: keep original if a code count check fails
        validated: list[str] = []
        for orig, trans in zip(to_translate_texts, translated, strict=True):
            if (
                len(FORMAT_SPEC_RE.findall(orig)) != len(FORMAT_SPEC_RE.findall(trans))
                or len(SECTION_SIGN_RE.findall(orig)) != len(SECTION_SIGN_RE.findall(trans))
                or len(_FTBQ_AMP_RE.findall(trans)) < len(_FTBQ_AMP_RE.findall(orig))
            ):
                validated.append(orig)
            else:
                validated.append(trans)
        translated = validated

        # Build translation map
        translations: dict[str, str] = dict(cached)
        for orig, trans in zip(to_translate_texts, translated, strict=True):
            translations[orig] = trans

        # Cache new translations
        if cache is not None:
            try:
                cache_entries = [
                    (orig, lang, orig, translations[orig])
                    for orig in to_translate_texts
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

        # ── Phase 3: Write target SNBT files ──
        total = len(file_states)
        for i, (sf, data) in enumerate(file_states):
            _check_cancel(cancel_event)
            new_data: dict[str, str | list[str]] = {}
            applied = 0
            for key, value in data.items():
                if isinstance(value, list):
                    new_list: list[str] = []
                    for item in value:
                        t = translations.get(item)
                        if t is not None and t != item:
                            new_list.append(t)
                            applied += 1
                        else:
                            new_list.append(item)
                    new_data[key] = new_list
                else:
                    t = translations.get(value)
                    if t is not None and t != value:
                        new_data[key] = t
                        applied += 1
                    else:
                        new_data[key] = value
            if applied:
                try:
                    target = ftbq_target_file(sf, source_locale, target_locale)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write(target, dump_snbt_lang(new_data).encode("utf-8"))
                    result.success_count += 1
                except Exception as e:
                    result.error_count += 1
                    result.errors.append((sf.name, str(e)))
            else:
                result.skip_count += 1
            if on_progress:
                on_progress("write", i + 1, total, sf.name)

    except CancelledError:
        result.errors.append(("", "Cancelled by user"))

    result.elapsed_seconds = _time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════
# Public API: Auto-detect and translate everything
# ═══════════════════════════════════════════════════════════════════


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
        if scan.mc_files:
            found_parts.append(f"{len(scan.mc_files)} MC JARs")
        if scan.kubejs_lang_files:
            found_parts.append(f"{len(scan.kubejs_lang_files)} KubeJS lang")
        if scan.ftbq_quests_dir is not None:
            found_parts.append("FTB Quests")
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
    if scan.pex_files or scan.archives:
        if on_progress:
            on_progress("scan", 0, 0, _msg(lang, "translating_pex", n=len(scan.pex_files)))
        try:
            _check_cancel(cancel_event)
            result.pex_result = batch_translate_pex(
                scan.pex_files,
                lang=lang, backend=backend, backend_label=backend_label,
                game=game, glossary=glossary, skip_translated=skip_translated,
                output_dir=output_dir, no_cache=no_cache,
                directory=directory, archives=scan.archives,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # MCM
    if (scan.has_mcm and scan.mcm_directory) or scan.archives:
        if on_progress:
            on_progress("scan", 0, 0, _msg(lang, "translating_mcm"))
        try:
            _check_cancel(cancel_event)
            result.mcm_result = batch_translate_mcm(
                scan.mcm_directory or directory,
                lang=lang, backend=backend, backend_label=backend_label,
                game=game, glossary=glossary, skip_translated=skip_translated,
                output_dir=output_dir, no_cache=no_cache,
                archives=scan.archives,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # Minecraft JARs and loose KubeJS lang files
    if scan.mc_files or scan.kubejs_lang_files:
        if on_progress:
            on_progress("scan", 0, 0, _msg(lang, "translating_mc", n=len(scan.mc_files)))
        try:
            _check_cancel(cancel_event)
            result.mc_result = batch_translate_mc(
                scan.mc_files,
                lang=lang, backend=backend, backend_label=backend_label,
                glossary=glossary, skip_translated=skip_translated,
                output_dir=output_dir, no_cache=no_cache,
                kubejs_files=scan.kubejs_lang_files,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # FTB Quests SNBT lang
    if scan.ftbq_quests_dir is not None:
        try:
            _check_cancel(cancel_event)
            result.ftbq_result = batch_translate_ftbq(
                scan.ftbq_quests_dir,
                lang=lang, backend=backend, backend_label=backend_label,
                glossary=glossary, skip_translated=skip_translated,
                no_cache=no_cache,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # Aggregate
    for r in (
        result.esp_result, result.pex_result, result.mcm_result,
        result.mc_result, result.ftbq_result,
    ):
        if r:
            result.total_success += r.success_count
            result.total_errors += r.error_count

    result.elapsed_seconds = _time.monotonic() - t0
    return result
