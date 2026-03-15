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
                atomic_write(out_path, b'\xff\xfe' + content.encode('utf-16-le'))
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
                json_bytes = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                )
                atomic_write(out_json, json_bytes)
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
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> BatchResult:
    """Translate Minecraft mod JAR files.

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
    """
    from modtranslator.core.mc_jar import rebuild_jar_with_lang, scan_jar
    from modtranslator.core.mc_lang_parser import (
        FORMAT_SPEC_RE,
        MC_LANG_MAP,
        SECTION_SIGN_RE,
        dump_json_lang,
        dump_legacy_lang,
        filter_translatable,
        merge_translations,
        parse_json_lang,
        parse_legacy_lang,
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

        # ── Phase 1: Scan JARs and collect translatable strings ──

        # Per-JAR state for Phase 3
        jar_entries: list[tuple[
            Path,  # jar_path
            Path | None,  # output_path
            list[tuple[  # per-mod entries in this JAR
                str,  # en_us zip path
                str,  # target zip path
                dict[str, str],  # en_us entries
                dict[str, str],  # existing target entries
                dict[str, str],  # keys to translate {key: en_text}
                str,  # format ("json" or "lang")
                str,  # indent
            ]],
        ]] = []

        unique_texts: list[str] = []
        text_to_index: dict[str, int] = {}
        signed_count = 0

        for i, jar_path in enumerate(files):
            _check_cancel(cancel_event)

            scan = scan_jar(jar_path, target_locale)

            if scan.error:
                result.error_count += 1
                result.errors.append((jar_path.name, scan.error))
                if on_progress:
                    on_progress("prepare", i + 1, len(files), jar_path.name)
                continue

            if scan.is_signed:
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
            mod_entries: list[tuple[str, str, dict, dict, dict, str, str]] = []

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

            if mod_entries:
                jar_entries.append((jar_path, out_path, mod_entries))

            if on_progress:
                on_progress("prepare", i + 1, len(files), jar_path.name)

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

        # ── Phase 3: Write JARs ──

        total_jars = len(jar_entries)
        for i, (jar_path, out_path, mod_entries) in enumerate(jar_entries):
            _check_cancel(cancel_event)
            try:
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
                    if fmt == "json":
                        content = dump_json_lang(merged, indent)
                    else:
                        content = dump_legacy_lang(merged)

                    new_files[target_zip] = content.encode("utf-8")

                if new_files:
                    rebuild_jar_with_lang(jar_path, new_files, output_path=out_path)
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

    # Minecraft JARs
    if scan.mc_files:
        if on_progress:
            on_progress("scan", 0, 0, _msg(lang, "translating_mc", n=len(scan.mc_files)))
        try:
            _check_cancel(cancel_event)
            result.mc_result = batch_translate_mc(
                scan.mc_files,
                lang=lang, backend=backend, backend_label=backend_label,
                glossary=glossary, skip_translated=skip_translated,
                output_dir=output_dir, no_cache=no_cache,
                on_progress=on_progress, cancel_event=cancel_event,
            )
        except CancelledError:
            result.elapsed_seconds = _time.monotonic() - t0
            return result

    # Aggregate
    for r in (result.esp_result, result.pex_result, result.mcm_result, result.mc_result):
        if r:
            result.total_success += r.success_count
            result.total_errors += r.error_count

    result.elapsed_seconds = _time.monotonic() - t0
    return result
