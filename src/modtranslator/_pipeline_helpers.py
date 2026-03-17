"""Pipeline helper functions shared across batch operations.

Extracted from pipeline.py for maintainability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Event

from modtranslator._pipeline_types import (
    CancelledError,
    GameChoice,
    ProgressCallback,
    ScanResult,
    _check_cancel,
    _FileContext,
)
from modtranslator.backends.base import TranslationBackend
from modtranslator.core.constants import Game

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

    if effective == GameChoice.minecraft:
        candidates = [
            glossaries_dir / "minecraft" / f"minecraft_base_{lang_lower}.toml",
            glossaries_dir / "minecraft" / f"minecraft_{lang_lower}.toml",
        ]
        return [p for p in candidates if p.exists()]

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
    from modtranslator.translation.extractor import (
        extract_scpt_data,
        extract_strings,
        scpt_to_translatable,
    )

    ctx = _FileContext(file_path=file_path, output_path=output_path)

    try:
        plugin = load_plugin(file_path)
        ctx.plugin = plugin

        strings = extract_strings(plugin)
        file_stem = file_path.stem
        for s in strings:
            s.source_file = file_stem

        # Also extract SCPT bytecode strings
        scpt_records = extract_scpt_data(plugin)
        if scpt_records:
            ctx.scpt_data = scpt_records
            scpt_strings = scpt_to_translatable(scpt_records, file_stem)
            strings.extend(scpt_strings)

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
            texts, ctx.gloss_mappings = gloss.protect_batch(texts)  # type: ignore[attr-defined]

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
    from modtranslator.translation.extractor import (
        extract_scpt_data,
        extract_strings,
        scpt_to_translatable,
    )
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
            translated = gloss.restore_batch(translated, ctx.gloss_mappings)  # type: ignore[attr-defined]

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
            # Re-extract SCPT data for the reloaded plugin
            scpt_records = extract_scpt_data(ctx.plugin)
            if scpt_records:
                ctx.scpt_data = scpt_records
                scpt_strings = scpt_to_translatable(scpt_records, file_stem)
                ctx.all_strings.extend(scpt_strings)

        from modtranslator.core.constants import encoding_for_lang
        st = getattr(ctx.plugin, "string_tables", None)

        # Separate SCPT strings from normal strings for different patching paths
        normal_strings = [s for s in ctx.all_strings if s.record_type != b"SCPT"]
        patched = apply_translations(
            normal_strings, translations,
            encoding=encoding_for_lang(lang),
            string_tables=st,
        )

        # Patch SCPT bytecode strings
        if ctx.scpt_data:
            from modtranslator.core.scpt_parser import patch_scpt_record

            for sr in ctx.scpt_data:
                scpt_translations: dict[int, str] = {}
                for ss in sr.strings:
                    ts_key = (
                        f"{ctx.file_path.stem}:"
                        f"{sr.record.form_id:08X}:SCTX:{ss.scda_offset}"
                    )
                    if ts_key in translations:
                        scpt_translations[ss.scda_offset] = translations[ts_key]
                if scpt_translations:
                    patched += patch_scpt_record(sr.record, scpt_translations)

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


# ── Newline splitting helpers ──


def _split_newlines(
    texts: list[str],
) -> tuple[list[str], list[tuple[str, list[tuple[str, int | str]]]]]:
    """Split multi-line texts into individual lines for translation.

    Neural MT models strip newlines, so each line must be translated
    independently and reassembled afterwards.

    Returns (flat_texts, reassembly_map) where:
    - flat_texts: list of individual non-empty lines to translate
    - reassembly_map: per-original-text entries, each being either
      ("single", [(idx, flat_index)]) for single-line texts, or
      ("multi", [(action, data), ...]) for multi-line texts where
      action is "translate" (data=flat_index) or "keep" (data=original_line)
    """
    flat: list[str] = []
    rmap: list[tuple[str, list[tuple[str, int | str]]]] = []
    for text in texts:
        if "\n" not in text:
            rmap.append(("single", [("translate", len(flat))]))
            flat.append(text)
        else:
            lines = text.split("\n")
            entries: list[tuple[str, int | str]] = []
            for line in lines:
                if line.strip():
                    entries.append(("translate", len(flat)))
                    flat.append(line)
                else:
                    entries.append(("keep", line))
            rmap.append(("multi", entries))
    return flat, rmap


def _rejoin_newlines(
    translated_flat: list[str],
    rmap: list[tuple[str, list[tuple[str, int | str]]]],
) -> list[str]:
    """Reassemble multi-line texts from translated individual lines."""
    result: list[str] = []
    for kind, entries in rmap:
        if kind == "single":
            _, flat_idx = entries[0]
            result.append(translated_flat[flat_idx])  # type: ignore[index]
        else:
            lines: list[str] = []
            for action, data in entries:
                if action == "translate":
                    lines.append(translated_flat[data])  # type: ignore[index]
                else:
                    lines.append(data)  # type: ignore[arg-type]
            result.append("\n".join(lines))
    return result


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

    Multi-line texts are split into individual lines before translation
    to prevent neural MT from stripping newlines.

    Returns (translated_texts, chunk_errors).
    """
    # Split multi-line texts so each line is translated independently
    flat_texts, rmap = _split_newlines(texts)

    translated_flat: list[str] = []
    chunk_errors: list[str] = []
    total = len(texts)
    flat_total = len(flat_texts)
    for chunk_start in range(0, flat_total, chunk_size):
        _check_cancel(cancel_event)
        chunk = flat_texts[chunk_start:chunk_start + chunk_size]
        try:
            translated_flat.extend(backend.translate_batch(chunk, lang))
        except CancelledError:
            raise
        except Exception as e:
            chunk_errors.append(
                f"Chunk {chunk_start // chunk_size + 1}: {e}"
            )
            translated_flat.extend(chunk)  # fallback: original text
        if on_progress:
            # Map flat progress back to original text count
            progress = min(
                int(len(translated_flat) / max(flat_total, 1) * total), total,
            )
            on_progress(phase_name, progress, total, "")

    # Reassemble multi-line texts
    translated = _rejoin_newlines(translated_flat, rmap)
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


# ── Utility functions ──


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


def scan_directory(directory: Path) -> ScanResult:
    """Scan a directory for translatable content (ESP/ESM, PEX, MCM, Minecraft JARs)."""
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

    # Minecraft JAR files (direct children or mods/ subdirectory)
    jar_files = list(directory.glob("*.jar"))
    mods_dir = directory / "mods"
    if mods_dir.is_dir():
        jar_files.extend(mods_dir.glob("*.jar"))
    result.mc_files = sorted(jar_files)

    return result
