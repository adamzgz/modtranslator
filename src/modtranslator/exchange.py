"""Export/apply translation strings to/from a portable JSON exchange format.

Bridges the internal pipeline (parsers, extractors, patchers, cache) with an
LLM-agent translation flow: `export_path()` dumps raw source strings to JSON,
`apply_path()` writes an edited translations JSON back into the source files.

No text protection (Qx/Gx/Cx/FMx placeholders) is applied — text is exported
raw and specifiers are validated on apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from modtranslator import __version__
from modtranslator._pipeline_helpers import (
    _setup_glossary,
    ftbq_source_files,
    ftbq_target_file,
    patch_scpt_translations,
    resolve_glossary_paths,
    scan_directory,
)
from modtranslator._pipeline_types import GameChoice
from modtranslator.core.constants import Game, encoding_for_lang
from modtranslator.core.mc_lang_parser import (
    FORMAT_SPEC_RE,
    MC_LANG_MAP,
    SECTION_SIGN_RE,
)
from modtranslator.pipeline import (
    _build_mod_lang_files,
    _collect_jar_mod_entries,
    _collect_kubejs_entries,
    _collect_mcm_archive_txts,
    _collect_mcm_targets,
    _collect_pex_archive_entries,
    _write_jar_translations,
    _write_kubejs_translations,
    _write_mcm_config,
    _write_mcm_json,
    _write_mcm_txt,
    _write_resource_pack,
    mcm_config_pointer,
)


class FormatChoice(str, Enum):
    """User-facing content-format selection."""

    esp = "esp"
    pex = "pex"
    mcm = "mcm"
    mc = "mc"
    ftbq = "ftbq"
    auto = "auto"


# Regex for FTB Quests ampersand colour/format codes: &7, &r, &l, ...
_AMP_CODE_RE = re.compile(r"&[0-9a-fk-or]", re.IGNORECASE)


@dataclass
class ExportResult:
    """Result of exporting strings to the exchange JSON format."""

    doc: dict[str, Any]
    files_scanned: int = 0
    total_entries: int = 0
    cache_skipped: int = 0


@dataclass
class ApplyResult:
    """Result of applying a translations JSON back into source files."""

    files_written: int = 0
    strings_applied: int = 0
    unmatched_keys: list[str] = field(default_factory=list)
    spec_rejected: list[str] = field(default_factory=list)


# ── Shared helpers ──


def _detect_file_format(path: Path) -> FormatChoice:
    """Detect the content format of a single file from its suffix."""
    suffix = path.suffix.lower()
    if suffix in (".esp", ".esm", ".esl"):
        return FormatChoice.esp
    if suffix == ".pex":
        return FormatChoice.pex
    if suffix == ".jar":
        return FormatChoice.mc
    if suffix == ".txt":
        return FormatChoice.mcm
    return FormatChoice.esp


def _mcm_base_dir(path: Path) -> Path:
    """Resolve the game-data base directory for an MCM .txt file."""
    if (
        path.parent.name.lower() == "translations"
        and path.parent.parent.name.lower() == "interface"
    ):
        return path.parent.parent.parent
    return path.parent


def _resolve_targets(
    path: Path, fmt: FormatChoice
) -> tuple[
    list[Path], list[Path], Path | None, list[Path], list[Path], Path | None,
    Path | None, list[Path],
]:
    """Return (esp, pex, mcm_dir, mc, kubejs, ftbq_dir, arc_dir, archives)."""
    esp_files: list[Path] = []
    pex_files: list[Path] = []
    mcm_dir: Path | None = None
    mc_files: list[Path] = []
    kubejs_files: list[Path] = []
    ftbq_dir: Path | None = None
    arc_dir: Path | None = None
    archives: list[Path] = []

    if path.is_dir():
        scan = scan_directory(path)
        if fmt in (FormatChoice.auto, FormatChoice.esp):
            esp_files = list(scan.esp_files)
        if fmt in (FormatChoice.auto, FormatChoice.pex):
            pex_files = list(scan.pex_files)
        if fmt in (FormatChoice.auto, FormatChoice.mcm) and scan.has_mcm:
            mcm_dir = scan.mcm_directory
        if fmt in (FormatChoice.auto, FormatChoice.mc):
            mc_files = list(scan.mc_files)
            kubejs_files = list(scan.kubejs_lang_files)
        if fmt in (FormatChoice.auto, FormatChoice.ftbq):
            ftbq_dir = scan.ftbq_quests_dir
        if fmt in (FormatChoice.auto, FormatChoice.pex, FormatChoice.mcm):
            arc_dir = path
            archives = list(scan.archives)
    else:
        detected = _detect_file_format(path) if fmt == FormatChoice.auto else fmt
        if detected == FormatChoice.esp:
            esp_files = [path]
        elif detected == FormatChoice.pex:
            pex_files = [path]
        elif detected == FormatChoice.mc:
            mc_files = [path]
        elif detected == FormatChoice.mcm:
            mcm_dir = _mcm_base_dir(path)

    return (
        esp_files, pex_files, mcm_dir, mc_files, kubejs_files, ftbq_dir,
        arc_dir, archives,
    )


def _resolve_game(game: GameChoice, esp_files: list[Path], has_mc: bool) -> GameChoice:
    """Resolve the effective GameChoice for glossary resolution and metadata."""
    if game != GameChoice.auto:
        return game
    if esp_files:
        from modtranslator.core.plugin import load_plugin

        try:
            detected = load_plugin(esp_files[0]).game
        except Exception:
            return GameChoice.fo3
        if detected == Game.SKYRIM:
            return GameChoice.skyrim
        if detected == Game.FALLOUT4:
            return GameChoice.fo4
        return GameChoice.fo3
    if has_mc:
        return GameChoice.minecraft
    return GameChoice.fo3


def _extract_esp_strings(plugin: Any, stem: str) -> tuple[list[Any], list[Any]]:
    """Extract normal + SCPT translatable strings from a plugin. Returns (strings, scpt_records)."""
    from modtranslator.translation.extractor import (
        extract_scpt_data,
        extract_strings,
        scpt_to_translatable,
    )

    strings = extract_strings(plugin)
    for s in strings:
        s.source_file = stem
    scpt_records = extract_scpt_data(plugin)
    if scpt_records:
        strings.extend(scpt_to_translatable(scpt_records, stem))
    return strings, scpt_records


def _specs_ok(orig: str, trad: str) -> bool:
    """True if format specifiers and section-sign codes counts match."""
    return (
        len(FORMAT_SPEC_RE.findall(orig)) == len(FORMAT_SPEC_RE.findall(trad))
        and len(SECTION_SIGN_RE.findall(orig)) == len(SECTION_SIGN_RE.findall(trad))
    )


def _ftbq_specs_ok(orig: str, trad: str) -> bool:
    """Like _specs_ok, plus reject when the count of &-codes decreases."""
    if not _specs_ok(orig, trad):
        return False
    # Conservative: only reject if the translation dropped &-codes (avoids
    # false positives when a translation legitimately introduces a literal '&').
    return len(_AMP_CODE_RE.findall(trad)) >= len(_AMP_CODE_RE.findall(orig))


# ── Export ──


def export_path(
    path: Path,
    *,
    lang: str = "ES",
    game: GameChoice = GameChoice.auto,
    fmt: FormatChoice = FormatChoice.auto,
    glossary: Path | None = None,
    use_cache: bool = True,
    skip_translated: bool = True,
) -> ExportResult:
    """Export raw translatable strings from *path* to the exchange JSON document."""
    from modtranslator.core.pex_parser import load_pex
    from modtranslator.core.plugin import load_plugin
    from modtranslator.translation.cache import TranslationCache
    from modtranslator.translation.lang_detect import should_translate

    path = Path(path)
    (
        esp_files, pex_files, mcm_dir, mc_files, kubejs_files, ftbq_dir,
        arc_dir, archives,
    ) = _resolve_targets(path, fmt)

    gc = _resolve_game(game, esp_files, bool(mc_files or kubejs_files or ftbq_dir))
    _gloss, glossary_terms, glossary_source_terms = _setup_glossary(
        glossary, lang, gc, Game.FALLOUT3,
    )
    gloss_paths = resolve_glossary_paths(glossary, lang, gc, Game.FALLOUT3)

    cache = TranslationCache() if use_cache else None
    files_out: list[dict[str, Any]] = []
    files_scanned = 0
    total_entries = 0
    cache_skipped = 0

    def _keep(text: str) -> bool:
        if not skip_translated:
            return True
        return should_translate(text, lang, glossary_terms, glossary_source_terms)

    try:
        # ── ESP/ESM ──
        for esp_path in esp_files:
            files_scanned += 1
            plugin = load_plugin(esp_path)
            strings, _scpt = _extract_esp_strings(plugin, esp_path.stem)
            strings = [s for s in strings if _keep(s.original_text)]
            cached: dict[str, str] = {}
            if cache is not None:
                cached = cache.get_batch([s.key for s in strings], lang)
            entries: list[dict[str, Any]] = []
            for s in strings:
                if s.key in cached:
                    cache_skipped += 1
                    continue
                entry: dict[str, Any] = {
                    "key": s.key,
                    "text": s.original_text,
                    "record": s.record_type.decode("ascii"),
                    "type": s.subrecord_type.decode("ascii"),
                }
                if s.editor_id:
                    entry["edid"] = s.editor_id
                entries.append(entry)
            if entries:
                files_out.append(
                    {"path": str(esp_path), "format": "esp", "entries": entries}
                )
                total_entries += len(entries)

        # ── PEX ──
        for pex_path in pex_files:
            files_scanned += 1
            pex = load_pex(pex_path)
            translatable = pex.get_translatable_strings()
            texts = [t for t in translatable.values() if _keep(t)]
            cached = cache.get_batch(texts, lang) if cache is not None else {}
            entries = []
            for text in texts:
                if text in cached:
                    cache_skipped += 1
                    continue
                entries.append({"key": text, "text": text})
            if entries:
                files_out.append(
                    {"path": str(pex_path), "format": "pex", "entries": entries}
                )
                total_entries += len(entries)

        # ── PEX inside .bsa/.ba2 archives ──
        if archives and arc_dir is not None:
            arc_pex, _arc_errs = _collect_pex_archive_entries(archives, arc_dir, None)
            for _target, source_id, _pex, translatable in arc_pex:
                files_scanned += 1
                texts = [t for t in translatable.values() if _keep(t)]
                cached = cache.get_batch(texts, lang) if cache is not None else {}
                entries = []
                seen_pex: set[str] = set()
                for text in texts:
                    if text in seen_pex:
                        continue
                    seen_pex.add(text)
                    if text in cached:
                        cache_skipped += 1
                        continue
                    entries.append({"key": text, "text": text})
                if entries:
                    files_out.append(
                        {"path": source_id, "format": "pex", "entries": entries}
                    )
                    total_entries += len(entries)

        # ── MCM ──
        if mcm_dir is not None:
            mcm_files, mcm_jsons, mcm_configs = _collect_mcm_targets(mcm_dir, lang, None)
            for eng_path, _out_path, lines, indices, mtexts in mcm_files:
                files_scanned += 1
                entries = []
                for idx, text in zip(indices, mtexts, strict=True):
                    if not _keep(text):
                        continue
                    if cache is not None and cache.get(text, lang) is not None:
                        cache_skipped += 1
                        continue
                    mcm_key = lines[idx].rstrip("\r\n").split("\t", 1)[0]
                    entries.append({"key": text, "text": text, "id": mcm_key})
                if entries:
                    files_out.append(
                        {"path": str(eng_path), "format": "mcm", "entries": entries}
                    )
                    total_entries += len(entries)
            for json_path, _out_json, _data, keys, jtexts in mcm_jsons:
                files_scanned += 1
                entries = []
                for jkey, text in zip(keys, jtexts, strict=True):
                    if not _keep(text):
                        continue
                    if cache is not None and cache.get(text, lang) is not None:
                        cache_skipped += 1
                        continue
                    entries.append({"key": text, "text": text, "id": jkey})
                if entries:
                    files_out.append(
                        {"path": str(json_path), "format": "mcm", "entries": entries}
                    )
                    total_entries += len(entries)
            for cfg_path, _out_cfg, _data, cpaths, ctexts, _indent in mcm_configs:
                files_scanned += 1
                try:
                    rel = cfg_path.relative_to(mcm_dir).as_posix()
                except ValueError:
                    rel = cfg_path.name
                entries = []
                for cpath, text in zip(cpaths, ctexts, strict=True):
                    if not _keep(text):
                        continue
                    if cache is not None and cache.get(text, lang) is not None:
                        cache_skipped += 1
                        continue
                    entry_id = f"{rel}:{mcm_config_pointer(cpath)}"
                    entries.append({"key": text, "text": text, "id": entry_id})
                if entries:
                    files_out.append(
                        {"path": str(cfg_path), "format": "mcm", "entries": entries}
                    )
                    total_entries += len(entries)

        # ── MCM translations txt inside .bsa/.ba2 archives ──
        if archives and arc_dir is not None:
            arc_txts, arc_ids, _txt_errs = _collect_mcm_archive_txts(
                archives, arc_dir, lang, None
            )
            for (_arc, _out_path, lines, indices, mtexts), src_id in zip(
                arc_txts, arc_ids, strict=True
            ):
                files_scanned += 1
                entries = []
                for idx, text in zip(indices, mtexts, strict=True):
                    if not _keep(text):
                        continue
                    if cache is not None and cache.get(text, lang) is not None:
                        cache_skipped += 1
                        continue
                    mcm_key = lines[idx].rstrip("\r\n").split("\t", 1)[0]
                    entries.append({"key": text, "text": text, "id": mcm_key})
                if entries:
                    files_out.append(
                        {"path": src_id, "format": "mcm", "entries": entries}
                    )
                    total_entries += len(entries)

        # ── Minecraft JARs ──
        if mc_files or kubejs_files:
            from modtranslator.core.mc_jar import scan_jar

            target_locale = MC_LANG_MAP.get(lang.upper(), "es_es")
            for jar_path in mc_files:
                files_scanned += 1
                scan = scan_jar(jar_path, target_locale)
                if scan.error or scan.is_signed or not scan.lang_entries:
                    continue
                pairs: list[tuple[str, str]] = []
                for mod_entry in _collect_jar_mod_entries(scan, target_locale):
                    to_translate = mod_entry[4]
                    for lang_key, text in to_translate.items():
                        pairs.append((lang_key, text))
                texts = [t for _k, t in pairs if _keep(t)]
                cached = cache.get_batch(texts, lang) if cache is not None else {}
                entries = []
                seen: set[str] = set()
                for lang_key, text in pairs:
                    if not _keep(text) or text in seen:
                        continue
                    if text in cached:
                        seen.add(text)
                        cache_skipped += 1
                        continue
                    seen.add(text)
                    entries.append({"key": text, "text": text, "id": lang_key})
                if entries:
                    files_out.append(
                        {"path": str(jar_path), "format": "mc", "entries": entries}
                    )
                    total_entries += len(entries)

            # Loose KubeJS lang files
            for kb in _collect_kubejs_entries(kubejs_files, target_locale):
                files_scanned += 1
                pairs = list(kb[4].items())  # (lang_key, text)
                texts = [t for _k, t in pairs if _keep(t)]
                cached = cache.get_batch(texts, lang) if cache is not None else {}
                entries = []
                seen = set()
                for lang_key, text in pairs:
                    if not _keep(text) or text in seen:
                        continue
                    if text in cached:
                        seen.add(text)
                        cache_skipped += 1
                        continue
                    seen.add(text)
                    entries.append({"key": text, "text": text, "id": lang_key})
                if entries:
                    files_out.append(
                        {"path": str(kb[0]), "format": "mc", "entries": entries}
                    )
                    total_entries += len(entries)

        # ── FTB Quests SNBT lang ──
        if ftbq_dir is not None:
            from modtranslator.core.ftbq_lang import parse_snbt_lang

            ftbq_seen: set[str] = set()
            for sf in ftbq_source_files(ftbq_dir):
                files_scanned += 1
                try:
                    data = parse_snbt_lang(sf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                rel = sf.relative_to(ftbq_dir).as_posix()
                entries = []
                for snbt_key, value in data.items():
                    items = value if isinstance(value, list) else [value]
                    for idx, text in enumerate(items):
                        if not _keep(text) or text in ftbq_seen:
                            continue
                        if cache is not None and cache.get(text, lang) is not None:
                            ftbq_seen.add(text)
                            cache_skipped += 1
                            continue
                        ftbq_seen.add(text)
                        entry_id = (
                            f"{rel}:{snbt_key}:{idx}"
                            if isinstance(value, list)
                            else f"{rel}:{snbt_key}"
                        )
                        entries.append({"key": text, "text": text, "id": entry_id})
                if entries:
                    files_out.append(
                        {"path": str(sf), "format": "ftbq", "entries": entries}
                    )
                    total_entries += len(entries)
    finally:
        if cache is not None:
            cache.close()

    doc: dict[str, Any] = {
        "version": 1,
        "generator": f"modtranslator {__version__}",
        "created": datetime.now(timezone.utc).isoformat(),
        "lang": lang,
        "game": gc.value,
        "glossaries": [str(p) for p in gloss_paths],
        "files": files_out,
    }

    return ExportResult(
        doc=doc,
        files_scanned=files_scanned,
        total_entries=total_entries,
        cache_skipped=cache_skipped,
    )


# ── Apply ──


def apply_path(
    path: Path,
    translations: dict[str, str],
    *,
    lang: str = "ES",
    game: GameChoice = GameChoice.auto,
    fmt: FormatChoice = FormatChoice.auto,
    output: Path | None = None,
    use_cache: bool = True,
    backend_label: str = "claude-agent",
    resource_pack: Path | None = None,
    pack_format: int = 34,
) -> ApplyResult:
    """Apply *translations* (key → translated text) back into source files."""
    from modtranslator.core.pex_parser import load_pex, save_pex
    from modtranslator.core.plugin import load_plugin, save_plugin
    from modtranslator.core.string_table import ISO_TO_FULL_LANGUAGE
    from modtranslator.translation.cache import TranslationCache
    from modtranslator.translation.patcher import apply_translations

    path = Path(path)
    single = path.is_file()

    # Discard empty translations up front.
    trans = {k: v for k, v in translations.items() if v}

    (
        esp_files, pex_files, mcm_dir, mc_files, kubejs_files, ftbq_dir,
        arc_dir, archives,
    ) = _resolve_targets(path, fmt)
    target_locale = MC_LANG_MAP.get(lang.upper(), "es_es")
    use_resource_pack = resource_pack is not None

    result = ApplyResult()
    matched: set[str] = set()
    spec_seen: set[str] = set()
    cache_entries: list[tuple[str, str, str, str]] = []

    def _reject(key: str) -> None:
        if key not in spec_seen:
            spec_seen.add(key)
            result.spec_rejected.append(key)

    output_language = ISO_TO_FULL_LANGUAGE.get(lang.upper(), lang)

    # ── ESP/ESM ──
    for esp_path in esp_files:
        plugin = load_plugin(esp_path)
        stem = esp_path.stem
        strings, scpt_records = _extract_esp_strings(plugin, stem)
        plugin_trans: dict[str, str] = {}
        for s in strings:
            trad = trans.get(s.key)
            if trad is None:
                continue
            matched.add(s.key)
            if not _specs_ok(s.original_text, trad):
                _reject(s.key)
                continue
            plugin_trans[s.key] = trad
            cache_entries.append((s.key, lang, s.original_text, trad))
        normal_strings = [s for s in strings if s.record_type != b"SCPT"]
        patched = apply_translations(
            normal_strings,
            plugin_trans,
            encoding=encoding_for_lang(lang),
            string_tables=getattr(plugin, "string_tables", None),
        )
        if scpt_records:
            patched += patch_scpt_translations(scpt_records, plugin_trans, stem)
        if patched:
            out = output if (single and output is not None) else esp_path.with_stem(
                f"{stem}_{lang}"
            )
            save_plugin(plugin, out, output_language=output_language)
            result.files_written += 1
            result.strings_applied += patched

    # ── PEX ──
    for pex_path in pex_files:
        pex = load_pex(pex_path)
        translatable = pex.get_translatable_strings()
        applied = 0
        for idx, text in translatable.items():
            trad = trans.get(text)
            if trad is None:
                continue
            matched.add(text)
            if not _specs_ok(text, trad):
                _reject(text)
                continue
            pex.string_table[idx] = trad
            cache_entries.append((text, lang, text, trad))
            applied += 1
        if applied:
            out = output if (single and output is not None) else pex_path
            save_pex(pex, out)
            result.files_written += 1
            result.strings_applied += applied

    # ── PEX inside .bsa/.ba2 archives → loose ──
    if archives and arc_dir is not None:
        arc_pex, _arc_errs = _collect_pex_archive_entries(archives, arc_dir, None)
        for target, _source_id, pex, translatable in arc_pex:
            applied = 0
            for idx, text in translatable.items():
                trad = trans.get(text)
                if trad is None:
                    continue
                matched.add(text)
                if not _specs_ok(text, trad):
                    _reject(text)
                    continue
                pex.string_table[idx] = trad
                cache_entries.append((text, lang, text, trad))
                applied += 1
            if applied:
                save_pex(pex, target)
                result.files_written += 1
                result.strings_applied += applied

    # ── MCM ──
    if mcm_dir is not None:
        mcm_files, mcm_jsons, mcm_configs = _collect_mcm_targets(mcm_dir, lang, output)
        for _eng_path, out_path, lines, indices, mtexts in mcm_files:
            mod_trans: dict[str, str] = {}
            applied = 0
            for text in mtexts:
                trad = trans.get(text)
                if trad is None:
                    continue
                matched.add(text)
                if not _specs_ok(text, trad):
                    _reject(text)
                    continue
                mod_trans[text] = trad
                cache_entries.append((text, lang, text, trad))
                applied += 1
            if applied:
                _write_mcm_txt(out_path, lines, indices, mtexts, mod_trans)
                result.files_written += 1
                result.strings_applied += applied
        for _json_path, out_json, data, keys, jtexts in mcm_jsons:
            mod_trans = {}
            applied = 0
            for text in jtexts:
                trad = trans.get(text)
                if trad is None:
                    continue
                matched.add(text)
                if not _specs_ok(text, trad):
                    _reject(text)
                    continue
                mod_trans[text] = trad
                cache_entries.append((text, lang, text, trad))
                applied += 1
            if applied:
                _write_mcm_json(out_json, data, keys, jtexts, mod_trans)
                result.files_written += 1
                result.strings_applied += applied
        for _cfg_path, out_cfg, data, cpaths, ctexts, indent in mcm_configs:
            mod_trans = {}
            applied = 0
            for text in ctexts:
                trad = trans.get(text)
                if trad is None:
                    continue
                matched.add(text)
                if not _specs_ok(text, trad):
                    _reject(text)
                    continue
                mod_trans[text] = trad
                cache_entries.append((text, lang, text, trad))
                applied += 1
            if applied:
                _write_mcm_config(out_cfg, data, cpaths, ctexts, mod_trans, indent)
                result.files_written += 1
                result.strings_applied += applied

    # ── MCM translations txt inside .bsa/.ba2 archives → loose ──
    if archives and arc_dir is not None:
        arc_txts, _arc_ids, _txt_errs = _collect_mcm_archive_txts(
            archives, arc_dir, lang, None
        )
        for _arc, out_path, lines, indices, mtexts in arc_txts:
            mod_trans = {}
            applied = 0
            for text in mtexts:
                trad = trans.get(text)
                if trad is None:
                    continue
                matched.add(text)
                if not _specs_ok(text, trad):
                    _reject(text)
                    continue
                mod_trans[text] = trad
                cache_entries.append((text, lang, text, trad))
                applied += 1
            if applied:
                _write_mcm_txt(out_path, lines, indices, mtexts, mod_trans)
                result.files_written += 1
                result.strings_applied += applied

    # ── Minecraft JARs and loose KubeJS lang ──
    if mc_files or kubejs_files:
        from modtranslator.core.mc_jar import scan_jar

        resource_pack_files: dict[str, bytes] = {}
        for jar_path in mc_files:
            scan = scan_jar(
                jar_path, target_locale, include_signed=use_resource_pack
            )
            if scan.error or not scan.lang_entries:
                continue
            if scan.is_signed and not use_resource_pack:
                continue
            mod_entries = _collect_jar_mod_entries(scan, target_locale)
            jar_trans: dict[str, str] = {}
            applied = 0
            for entry in mod_entries:
                for text in entry[4].values():
                    trad = trans.get(text)
                    if trad is None:
                        continue
                    matched.add(text)
                    if not _specs_ok(text, trad):
                        _reject(text)
                        continue
                    if text not in jar_trans:
                        jar_trans[text] = trad
                        cache_entries.append((text, lang, text, trad))
                    applied += 1
            if applied:
                if use_resource_pack:
                    resource_pack_files.update(
                        _build_mod_lang_files(mod_entries, jar_trans)
                    )
                    result.strings_applied += applied
                else:
                    out_jar = output if (single and output is not None) else None
                    if _write_jar_translations(
                        jar_path, out_jar, mod_entries, jar_trans
                    ):
                        result.files_written += 1
                        result.strings_applied += applied

        # Loose KubeJS lang files (always written to their kubejs/assets/ location)
        for kb in _collect_kubejs_entries(kubejs_files, target_locale):
            kb_trans: dict[str, str] = {}
            applied = 0
            for text in kb[4].values():
                trad = trans.get(text)
                if trad is None:
                    continue
                matched.add(text)
                if not _specs_ok(text, trad):
                    _reject(text)
                    continue
                if text not in kb_trans:
                    kb_trans[text] = trad
                    cache_entries.append((text, lang, text, trad))
                applied += 1
            if applied:
                _write_kubejs_translations([kb], kb_trans)
                result.files_written += 1
                result.strings_applied += applied

        # Write the resource pack zip (JARs are left untouched)
        if use_resource_pack and resource_pack_files:
            _write_resource_pack(
                resource_pack, pack_format, lang, resource_pack_files  # type: ignore[arg-type]
            )
            result.files_written += 1

    # ── FTB Quests SNBT lang ──
    if ftbq_dir is not None:
        from modtranslator.core.ftbq_lang import dump_snbt_lang, parse_snbt_lang

        for sf in ftbq_source_files(ftbq_dir):
            try:
                snbt_data = parse_snbt_lang(sf.read_text(encoding="utf-8"))
            except Exception:
                continue
            new_data: dict[str, str | list[str]] = {}
            applied = 0
            for snbt_key, value in snbt_data.items():
                if isinstance(value, list):
                    new_list: list[str] = []
                    for item in value:
                        trad = trans.get(item)
                        if trad is None:
                            new_list.append(item)
                            continue
                        matched.add(item)
                        if not _ftbq_specs_ok(item, trad):
                            _reject(item)
                            new_list.append(item)
                            continue
                        new_list.append(trad)
                        cache_entries.append((item, lang, item, trad))
                        applied += 1
                    new_data[snbt_key] = new_list
                else:
                    trad = trans.get(value)
                    if trad is None:
                        new_data[snbt_key] = value
                        continue
                    matched.add(value)
                    if not _ftbq_specs_ok(value, trad):
                        _reject(value)
                        new_data[snbt_key] = value
                        continue
                    new_data[snbt_key] = trad
                    cache_entries.append((value, lang, value, trad))
                    applied += 1
            if applied:
                target = ftbq_target_file(sf, "en_us", target_locale)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(dump_snbt_lang(new_data), encoding="utf-8")
                result.files_written += 1
                result.strings_applied += applied

    result.unmatched_keys = [k for k in trans if k not in matched]

    if use_cache and cache_entries:
        cache = TranslationCache()
        try:
            cache.put_batch(cache_entries, backend=backend_label)
        finally:
            cache.close()

    return result
