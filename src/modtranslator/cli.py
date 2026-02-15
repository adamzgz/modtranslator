"""CLI interface for modtranslator using Typer."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from modtranslator import __version__
from modtranslator.backends.base import TranslationBackend
from modtranslator.core.constants import Game

# Files larger than this are parsed twice (extract → free → translate → re-parse → patch)
# to avoid holding both the plugin (~2GB for Fallout3.esm) and the translation model
# (~300MB-2GB for CTranslate2) in memory simultaneously, which freezes the process.
_LARGE_FILE_THRESHOLD = 150 * 1024 * 1024  # 150 MB


class GameChoice(str, Enum):
    """User-facing game selection (separate from binary detection Game enum)."""
    auto = "auto"
    fo3 = "fo3"
    fnv = "fnv"
    skyrim = "skyrim"

app = typer.Typer(
    name="modtranslator",
    help="Automatic translator for Bethesda ESP/ESM mod files.",
    add_completion=False,
)
console = Console()

_verbose = False
_quiet = False


def _print(msg: str, *, verbose_only: bool = False) -> None:
    """Print respecting --verbose/--quiet flags. Errors bypass --quiet."""
    if _quiet:
        return
    if verbose_only and not _verbose:
        return
    console.print(msg)


def _resolve_glossary_paths(
    glossary: Path | None,
    lang: str,
    game: GameChoice,
    detected_game: Game,
) -> list[Path]:
    """Resolve which glossary files to load based on --glossary, --game, and detected game."""
    glossaries_dir = Path(__file__).resolve().parent.parent.parent / "glossaries"

    # Explicit --glossary overrides everything
    if glossary is not None:
        return [glossary] if glossary.exists() else []

    # Only auto-load for Spanish
    if lang.upper() != "ES":
        return []

    # Determine effective game
    if game != GameChoice.auto:
        effective = game
    elif detected_game == Game.SKYRIM:
        effective = GameChoice.skyrim
    else:
        effective = GameChoice.fo3

    # Resolve paths
    if effective == GameChoice.fo3:
        candidates = [
            glossaries_dir / "fallout_base_es.toml",
            glossaries_dir / "fallout3_es.toml",
        ]
    elif effective == GameChoice.fnv:
        candidates = [
            glossaries_dir / "fallout_base_es.toml",
            glossaries_dir / "falloutnv_es.toml",
        ]
    elif effective == GameChoice.skyrim:
        candidates = [
            glossaries_dir / "skyrim_base_es.toml",
        ]
    else:
        return []

    return [p for p in candidates if p.exists()]


def _create_backend(
    backend_name: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    device: str | None = None,
    use_dummy: bool = False,
) -> tuple[TranslationBackend, str]:
    """Create a translation backend instance.

    Returns:
        Tuple of (backend_instance, backend_label_for_report).
    """
    from modtranslator.backends.deepl import DeepLBackend
    from modtranslator.backends.dummy import DummyBackend

    if use_dummy:
        backend_name = "dummy"

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
            console.print(
                "[red]Error:[/red] DeepL API key required."
                " Use --api-key or set DEEPL_API_KEY."
            )
            raise typer.Exit(1)
        return DeepLBackend(api_key), "deepl"


@dataclass
class _FileContext:
    """Per-file state carried through the 3-phase parallel pipeline."""

    file_path: Path
    output_path: Path | None
    plugin: object | None = None  # PluginFile (avoid import at module level)
    all_strings: list = field(default_factory=list)  # list[TranslatableString]
    to_translate: list = field(default_factory=list)  # list[TranslatableString]
    to_translate_keys: list[str] = field(default_factory=list)
    to_translate_originals: list[str] = field(default_factory=list)
    cached: dict[str, str] = field(default_factory=dict)
    protected_texts: list[str] = field(default_factory=list)
    gloss_mappings: list[dict[str, str]] | None = None
    es_mappings: list[dict[str, str]] | None = None
    dedup_indices: list[int] = field(default_factory=list)
    translations: dict[str, str] = field(default_factory=dict)
    patched_count: int = 0
    status: str = "pending"  # pending|prepared|skipped|written|error
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
    """Phase 1 worker: parse, extract, filter, protect (no cache — cache is main-thread only).

    Thread-safe: all operations are stateless or read-only.
    Cache lookup is done in the main thread after Phase 1 (SQLite is not thread-safe).
    """
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

        # Filter already-translated strings
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

        # to_translate will be refined after cache check in main thread
        ctx.to_translate = list(strings)
        texts = [s.original_text for s in ctx.to_translate]

        # Protect glossary terms
        if gloss and texts:
            texts, ctx.gloss_mappings = gloss.protect_batch(texts)  # type: ignore[union-attr]

        # Protect Spanish words in mixed strings (only for ES target)
        if texts and lang.upper() == "ES":
            from modtranslator.translation.spanish_protect import protect_spanish_batch

            texts, ctx.es_mappings = protect_spanish_batch(texts)

        ctx.protected_texts = texts
        ctx.status = "prepared"

    except Exception as e:
        ctx.status = "error"
        ctx.error_message = str(e)

    return ctx


def _build_dedup_map(
    contexts: list[_FileContext],
) -> tuple[list[str], dict[str, int]]:
    """Build deduplicated text list from all prepared contexts.

    Returns (unique_texts, text_to_index). Also sets ctx.dedup_indices
    for each context to map back from unique results.
    """
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
    """Phase 3 worker: restore placeholders, patch, save.

    Thread-safe: each ctx owns its plugin/subrecords independently.
    """
    from modtranslator.core.plugin import load_plugin, save_plugin
    from modtranslator.translation.extractor import extract_strings
    from modtranslator.translation.patcher import apply_translations

    try:
        # Map dedup indices back to per-file translated texts
        if ctx.dedup_indices and translated_unique:
            translated = [translated_unique[i] for i in ctx.dedup_indices]
        else:
            translated = []

        # Restore Spanish words (inner layer, before glossary)
        if ctx.es_mappings is not None and translated:
            from modtranslator.translation.spanish_protect import restore_spanish_batch

            translated = restore_spanish_batch(translated, ctx.es_mappings)

        # Restore glossary terms (outer layer)
        if gloss and translated:
            translated = gloss.restore_batch(translated, ctx.gloss_mappings)  # type: ignore[union-attr]

        # Build translations dict: cached + newly translated
        translations: dict[str, str] = dict(ctx.cached)
        for key, t in zip(ctx.to_translate_keys, translated, strict=True):
            translations[key] = t

        ctx.translations = translations

        # Re-parse plugin if it was freed (large file memory optimization)
        if ctx.plugin is None:
            ctx.plugin = load_plugin(ctx.file_path)
            ctx.all_strings = extract_strings(ctx.plugin)
            file_stem = ctx.file_path.stem
            for s in ctx.all_strings:
                s.source_file = file_stem

        # Patch plugin in memory
        st = getattr(ctx.plugin, "string_tables", None)
        patched = apply_translations(ctx.all_strings, translations, string_tables=st)
        ctx.patched_count = patched

        # Write to disk
        if ctx.output_path is not None:
            save_plugin(ctx.plugin, ctx.output_path)  # type: ignore[arg-type]
        else:
            out = ctx.file_path.with_stem(f"{ctx.file_path.stem}_{lang}")
            save_plugin(ctx.plugin, out)  # type: ignore[arg-type]

        ctx.status = "written"

    except Exception as e:
        ctx.status = "error"
        ctx.error_message = str(e)

    return ctx


def _batch_translate_parallel(
    files: list[Path],
    *,
    lang: str,
    backend: TranslationBackend,
    backend_label: str,
    gloss: object | None,
    glossary_terms: set[str] | None,
    glossary_source_terms: set[str] | None,
    skip_translated: bool,
    output_dir: Path | None,
    no_cache: bool,
) -> tuple[int, int, int, list[tuple[str, str]]]:
    """Orchestrate the 3-phase sequential batch pipeline.

    All phases run sequentially. Parallel I/O in phases 1 and 3 was removed
    because Fallout 3 determines plugin load order from file timestamps (not
    plugins.txt). Parallel writes give multiple files the same NTFS timestamp
    (1-second resolution), scrambling the load order and causing crashes when a
    patch loads before its master. Sequential processing preserves ordered
    timestamps.  The performance cost is minimal (~10-15%) since Phase 2 (GPU
    translation) dominates at ~80% of total time and is always single-threaded.

    Returns (success_count, skip_count, error_count, errors_list).
    """
    import time as _time

    from modtranslator.translation.cache import TranslationCache

    cache = None if no_cache else TranslationCache()

    # Pre-initialize singletons before threading
    from modtranslator.translation.lang_detect import _load_spanish_dictionary
    _load_spanish_dictionary()

    # Build file list with output paths
    file_pairs: list[tuple[Path, Path | None]] = []
    for f in files:
        out = (output_dir / f.name) if output_dir is not None else None
        file_pairs.append((f, out))

    # ── Phase 1: Prepare sequentially (parse, extract, filter, protect) ──
    contexts: list[_FileContext] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Phase 1: Preparing files", total=len(files))

        for fp, out in file_pairs:
            ctx = _prepare_file(fp, out, lang, gloss, glossary_terms,
                                glossary_source_terms, skip_translated)
            contexts.append(ctx)
            progress.advance(task)

    # ── Cache check (main thread — SQLite is not thread-safe) ──
    if cache is not None:
        for ctx in contexts:
            if ctx.status != "prepared":
                continue
            keys = [s.key for s in ctx.all_strings]
            ctx.cached = cache.get_batch(keys, lang)
            if ctx.cached:
                # Rebuild to_translate and protected_texts excluding cached strings
                ctx.to_translate = [s for s in ctx.all_strings if s.key not in ctx.cached]
                texts = [s.original_text for s in ctx.to_translate]
                # Re-protect (texts changed after removing cached)
                if gloss and texts:
                    texts, ctx.gloss_mappings = gloss.protect_batch(texts)  # type: ignore[union-attr]
                else:
                    ctx.gloss_mappings = None
                if texts and lang.upper() == "ES":
                    from modtranslator.translation.spanish_protect import protect_spanish_batch
                    texts, ctx.es_mappings = protect_spanish_batch(texts)
                else:
                    ctx.es_mappings = None
                ctx.protected_texts = texts

    # ── Save keys/originals for all prepared contexts (needed for writeback + cache) ──
    for ctx in contexts:
        if ctx.status == "prepared":
            ctx.to_translate_keys = [s.key for s in ctx.to_translate]
            ctx.to_translate_originals = [s.original_text for s in ctx.to_translate]

    # ── Deduplication ──
    unique_texts, _ = _build_dedup_map(contexts)

    # ── Free large plugins to make room for translation model ──
    for ctx in contexts:
        if ctx.status != "prepared" or not ctx.protected_texts:
            continue
        if ctx.file_path.stat().st_size > _LARGE_FILE_THRESHOLD:
            ctx.plugin = None
            ctx.all_strings = []
            ctx.to_translate = []
            gc.collect()

    prepared_count = sum(1 for c in contexts if c.status == "prepared" and c.protected_texts)
    _print(
        f"Prepared {prepared_count} files, "
        f"{len(unique_texts)} unique strings to translate",
        verbose_only=True,
    )

    # ── Phase 2: Translate unique texts in chunks ──
    chunk_size = 500
    translated_unique: list[str] = []

    if unique_texts:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Phase 2: Translating",
                total=len(unique_texts),
            )
            _t0 = _time.monotonic()
            try:
                for chunk_start in range(0, len(unique_texts), chunk_size):
                    chunk = unique_texts[chunk_start:chunk_start + chunk_size]
                    translated_unique.extend(backend.translate_batch(chunk, lang))
                    progress.advance(task, len(chunk))
            except Exception as e:
                console.print(f"[red]Translation error:[/red] {e}")
                for ctx in contexts:
                    if ctx.status == "prepared":
                        ctx.status = "error"
                        ctx.error_message = f"Backend error: {e}"
                translated_unique = []
            _elapsed = _time.monotonic() - _t0

        if translated_unique and _elapsed > 0:
            n = len(translated_unique)
            rate = n / _elapsed
            _print(
                f"Translated {n} strings in"
                f" {_elapsed:.1f}s ({rate:.0f} strings/s)",
            )

    # ── Phase 3: Write back sequentially ──
    # Include files that are prepared (have texts to translate OR all cached)
    writable = [c for c in contexts if c.status == "prepared"]

    if writable:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Phase 3: Writing files", total=len(writable))

            for ctx in writable:
                _writeback_file(ctx, translated_unique, gloss, lang, backend_label)
                progress.advance(task)

    # ── Roundtrip skipped files when --output-dir is set ──
    # Skipped files (no translatable strings after filtering) still need to
    # be written to the output directory so it is self-contained.  Without
    # this, the game directory has a mix of freshly-written translated files
    # and stale originals, which can cause load-order or compatibility issues.
    skipped_with_output = [
        c for c in contexts
        if c.status == "skipped" and c.output_path is not None
    ]
    if skipped_with_output:
        from modtranslator.core.plugin import load_plugin, save_plugin

        for ctx in skipped_with_output:
            try:
                plugin = load_plugin(ctx.file_path)
                save_plugin(plugin, ctx.output_path)  # type: ignore[arg-type]
                ctx.status = "written"
            except Exception as e:
                ctx.status = "error"
                ctx.error_message = str(e)

    # ── Cache write-back (serialized) ──
    if cache is not None:
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
        cache.close()

    # ── Collect results ──
    success_count = 0
    skip_count = 0
    error_count = 0
    errors_list: list[tuple[str, str]] = []

    for ctx in contexts:
        if ctx.status == "written":
            success_count += 1
        elif ctx.status == "skipped":
            skip_count += 1
        elif ctx.status == "error":
            error_count += 1
            errors_list.append((ctx.file_path.name, ctx.error_message))
        elif ctx.status == "prepared":
            # Prepared but no unique_texts (all from cache) — still write
            # This shouldn't happen normally, but handle gracefully
            skip_count += 1

    return success_count, skip_count, error_count, errors_list


def version_callback(value: bool) -> None:
    if value:
        console.print(f"modtranslator {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show extra info (backend, timing).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only show errors.",
    ),
) -> None:
    """modtranslator: Translate Bethesda mod files (ESP/ESM) automatically."""
    global _verbose, _quiet
    _verbose = verbose
    _quiet = quiet


@app.command()
def translate(
    file: Path = typer.Argument(
        ..., help="Path to the ESP/ESM file to translate.",
    ),
    lang: str = typer.Option(
        "ES", "--lang", "-l",
        help="Target language code (e.g. ES, FR, DE).",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-k",
        envvar="DEEPL_API_KEY", help="DeepL API key.",
    ),
    backend_name: str = typer.Option(
        "deepl", "--backend", "-b",
        help="Backend: deepl, opus-mt, nllb, hybrid, dummy.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Model: base/tc-big (Opus-MT), 600M/1.3B (NLLB).",
    ),
    device: str | None = typer.Option(
        None, "--device",
        help="Device for Opus-MT/NLLB: auto, cpu, cuda.",
    ),
    glossary: Path | None = typer.Option(
        None, "--glossary", "-g",
        help="Path to glossary TOML file.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Output file path. Defaults to <name>_<lang>.esp.",
    ),
    report: Path | None = typer.Option(
        None, "--report", "-r",
        help="Save report to file (json/md/csv).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Extract and translate but don't write the file.",
    ),
    use_dummy: bool = typer.Option(
        False, "--dummy",
        help="Use dummy backend (shortcut for --backend dummy).",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable translation cache.",
    ),
    skip_translated: bool = typer.Option(
        True, "--skip-translated/--no-skip-translated",
        help="Skip strings already in target language.",
    ),
    game: GameChoice = typer.Option(
        GameChoice.auto, "--game",
        help="Game: fo3, fnv, skyrim, auto.",
    ),
) -> None:
    """Translate an ESP/ESM file."""
    _translate_file(
        file=file, lang=lang, api_key=api_key,
        backend_name=backend_name, model=model,
        device=device, glossary=glossary, output=output, report=report,
        dry_run=dry_run, use_dummy=use_dummy, no_cache=no_cache,
        skip_translated=skip_translated, game=game,
    )


def _translate_file(
    *,
    file: Path,
    lang: str = "ES",
    api_key: str | None = None,
    backend_name: str = "deepl",
    model: str | None = None,
    device: str | None = None,
    glossary: Path | None = None,
    output: Path | None = None,
    report: Path | None = None,
    dry_run: bool = False,
    use_dummy: bool = False,
    no_cache: bool = False,
    skip_translated: bool = False,
    game: GameChoice = GameChoice.auto,
    _backend_instance: TranslationBackend | None = None,
) -> None:
    """Internal translate logic. Accepts an optional pre-created backend."""
    from modtranslator.core.plugin import load_plugin, save_plugin
    from modtranslator.reporting.formatters import save_report
    from modtranslator.reporting.report import TranslationReport
    from modtranslator.translation.cache import TranslationCache
    from modtranslator.translation.extractor import extract_strings
    from modtranslator.translation.glossary import Glossary
    from modtranslator.translation.patcher import apply_translations

    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    rpt = TranslationReport(
        source_file=str(file),
        target_lang=lang,
        dry_run=dry_run,
    )

    # Resolve backend — reuse if provided, otherwise create new
    if _backend_instance is not None:
        backend = _backend_instance
        rpt.backend = type(backend).__name__
    else:
        backend, rpt.backend = _create_backend(
            backend_name,
            api_key=api_key,
            model=model,
            device=device,
            use_dummy=use_dummy,
        )

    _print(f"Backend: [cyan]{rpt.backend}[/cyan]", verbose_only=True)

    # Load plugin
    with console.status("Parsing plugin file..."):
        plugin = load_plugin(file)
        rpt.game_detected = plugin.game.name

    _print(f"Game detected: [cyan]{plugin.game.name}[/cyan]")

    # Extract strings
    with console.status("Extracting strings..."):
        strings = extract_strings(plugin)
        file_stem = file.stem
        for s in strings:
            s.source_file = file_stem
        rpt.total_strings_found = len(strings)

    _print(f"Found [green]{len(strings)}[/green] translatable strings")

    # Load glossary (before language filter so terms are available for should_translate)
    gloss = None
    gloss_paths = _resolve_glossary_paths(glossary, lang, game, plugin.game)
    if gloss_paths:
        gloss = Glossary.from_multiple_toml(gloss_paths)
        rpt.glossary_file = ", ".join(str(p) for p in gloss_paths)
        rpt.glossary_terms = len(gloss.terms)
        n_terms = len(gloss.terms)
        _print(
            f"Loaded glossary: [cyan]{n_terms}[/cyan] terms"
            f" from {len(gloss_paths)} file(s)",
        )

    # Filter out strings already in target language
    if skip_translated:
        from modtranslator.translation.lang_detect import should_translate

        glossary_terms: set[str] | None = None
        glossary_source_terms: set[str] | None = None
        if gloss:
            # Build set of lowercased glossary target values for exact-match skipping
            glossary_terms = {v.lower() for v in gloss.terms.values()}
            # Source terms that have a different translation — force-translate these
            # so short terms like "Dad" → "Papá" bypass the MIN_LENGTH filter
            glossary_source_terms = {
                k.lower() for k, v in gloss.terms.items()
                if k.lower() != v.lower()
            }

        before = len(strings)
        strings = [
            s for s in strings
            if should_translate(s.original_text, lang, glossary_terms, glossary_source_terms)
        ]
        skipped = before - len(strings)
        if skipped:
            _print(f"Skipped [yellow]{skipped}[/yellow] strings already in target language")

    if not strings:
        console.print("[yellow]No translatable strings found.[/yellow]")
        raise typer.Exit()

    # Check cache
    cache = None
    cached: dict[str, str] = {}
    if not no_cache:
        cache = TranslationCache()
        keys = [s.key for s in strings]
        cached = cache.get_batch(keys, lang)
        rpt.strings_from_cache = len(cached)
        if cached:
            _print(f"Cache hits: [green]{len(cached)}[/green]")

    # Determine which strings need translation
    to_translate = [s for s in strings if s.key not in cached]
    texts = [s.original_text for s in to_translate]

    # Protect glossary terms
    gloss_mappings: list[dict[str, str]] | None = None
    if gloss and texts:
        texts, gloss_mappings = gloss.protect_batch(texts)

    # Protect Spanish words in mixed strings (only for ES target)
    es_mappings: list[dict[str, str]] | None = None
    if texts and lang.upper() == "ES":
        from modtranslator.translation.spanish_protect import protect_spanish_batch

        texts, es_mappings = protect_spanish_batch(texts)

    # Memory optimization for large files: free plugin before loading translation model.
    # Fallout3.esm (~275MB on disk) expands to ~2GB in Python; CTranslate2 adds ~300MB-2GB.
    # Keeping both in memory simultaneously exhausts RAM and freezes the process.
    _plugin_freed = False
    to_translate_keys = [s.key for s in to_translate]
    to_translate_originals = [s.original_text for s in to_translate]
    if texts and file.stat().st_size > _LARGE_FILE_THRESHOLD:
        _plugin_freed = True
        _print("Large file: freeing plugin memory before translation...", verbose_only=True)
        del plugin, strings, to_translate
        gc.collect()

    # Translate
    translations: dict[str, str] = dict(cached)

    if texts:
        import time as _time

        chunk_size = 500
        translated: list[str] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Translating", total=len(texts))
            _t0 = _time.monotonic()
            try:
                for chunk_start in range(0, len(texts), chunk_size):
                    chunk = texts[chunk_start:chunk_start + chunk_size]
                    translated.extend(backend.translate_batch(chunk, lang))
                    progress.advance(task, len(chunk))
                _elapsed = _time.monotonic() - _t0
                rpt.strings_translated = len(translated)
            except Exception as e:
                console.print(f"[red]Translation error:[/red] {e}")
                rpt.errors.append(str(e))
                rpt.finish()
                if report:
                    save_report(rpt, report)
                raise typer.Exit(1) from e

        if _elapsed > 0:
            rate = len(translated) / _elapsed
            _print(
                f"Translated {len(translated)} strings in"
                f" {_elapsed:.1f}s ({rate:.0f} strings/s)",
            )

        # Restore Spanish words (Cx0 → originals) before glossary
        if es_mappings is not None:
            from modtranslator.translation.spanish_protect import restore_spanish_batch

            translated = restore_spanish_batch(translated, es_mappings)

        # Restore glossary terms
        if gloss:
            translated = gloss.restore_batch(translated, gloss_mappings)

        # Build translations dict and cache new entries
        cache_entries = []
        for key, orig, t in zip(to_translate_keys, to_translate_originals, translated, strict=True):
            translations[key] = t
            cache_entries.append((key, lang, orig, t))

        if cache and cache_entries:
            cache.put_batch(cache_entries, backend=rpt.backend)

    # Apply translations
    if not dry_run:
        # Re-parse plugin if it was freed (large file memory optimization)
        if _plugin_freed:
            with console.status("Re-parsing plugin for patching..."):
                plugin = load_plugin(file)
                strings = extract_strings(plugin)
                for s in strings:
                    s.source_file = file_stem

        with console.status("Patching plugin..."):
            st = getattr(plugin, "string_tables", None)
            patched = apply_translations(strings, translations, string_tables=st)
            rpt.strings_patched = patched

        _print(f"Patched [green]{patched}[/green] strings")

        # Write output
        if output is None:
            output = file.with_stem(f"{file.stem}_{lang}")
        rpt.output_file = str(output)

        with console.status("Writing output file..."):
            save_plugin(plugin, output)

        _print(f"Saved: [cyan]{output}[/cyan]")
    else:
        # Re-parse for preview if plugin was freed
        if _plugin_freed:
            plugin = load_plugin(file)
            strings = extract_strings(plugin)
            for s in strings:
                s.source_file = file_stem

        console.print("[yellow]Dry run — no file written.[/yellow]")
        # Show preview of translations
        table = Table(title="Translation Preview (first 20)")
        table.add_column("FormID", style="dim")
        table.add_column("Type")
        table.add_column("Original")
        table.add_column("Translated", style="green")

        for s in strings[:20]:
            t = translations.get(s.key, "")
            table.add_row(
                f"{s.form_id:08X}",
                s.subrecord_type.decode("ascii"),
                s.original_text[:50],
                t[:50],
            )
        console.print(table)

    rpt.finish()

    if report:
        save_report(rpt, report)
        _print(f"Report saved: [cyan]{report}[/cyan]")

    if cache:
        cache.close()


@app.command()
def scan(
    file: Path = typer.Argument(..., help="Path to the ESP/ESM file to scan."),
) -> None:
    """Scan an ESP/ESM file and list translatable strings."""
    from modtranslator.core.plugin import load_plugin
    from modtranslator.translation.extractor import extract_strings

    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    with console.status("Parsing..."):
        plugin = load_plugin(file)

    console.print(f"Game: [cyan]{plugin.game.name}[/cyan]")

    strings = extract_strings(plugin)
    console.print(f"Found [green]{len(strings)}[/green] translatable strings\n")

    table = Table(title=f"Translatable strings in {file.name}")
    table.add_column("FormID", style="dim")
    table.add_column("Record")
    table.add_column("Sub")
    table.add_column("EDID", style="dim")
    table.add_column("Text")

    for s in strings:
        table.add_row(
            f"{s.form_id:08X}",
            s.record_type.decode("ascii"),
            s.subrecord_type.decode("ascii"),
            s.editor_id[:20] if s.editor_id else "",
            s.original_text[:60],
        )

    console.print(table)


@app.command()
def batch(
    directory: Path = typer.Argument(
        ..., help="Directory containing ESP/ESM files.",
    ),
    lang: str = typer.Option(
        "ES", "--lang", "-l", help="Target language code.",
    ),
    pattern: str = typer.Option(
        "*.esp", "--pattern", "-p", help="File glob pattern.",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-k", envvar="DEEPL_API_KEY",
    ),
    backend_name: str = typer.Option(
        "deepl", "--backend", "-b",
        help="Backend: deepl, opus-mt, nllb, hybrid, dummy.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Model: base/tc-big (Opus-MT), 600M/1.3B (NLLB).",
    ),
    device: str | None = typer.Option(
        None, "--device",
        help="Device for Opus-MT/NLLB: auto, cpu, cuda.",
    ),
    glossary: Path | None = typer.Option(
        None, "--glossary", "-g",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-O",
        help="Output directory (keeps original filenames).",
    ),
    use_dummy: bool = typer.Option(False, "--dummy"),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable translation cache.",
    ),
    skip_translated: bool = typer.Option(
        True, "--skip-translated/--no-skip-translated",
        help="Skip strings already in target language.",
    ),
    game: GameChoice = typer.Option(
        GameChoice.auto, "--game",
        help="Game: fo3, fnv, skyrim, auto.",
    ),
) -> None:
    """Translate all matching files in a directory."""
    if not directory.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {directory}")
        raise typer.Exit(1)

    # Default pattern includes both ESP and ESM
    if pattern == "*.esp":
        files = sorted(list(directory.glob("*.esp")) + list(directory.glob("*.esm")))
    else:
        files = sorted(directory.glob(pattern))

    if not files:
        console.print(f"[yellow]No files matching '{pattern}' in {directory}[/yellow]")
        raise typer.Exit()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create backend once — CUDA fallback happens here, not per-file
    shared_backend, backend_label = _create_backend(
        backend_name,
        api_key=api_key,
        model=model,
        device=device,
        use_dummy=use_dummy,
    )

    console.print(f"Found [green]{len(files)}[/green] files matching '{pattern}'\n")
    _print(f"Backend: [cyan]{backend_label}[/cyan]", verbose_only=True)

    # Resolve glossary once before parallelizing
    # For --game auto, parse first file to detect game format
    from modtranslator.translation.glossary import Glossary

    gloss = None
    glossary_terms: set[str] | None = None

    if game == GameChoice.auto:
        from modtranslator.core.plugin import load_plugin
        try:
            first_plugin = load_plugin(files[0])
            detected_game = first_plugin.game
        except Exception:
            detected_game = Game.FALLOUT3
    elif game == GameChoice.skyrim:
        detected_game = Game.SKYRIM
    else:
        detected_game = Game.FALLOUT3

    glossary_source_terms: set[str] | None = None

    gloss_paths = _resolve_glossary_paths(glossary, lang, game, detected_game)
    if gloss_paths:
        gloss = Glossary.from_multiple_toml(gloss_paths)
        glossary_terms = {v.lower() for v in gloss.terms.values()}
        glossary_source_terms = {
            k.lower() for k, v in gloss.terms.items()
            if k.lower() != v.lower()
        }
        n_terms = len(gloss.terms)
        _print(
            f"Loaded glossary: [cyan]{n_terms}[/cyan] terms"
            f" from {len(gloss_paths)} file(s)",
        )

    # Run parallel pipeline
    success_count, skip_count, error_count, errors_list = _batch_translate_parallel(
        files,
        lang=lang,
        backend=shared_backend,
        backend_label=backend_label,
        gloss=gloss,
        glossary_terms=glossary_terms,
        glossary_source_terms=glossary_source_terms,
        skip_translated=skip_translated,
        output_dir=output_dir,
        no_cache=no_cache,
    )

    # Summary table
    summary = Table(title="Batch Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_row("Translated", f"[green]{success_count}[/green]")
    summary.add_row("Skipped", f"[yellow]{skip_count}[/yellow]")
    summary.add_row("Errors", f"[red]{error_count}[/red]")
    summary.add_row("Total", str(len(files)))
    console.print(summary)

    if errors_list:
        err_table = Table(title="Errors")
        err_table.add_column("File", style="red")
        err_table.add_column("Error")
        for fname, err_msg in errors_list:
            err_table.add_row(fname, err_msg)
        console.print(err_table)


@app.command(name="batch-pex")
def batch_pex(
    directory: Path = typer.Argument(
        ..., help="Directory containing .pex script files.",
    ),
    lang: str = typer.Option(
        "ES", "--lang", "-l", help="Target language code.",
    ),
    backend_name: str = typer.Option(
        "deepl", "--backend", "-b",
        help="Backend: deepl, opus-mt, nllb, hybrid, dummy.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Model: base/tc-big (Opus-MT), 600M/1.3B (NLLB).",
    ),
    device: str | None = typer.Option(
        None, "--device",
        help="Device for Opus-MT/NLLB: auto, cpu, cuda.",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-k", envvar="DEEPL_API_KEY",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-O",
        help="Output directory (keeps original filenames).",
    ),
    use_dummy: bool = typer.Option(False, "--dummy"),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable translation cache.",
    ),
    skip_translated: bool = typer.Option(
        True, "--skip-translated/--no-skip-translated",
        help="Skip strings already in target language.",
    ),
    glossary: Path | None = typer.Option(
        None, "--glossary", "-g",
    ),
    game: GameChoice = typer.Option(
        GameChoice.skyrim, "--game",
        help="Game: fo3, fnv, skyrim, auto.",
    ),
) -> None:
    """Translate all .pex Papyrus script files in a directory."""
    import time as _time

    from modtranslator.core.pex_parser import load_pex, save_pex
    from modtranslator.translation.cache import TranslationCache

    if not directory.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {directory}")
        raise typer.Exit(1)

    files = sorted(directory.glob("*.pex"))
    if not files:
        console.print(f"[yellow]No .pex files in {directory}[/yellow]")
        raise typer.Exit()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create backend
    shared_backend, backend_label = _create_backend(
        backend_name, api_key=api_key, model=model, device=device, use_dummy=use_dummy,
    )

    console.print(f"Found [green]{len(files)}[/green] .pex files")
    _print(f"Backend: [cyan]{backend_label}[/cyan]", verbose_only=True)

    # Load glossary
    from modtranslator.translation.glossary import Glossary

    gloss = None
    glossary_terms: set[str] | None = None
    glossary_source_terms: set[str] | None = None
    detected_game = Game.SKYRIM if game == GameChoice.skyrim else Game.FALLOUT3
    gloss_paths = _resolve_glossary_paths(glossary, lang, game, detected_game)
    if gloss_paths:
        gloss = Glossary.from_multiple_toml(gloss_paths)
        glossary_terms = {v.lower() for v in gloss.terms.values()}
        glossary_source_terms = {
            k.lower() for k, v in gloss.terms.items() if k.lower() != v.lower()
        }
        _print(f"Loaded glossary: [cyan]{len(gloss.terms)}[/cyan] terms")

    # Phase 1: Parse all .pex files and collect translatable strings
    cache = None if no_cache else TranslationCache()
    all_entries: list[tuple[Path, dict[int, str]]] = []  # (path, {idx: text})
    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("Phase 1: Scanning .pex files", total=len(files))
        for pex_path in files:
            try:
                pex = load_pex(pex_path)
                translatable = pex.get_translatable_strings()

                # Language detection filter
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
                _print(f"[yellow]Warning:[/yellow] Failed to parse {pex_path.name}: {e}",
                       verbose_only=True)
            progress.advance(task)

    files_with_text = len(all_entries)
    console.print(
        f"Found [green]{len(unique_texts)}[/green] unique translatable strings"
        f" in [green]{files_with_text}[/green] files"
    )

    if not unique_texts:
        console.print("[yellow]No translatable strings found.[/yellow]")
        raise typer.Exit()

    # Cache check
    cached: dict[str, str] = {}
    if cache is not None:
        cached = cache.get_batch(unique_texts, lang)
        if cached:
            _print(f"Cache hits: [green]{len(cached)}[/green]")

    to_translate = [t for t in unique_texts if t not in cached]

    # Protect glossary terms
    gloss_mappings: list[dict[str, str]] | None = None
    if gloss and to_translate:
        to_translate, gloss_mappings = gloss.protect_batch(to_translate)

    # Protect Spanish words
    es_mappings: list[dict[str, str]] | None = None
    if to_translate and lang.upper() == "ES":
        from modtranslator.translation.spanish_protect import protect_spanish_batch
        to_translate, es_mappings = protect_spanish_batch(to_translate)

    # Phase 2: Translate
    translated: list[str] = []
    if to_translate:
        chunk_size = 500
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TextColumn("•"),
            TimeElapsedColumn(), TextColumn("ETA"), TimeRemainingColumn(),
            console=console, transient=True,
        ) as progress:
            task = progress.add_task("Phase 2: Translating", total=len(to_translate))
            _t0 = _time.monotonic()
            for chunk_start in range(0, len(to_translate), chunk_size):
                chunk = to_translate[chunk_start:chunk_start + chunk_size]
                translated.extend(shared_backend.translate_batch(chunk, lang))
                progress.advance(task, len(chunk))
            _elapsed = _time.monotonic() - _t0

        if _elapsed > 0:
            rate = len(translated) / _elapsed
            _print(
                f"Translated {len(translated)} strings"
                f" in {_elapsed:.1f}s ({rate:.0f} strings/s)",
            )

        # Restore Spanish words
        if es_mappings is not None:
            from modtranslator.translation.spanish_protect import restore_spanish_batch
            translated = restore_spanish_batch(translated, es_mappings)

        # Restore glossary
        if gloss and gloss_mappings:
            translated = gloss.restore_batch(translated, gloss_mappings)

    # Build full translation map: original_text → translated_text
    translations: dict[str, str] = dict(cached)
    uncached_originals = [t for t in unique_texts if t not in cached]
    for orig, trans in zip(uncached_originals, translated, strict=True):
        translations[orig] = trans

    # Cache new translations
    if cache is not None and translated:
        cache_entries = [
            (orig, lang, orig, translations[orig])
            for orig in uncached_originals
            if orig in translations
        ]
        if cache_entries:
            cache.put_batch(cache_entries, backend=backend_label)
        cache.close()

    # Phase 3: Patch and write .pex files
    success_count = 0
    error_count = 0
    errors_list: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("Phase 3: Writing .pex files", total=files_with_text)
        for pex_path, translatable in all_entries:
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
                    success_count += 1
            except Exception as e:
                error_count += 1
                errors_list.append((pex_path.name, str(e)))
            progress.advance(task)

    # Summary
    summary = Table(title="PEX Batch Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_row("Translated", f"[green]{success_count}[/green]")
    summary.add_row("Skipped", f"[yellow]{len(files) - files_with_text}[/yellow]")
    summary.add_row("Errors", f"[red]{error_count}[/red]")
    summary.add_row("Total", str(len(files)))
    console.print(summary)

    if errors_list:
        err_table = Table(title="Errors")
        err_table.add_column("File", style="red")
        err_table.add_column("Error")
        for fname, err_msg in errors_list:
            err_table.add_row(fname, err_msg)
        console.print(err_table)


@app.command(name="batch-mcm")
def batch_mcm(
    directory: Path = typer.Argument(
        ..., help="Skyrim Data directory (contains Interface/translations/).",
    ),
    lang: str = typer.Option(
        "ES", "--lang", "-l", help="Target language code.",
    ),
    backend_name: str = typer.Option(
        "deepl", "--backend", "-b",
        help="Backend: deepl, opus-mt, nllb, hybrid, dummy.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
    ),
    device: str | None = typer.Option(
        None, "--device",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-k", envvar="DEEPL_API_KEY",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-O",
        help="Output directory (mirrors Interface/translations/ structure).",
    ),
    use_dummy: bool = typer.Option(False, "--dummy"),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable translation cache.",
    ),
    skip_translated: bool = typer.Option(
        True, "--skip-translated/--no-skip-translated",
        help="Skip strings already in target language.",
    ),
    glossary: Path | None = typer.Option(
        None, "--glossary", "-g",
    ),
    game: GameChoice = typer.Option(
        GameChoice.skyrim, "--game",
    ),
) -> None:
    """Translate MCM Interface/translations/ files and MCM Recorder JSON."""
    import json
    import time as _time

    from modtranslator.translation.cache import TranslationCache

    # Language name mapping for filenames
    lang_names = {
        "ES": "spanish", "FR": "french", "DE": "german", "IT": "italian",
        "PT": "portuguese", "RU": "russian", "PL": "polish", "CS": "czech",
        "JA": "japanese", "ZH": "chinese",
    }
    target_lang_name = lang_names.get(lang.upper(), lang.lower())

    # Locate Interface/translations/ directory
    translations_dir = directory / "Interface" / "translations"
    mcm_recorder_dir = directory / "McmRecorder"

    if not translations_dir.is_dir() and not mcm_recorder_dir.is_dir():
        console.print(
            f"[red]Error:[/red] No Interface/translations/ or McmRecorder/ in {directory}"
        )
        raise typer.Exit(1)

    # Create backend
    shared_backend, backend_label = _create_backend(
        backend_name, api_key=api_key, model=model, device=device, use_dummy=use_dummy,
    )
    _print(f"Backend: [cyan]{backend_label}[/cyan]", verbose_only=True)

    # Load glossary
    from modtranslator.translation.glossary import Glossary

    gloss = None
    glossary_terms: set[str] | None = None
    glossary_source_terms: set[str] | None = None
    detected_game = Game.SKYRIM if game == GameChoice.skyrim else Game.FALLOUT3
    gloss_paths = _resolve_glossary_paths(glossary, lang, game, detected_game)
    if gloss_paths:
        gloss = Glossary.from_multiple_toml(gloss_paths)
        glossary_terms = {v.lower() for v in gloss.terms.values()}
        glossary_source_terms = {
            k.lower() for k, v in gloss.terms.items() if k.lower() != v.lower()
        }
        _print(f"Loaded glossary: [cyan]{len(gloss.terms)}[/cyan] terms")

    cache = None if no_cache else TranslationCache()

    # ── Phase 1: Collect all translatable strings from MCM files ──
    # Each entry: (file_path, output_path, lines, translatable_indices, translatable_texts)
    mcm_files: list[tuple[Path, Path, list[str], list[int], list[str]]] = []
    # MCM Recorder JSON: (file_path, output_path, data_dict, translatable_keys, translatable_texts)
    mcm_jsons: list[tuple[Path, Path, dict, list[str], list[str]]] = []

    if translations_dir.is_dir():
        # Find all _english.txt files (case insensitive via manual filter)
        import re as _re

        all_txt = sorted(translations_dir.iterdir())
        eng_files = [f for f in all_txt if _re.search(r'_english\.txt$', f.name, _re.IGNORECASE)]

        for eng_path in eng_files:
            stem = _re.sub(r'(?i)_english\.txt$', '', eng_path.name)

            # Check if Spanish translation already exists and is genuine
            target_name_variants = [
                f"{stem}_{target_lang_name}.txt",
                f"{stem}_{target_lang_name.upper()}.txt",
                f"{stem}_{target_lang_name.capitalize()}.txt",
            ]
            existing_spa = None
            for variant in target_name_variants:
                candidate = translations_dir / variant
                if candidate.exists():
                    existing_spa = candidate
                    break

            # Skip if already genuinely translated (different content from English)
            if existing_spa is not None:
                import hashlib
                eng_hash = hashlib.md5(eng_path.read_bytes()).hexdigest()
                spa_hash = hashlib.md5(existing_spa.read_bytes()).hexdigest()
                if eng_hash != spa_hash:
                    _print(
                        f"[dim]Skipping {stem} (already translated)[/dim]",
                        verbose_only=True,
                    )
                    continue

            # Determine output path
            # Match casing of existing spanish file or english file
            if existing_spa is not None:
                out_name = existing_spa.name
            else:
                # Match casing pattern from English filename
                if "_ENGLISH." in eng_path.name:
                    out_name = f"{stem}_{target_lang_name.upper()}.txt"
                else:
                    out_name = f"{stem}_{target_lang_name}.txt"

            if output_dir is not None:
                out_path = output_dir / out_name
            else:
                out_path = translations_dir / out_name

            # Parse the English file
            try:
                raw = eng_path.read_bytes()
                # Detect encoding: UTF-16-LE BOM = FF FE
                if raw[:2] == b'\xff\xfe':
                    text = raw.decode('utf-16-le')
                    # Remove BOM character
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
                    _print(
                        f"  {stem}: {len(translatable_texts)} strings",
                        verbose_only=True,
                    )
            except Exception as e:
                _print(f"[yellow]Warning:[/yellow] Failed to parse {eng_path.name}: {e}")

    # MCM Recorder JSON files
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
            except Exception as e:
                _print(f"[yellow]Warning:[/yellow] Failed to parse {json_path.name}: {e}")

    total_files = len(mcm_files) + len(mcm_jsons)
    if total_files == 0:
        console.print("[yellow]No MCM files need translation.[/yellow]")
        raise typer.Exit()

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

    console.print(
        f"Found [green]{len(unique_texts)}[/green] unique strings"
        f" in [green]{total_files}[/green] files"
    )

    # Language filter
    if skip_translated:
        from modtranslator.translation.lang_detect import should_translate

        to_translate_set = {
            t for t in unique_texts
            if should_translate(t, lang, glossary_terms, glossary_source_terms)
        }
        skipped = len(unique_texts) - len(to_translate_set)
        if skipped:
            _print(f"Skipped [yellow]{skipped}[/yellow] strings already in target language")
        texts_for_translation = [t for t in unique_texts if t in to_translate_set]
    else:
        texts_for_translation = list(unique_texts)

    # Cache check
    cached: dict[str, str] = {}
    if cache is not None:
        cached = cache.get_batch(texts_for_translation, lang)
        if cached:
            _print(f"Cache hits: [green]{len(cached)}[/green]")

    to_translate = [t for t in texts_for_translation if t not in cached]

    # Protect glossary terms
    gloss_mappings: list[dict[str, str]] | None = None
    if gloss and to_translate:
        to_translate, gloss_mappings = gloss.protect_batch(to_translate)

    # Protect Spanish words
    es_mappings: list[dict[str, str]] | None = None
    if to_translate and lang.upper() == "ES":
        from modtranslator.translation.spanish_protect import protect_spanish_batch
        to_translate, es_mappings = protect_spanish_batch(to_translate)

    # ── Phase 2: Translate ──
    translated: list[str] = []
    if to_translate:
        chunk_size = 500
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TextColumn("•"),
            TimeElapsedColumn(), TextColumn("ETA"), TimeRemainingColumn(),
            console=console, transient=True,
        ) as progress:
            task = progress.add_task("Translating", total=len(to_translate))
            _t0 = _time.monotonic()
            for chunk_start in range(0, len(to_translate), chunk_size):
                chunk = to_translate[chunk_start:chunk_start + chunk_size]
                translated.extend(shared_backend.translate_batch(chunk, lang))
                progress.advance(task, len(chunk))
            _elapsed = _time.monotonic() - _t0

        if _elapsed > 0:
            rate = len(translated) / _elapsed
            _print(
                f"Translated {len(translated)} strings"
                f" in {_elapsed:.1f}s ({rate:.0f} strings/s)",
            )

        # Restore Spanish words
        if es_mappings is not None:
            from modtranslator.translation.spanish_protect import restore_spanish_batch
            translated = restore_spanish_batch(translated, es_mappings)

        # Restore glossary
        if gloss and gloss_mappings:
            translated = gloss.restore_batch(translated, gloss_mappings)

    # Build full translation map
    translations: dict[str, str] = dict(cached)
    uncached_originals = [t for t in texts_for_translation if t not in cached]
    for orig, trans in zip(uncached_originals, translated, strict=True):
        translations[orig] = trans

    # Cache new translations
    if cache is not None and translated:
        cache_entries = [
            (orig, lang, orig, translations[orig])
            for orig in uncached_originals
            if orig in translations
        ]
        if cache_entries:
            cache.put_batch(cache_entries, backend=backend_label)
        cache.close()

    # ── Phase 3: Write output files ──
    success_count = 0
    error_count = 0
    errors_list: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("Writing files", total=total_files)

        # Write MCM translation files
        for eng_path, out_path, lines, indices, orig_texts in mcm_files:
            try:
                # Build translated lines
                new_lines = list(lines)
                for idx, orig_text in zip(indices, orig_texts, strict=True):
                    if orig_text in translations:
                        line = new_lines[idx]
                        stripped = line.rstrip('\r\n')
                        ending = line[len(stripped):]  # preserve line ending
                        parts = stripped.split('\t', 1)
                        new_lines[idx] = f"{parts[0]}\t{translations[orig_text]}{ending}"

                # Write as UTF-16-LE with BOM
                out_path.parent.mkdir(parents=True, exist_ok=True)
                content = ''.join(new_lines)
                out_path.write_bytes(b'\xff\xfe' + content.encode('utf-16-le'))
                success_count += 1
            except Exception as e:
                error_count += 1
                errors_list.append((eng_path.name, str(e)))
            progress.advance(task)

        # Write MCM Recorder JSON files
        for json_path, out_json, data, keys, orig_texts in mcm_jsons:
            try:
                for key, orig_text in zip(keys, orig_texts, strict=True):
                    if orig_text in translations:
                        data[key] = translations[orig_text]
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                success_count += 1
            except Exception as e:
                error_count += 1
                errors_list.append((json_path.name, str(e)))
            progress.advance(task)

    # Summary
    summary = Table(title="MCM Batch Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_row("Translation files", f"[green]{len(mcm_files)}[/green]")
    summary.add_row("JSON files", f"[green]{len(mcm_jsons)}[/green]")
    summary.add_row("Unique strings", str(len(unique_texts)))
    summary.add_row("Translated", f"[green]{success_count}[/green]")
    summary.add_row("Errors", f"[red]{error_count}[/red]")
    console.print(summary)

    if errors_list:
        err_table = Table(title="Errors")
        err_table.add_column("File", style="red")
        err_table.add_column("Error")
        for fname, err_msg in errors_list:
            err_table.add_row(fname, err_msg)
        console.print(err_table)


@app.command(name="cache-info")
def cache_info() -> None:
    """Show translation cache statistics."""
    from modtranslator.translation.cache import TranslationCache

    cache = TranslationCache()
    count = cache.count()
    console.print(f"Cached translations: [green]{count}[/green]")
    console.print(f"Cache location: [dim]{cache._db_path}[/dim]")
    cache.close()


@app.command(name="cache-clear")
def cache_clear() -> None:
    """Clear the translation cache."""
    from modtranslator.translation.cache import TranslationCache

    cache = TranslationCache()
    deleted = cache.clear()
    console.print(f"Cleared [yellow]{deleted}[/yellow] cached translations.")
    cache.close()


if __name__ == "__main__":
    app()
