"""CLI interface for modtranslator using Typer."""

from __future__ import annotations

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
from modtranslator.pipeline import (
    BatchResult,
    GameChoice,
    batch_translate_esp,
    batch_translate_mcm,
    batch_translate_pex,
    clear_cache,
    create_backend,
    get_cache_info,
    resolve_glossary_paths,
    scan_file,
)

app = typer.Typer(
    name="modtranslator",
    help="Automatic translator for Bethesda ESP/ESM mod files.",
    add_completion=False,
)
console = Console()

_verbose = False
_quiet = False


def _print(msg: str, *, verbose_only: bool = False) -> None:
    """Print respecting --verbose/--quiet flags."""
    if _quiet:
        return
    if verbose_only and not _verbose:
        return
    console.print(msg)


def _make_rich_progress_callback(progress: Progress, task_id: object) -> None:
    """Create a callback adapter from pipeline progress to Rich progress bar."""
    # This is handled inline in each command for now since Rich Progress
    # needs different configurations per phase.
    pass


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
    _backend_instance: object | None = None,
) -> None:
    """Internal translate logic for a single file."""
    import gc

    from modtranslator.backends.base import TranslationBackend
    from modtranslator.core.plugin import load_plugin, save_plugin
    from modtranslator.reporting.formatters import save_report
    from modtranslator.reporting.report import TranslationReport
    from modtranslator.translation.cache import TranslationCache
    from modtranslator.translation.extractor import extract_strings
    from modtranslator.translation.glossary import Glossary
    from modtranslator.translation.patcher import apply_translations

    large_file_threshold = 150 * 1024 * 1024

    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    rpt = TranslationReport(
        source_file=str(file),
        target_lang=lang,
        dry_run=dry_run,
    )

    if use_dummy:
        backend_name = "dummy"

    if _backend_instance is not None:
        backend: TranslationBackend = _backend_instance  # type: ignore[assignment]
        rpt.backend = type(backend).__name__
    else:
        try:
            backend, rpt.backend = create_backend(
                backend_name, api_key=api_key, model=model, device=device,
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

    _print(f"Backend: [cyan]{rpt.backend}[/cyan]", verbose_only=True)

    with console.status("Parsing plugin file..."):
        plugin = load_plugin(file)
        rpt.game_detected = plugin.game.name

    _print(f"Game detected: [cyan]{plugin.game.name}[/cyan]")

    with console.status("Extracting strings..."):
        strings = extract_strings(plugin)
        file_stem = file.stem
        for s in strings:
            s.source_file = file_stem
        rpt.total_strings_found = len(strings)

    _print(f"Found [green]{len(strings)}[/green] translatable strings")

    gloss = None
    gloss_paths = resolve_glossary_paths(glossary, lang, game, plugin.game)
    if gloss_paths:
        gloss = Glossary.from_multiple_toml(gloss_paths)
        rpt.glossary_file = ", ".join(str(p) for p in gloss_paths)
        rpt.glossary_terms = len(gloss.terms)
        n_terms = len(gloss.terms)
        _print(
            f"Loaded glossary: [cyan]{n_terms}[/cyan] terms"
            f" from {len(gloss_paths)} file(s)",
        )

    if skip_translated:
        from modtranslator.translation.lang_detect import should_translate

        glossary_terms: set[str] | None = None
        glossary_source_terms: set[str] | None = None
        if gloss:
            glossary_terms = {v.lower() for v in gloss.terms.values()}
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

    cache = None
    cached: dict[str, str] = {}
    if not no_cache:
        cache = TranslationCache()
        keys = [s.key for s in strings]
        cached = cache.get_batch(keys, lang)
        rpt.strings_from_cache = len(cached)
        if cached:
            _print(f"Cache hits: [green]{len(cached)}[/green]")

    to_translate = [s for s in strings if s.key not in cached]
    texts = [s.original_text for s in to_translate]

    gloss_mappings: list[dict[str, str]] | None = None
    if gloss and texts:
        texts, gloss_mappings = gloss.protect_batch(texts)

    es_mappings: list[dict[str, str]] | None = None
    if texts and lang.upper() == "ES":
        from modtranslator.translation.spanish_protect import protect_spanish_batch

        texts, es_mappings = protect_spanish_batch(texts)

    _plugin_freed = False
    to_translate_keys = [s.key for s in to_translate]
    to_translate_originals = [s.original_text for s in to_translate]
    if texts and file.stat().st_size > large_file_threshold:
        _plugin_freed = True
        _print("Large file: freeing plugin memory before translation...", verbose_only=True)
        del plugin, strings, to_translate
        gc.collect()

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

        if es_mappings is not None:
            from modtranslator.translation.spanish_protect import restore_spanish_batch

            translated = restore_spanish_batch(translated, es_mappings)

        if gloss:
            translated = gloss.restore_batch(translated, gloss_mappings)

        cache_entries = []
        for key, orig, t in zip(to_translate_keys, to_translate_originals, translated, strict=True):
            translations[key] = t
            cache_entries.append((key, lang, orig, t))

        if cache and cache_entries:
            cache.put_batch(cache_entries, backend=rpt.backend)

    if not dry_run:
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

        if output is None:
            output = file.with_stem(f"{file.stem}_{lang}")
        rpt.output_file = str(output)

        with console.status("Writing output file..."):
            save_plugin(plugin, output)

        _print(f"Saved: [cyan]{output}[/cyan]")
    else:
        if _plugin_freed:
            plugin = load_plugin(file)
            strings = extract_strings(plugin)
            for s in strings:
                s.source_file = file_stem

        console.print("[yellow]Dry run — no file written.[/yellow]")
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
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    with console.status("Parsing..."):
        strings = scan_file(file)

    from modtranslator.core.plugin import load_plugin
    plugin = load_plugin(file)
    console.print(f"Game: [cyan]{plugin.game.name}[/cyan]")

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

    if pattern == "*.esp":
        files = sorted(list(directory.glob("*.esp")) + list(directory.glob("*.esm")))
    else:
        files = sorted(directory.glob(pattern))

    if not files:
        console.print(f"[yellow]No files matching '{pattern}' in {directory}[/yellow]")
        raise typer.Exit()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if use_dummy:
        backend_name = "dummy"

    try:
        shared_backend, backend_label = create_backend(
            backend_name, api_key=api_key, model=model, device=device,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    console.print(f"Found [green]{len(files)}[/green] files matching '{pattern}'\n")
    _print(f"Backend: [cyan]{backend_label}[/cyan]", verbose_only=True)

    result = batch_translate_esp(
        files,
        lang=lang,
        backend=shared_backend,
        backend_label=backend_label,
        game=game,
        glossary=glossary,
        skip_translated=skip_translated,
        output_dir=output_dir,
        no_cache=no_cache,
    )

    _print_batch_summary("Batch Summary", result, len(files))


@app.command(name="batch-pex")
def batch_pex_cmd(
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
    if not directory.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {directory}")
        raise typer.Exit(1)

    files = sorted(directory.glob("*.pex"))
    if not files:
        console.print(f"[yellow]No .pex files in {directory}[/yellow]")
        raise typer.Exit()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if use_dummy:
        backend_name = "dummy"

    try:
        shared_backend, backend_label = create_backend(
            backend_name, api_key=api_key, model=model, device=device,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    console.print(f"Found [green]{len(files)}[/green] .pex files")
    _print(f"Backend: [cyan]{backend_label}[/cyan]", verbose_only=True)

    result = batch_translate_pex(
        files,
        lang=lang,
        backend=shared_backend,
        backend_label=backend_label,
        game=game,
        glossary=glossary,
        skip_translated=skip_translated,
        output_dir=output_dir,
        no_cache=no_cache,
    )

    _print_batch_summary("PEX Batch Summary", result, len(files))


@app.command(name="batch-mcm")
def batch_mcm_cmd(
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
    if use_dummy:
        backend_name = "dummy"

    try:
        shared_backend, backend_label = create_backend(
            backend_name, api_key=api_key, model=model, device=device,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    _print(f"Backend: [cyan]{backend_label}[/cyan]", verbose_only=True)

    try:
        result = batch_translate_mcm(
            directory,
            lang=lang,
            backend=shared_backend,
            backend_label=backend_label,
            game=game,
            glossary=glossary,
            skip_translated=skip_translated,
            output_dir=output_dir,
            no_cache=no_cache,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if result.success_count == 0 and result.error_count == 0:
        console.print("[yellow]No MCM files need translation.[/yellow]")
        raise typer.Exit()

    _print_batch_summary("MCM Batch Summary", result, result.success_count + result.error_count)


def _print_batch_summary(title: str, result: BatchResult, total_files: int) -> None:
    """Print a Rich table with batch results."""
    summary = Table(title=title)
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_row("Translated", f"[green]{result.success_count}[/green]")
    summary.add_row("Skipped", f"[yellow]{result.skip_count}[/yellow]")
    summary.add_row("Errors", f"[red]{result.error_count}[/red]")
    summary.add_row("Total", str(total_files))
    console.print(summary)

    if result.errors:
        err_table = Table(title="Errors")
        err_table.add_column("File", style="red")
        err_table.add_column("Error")
        for fname, err_msg in result.errors:
            if fname:  # skip "Cancelled" pseudo-error
                err_table.add_row(fname, err_msg)
        if err_table.row_count:
            console.print(err_table)


@app.command(name="cache-info")
def cache_info_cmd() -> None:
    """Show translation cache statistics."""
    info = get_cache_info()
    console.print(f"Cached translations: [green]{info['count']}[/green]")
    console.print(f"Cache location: [dim]{info['path']}[/dim]")


@app.command(name="cache-clear")
def cache_clear_cmd() -> None:
    """Clear the translation cache."""
    deleted = clear_cache()
    console.print(f"Cleared [yellow]{deleted}[/yellow] cached translations.")


if __name__ == "__main__":
    app()
