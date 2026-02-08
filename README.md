# modtranslator

**Automatic offline translator for Bethesda game mods (ESP/ESM files).**

Parses the TES4 binary format used by Fallout 3 and Fallout: New Vegas — extracts every player-visible string, translates it through neural machine translation, and rewrites the plugin file with recalculated sizes. No cloud APIs required.

> **Tested on a full Fallout 3 installation** — 124 files (base game + 5 DLCs + 50+ mods), 96,456 strings translated with zero errors, zero broken placeholders. The game loads and plays correctly with translated files.

## Why?

Thousands of Fallout mods on Nexus are English-only. Existing translation tools (xTranslator) require manual per-string work or importing pre-made translations that may not exist. There's no automated solution for mass-translating mod files offline.

modtranslator solves this: point it at a game's Data folder, pick a backend, and walk away. It handles the binary format, protects game terminology with glossaries, skips internal IDs that would break the game, and outputs ready-to-use ESP/ESM files.

## Results

| Metric | Value |
|--------|-------|
| Files translated | 124 (Fallout 3 full install) |
| Strings processed | 96,456 |
| Parse errors | 0 |
| Broken placeholders | 0 |
| Game functional | Yes |
| Backend used | Hybrid (tc-big + NLLB), CUDA |

## Supported Games

| Game | Status |
|------|--------|
| **Fallout 3** | Fully tested — 124 files, game verified |
| **Fallout: New Vegas** | Supported — same binary format as FO3 |

## Installation

```bash
git clone https://github.com/adamzgz/modtranslator.git
cd modtranslator

# Base install (CLI + DeepL backend)
pip install -e .

# Offline translation — recommended
pip install -e ".[opus-mt,nllb]"

# Development (tests, lint, type checking)
pip install -e ".[dev]"
```

### GPU Acceleration (Optional)

```bash
pip install torch nvidia-cublas-cu12
# Then use --device cuda or --device auto
```

## Quick Start

```bash
# Scan a mod to preview translatable strings
modtranslator scan MyMod.esp

# Translate a single file
modtranslator translate MyMod.esp --backend hybrid --device cuda

# Translate an entire game folder
modtranslator batch "C:\Games\Fallout 3\Data" --backend hybrid --device cuda --game fo3 --output-dir translated/Data
```

## Translation Backends

Five interchangeable backends — from fast offline models to cloud APIs:

| Backend | Speed (GPU) | Cost | Quality EN→ES | Offline |
|---------|-------------|------|---------------|---------|
| **Hybrid (tc-big+NLLB)** | ~33 str/s | Free | Best combined | Yes |
| **Opus-MT tc-big** | 241–468 str/s | Free | Very good | Yes |
| Opus-MT base | 188–291 str/s | Free | Good | Yes |
| NLLB 1.3B | 32–71 str/s | Free | Excellent (long text) | Yes |
| DeepL | Online | API key | Very good | No |

The **hybrid** backend routes short strings (1–3 words) to Opus-MT tc-big and longer strings (4+ words) to NLLB 1.3B — combining the speed of tc-big with the fluency of NLLB on complex sentences while avoiding NLLB's hallucination issues on short inputs.

## How It Works

```
ESP/ESM file (binary, Windows-1252 encoded)
    │
    ▼
1. PARSE ─────────── Binary reader → Record/Group tree (TES4 format)
    │                 Handles compressed records (zlib), 24-byte headers
    ▼
2. EXTRACT ───────── Registry filters translatable subrecords
    │                 FULL (37 record types), DESC, NAM1, RNAM, TNAM, NNAM, ITXT, CNAM
    │                 Covers inventory, map markers, terminals, character creation
    │                 Skips editor IDs, binary data, internal references
    ▼
3. DETECT LANGUAGE ─ 4-layer heuristic skips already-translated strings
    │                 glossary match → langid → spanish dict → english dict
    ▼
4. PROTECT ───────── Glossary terms → Gx0, Gx1 placeholders
    │                 Spanish words in mixed strings → Cx0, Cx1 placeholders
    │                 (3 SentencePiece tokens each, 100% MT survival rate)
    ▼
5. TRANSLATE ─────── Backend processes strings in batches
    │                 SQLite cache skips previously translated strings
    ▼
6. RESTORE ───────── Placeholders → original protected terms
    │
    ▼
7. WRITE ─────────── Serialize tree with recalculated sizes → valid ESP/ESM
```

### Batch Pipeline

The `batch` command runs three sequential phases to translate entire game folders:

```
Phase 1: PREPARE — parse, extract, filter, protect (per file)
Phase 2: TRANSLATE — deduplicate across files, translate unique strings in chunks
Phase 3: WRITE — restore placeholders, patch subrecords, save files
```

All phases are sequential. Fallout 3 determines mod load order from file timestamps — parallel writes would give files identical NTFS timestamps, scrambling the order and causing crashes.

For files over 150 MB (like Fallout3.esm at ~280 MB), the pipeline uses a double-parse strategy: parse → extract → free plugin from memory → translate → re-parse → patch → write. This avoids holding both the plugin tree (~2 GB) and the ML model (~2 GB) in RAM simultaneously.

## Game-Specific Glossaries

TOML glossaries protect official Bethesda terminology from machine translation:

```toml
[terms]
"Vault" = "Refugio"
"Stimpak" = "Estimulante"
"Brotherhood of Steel" = "Hermandad del Acero"
"Deathclaw" = "Sanguinario"
"Pip-Boy" = "Pip-Boy"
```

| Glossary | Scope | Example terms |
|----------|-------|---------------|
| `fallout_base_es.toml` | Shared FO3/FNV | SPECIAL stats, consumables, factions, weapons |
| `fallout3_es.toml` | Capital Wasteland | Megaton, Three Dog, Talon Company |
| `falloutnv_es.toml` | Mojave Wasteland | NCR→RNC, Mr. House→Sr. House, Yes Man→Servibot |

The `--game` flag auto-loads the correct combination:

```bash
modtranslator translate mod.esp --game fo3       # fallout_base + fallout3
modtranslator translate mod.esp --game fnv       # fallout_base + falloutnv
```

## CLI Reference

```
modtranslator [--verbose|-v] [--quiet|-q] COMMAND

Commands:
  scan           List translatable strings (no changes)
  translate      Translate a single ESP/ESM file
  batch          Translate all matching files in a directory
  cache-info     Show translation cache statistics
  cache-clear    Clear the translation cache

Key flags:
  --backend      {dummy, deepl, opus-mt, nllb, hybrid}
  --model        {base, tc-big} for Opus-MT / {600M, 1.3B} for NLLB
  --device       {auto, cpu, cuda}
  --game         {auto, fo3, fnv}
  --glossary     path/to/custom.toml
  --output-dir   write files to a separate directory
  --lang         target language (default: ES)
  --dry-run      preview without writing
  --no-cache     skip translation cache
  --skip-translated / --no-skip-translated
```

## Architecture

```
src/modtranslator/          4,200+ lines
├── cli.py                  Typer CLI, batch pipeline (3-phase sequential)
├── core/                   Binary ESP/ESM parser and writer
│   ├── parser.py             bytes → Record tree (TES4 format)
│   ├── writer.py             Record tree → bytes (bottom-up size recalc)
│   ├── records.py            Dataclasses: Subrecord, Record, GroupRecord, PluginFile
│   ├── constants.py          Header sizes, flags, Game enum
│   ├── compression.py        zlib for compressed records
│   └── plugin.py             Facade: load_plugin / save_plugin
├── backends/               Translation backends (TranslationBackend ABC)
│   ├── opus_mt.py            Helsinki-NLP + CTranslate2 (base + tc-big)
│   ├── nllb.py               Meta NLLB-200 + CTranslate2 (600M/1.3B)
│   ├── hybrid.py             tc-big (short) + NLLB (long) routing
│   ├── deepl.py              DeepL API
│   └── dummy.py              Test backend: prefixes [XX]
├── translation/            String extraction, filtering, and patching
│   ├── extractor.py          Record tree → list[TranslatableString]
│   ├── registry.py           (record_type, sub_type) → translatable?
│   ├── glossary.py           TOML glossaries + Gx{i} placeholder system
│   ├── spanish_protect.py    Spanish word protection with Cx{i} placeholders
│   ├── lang_detect.py        4-layer language detection heuristic
│   ├── patcher.py            Apply translations to subrecord bytearrays
│   └── cache.py              SQLite translation cache (~/.modtranslator/)
├── reporting/              Translation reports (JSON, Markdown, CSV)
└── data/                   Spanish dictionary (~2,100 words)

glossaries/                 Game-specific TOML terminology files
tests/                      378 tests, ~84% branch coverage
```

### TES4 Binary Format

The parser handles the TES4 plugin format used by Gamebryo/Creation Engine games:

- **Record Header**: `Type(4) + DataSize(4) + Flags(4) + FormID(4) + VCS1(4) + VCS2(4)` — 24 bytes
- **Subrecord**: `Type(4) + Size(2, uint16) + Data(N)`
- **Group (GRUP)**: 24-byte header, `GroupSize` includes the header itself
- **Strings**: null-terminated, Windows-1252 encoding
- **Compressed records**: flag `0x00040000`, payload = `decompressed_size(4) + zlib`
- **Game detection**: HEDR subrecord version float (0.94 = FO3/FNV)

### Key Design Decisions

- **Mutable subrecords**: `Subrecord.data` is `bytearray` — the patcher modifies in-place with zero copies
- **Computed sizes**: `Subrecord.size` is a property (`len(self.data)`) — auto-updates on mutation
- **Bottom-up serialization**: the writer recalculates all sizes from the record tree, so translated strings of different length always produce valid files
- **Compact placeholders**: `Gx{i}` tokenizes as 3 SentencePiece tokens (vs 6+ for longer formats), achieving 100% survival through neural MT pipelines
- **Encoding chain**: cp1252 → UTF-8 → latin-1 fallback for edge-case bytes (0x90, 0x8D, 0x9D)

## Tests

378 tests covering the full pipeline:

```bash
pytest              # with coverage
pytest --no-cov     # faster
ruff check src/     # lint
mypy src/           # type check
```

- **Binary roundtrip**: `parse(write(parse(file)))` preserves data byte-for-byte
- **All backends**: mocked dependencies, device selection, model variants
- **Parser robustness**: empty files, truncated headers, corrupt zlib, zero-length subrecords
- **CLI integration**: Typer CliRunner, batch pipeline, verbose/quiet modes, error isolation
- **Stress tests**: 500-record files through full parse → extract → translate → roundtrip

## Known Limitations

1. **Skyrim/Fallout 4/Starfield** use external string tables (`.STRINGS`, `.ILSTRINGS`, `.DLSTRINGS`) instead of inline strings — not yet supported.
2. **NLLB hallucinations on short strings** — NLLB duplicates 1–3 word inputs. The hybrid backend routes these to tc-big instead.

## Requirements

- Python >= 3.10
- Windows, macOS, or Linux
- GPU optional (NVIDIA CUDA for acceleration)

## License

This project is licensed under the MIT License.
