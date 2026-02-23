# Changelog

All notable changes to this project will be documented in this file.

---

## [0.2.0] - 2026-02-22

### Added

- **Skyrim SE/AE support** — ESP/ESM plugins with string tables (`.STRINGS`, `.DLSTRINGS`, `.ILSTRINGS`). LOCALIZED flag auto-detected; strings loaded from `{Plugin}_English.*` and written to `{Plugin}_Spanish.*`.
- **Fallout 4 support** — ESP/ESM (inline + string tables), PEX scripts (little-endian), MCM files. String tables use short language codes (`_En`/`_es`); both naming conventions tried automatically.
- **PEX script translation** (`batch-pex`) — Compiled Papyrus scripts for Skyrim (big-endian) and Fallout 4 (little-endian). Only pure `0x02` string literals translated; `0x01` identifiers preserved.
- **MCM translation file support** (`batch-mcm`) — UTF-16-LE tab-separated files and MCM Recorder JSON. HTML-like tags preserved across translation.
- **GUI** — CustomTkinter interface with backup/restore, progress bar, model download indicator, and CUDA auto-detection.
- **Multi-language support** — 6 new target languages: French (FR), German (DE), Italian (IT), Portuguese (PT), Russian (RU), Polish (PL).
- **Glossaries for all languages** — `fallout_base`, `fallout3`, `falloutnv`, `fallout4`, `skyrim_base` glossaries for all 7 target languages with official Bethesda localisation terms.
- **GUI i18n** — Interface language follows the selected target language automatically.
- **`batch-translate-all` command** — Translates ESP/ESM + PEX + MCM in a single pass.
- **Reporting** — JSON/Markdown/CSV output with strings translated, cache hits, and duration.

### Fixed

- Patcher string counter off-by-one after multi-string subrecords.
- `restore_batch` raised on missing files instead of skipping silently.
- Dead code in `encode_string` path removed.
- Writer import error in edge-case module resolution.
- PEX error count not propagated to batch summary.

### Changed

- `Development Status` classifier promoted to `3 - Alpha`.
- Backend table in README updated to reflect available backends.

---

## [0.1.0] - 2026-02-08

### Added

- **Fallout 3 / Fallout: New Vegas support** — ESP/ESM binary parser/writer with full bottom-up size recalculation. cp1252 string encoding with UTF-8 → cp1252 → latin-1 fallback chain.
- **Translatable record registry** — Curated list of `(record_type, subrecord_type)` pairs safe for translation (FULL, DESC, NAM1, etc.).
- **CTranslate2 backends** — `opus-mt` (base and tc-big), `nllb` (1.3B and 3.3B), `hybrid` (Opus-MT for short strings, NLLB for longer ones).
- **DeepL backend** — Cloud API translation with `DEEPL_API_KEY` environment variable.
- **SQLite translation cache** — Deduplication across batches; previously translated strings never re-sent to the model.
- **Glossary system** — TOML glossaries with `Gx{i}` placeholders to protect game-specific terminology during translation.
- **Spanish word protection** — `Cx{i}` placeholders prevent already-Spanish words from being re-translated.
- **Language detection** — 4-layer heuristic to skip strings already in the target language.
- **CLI** — `scan`, `translate`, `batch`, `cache-info`, `cache-clear` commands via Typer + Rich.
- **Compressed record support** — zlib-compressed records (flag `0x00040000`) transparently decompressed and recompressed.
- **Test suite** — pytest with 60% coverage threshold.

---

# Registro de Cambios

Todos los cambios relevantes de este proyecto se documentan en este archivo.

---

## [0.2.0] - 2026-02-22

### Añadido

- **Soporte para Skyrim SE/AE** — plugins ESP/ESM con string tables (`.STRINGS`, `.DLSTRINGS`, `.ILSTRINGS`). Flag LOCALIZED detectado automáticamente; strings cargados desde `{Plugin}_English.*` y escritos en `{Plugin}_Spanish.*`.
- **Soporte para Fallout 4** — ESP/ESM (inline + string tables), scripts PEX (little-endian), archivos MCM. Las string tables usan códigos de idioma cortos (`_En`/`_es`); ambas convenciones de nombre se prueban automáticamente.
- **Traducción de scripts PEX** (`batch-pex`) — Scripts Papyrus compilados para Skyrim (big-endian) y Fallout 4 (little-endian). Solo se traducen literales de string `0x02` puros; los identificadores `0x01` se preservan intactos.
- **Soporte de archivos MCM** (`batch-mcm`) — Archivos tab-separated UTF-16-LE y JSON de MCM Recorder. Las etiquetas similares a HTML se preservan durante la traducción.
- **Interfaz gráfica (GUI)** — Aplicación CustomTkinter con backup/restauración, barra de progreso, indicador de descarga de modelos y detección automática de CUDA.
- **Soporte multilingüe** — 6 nuevos idiomas destino: francés (FR), alemán (DE), italiano (IT), portugués (PT), ruso (RU) y polaco (PL).
- **Glosarios para todos los idiomas** — Glosarios `fallout_base`, `fallout3`, `falloutnv`, `fallout4`, `skyrim_base` para los 7 idiomas destino con terminología oficial de las localizaciones de Bethesda.
- **GUI multilingüe (i18n)** — La interfaz se adapta automáticamente al idioma de destino seleccionado.
- **Comando `batch-translate-all`** — Traduce ESP/ESM + PEX + MCM en un solo paso.
- **Informes de traducción** — Salida en JSON/Markdown/CSV con strings traducidos, aciertos de caché y duración.

### Corregido

- Contador de strings del patcher incorrecto tras subrecords con múltiples strings.
- `restore_batch` lanzaba excepción en archivos no encontrados en vez de ignorarlos.
- Código muerto en la ruta de `encode_string` eliminado.
- Error de importación del writer en resolución de módulo en casos límite.
- Contador de errores PEX no se propagaba al resumen del batch.

### Cambiado

- Clasificador `Development Status` promovido a `3 - Alpha`.
- Tabla de backends en el README actualizada.

---

## [0.1.0] - 2026-02-08

### Añadido

- **Soporte para Fallout 3 / Fallout: New Vegas** — Parser/writer binario ESP/ESM con recalculación de tamaños bottom-up. Codificación cp1252 con cadena de fallback UTF-8 → cp1252 → latin-1.
- **Registro de records traducibles** — Lista curada de pares `(record_type, subrecord_type)` seguros para traducción (FULL, DESC, NAM1, etc.).
- **Backends CTranslate2** — `opus-mt` (base y tc-big), `nllb` (1.3B y 3.3B), `hybrid` (Opus-MT para strings cortos, NLLB para más largos).
- **Backend DeepL** — Traducción por API cloud con variable de entorno `DEEPL_API_KEY`.
- **Caché SQLite** — Deduplicación entre batches; strings ya traducidos no se reenvían al modelo.
- **Sistema de glosarios** — Archivos TOML con marcadores `Gx{i}` para proteger terminología específica del juego durante la traducción.
- **Protección de palabras en español** — Marcadores `Cx{i}` para evitar que palabras ya en español sean retraducidas.
- **Detección de idioma** — Heurística de 4 capas para omitir strings ya en el idioma de destino.
- **CLI** — Comandos `scan`, `translate`, `batch`, `cache-info`, `cache-clear` con Typer + Rich.
- **Soporte de records comprimidos** — Records con zlib (flag `0x00040000`) descomprimidos y recomprimidos de forma transparente.
- **Suite de tests** — pytest con umbral de cobertura del 60%.
