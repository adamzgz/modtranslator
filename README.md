# modtranslator

**Traductor automático offline de mods para juegos Bethesda.**

Traduce archivos ESP/ESM, scripts PEX y archivos MCM de forma completamente offline usando modelos de traducción neuronal. Compatible con Fallout 3, Fallout: New Vegas y Skyrim (SE/AE).

> **Probado con los 3 juegos soportados.** Fallout 3 (124 archivos, ~97K strings), Fallout: New Vegas (244 archivos, ~142K strings) y Skyrim SE/AE (542 archivos, ~128K strings + 2.302 scripts PEX + MCM) — todos funcionan correctamente con los mods traducidos.

---

## Características

- **Traducción in-place** — traduce directamente sobre la carpeta Data/ del juego
- **Backup automático** — respalda solo los archivos que se van a modificar antes de traducir
- **Restaurar backup** — un click para volver a los archivos originales
- **GUI incluida** — interfaz gráfica con CustomTkinter, sin necesidad de terminal
- **CLI completa** — para usuarios avanzados y automatización
- **3 juegos soportados** — Fallout 3, Fallout: New Vegas, Skyrim SE/AE
- **3 tipos de archivo** — ESP/ESM (plugins), PEX (scripts de Papyrus), MCM (menús de configuración)
- **5 backends de traducción** — desde modelos offline hasta APIs cloud
- **Cache de traducciones** — no retraduce strings ya procesados
- **Glosarios por juego** — protege terminología oficial (Refugio, Estimulante, Sanguinario...)

## Juegos Soportados

| Juego | Estado | Tipos de archivo |
|-------|--------|------------------|
| **Fallout 3** | Probado — 124 archivos, ~97K strings | ESP/ESM |
| **Fallout: New Vegas** | Probado — 244 archivos, ~142K strings | ESP/ESM |
| **Skyrim SE/AE** | Probado — 542 archivos, ~128K strings + 2.302 PEX + MCM | ESP/ESM, PEX, MCM |

## Resultados

| Juego | Archivos | Strings | Errores | Juego funcional |
|-------|----------|---------|---------|-----------------|
| Fallout 3 | 124 | ~97K | 0 | Sí |
| Fallout: New Vegas | 244 | ~142K | 0 | Sí |
| Skyrim SE/AE | 542 + 2.302 PEX + MCM | ~128K | 0 | Sí |

## GUI

La interfaz gráfica permite traducir mods sin usar la terminal:

1. Selecciona la carpeta `Data/` del juego
2. Elige el motor de traducción
3. Pulsa **Traducir**

La GUI hace backup automático de los archivos que va a modificar. Si algo sale mal, pulsa **Restaurar backup** para volver a los originales.

### Ejecutable (Windows)

Descarga el ZIP de la sección Releases — no necesita Python ni dependencias.

### Desde código

```bash
pip install -e ".[opus-mt,nllb]"
python modtranslator_gui.py
```

## Instalación (CLI)

```bash
git clone https://github.com/adamzgz/modtranslator.git
cd modtranslator

# Instalación base (CLI + DeepL)
pip install -e .

# Traducción offline — recomendado
pip install -e ".[opus-mt,nllb]"

# Desarrollo (tests, lint, type checking)
pip install -e ".[dev]"
```

### Aceleración GPU (Opcional)

```bash
pip install torch nvidia-cublas-cu12
# Usar --device cuda o --device auto
```

## Uso rápido (CLI)

```bash
# Escanear un mod para ver strings traducibles
modtranslator scan MyMod.esp

# Traducir un archivo
modtranslator translate MyMod.esp --backend hybrid --device cuda

# Traducir toda una carpeta de juego
modtranslator batch "C:\Games\Fallout 3\Data" --backend hybrid --device cuda --game fo3

# Traducir scripts PEX de Skyrim
modtranslator batch-pex "C:\Games\Skyrim\Data" --backend opus-mt

# Traducir archivos MCM de Skyrim
modtranslator batch-mcm "C:\Games\Skyrim\Data" --backend opus-mt
```

## Backends de Traducción

| Backend | Velocidad (GPU) | Coste | Calidad EN→ES | Offline |
|---------|-----------------|-------|---------------|---------|
| **Híbrido (tc-big+NLLB)** | ~33 str/s | Gratis | Mejor combinada | Sí |
| **Opus-MT tc-big** | 241–468 str/s | Gratis | Muy buena | Sí |
| Opus-MT base | 188–291 str/s | Gratis | Buena | Sí |
| NLLB 1.3B | 32–71 str/s | Gratis | Excelente (texto largo) | Sí |
| DeepL | Online | API key | Muy buena | No |

El backend **híbrido** enruta strings cortos (1–3 palabras) a Opus-MT tc-big y strings largos (4+ palabras) a NLLB 1.3B — combinando la velocidad de tc-big con la fluidez de NLLB en oraciones complejas.

## Cómo Funciona

```
Archivo ESP/ESM (binario, codificación Windows-1252)
    │
    ▼
1. PARSEAR ────────── Lector binario → Árbol Record/Group (formato TES4)
    │                  Gestiona records comprimidos (zlib), cabeceras de 24 bytes
    ▼
2. EXTRAER ────────── El registro filtra subrecords traducibles
    │                  FULL (37 tipos), DESC, NAM1, RNAM, TNAM, NNAM, ITXT, CNAM
    ▼
3. DETECTAR IDIOMA ── Heurística de 4 capas para saltar strings ya traducidos
    │                  glosario → langid → diccionario español → diccionario inglés
    ▼
4. PROTEGER ───────── Términos del glosario → placeholders Gx0, Gx1
    │                  Palabras españolas en strings mixtos → placeholders Cx0, Cx1
    ▼
5. TRADUCIR ───────── El backend procesa strings en lotes
    │                  Cache SQLite para saltar strings ya traducidos
    ▼
6. RESTAURAR ──────── Placeholders → términos originales protegidos
    │
    ▼
7. ESCRIBIR ───────── Serializar árbol con tamaños recalculados → ESP/ESM válido
```

## Glosarios por Juego

Glosarios TOML que protegen la terminología oficial de Bethesda:

```toml
[terms]
"Vault" = "Refugio"
"Stimpak" = "Estimulante"
"Brotherhood of Steel" = "Hermandad del Acero"
"Deathclaw" = "Sanguinario"
```

| Glosario | Alcance | Ejemplos |
|----------|---------|----------|
| `fallout_base_es.toml` | Compartido FO3/FNV | Stats SPECIAL, consumibles, facciones, armas |
| `fallout3_es.toml` | Capital Wasteland | Megaton, Three Dog, Talon Company |
| `falloutnv_es.toml` | Mojave Wasteland | NCR→RNC, Mr. House→Sr. House, Yes Man→Servibot |
| `skyrim_es.toml` | Skyrim | Dovahkiin, Whiterun→Carrera Blanca, Stormcloaks→Capas de la Tormenta |

## Arquitectura

```
src/modtranslator/
├── cli.py                  CLI con Typer, pipeline batch (3 fases secuenciales)
├── pipeline.py             Lógica compartida CLI/GUI
├── core/                   Parser/writer binario ESP/ESM + PEX
│   ├── parser.py             bytes → árbol de Records (formato TES4)
│   ├── writer.py             árbol de Records → bytes (recálculo bottom-up)
│   ├── records.py            Dataclasses: Subrecord, Record, GroupRecord, PluginFile
│   ├── string_table.py       String tables externas de Skyrim (.STRINGS/.DLSTRINGS/.ILSTRINGS)
│   ├── pex_parser.py         Parser de scripts PEX de Papyrus
│   └── plugin.py             Fachada: load_plugin / save_plugin
├── backends/               Backends de traducción (TranslationBackend ABC)
│   ├── opus_mt.py            Helsinki-NLP + CTranslate2 (base + tc-big)
│   ├── nllb.py               Meta NLLB-200 + CTranslate2 (600M/1.3B)
│   ├── hybrid.py             tc-big (corto) + NLLB (largo)
│   ├── deepl.py              API de DeepL
│   └── dummy.py              Backend de test: prefija [XX]
├── translation/            Extracción, filtrado y parcheo de strings
│   ├── extractor.py          árbol → list[TranslatableString]
│   ├── registry.py           (record_type, sub_type) → ¿traducible?
│   ├── glossary.py           Glosarios TOML + sistema de placeholders Gx{i}
│   ├── spanish_protect.py    Protección de palabras españolas con placeholders Cx{i}
│   ├── lang_detect.py        Heurística de detección de idioma (4 capas)
│   ├── patcher.py            Aplicar traducciones a bytearrays de subrecords
│   └── cache.py              Cache SQLite de traducciones (~/.modtranslator/)
├── gui/                    Interfaz gráfica con CustomTkinter
│   ├── app.py                Ventana principal, backup selectivo, traducción in-place
│   ├── worker.py             Worker en background thread
│   └── model_manager.py      Detección GPU y gestión de modelos
└── data/                   Diccionario español (~2.100 palabras)

glossaries/                 Archivos TOML de terminología por juego
tests/                      378 tests, ~84% cobertura
```

## Tests

```bash
pytest              # con cobertura
pytest --no-cov     # más rápido
ruff check src/     # lint
mypy src/           # type check
```

## Requisitos

- Python >= 3.10
- Windows, macOS o Linux
- GPU opcional (NVIDIA CUDA para aceleración)

## Licencia

Este proyecto está bajo la licencia MIT.

---
---

# modtranslator (English)

**Automatic offline translator for Bethesda game mods.**

Translates ESP/ESM files, PEX scripts, and MCM files completely offline using neural machine translation models. Compatible with Fallout 3, Fallout: New Vegas, and Skyrim (SE/AE).

> **Tested on all 3 supported games.** Fallout 3 (124 files, ~97K strings), Fallout: New Vegas (244 files, ~142K strings) and Skyrim SE/AE (542 files, ~128K strings + 2,302 PEX scripts + MCM) — all games work correctly with translated mods.

## Features

- **In-place translation** — translates directly in the game's Data/ folder
- **Automatic backup** — backs up only the files that will be modified before translating
- **Restore backup** — one click to revert to original files
- **GUI included** — graphical interface with CustomTkinter, no terminal needed
- **Full CLI** — for advanced users and automation
- **3 supported games** — Fallout 3, Fallout: New Vegas, Skyrim SE/AE
- **3 file types** — ESP/ESM (plugins), PEX (Papyrus scripts), MCM (configuration menus)
- **5 translation backends** — from offline models to cloud APIs
- **Translation cache** — skips previously translated strings
- **Per-game glossaries** — protects official terminology

## Supported Games

| Game | Status | File types |
|------|--------|------------|
| **Fallout 3** | Tested — 124 files, ~97K strings | ESP/ESM |
| **Fallout: New Vegas** | Tested — 244 files, ~142K strings | ESP/ESM |
| **Skyrim SE/AE** | Tested — 542 files, ~128K strings + 2,302 PEX + MCM | ESP/ESM, PEX, MCM |

## GUI

The graphical interface lets you translate mods without using the terminal:

1. Select the game's `Data/` folder
2. Choose the translation engine
3. Click **Translate**

The GUI automatically backs up files before modifying them. If anything goes wrong, click **Restore backup** to revert.

### Executable (Windows)

Download the ZIP from Releases — no Python or dependencies needed.

### From source

```bash
pip install -e ".[opus-mt,nllb]"
python modtranslator_gui.py
```

## Installation (CLI)

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

## Quick Start (CLI)

```bash
# Scan a mod to preview translatable strings
modtranslator scan MyMod.esp

# Translate a single file
modtranslator translate MyMod.esp --backend hybrid --device cuda

# Translate an entire game folder
modtranslator batch "C:\Games\Fallout 3\Data" --backend hybrid --device cuda --game fo3

# Translate Skyrim PEX scripts
modtranslator batch-pex "C:\Games\Skyrim\Data" --backend opus-mt

# Translate Skyrim MCM files
modtranslator batch-mcm "C:\Games\Skyrim\Data" --backend opus-mt
```

## Translation Backends

| Backend | Speed (GPU) | Cost | Quality EN→ES | Offline |
|---------|-------------|------|---------------|---------|
| **Hybrid (tc-big+NLLB)** | ~33 str/s | Free | Best combined | Yes |
| **Opus-MT tc-big** | 241–468 str/s | Free | Very good | Yes |
| Opus-MT base | 188–291 str/s | Free | Good | Yes |
| NLLB 1.3B | 32–71 str/s | Free | Excellent (long text) | Yes |
| DeepL | Online | API key | Very good | No |

The **hybrid** backend routes short strings (1–3 words) to Opus-MT tc-big and longer strings (4+ words) to NLLB 1.3B — combining tc-big's speed with NLLB's fluency on complex sentences.

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
    ▼
3. DETECT LANGUAGE ─ 4-layer heuristic skips already-translated strings
    │                 glossary → langid → spanish dict → english dict
    ▼
4. PROTECT ───────── Glossary terms → Gx0, Gx1 placeholders
    │                 Spanish words in mixed strings → Cx0, Cx1 placeholders
    ▼
5. TRANSLATE ─────── Backend processes strings in batches
    │                 SQLite cache skips previously translated strings
    ▼
6. RESTORE ───────── Placeholders → original protected terms
    │
    ▼
7. WRITE ─────────── Serialize tree with recalculated sizes → valid ESP/ESM
```

## Per-Game Glossaries

TOML glossaries protect official Bethesda terminology from machine translation:

```toml
[terms]
"Vault" = "Refugio"
"Stimpak" = "Stimpack"
"Brotherhood of Steel" = "Hermandad del Acero"
"Deathclaw" = "Sanguinario"
```

| Glossary | Scope | Examples |
|----------|-------|----------|
| `fallout_base_es.toml` | Shared FO3/FNV | SPECIAL stats, consumables, factions, weapons |
| `fallout3_es.toml` | Capital Wasteland | Megaton, Three Dog, Talon Company |
| `falloutnv_es.toml` | Mojave Wasteland | NCR→RNC, Mr. House→Sr. House, Yes Man→Servibot |
| `skyrim_es.toml` | Skyrim | Dovahkiin, Whiterun→Carrera Blanca, Stormcloaks→Capas de la Tormenta |

## Tests

```bash
pytest              # with coverage
pytest --no-cov     # faster
ruff check src/     # lint
mypy src/           # type check
```

## Requirements

- Python >= 3.10
- Windows, macOS, or Linux
- GPU optional (NVIDIA CUDA for acceleration)

## License

This project is licensed under the MIT License.
