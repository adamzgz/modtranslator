# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for modtranslator GUI — Windows x64.

Bundles: core + GUI + CTranslate2 + sentencepiece + transformers (tokenizer/conversion)
Excludes: torch, tensorflow, jax (not needed at runtime after CT2 conversion)
Models: download on first use via huggingface_hub (~/.modtranslator/models/)
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ── Data files ────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("customtkinter")
datas += collect_data_files("modtranslator")          # spanish_words.txt via importlib.resources
datas += [("glossaries", "glossaries")]               # TOML glossaries

# ── Hidden imports ────────────────────────────────────────────────────────────
# collect_submodules ensures entire packages are bundled (not just top-level)
hiddenimports = [
    # GUI
    "customtkinter",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    # CTranslate2
    "ctranslate2",
    "ctranslate2.converters",
    "ctranslate2.converters.transformers",
    # Transformers tokenizers (needed for MarianTokenizer + NLLB tokenizer)
    "transformers",
    "transformers.models.marian",
    "transformers.models.marian.tokenization_marian",
    "transformers.models.nllb",
    "transformers.models.m2m_100",
    "transformers.models.m2m_100.tokenization_m2m_100",
    "sentencepiece",
    "tokenizers",
    # HuggingFace hub (model download)
    "huggingface_hub",
    "huggingface_hub.file_download",
    # Utils
    "filelock",
    "tqdm",
    "tqdm.auto",
    "regex",
    "packaging",
    "packaging.version",
    "safetensors",
    "hf_xet",
    # modtranslator internals (dynamic imports in pipeline/backends)
    "modtranslator.backends.opus_mt",
    "modtranslator.backends.nllb",
    "modtranslator.backends.hybrid",
    "modtranslator.backends.deepl",
    "modtranslator.backends.hybrid_deepl",
    "modtranslator.backends.dummy",
    "modtranslator.gui.worker",
    "modtranslator.gui.model_manager",
]

# Packages that PyInstaller misses because they're only imported dynamically
# or are transitive dependencies not reachable from static analysis.
# collect_submodules pulls all submodules so the full package lands in the bundle.
_force_collect = [
    "requests", "urllib3", "charset_normalizer", "certifi", "idna",
    "httpx", "httpcore", "h11", "anyio", "darkdetect", "deepl",
]
for _pkg in _force_collect:
    hiddenimports += collect_submodules(_pkg)

# ── Excludes (not needed at runtime) ─────────────────────────────────────────
excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tensorflow_core",
    "jax",
    "flax",
    "IPython",
    "jupyter",
    "notebook",
    "matplotlib",
    "PIL",
    "cv2",
    "sklearn",
    "scipy",
    "pandas",
    "pytest",
    "sphinx",
    "pyarrow",
    "yt_dlp",
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["modtranslator_gui.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="modtranslator-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,     # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="modtranslator-gui",
)
