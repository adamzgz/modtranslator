# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ModTranslator GUI (--onedir build)."""

import os
import sys

block_cipher = None

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

a = Analysis(
    ['modtranslator_gui.py'],
    pathex=[SRC_DIR],
    binaries=[],
    datas=[
        # Glossary TOML files
        (os.path.join(PROJECT_ROOT, 'glossaries'), 'glossaries'),
        # Spanish dictionary for language detection
        (os.path.join(SRC_DIR, 'modtranslator', 'data'), os.path.join('modtranslator', 'data')),
    ],
    hiddenimports=[
        'modtranslator',
        'modtranslator.cli',
        'modtranslator.pipeline',
        'modtranslator.gui',
        'modtranslator.gui.app',
        'modtranslator.gui.worker',
        'modtranslator.gui.model_manager',
        'modtranslator.core',
        'modtranslator.core.parser',
        'modtranslator.core.writer',
        'modtranslator.core.plugin',
        'modtranslator.core.constants',
        'modtranslator.core.compression',
        'modtranslator.core.records',
        'modtranslator.core.string_table',
        'modtranslator.core.pex_parser',
        'modtranslator.translation',
        'modtranslator.translation.cache',
        'modtranslator.translation.extractor',
        'modtranslator.translation.glossary',
        'modtranslator.translation.lang_detect',
        'modtranslator.translation.patcher',
        'modtranslator.translation.registry',
        'modtranslator.translation.spanish_protect',
        'modtranslator.backends',
        'modtranslator.backends.base',
        'modtranslator.backends.dummy',
        'modtranslator.backends.deepl',
        'modtranslator.backends.opus_mt',
        'modtranslator.backends.nllb',
        'modtranslator.backends.hybrid',
        'modtranslator.reporting',
        'modtranslator.reporting.report',
        'modtranslator.reporting.formatters',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # torch is ~2GB and not needed (ctranslate2 has its own CUDA bindings)
        'torch',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ModTranslator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ModTranslator',
)
