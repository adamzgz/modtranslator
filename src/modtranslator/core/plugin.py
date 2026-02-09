"""Facade for loading and saving TES4 plugin files."""

from __future__ import annotations

import io
from pathlib import Path

from modtranslator.core.parser import parse_plugin
from modtranslator.core.records import PluginFile
from modtranslator.core.string_table import load_string_tables, save_string_tables
from modtranslator.core.writer import write_plugin


def load_plugin(path: str | Path) -> PluginFile:
    """Load and parse an ESP/ESM file from disk.

    If the plugin has the LOCALIZED flag, also loads the external string tables.
    """
    path = Path(path)
    with open(path, "rb") as f:
        plugin = parse_plugin(f)

    if plugin.is_localized:
        plugin.string_tables = load_string_tables(path)

    return plugin


def save_plugin(
    plugin: PluginFile,
    path: str | Path,
    output_language: str = "Spanish",
) -> None:
    """Serialize and write a PluginFile to disk.

    If the plugin has string tables, also writes the external string table files.
    """
    path = Path(path)
    with open(path, "wb") as f:
        write_plugin(plugin, f)

    if plugin.string_tables is not None:
        save_string_tables(plugin.string_tables, path, language=output_language)


def plugin_to_bytes(plugin: PluginFile) -> bytes:
    """Serialize a PluginFile to bytes in memory."""
    buf = io.BytesIO()
    write_plugin(plugin, buf)
    return buf.getvalue()


def plugin_from_bytes(data: bytes) -> PluginFile:
    """Parse a PluginFile from raw bytes."""
    buf = io.BytesIO(data)
    return parse_plugin(buf)
