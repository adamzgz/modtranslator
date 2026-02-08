"""Facade for loading and saving TES4 plugin files."""

from __future__ import annotations

import io
from pathlib import Path

from modtranslator.core.parser import parse_plugin
from modtranslator.core.records import PluginFile
from modtranslator.core.writer import write_plugin


def load_plugin(path: str | Path) -> PluginFile:
    """Load and parse an ESP/ESM file from disk."""
    path = Path(path)
    with open(path, "rb") as f:
        return parse_plugin(f)


def save_plugin(plugin: PluginFile, path: str | Path) -> None:
    """Serialize and write a PluginFile to disk."""
    path = Path(path)
    with open(path, "wb") as f:
        write_plugin(plugin, f)


def plugin_to_bytes(plugin: PluginFile) -> bytes:
    """Serialize a PluginFile to bytes in memory."""
    buf = io.BytesIO()
    write_plugin(plugin, buf)
    return buf.getvalue()


def plugin_from_bytes(data: bytes) -> PluginFile:
    """Parse a PluginFile from raw bytes."""
    buf = io.BytesIO(data)
    return parse_plugin(buf)
