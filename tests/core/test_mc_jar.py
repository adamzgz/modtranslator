"""Tests for Minecraft JAR (ZIP) file handling."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from modtranslator.core.mc_jar import (
    is_jar_signed,
    rebuild_jar_with_lang,
    scan_jar,
)


def _make_jar(tmp_path: Path, name: str, files: dict[str, str | bytes]) -> Path:
    """Create a test JAR with given files."""
    jar_path = tmp_path / name
    with zipfile.ZipFile(jar_path, "w") as zf:
        for path, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(path, content)
    return jar_path


class TestIsJarSigned:
    def test_unsigned(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "test.jar", {"file.txt": "hello"})
        with zipfile.ZipFile(jar, "r") as zf:
            assert not is_jar_signed(zf)

    def test_signed_sf(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "signed.jar", {
            "file.txt": "hello",
            "META-INF/MANIFEST.MF": "manifest",
            "META-INF/CERT.SF": "signature",
        })
        with zipfile.ZipFile(jar, "r") as zf:
            assert is_jar_signed(zf)

    def test_signed_rsa(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "signed.jar", {
            "META-INF/CERT.RSA": "rsa data",
        })
        with zipfile.ZipFile(jar, "r") as zf:
            assert is_jar_signed(zf)


class TestScanJar:
    def test_basic_json(self, tmp_path: Path):
        en_us = json.dumps({"item.sword": "Iron Sword"})
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })
        result = scan_jar(jar, "es_es")
        assert not result.is_signed
        assert len(result.lang_entries) == 1
        entry = result.lang_entries[0]
        assert entry.mod_id == "mymod"
        assert entry.format == "json"
        assert entry.en_us_content == en_us

    def test_with_existing_target(self, tmp_path: Path):
        en_us = json.dumps({"key1": "Hello", "key2": "World"})
        es_es = json.dumps({"key1": "Hola"})
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
            "assets/mymod/lang/es_es.json": es_es,
        })
        result = scan_jar(jar, "es_es")
        entry = result.lang_entries[0]
        assert entry.target_content is not None
        assert "Hola" in entry.target_content

    def test_signed_jar_skipped(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "signed.jar", {
            "META-INF/CERT.SF": "sig",
            "assets/mymod/lang/en_us.json": '{"key": "val"}',
        })
        result = scan_jar(jar, "es_es")
        assert result.is_signed
        assert len(result.lang_entries) == 0

    def test_no_lang_files(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "nolang.jar", {
            "assets/mymod/textures/block.png": b"\x89PNG",
        })
        result = scan_jar(jar, "es_es")
        assert len(result.lang_entries) == 0

    def test_bad_zip(self, tmp_path: Path):
        bad_jar = tmp_path / "bad.jar"
        bad_jar.write_bytes(b"not a zip file")
        result = scan_jar(bad_jar, "es_es")
        assert result.error is not None

    def test_legacy_lang(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "legacy.jar", {
            "assets/oldmod/lang/en_US.lang": "item.sword=Iron Sword",
        })
        result = scan_jar(jar, "es_es")
        assert len(result.lang_entries) == 1
        assert result.lang_entries[0].format == "lang"

    def test_multiple_mods_in_jar(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "multi.jar", {
            "assets/mod1/lang/en_us.json": '{"k1": "v1"}',
            "assets/mod2/lang/en_us.json": '{"k2": "v2"}',
        })
        result = scan_jar(jar, "es_es")
        assert len(result.lang_entries) == 2


class TestRebuildJar:
    def test_add_new_lang(self, tmp_path: Path):
        en_us = json.dumps({"key": "Hello"})
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
            "assets/mymod/textures/icon.png": b"\x89PNG",
        })

        es_es = json.dumps({"key": "Hola"}).encode("utf-8")
        rebuild_jar_with_lang(jar, {"assets/mymod/lang/es_es.json": es_es})

        # Verify the rebuild
        with zipfile.ZipFile(jar, "r") as zf:
            names = zf.namelist()
            assert "assets/mymod/lang/es_es.json" in names
            assert "assets/mymod/lang/en_us.json" in names
            assert "assets/mymod/textures/icon.png" in names

            data = json.loads(zf.read("assets/mymod/lang/es_es.json"))
            assert data["key"] == "Hola"

    def test_replace_existing_lang(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": '{"key": "Hello"}',
            "assets/mymod/lang/es_es.json": '{"key": "old"}',
        })

        new_content = json.dumps({"key": "Hola"}).encode("utf-8")
        rebuild_jar_with_lang(jar, {"assets/mymod/lang/es_es.json": new_content})

        with zipfile.ZipFile(jar, "r") as zf:
            data = json.loads(zf.read("assets/mymod/lang/es_es.json"))
            assert data["key"] == "Hola"
            # Should not have duplicates
            assert zf.namelist().count("assets/mymod/lang/es_es.json") == 1

    def test_output_to_different_path(self, tmp_path: Path):
        jar = _make_jar(tmp_path, "original.jar", {
            "assets/mymod/lang/en_us.json": '{"key": "Hello"}',
        })
        output = tmp_path / "output.jar"

        rebuild_jar_with_lang(
            jar,
            {"assets/mymod/lang/es_es.json": b'{"key": "Hola"}'},
            output_path=output,
        )

        assert output.exists()
        # Original should still exist unchanged
        with zipfile.ZipFile(jar, "r") as zf:
            assert "assets/mymod/lang/es_es.json" not in zf.namelist()

    def test_atomic_write_cleanup_on_error(self, tmp_path: Path):
        """If rebuild fails, temp file should be cleaned up."""
        jar = tmp_path / "nonexistent.jar"
        with pytest.raises(FileNotFoundError):
            rebuild_jar_with_lang(jar, {"test": b"data"})
        # No temp files left
        assert not list(tmp_path.glob("*.tmp"))
