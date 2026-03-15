"""Tests for Minecraft JAR batch translation pipeline."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from modtranslator.pipeline import batch_translate_mc


def _make_jar(tmp_path: Path, name: str, files: dict[str, str | bytes]) -> Path:
    jar_path = tmp_path / name
    with zipfile.ZipFile(jar_path, "w") as zf:
        for path, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(path, content)
    return jar_path


@pytest.fixture()
def dummy_backend():
    from modtranslator.backends.dummy import DummyBackend
    return DummyBackend()


class TestBatchTranslateMc:
    def test_basic_translation(self, tmp_path: Path, dummy_backend):
        en_us = json.dumps({"item.sword": "Iron Sword", "item.bow": "Bow"})
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
        )

        assert result.success_count == 1
        assert result.error_count == 0

        # Verify the JAR was modified
        with zipfile.ZipFile(jar, "r") as zf:
            assert "assets/mymod/lang/es_es.json" in zf.namelist()
            data = json.loads(zf.read("assets/mymod/lang/es_es.json"))
            assert "item.sword" in data
            assert "item.bow" in data

    def test_signed_jar_skipped(self, tmp_path: Path, dummy_backend):
        jar = _make_jar(tmp_path, "signed.jar", {
            "META-INF/CERT.SF": "sig",
            "assets/mymod/lang/en_us.json": '{"key": "Hello"}',
        })

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
        )

        assert result.skip_count == 1
        assert result.success_count == 0

    def test_no_lang_files_skipped(self, tmp_path: Path, dummy_backend):
        jar = _make_jar(tmp_path, "empty.jar", {
            "assets/mymod/textures/icon.png": b"PNG",
        })

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
        )

        assert result.skip_count == 1

    def test_output_dir(self, tmp_path: Path, dummy_backend):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        en_us = json.dumps({"key": "Hello"})
        jar = _make_jar(input_dir, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            output_dir=output_dir,
            no_cache=True,
        )

        assert result.success_count == 1
        output_jar = output_dir / "mod.jar"
        assert output_jar.exists()

    def test_existing_translation_preserved(self, tmp_path: Path, dummy_backend):
        en_us = json.dumps({"key1": "Hello", "key2": "World"})
        es_es = json.dumps({"key1": "Hola manual"})
        jar = _make_jar(tmp_path, "partial.jar", {
            "assets/mymod/lang/en_us.json": en_us,
            "assets/mymod/lang/es_es.json": es_es,
        })

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
            skip_translated=False,
        )

        assert result.success_count == 1
        with zipfile.ZipFile(jar, "r") as zf:
            data = json.loads(zf.read("assets/mymod/lang/es_es.json"))
            # Existing translation should be preserved
            assert data["key1"] == "Hola manual"
            # New key should be translated (dummy prefixes with [LANG])
            assert "World" in data["key2"]

    def test_format_specifiers_preserved(self, tmp_path: Path, dummy_backend):
        en_us = json.dumps({"msg": "Hello %s, you have %d items"})
        jar = _make_jar(tmp_path, "fmt.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
            skip_translated=False,
        )

        assert result.success_count == 1
        with zipfile.ZipFile(jar, "r") as zf:
            data = json.loads(zf.read("assets/mymod/lang/es_es.json"))
            # Dummy returns original text — format specifiers should be intact
            assert "%s" in data["msg"]
            assert "%d" in data["msg"]

    def test_multiple_jars(self, tmp_path: Path, dummy_backend):
        jar1 = _make_jar(tmp_path, "mod1.jar", {
            "assets/mod1/lang/en_us.json": '{"k1": "Hello"}',
        })
        jar2 = _make_jar(tmp_path, "mod2.jar", {
            "assets/mod2/lang/en_us.json": '{"k2": "World"}',
        })

        result = batch_translate_mc(
            [jar1, jar2],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
        )

        assert result.success_count == 2

    def test_skip_non_alpha_values(self, tmp_path: Path, dummy_backend):
        en_us = json.dumps({"numeric": "123", "alpha": "Hello"})
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
            skip_translated=False,
        )

        assert result.success_count == 1
        with zipfile.ZipFile(jar, "r") as zf:
            data = json.loads(zf.read("assets/mymod/lang/es_es.json"))
            # "Hello" should be translated (dummy prefixes with [LANG])
            assert "Hello" in data["alpha"]
            # "123" is not translatable — value comes from en_us
            assert data["numeric"] == "123"

    def test_bad_jar(self, tmp_path: Path, dummy_backend):
        bad_jar = tmp_path / "bad.jar"
        bad_jar.write_bytes(b"not a zip file")

        result = batch_translate_mc(
            [bad_jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
        )

        assert result.error_count == 1

    def test_different_languages(self, tmp_path: Path, dummy_backend):
        en_us = json.dumps({"key": "Hello"})
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })

        result = batch_translate_mc(
            [jar],
            lang="FR",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
        )

        assert result.success_count == 1
        with zipfile.ZipFile(jar, "r") as zf:
            assert "assets/mymod/lang/fr_fr.json" in zf.namelist()

    def test_indent_preserved(self, tmp_path: Path, dummy_backend):
        # 4-space indented JSON
        en_us = json.dumps({"key": "Hello"}, indent=4)
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })

        result = batch_translate_mc(
            [jar],
            lang="ES",
            backend=dummy_backend,
            backend_label="dummy",
            no_cache=True,
            skip_translated=False,
        )

        assert result.success_count == 1
        with zipfile.ZipFile(jar, "r") as zf:
            content = zf.read("assets/mymod/lang/es_es.json").decode("utf-8")
            # Should use 4-space indent
            assert "\n    " in content
