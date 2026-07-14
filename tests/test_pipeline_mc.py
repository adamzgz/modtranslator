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


def _make_kubejs(tmp_path: Path, namespace: str, entries: dict) -> Path:
    lang_dir = tmp_path / "kubejs" / "assets" / namespace / "lang"
    lang_dir.mkdir(parents=True)
    en_us = lang_dir / "en_us.json"
    en_us.write_text(json.dumps(entries), encoding="utf-8")
    return en_us


class TestBatchTranslateKubejs:
    def test_loose_lang_translated(self, tmp_path: Path, dummy_backend):
        en_us = _make_kubejs(tmp_path, "mymod", {"a": "Hello", "b": "World"})

        result = batch_translate_mc(
            [], lang="ES", backend=dummy_backend, backend_label="dummy",
            no_cache=True, kubejs_files=[en_us],
        )
        assert result.success_count == 1
        data = json.loads((en_us.parent / "es_es.json").read_text(encoding="utf-8"))
        assert "Hello" in data["a"]
        assert "World" in data["b"]

    def test_kubejs_not_in_resource_pack(self, tmp_path: Path, dummy_backend):
        # KubeJS lang must be written to disk even in resource-pack mode.
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mod/lang/en_us.json": json.dumps({"j": "JarText"}),
        })
        en_us = _make_kubejs(tmp_path, "kjmod", {"k": "KubeText"})
        rp = tmp_path / "pack.zip"

        result = batch_translate_mc(
            [jar], lang="ES", backend=dummy_backend, backend_label="dummy",
            no_cache=True, kubejs_files=[en_us], resource_pack=rp,
        )
        assert result.success_count == 2  # jar + kubejs
        # KubeJS written to disk
        assert (en_us.parent / "es_es.json").exists()
        # Resource pack has the jar override but NOT the kubejs namespace
        with zipfile.ZipFile(rp) as zf:
            names = set(zf.namelist())
            assert "assets/mod/lang/es_es.json" in names
            assert not any("kjmod" in n for n in names)


class TestBatchTranslateResourcePack:
    def test_resource_pack_merges_and_leaves_jars(self, tmp_path: Path, dummy_backend):
        j1 = _make_jar(tmp_path, "m1.jar", {
            "assets/m1/lang/en_us.json": json.dumps({"a": "Hello"}),
        })
        j2 = _make_jar(tmp_path, "m2.jar", {
            "assets/m2/lang/en_us.json": json.dumps({"b": "World"}),
        })
        rp = tmp_path / "pack.zip"

        result = batch_translate_mc(
            [j1, j2], lang="ES", backend=dummy_backend, backend_label="dummy",
            no_cache=True, resource_pack=rp, pack_format=48,
        )
        assert result.success_count == 2
        with zipfile.ZipFile(rp) as zf:
            names = set(zf.namelist())
            assert "pack.mcmeta" in names
            assert "assets/m1/lang/es_es.json" in names
            assert "assets/m2/lang/es_es.json" in names
            meta = json.loads(zf.read("pack.mcmeta"))
            assert meta["pack"]["pack_format"] == 48
            assert meta["pack"]["description"] == "modtranslator ES overlay"
        for jar in (j1, j2):
            with zipfile.ZipFile(jar) as zf:
                assert not any("es_es" in n for n in zf.namelist())

    def test_signed_jar_included_in_resource_pack(self, tmp_path: Path, dummy_backend):
        signed = _make_jar(tmp_path, "signed.jar", {
            "META-INF/CERT.SF": "sig",
            "assets/sm/lang/en_us.json": json.dumps({"c": "Signed Text"}),
        })
        rp = tmp_path / "pack.zip"

        result = batch_translate_mc(
            [signed], lang="ES", backend=dummy_backend, backend_label="dummy",
            no_cache=True, resource_pack=rp,
        )
        assert result.success_count == 1
        with zipfile.ZipFile(rp) as zf:
            assert "assets/sm/lang/es_es.json" in zf.namelist()
        # Signed jar itself untouched
        with zipfile.ZipFile(signed) as zf:
            assert not any("es_es" in n for n in zf.namelist())


class TestBatchTranslateFtbq:
    def _make_single(self, tmp_path: Path) -> Path:
        quests = tmp_path / "config" / "ftbquests" / "quests"
        lang_dir = quests / "lang"
        lang_dir.mkdir(parents=True)
        (lang_dir / "en_us.snbt").write_text(
            "{\n"
            '    q.a.title: "&7Holystone Weapons"\n'
            "    q.a.desc: [\n"
            '        "First paragraph"\n'
            '        "Second paragraph"\n'
            "    ]\n"
            "}\n",
            encoding="utf-8",
        )
        return quests

    def test_single_file(self, tmp_path: Path, dummy_backend):
        from modtranslator.core.ftbq_lang import parse_snbt_lang
        from modtranslator.pipeline import batch_translate_ftbq

        quests = self._make_single(tmp_path)
        result = batch_translate_ftbq(
            quests, lang="ES", backend=dummy_backend,
            backend_label="dummy", no_cache=True,
        )
        assert result.success_count == 1
        out = quests / "lang" / "es_es.snbt"
        data = parse_snbt_lang(out.read_text(encoding="utf-8"))
        # &7 code survives, list preserved
        assert "&7" in data["q.a.title"]
        assert len(data["q.a.desc"]) == 2

    def test_folder_layout(self, tmp_path: Path, dummy_backend):
        from modtranslator.core.ftbq_lang import parse_snbt_lang
        from modtranslator.pipeline import batch_translate_ftbq

        quests = tmp_path / "config" / "ftbquests" / "quests"
        base = quests / "lang" / "en_us" / "chapters"
        base.mkdir(parents=True)
        (base / "c1.snbt").write_text(
            '{\n    q.x.title: "Hello there"\n}\n', encoding="utf-8"
        )
        result = batch_translate_ftbq(
            quests, lang="ES", backend=dummy_backend,
            backend_label="dummy", no_cache=True,
        )
        assert result.success_count == 1
        out = quests / "lang" / "es_es" / "chapters" / "c1.snbt"
        data = parse_snbt_lang(out.read_text(encoding="utf-8"))
        assert "Hello there" in data["q.x.title"]


class TestBatchTranslateAll:
    def test_all_detects_kubejs_and_ftbq(self, tmp_path: Path, dummy_backend):
        from modtranslator.pipeline import batch_translate_all

        _make_jar(tmp_path, "mod.jar", {
            "assets/mod/lang/en_us.json": json.dumps({"j": "JarText"}),
        })
        _make_kubejs(tmp_path, "kjmod", {"k": "KubeText"})
        quests = tmp_path / "config" / "ftbquests" / "quests"
        (quests / "lang").mkdir(parents=True)
        (quests / "lang" / "en_us.snbt").write_text(
            '{\n    q.a: "Quest text here"\n}\n', encoding="utf-8"
        )

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = batch_translate_all(
            tmp_path, output_dir=out_dir, lang="ES",
            backend=dummy_backend, backend_label="dummy", no_cache=True,
        )
        assert result.mc_result is not None
        assert result.ftbq_result is not None
        assert result.ftbq_result.success_count == 1
        # KubeJS + JAR both handled by mc_result
        assert result.mc_result.success_count == 2
