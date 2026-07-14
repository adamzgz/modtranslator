"""Tests for the JSON exchange export/apply flow."""

from __future__ import annotations

import json
import shutil
import struct
import zipfile
from pathlib import Path

import pytest

from modtranslator._pipeline_helpers import scan_file
from modtranslator._pipeline_types import GameChoice
from modtranslator.core.pex_parser import PEX_MAGIC, load_pex
from modtranslator.exchange import FormatChoice, apply_path, export_path

FIXTURES = Path(__file__).parent / "fixtures"
ESP_KEY = "minimal_fo3:00001000:FULL:0"


def _copy_esp(tmp_path: Path) -> Path:
    esp = tmp_path / "minimal_fo3.esp"
    shutil.copy(FIXTURES / "minimal_fo3.esp", esp)
    return esp


def _make_pex_bytes(strings: list[str], post_data: bytes = b"") -> bytes:
    parts: list[bytes] = []
    parts.append(struct.pack(">I", PEX_MAGIC))
    parts.append(struct.pack(">BB", 3, 2))
    parts.append(struct.pack(">H", 1))
    parts.append(struct.pack(">Q", 0))
    for _ in range(3):
        parts.append(struct.pack(">H", 0))
    parts.append(struct.pack(">H", len(strings)))
    for s in strings:
        encoded = s.encode("utf-8")
        parts.append(struct.pack(">H", len(encoded)))
        parts.append(encoded)
    parts.append(post_data)
    return b"".join(parts)


def _make_jar(tmp_path: Path, name: str, files: dict[str, str | bytes]) -> Path:
    jar_path = tmp_path / name
    with zipfile.ZipFile(jar_path, "w") as zf:
        for path, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(path, content)
    return jar_path


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch):
    """Point the default cache DB at a per-test temp file."""
    monkeypatch.setattr(
        "modtranslator.translation.cache.DEFAULT_CACHE_DB",
        tmp_path / "cache.db",
    )


class TestExportEsp:
    def test_roundtrip_esp(self, tmp_path: Path):
        esp = _copy_esp(tmp_path)

        result = export_path(esp, use_cache=False)
        assert result.files_scanned == 1
        assert result.total_entries == 1
        assert result.doc["lang"] == "ES"
        assert result.doc["game"] == "fo3"
        entry = result.doc["files"][0]["entries"][0]
        assert entry["key"] == ESP_KEY
        assert entry["text"] == "Iron Sword"
        assert entry["record"] == "WEAP"
        assert entry["type"] == "FULL"
        assert entry["edid"] == "TestWeapon"

        out = tmp_path / "out.esp"
        ap = apply_path(esp, {ESP_KEY: "Espada de hierro"}, output=out, use_cache=False)
        assert ap.strings_applied == 1
        assert ap.files_written == 1
        assert ap.unmatched_keys == []
        assert ap.spec_rejected == []

        strings = scan_file(out)
        assert any(s.original_text == "Espada de hierro" for s in strings)

    def test_apply_default_output_name(self, tmp_path: Path):
        esp = _copy_esp(tmp_path)
        ap = apply_path(esp, {ESP_KEY: "Espada de hierro"}, use_cache=False)
        assert ap.files_written == 1
        assert (tmp_path / "minimal_fo3_ES.esp").exists()

    def test_unmatched_keys(self, tmp_path: Path):
        esp = _copy_esp(tmp_path)
        ap = apply_path(
            esp,
            {ESP_KEY: "Espada de hierro", "does:not:exist:0": "Nada"},
            output=tmp_path / "out.esp",
            use_cache=False,
        )
        assert ap.strings_applied == 1
        assert ap.unmatched_keys == ["does:not:exist:0"]

    def test_empty_translation_discarded(self, tmp_path: Path):
        esp = _copy_esp(tmp_path)
        ap = apply_path(esp, {ESP_KEY: ""}, output=tmp_path / "out.esp", use_cache=False)
        assert ap.strings_applied == 0
        assert ap.files_written == 0
        assert ap.unmatched_keys == []


class TestExportCache:
    def test_cache_excludes_already_translated(self, tmp_path: Path):
        esp = _copy_esp(tmp_path)

        r1 = export_path(esp, use_cache=True)
        assert r1.total_entries == 1
        assert r1.cache_skipped == 0

        apply_path(
            esp, {ESP_KEY: "Espada de hierro"},
            output=tmp_path / "out.esp", use_cache=True,
        )

        r2 = export_path(esp, use_cache=True)
        assert r2.total_entries == 0
        assert r2.cache_skipped == 1

    def test_apply_writes_cache(self, tmp_path: Path):
        from modtranslator.translation.cache import TranslationCache

        esp = _copy_esp(tmp_path)
        apply_path(
            esp, {ESP_KEY: "Espada de hierro"},
            output=tmp_path / "out.esp", use_cache=True,
            backend_label="claude-agent",
        )

        cache = TranslationCache()
        try:
            assert cache.get(ESP_KEY, "ES") == "Espada de hierro"
        finally:
            cache.close()


class TestExportPex:
    def test_pex_roundtrip(self, tmp_path: Path):
        strings = ["identifier", "Restore health to the player now"]
        post = b"\x01" + struct.pack(">H", 0) + b"\x02" + struct.pack(">H", 1)
        pex_path = tmp_path / "script.pex"
        pex_path.write_bytes(_make_pex_bytes(strings, post))

        result = export_path(pex_path, use_cache=False)
        assert result.total_entries == 1
        entry = result.doc["files"][0]["entries"][0]
        assert entry["key"] == "Restore health to the player now"
        assert result.doc["files"][0]["format"] == "pex"

        ap = apply_path(
            pex_path,
            {"Restore health to the player now": "Restaura la salud del jugador ahora"},
            use_cache=False,
        )
        assert ap.strings_applied == 1
        assert ap.files_written == 1

        pex = load_pex(pex_path)
        assert "Restaura la salud del jugador ahora" in pex.string_table

    def test_pex_spec_rejected(self, tmp_path: Path):
        text = "Restore %s health to the player"
        strings = ["identifier", text]
        post = b"\x01" + struct.pack(">H", 0) + b"\x02" + struct.pack(">H", 1)
        pex_path = tmp_path / "script.pex"
        pex_path.write_bytes(_make_pex_bytes(strings, post))

        # Translation drops the %s specifier → rejected.
        ap = apply_path(
            pex_path, {text: "Restaura la salud del jugador"}, use_cache=False,
        )
        assert ap.strings_applied == 0
        assert ap.spec_rejected == [text]

        pex = load_pex(pex_path)
        assert text in pex.string_table  # original preserved


class TestExportMc:
    def test_mc_roundtrip(self, tmp_path: Path):
        en_us = json.dumps({"item.sword": "Iron Sword", "item.bow": "Wooden Bow"})
        jar = _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": en_us,
        })

        result = export_path(jar, use_cache=False)
        assert result.total_entries == 2
        f = result.doc["files"][0]
        assert f["format"] == "mc"
        ids = {e["id"] for e in f["entries"]}
        assert ids == {"item.sword", "item.bow"}

        ap = apply_path(
            jar,
            {"Iron Sword": "Espada de hierro", "Wooden Bow": "Arco de madera"},
            use_cache=False,
        )
        assert ap.strings_applied == 2
        assert ap.files_written == 1

        with zipfile.ZipFile(jar, "r") as zf:
            assert "assets/mymod/lang/es_es.json" in zf.namelist()
            data = json.loads(zf.read("assets/mymod/lang/es_es.json"))
            assert data["item.sword"] == "Espada de hierro"
            assert data["item.bow"] == "Arco de madera"


class TestExportMcm:
    def _make_mcm(self, tmp_path: Path) -> Path:
        trans_dir = tmp_path / "Interface" / "translations"
        trans_dir.mkdir(parents=True)
        mcm_file = trans_dir / "Mod_english.txt"
        content = "﻿$MYKEY\tWelcome to the wasteland\n"
        mcm_file.write_bytes(content.encode("utf-16-le"))
        return trans_dir

    def test_mcm_roundtrip(self, tmp_path: Path):
        trans_dir = self._make_mcm(tmp_path)

        result = export_path(tmp_path, use_cache=False)
        assert result.total_entries == 1
        f = result.doc["files"][0]
        assert f["format"] == "mcm"
        entry = f["entries"][0]
        assert entry["id"] == "$MYKEY"
        assert entry["text"] == "Welcome to the wasteland"

        ap = apply_path(
            tmp_path,
            {"Welcome to the wasteland": "Bienvenido al yermo"},
            use_cache=False,
        )
        assert ap.strings_applied == 1
        assert ap.files_written == 1

        out_file = trans_dir / "Mod_spanish.txt"
        assert out_file.exists()
        raw = out_file.read_bytes()
        assert raw[:2] == b"\xff\xfe"
        text = raw.decode("utf-16-le").lstrip("﻿")
        assert "Bienvenido al yermo" in text


class TestExportDirectory:
    def test_mixed_directory_auto(self, tmp_path: Path):
        # ESP
        shutil.copy(FIXTURES / "minimal_fo3.esp", tmp_path / "minimal_fo3.esp")
        # MCM
        trans_dir = tmp_path / "Interface" / "translations"
        trans_dir.mkdir(parents=True)
        (trans_dir / "Mod_english.txt").write_bytes(
            "﻿$K\tWelcome to the wasteland\n".encode("utf-16-le")
        )
        # MC JAR
        _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": json.dumps({"item.sword": "Iron Sword"}),
        })

        result = export_path(tmp_path, use_cache=False)
        formats = {f["format"] for f in result.doc["files"]}
        assert formats == {"esp", "mcm", "mc"}
        assert result.total_entries == 3

    def test_format_filter(self, tmp_path: Path):
        shutil.copy(FIXTURES / "minimal_fo3.esp", tmp_path / "minimal_fo3.esp")
        _make_jar(tmp_path, "mod.jar", {
            "assets/mymod/lang/en_us.json": json.dumps({"item.sword": "Iron Sword"}),
        })

        result = export_path(tmp_path, fmt=FormatChoice.esp, use_cache=False)
        formats = {f["format"] for f in result.doc["files"]}
        assert formats == {"esp"}


class TestExportMetadata:
    def test_doc_metadata(self, tmp_path: Path):
        esp = _copy_esp(tmp_path)
        result = export_path(esp, lang="FR", game=GameChoice.fnv, use_cache=False)
        doc = result.doc
        assert doc["version"] == 1
        assert doc["generator"].startswith("modtranslator ")
        assert "created" in doc
        assert doc["lang"] == "FR"
        assert doc["game"] == "fnv"
        assert isinstance(doc["glossaries"], list)


def _make_kubejs(tmp_path: Path, namespace: str, entries: dict) -> Path:
    lang_dir = tmp_path / "kubejs" / "assets" / namespace / "lang"
    lang_dir.mkdir(parents=True)
    en_us = lang_dir / "en_us.json"
    en_us.write_text(json.dumps(entries), encoding="utf-8")
    return en_us


def _make_ftbq_single(tmp_path: Path, content: str) -> Path:
    lang_dir = tmp_path / "config" / "ftbquests" / "quests" / "lang"
    lang_dir.mkdir(parents=True)
    snbt = lang_dir / "en_us.snbt"
    snbt.write_text(content, encoding="utf-8")
    return snbt


class TestExportKubejs:
    def test_kubejs_roundtrip(self, tmp_path: Path):
        _make_kubejs(tmp_path, "mymod", {
            "block.thing": "Cool Block", "item.gadget": "Shiny Gadget",
        })

        result = export_path(tmp_path, use_cache=False)
        assert result.total_entries == 2
        f = result.doc["files"][0]
        assert f["format"] == "mc"
        ids = {e["id"] for e in f["entries"]}
        assert ids == {"block.thing", "item.gadget"}

        ap = apply_path(
            tmp_path,
            {"Cool Block": "Bloque genial", "Shiny Gadget": "Artilugio brillante"},
            use_cache=False,
        )
        assert ap.strings_applied == 2
        assert ap.files_written == 1

        es = tmp_path / "kubejs" / "assets" / "mymod" / "lang" / "es_es.json"
        assert es.exists()
        data = json.loads(es.read_text(encoding="utf-8"))
        assert data["block.thing"] == "Bloque genial"
        assert data["item.gadget"] == "Artilugio brillante"

    def test_kubejs_merges_existing(self, tmp_path: Path):
        _make_kubejs(tmp_path, "mymod", {"a": "Hello", "b": "World"})
        es_dir = tmp_path / "kubejs" / "assets" / "mymod" / "lang"
        (es_dir / "es_es.json").write_text(
            json.dumps({"a": "Hola manual"}), encoding="utf-8"
        )

        # 'a' is already translated → not exported
        result = export_path(tmp_path, use_cache=False)
        ids = {e["id"] for f in result.doc["files"] for e in f["entries"]}
        assert ids == {"b"}

        apply_path(tmp_path, {"World": "Mundo"}, use_cache=False)
        data = json.loads((es_dir / "es_es.json").read_text(encoding="utf-8"))
        assert data["a"] == "Hola manual"  # existing preserved
        assert data["b"] == "Mundo"

    def test_kubejs_batch(self, tmp_path: Path):
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import batch_translate_mc

        en_us = _make_kubejs(tmp_path, "mymod", {"a": "Hello"})
        result = batch_translate_mc(
            [], lang="ES", backend=DummyBackend(), backend_label="dummy",
            no_cache=True, kubejs_files=[en_us],
        )
        assert result.success_count == 1
        es = en_us.parent / "es_es.json"
        data = json.loads(es.read_text(encoding="utf-8"))
        assert "Hello" in data["a"]


class TestExportFtbq:
    SINGLE = (
        "{\n"
        '    quest.A.title: "&7Holystone Weapons"\n'
        "    quest.A.desc: [\n"
        '        "First paragraph"\n'
        '        "Second paragraph &7here"\n'
        "    ]\n"
        "}\n"
    )

    def test_ftbq_single_file_roundtrip(self, tmp_path: Path):
        _make_ftbq_single(tmp_path, self.SINGLE)

        result = export_path(tmp_path, fmt=FormatChoice.ftbq, use_cache=False)
        assert result.total_entries == 3
        f = result.doc["files"][0]
        assert f["format"] == "ftbq"
        ids = {e["id"] for e in f["entries"]}
        assert "lang/en_us.snbt:quest.A.title" in ids
        assert "lang/en_us.snbt:quest.A.desc:0" in ids
        assert "lang/en_us.snbt:quest.A.desc:1" in ids

        ap = apply_path(
            tmp_path,
            {
                "&7Holystone Weapons": "&7Armas de Piedra Sagrada",
                "First paragraph": "Primer parrafo",
                "Second paragraph &7here": "Segundo parrafo &7aqui",
            },
            fmt=FormatChoice.ftbq,
            use_cache=False,
        )
        assert ap.strings_applied == 3
        assert ap.files_written == 1
        assert ap.spec_rejected == []

        from modtranslator.core.ftbq_lang import parse_snbt_lang

        out = tmp_path / "config" / "ftbquests" / "quests" / "lang" / "es_es.snbt"
        assert out.exists()
        data = parse_snbt_lang(out.read_text(encoding="utf-8"))
        assert data["quest.A.title"] == "&7Armas de Piedra Sagrada"
        assert data["quest.A.desc"] == ["Primer parrafo", "Segundo parrafo &7aqui"]

    def test_ftbq_folder_roundtrip(self, tmp_path: Path):
        base = tmp_path / "config" / "ftbquests" / "quests" / "lang" / "en_us"
        (base / "chapters").mkdir(parents=True)
        (base / "chapters" / "c1.snbt").write_text(
            '{\n    q.x.title: "Sword"\n    q.x.desc: [\n        "First line here"\n    ]\n}\n',
            encoding="utf-8",
        )

        result = export_path(tmp_path, fmt=FormatChoice.ftbq, use_cache=False)
        assert result.total_entries == 2
        ids = {e["id"] for f in result.doc["files"] for e in f["entries"]}
        assert "lang/en_us/chapters/c1.snbt:q.x.title" in ids
        assert "lang/en_us/chapters/c1.snbt:q.x.desc:0" in ids

        ap = apply_path(
            tmp_path,
            {"Sword": "Espada", "First line here": "Primera linea aqui"},
            fmt=FormatChoice.ftbq,
            use_cache=False,
        )
        assert ap.strings_applied == 2

        from modtranslator.core.ftbq_lang import parse_snbt_lang

        out = (
            tmp_path / "config" / "ftbquests" / "quests"
            / "lang" / "es_es" / "chapters" / "c1.snbt"
        )
        assert out.exists()
        data = parse_snbt_lang(out.read_text(encoding="utf-8"))
        assert data["q.x.title"] == "Espada"
        assert data["q.x.desc"] == ["Primera linea aqui"]

    def test_ftbq_amp_code_dropped_rejected(self, tmp_path: Path):
        _make_ftbq_single(tmp_path, '{\n    q.a: "&7Colored text"\n}\n')
        ap = apply_path(
            tmp_path, {"&7Colored text": "Texto sin codigo"},
            fmt=FormatChoice.ftbq, use_cache=False,
        )
        assert ap.strings_applied == 0
        assert ap.spec_rejected == ["&7Colored text"]

    def test_ftbq_amp_code_added_allowed(self, tmp_path: Path):
        # Translation introduces a literal '&' — conservative check must allow it.
        _make_ftbq_single(tmp_path, '{\n    q.a: "Tom and Jerry"\n}\n')
        ap = apply_path(
            tmp_path, {"Tom and Jerry": "Tom & Jerry"},
            fmt=FormatChoice.ftbq, use_cache=False,
        )
        assert ap.strings_applied == 1
        assert ap.spec_rejected == []

    def test_ftbq_batch(self, tmp_path: Path):
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.core.ftbq_lang import parse_snbt_lang
        from modtranslator.pipeline import batch_translate_ftbq

        _make_ftbq_single(tmp_path, self.SINGLE)
        quests_dir = tmp_path / "config" / "ftbquests" / "quests"
        result = batch_translate_ftbq(
            quests_dir, lang="ES", backend=DummyBackend(),
            backend_label="dummy", no_cache=True,
        )
        assert result.success_count == 1
        out = quests_dir / "lang" / "es_es.snbt"
        data = parse_snbt_lang(out.read_text(encoding="utf-8"))
        # &7 codes preserved; dummy prefixes [ES]
        assert "&7" in data["quest.A.title"]
        assert isinstance(data["quest.A.desc"], list)


class TestExportResourcePack:
    def test_resource_pack_merges_mods(self, tmp_path: Path):
        j1 = _make_jar(tmp_path, "m1.jar", {
            "assets/m1/lang/en_us.json": json.dumps({"a": "Hello"}),
        })
        j2 = _make_jar(tmp_path, "m2.jar", {
            "assets/m2/lang/en_us.json": json.dumps({"b": "World"}),
        })
        rp = tmp_path / "pack.zip"

        ap = apply_path(
            tmp_path,
            {"Hello": "Hola", "World": "Mundo"},
            use_cache=False,
            resource_pack=rp,
            pack_format=42,
        )
        assert ap.strings_applied == 2
        assert rp.exists()

        with zipfile.ZipFile(rp) as zf:
            names = set(zf.namelist())
            assert "pack.mcmeta" in names
            assert "assets/m1/lang/es_es.json" in names
            assert "assets/m2/lang/es_es.json" in names
            meta = json.loads(zf.read("pack.mcmeta"))
            assert meta["pack"]["pack_format"] == 42
            assert meta["pack"]["description"] == "modtranslator ES overlay"
            assert json.loads(zf.read("assets/m1/lang/es_es.json"))["a"] == "Hola"

        # JARs must be left untouched (no es_es added)
        for jar in (j1, j2):
            with zipfile.ZipFile(jar) as zf:
                assert not any("es_es" in n for n in zf.namelist())

    def test_resource_pack_includes_signed(self, tmp_path: Path):
        _make_jar(tmp_path, "signed.jar", {
            "META-INF/CERT.SF": "sig",
            "assets/sm/lang/en_us.json": json.dumps({"c": "Signed Text"}),
        })
        rp = tmp_path / "pack.zip"

        ap = apply_path(
            tmp_path, {"Signed Text": "Texto firmado"},
            use_cache=False, resource_pack=rp,
        )
        assert ap.strings_applied == 1
        with zipfile.ZipFile(rp) as zf:
            assert "assets/sm/lang/es_es.json" in zf.namelist()

    def test_resource_pack_batch_multiple(self, tmp_path: Path):
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import batch_translate_mc

        j1 = _make_jar(tmp_path, "m1.jar", {
            "assets/m1/lang/en_us.json": json.dumps({"a": "Hello"}),
        })
        signed = _make_jar(tmp_path, "signed.jar", {
            "META-INF/CERT.SF": "sig",
            "assets/sm/lang/en_us.json": json.dumps({"c": "Signed"}),
        })
        rp = tmp_path / "pack.zip"

        result = batch_translate_mc(
            [j1, signed], lang="ES", backend=DummyBackend(),
            backend_label="dummy", no_cache=True, resource_pack=rp, pack_format=15,
        )
        assert result.success_count == 2
        with zipfile.ZipFile(rp) as zf:
            names = set(zf.namelist())
            assert "assets/m1/lang/es_es.json" in names
            assert "assets/sm/lang/es_es.json" in names
            assert json.loads(zf.read("pack.mcmeta"))["pack"]["pack_format"] == 15
        # Signed jar untouched
        with zipfile.ZipFile(signed) as zf:
            assert not any("es_es" in n for n in zf.namelist())
