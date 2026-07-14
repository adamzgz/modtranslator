"""Tests for FO4 MCM config.json translation and BSA/BA2 loose-file output."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from modtranslator.backends.dummy import DummyBackend
from modtranslator.core.pex_parser import PEX_MAGIC, load_pex
from modtranslator.exchange import FormatChoice, apply_path, export_path
from modtranslator.pipeline import batch_translate_mcm, batch_translate_pex
from tests.core.test_bsa_archive import build_ba2, build_bsa


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch):
    """Point the default cache DB at a per-test temp file."""
    monkeypatch.setattr(
        "modtranslator.translation.cache.DEFAULT_CACHE_DB",
        tmp_path / "cache.db",
    )


def _make_pex_bytes(strings: list[str], post_data: bytes = b"") -> bytes:
    parts: list[bytes] = [
        struct.pack(">I", PEX_MAGIC),
        struct.pack(">BB", 3, 2),
        struct.pack(">H", 1),
        struct.pack(">Q", 0),
    ]
    for _ in range(3):
        parts.append(struct.pack(">H", 0))
    parts.append(struct.pack(">H", len(strings)))
    for s in strings:
        encoded = s.encode("utf-8")
        parts.append(struct.pack(">H", len(encoded)))
        parts.append(encoded)
    parts.append(post_data)
    return b"".join(parts)


def _translatable_pex(text: str) -> bytes:
    """A .pex where index 1 (`text`) is a pure 0x02 string literal."""
    post = b"\x01" + struct.pack(">H", 0) + b"\x02" + struct.pack(">H", 1)
    return _make_pex_bytes(["identifier", text], post)


def _config_dict() -> dict:
    return {
        "modName": "MyMod",
        "displayName": "My Mod",
        "content": [
            {"text": "Welcome!", "type": "text"},
            {"text": "$sectionKey", "type": "section"},
            {"text": "Enable", "type": "switcher", "help": "Turn it on"},
        ],
        "pages": [
            {
                "pageDisplayName": "Page One",
                "content": [{"text": "Nested text", "type": "text"}],
            }
        ],
    }


def _write_config(base: Path) -> Path:
    cfg = base / "MCM" / "Config" / "MyMod" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps(_config_dict(), indent="\t"), encoding="utf-8")
    return cfg


# ── MCM config.json ──────────────────────────────────────────────────────


class TestMcmConfig:
    def test_batch_translates_in_place(self, tmp_path: Path):
        cfg = _write_config(tmp_path)

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            no_cache=True,
        )
        assert result.success_count == 1
        assert result.error_count == 0

        data = json.loads(cfg.read_text(encoding="utf-8"))
        # Visible text translated
        assert data["displayName"] == "[ES] My Mod"
        assert data["content"][0]["text"] == "[ES] Welcome!"
        assert data["content"][2]["text"] == "[ES] Enable"
        assert data["content"][2]["help"] == "[ES] Turn it on"
        assert data["pages"][0]["pageDisplayName"] == "[ES] Page One"
        assert data["pages"][0]["content"][0]["text"] == "[ES] Nested text"
        # Identifiers / translation keys / structural keys untouched
        assert data["modName"] == "MyMod"
        assert data["content"][1]["text"] == "$sectionKey"
        assert data["content"][0]["type"] == "text"

    def test_export_apply_roundtrip(self, tmp_path: Path):
        cfg = _write_config(tmp_path)

        exp = export_path(
            tmp_path, fmt=FormatChoice.mcm, use_cache=False, skip_translated=False
        )
        cfg_files = [f for f in exp.doc["files"] if f["path"] == str(cfg)]
        assert len(cfg_files) == 1
        entry = cfg_files[0]
        assert entry["format"] == "mcm"
        texts = {e["text"] for e in entry["entries"]}
        assert texts == {
            "My Mod", "Welcome!", "Enable", "Turn it on",
            "Page One", "Nested text",
        }
        ids = {e["id"] for e in entry["entries"]}
        assert "MCM/Config/MyMod/config.json:content[0].text" in ids
        assert "MCM/Config/MyMod/config.json:pages[0].content[0].text" in ids

        trans = {t: f"[X] {t}" for t in texts}
        ap = apply_path(
            tmp_path, trans, fmt=FormatChoice.mcm, use_cache=False,
        )
        assert ap.strings_applied == 6
        assert ap.files_written == 1

        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["displayName"] == "[X] My Mod"
        assert data["content"][2]["help"] == "[X] Turn it on"
        assert data["modName"] == "MyMod"
        assert data["content"][1]["text"] == "$sectionKey"


# ── BSA/BA2 → loose files ────────────────────────────────────────────────


def _decode_mcm_txt(path: Path) -> str:
    raw = path.read_bytes()
    assert raw[:2] == b"\xff\xfe"
    text = raw.decode("utf-16-le")
    return text[1:] if text and text[0] == "﻿" else text


class TestPexArchives:
    def test_batch_pex_extracts_to_loose(self, tmp_path: Path):
        pex_bytes = _translatable_pex("Restore health to the player")
        bsa = tmp_path / "assets.bsa"
        bsa.write_bytes(build_bsa(104, [("scripts", "quest.pex", pex_bytes, False)]))

        result = batch_translate_pex(
            [],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            no_cache=True,
            directory=tmp_path,
            archives=[bsa],
        )
        assert result.success_count == 1

        loose = tmp_path / "Scripts" / "quest.pex"
        assert loose.exists()
        pex = load_pex(loose)
        # The archived literal was translated (dummy backend prefixes "[ES] ").
        assert any(s.startswith("[ES] ") for s in pex.string_table)
        assert "identifier" in pex.string_table  # 0x01 identifier untouched

    def test_loose_wins_precedence(self, tmp_path: Path):
        pex_bytes = _translatable_pex("Restore health to the player")
        bsa = tmp_path / "assets.bsa"
        bsa.write_bytes(build_bsa(104, [("scripts", "quest.pex", pex_bytes, False)]))

        # A pre-existing loose file with the same relative path wins.
        loose = tmp_path / "Scripts" / "quest.pex"
        loose.parent.mkdir(parents=True)
        loose.write_bytes(b"PLACEHOLDER")

        result = batch_translate_pex(
            [],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            no_cache=True,
            directory=tmp_path,
            archives=[bsa],
        )
        assert result.success_count == 0
        assert loose.read_bytes() == b"PLACEHOLDER"

    def test_export_apply_with_double_colon_path(self, tmp_path: Path):
        text = "Restore health to the player"
        pex_bytes = _translatable_pex(text)
        bsa = tmp_path / "assets.bsa"
        bsa.write_bytes(build_bsa(104, [("scripts", "quest.pex", pex_bytes, False)]))

        exp = export_path(
            tmp_path, fmt=FormatChoice.pex, use_cache=False, skip_translated=False
        )
        pex_files = [f for f in exp.doc["files"] if f["format"] == "pex"]
        assert len(pex_files) == 1
        assert pex_files[0]["path"] == f"{bsa}::scripts/quest.pex"

        ap = apply_path(
            tmp_path,
            {text: "Restaura la salud del jugador"},
            fmt=FormatChoice.pex,
            use_cache=False,
        )
        assert ap.strings_applied == 1
        assert ap.files_written == 1

        loose = tmp_path / "Scripts" / "quest.pex"
        assert loose.exists()
        pex = load_pex(loose)
        assert "Restaura la salud del jugador" in pex.string_table


class TestMcmArchives:
    def test_batch_mcm_extracts_translations_txt(self, tmp_path: Path):
        txt = b"$sHello\tHello World\n$sBye\tGoodbye Friend\n"
        ba2 = tmp_path / "mod.ba2"
        ba2.write_bytes(
            build_ba2([(r"Interface\Translations\Mod_english.txt", txt, False)])
        )

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            no_cache=True,
            archives=[ba2],
        )
        assert result.success_count == 1

        out = tmp_path / "Interface" / "translations" / "Mod_spanish.txt"
        assert out.exists()
        content = _decode_mcm_txt(out)
        assert "[ES] Hello World" in content
        assert "[ES] Goodbye Friend" in content
        assert "$sHello" in content

    def test_loose_wins_precedence(self, tmp_path: Path):
        txt = b"$sHello\tHello World\n"
        ba2 = tmp_path / "mod.ba2"
        ba2.write_bytes(
            build_ba2([(r"Interface\Translations\Mod_english.txt", txt, False)])
        )
        # Pre-existing loose target wins.
        tdir = tmp_path / "Interface" / "translations"
        tdir.mkdir(parents=True)
        existing = tdir / "Mod_spanish.txt"
        existing.write_bytes(b"\xff\xfeplaceholder".ljust(4, b"\x00"))
        before = existing.read_bytes()

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            no_cache=True,
            archives=[ba2],
        )
        assert result.success_count == 0
        assert existing.read_bytes() == before

    def test_export_apply_with_double_colon_path(self, tmp_path: Path):
        txt = b"$sHello\tHello World\n"
        ba2 = tmp_path / "mod.ba2"
        ba2.write_bytes(
            build_ba2([(r"Interface\Translations\Mod_english.txt", txt, False)])
        )

        exp = export_path(
            tmp_path, fmt=FormatChoice.mcm, use_cache=False, skip_translated=False
        )
        mcm_files = [f for f in exp.doc["files"] if f["format"] == "mcm"]
        assert len(mcm_files) == 1
        assert mcm_files[0]["path"] == f"{ba2}::interface/translations/mod_english.txt"
        assert mcm_files[0]["entries"][0]["text"] == "Hello World"

        ap = apply_path(
            tmp_path,
            {"Hello World": "Hola Mundo"},
            fmt=FormatChoice.mcm,
            use_cache=False,
        )
        assert ap.strings_applied == 1

        out = tmp_path / "Interface" / "translations" / "Mod_spanish.txt"
        assert out.exists()
        assert "Hola Mundo" in _decode_mcm_txt(out)
