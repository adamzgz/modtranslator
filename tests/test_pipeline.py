"""Tests for pipeline.py: batch functions, cancellation, MCM, PEX, utilities."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from threading import Event

import pytest

from modtranslator.backends.dummy import DummyBackend
from modtranslator.core.writer import write_plugin
from modtranslator.pipeline import (
    BatchResult,
    CancelledError,
    GameChoice,
    _build_dedup_map,
    _check_cancel,
    _FileContext,
    _translate_chunks,
    batch_translate_esp,
    batch_translate_mcm,
    create_backend,
    scan_directory,
    scan_file,
)
from tests.conftest import make_plugin, make_subrecord

# ── Helpers ──


def _write_esp(path: Path, records=None) -> None:
    """Write a minimal binary ESP to disk."""
    plugin = make_plugin(records or [
        ("WEAP", 0x1000, [
            make_subrecord("EDID", "TestWeapon"),
            make_subrecord("FULL", "Iron Sword"),
        ]),
    ])
    with open(path, "wb") as f:
        write_plugin(plugin, f)


def _write_mcm(path: Path, entries: list[tuple[str, str]]) -> None:
    """Write a UTF-16-LE MCM translation file."""
    lines = [f"${key}\t{value}" for key, value in entries]
    content = "\r\n".join(lines) + "\r\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe" + content.encode("utf-16-le"))


def _write_pex(path: Path, strings: list[tuple[int, str]]) -> None:
    """Write a minimal Skyrim PEX file (big-endian)."""
    buf = io.BytesIO()
    # Magic
    buf.write(struct.pack(">I", 0xFA57C0DE))
    # Major version, minor version, game_id, compilation_time
    buf.write(struct.pack(">B", 3))
    buf.write(struct.pack(">B", 1))
    buf.write(struct.pack(">H", 1))  # game_id=1 (Skyrim)
    buf.write(struct.pack(">Q", 0))  # compilation_time
    # Source filename
    buf.write(struct.pack(">H", 4))
    buf.write(b"test")
    # Username
    buf.write(struct.pack(">H", 4))
    buf.write(b"user")
    # Machine name
    buf.write(struct.pack(">H", 4))
    buf.write(b"host")
    # String table
    buf.write(struct.pack(">H", len(strings)))
    for _type_tag, text in strings:
        encoded = text.encode("utf-8")
        buf.write(struct.pack(">H", len(encoded)))
        buf.write(encoded)
    # String type tags
    # Need to write object count = 0 to end parsing
    # Actually the PEX parser reads the full structure, let's write a minimal valid one
    # For simplicity, write 0 objects (the parser stops after string table for types)
    # Actually the parser reads debug info, then user flags, then objects...
    # Let's use the actual pex_parser to save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


# ── Unit tests ──


class TestCheckCancel:
    def test_no_event_is_noop(self):
        _check_cancel(None)

    def test_unset_event_is_noop(self):
        ev = Event()
        _check_cancel(ev)

    def test_set_event_raises(self):
        ev = Event()
        ev.set()
        with pytest.raises(CancelledError):
            _check_cancel(ev)


class TestBuildDedupMap:
    def test_dedup_across_contexts(self):
        ctx1 = _FileContext(file_path=Path("a.esp"), output_path=None)
        ctx1.status = "prepared"
        ctx1.protected_texts = ["hello", "world"]

        ctx2 = _FileContext(file_path=Path("b.esp"), output_path=None)
        ctx2.status = "prepared"
        ctx2.protected_texts = ["world", "foo"]

        unique, mapping = _build_dedup_map([ctx1, ctx2])
        assert unique == ["hello", "world", "foo"]
        assert ctx1.dedup_indices == [0, 1]
        assert ctx2.dedup_indices == [1, 2]

    def test_skipped_contexts_ignored(self):
        ctx = _FileContext(file_path=Path("a.esp"), output_path=None)
        ctx.status = "skipped"
        ctx.protected_texts = ["hello"]

        unique, _ = _build_dedup_map([ctx])
        assert unique == []

    def test_empty_protected_texts_ignored(self):
        ctx = _FileContext(file_path=Path("a.esp"), output_path=None)
        ctx.status = "prepared"
        ctx.protected_texts = []

        unique, _ = _build_dedup_map([ctx])
        assert unique == []


class TestTranslateChunks:
    def test_basic_translation(self):
        backend = DummyBackend()
        result, errors = _translate_chunks(["hello", "world"], backend, "ES")
        assert result == ["[ES] hello", "[ES] world"]
        assert errors == []

    def test_progress_callback(self):
        backend = DummyBackend()
        calls = []
        result, errors = _translate_chunks(
            ["a", "b", "c"], backend, "ES", chunk_size=2,
            on_progress=lambda phase, cur, total, msg: calls.append((phase, cur, total)),
        )
        assert result == ["[ES] a", "[ES] b", "[ES] c"]
        assert errors == []
        assert len(calls) == 2  # 2 chunks

    def test_cancel_between_chunks(self):
        backend = DummyBackend()
        ev = Event()

        def on_progress(phase, cur, total, msg):
            if cur >= 2:
                ev.set()  # cancel after first chunk

        with pytest.raises(CancelledError):
            _translate_chunks(
                ["a", "b", "c", "d"], backend, "ES", chunk_size=2,
                cancel_event=ev, on_progress=on_progress,
            )


class TestCreateBackend:
    def test_dummy(self):
        backend, label = create_backend("dummy")
        assert label == "dummy"

    def test_deepl_no_key_raises(self):
        with pytest.raises(ValueError, match="DeepL API key required"):
            create_backend("deepl")

    def test_deepl_default_no_key_raises(self):
        with pytest.raises(ValueError, match="DeepL API key required"):
            create_backend("unknown_backend")


class TestScanDirectory:
    def test_scan_esp_files(self, tmp_path):
        (tmp_path / "mod1.esp").write_bytes(b"")
        (tmp_path / "mod2.esm").write_bytes(b"")
        (tmp_path / "readme.txt").write_bytes(b"")

        result = scan_directory(tmp_path)
        assert len(result.esp_files) == 2
        assert result.pex_files == []
        assert result.has_mcm is False

    def test_scan_pex_in_scripts_dir(self, tmp_path):
        scripts = tmp_path / "Scripts"
        scripts.mkdir()
        (scripts / "test.pex").write_bytes(b"")
        (tmp_path / "direct.pex").write_bytes(b"")

        result = scan_directory(tmp_path)
        assert len(result.pex_files) == 2

    def test_scan_mcm(self, tmp_path):
        trans = tmp_path / "Interface" / "translations"
        trans.mkdir(parents=True)
        (trans / "mod_english.txt").write_bytes(b"")

        result = scan_directory(tmp_path)
        assert result.has_mcm is True
        assert result.mcm_directory == tmp_path

    def test_scan_mcm_recorder(self, tmp_path):
        recorder = tmp_path / "McmRecorder"
        recorder.mkdir()

        result = scan_directory(tmp_path)
        assert result.has_mcm is True

    def test_scan_esl_files(self, tmp_path):
        (tmp_path / "patch.esl").write_bytes(b"")
        result = scan_directory(tmp_path)
        assert len(result.esp_files) == 1


class TestScanFile:
    def test_scan_file_returns_strings(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)
        strings = scan_file(esp)
        assert len(strings) > 0
        assert any(s.original_text == "Iron Sword" for s in strings)

    def test_scan_file_sets_source_file(self, tmp_path):
        esp = tmp_path / "mymod.esp"
        _write_esp(esp)
        strings = scan_file(esp)
        for s in strings:
            assert s.source_file == "mymod"


class TestBatchTranslateEsp:
    def test_basic_batch(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_esp(
            [esp],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            game=GameChoice.fo3,
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
        )
        assert isinstance(result, BatchResult)
        assert result.success_count == 1
        assert result.error_count == 0

    def test_cancellation(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)

        ev = Event()
        ev.set()  # pre-cancel

        result = batch_translate_esp(
            [esp],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            game=GameChoice.fo3,
            no_cache=True,
            cancel_event=ev,
        )
        assert any("Cancelled" in e[1] for e in result.errors)

    def test_auto_game_detection(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_esp(
            [esp],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            game=GameChoice.auto,
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
        )
        assert result.success_count == 1

    def test_progress_callback(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        phases = []
        batch_translate_esp(
            [esp],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            game=GameChoice.fo3,
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
            on_progress=lambda phase, *_: phases.append(phase),
        )
        assert "prepare" in phases

    def test_skyrim_game_choice(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_esp(
            [esp],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            game=GameChoice.skyrim,
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
        )
        assert isinstance(result, BatchResult)

    def test_fo4_game_choice(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_esp(
            [esp],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            game=GameChoice.fo4,
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
        )
        assert isinstance(result, BatchResult)

    def test_no_output_dir_creates_suffixed_file(self, tmp_path):
        esp = tmp_path / "test.esp"
        _write_esp(esp)

        result = batch_translate_esp(
            [esp],
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            game=GameChoice.fo3,
            skip_translated=False,
            no_cache=True,
        )
        assert result.success_count == 1


class TestBatchTranslateMcm:
    def test_basic_mcm_translation(self, tmp_path):
        trans = tmp_path / "Interface" / "translations"
        trans.mkdir(parents=True)
        _write_mcm(
            trans / "mymod_english.txt",
            [("sOption1", "Enable Feature"), ("sOption2", "Disable Feature")],
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
        )
        assert result.success_count >= 1
        assert result.error_count == 0

    def test_mcm_no_translations_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            batch_translate_mcm(
                tmp_path,
                lang="ES",
                backend=DummyBackend(),
                backend_label="dummy",
                no_cache=True,
            )

    def test_mcm_json_translation(self, tmp_path):
        # Create McmRecorder dir with a JSON file
        recorder = tmp_path / "McmRecorder"
        recorder.mkdir()
        data = {"welcome": "Welcome to the mod!", "complete": "Setup is complete.", "version": 1}
        (recorder / "config.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
        )
        assert result.success_count >= 1

    def test_mcm_empty_dir_returns_zero(self, tmp_path):
        trans = tmp_path / "Interface" / "translations"
        trans.mkdir(parents=True)
        # No english files

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            no_cache=True,
        )
        assert result.success_count == 0
        assert result.error_count == 0

    def test_mcm_cancellation(self, tmp_path):
        trans = tmp_path / "Interface" / "translations"
        trans.mkdir(parents=True)
        _write_mcm(
            trans / "mod_english.txt",
            [("sTest", "Hello World")],
        )

        ev = Event()
        ev.set()

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            no_cache=True,
            cancel_event=ev,
        )
        assert any("Cancelled" in e[1] for e in result.errors)

    def test_mcm_html_tags_preserved(self, tmp_path):
        trans = tmp_path / "Interface" / "translations"
        trans.mkdir(parents=True)
        _write_mcm(
            trans / "mod_english.txt",
            [("sColored", "<font color='#ff0000'>Red Text</font>")],
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
        )
        assert result.success_count >= 1

    def test_mcm_skip_already_translated(self, tmp_path):
        """When target file exists with different content, skip it."""
        trans = tmp_path / "Interface" / "translations"
        trans.mkdir(parents=True)
        _write_mcm(
            trans / "mod_english.txt",
            [("sTest", "Hello")],
        )
        # Create a different spanish file (different hash → skip)
        _write_mcm(
            trans / "mod_spanish.txt",
            [("sTest", "Hola amigo")],
        )

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            no_cache=True,
        )
        # Should skip because target already exists with different content
        assert result.success_count == 0

    def test_mcm_fo4_short_codes(self, tmp_path):
        """MCM with FO4-style _en.txt naming."""
        trans = tmp_path / "Interface" / "translations"
        trans.mkdir(parents=True)
        _write_mcm(
            trans / "mymod_en.txt",
            [("sOption", "Enable")],
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = batch_translate_mcm(
            tmp_path,
            lang="ES",
            backend=DummyBackend(),
            backend_label="dummy",
            skip_translated=False,
            output_dir=output_dir,
            no_cache=True,
            game=GameChoice.fo4,
        )
        assert result.success_count >= 1
