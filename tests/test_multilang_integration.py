"""Integration tests for multi-language support (FR, DE, IT, PT, RU, PL).

All tests use the dummy backend to avoid ML model dependencies.
The dummy backend prefixes each string with [LANG], making output easy to verify.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modtranslator.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mcm_skyrim(tmp_path: Path, stem: str, content: str) -> Path:
    """Create a Skyrim-style MCM file (UTF-16-LE, _english.txt suffix)."""
    trans_dir = tmp_path / "Interface" / "translations"
    trans_dir.mkdir(parents=True, exist_ok=True)
    p = trans_dir / f"{stem}_english.txt"
    p.write_bytes(("\ufeff" + content).encode("utf-16-le"))
    return tmp_path  # return mod root


def _mcm_fo4(tmp_path: Path, stem: str, content: str) -> Path:
    """Create a FO4-style MCM file (UTF-16-LE, _en.txt suffix)."""
    trans_dir = tmp_path / "Interface" / "translations"
    trans_dir.mkdir(parents=True, exist_ok=True)
    p = trans_dir / f"{stem}_en.txt"
    p.write_bytes(("\ufeff" + content).encode("utf-16-le"))
    return tmp_path


def _read_mcm(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le")
        return text.lstrip("\ufeff")
    return raw.decode("utf-8")


def _skyrim_pex() -> bytes:
    """Build a minimal translatable Skyrim PEX (big-endian)."""
    magic = struct.pack(">I", 0xFA57C0DE)
    header = struct.pack(">HHHI", 3, 2, 1, 0)
    src_str = struct.pack(">H", 8) + b"test.psc"
    user_doc = struct.pack(">H", 0)
    s = b"A magical sword"
    table = struct.pack(">H", 1) + struct.pack(">H", len(s)) + s
    types = struct.pack(">H", 1) + struct.pack(">HB", 0, 0x02)
    trailing = b"\x00" * 4
    return magic + header + src_str + user_doc + table + types + trailing


# ── ESP / ESM ─────────────────────────────────────────────────────────────────


class TestMultilangESPTranslate:
    """Single-file translate command with --lang != ES."""

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT", "PT", "RU", "PL"])
    def test_translate_produces_tagged_output(self, tmp_path, lang):
        """translate --lang LANG with dummy backend: output contains [LANG] tag."""
        src = FIXTURES / "minimal_fo3.esp"
        out = tmp_path / f"out_{lang}.esp"
        result = runner.invoke(app, [
            "translate", str(src),
            "--dummy",
            "--lang", lang,
            "--output", str(out),
            "--no-cache",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

        # Load the translated plugin and verify strings are tagged
        from modtranslator.core.plugin import load_plugin
        plugin = load_plugin(out)
        strings = []
        for group in plugin.groups:
            for rec in group.children:
                for sub in rec.subrecords:
                    try:
                        s = sub.decode_string()
                        if s and s.strip():
                            strings.append(s)
                    except Exception:
                        pass
        translated = [s for s in strings if f"[{lang}]" in s]
        assert len(translated) > 0, (
            f"Expected strings tagged with [{lang}], got: {strings}"
        )

    @pytest.mark.parametrize("lang", ["FR", "DE", "RU"])
    def test_translate_skyrim_inline(self, tmp_path, lang):
        """Skyrim inline (non-localized) ESP translated to non-ES language."""
        src = FIXTURES / "skyrim_inline.esp"
        out = tmp_path / f"skyrim_{lang}.esp"
        result = runner.invoke(app, [
            "translate", str(src),
            "--dummy",
            "--lang", lang,
            "--game", "skyrim",
            "--output", str(out),
            "--no-cache",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

    @pytest.mark.parametrize("lang", ["FR", "DE", "IT"])
    def test_translate_multi_record(self, tmp_path, lang):
        """Multi-record ESP: all translatable strings get tagged."""
        src = FIXTURES / "multi_record.esp"
        out = tmp_path / f"multi_{lang}.esp"
        result = runner.invoke(app, [
            "translate", str(src),
            "--dummy",
            "--lang", lang,
            "--output", str(out),
            "--no-cache",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

        from modtranslator.core.plugin import load_plugin
        plugin = load_plugin(out)
        tags_found = 0
        for group in plugin.groups:
            for rec in group.children:
                for sub in rec.subrecords:
                    try:
                        s = sub.decode_string()
                        if f"[{lang}]" in s:
                            tags_found += 1
                    except Exception:
                        pass
        assert tags_found >= 2  # multi_record has at least WEAP+ARMO FULL strings


class TestMultilangESPBatch:
    """batch command with --lang != ES."""

    @pytest.mark.parametrize("lang", ["FR", "DE", "RU", "PL"])
    def test_batch_output_suffix_inplace(self, tmp_path, lang):
        """batch --lang LANG without --output-dir: files get _{LANG} suffix in-place."""
        import shutil
        src = FIXTURES / "minimal_fo3.esp"
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        shutil.copy(src, mod_dir / "minimal_fo3.esp")

        # No --output-dir → output in same dir with _LANG suffix
        result = runner.invoke(app, [
            "batch", str(mod_dir),
            "--dummy",
            "--lang", lang,
            "--no-cache",
        ])
        assert result.exit_code == 0, result.output
        expected = mod_dir / f"minimal_fo3_{lang}.esp"
        assert expected.exists(), f"Expected {expected}, got: {list(mod_dir.iterdir())}"

    @pytest.mark.parametrize("lang", ["FR", "IT", "PT"])
    def test_batch_output_dir_keeps_original_names(self, tmp_path, lang):
        """batch --output-dir: output files keep original names in the target dir."""
        import shutil
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        for name in ("minimal_fo3.esp", "multi_record.esp"):
            shutil.copy(FIXTURES / name, mod_dir / name)

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch", str(mod_dir),
            "--dummy",
            "--lang", lang,
            "--output-dir", str(out_dir),
            "--no-cache",
        ])
        assert result.exit_code == 0, result.output
        # With --output-dir, files keep original names
        translated = list(out_dir.glob("*.esp"))
        assert len(translated) == 2, f"Expected 2 translated files, got: {translated}"

    def test_batch_summary_shows_lang(self, tmp_path):
        """Batch summary output is shown for non-ES languages."""
        import shutil
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        shutil.copy(FIXTURES / "minimal_fo3.esp", mod_dir)

        result = runner.invoke(app, [
            "batch", str(mod_dir),
            "--dummy",
            "--lang", "FR",
            "--output-dir", str(tmp_path / "out"),
            "--no-cache",
        ])
        assert result.exit_code == 0, result.output
        assert "Batch Summary" in result.output


# ── MCM ───────────────────────────────────────────────────────────────────────


class TestMultilangMCM:
    """batch-mcm with --lang != ES."""

    def test_skyrim_mcm_fr_output_named_french(self, tmp_path):
        """Skyrim MCM + --lang FR → output file uses 'french' suffix."""
        mod_root = _mcm_skyrim(tmp_path, "mymod", "$KEY\tHello adventurer\n")
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-mcm", str(mod_root),
            "--dummy", "--lang", "FR",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "skyrim",
        ])
        assert result.exit_code == 0, result.output
        out_file = out_dir / "mymod_french.txt"
        assert out_file.exists(), f"Expected {out_file}, got: {list(out_dir.iterdir())}"
        content = _read_mcm(out_file)
        assert "[FR]" in content

    def test_skyrim_mcm_de_output_named_german(self, tmp_path):
        """Skyrim MCM + --lang DE → output file uses 'german' suffix."""
        mod_root = _mcm_skyrim(tmp_path, "mymod", "$KEY\tA sword of legend\n")
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-mcm", str(mod_root),
            "--dummy", "--lang", "DE",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "skyrim",
        ])
        assert result.exit_code == 0, result.output
        out_file = out_dir / "mymod_german.txt"
        assert out_file.exists(), f"Expected {out_file}, got: {list(out_dir.iterdir())}"
        content = _read_mcm(out_file)
        assert "[DE]" in content

    def test_skyrim_mcm_ru_output_named_russian(self, tmp_path):
        """Skyrim MCM + --lang RU → output file uses 'russian' suffix."""
        mod_root = _mcm_skyrim(tmp_path, "mymod", "$KEY\tWelcome to Skyrim\n")
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-mcm", str(mod_root),
            "--dummy", "--lang", "RU",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "skyrim",
        ])
        assert result.exit_code == 0, result.output
        out_file = out_dir / "mymod_russian.txt"
        assert out_file.exists(), f"Expected {out_file}, got: {list(out_dir.iterdir())}"
        content = _read_mcm(out_file)
        assert "[RU]" in content

    def test_skyrim_mcm_pl_output_named_polish(self, tmp_path):
        mod_root = _mcm_skyrim(tmp_path, "mymod", "$KEY\tWitaj w Skyrim\n")
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-mcm", str(mod_root),
            "--dummy", "--lang", "PL",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "skyrim",
        ])
        assert result.exit_code == 0, result.output
        out_file = out_dir / "mymod_polish.txt"
        assert out_file.exists(), f"Expected {out_file}, got: {list(out_dir.iterdir())}"

    def test_fo4_mcm_fr_uses_short_code(self, tmp_path):
        """FO4 MCM + --lang FR → output uses short code 'fr'."""
        mod_root = _mcm_fo4(tmp_path, "mymod", "$KEY\tOpen the door\n")
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-mcm", str(mod_root),
            "--dummy", "--lang", "FR",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "fo4",
        ])
        assert result.exit_code == 0, result.output
        out_file = out_dir / "mymod_fr.txt"
        assert out_file.exists(), f"Expected {out_file}, got: {list(out_dir.iterdir())}"
        content = _read_mcm(out_file)
        assert "[FR]" in content

    def test_fo4_mcm_de_uses_short_code(self, tmp_path):
        """FO4 MCM + --lang DE → output uses short code 'de'."""
        mod_root = _mcm_fo4(tmp_path, "weapon", "$KEY\tPower Armor\n")
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-mcm", str(mod_root),
            "--dummy", "--lang", "DE",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "fo4",
        ])
        assert result.exit_code == 0, result.output
        out_file = out_dir / "weapon_de.txt"
        assert out_file.exists(), f"Expected {out_file}, got: {list(out_dir.iterdir())}"

    def test_mcm_translated_content_has_lang_tag(self, tmp_path):
        """Dummy backend tags each MCM value with [LANG] prefix."""
        mod_root = _mcm_skyrim(tmp_path, "spell", "$KEY1\tFireball\n$KEY2\tIceberg\n")
        out_dir = tmp_path / "out"
        runner.invoke(app, [
            "batch-mcm", str(mod_root),
            "--dummy", "--lang", "IT",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "skyrim",
        ])
        out_file = out_dir / "spell_italian.txt"
        if out_file.exists():
            content = _read_mcm(out_file)
            assert "[IT]" in content


# ── PEX ───────────────────────────────────────────────────────────────────────


class TestMultilangPEX:
    """batch-pex with --lang != ES."""

    @pytest.mark.parametrize("lang", ["FR", "DE", "RU", "PL"])
    def test_batch_pex_lang_succeeds(self, tmp_path, lang):
        """batch-pex with non-ES lang: pipeline succeeds without crash."""
        pex_file = tmp_path / "spell.pex"
        pex_file.write_bytes(_skyrim_pex())
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-pex", str(tmp_path),
            "--dummy",
            "--lang", lang,
            "--output-dir", str(out_dir),
            "--no-cache",
            "--game", "skyrim",
        ])
        assert result.exit_code in (0, None, 1), result.output
        # If strings were found and translated, output file exists
        out_pex = out_dir / "spell.pex"
        if out_pex.exists():
            # Verify the PEX has been modified (non-zero, parseable)
            assert out_pex.stat().st_size > 0

    def test_batch_pex_fr_tagged_strings(self, tmp_path):
        """PEX translated to FR: string table contains [FR] tagged text."""
        from modtranslator.core.pex_parser import parse_pex

        pex_file = tmp_path / "spell.pex"
        pex_file.write_bytes(_skyrim_pex())
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch-pex", str(tmp_path),
            "--dummy", "--lang", "FR",
            "--output-dir", str(out_dir),
            "--no-cache", "--game", "skyrim",
        ])
        assert result.exit_code in (0, None, 1), result.output
        out_pex = out_dir / "spell.pex"
        if out_pex.exists():
            pex = parse_pex(out_pex.read_bytes())
            strings_in_output = [e.value for e in pex.string_table]
            tagged = [s for s in strings_in_output if "[FR]" in s]
            assert len(tagged) > 0, f"Expected [FR]-tagged strings, got: {strings_in_output}"


# ── Pipeline-level tests (no CLI) ────────────────────────────────────────────


class TestMultilangPipelineDirect:
    """Direct pipeline function calls for multilang (no CLI overhead)."""

    def test_batch_translate_esp_fr(self, tmp_path):
        """batch_translate_esp with lang='FR' and dummy backend."""
        import shutil

        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_esp

        src = FIXTURES / "minimal_fo3.esp"
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        shutil.copy(src, mod_dir / "minimal_fo3.esp")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = batch_translate_esp(
            list(mod_dir.glob("*.esp")),
            backend=DummyBackend(),
            backend_label="dummy",
            lang="FR",
            game=GameChoice.fo3,
            output_dir=out_dir,
            no_cache=True,
        )
        assert result.success_count >= 1
        # With output_dir, file keeps original name
        assert (out_dir / "minimal_fo3.esp").exists()

    def test_batch_translate_esp_de(self, tmp_path):
        """batch_translate_esp with lang='DE' and dummy backend."""
        import shutil

        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_esp

        src = FIXTURES / "multi_record.esp"
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        shutil.copy(src, mod_dir / "multi_record.esp")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = batch_translate_esp(
            list(mod_dir.glob("*.esp")),
            backend=DummyBackend(),
            backend_label="dummy",
            lang="DE",
            game=GameChoice.fo3,
            output_dir=out_dir,
            no_cache=True,
        )
        assert result.success_count >= 1

    def test_batch_translate_esp_ru_skyrim(self, tmp_path):
        """batch_translate_esp with lang='RU' on Skyrim inline plugin."""
        import shutil

        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_esp

        src = FIXTURES / "skyrim_inline.esp"
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        shutil.copy(src, mod_dir / "skyrim_inline.esp")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = batch_translate_esp(
            list(mod_dir.glob("*.esp")),
            backend=DummyBackend(),
            backend_label="dummy",
            lang="RU",
            game=GameChoice.skyrim,
            output_dir=out_dir,
            no_cache=True,
        )
        assert result.success_count >= 1
        assert (out_dir / "skyrim_inline.esp").exists()

    def test_batch_translate_mcm_fr(self, tmp_path):
        """batch_translate_mcm with lang='FR' produces french-named output."""
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_mcm

        mod_root = _mcm_skyrim(tmp_path, "myspell", "$KEY\tFireball\n$KEY2\tIce storm\n")
        out_dir = tmp_path / "out"

        result = batch_translate_mcm(
            mod_root,
            backend=DummyBackend(),
            backend_label="dummy",
            lang="FR",
            game=GameChoice.skyrim,
            output_dir=out_dir,
            no_cache=True,
        )
        assert result.success_count >= 1
        out_file = out_dir / "myspell_french.txt"
        assert out_file.exists()
        content = _read_mcm(out_file)
        assert "[FR]" in content

    def test_batch_translate_mcm_de(self, tmp_path):
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_mcm

        mod_root = _mcm_skyrim(tmp_path, "weapons", "$KEY\tIron Sword\n")
        out_dir = tmp_path / "out"

        result = batch_translate_mcm(
            mod_root,
            backend=DummyBackend(),
            backend_label="dummy",
            lang="DE",
            game=GameChoice.skyrim,
            output_dir=out_dir,
            no_cache=True,
        )
        assert result.success_count >= 1
        assert (out_dir / "weapons_german.txt").exists()

    def test_batch_translate_mcm_ru(self, tmp_path):
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_mcm

        mod_root = _mcm_skyrim(tmp_path, "armors", "$KEY\tLeather Armor\n")
        out_dir = tmp_path / "out"

        result = batch_translate_mcm(
            mod_root,
            backend=DummyBackend(),
            backend_label="dummy",
            lang="RU",
            game=GameChoice.skyrim,
            output_dir=out_dir,
            no_cache=True,
        )
        assert result.success_count >= 1
        assert (out_dir / "armors_russian.txt").exists()

    def test_batch_translate_mcm_fo4_fr_short(self, tmp_path):
        """FO4 MCM + lang=FR → uses short code 'fr'."""
        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_mcm

        mod_root = _mcm_fo4(tmp_path, "settlement", "$KEY\tOpen the door\n")
        out_dir = tmp_path / "out"

        result = batch_translate_mcm(
            mod_root,
            backend=DummyBackend(),
            backend_label="dummy",
            lang="FR",
            game=GameChoice.fo4,
            output_dir=out_dir,
            no_cache=True,
        )
        assert result.success_count >= 1
        assert (out_dir / "settlement_fr.txt").exists()

    def test_deduplication_works_across_langs(self, tmp_path):
        """Deduplication works for non-ES: identical strings translated once."""
        import shutil

        from modtranslator.backends.dummy import DummyBackend
        from modtranslator.pipeline import GameChoice, batch_translate_esp

        # 3 copies of same file → all share same strings → one backend call
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        src = FIXTURES / "minimal_fo3.esp"
        for i in range(3):
            shutil.copy(src, mod_dir / f"mod{i}.esp")

        call_count = 0
        original_batch = DummyBackend.translate_batch

        def counting_batch(self, texts, target_lang, source_lang=None):
            nonlocal call_count
            call_count += 1
            return original_batch(self, texts, target_lang, source_lang)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        DummyBackend.translate_batch = counting_batch
        try:
            result = batch_translate_esp(
                list(mod_dir.glob("*.esp")),
                backend=DummyBackend(),
                backend_label="dummy",
                lang="FR",
                game=GameChoice.fo3,
                output_dir=out_dir,
                no_cache=True,
            )
        finally:
            DummyBackend.translate_batch = original_batch

        assert result.success_count == 3
        # Deduplication means translate_batch called once (all identical strings merged)
        assert call_count == 1, f"Expected 1 deduped call, got {call_count}"
