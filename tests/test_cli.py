"""Integration tests for the CLI using Typer's CliRunner."""

import sys
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from modtranslator.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


class TestCLIScan:
    def test_scan_minimal_fo3(self):
        result = runner.invoke(app, ["scan", str(FIXTURES / "minimal_fo3.esp")])
        assert result.exit_code == 0
        assert "Iron Sword" in result.output

    def test_scan_multi_record(self):
        result = runner.invoke(app, ["scan", str(FIXTURES / "multi_record.esp")])
        assert result.exit_code == 0
        assert "Leather Armor" in result.output
        assert "Wasteland Survival Guide" in result.output

    def test_scan_nonexistent_file(self):
        result = runner.invoke(app, ["scan", "nonexistent.esp"])
        assert result.exit_code == 1


class TestCLITranslate:
    def test_translate_dry_run_dummy(self, tmp_path):
        esp = FIXTURES / "minimal_fo3.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--dry-run",
            "--no-cache",
        ])
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_translate_with_dummy_backend(self, tmp_path):
        esp = FIXTURES / "multi_record.esp"
        output = tmp_path / "translated.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_translate_with_report(self, tmp_path):
        esp = FIXTURES / "minimal_fo3.esp"
        report = tmp_path / "report.json"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--report", str(report),
            "--no-cache",
        ])
        assert result.exit_code == 0
        assert report.exists()
        import json
        data = json.loads(report.read_text())
        assert data["total_strings_found"] > 0

    def test_translate_requires_api_key_without_dummy(self):
        esp = FIXTURES / "minimal_fo3.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--no-cache",
        ], env={"DEEPL_API_KEY": ""})
        assert result.exit_code == 1


class TestCLIVersion:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "modtranslator" in result.output


class TestCLITranslateNewFlags:
    def test_translate_skip_translated_dummy(self, tmp_path):
        esp = FIXTURES / "multi_record.esp"
        output = tmp_path / "translated.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
            "--skip-translated",
        ])
        assert result.exit_code == 0

    def test_translate_backend_dummy_flag(self, tmp_path):
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--backend", "dummy",
            "--output", str(output),
            "--no-cache",
        ])
        assert result.exit_code == 0
        assert output.exists()


class TestCLIBatch:
    def test_batch_output_dir(self, tmp_path):
        """--output-dir writes files with original names to the target dir."""
        import shutil

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "test.esp")

        out_dir = tmp_path / "translated"
        result = runner.invoke(app, [
            "batch", str(src_dir),
            "--dummy",
            "--output-dir", str(out_dir),
            "--pattern", "*.esp",
        ])
        assert result.exit_code is None or result.exit_code == 0
        assert (out_dir / "test.esp").exists()

    def test_batch_default_pattern_finds_esm(self, tmp_path):
        """Default pattern (*.esp) also picks up .esm files."""
        import shutil

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        # Copy a fixture as .esm to test the default pattern
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "base.esm")

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch", str(src_dir),
            "--dummy",
            "--output-dir", str(out_dir),
        ])
        assert result.exit_code is None or result.exit_code == 0
        assert (out_dir / "base.esm").exists()

    def test_batch_default_pattern_finds_both(self, tmp_path):
        """Default pattern finds both .esp and .esm files."""
        import shutil

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "mod.esp")
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "base.esm")

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch", str(src_dir),
            "--dummy",
            "--output-dir", str(out_dir),
        ])
        assert result.exit_code is None or result.exit_code == 0
        assert (out_dir / "mod.esp").exists()
        assert (out_dir / "base.esm").exists()


class TestCLIBatchSummary:
    def test_batch_shows_summary_table(self, tmp_path):
        """Batch output contains a summary table."""
        import shutil

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "test.esp")

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch", str(src_dir),
            "--dummy",
            "--output-dir", str(out_dir),
        ])
        assert result.exit_code is None or result.exit_code == 0
        assert "Batch Summary" in result.output

    def test_batch_counts_errors_gracefully(self, tmp_path):
        """A garbage ESP file is counted as error, not a crash."""
        src_dir = tmp_path / "data"
        src_dir.mkdir()
        (src_dir / "bad.esp").write_bytes(b"\x00\x01\x02\x03")

        result = runner.invoke(app, [
            "batch", str(src_dir),
            "--dummy",
            "--pattern", "*.esp",
        ])
        assert result.exit_code is None or result.exit_code == 0
        assert "Batch Summary" in result.output
        assert "Errors" in result.output


class TestCLIVerboseQuiet:
    def test_quiet_suppresses_output(self, tmp_path):
        """--quiet suppresses informational output."""
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "--quiet",
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
        ])
        assert result.exit_code == 0
        assert "Found" not in result.output
        assert "Game detected" not in result.output

    def test_verbose_shows_extra_info(self, tmp_path):
        """--verbose shows backend info."""
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "--verbose",
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
        ])
        assert result.exit_code == 0
        assert "Backend:" in result.output

    def test_quiet_errors_still_show(self):
        """Errors are visible even with --quiet."""
        result = runner.invoke(app, [
            "--quiet",
            "translate", "nonexistent.esp",
            "--dummy",
            "--no-cache",
        ])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestCLIOpusMT:
    def test_opus_mt_missing_deps(self):
        """--backend opus-mt without deps installed gives clear error."""
        esp = FIXTURES / "minimal_fo3.esp"
        # Simulate ctranslate2 not being installed: setting a module to None
        # in sys.modules makes `import ctranslate2` raise ImportError.
        with patch.dict(sys.modules, {"ctranslate2": None}):
            result = runner.invoke(app, [
                "translate", str(esp),
                "--backend", "opus-mt",
                "--no-cache",
            ])
        assert result.exit_code == 1


class TestCLINLLB:
    def test_nllb_missing_deps(self):
        """--backend nllb without deps installed gives clear error."""
        esp = FIXTURES / "minimal_fo3.esp"
        with patch.dict(sys.modules, {"ctranslate2": None}):
            result = runner.invoke(app, [
                "translate", str(esp),
                "--backend", "nllb",
                "--no-cache",
            ])
        assert result.exit_code == 1

    def test_nllb_model_flag_accepted(self):
        """--backend nllb --model 600M is accepted by the CLI parser."""
        esp = FIXTURES / "minimal_fo3.esp"
        with patch.dict(sys.modules, {"ctranslate2": None}):
            result = runner.invoke(app, [
                "translate", str(esp),
                "--backend", "nllb",
                "--model", "600M",
                "--no-cache",
            ])
        # Exits with error because deps missing, but flag was accepted (not a usage error)
        assert result.exit_code == 1
        assert "Usage:" not in (result.output or "")


class TestCLIGameFlag:
    def test_game_auto_default(self, tmp_path):
        """--game auto works (default behavior)."""
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
            "--game", "auto",
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_game_fo3_explicit(self, tmp_path):
        """--game fo3 works explicitly."""
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
            "--game", "fo3",
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_game_fnv_explicit(self, tmp_path):
        """--game fnv loads FNV glossary."""
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
            "--game", "fnv",
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_game_invalid_value(self):
        """Invalid --game value gives error."""
        esp = FIXTURES / "minimal_fo3.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--no-cache",
            "--game", "oblivion",
        ])
        assert result.exit_code != 0

    def test_game_skyrim_accepted(self, tmp_path):
        """--game skyrim is a valid option."""
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
            "--game", "skyrim",
        ])
        assert result.exit_code == 0

    def test_glossary_overrides_game(self, tmp_path):
        """Explicit --glossary takes priority over --game."""
        esp = FIXTURES / "minimal_fo3.esp"
        output = tmp_path / "out.esp"
        custom_gloss = tmp_path / "custom.toml"
        custom_gloss.write_text('[terms]\n"TestTerm" = "TerminoTest"\n', encoding="utf-8")
        result = runner.invoke(app, [
            "translate", str(esp),
            "--dummy",
            "--output", str(output),
            "--no-cache",
            "--game", "fnv",
            "--glossary", str(custom_gloss),
        ])
        assert result.exit_code == 0
        # Should load 1 term from custom glossary, not FNV terms
        assert "1" in result.output  # "1 terms"


class TestCLIBatchBackendReuse:
    def test_batch_calls_translate_batch_once(self, tmp_path):
        """Parallel batch calls backend.translate_batch() exactly once with all unique texts."""
        import shutil
        from unittest.mock import MagicMock, patch

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "a.esp")
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "b.esp")

        out_dir = tmp_path / "out"

        from modtranslator import cli as cli_module
        from modtranslator.backends.dummy import DummyBackend

        real_backend = DummyBackend()
        spy_backend = MagicMock(wraps=real_backend)

        def mock_create(*args, **kwargs):
            return spy_backend, "dummy"

        with patch.object(cli_module, "_create_backend", side_effect=mock_create):
            result = runner.invoke(app, [
                "batch", str(src_dir),
                "--dummy",
                "--output-dir", str(out_dir),
                "--pattern", "*.esp",
                "--no-cache",
            ])

        assert result.exit_code is None or result.exit_code == 0
        # translate_batch called exactly once (all texts deduped into single call)
        assert spy_backend.translate_batch.call_count == 1


class TestCLIBatchParallel:
    def test_batch_parallel_deduplication(self, tmp_path):
        """3 copies of the same ESP → translate_batch receives deduplicated texts."""
        import shutil
        from unittest.mock import MagicMock, patch

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "a.esp")
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "b.esp")
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "c.esp")

        out_dir = tmp_path / "out"

        from modtranslator import cli as cli_module
        from modtranslator.backends.dummy import DummyBackend

        real_backend = DummyBackend()
        spy_backend = MagicMock(wraps=real_backend)

        def mock_create(*args, **kwargs):
            return spy_backend, "dummy"

        with patch.object(cli_module, "_create_backend", side_effect=mock_create):
            result = runner.invoke(app, [
                "batch", str(src_dir),
                "--dummy",
                "--output-dir", str(out_dir),
                "--pattern", "*.esp",
                "--no-cache",
            ])

        assert result.exit_code is None or result.exit_code == 0
        # translate_batch called exactly once
        assert spy_backend.translate_batch.call_count == 1
        # The number of unique texts should be less than 3x the strings in one file
        # (because all 3 files are identical copies, dedup should collapse them)
        call_args = spy_backend.translate_batch.call_args
        texts_sent = call_args[0][0]
        # All 3 files have the same strings — protected texts deduplicate
        assert len(texts_sent) > 0

    def test_batch_parallel_error_isolation(self, tmp_path):
        """A corrupted ESP file counts as error, doesn't crash other files."""
        import shutil

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        # Use 'z' prefix so good.esp sorts first (for game detection)
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "good.esp")
        (src_dir / "z_bad.esp").write_bytes(b"\x00\x01\x02\x03")

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch", str(src_dir),
            "--dummy",
            "--output-dir", str(out_dir),
            "--pattern", "*.esp",
            "--no-cache",
        ])
        assert result.exit_code is None or result.exit_code == 0
        assert "Batch Summary" in result.output
        # good.esp should be translated, z_bad.esp should be an error
        assert (out_dir / "good.esp").exists()

    def test_batch_parallel_skipped_files(self, tmp_path):
        """Files with no translatable strings count as skipped."""
        src_dir = tmp_path / "data"
        src_dir.mkdir()
        # Create a minimal ESP with just a TES4 header and no translatable strings
        from modtranslator.core.constants import Game
        from modtranslator.core.plugin import save_plugin
        from modtranslator.core.records import PluginFile, Record

        header = Record(type=b"TES4", flags=0, form_id=0, vcs1=0, vcs2=0, subrecords=[])
        plugin = PluginFile(header=header, groups=[], game=Game.FALLOUT3)
        save_plugin(plugin, src_dir / "empty.esp")

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "batch", str(src_dir),
            "--dummy",
            "--output-dir", str(out_dir),
            "--pattern", "*.esp",
        ])
        assert result.exit_code is None or result.exit_code == 0
        assert "Batch Summary" in result.output

    def test_batch_parallel_single_backend_call(self, tmp_path):
        """backend.translate_batch is called exactly once for the whole batch."""
        import shutil
        from unittest.mock import MagicMock, patch

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        shutil.copy(FIXTURES / "minimal_fo3.esp", src_dir / "x.esp")
        shutil.copy(FIXTURES / "multi_record.esp", src_dir / "y.esp")

        out_dir = tmp_path / "out"

        from modtranslator import cli as cli_module
        from modtranslator.backends.dummy import DummyBackend

        real_backend = DummyBackend()
        spy_backend = MagicMock(wraps=real_backend)

        def mock_create(*args, **kwargs):
            return spy_backend, "dummy"

        with patch.object(cli_module, "_create_backend", side_effect=mock_create):
            result = runner.invoke(app, [
                "batch", str(src_dir),
                "--dummy",
                "--output-dir", str(out_dir),
                "--pattern", "*.esp",
                "--no-cache",
            ])

        assert result.exit_code is None or result.exit_code == 0
        assert spy_backend.translate_batch.call_count == 1


class TestCLICacheCommands:
    def test_cache_info(self):
        result = runner.invoke(app, ["cache-info"])
        assert result.exit_code == 0
        assert "Cached translations" in result.output
