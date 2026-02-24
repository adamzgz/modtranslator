"""Tests for GUI app helpers and standalone functions."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level stubs for customtkinter (avoid importing real Tk)
# ---------------------------------------------------------------------------

class _StubBase:
    """Minimal base class that accepts arbitrary kwargs and supports configure/bind."""
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def configure(self, **kwargs: object) -> None:
        pass

    def bind(self, *args: object, **kwargs: object) -> None:
        pass

    def after(self, *args: object, **kwargs: object) -> None:
        pass


def _ensure_ctk_stub() -> None:
    """Stub customtkinter and tkinter bits so app.py can be imported without a display."""
    if "customtkinter" not in sys.modules:
        ctk = types.ModuleType("customtkinter")
        # Classes that app.py inherits from — must be real classes, not MagicMock,
        # so that __new__ / __setattr__ work properly on subclasses.
        for cls_name in (
            "CTk", "CTkToplevel", "CTkFrame", "CTkTextbox", "CTkBaseClass",
            "CTkLabel", "CTkButton", "CTkEntry", "CTkProgressBar",
            "CTkOptionMenu", "CTkCheckBox", "CTkScrollableFrame",
            "StringVar", "BooleanVar",
        ):
            setattr(ctk, cls_name, _StubBase)
        # CTkFont is called as a function
        ctk.CTkFont = lambda **kwargs: None  # type: ignore[attr-defined]
        ctk.set_appearance_mode = lambda *a: None  # type: ignore[attr-defined]
        ctk.set_default_color_theme = lambda *a: None  # type: ignore[attr-defined]
        sys.modules["customtkinter"] = ctk

    if "tkinter" not in sys.modules:
        tk = types.ModuleType("tkinter")
        tk.filedialog = MagicMock()  # type: ignore[attr-defined]
        tk.messagebox = MagicMock()  # type: ignore[attr-defined]
        sys.modules["tkinter"] = tk
        sys.modules["tkinter.filedialog"] = tk.filedialog  # type: ignore[attr-defined]
        sys.modules["tkinter.messagebox"] = tk.messagebox  # type: ignore[attr-defined]


_ensure_ctk_stub()

from modtranslator.gui.app import (  # noqa: E402
    _UI_STRINGS,
    _detect_gpu_type,
    _has_any_ml_backend,
    _load_settings,
    _save_settings,
)

# ---------------------------------------------------------------------------
# _UI_STRINGS completeness
# ---------------------------------------------------------------------------


class TestUIStrings:
    def test_all_languages_have_same_keys(self) -> None:
        """Every language dict must have exactly the same keys as EN."""
        en_keys = set(_UI_STRINGS["EN"].keys())
        for lang, strings in _UI_STRINGS.items():
            missing = en_keys - set(strings.keys())
            extra = set(strings.keys()) - en_keys
            assert not missing, f"{lang} missing keys: {missing}"
            assert not extra, f"{lang} extra keys: {extra}"

    def test_expected_languages_present(self) -> None:
        expected = {"EN", "ES", "FR", "DE", "IT", "PT", "RU", "PL"}
        assert set(_UI_STRINGS.keys()) == expected

    def test_no_empty_values(self) -> None:
        for lang, strings in _UI_STRINGS.items():
            for key, value in strings.items():
                assert value, f"{lang}.{key} is empty"


# ---------------------------------------------------------------------------
# _load_settings / _save_settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_load_returns_empty_dict_when_file_missing(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "nonexistent.json"
        with patch("modtranslator.gui.app._SETTINGS_FILE", fake_file):
            result = _load_settings()
        assert result == {}

    def test_load_returns_empty_dict_on_corrupt_json(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "bad.json"
        fake_file.write_text("{invalid json", encoding="utf-8")
        with patch("modtranslator.gui.app._SETTINGS_FILE", fake_file):
            result = _load_settings()
        assert result == {}

    def test_load_returns_dict_from_valid_json(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "settings.json"
        data = {"lang": "ES", "backend": "hybrid"}
        fake_file.write_text(json.dumps(data), encoding="utf-8")
        with patch("modtranslator.gui.app._SETTINGS_FILE", fake_file):
            result = _load_settings()
        assert result == data

    def test_save_creates_file(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "subdir"
        settings_file = settings_dir / "settings.json"
        with (
            patch("modtranslator.gui.app._SETTINGS_DIR", settings_dir),
            patch("modtranslator.gui.app._SETTINGS_FILE", settings_file),
        ):
            _save_settings({"key": "value"})
        assert settings_file.exists()
        loaded = json.loads(settings_file.read_text(encoding="utf-8"))
        assert loaded == {"key": "value"}

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        deep_dir = tmp_path / "a" / "b" / "c"
        settings_file = deep_dir / "settings.json"
        with (
            patch("modtranslator.gui.app._SETTINGS_DIR", deep_dir),
            patch("modtranslator.gui.app._SETTINGS_FILE", settings_file),
        ):
            _save_settings({"nested": True})
        assert deep_dir.exists()
        assert settings_file.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "rt"
        settings_file = settings_dir / "settings.json"
        data = {"lang": "FR", "cache": True, "deepl_key": "abc123"}
        with (
            patch("modtranslator.gui.app._SETTINGS_DIR", settings_dir),
            patch("modtranslator.gui.app._SETTINGS_FILE", settings_file),
        ):
            _save_settings(data)
            result = _load_settings()
        assert result == data


# ---------------------------------------------------------------------------
# _has_any_ml_backend
# ---------------------------------------------------------------------------


class TestHasAnyMLBackend:
    def test_true_when_ctranslate2_available(self) -> None:
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.side_effect = lambda name: MagicMock() if name == "ctranslate2" else None
            assert _has_any_ml_backend() is True

    def test_true_when_torch_available(self) -> None:
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.side_effect = lambda name: MagicMock() if name == "torch" else None
            assert _has_any_ml_backend() is True

    def test_false_when_neither_available(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            assert _has_any_ml_backend() is False


# ---------------------------------------------------------------------------
# _detect_gpu_type
# ---------------------------------------------------------------------------


class TestDetectGpuType:
    def test_nvidia_detected_via_wmic(self) -> None:
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "Name=NVIDIA GeForce RTX 3080\n"
        with patch("modtranslator.gui.app.subprocess.run", return_value=fake_result):
            gpu_type, name = _detect_gpu_type()
        assert gpu_type == "nvidia"
        assert "RTX 3080" in name

    def test_quadro_detected_via_wmic(self) -> None:
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "Name=NVIDIA Quadro P4000\n"
        with patch("modtranslator.gui.app.subprocess.run", return_value=fake_result):
            gpu_type, name = _detect_gpu_type()
        assert gpu_type == "nvidia"
        assert "Quadro" in name

    def test_no_gpu_returns_cpu(self) -> None:
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "Name=Intel UHD Graphics 630\n"

        # wmic returns non-NVIDIA, nvidia-smi fails
        def fake_run(cmd: list, **kwargs: object) -> MagicMock:
            if "nvidia-smi" in cmd:
                raise FileNotFoundError("nvidia-smi not found")
            return fake_result

        with patch("modtranslator.gui.app.subprocess.run", side_effect=fake_run):
            gpu_type, name = _detect_gpu_type()
        assert gpu_type == "cpu"
        assert name == "CPU"

    def test_nvidia_smi_fallback(self) -> None:
        """If wmic fails, fall back to nvidia-smi."""
        wmic_fail = MagicMock()
        wmic_fail.returncode = 1
        wmic_fail.stdout = ""

        smi_ok = MagicMock()
        smi_ok.returncode = 0
        smi_ok.stdout = "Tesla V100-SXM2\n"

        def fake_run(cmd: list, **kwargs: object) -> MagicMock:
            if "nvidia-smi" in cmd:
                return smi_ok
            return wmic_fail

        with patch("modtranslator.gui.app.subprocess.run", side_effect=fake_run):
            gpu_type, name = _detect_gpu_type()
        assert gpu_type == "nvidia"
        assert "Tesla" in name

    def test_all_commands_fail(self) -> None:
        """When every subprocess call raises, return cpu."""
        with patch(
            "modtranslator.gui.app.subprocess.run",
            side_effect=Exception("fail"),
        ):
            gpu_type, name = _detect_gpu_type()
        assert gpu_type == "cpu"
        assert name == "CPU"


# ---------------------------------------------------------------------------
# LogConsole (mocked)
# ---------------------------------------------------------------------------


class TestLogConsole:
    def test_append_inserts_text(self) -> None:
        from modtranslator.gui.app import LogConsole

        console = LogConsole(MagicMock())
        console.configure = MagicMock()
        console.insert = MagicMock()
        console.see = MagicMock()

        console.append("hello")
        console.configure.assert_called()
        console.insert.assert_called_once_with("end", "hello\n")
        console.see.assert_called_once_with("end")

    def test_clear_deletes_all(self) -> None:
        from modtranslator.gui.app import LogConsole

        console = LogConsole(MagicMock())
        console.configure = MagicMock()
        console.delete = MagicMock()

        console.clear()
        console.delete.assert_called_once_with("1.0", "end")


# ---------------------------------------------------------------------------
# ModTranslatorApp._t (translation helper)
# ---------------------------------------------------------------------------


def _make_bare_app() -> object:
    """Create a bare ModTranslatorApp without calling __init__."""
    from modtranslator.gui.app import ModTranslatorApp
    app = object.__new__(ModTranslatorApp)
    return app


class TestTranslationHelper:
    def test_returns_translated_string(self) -> None:
        """_t should return the string for the current language."""
        app = _make_bare_app()
        app._ui_lang = "ES"  # type: ignore[attr-defined]
        result = app._t("btn_translate")  # type: ignore[attr-defined]
        assert result == "Traducir"

    def test_fallback_to_en(self) -> None:
        """Unknown language should fall back to EN."""
        app = _make_bare_app()
        app._ui_lang = "XX"  # type: ignore[attr-defined]
        result = app._t("btn_translate")  # type: ignore[attr-defined]
        assert result == "Translate"

    def test_unknown_key_returns_key(self) -> None:
        """Missing key should return the key itself."""
        app = _make_bare_app()
        app._ui_lang = "EN"  # type: ignore[attr-defined]
        result = app._t("nonexistent_key_xyz")  # type: ignore[attr-defined]
        assert result == "nonexistent_key_xyz"


# ---------------------------------------------------------------------------
# ModTranslatorApp._handle_message
# ---------------------------------------------------------------------------


class TestHandleMessage:
    def _make_app(self) -> object:
        """Create a bare ModTranslatorApp with mocked widgets."""
        app = _make_bare_app()
        app._ui_lang = "EN"  # type: ignore[attr-defined]
        app.progress_bar = MagicMock()  # type: ignore[attr-defined]
        app.progress_label = MagicMock()  # type: ignore[attr-defined]
        app.log = MagicMock()  # type: ignore[attr-defined]
        app.translate_btn = MagicMock()  # type: ignore[attr-defined]
        app.cancel_btn = MagicMock()  # type: ignore[attr-defined]
        return app

    def test_progress_updates_bar_and_label(self) -> None:
        from modtranslator.gui.worker import WorkerMessage

        app = self._make_app()
        msg = WorkerMessage(type="progress", phase="translate", current=5, total=10)
        app._handle_message(msg)  # type: ignore[union-attr]

        app.progress_bar.set.assert_called_once_with(0.5)  # type: ignore[union-attr]
        app.progress_label.configure.assert_called_once()  # type: ignore[union-attr]

    def test_progress_with_message_appends_to_log(self) -> None:
        from modtranslator.gui.worker import WorkerMessage

        app = self._make_app()
        msg = WorkerMessage(
            type="progress", phase="scan", current=1, total=5, message="file.esp",
        )
        app._handle_message(msg)  # type: ignore[union-attr]
        app.log.append.assert_called_once_with("> file.esp")  # type: ignore[union-attr]

    def test_error_resets_bar_and_buttons(self) -> None:
        from modtranslator.gui.worker import WorkerMessage

        app = self._make_app()
        msg = WorkerMessage(type="error", message="something broke")
        app._handle_message(msg)  # type: ignore[union-attr]

        app.progress_bar.set.assert_called_once_with(0)  # type: ignore[union-attr]
        app.translate_btn.configure.assert_called()  # type: ignore[union-attr]
        app.cancel_btn.configure.assert_called()  # type: ignore[union-attr]
        app.log.append.assert_called()  # type: ignore[union-attr]

    def test_done_sets_bar_to_full(self) -> None:
        from modtranslator.gui.worker import WorkerMessage
        from modtranslator.pipeline import BatchAllResult

        result = BatchAllResult()
        result.elapsed_seconds = 1.5
        result.total_success = 3

        app = self._make_app()
        msg = WorkerMessage(type="done", result=result)
        app._handle_message(msg)  # type: ignore[union-attr]

        app.progress_bar.set.assert_called_once_with(1.0)  # type: ignore[union-attr]
        app.translate_btn.configure.assert_called()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# ModTranslatorApp._on_input_path_change
# ---------------------------------------------------------------------------


class TestOnInputPathChange:
    def _make_app(self) -> object:
        app = _make_bare_app()
        app._ui_lang = "EN"  # type: ignore[attr-defined]
        app.scan_label = MagicMock()  # type: ignore[attr-defined]
        return app

    def test_esp_file_shows_green(self, tmp_path: Path) -> None:
        esp = tmp_path / "test.esp"
        esp.write_bytes(b"\x00")

        app = self._make_app()
        app._on_input_path_change(str(esp))  # type: ignore[union-attr]

        call_kwargs = app.scan_label.configure.call_args  # type: ignore[union-attr]
        assert "test.esp" in call_kwargs.kwargs.get("text", call_kwargs[1].get("text", ""))
        assert "#28a745" in str(call_kwargs)

    def test_pex_file_shows_green(self, tmp_path: Path) -> None:
        pex = tmp_path / "script.pex"
        pex.write_bytes(b"\x00")

        app = self._make_app()
        app._on_input_path_change(str(pex))  # type: ignore[union-attr]

        call_kwargs = app.scan_label.configure.call_args  # type: ignore[union-attr]
        assert "script.pex" in str(call_kwargs)

    def test_unsupported_file_shows_red(self, tmp_path: Path) -> None:
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")

        app = self._make_app()
        app._on_input_path_change(str(txt))  # type: ignore[union-attr]

        call_kwargs = app.scan_label.configure.call_args  # type: ignore[union-attr]
        assert "#dc3545" in str(call_kwargs)

    def test_invalid_path_shows_red(self) -> None:
        app = self._make_app()
        app._on_input_path_change("/nonexistent/path/xyz")  # type: ignore[union-attr]

        call_kwargs = app.scan_label.configure.call_args  # type: ignore[union-attr]
        assert "#dc3545" in str(call_kwargs)

    def test_directory_with_files(self, tmp_path: Path) -> None:
        scan_result = MagicMock()
        scan_result.esp_files = ["a.esp", "b.esm"]
        scan_result.pex_files = []
        scan_result.has_mcm = False

        app = self._make_app()
        with patch("modtranslator.gui.app.scan_directory", return_value=scan_result):
            app._on_input_path_change(str(tmp_path))  # type: ignore[union-attr]

        call_kwargs = app.scan_label.configure.call_args  # type: ignore[union-attr]
        assert "#28a745" in str(call_kwargs)
        assert "2" in str(call_kwargs)  # 2 plugins

    def test_empty_directory(self, tmp_path: Path) -> None:
        scan_result = MagicMock()
        scan_result.esp_files = []
        scan_result.pex_files = []
        scan_result.has_mcm = False

        app = self._make_app()
        with patch("modtranslator.gui.app.scan_directory", return_value=scan_result):
            app._on_input_path_change(str(tmp_path))  # type: ignore[union-attr]

        call_kwargs = app.scan_label.configure.call_args  # type: ignore[union-attr]
        assert "#dc3545" in str(call_kwargs)
