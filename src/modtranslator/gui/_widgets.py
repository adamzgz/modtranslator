"""GUI reusable widgets and helper functions.

Extracted from gui/app.py for maintainability.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from modtranslator.pipeline import (
    BatchAllResult,
    GameChoice,
    batch_translate_esp,
    batch_translate_pex,
)

# Settings persistence
_SETTINGS_DIR = Path.home() / ".modtranslator"
_SETTINGS_FILE = _SETTINGS_DIR / "gui_settings.json"

_ESP_EXTENSIONS = {".esp", ".esm", ".esl"}
_PEX_EXTENSION = ".pex"


def _load_settings() -> dict:
    try:
        if _SETTINGS_FILE.exists():
            return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_settings(settings: dict) -> None:
    try:
        _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _translate_single_file(
    file_path: Path,
    *,
    output_dir: Path,
    lang: str,
    backend: object,
    backend_label: str,
    game: GameChoice,
    skip_translated: bool,
    no_cache: bool,
    on_progress: object,
    cancel_event: object,
) -> BatchAllResult:
    """Translate a single ESP/ESM/ESL or PEX file. Returns BatchAllResult for uniform handling."""
    t0 = time.monotonic()
    result = BatchAllResult()

    if file_path.suffix.lower() == _PEX_EXTENSION:
        r = batch_translate_pex(
            [file_path],
            lang=lang, backend=backend, backend_label=backend_label,  # type: ignore[arg-type]
            game=game, skip_translated=skip_translated,
            output_dir=output_dir, no_cache=no_cache,
            on_progress=on_progress, cancel_event=cancel_event,  # type: ignore[arg-type]
        )
        result.pex_result = r
    else:
        r = batch_translate_esp(
            [file_path],
            lang=lang, backend=backend, backend_label=backend_label,  # type: ignore[arg-type]
            game=game, skip_translated=skip_translated,
            output_dir=output_dir, no_cache=no_cache,
            on_progress=on_progress, cancel_event=cancel_event,  # type: ignore[arg-type]
        )
        result.esp_result = r

    result.total_success = r.success_count
    result.total_errors = r.error_count
    result.elapsed_seconds = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════
# Reusable widgets
# ═══════════════════════════════════════════════════════════════


class Tooltip:
    """Hover tooltip for any widget."""

    def __init__(self, widget: ctk.CTkBaseClass, text: str) -> None:
        self.widget = widget
        self.text = text
        self._after_id: str | None = None
        self.tw: ctk.CTkToplevel | None = None
        widget.bind("<Enter>", self._schedule_show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _schedule_show(self, _event: object) -> None:
        self._hide()
        self._after_id = self.widget.after(400, self._show)

    def _show(self, _event: object = None) -> None:
        self._after_id = None
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tw = ctk.CTkToplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_attributes("-topmost", True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = ctk.CTkLabel(
            self.tw, text=self.text,
            fg_color="gray20", corner_radius=6,
            padx=8, pady=4,
            font=ctk.CTkFont(size=12),
        )
        label.pack()

    def _hide(self, _event: object = None) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self.tw:
            self.tw.destroy()
            self.tw = None


class LogConsole(ctk.CTkTextbox):
    """Read-only log area with auto-scroll."""

    def __init__(self, master: ctk.CTkBaseClass, **kwargs: object) -> None:
        kwargs.setdefault("height", 100)
        kwargs.setdefault("state", "disabled")
        kwargs.setdefault("font", ctk.CTkFont(family="Consolas", size=12))
        super().__init__(master, **kwargs)

    def append(self, text: str) -> None:
        self.configure(state="normal")
        self.insert("end", text + "\n")
        self.see("end")
        self.configure(state="disabled")

    def clear(self) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class PathSelector(ctk.CTkFrame):
    """Label + entry + browse button(s). Supports folder-only or folder+file mode."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        label: str,
        initial_value: str = "",
        placeholder: str = "",
        on_change: object = None,
        allow_files: bool = False,
        file_types: list[tuple[str, str]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(1, weight=1)
        self._on_change = on_change
        self._file_types = file_types or [("Todos", "*.*")]

        self._label_widget = ctk.CTkLabel(self, text=label, width=130, anchor="w")
        self._label_widget.grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.var = ctk.StringVar(value=initial_value)
        self.entry = ctk.CTkEntry(
            self, textvariable=self.var,
            placeholder_text=placeholder,
        )
        self.entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        self._btn_folder_widget: ctk.CTkButton | None = None
        self._btn_file_widget: ctk.CTkButton | None = None
        self._btn_browse_widget: ctk.CTkButton | None = None

        if allow_files:
            self._btn_folder_widget = ctk.CTkButton(
                self, text="Carpeta", width=72,
                command=self._browse_folder,
            )
            self._btn_folder_widget.grid(row=0, column=2, padx=(0, 3))
            self._btn_file_widget = ctk.CTkButton(
                self, text="Archivo", width=72,
                command=self._browse_file,
            )
            self._btn_file_widget.grid(row=0, column=3)
        else:
            self._btn_browse_widget = ctk.CTkButton(
                self, text="Examinar", width=80,
                command=self._browse_folder,
            )
            self._btn_browse_widget.grid(row=0, column=2)

    def update_label(self, text: str) -> None:
        self._label_widget.configure(text=text)

    def update_buttons(
        self,
        *,
        folder: str | None = None,
        file: str | None = None,
        browse: str | None = None,
    ) -> None:
        if folder is not None and self._btn_folder_widget is not None:
            self._btn_folder_widget.configure(text=folder)
        if file is not None and self._btn_file_widget is not None:
            self._btn_file_widget.configure(text=file)
        if browse is not None and self._btn_browse_widget is not None:
            self._btn_browse_widget.configure(text=browse)

    def update_placeholder(self, text: str) -> None:
        self.entry.configure(placeholder_text=text)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Seleccionar carpeta")
        if path:
            self.var.set(path)
            if callable(self._on_change):
                self._on_change(path)

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=self._file_types,
        )
        if path:
            self.var.set(path)
            if callable(self._on_change):
                self._on_change(path)

    @property
    def path(self) -> str:
        return self.var.get().strip()


# Alias for backwards compatibility
FolderSelector = PathSelector
