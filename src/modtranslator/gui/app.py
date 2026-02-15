"""ModTranslator GUI - Main application window.

Single-view interface: select Data/ folder, click translate (in-place).
Auto-detects ESP/ESM, PEX, and MCM files and translates everything.
Backs up only the files that will be modified before translating.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from modtranslator import __version__
from modtranslator.gui.model_manager import (
    check_backend_ready,
    detect_cuda,
    download_model,
    get_model_status,
)
from modtranslator.gui.worker import TranslationWorker, WorkerMessage
from modtranslator.pipeline import (
    BatchAllResult,
    GameChoice,
    batch_translate_all,
    clear_cache,
    create_backend,
    get_cache_info,
    scan_directory,
)

# Settings persistence
_SETTINGS_DIR = Path.home() / ".modtranslator"
_SETTINGS_FILE = _SETTINGS_DIR / "gui_settings.json"

_PHASE_NAMES = {
    "scan": "Escaneando",
    "prepare": "Fase 1: Preparando",
    "translate": "Fase 2: Traduciendo",
    "write": "Fase 3: Escribiendo",
}


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


class FolderSelector(ctk.CTkFrame):
    """Label + entry + browse button for folder selection."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        label: str,
        initial_value: str = "",
        placeholder: str = "",
        on_change: object = None,
        **kwargs: object,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(1, weight=1)
        self._on_change = on_change

        ctk.CTkLabel(self, text=label, width=130, anchor="w").grid(
            row=0, column=0, padx=(0, 5), sticky="w",
        )
        self.var = ctk.StringVar(value=initial_value)
        self.entry = ctk.CTkEntry(
            self, textvariable=self.var,
            placeholder_text=placeholder,
        )
        self.entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        ctk.CTkButton(
            self, text="Examinar", width=80,
            command=self._browse,
        ).grid(row=0, column=2)

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Seleccionar carpeta")
        if path:
            self.var.set(path)
            if callable(self._on_change):
                self._on_change(path)

    @property
    def path(self) -> str:
        return self.var.get().strip()


# ═══════════════════════════════════════════════════════════════
# Main application window
# ═══════════════════════════════════════════════════════════════


class ModTranslatorApp(ctk.CTk):
    """Main application window — single view."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"ModTranslator v{__version__}")
        self.geometry("700x580")
        self.minsize(600, 480)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.settings = _load_settings()
        self.worker = TranslationWorker()

        self._build_ui()
        self._poll_worker()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main = ctk.CTkFrame(self)
        main.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main.grid_columnconfigure(0, weight=1)

        row = 0

        # ── Title ──
        ctk.CTkLabel(
            main, text=f"ModTranslator v{__version__}",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=row, column=0, sticky="w", padx=10, pady=(10, 5))
        row += 1

        ctk.CTkLabel(
            main,
            text="Traductor de mods",
            text_color="gray",
        ).grid(row=row, column=0, sticky="w", padx=10, pady=(0, 10))
        row += 1

        # ── Carpetas ──
        self.input_folder = FolderSelector(
            main, "Carpeta Data/:",
            self.settings.get("input_dir", ""),
            placeholder="Selecciona la carpeta Data/ del juego...",
            on_change=self._on_input_folder_change,
        )
        self.input_folder.grid(row=row, column=0, sticky="ew", padx=10, pady=2)
        row += 1

        self.scan_label = ctk.CTkLabel(main, text="", font=ctk.CTkFont(size=12))
        self.scan_label.grid(row=row, column=0, sticky="w", padx=15, pady=(0, 2))
        row += 1

        # ── Opciones ──
        opts = ctk.CTkFrame(main, fg_color="transparent")
        opts.grid(row=row, column=0, sticky="ew", padx=10, pady=(8, 2))
        row += 1

        ctk.CTkLabel(opts, text="Juego:").pack(side="left", padx=(0, 5))
        self.game_var = ctk.StringVar(value=self.settings.get("game", "Auto"))
        ctk.CTkOptionMenu(
            opts, variable=self.game_var,
            values=["Auto", "Fallout 3", "Fallout NV", "Skyrim"],
            width=120,
        ).pack(side="left", padx=(0, 15))

        motor_label = ctk.CTkLabel(opts, text="Motor de traducción:")
        motor_label.pack(side="left", padx=(0, 5))
        # Use saved backend preference; GPU detection happens in Settings
        has_gpu = self.settings.get("backend", "") == "Hybrid"
        self._backend_display_to_value = {
            "Híbrido (Recomendado con GPU)": "Hybrid",
            "Opus-MT (Recomendado sin GPU)": "Opus-MT",
            "DeepL (necesita API key)": "DeepL",
        }
        self._backend_value_to_display = {v: k for k, v in self._backend_display_to_value.items()}
        default_backend = "Hybrid" if has_gpu else "Opus-MT"
        saved_backend = self.settings.get("backend", default_backend)
        if saved_backend not in self._backend_value_to_display:
            saved_backend = default_backend
        self.backend_var = ctk.StringVar(
            value=self._backend_value_to_display[saved_backend],
        )
        motor_menu = ctk.CTkOptionMenu(
            opts, variable=self.backend_var,
            values=list(self._backend_display_to_value.keys()),
            width=250,
        )
        motor_menu.pack(side="left")
        Tooltip(
            motor_label,
            "Híbrido: usa dos modelos (mejor calidad, necesita GPU).\n"
            "Opus-MT: un solo modelo, funciona bien sin GPU.\n"
            "DeepL: servicio online de alta calidad (necesita API key).",
        )

        # ── Checkboxes ──
        checks = ctk.CTkFrame(main, fg_color="transparent")
        checks.grid(row=row, column=0, sticky="w", padx=10, pady=2)
        row += 1

        self.cache_var = ctk.BooleanVar(value=self.settings.get("use_cache", True))
        cache_cb = ctk.CTkCheckBox(
            checks, text="Recordar traducciones anteriores",
            variable=self.cache_var,
        )
        cache_cb.pack(side="left")
        Tooltip(
            cache_cb,
            "Guarda las traducciones en disco para no volver\n"
            "a traducir los mismos textos en futuras ejecuciones.\n"
            "Ahorra tiempo si traduces los mismos mods otra vez.",
        )

        # ── Botones ──
        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(10, 5))
        row += 1

        self.translate_btn = ctk.CTkButton(
            btn_frame, text="Traducir", width=160, height=36,
            fg_color="#28a745", hover_color="#218838",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start,
        )
        self.translate_btn.pack(side="left", padx=(0, 10))

        self.cancel_btn = ctk.CTkButton(
            btn_frame, text="Cancelar", width=100, height=36,
            fg_color="#dc3545", hover_color="#c82333",
            state="disabled",
            command=self._cancel,
        )
        self.cancel_btn.pack(side="left", padx=(0, 15))

        # Right side buttons
        self.settings_btn = ctk.CTkButton(
            btn_frame, text="Ajustes", width=80, height=36,
            fg_color="gray30", hover_color="gray40",
            command=self._open_settings,
        )
        self.settings_btn.pack(side="right")

        self.restore_btn = ctk.CTkButton(
            btn_frame, text="Restaurar backup", width=130, height=36,
            fg_color="gray30", hover_color="gray40",
            command=self._restore_backup,
        )
        self.restore_btn.pack(side="right", padx=(0, 5))

        # ── Progress ──
        self.progress_bar = ctk.CTkProgressBar(main)
        self.progress_bar.grid(row=row, column=0, sticky="ew", padx=10, pady=(5, 0))
        self.progress_bar.set(0)
        row += 1

        self.progress_label = ctk.CTkLabel(
            main, text="Selecciona la carpeta Data/ y pulsa Traducir para empezar",
            font=ctk.CTkFont(size=12),
        )
        self.progress_label.grid(row=row, column=0, sticky="w", padx=10)
        row += 1

        # ── Log ──
        ctk.CTkLabel(
            main, text="Registro de actividad:", font=ctk.CTkFont(size=12),
            text_color="gray",
        ).grid(row=row, column=0, sticky="w", padx=10, pady=(5, 0))
        row += 1

        main.grid_rowconfigure(row, weight=1)
        self.log = LogConsole(main)
        self.log.grid(row=row, column=0, sticky="nsew", padx=10, pady=(2, 10))

    # ── Auto-scan on folder selection ──

    def _on_input_folder_change(self, path: str) -> None:
        """Scan selected input folder and show summary."""
        folder = Path(path)
        if not folder.is_dir():
            self.scan_label.configure(text="Carpeta no válida", text_color="#dc3545")
            return
        scan = scan_directory(folder)
        parts = []
        if scan.esp_files:
            parts.append(f"{len(scan.esp_files)} plugins (ESP/ESM)")
        if scan.pex_files:
            parts.append(f"{len(scan.pex_files)} scripts (PEX)")
        if scan.has_mcm:
            parts.append("archivos MCM")
        if parts:
            self.scan_label.configure(
                text=f"Encontrados: {', '.join(parts)}",
                text_color="#28a745",
            )
        else:
            self.scan_label.configure(
                text="No se encontraron archivos traducibles",
                text_color="#dc3545",
            )

    # ── Translation ──

    def _get_game_choice(self) -> GameChoice:
        return {
            "Auto": GameChoice.auto,
            "Fallout 3": GameChoice.fo3,
            "Fallout NV": GameChoice.fnv,
            "Skyrim": GameChoice.skyrim,
        }.get(self.game_var.get(), GameChoice.auto)

    def _get_backend_name(self) -> str:
        display = self.backend_var.get()
        internal = self._backend_display_to_value.get(display, "Hybrid")
        return {
            "Hybrid": "hybrid",
            "Opus-MT": "opus-mt",
            "DeepL": "deepl",
        }.get(internal, "hybrid")

    def _validate(self) -> bool:
        if not self.input_folder.path:
            messagebox.showerror("Error", "Selecciona la carpeta Data/ del juego.")
            return False
        if not Path(self.input_folder.path).is_dir():
            messagebox.showerror("Error", f"Carpeta no existe: {self.input_folder.path}")
            return False

        backend_name = self._get_backend_name()
        if backend_name == "deepl" and not self.settings.get("deepl_api_key", "").strip():
            messagebox.showerror(
                "Error",
                "Introduce la API key de DeepL en Ajustes.",
            )
            return False

        ready, msg = check_backend_ready(backend_name)
        if not ready:
            messagebox.showerror("Backend no disponible", msg)
            return False

        # Quick scan to check there's something to translate
        scan = scan_directory(Path(self.input_folder.path))
        if not scan.esp_files and not scan.pex_files and not scan.has_mcm:
            messagebox.showwarning(
                "Sin archivos",
                "No se encontraron archivos traducibles (ESP/ESM, PEX, MCM) en la carpeta.",
            )
            return False

        return True

    def _save_current_settings(self) -> None:
        self.settings["input_dir"] = self.input_folder.path
        self.settings["game"] = self.game_var.get()
        display = self.backend_var.get()
        self.settings["backend"] = self._backend_display_to_value.get(display, "Hybrid")
        self.settings["use_cache"] = self.cache_var.get()
        _save_settings(self.settings)

    def _start(self) -> None:
        if self.worker.is_running:
            return
        if not self._validate():
            return

        self._save_current_settings()
        self.log.clear()
        self.progress_bar.set(0)
        self.progress_label.configure(text="Iniciando...")
        self.translate_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        backend_name = self._get_backend_name()
        api_key = self.settings.get("deepl_api_key", "").strip() or None

        try:
            backend, backend_label = create_backend(backend_name, api_key=api_key)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self._reset_buttons()
            return

        self.log.append(f"Backend: {backend_label}")

        input_path = Path(self.input_folder.path)

        # Backup selectivo: solo archivos que se van a modificar
        backup_dir = _SETTINGS_DIR / "backups" / input_path.name
        try:
            scan = scan_directory(input_path)
            files_to_backup: list[Path] = []
            # ESP/ESM y PEX se sobreescriben in-place
            files_to_backup.extend(scan.esp_files)
            files_to_backup.extend(scan.pex_files)
            # MCM y string tables crean archivos nuevos, no necesitan backup

            if files_to_backup:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
                for f in files_to_backup:
                    rel = f.relative_to(input_path)
                    dest = backup_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                self.settings["last_backup"] = str(backup_dir)
                self.settings["last_backup_source"] = str(input_path)
                _save_settings(self.settings)
                self.log.append(
                    f"Backup de {len(files_to_backup)} archivos en: {backup_dir}"
                )
            else:
                self.log.append("Sin archivos ESP/ESM/PEX para respaldar")
        except Exception as e:
            self.log.append(f"Aviso: no se pudo crear backup: {e}")

        # Traducir in-place: output_dir = input_path (sobreescribe originales)
        self.worker.start(
            batch_translate_all,
            directory=input_path,
            output_dir=input_path,
            lang="ES",
            backend=backend,
            backend_label=backend_label,
            game=self._get_game_choice(),
            skip_translated=True,
            no_cache=not self.cache_var.get(),
        )

    def _cancel(self) -> None:
        if self.worker.is_running:
            self.worker.cancel()
            self.log.append("Cancelando...")
            self.cancel_btn.configure(state="disabled")

    def _restore_backup(self) -> None:
        backup_dir = self.settings.get("last_backup", "")
        source_dir = self.settings.get("last_backup_source", "")
        if not backup_dir or not Path(backup_dir).exists():
            messagebox.showinfo(
                "Sin backup",
                "No hay ningún backup disponible.\n"
                "Se crea uno automáticamente cada vez que traduces.",
            )
            return

        backup_path = Path(backup_dir)
        backed_up_files = list(backup_path.rglob("*"))
        backed_up_files = [f for f in backed_up_files if f.is_file()]

        confirm = messagebox.askyesno(
            "Restaurar backup",
            f"Esto restaurará {len(backed_up_files)} archivos originales en:\n"
            f"{source_dir}\n\n¿Continuar?",
        )
        if not confirm:
            return
        try:
            dest = Path(source_dir)
            restored = 0
            for f in backed_up_files:
                rel = f.relative_to(backup_path)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                restored += 1
            self.log.append(f"Restaurados {restored} archivos en: {dest}")
            messagebox.showinfo("Restaurado", f"{restored} archivos restaurados correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error restaurando backup:\n{e}")

    def _reset_buttons(self) -> None:
        self.translate_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    # ── Worker polling ──

    def _poll_worker(self) -> None:
        while not self.worker.queue.empty():
            msg: WorkerMessage = self.worker.queue.get_nowait()
            self._handle_message(msg)
        self.after(100, self._poll_worker)

    def _handle_message(self, msg: WorkerMessage) -> None:
        if msg.type == "progress":
            phase_name = _PHASE_NAMES.get(msg.phase, msg.phase)
            if msg.total > 0:
                self.progress_bar.set(msg.current / msg.total)
                self.progress_label.configure(
                    text=f"{phase_name}: {msg.current}/{msg.total}",
                )
            if msg.message:
                self.log.append(f"> {msg.message}")

        elif msg.type == "done":
            result: BatchAllResult = msg.result
            self.progress_bar.set(1.0)

            elapsed = f"{result.elapsed_seconds:.1f}s"
            self.log.append(f"\n{'='*40}")
            self.log.append(f"Completado en {elapsed}")

            for label, r in [
                ("ESP/ESM", result.esp_result),
                ("PEX", result.pex_result),
                ("MCM", result.mcm_result),
            ]:
                if r:
                    self.log.append(
                        f"  {label}: {r.success_count} traducidos, "
                        f"{r.skip_count} omitidos, {r.total_strings} strings"
                    )
                    if r.error_count:
                        for fname, err in r.errors:
                            if fname:
                                self.log.append(f"    ERROR {fname}: {err}")

            self.progress_label.configure(
                text=f"Completado - {result.total_success} archivos traducidos",
            )
            self._reset_buttons()

        elif msg.type == "error":
            self.progress_bar.set(0)
            self.progress_label.configure(text="Error")
            self.log.append(f"\nERROR: {msg.message}")
            self._reset_buttons()

    # ── Settings dialog ──

    def _open_settings(self) -> None:
        SettingsWindow(self, self.settings)


# ═══════════════════════════════════════════════════════════════
# Settings window (popup)
# ═══════════════════════════════════════════════════════════════


class SettingsWindow(ctk.CTkToplevel):
    """Popup settings window."""

    def __init__(self, parent: ctk.CTk, settings: dict) -> None:
        super().__init__(parent)
        self.settings = settings

        self.title("Ajustes")
        self.geometry("500x450")
        self.minsize(400, 350)
        self.transient(parent)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)

        row = 0

        # ── GPU ──
        gpu_frame = ctk.CTkFrame(self)
        gpu_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(10, 5))
        row += 1

        ctk.CTkLabel(
            gpu_frame, text="GPU / CUDA",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self._gpu_label = ctk.CTkLabel(
            gpu_frame, text="Detectando GPU...", text_color="gray",
        )
        self._gpu_label.pack(anchor="w", padx=10, pady=(0, 10))

        # ── Models ──
        self._models_frame = ctk.CTkFrame(self)
        self._models_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        self._models_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(
            self._models_frame, text="Modelos ML",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5))

        self._models_loading = ctk.CTkLabel(
            self._models_frame, text="Comprobando modelos...", text_color="gray",
        )
        self._models_loading.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        # ── Cache ──
        cache_frame = ctk.CTkFrame(self)
        cache_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        row += 1

        ctk.CTkLabel(
            cache_frame, text="Cache de traducciones",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self.cache_label = ctk.CTkLabel(
            cache_frame, text="Comprobando cache...", text_color="gray",
        )
        self.cache_label.pack(anchor="w", padx=10)

        self._cache_path_label = ctk.CTkLabel(
            cache_frame, text="", text_color="gray",
        )
        self._cache_path_label.pack(anchor="w", padx=10, pady=(0, 5))

        ctk.CTkButton(
            cache_frame, text="Limpiar cache", width=120,
            fg_color="#dc3545", hover_color="#c82333",
            command=self._clear_cache,
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # ── DeepL API key ──
        api_frame = ctk.CTkFrame(self)
        api_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        api_frame.grid_columnconfigure(0, weight=1)
        row += 1

        ctk.CTkLabel(
            api_frame, text="DeepL API Key",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        self.api_key_var = ctk.StringVar(value=settings.get("deepl_api_key", ""))
        ctk.CTkEntry(
            api_frame, textvariable=self.api_key_var, show="*",
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        ctk.CTkButton(
            api_frame, text="Guardar", width=80,
            command=self._save_api_key,
        ).grid(row=1, column=1, padx=(5, 10), pady=(0, 5))

        # Load heavy data in background thread
        threading.Thread(target=self._load_settings_data, daemon=True).start()

    def _load_settings_data(self) -> None:
        """Load GPU, model, and cache info in background."""
        cuda_info = detect_cuda()
        models = get_model_status()
        cache_info = get_cache_info()
        self.after(0, lambda: self._populate_settings(cuda_info, models, cache_info))

    def _populate_settings(
        self, cuda_info: dict, models: list, cache_info: dict,
    ) -> None:
        """Populate settings UI with loaded data (runs on main thread)."""
        # GPU
        if cuda_info["available"]:
            self._gpu_label.configure(
                text=f"GPU detectada: {cuda_info['gpu_name']}",
                text_color="#28a745",
            )
        else:
            self._gpu_label.configure(
                text="No se detecta GPU CUDA (se usara CPU)",
                text_color="#ffc107",
            )

        # Models
        self._models_loading.destroy()
        for i, model in enumerate(models):
            status = "Descargado" if model.is_downloaded else "No descargado"
            color = "#28a745" if model.is_downloaded else "#dc3545"

            ctk.CTkLabel(
                self._models_frame, text=f"{model.name} ({model.size_hint})",
            ).grid(row=i + 1, column=0, sticky="w", padx=10, pady=2)

            ctk.CTkLabel(
                self._models_frame, text=status, text_color=color,
            ).grid(row=i + 1, column=1, sticky="w", padx=5, pady=2)

            if not model.is_downloaded:
                ctk.CTkButton(
                    self._models_frame, text="Descargar", width=80,
                    command=lambda m=model: self._download(m),
                ).grid(row=i + 1, column=2, padx=10, pady=2)

        # Cache
        self.cache_label.configure(
            text=f"Traducciones en cache: {cache_info['count']}",
            text_color=("white", "white"),
        )
        self._cache_path_label.configure(text=f"Ubicacion: {cache_info['path']}")

    def _download(self, model: object) -> None:
        from modtranslator.gui.model_manager import ModelInfo
        assert isinstance(model, ModelInfo)

        def _do_download() -> None:
            success = download_model(model.description)
            self.after(0, lambda: self._on_download_done(model.name, success))

        self._download_status = ctk.CTkLabel(
            self, text=f"Descargando {model.name}...", text_color="#ffc107",
        )
        self._download_status.grid(row=10, column=0, padx=10, pady=5)
        threading.Thread(target=_do_download, daemon=True).start()

    def _on_download_done(self, name: str, success: bool) -> None:
        if hasattr(self, "_download_status"):
            self._download_status.destroy()
        if success:
            messagebox.showinfo("Éxito", f"{name} descargado correctamente.")
        else:
            messagebox.showerror("Error", f"Error descargando {name}.")

    def _clear_cache(self) -> None:
        deleted = clear_cache()
        self.cache_label.configure(text="Traducciones en cache: 0")
        messagebox.showinfo("Cache", f"Se eliminaron {deleted} traducciones del cache.")

    def _save_api_key(self) -> None:
        self.settings["deepl_api_key"] = self.api_key_var.get().strip()
        _save_settings(self.settings)
        messagebox.showinfo("Guardado", "API key guardada.")


def run_gui() -> None:
    """Launch the ModTranslator GUI."""
    app = ModTranslatorApp()
    app.mainloop()
