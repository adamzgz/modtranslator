"""ModTranslator GUI - Main application window.

Single-view interface: select Data/ folder or single mod file, click translate.
Auto-detects ESP/ESM, PEX, and MCM files and translates everything.
Backs up only the files that will be modified before translating.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from modtranslator import __version__

# ── Re-exports from _strings ──
from modtranslator.gui._strings import _UI_STRINGS  # noqa: F401

# ── Re-exports from _widgets ──
from modtranslator.gui._widgets import (  # noqa: F401
    _ESP_EXTENSIONS,
    _PEX_EXTENSION,
    _SETTINGS_DIR,
    _SETTINGS_FILE,
    FolderSelector,
    LogConsole,
    PathSelector,
    Tooltip,
    _load_settings,
    _save_settings,
    _translate_single_file,
)
from modtranslator.gui.model_manager import (
    check_backend_ready,
    detect_cuda,
    download_model,
    get_missing_model_ids,
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

# ═══════════════════════════════════════════════════════════════
# Main application window
# ═══════════════════════════════════════════════════════════════


class ModTranslatorApp(ctk.CTk):
    """Main application window — single view."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"ModTranslator v{__version__}")
        self.geometry("700x620")
        self.minsize(600, 520)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.settings = _load_settings()
        self.worker = TranslationWorker()
        self._setup_proc: subprocess.Popen | None = None
        self._setup_cancel = threading.Event()
        self._has_hybrid = True  # Assume hybrid until GPU detection overrides
        # UI language: use saved lang if it exists, otherwise default to English
        self._ui_lang: str = self.settings.get("lang", "EN")

        self._build_ui()
        self._apply_lang()  # Apply UI language on startup
        self.lang_var.trace_add("write", self._on_lang_change)
        self._poll_worker()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Detectar GPU siempre al arrancar para filtrar opciones del dropdown
        threading.Thread(target=self._detect_gpu_for_default, daemon=True).start()
        # Primera ejecución: instalar backends ML en background sin bloquear la GUI
        self._maybe_start_setup()

    def _on_close(self) -> None:
        """Terminate child processes and exit cleanly."""
        self._setup_cancel.set()
        if self._setup_proc is not None and self._setup_proc.poll() is None:
            self._setup_proc.terminate()
        if self.worker.is_running:
            self.worker.cancel()
        self.destroy()

    # ── First-run background setup ──────────────────────────────

    def _maybe_start_setup(self) -> None:
        if _has_any_ml_backend():
            return
        self.translate_btn.configure(state="disabled")
        self.log.append("Primera ejecución: instalando dependencias ML en background...")
        self.log.append("Puedes explorar la interfaz mientras tanto.")
        threading.Thread(target=self._run_background_setup, daemon=True).start()

    def _run_background_setup(self) -> None:
        def log(msg: str) -> None:
            self.after(0, lambda m=msg: self.log.append(m))

        gpu_type, gpu_name = _detect_gpu_type()
        gpu_labels = {
            "nvidia": f"GPU NVIDIA detectada: {gpu_name}",
            "cpu": "Sin GPU dedicada — instalando backends CPU",
        }
        log(gpu_labels[gpu_type])

        steps = _SETUP_STEPS[gpu_type]
        total = len(steps)
        for i, (desc, packages) in enumerate(steps, 1):
            if self._setup_cancel.is_set():
                return
            log(f"[{i}/{total}] {desc}")
            self._setup_proc = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", *packages],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            proc = self._setup_proc
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line and any(
                    kw in line for kw in ("Downloading", "Installing", "Successfully")
                ):
                    log(f"  {line}")
            proc.wait()

        importlib.invalidate_caches()
        log("Dependencias instaladas. ¡Listo para traducir!")
        log("Los modelos se descargarán automáticamente la primera vez que traduzcas.")
        # Re-detectar GPU ahora que los paquetes están instalados
        threading.Thread(target=self._detect_gpu_for_default, daemon=True).start()
        self.after(0, lambda: self.translate_btn.configure(state="normal"))

    # ── UI ──────────────────────────────────────────────────────

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

        self._subtitle_label = ctk.CTkLabel(
            main,
            text="Traductor de mods",
            text_color="gray",
        )
        self._subtitle_label.grid(row=row, column=0, sticky="w", padx=10, pady=(0, 10))
        row += 1

        # ── Entrada ──
        self.input_folder = PathSelector(
            main, "Entrada:",
            self.settings.get("input_dir", ""),
            placeholder="Carpeta Data/ o archivo .esp/.esm/.pex...",
            on_change=self._on_input_path_change,
            allow_files=True,
            file_types=[
                ("Archivos de mod", "*.esp;*.esm;*.esl;*.pex"),
                ("Todos", "*.*"),
            ],
        )
        self.input_folder.grid(row=row, column=0, sticky="ew", padx=10, pady=2)
        row += 1

        self.scan_label = ctk.CTkLabel(main, text="", font=ctk.CTkFont(size=12))
        self.scan_label.grid(row=row, column=0, sticky="w", padx=15, pady=(0, 2))
        row += 1

        # ── Salida ──
        self.output_folder = PathSelector(
            main, "Carpeta de salida:",
            self.settings.get("output_dir", ""),
            placeholder="(misma que la entrada — in-place)",
        )
        self.output_folder.grid(row=row, column=0, sticky="ew", padx=10, pady=2)
        row += 1

        # ── Opciones ──
        opts = ctk.CTkFrame(main, fg_color="transparent")
        opts.grid(row=row, column=0, sticky="ew", padx=10, pady=(8, 2))
        row += 1

        self._game_label = ctk.CTkLabel(opts, text="Juego:")
        self._game_label.pack(side="left", padx=(0, 5))
        self.game_var = ctk.StringVar(value=self.settings.get("game", "Auto"))
        ctk.CTkOptionMenu(
            opts, variable=self.game_var,
            values=["Auto", "Fallout 3", "Fallout NV", "Fallout 4", "Skyrim", "Minecraft"],
            width=120,
        ).pack(side="left", padx=(0, 15))

        self._lang_label = ctk.CTkLabel(opts, text="Idioma:")
        self._lang_label.pack(side="left", padx=(0, 5))
        self.lang_var = ctk.StringVar(value=self.settings.get("lang", "ES"))
        ctk.CTkOptionMenu(
            opts, variable=self.lang_var,
            values=["ES", "FR", "DE", "IT", "PT", "RU"],
            width=80,
        ).pack(side="left", padx=(0, 15))

        self._backend_label = ctk.CTkLabel(opts, text="Motor de traducción:")
        self._backend_label.pack(side="left", padx=(0, 5))
        # Opciones iniciales (todas) — _apply_gpu_default filtrará según GPU detectada
        self._backend_display_to_value = {
            "Híbrido (Recomendado con GPU Nvidia)": "Hybrid",
            "Opus-MT (sin GPU)": "Opus-MT",
            "DeepL (necesita API key)": "DeepL",
        }
        self._backend_value_to_display = {v: k for k, v in self._backend_display_to_value.items()}
        saved_backend = self.settings.get("backend", "Opus-MT")
        if saved_backend not in self._backend_value_to_display:
            saved_backend = "Opus-MT"
        self.backend_var = ctk.StringVar(
            value=self._backend_value_to_display[saved_backend],
        )
        self._motor_menu = ctk.CTkOptionMenu(
            opts, variable=self.backend_var,
            values=list(self._backend_display_to_value.keys()),
            width=250,
        )
        self._motor_menu.pack(side="left")
        self._motor_tooltip = Tooltip(
            self._backend_label,
            "Híbrido: Opus-MT + NLLB via CTranslate2 (GPU Nvidia/CUDA).\n"
            "Opus-MT: un modelo CTranslate2, funciona bien sin GPU.\n"
            "DeepL: servicio online de alta calidad (necesita API key).",
        )

        # ── Checkboxes ──
        checks = ctk.CTkFrame(main, fg_color="transparent")
        checks.grid(row=row, column=0, sticky="w", padx=10, pady=2)
        row += 1

        self.cache_var = ctk.BooleanVar(value=self.settings.get("use_cache", True))
        self._cache_cb = ctk.CTkCheckBox(
            checks, text="Recordar traducciones anteriores",
            variable=self.cache_var,
        )
        self._cache_cb.pack(side="left")
        self._cache_tooltip = Tooltip(
            self._cache_cb,
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
            main, text="Selecciona la entrada y pulsa Traducir para empezar",
            font=ctk.CTkFont(size=12),
        )
        self.progress_label.grid(row=row, column=0, sticky="w", padx=10)
        row += 1

        # ── Log ──
        self._log_label = ctk.CTkLabel(
            main, text="Registro de actividad:", font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._log_label.grid(row=row, column=0, sticky="w", padx=10, pady=(5, 0))
        row += 1

        main.grid_rowconfigure(row, weight=1)
        self.log = LogConsole(main)
        self.log.grid(row=row, column=0, sticky="nsew", padx=10, pady=(2, 10))

    # ── i18n ────────────────────────────────────────────────────

    def _t(self, key: str) -> str:
        """Return translated string for the current UI language."""
        return _UI_STRINGS.get(self._ui_lang, _UI_STRINGS["EN"]).get(key, key)

    def _on_lang_change(self, *_: object) -> None:
        self._ui_lang = self.lang_var.get()
        self._apply_lang()

    def _apply_lang(self) -> None:
        """Update all translatable UI widgets to the current target language."""
        self._subtitle_label.configure(text=self._t("subtitle"))
        self.input_folder.update_label(self._t("label_input"))
        self.input_folder.update_buttons(
            folder=self._t("btn_folder"),
            file=self._t("btn_file"),
        )
        self.input_folder.update_placeholder(self._t("ph_input"))
        self.output_folder.update_label(self._t("label_output"))
        self.output_folder.update_buttons(browse=self._t("btn_browse"))
        self.output_folder.update_placeholder(self._t("ph_output"))
        self._game_label.configure(text=self._t("label_game"))
        self._lang_label.configure(text=self._t("label_lang"))
        self._backend_label.configure(text=self._t("label_backend"))
        self.translate_btn.configure(text=self._t("btn_translate"))
        self.cancel_btn.configure(text=self._t("btn_cancel"))
        self.settings_btn.configure(text=self._t("btn_settings"))
        self.restore_btn.configure(text=self._t("btn_restore"))
        self._cache_cb.configure(text=self._t("cb_cache"))
        self.progress_label.configure(text=self._t("progress_initial"))
        self._log_label.configure(text=self._t("log_label"))
        self._motor_tooltip.text = self._t("tooltip_backend")
        self._cache_tooltip.text = self._t("tooltip_cache")
        self._rebuild_backend_options()

    def _rebuild_backend_options(self) -> None:
        """Rebuild the backend dropdown with translated names, preserving selection."""
        current_value = self._backend_display_to_value.get(self.backend_var.get())

        new_options: dict[str, str] = {}
        if self._has_hybrid:
            new_options[self._t("backend_hybrid")] = "Hybrid"
        new_options[self._t("backend_opus")] = "Opus-MT"
        new_options[self._t("backend_deepl")] = "DeepL"

        self._backend_display_to_value = new_options
        self._backend_value_to_display = {v: k for k, v in new_options.items()}
        self._motor_menu.configure(values=list(new_options.keys()))

        if current_value and current_value in self._backend_value_to_display:
            self.backend_var.set(self._backend_value_to_display[current_value])
        else:
            self.backend_var.set(list(new_options.keys())[0])

    # ── GPU auto-detection on first launch ──

    def _detect_gpu_for_default(self) -> None:
        """Detectar GPU en background y actualizar el dropdown si no hay preferencia guardada."""
        cuda_info = detect_cuda()
        self.after(0, lambda: self._apply_gpu_default(cuda_info))

    def _apply_gpu_default(self, cuda_info: dict) -> None:
        """Filtrar opciones del dropdown según GPU y seleccionar el mejor backend."""
        self._has_hybrid = cuda_info["available"]
        best = "Hybrid" if self._has_hybrid else "Opus-MT"
        self._rebuild_backend_options()

        # Respetar preferencia guardada si sigue siendo válida para esta GPU;
        # si no (cambio de GPU o primer arranque) → seleccionar el mejor
        saved = self.settings.get("backend", "")
        if saved in self._backend_value_to_display:
            self.backend_var.set(self._backend_value_to_display[saved])
        else:
            best_display = self._backend_value_to_display.get(best)
            if best_display is None:
                best_display = list(self._backend_value_to_display.keys())[0]
            self.backend_var.set(best_display)

    # ── Input path change ──

    def _on_input_path_change(self, path: str) -> None:
        """Detectar si es archivo o carpeta y mostrar resumen."""
        p = Path(path)
        if p.is_file():
            suffix = p.suffix.lower()
            if suffix in _ESP_EXTENSIONS:
                self.scan_label.configure(
                    text=self._t("scan_esp").format(name=p.name),
                    text_color="#28a745",
                )
            elif suffix == _PEX_EXTENSION:
                self.scan_label.configure(
                    text=self._t("scan_pex").format(name=p.name),
                    text_color="#28a745",
                )
            else:
                self.scan_label.configure(
                    text=self._t("scan_unsupported").format(ext=p.suffix),
                    text_color="#dc3545",
                )
        elif p.is_dir():
            scan = scan_directory(p)
            parts = []
            if scan.esp_files:
                parts.append(self._t("scan_plugins").format(n=len(scan.esp_files)))
            if scan.pex_files:
                parts.append(self._t("scan_scripts").format(n=len(scan.pex_files)))
            if scan.has_mcm:
                parts.append(self._t("scan_mcm"))
            if parts:
                self.scan_label.configure(
                    text=self._t("scan_found").format(parts=", ".join(parts)),
                    text_color="#28a745",
                )
            else:
                self.scan_label.configure(
                    text=self._t("scan_nothing"),
                    text_color="#dc3545",
                )
        else:
            self.scan_label.configure(text=self._t("scan_invalid"), text_color="#dc3545")

    # ── Translation ──

    def _get_game_choice(self) -> GameChoice:
        return {
            "Auto": GameChoice.auto,
            "Fallout 3": GameChoice.fo3,
            "Fallout NV": GameChoice.fnv,
            "Fallout 4": GameChoice.fo4,
            "Skyrim": GameChoice.skyrim,
            "Minecraft": GameChoice.minecraft,
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
        input_path_str = self.input_folder.path
        if not input_path_str:
            messagebox.showerror("Error", "Selecciona la carpeta Data/ o un archivo de mod.")
            return False

        input_path = Path(input_path_str)
        if not input_path.exists():
            messagebox.showerror("Error", f"No existe: {input_path_str}")
            return False

        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix not in _ESP_EXTENSIONS and suffix != _PEX_EXTENSION:
                messagebox.showerror(
                    "Error",
                    f"Tipo de archivo no soportado: {input_path.suffix}\n"
                    "Usa archivos .esp, .esm, .esl o .pex.",
                )
                return False
        elif input_path.is_dir():
            scan = scan_directory(input_path)
            if not scan.esp_files and not scan.pex_files and not scan.has_mcm:
                messagebox.showwarning(
                    "Sin archivos",
                    "No se encontraron archivos traducibles (ESP/ESM, PEX, MCM) en la carpeta.",
                )
                return False
        else:
            messagebox.showerror("Error", f"Ruta no válida: {input_path_str}")
            return False

        backend_name = self._get_backend_name()
        if backend_name == "deepl" and not self.settings.get("deepl_api_key", "").strip():
            messagebox.showerror("Error", "Introduce la API key de DeepL en Ajustes.")
            return False

        ready, msg = check_backend_ready(backend_name, lang=self.lang_var.get())
        if not ready and "not downloaded" not in msg:
            messagebox.showerror("Backend no disponible", msg)
            return False

        output_path_str = self.output_folder.path
        if output_path_str:
            output_path = Path(output_path_str)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"No se pudo crear la carpeta de salida:\n{e}",
                    )
                    return False

        return True

    def _save_current_settings(self) -> None:
        self.settings["input_dir"] = self.input_folder.path
        self.settings["output_dir"] = self.output_folder.path
        self.settings["game"] = self.game_var.get()
        self.settings["lang"] = self.lang_var.get()
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
        self.progress_label.configure(text=self._t("msg_starting"))
        self.translate_btn.configure(state="disabled")
        self.cancel_btn.configure(text=self._t("btn_cancel"))

        backend_name = self._get_backend_name()
        api_key = self.settings.get("deepl_api_key", "").strip() or None

        input_path = Path(self.input_folder.path)
        output_path_str = self.output_folder.path

        # Resolver carpeta de salida
        if output_path_str:
            output_dir = Path(output_path_str)
            is_inplace = False
            self.log.append(f"Salida: {output_dir}")
        else:
            # In-place: la salida va a la misma carpeta que la entrada
            output_dir = input_path if input_path.is_dir() else input_path.parent
            is_inplace = True

        # Backup selectivo (solo para in-place, donde se sobreescriben los originales)
        if is_inplace:
            backup_name = input_path.stem if input_path.is_file() else input_path.name
            backup_dir = _SETTINGS_DIR / "backups" / backup_name
            try:
                if input_path.is_file():
                    files_to_backup = [input_path]
                    backup_base = input_path.parent
                else:
                    scan = scan_directory(input_path)
                    files_to_backup = scan.esp_files + scan.pex_files
                    backup_base = input_path

                if files_to_backup:
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    for f in files_to_backup:
                        rel = f.relative_to(backup_base)
                        dest = backup_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)
                    self.settings["last_backup"] = str(backup_dir)
                    self.settings["last_backup_source"] = str(backup_base)
                    _save_settings(self.settings)
                    self.log.append(
                        f"Backup de {len(files_to_backup)} archivos en: {backup_dir}"
                    )
                else:
                    self.log.append("Sin archivos ESP/ESM/PEX para respaldar")
            except Exception as e:
                self.log.append(f"Aviso: no se pudo crear backup: {e}")

        # Parámetros base (sin backend aún — se crea en background)
        base_kwargs = dict(
            lang=self.lang_var.get(),
            game=self._get_game_choice(),
            skip_translated=True,
            no_cache=not self.cache_var.get(),
        )

        # Si faltan modelos, descargar + crear backend todo en background
        missing_models = get_missing_model_ids(backend_name, lang=self.lang_var.get())
        if missing_models:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
            threading.Thread(
                target=self._auto_download_models,
                args=(missing_models, input_path, output_dir, base_kwargs,
                      backend_name, api_key),
                daemon=True,
            ).start()
            return

        # Crear backend (puede ser lento por imports de torch/ctranslate2)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.progress_label.configure(text=self._t("msg_starting"))
        threading.Thread(
            target=self._create_backend_and_launch,
            args=(backend_name, api_key, input_path, output_dir, base_kwargs),
            daemon=True,
        ).start()

    def _cancel(self) -> None:
        if self.worker.is_running:
            self.worker.cancel()
            self.log.append(self._t("msg_cancelling"))
            self.cancel_btn.configure(state="disabled")

    def _launch_worker(self, input_path: Path, output_dir: Path, common_kwargs: dict) -> None:
        """Enable cancel button and start the translation worker."""
        self.cancel_btn.configure(state="normal")
        if input_path.is_file():
            self.worker.start(
                _translate_single_file,
                file_path=input_path,
                output_dir=output_dir,
                **common_kwargs,
            )
        else:
            self.worker.start(
                batch_translate_all,
                directory=input_path,
                output_dir=output_dir,
                **common_kwargs,
            )

    def _create_backend_and_launch(
        self,
        backend_name: str,
        api_key: str | None,
        input_path: Path,
        output_dir: Path,
        base_kwargs: dict,
    ) -> None:
        """Create backend in background thread, then launch worker on main thread."""
        def log(msg: str) -> None:
            self.after(0, lambda m=msg: self.log.append(m))

        try:
            backend, backend_label = create_backend(backend_name, api_key=api_key)
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            self.after(0, lambda: self._on_backend_create_error(err_msg))
            return

        log(f"Backend: {backend_label}")
        common_kwargs = {**base_kwargs, "backend": backend, "backend_label": backend_label}
        self.after(0, lambda: self._stop_indeterminate_and_launch(
            input_path, output_dir, common_kwargs,
        ))

    def _on_backend_create_error(self, error: str) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        messagebox.showerror("Error", error)
        self._reset_buttons()

    def _auto_download_models(
        self,
        missing: list[tuple[str, str]],
        input_path: Path,
        output_dir: Path,
        base_kwargs: dict,
        backend_name: str,
        api_key: str | None,
    ) -> None:
        """Download missing models in background, then create backend and start worker."""
        def log(msg: str) -> None:
            self.after(0, lambda m=msg: self.log.append(m))

        total = len(missing)
        log(self._t("dl_start").format(n=total))

        for i, (name, model_id) in enumerate(missing, 1):
            log(self._t("dl_progress").format(i=i, n=total, name=name))
            msg = self._t("dl_progress_short").format(name=name)
            self.after(0, lambda m=msg: self.progress_label.configure(text=m))
            ok, err = download_model(model_id)
            if not ok:
                log(self._t("dl_error").format(name=name, err=err))
                self.after(0, lambda n=name: self._on_model_download_error(n))
                return
            log(self._t("dl_done").format(name=name))

        log(self._t("dl_all_done"))
        self._create_backend_and_launch(
            backend_name, api_key, input_path, output_dir, base_kwargs,
        )

    def _stop_indeterminate_and_launch(
        self, input_path: Path, output_dir: Path, common_kwargs: dict,
    ) -> None:
        """Stop indeterminate progress bar and launch translation worker."""
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        self._launch_worker(input_path, output_dir, common_kwargs)

    def _on_model_download_error(self, name: str) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        messagebox.showerror(
            self._t("msg_error"),
            self._t("dl_error_dialog").format(name=name),
        )
        self._reset_buttons()

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
        backed_up_files = [f for f in backup_path.rglob("*") if f.is_file()]

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
            phase_key = f"phase_{msg.phase}"
            phase_name = self._t(phase_key) if phase_key in _UI_STRINGS["EN"] else msg.phase
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
            self.log.append(self._t("completed_in").format(elapsed=elapsed))

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
                text=self._t("completed_files").format(n=result.total_success),
            )
            self._reset_buttons()

        elif msg.type == "error":
            self.progress_bar.set(0)
            self.progress_label.configure(text=self._t("msg_error"))
            self.log.append(f"\nERROR: {msg.message}")
            self._reset_buttons()

    # ── Settings dialog ──

    def _open_settings(self) -> None:
        SettingsWindow(self, self.settings, self._ui_lang)


# ═══════════════════════════════════════════════════════════════
# Settings window (popup)
# ═══════════════════════════════════════════════════════════════


class SettingsWindow(ctk.CTkToplevel):
    """Popup settings window."""

    def __init__(self, parent: ctk.CTk, settings: dict, ui_lang: str) -> None:
        super().__init__(parent)
        self.settings = settings
        self._ui_lang = ui_lang

        self.title(self._t("settings_title"))
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
            gpu_frame, text=self._t("settings_gpu_section"),
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self._gpu_label = ctk.CTkLabel(
            gpu_frame, text=self._t("settings_gpu_detecting"), text_color="gray",
        )
        self._gpu_label.pack(anchor="w", padx=10, pady=(0, 10))

        # ── Models ──
        self._models_frame = ctk.CTkFrame(self)
        self._models_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        self._models_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(
            self._models_frame, text=self._t("settings_models_section"),
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5))

        self._models_loading = ctk.CTkLabel(
            self._models_frame, text=self._t("settings_models_checking"), text_color="gray",
        )
        self._models_loading.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        # ── Cache ──
        cache_frame = ctk.CTkFrame(self)
        cache_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        row += 1

        ctk.CTkLabel(
            cache_frame, text=self._t("settings_cache_section"),
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self.cache_label = ctk.CTkLabel(
            cache_frame, text=self._t("settings_cache_checking"), text_color="gray",
        )
        self.cache_label.pack(anchor="w", padx=10)

        self._cache_path_label = ctk.CTkLabel(
            cache_frame, text="", text_color="gray",
        )
        self._cache_path_label.pack(anchor="w", padx=10, pady=(0, 5))

        ctk.CTkButton(
            cache_frame, text=self._t("settings_cache_clear_btn"), width=120,
            fg_color="#dc3545", hover_color="#c82333",
            command=self._clear_cache,
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # ── DeepL API key ──
        api_frame = ctk.CTkFrame(self)
        api_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        api_frame.grid_columnconfigure(0, weight=1)
        row += 1

        ctk.CTkLabel(
            api_frame, text=self._t("settings_deepl_section"),
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        self.api_key_var = ctk.StringVar(value=settings.get("deepl_api_key", ""))
        ctk.CTkEntry(
            api_frame, textvariable=self.api_key_var, show="*",
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        ctk.CTkButton(
            api_frame, text=self._t("settings_save_btn"), width=80,
            command=self._save_api_key,
        ).grid(row=1, column=1, padx=(5, 10), pady=(0, 5))

        # Load heavy data in background thread
        threading.Thread(target=self._load_settings_data, daemon=True).start()

    def _t(self, key: str) -> str:
        return _UI_STRINGS.get(self._ui_lang, _UI_STRINGS["EN"]).get(key, key)

    def _load_settings_data(self) -> None:
        """Load GPU, model, and cache info in background."""
        cuda_info = detect_cuda()
        models = get_model_status(lang=self._ui_lang)
        cache_info = get_cache_info()
        self.after(0, lambda: self._populate_settings(cuda_info, models, cache_info))

    def _populate_settings(
        self, cuda_info: dict, models: list, cache_info: dict,
    ) -> None:
        """Populate settings UI with loaded data (runs on main thread)."""
        # GPU
        if cuda_info["available"]:
            self._gpu_label.configure(
                text=self._t("settings_gpu_nvidia").format(name=cuda_info["gpu_name"]),
                text_color="#28a745",
            )
        else:
            self._gpu_label.configure(
                text=self._t("settings_gpu_none"),
                text_color="#ffc107",
            )

        # Models
        self._models_loading.destroy()
        for i, model in enumerate(models):
            status = (
                self._t("settings_model_downloaded")
                if model.is_downloaded
                else self._t("settings_model_not_downloaded")
            )
            color = "#28a745" if model.is_downloaded else "#dc3545"

            ctk.CTkLabel(
                self._models_frame, text=f"{model.name} ({model.size_hint})",
            ).grid(row=i + 1, column=0, sticky="w", padx=10, pady=2)

            ctk.CTkLabel(
                self._models_frame, text=status, text_color=color,
            ).grid(row=i + 1, column=1, sticky="w", padx=5, pady=2)

            if not model.is_downloaded:
                ctk.CTkButton(
                    self._models_frame, text=self._t("settings_model_download_btn"), width=80,
                    command=lambda m=model: self._download(m),
                ).grid(row=i + 1, column=2, padx=10, pady=2)

        # Cache
        self.cache_label.configure(
            text=self._t("settings_cache_count").format(n=cache_info["count"]),
            text_color=("white", "white"),
        )
        self._cache_path_label.configure(
            text=self._t("settings_cache_path").format(path=cache_info["path"]),
        )

    def _download(self, model: object) -> None:
        from modtranslator.gui.model_manager import ModelInfo
        assert isinstance(model, ModelInfo)

        def _do_download() -> None:
            success, err = download_model(model.description)
            if not success:
                self.after(0, lambda: self._on_download_done(model.name, False, err))
            else:
                self.after(0, lambda: self._on_download_done(model.name, True, ""))


        self._download_status = ctk.CTkLabel(
            self, text=self._t("settings_model_downloading").format(name=model.name),
            text_color="#ffc107",
        )
        self._download_status.grid(row=10, column=0, padx=10, pady=5)
        threading.Thread(target=_do_download, daemon=True).start()

    def _on_download_done(self, name: str, success: bool, error: str = "") -> None:
        if hasattr(self, "_download_status"):
            self._download_status.destroy()
        if success:
            messagebox.showinfo(
                self._t("msg_success"),
                self._t("settings_download_ok").format(name=name),
            )
        else:
            detail = f"\n\n{error}" if error else ""
            messagebox.showerror(
                self._t("msg_error"),
                self._t("settings_download_err").format(name=name) + detail,
            )

    def _clear_cache(self) -> None:
        deleted = clear_cache()
        self.cache_label.configure(text=self._t("settings_cache_count").format(n=0))
        messagebox.showinfo("Cache", self._t("settings_cache_cleared").format(n=deleted))

    def _save_api_key(self) -> None:
        self.settings["deepl_api_key"] = self.api_key_var.get().strip()
        _save_settings(self.settings)
        messagebox.showinfo(self._t("msg_success"), self._t("settings_api_saved"))


# ═══════════════════════════════════════════════════════════════
# First-run setup
# ═══════════════════════════════════════════════════════════════

def _has_any_ml_backend() -> bool:
    """Check instantly (no import side effects) if any ML backend is installed."""
    import importlib.util
    return (
        importlib.util.find_spec("ctranslate2") is not None
        or importlib.util.find_spec("torch") is not None
    )


def _detect_gpu_type() -> tuple[str, str]:
    """Returns (type, name): type is 'nvidia' or 'cpu'."""
    try:
        r = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name", "/value"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.startswith("Name="):
                    name = line.split("=", 1)[1].strip()
                    if not name:
                        continue
                    if any(kw in name.upper() for kw in ("NVIDIA", "GEFORCE", "QUADRO", "TESLA")):
                        return "nvidia", name
    except Exception:
        pass
    # Fallback: nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "nvidia", r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "cpu", "CPU"


_SETUP_STEPS: dict[str, list[tuple[str, list[str]]]] = {
    "nvidia": [
        ("Instalando PyTorch CUDA (NVIDIA)...",
         ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"]),
        ("Instalando motor de traduccion...",
         ["ctranslate2", "transformers", "sentencepiece"]),
    ],
    "cpu": [
        ("Instalando PyTorch CPU...",
         ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]),
        ("Instalando motor de traduccion...",
         ["ctranslate2", "transformers", "sentencepiece"]),
    ],
}


def run_gui() -> None:
    """Launch the ModTranslator GUI."""
    app = ModTranslatorApp()
    app.mainloop()
