"""ModTranslator GUI - Main application window.

Single-view interface: select Data/ folder or single mod file, click translate.
Auto-detects ESP/ESM, PEX, and MCM files and translates everything.
Backs up only the files that will be modified before translating.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from modtranslator import __version__
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
    batch_translate_esp,
    batch_translate_pex,
    clear_cache,
    create_backend,
    get_cache_info,
    scan_directory,
)

# Settings persistence
_SETTINGS_DIR = Path.home() / ".modtranslator"
_SETTINGS_FILE = _SETTINGS_DIR / "gui_settings.json"

_ESP_EXTENSIONS = {".esp", ".esm", ".esl"}
_PEX_EXTENSION = ".pex"

# ── UI translations ─────────────────────────────────────────────────────────

_UI_STRINGS: dict[str, dict[str, str]] = {
    "EN": {
        "subtitle": "Mod Translator",
        "label_input": "Input:",
        "label_output": "Output folder:",
        "label_game": "Game:",
        "label_lang": "Language:",
        "label_backend": "Translation engine:",
        "btn_folder": "Folder",
        "btn_file": "File",
        "btn_browse": "Browse",
        "btn_translate": "Translate",
        "btn_cancel": "Cancel",
        "btn_settings": "Settings",
        "btn_restore": "Restore backup",
        "cb_cache": "Remember previous translations",
        "progress_initial": "Select input and click Translate to start",
        "log_label": "Activity log:",
        "ph_input": "Data/ folder or .esp/.esm/.pex file...",
        "ph_output": "(same as input — in-place)",
        "phase_scan": "Scanning",
        "phase_prepare": "Phase 1: Preparing",
        "phase_translate": "Phase 2: Translating",
        "phase_write": "Phase 3: Writing",
        "msg_starting": "Starting...",
        "msg_cancelling": "Cancelling...",
        "msg_error": "Error",
        "backend_hybrid": "Hybrid (Recommended with Nvidia GPU)",
        "backend_opus": "Opus-MT (no GPU)",
        "backend_deepl": "DeepL (needs API key)",
        "tooltip_backend": (
            "Hybrid: Opus-MT + NLLB via CTranslate2 (Nvidia GPU/CUDA).\n"
            "Opus-MT: CTranslate2 model, works well without GPU.\n"
            "DeepL: high-quality online service (needs API key)."
        ),
        "tooltip_cache": (
            "Saves translations to disk to avoid\n"
            "retranslating the same text in future runs.\n"
            "Saves time if you translate the same mods again."
        ),
        "scan_esp": "File: {name} (ESP/ESM)",
        "scan_pex": "File: {name} (PEX)",
        "scan_unsupported": "Unsupported type: {ext}",
        "scan_found": "Found: {parts}",
        "scan_nothing": "No translatable files found",
        "scan_invalid": "Invalid path",
        "scan_plugins": "{n} plugins (ESP/ESM)",
        "scan_scripts": "{n} scripts (PEX)",
        "scan_mcm": "MCM files",
        "completed_files": "Completed - {n} files translated",
        "completed_in": "Completed in {elapsed}",
        "msg_success": "Success",
        "settings_title": "Settings",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "Detecting GPU...",
        "settings_models_section": "ML Models",
        "settings_models_checking": "Checking models...",
        "settings_cache_section": "Translation cache",
        "settings_cache_checking": "Checking cache...",
        "settings_cache_clear_btn": "Clear cache",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Save",
        "settings_gpu_nvidia": "NVIDIA GPU detected: {name} — use Hybrid backend",
        "settings_gpu_none": "No GPU detected (CPU will be used)",
        "settings_model_downloaded": "Downloaded",
        "settings_model_not_downloaded": "Not downloaded",
        "settings_model_download_btn": "Download",
        "settings_model_downloading": "Downloading {name}...",
        "settings_download_ok": "{name} downloaded successfully.",
        "settings_download_err": "Error downloading {name}.",
        "settings_cache_count": "Cached translations: {n}",
        "settings_cache_path": "Location: {path}",
        "settings_cache_cleared": "{n} translations removed from cache.",
        "settings_api_saved": "API key saved.",
    },
    "ES": {
        "subtitle": "Traductor de mods",
        "label_input": "Entrada:",
        "label_output": "Carpeta de salida:",
        "label_game": "Juego:",
        "label_lang": "Idioma:",
        "label_backend": "Motor de traducción:",
        "btn_folder": "Carpeta",
        "btn_file": "Archivo",
        "btn_browse": "Examinar",
        "btn_translate": "Traducir",
        "btn_cancel": "Cancelar",
        "btn_settings": "Ajustes",
        "btn_restore": "Restaurar backup",
        "cb_cache": "Recordar traducciones anteriores",
        "progress_initial": "Selecciona la entrada y pulsa Traducir para empezar",
        "log_label": "Registro de actividad:",
        "ph_input": "Carpeta Data/ o archivo .esp/.esm/.pex...",
        "ph_output": "(misma que la entrada — in-place)",
        "phase_scan": "Escaneando",
        "phase_prepare": "Fase 1: Preparando",
        "phase_translate": "Fase 2: Traduciendo",
        "phase_write": "Fase 3: Escribiendo",
        "msg_starting": "Iniciando...",
        "msg_cancelling": "Cancelando...",
        "msg_error": "Error",
        "backend_hybrid": "Híbrido (Recomendado con GPU Nvidia)",
        "backend_opus": "Opus-MT (sin GPU)",
        "backend_deepl": "DeepL (necesita API key)",
        "tooltip_backend": (
            "Híbrido: Opus-MT + NLLB via CTranslate2 (GPU Nvidia/CUDA).\n"
            "Opus-MT: un modelo CTranslate2, funciona bien sin GPU.\n"
            "DeepL: servicio online de alta calidad (necesita API key)."
        ),
        "tooltip_cache": (
            "Guarda las traducciones en disco para no volver\n"
            "a traducir los mismos textos en futuras ejecuciones.\n"
            "Ahorra tiempo si traduces los mismos mods otra vez."
        ),
        "scan_esp": "Archivo: {name} (ESP/ESM)",
        "scan_pex": "Archivo: {name} (PEX)",
        "scan_unsupported": "Tipo no soportado: {ext}",
        "scan_found": "Encontrados: {parts}",
        "scan_nothing": "No se encontraron archivos traducibles",
        "scan_invalid": "Ruta no válida",
        "scan_plugins": "{n} plugins (ESP/ESM)",
        "scan_scripts": "{n} scripts (PEX)",
        "scan_mcm": "archivos MCM",
        "completed_files": "Completado - {n} archivos traducidos",
        "completed_in": "Completado en {elapsed}",
        "msg_success": "Éxito",
        "settings_title": "Ajustes",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "Detectando GPU...",
        "settings_models_section": "Modelos ML",
        "settings_models_checking": "Comprobando modelos...",
        "settings_cache_section": "Cache de traducciones",
        "settings_cache_checking": "Comprobando cache...",
        "settings_cache_clear_btn": "Limpiar cache",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Guardar",
        "settings_gpu_nvidia": "GPU NVIDIA detectada: {name} — usar backend Híbrido",
        "settings_gpu_none": "No se detecta GPU (se usará CPU)",
        "settings_model_downloaded": "Descargado",
        "settings_model_not_downloaded": "No descargado",
        "settings_model_download_btn": "Descargar",
        "settings_model_downloading": "Descargando {name}...",
        "settings_download_ok": "{name} descargado correctamente.",
        "settings_download_err": "Error descargando {name}.",
        "settings_cache_count": "Traducciones en cache: {n}",
        "settings_cache_path": "Ubicación: {path}",
        "settings_cache_cleared": "Se eliminaron {n} traducciones del cache.",
        "settings_api_saved": "API key guardada.",
    },
    "FR": {
        "subtitle": "Traducteur de mods",
        "label_input": "Entrée :",
        "label_output": "Dossier de sortie :",
        "label_game": "Jeu :",
        "label_lang": "Langue :",
        "label_backend": "Moteur de traduction :",
        "btn_folder": "Dossier",
        "btn_file": "Fichier",
        "btn_browse": "Parcourir",
        "btn_translate": "Traduire",
        "btn_cancel": "Annuler",
        "btn_settings": "Paramètres",
        "btn_restore": "Restaurer sauvegarde",
        "cb_cache": "Mémoriser les traductions précédentes",
        "progress_initial": "Sélectionne l'entrée et clique sur Traduire pour commencer",
        "log_label": "Journal d'activité :",
        "ph_input": "Dossier Data/ ou fichier .esp/.esm/.pex...",
        "ph_output": "(même que l'entrée — en place)",
        "phase_scan": "Analyse en cours",
        "phase_prepare": "Phase 1 : Préparation",
        "phase_translate": "Phase 2 : Traduction",
        "phase_write": "Phase 3 : Écriture",
        "msg_starting": "Démarrage...",
        "msg_cancelling": "Annulation...",
        "msg_error": "Erreur",
        "backend_hybrid": "Hybride (Recommandé avec GPU Nvidia)",
        "backend_opus": "Opus-MT (sans GPU)",
        "backend_deepl": "DeepL (nécessite une clé API)",
        "tooltip_backend": (
            "Hybride : Opus-MT + NLLB via CTranslate2 (GPU Nvidia/CUDA).\n"
            "Opus-MT : modèle CTranslate2, fonctionne sans GPU.\n"
            "DeepL : service en ligne haute qualité (clé API requise)."
        ),
        "tooltip_cache": (
            "Enregistre les traductions sur disque pour ne pas\n"
            "retraduire les mêmes textes lors des prochaines exécutions.\n"
            "Gain de temps si vous traduisez les mêmes mods à nouveau."
        ),
        "scan_esp": "Fichier : {name} (ESP/ESM)",
        "scan_pex": "Fichier : {name} (PEX)",
        "scan_unsupported": "Type non supporté : {ext}",
        "scan_found": "Trouvé : {parts}",
        "scan_nothing": "Aucun fichier traduisible trouvé",
        "scan_invalid": "Chemin invalide",
        "scan_plugins": "{n} plugins (ESP/ESM)",
        "scan_scripts": "{n} scripts (PEX)",
        "scan_mcm": "fichiers MCM",
        "completed_files": "Terminé - {n} fichiers traduits",
        "completed_in": "Terminé en {elapsed}",
        "msg_success": "Succès",
        "settings_title": "Paramètres",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "Détection du GPU...",
        "settings_models_section": "Modèles ML",
        "settings_models_checking": "Vérification des modèles...",
        "settings_cache_section": "Cache de traductions",
        "settings_cache_checking": "Vérification du cache...",
        "settings_cache_clear_btn": "Vider le cache",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Enregistrer",
        "settings_gpu_nvidia": "GPU NVIDIA détecté : {name} — utiliser le backend Hybride",
        "settings_gpu_none": "Pas de GPU détecté (utilisation du CPU)",
        "settings_model_downloaded": "Téléchargé",
        "settings_model_not_downloaded": "Non téléchargé",
        "settings_model_download_btn": "Télécharger",
        "settings_model_downloading": "Téléchargement de {name}...",
        "settings_download_ok": "{name} téléchargé avec succès.",
        "settings_download_err": "Erreur lors du téléchargement de {name}.",
        "settings_cache_count": "Traductions en cache : {n}",
        "settings_cache_path": "Emplacement : {path}",
        "settings_cache_cleared": "{n} traductions supprimées du cache.",
        "settings_api_saved": "Clé API enregistrée.",
    },
    "DE": {
        "subtitle": "Mod-Übersetzer",
        "label_input": "Eingabe:",
        "label_output": "Ausgabeordner:",
        "label_game": "Spiel:",
        "label_lang": "Sprache:",
        "label_backend": "Übersetzungsmotor:",
        "btn_folder": "Ordner",
        "btn_file": "Datei",
        "btn_browse": "Durchsuchen",
        "btn_translate": "Übersetzen",
        "btn_cancel": "Abbrechen",
        "btn_settings": "Einstellungen",
        "btn_restore": "Backup wiederherstellen",
        "cb_cache": "Frühere Übersetzungen speichern",
        "progress_initial": "Eingabe auswählen und auf Übersetzen klicken",
        "log_label": "Aktivitätsprotokoll:",
        "ph_input": "Data/-Ordner oder .esp/.esm/.pex-Datei...",
        "ph_output": "(gleicher Ordner wie Eingabe — in-place)",
        "phase_scan": "Scannen",
        "phase_prepare": "Phase 1: Vorbereitung",
        "phase_translate": "Phase 2: Übersetzung",
        "phase_write": "Phase 3: Schreiben",
        "msg_starting": "Starten...",
        "msg_cancelling": "Abbrechen...",
        "msg_error": "Fehler",
        "backend_hybrid": "Hybrid (Empfohlen mit Nvidia GPU)",
        "backend_opus": "Opus-MT (ohne GPU)",
        "backend_deepl": "DeepL (braucht API-Schlüssel)",
        "tooltip_backend": (
            "Hybrid: Opus-MT + NLLB via CTranslate2 (Nvidia GPU/CUDA).\n"
            "Opus-MT: CTranslate2-Modell, funktioniert ohne GPU.\n"
            "DeepL: Online-Dienst hoher Qualität (API-Schlüssel erforderlich)."
        ),
        "tooltip_cache": (
            "Speichert Übersetzungen auf Disk, damit dieselben Texte\n"
            "bei zukünftigen Ausführungen nicht erneut übersetzt werden.\n"
            "Spart Zeit beim erneuten Übersetzen derselben Mods."
        ),
        "scan_esp": "Datei: {name} (ESP/ESM)",
        "scan_pex": "Datei: {name} (PEX)",
        "scan_unsupported": "Nicht unterstützter Typ: {ext}",
        "scan_found": "Gefunden: {parts}",
        "scan_nothing": "Keine übersetzbaren Dateien gefunden",
        "scan_invalid": "Ungültiger Pfad",
        "scan_plugins": "{n} Plugins (ESP/ESM)",
        "scan_scripts": "{n} Skripte (PEX)",
        "scan_mcm": "MCM-Dateien",
        "completed_files": "Abgeschlossen - {n} Dateien übersetzt",
        "completed_in": "Abgeschlossen in {elapsed}",
        "msg_success": "Erfolg",
        "settings_title": "Einstellungen",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "GPU wird erkannt...",
        "settings_models_section": "ML-Modelle",
        "settings_models_checking": "Modelle werden geprüft...",
        "settings_cache_section": "Übersetzungs-Cache",
        "settings_cache_checking": "Cache wird geprüft...",
        "settings_cache_clear_btn": "Cache leeren",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Speichern",
        "settings_gpu_nvidia": "NVIDIA-GPU erkannt: {name} — Hybrid-Backend verwenden",
        "settings_gpu_none": "Keine GPU erkannt (CPU wird verwendet)",
        "settings_model_downloaded": "Heruntergeladen",
        "settings_model_not_downloaded": "Nicht heruntergeladen",
        "settings_model_download_btn": "Herunterladen",
        "settings_model_downloading": "{name} wird heruntergeladen...",
        "settings_download_ok": "{name} erfolgreich heruntergeladen.",
        "settings_download_err": "Fehler beim Herunterladen von {name}.",
        "settings_cache_count": "Gecachte Übersetzungen: {n}",
        "settings_cache_path": "Speicherort: {path}",
        "settings_cache_cleared": "{n} Übersetzungen aus dem Cache gelöscht.",
        "settings_api_saved": "API-Schlüssel gespeichert.",
    },
    "IT": {
        "subtitle": "Traduttore di mod",
        "label_input": "Ingresso:",
        "label_output": "Cartella di uscita:",
        "label_game": "Gioco:",
        "label_lang": "Lingua:",
        "label_backend": "Motore di traduzione:",
        "btn_folder": "Cartella",
        "btn_file": "File",
        "btn_browse": "Sfoglia",
        "btn_translate": "Traduci",
        "btn_cancel": "Annulla",
        "btn_settings": "Impostazioni",
        "btn_restore": "Ripristina backup",
        "cb_cache": "Ricorda le traduzioni precedenti",
        "progress_initial": "Seleziona l'ingresso e clicca Traduci per iniziare",
        "log_label": "Registro attività:",
        "ph_input": "Cartella Data/ o file .esp/.esm/.pex...",
        "ph_output": "(stessa dell'ingresso — in-place)",
        "phase_scan": "Scansione",
        "phase_prepare": "Fase 1: Preparazione",
        "phase_translate": "Fase 2: Traduzione",
        "phase_write": "Fase 3: Scrittura",
        "msg_starting": "Avvio...",
        "msg_cancelling": "Annullamento...",
        "msg_error": "Errore",
        "backend_hybrid": "Ibrido (Consigliato con GPU Nvidia)",
        "backend_opus": "Opus-MT (senza GPU)",
        "backend_deepl": "DeepL (richiede chiave API)",
        "tooltip_backend": (
            "Ibrido: Opus-MT + NLLB via CTranslate2 (GPU Nvidia/CUDA).\n"
            "Opus-MT: modello CTranslate2, funziona senza GPU.\n"
            "DeepL: servizio online di alta qualità (richiede chiave API)."
        ),
        "tooltip_cache": (
            "Salva le traduzioni su disco per non ritradurre\n"
            "gli stessi testi nelle future esecuzioni.\n"
            "Risparmia tempo se traduci gli stessi mod di nuovo."
        ),
        "scan_esp": "File: {name} (ESP/ESM)",
        "scan_pex": "File: {name} (PEX)",
        "scan_unsupported": "Tipo non supportato: {ext}",
        "scan_found": "Trovati: {parts}",
        "scan_nothing": "Nessun file traducibile trovato",
        "scan_invalid": "Percorso non valido",
        "scan_plugins": "{n} plugin (ESP/ESM)",
        "scan_scripts": "{n} script (PEX)",
        "scan_mcm": "file MCM",
        "completed_files": "Completato - {n} file tradotti",
        "completed_in": "Completato in {elapsed}",
        "msg_success": "Successo",
        "settings_title": "Impostazioni",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "Rilevamento GPU...",
        "settings_models_section": "Modelli ML",
        "settings_models_checking": "Controllo modelli...",
        "settings_cache_section": "Cache traduzioni",
        "settings_cache_checking": "Controllo cache...",
        "settings_cache_clear_btn": "Svuota cache",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Salva",
        "settings_gpu_nvidia": "GPU NVIDIA rilevata: {name} — usare il backend Ibrido",
        "settings_gpu_none": "Nessuna GPU rilevata (verrà usata la CPU)",
        "settings_model_downloaded": "Scaricato",
        "settings_model_not_downloaded": "Non scaricato",
        "settings_model_download_btn": "Scarica",
        "settings_model_downloading": "Download di {name}...",
        "settings_download_ok": "{name} scaricato correttamente.",
        "settings_download_err": "Errore nel download di {name}.",
        "settings_cache_count": "Traduzioni in cache: {n}",
        "settings_cache_path": "Posizione: {path}",
        "settings_cache_cleared": "{n} traduzioni rimosse dalla cache.",
        "settings_api_saved": "Chiave API salvata.",
    },
    "PT": {
        "subtitle": "Tradutor de mods",
        "label_input": "Entrada:",
        "label_output": "Pasta de saída:",
        "label_game": "Jogo:",
        "label_lang": "Idioma:",
        "label_backend": "Motor de tradução:",
        "btn_folder": "Pasta",
        "btn_file": "Arquivo",
        "btn_browse": "Procurar",
        "btn_translate": "Traduzir",
        "btn_cancel": "Cancelar",
        "btn_settings": "Configurações",
        "btn_restore": "Restaurar backup",
        "cb_cache": "Lembrar traduções anteriores",
        "progress_initial": "Selecione a entrada e clique em Traduzir para começar",
        "log_label": "Registro de atividade:",
        "ph_input": "Pasta Data/ ou arquivo .esp/.esm/.pex...",
        "ph_output": "(mesma que a entrada — in-place)",
        "phase_scan": "Varrendo",
        "phase_prepare": "Fase 1: Preparando",
        "phase_translate": "Fase 2: Traduzindo",
        "phase_write": "Fase 3: Escrevendo",
        "msg_starting": "Iniciando...",
        "msg_cancelling": "Cancelando...",
        "msg_error": "Erro",
        "backend_hybrid": "Híbrido (Recomendado com GPU Nvidia)",
        "backend_opus": "Opus-MT (sem GPU)",
        "backend_deepl": "DeepL (precisa de chave API)",
        "tooltip_backend": (
            "Híbrido: Opus-MT + NLLB via CTranslate2 (GPU Nvidia/CUDA).\n"
            "Opus-MT: modelo CTranslate2, funciona sem GPU.\n"
            "DeepL: serviço online de alta qualidade (requer chave API)."
        ),
        "tooltip_cache": (
            "Salva traduções em disco para não traduzir\n"
            "os mesmos textos em execuções futuras.\n"
            "Economiza tempo ao traduzir os mesmos mods novamente."
        ),
        "scan_esp": "Arquivo: {name} (ESP/ESM)",
        "scan_pex": "Arquivo: {name} (PEX)",
        "scan_unsupported": "Tipo não suportado: {ext}",
        "scan_found": "Encontrados: {parts}",
        "scan_nothing": "Nenhum arquivo traduzível encontrado",
        "scan_invalid": "Caminho inválido",
        "scan_plugins": "{n} plugins (ESP/ESM)",
        "scan_scripts": "{n} scripts (PEX)",
        "scan_mcm": "arquivos MCM",
        "completed_files": "Concluído - {n} arquivos traduzidos",
        "completed_in": "Concluído em {elapsed}",
        "msg_success": "Êxito",
        "settings_title": "Configurações",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "Detectando GPU...",
        "settings_models_section": "Modelos ML",
        "settings_models_checking": "Verificando modelos...",
        "settings_cache_section": "Cache de traduções",
        "settings_cache_checking": "Verificando cache...",
        "settings_cache_clear_btn": "Limpar cache",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Salvar",
        "settings_gpu_nvidia": "GPU NVIDIA detectada: {name} — usar backend Híbrido",
        "settings_gpu_none": "Nenhuma GPU detectada (será usada a CPU)",
        "settings_model_downloaded": "Baixado",
        "settings_model_not_downloaded": "Não baixado",
        "settings_model_download_btn": "Baixar",
        "settings_model_downloading": "Baixando {name}...",
        "settings_download_ok": "{name} baixado com sucesso.",
        "settings_download_err": "Erro ao baixar {name}.",
        "settings_cache_count": "Traduções em cache: {n}",
        "settings_cache_path": "Localização: {path}",
        "settings_cache_cleared": "{n} traduções removidas do cache.",
        "settings_api_saved": "Chave API salva.",
    },
    "RU": {
        "subtitle": "Переводчик модов",
        "label_input": "Вход:",
        "label_output": "Папка вывода:",
        "label_game": "Игра:",
        "label_lang": "Язык:",
        "label_backend": "Движок перевода:",
        "btn_folder": "Папка",
        "btn_file": "Файл",
        "btn_browse": "Обзор",
        "btn_translate": "Перевести",
        "btn_cancel": "Отмена",
        "btn_settings": "Настройки",
        "btn_restore": "Восстановить копию",
        "cb_cache": "Запомнить переводы",
        "progress_initial": "Выберите источник и нажмите Перевести",
        "log_label": "Журнал активности:",
        "ph_input": "Папка Data/ или файл .esp/.esm/.pex...",
        "ph_output": "(та же, что и вход)",
        "phase_scan": "Сканирование",
        "phase_prepare": "Фаза 1: Подготовка",
        "phase_translate": "Фаза 2: Перевод",
        "phase_write": "Фаза 3: Запись",
        "msg_starting": "Запуск...",
        "msg_cancelling": "Отмена...",
        "msg_error": "Ошибка",
        "backend_hybrid": "Гибрид (рек. с GPU Nvidia)",
        "backend_opus": "Opus-MT (без GPU)",
        "backend_deepl": "DeepL (нужен API ключ)",
        "tooltip_backend": (
            "Гибрид: Opus-MT + NLLB через CTranslate2 (GPU Nvidia/CUDA).\n"
            "Opus-MT: модель CTranslate2, работает без GPU.\n"
            "DeepL: онлайн-сервис высокого качества (нужен API ключ)."
        ),
        "tooltip_cache": (
            "Сохраняет переводы на диске, чтобы не переводить\n"
            "те же тексты при следующих запусках.\n"
            "Экономит время при повторном переводе тех же модов."
        ),
        "scan_esp": "Файл: {name} (ESP/ESM)",
        "scan_pex": "Файл: {name} (PEX)",
        "scan_unsupported": "Тип не поддерживается: {ext}",
        "scan_found": "Найдено: {parts}",
        "scan_nothing": "Переводимые файлы не найдены",
        "scan_invalid": "Неверный путь",
        "scan_plugins": "{n} плагинов (ESP/ESM)",
        "scan_scripts": "{n} скриптов (PEX)",
        "scan_mcm": "файлы MCM",
        "completed_files": "Завершено — {n} файлов переведено",
        "completed_in": "Завершено за {elapsed}",
        "msg_success": "Успех",
        "settings_title": "Настройки",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "Определение GPU...",
        "settings_models_section": "Модели ML",
        "settings_models_checking": "Проверка моделей...",
        "settings_cache_section": "Кэш переводов",
        "settings_cache_checking": "Проверка кэша...",
        "settings_cache_clear_btn": "Очистить кэш",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Сохранить",
        "settings_gpu_nvidia": "Обнаружена GPU NVIDIA: {name} — использовать гибридный бэкенд",
        "settings_gpu_none": "GPU не обнаружена (будет использоваться CPU)",
        "settings_model_downloaded": "Загружено",
        "settings_model_not_downloaded": "Не загружено",
        "settings_model_download_btn": "Загрузить",
        "settings_model_downloading": "Загрузка {name}...",
        "settings_download_ok": "{name} успешно загружено.",
        "settings_download_err": "Ошибка загрузки {name}.",
        "settings_cache_count": "Переводов в кэше: {n}",
        "settings_cache_path": "Расположение: {path}",
        "settings_cache_cleared": "{n} переводов удалено из кэша.",
        "settings_api_saved": "API-ключ сохранён.",
    },
    "PL": {
        "subtitle": "Tłumacz modów",
        "label_input": "Wejście:",
        "label_output": "Folder wyjściowy:",
        "label_game": "Gra:",
        "label_lang": "Język:",
        "label_backend": "Silnik tłumaczenia:",
        "btn_folder": "Folder",
        "btn_file": "Plik",
        "btn_browse": "Przeglądaj",
        "btn_translate": "Tłumacz",
        "btn_cancel": "Anuluj",
        "btn_settings": "Ustawienia",
        "btn_restore": "Przywróć kopię",
        "cb_cache": "Pamiętaj tłumaczenia",
        "progress_initial": "Wybierz wejście i kliknij Tłumacz, aby rozpocząć",
        "log_label": "Dziennik aktywności:",
        "ph_input": "Folder Data/ lub plik .esp/.esm/.pex...",
        "ph_output": "(ten sam co wejście — w miejscu)",
        "phase_scan": "Skanowanie",
        "phase_prepare": "Faza 1: Przygotowanie",
        "phase_translate": "Faza 2: Tłumaczenie",
        "phase_write": "Faza 3: Zapis",
        "msg_starting": "Uruchamianie...",
        "msg_cancelling": "Anulowanie...",
        "msg_error": "Błąd",
        "backend_hybrid": "Hybrydowy (z GPU Nvidia)",
        "backend_opus": "Opus-MT (bez GPU)",
        "backend_deepl": "DeepL (wymaga klucza API)",
        "tooltip_backend": (
            "Hybrydowy: Opus-MT + NLLB przez CTranslate2 (GPU Nvidia/CUDA).\n"
            "Opus-MT: model CTranslate2, działa bez GPU.\n"
            "DeepL: usługa online wysokiej jakości (wymagany klucz API)."
        ),
        "tooltip_cache": (
            "Zapisuje tłumaczenia na dysk, aby nie tłumaczyć\n"
            "tych samych tekstów w przyszłych uruchomieniach.\n"
            "Oszczędza czas przy ponownym tłumaczeniu tych samych modów."
        ),
        "scan_esp": "Plik: {name} (ESP/ESM)",
        "scan_pex": "Plik: {name} (PEX)",
        "scan_unsupported": "Nieobsługiwany typ: {ext}",
        "scan_found": "Znaleziono: {parts}",
        "scan_nothing": "Nie znaleziono plików do tłumaczenia",
        "scan_invalid": "Nieprawidłowa ścieżka",
        "scan_plugins": "{n} pluginów (ESP/ESM)",
        "scan_scripts": "{n} skryptów (PEX)",
        "scan_mcm": "pliki MCM",
        "completed_files": "Ukończono — {n} pliki przetłumaczone",
        "completed_in": "Ukończono w {elapsed}",
        "msg_success": "Sukces",
        "settings_title": "Ustawienia",
        "settings_gpu_section": "GPU / CUDA",
        "settings_gpu_detecting": "Wykrywanie GPU...",
        "settings_models_section": "Modele ML",
        "settings_models_checking": "Sprawdzanie modeli...",
        "settings_cache_section": "Pamięć podręczna",
        "settings_cache_checking": "Sprawdzanie cache...",
        "settings_cache_clear_btn": "Wyczyść cache",
        "settings_deepl_section": "DeepL API Key",
        "settings_save_btn": "Zapisz",
        "settings_gpu_nvidia": "Wykryto GPU NVIDIA: {name} — użyj backendu hybrydowego",
        "settings_gpu_none": "Nie wykryto GPU (zostanie użyty CPU)",
        "settings_model_downloaded": "Pobrano",
        "settings_model_not_downloaded": "Nie pobrano",
        "settings_model_download_btn": "Pobierz",
        "settings_model_downloading": "Pobieranie {name}...",
        "settings_download_ok": "{name} pobrano pomyślnie.",
        "settings_download_err": "Błąd pobierania {name}.",
        "settings_cache_count": "Tłumaczenia w cache: {n}",
        "settings_cache_path": "Lokalizacja: {path}",
        "settings_cache_cleared": "{n} tłumaczeń usuniętych z cache.",
        "settings_api_saved": "Klucz API zapisany.",
    },
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
            values=["Auto", "Fallout 3", "Fallout NV", "Fallout 4", "Skyrim"],
            width=120,
        ).pack(side="left", padx=(0, 15))

        self._lang_label = ctk.CTkLabel(opts, text="Idioma:")
        self._lang_label.pack(side="left", padx=(0, 5))
        self.lang_var = ctk.StringVar(value=self.settings.get("lang", "ES"))
        ctk.CTkOptionMenu(
            opts, variable=self.lang_var,
            values=["ES", "FR", "DE", "IT", "PT", "RU", "PL"],
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
        if not ready and "descargado" not in msg:
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

        try:
            backend, backend_label = create_backend(backend_name, api_key=api_key)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self._reset_buttons()
            return

        self.log.append(f"Backend: {backend_label}")

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

        # Lanzar traducción
        common_kwargs = dict(
            lang=self.lang_var.get(),
            backend=backend,
            backend_label=backend_label,
            game=self._get_game_choice(),
            skip_translated=True,
            no_cache=not self.cache_var.get(),
        )

        missing_models = get_missing_model_ids(backend_name, lang=self.lang_var.get())
        if missing_models:
            threading.Thread(
                target=self._auto_download_models,
                args=(missing_models, input_path, output_dir, common_kwargs),
                daemon=True,
            ).start()
            return

        self._launch_worker(input_path, output_dir, common_kwargs)

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

    def _auto_download_models(
        self,
        missing: list[tuple[str, str]],
        input_path: Path,
        output_dir: Path,
        common_kwargs: dict,
    ) -> None:
        """Download missing models in background, then start the worker."""
        def log(msg: str) -> None:
            self.after(0, lambda m=msg: self.log.append(m))

        total = len(missing)
        log(f"Descargando {total} modelo(s) necesarios (puede tardar varios minutos)...")

        for i, (name, model_id) in enumerate(missing, 1):
            log(f"[{i}/{total}] Descargando {name}...")
            self.after(0, lambda n=name: self.progress_label.configure(
                text=f"Descargando {n}...",
            ))
            ok = download_model(model_id)
            if not ok:
                log(f"  Error al descargar {name}.")
                self.after(0, lambda n=name: self._on_model_download_error(n))
                return
            log(f"  {name} listo.")

        log("Modelos listos. Iniciando traducción...")
        self.after(0, lambda: self._launch_worker(input_path, output_dir, common_kwargs))

    def _on_model_download_error(self, name: str) -> None:
        messagebox.showerror(
            "Error",
            f"No se pudo descargar el modelo:\n{name}\n\nComprueba tu conexión a internet.",
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
        models = get_model_status(lang=self.lang_var.get())
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
            success = download_model(model.description)
            self.after(0, lambda: self._on_download_done(model.name, success))

        self._download_status = ctk.CTkLabel(
            self, text=self._t("settings_model_downloading").format(name=model.name),
            text_color="#ffc107",
        )
        self._download_status.grid(row=10, column=0, padx=10, pady=5)
        threading.Thread(target=_do_download, daemon=True).start()

    def _on_download_done(self, name: str, success: bool) -> None:
        if hasattr(self, "_download_status"):
            self._download_status.destroy()
        if success:
            messagebox.showinfo(
                self._t("msg_success"),
                self._t("settings_download_ok").format(name=name),
            )
        else:
            messagebox.showerror(
                self._t("msg_error"),
                self._t("settings_download_err").format(name=name),
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
