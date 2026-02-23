"""Model manager for ML backend detection, download, and CUDA support."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    """Status of a downloadable model."""
    name: str
    description: str
    size_hint: str
    is_downloaded: bool
    required_for: list[str]  # backend names that need this model


# Opus-MT model availability per target language.
# tc-big has better quality; base is smaller fallback.
_OPUS_VARIANTS: dict[str, list[str]] = {
    "ES": ["tc-big", "base"],
    "FR": ["tc-big", "base"],
    "IT": ["tc-big", "base"],
    "PT": ["tc-big", "base"],
    "DE": ["base"],
    "RU": ["base"],
    "PL": ["base"],
}


def _opus_model_id(lang: str) -> str | None:
    """Return the best available Opus-MT HuggingFace model ID for a target language.

    Prefers tc-big over base. Returns None if no Opus-MT model exists for the language.
    """
    tgt = lang.lower()
    variants = _OPUS_VARIANTS.get(lang.upper(), [])
    for variant in variants:
        if variant == "tc-big":
            model_id = f"Helsinki-NLP/opus-mt-tc-big-en-{tgt}"
        else:
            model_id = f"Helsinki-NLP/opus-mt-en-{tgt}"
        if _check_model_exists(model_id):
            return model_id
    # None downloaded yet → return the best one to download
    if variants:
        v = variants[0]
        if v == "tc-big":
            return f"Helsinki-NLP/opus-mt-tc-big-en-{tgt}"
        return f"Helsinki-NLP/opus-mt-en-{tgt}"
    return None


def _opus_display_name(model_id: str) -> str:
    """Human-readable name from a HuggingFace Opus-MT model ID."""
    short = model_id.split("/")[-1]  # opus-mt-tc-big-en-es
    return short.replace("Helsinki-NLP/", "").replace("opus-mt-", "Opus-MT ")


def detect_cuda() -> dict[str, object]:
    """Detect CUDA availability.

    Returns dict with keys:
        available: bool
        gpu_name: str or None
    """
    result: dict[str, object] = {
        "available": False,
        "gpu_name": None,
    }

    # Check nvidia-smi (try common paths on Windows)
    nvidia_smi_paths = ["nvidia-smi"]
    if sys.platform == "win32":
        nvidia_smi_paths.extend([
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ])
    for nvidia_smi in nvidia_smi_paths:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                result["available"] = True
                result["gpu_name"] = proc.stdout.strip().split("\n")[0]
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return result


def _check_model_exists(model_id: str) -> bool:
    """Check if a model is available as CT2 (ready to use)."""
    short_name = model_id.split("/")[-1]

    ct2_dir = Path.home() / ".modtranslator" / "models"
    for candidate in ct2_dir.glob(f"{short_name}-ct2-*"):
        if (candidate / "model.bin").exists():
            return True

    return False


def get_model_status(lang: str = "ES") -> list[ModelInfo]:
    """Return status of all downloadable models for the given target language."""
    models: list[ModelInfo] = []

    opus_id = _opus_model_id(lang)
    if opus_id is not None:
        models.append(ModelInfo(
            name=_opus_display_name(opus_id),
            description=opus_id,
            size_hint="~300 MB",
            is_downloaded=_check_model_exists(opus_id),
            required_for=["opus-mt", "hybrid"],
        ))

    models.append(ModelInfo(
        name="NLLB 1.3B (CTranslate2)",
        description="facebook/nllb-200-distilled-1.3B",
        size_hint="~2.5 GB",
        is_downloaded=_check_model_exists("facebook/nllb-200-distilled-1.3B"),
        required_for=["hybrid"],
    ))
    return models


def download_model(model_description: str, on_progress: object = None) -> bool:
    """Download a model by its description (HuggingFace model ID).

    Returns True on success, False on failure.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_description)
        return True
    except Exception:
        return False


def check_backend_ready(backend_name: str, lang: str = "ES") -> tuple[bool, str]:
    """Check if a backend has all required dependencies and models.

    Returns (ready, message).
    """
    if backend_name == "dummy":
        return True, "Ready"

    if backend_name == "deepl":
        try:
            import deepl  # noqa: F401
            return True, "deepl package installed"
        except ImportError:
            return False, "Missing package: pip install deepl"

    if backend_name == "opus-mt":
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError:
            return False, "Missing packages: pip install ctranslate2 sentencepiece transformers"
        opus_id = _opus_model_id(lang)
        if opus_id is None:
            return False, f"No Opus-MT model available for {lang}"
        if not _check_model_exists(opus_id):
            return False, "Opus-MT model not downloaded"
        return True, "Ready"

    if backend_name == "nllb":
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError:
            return False, "Missing packages: pip install ctranslate2 sentencepiece transformers"
        if not _check_model_exists("facebook/nllb-200-distilled-1.3B"):
            return False, "NLLB model not downloaded"
        return True, "Ready"

    if backend_name == "hybrid":
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError:
            return False, "Missing packages: pip install ctranslate2 sentencepiece transformers"
        opus_id = _opus_model_id(lang)
        opus_ok = opus_id is not None and _check_model_exists(opus_id)
        nllb_ok = _check_model_exists("facebook/nllb-200-distilled-1.3B")
        if not opus_ok and not nllb_ok:
            return False, "Opus-MT and NLLB models not downloaded"
        if not opus_ok:
            return False, "Opus-MT model not downloaded"
        if not nllb_ok:
            return False, "NLLB model not downloaded"
        return True, "Ready"

    return False, f"Unknown backend: {backend_name}"


def get_missing_model_ids(backend_name: str, lang: str = "ES") -> list[tuple[str, str]]:
    """Return (display_name, hf_model_id) for models needed by backend that are not downloaded.

    Returns an empty list if the backend needs no models or all models are present.
    """
    nllb = ("NLLB 1.3B", "facebook/nllb-200-distilled-1.3B")

    opus_id = _opus_model_id(lang)
    opus_entry = (_opus_display_name(opus_id), opus_id) if opus_id else None

    missing: list[tuple[str, str]] = []
    if backend_name == "opus-mt":
        if opus_entry and not _check_model_exists(opus_entry[1]):
            missing.append(opus_entry)
    elif backend_name == "nllb":
        if not _check_model_exists(nllb[1]):
            missing.append(nllb)
    elif backend_name == "hybrid":
        if opus_entry and not _check_model_exists(opus_entry[1]):
            missing.append(opus_entry)
        if not _check_model_exists(nllb[1]):
            missing.append(nllb)
    return missing
