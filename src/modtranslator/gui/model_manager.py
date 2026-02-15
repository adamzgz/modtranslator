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


def detect_cuda() -> dict[str, object]:
    """Detect CUDA availability.

    Returns dict with keys:
        available: bool
        gpu_name: str or None
        torch_available: bool
    """
    result: dict[str, object] = {
        "available": False,
        "gpu_name": None,
        "torch_available": False,
    }

    # Try torch first
    try:
        import torch
        result["torch_available"] = True
        if torch.cuda.is_available():
            result["available"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            return result
    except ImportError:
        pass

    # Fallback: check nvidia-smi (try common paths on Windows)
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


def _get_models_dir() -> Path:
    """Get the HuggingFace cache directory where models are stored."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_dir


def _check_model_exists(model_id: str) -> bool:
    """Check if a HuggingFace model is already cached locally."""
    cache_dir = _get_models_dir()
    # HF cache uses models--org--name format
    safe_name = "models--" + model_id.replace("/", "--")
    model_dir = cache_dir / safe_name
    if model_dir.is_dir():
        # Check for snapshots directory (indicates completed download)
        snapshots = model_dir / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            return True
    return False


def get_model_status() -> list[ModelInfo]:
    """Return status of all downloadable models."""
    models = [
        ModelInfo(
            name="Opus-MT en-es (tc-big)",
            description="Helsinki-NLP/opus-mt-tc-big-en-es",
            size_hint="~300 MB",
            is_downloaded=_check_model_exists("Helsinki-NLP/opus-mt-tc-big-en-es"),
            required_for=["opus-mt", "hybrid"],
        ),
        ModelInfo(
            name="NLLB 1.3B (CTranslate2)",
            description="facebook/nllb-200-distilled-1.3B",
            size_hint="~2.5 GB",
            is_downloaded=_check_model_exists("facebook/nllb-200-distilled-1.3B"),
            required_for=["hybrid"],
        ),
    ]
    return models


def download_model(model_description: str, on_progress: object = None) -> bool:
    """Download a model by its description (HuggingFace model ID).

    Returns True on success, False on failure.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_description)
        return True
    except ImportError:
        # Try with transformers
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            AutoTokenizer.from_pretrained(model_description)
            AutoModelForSeq2SeqLM.from_pretrained(model_description)
            return True
        except Exception:
            return False
    except Exception:
        return False


def check_backend_ready(backend_name: str) -> tuple[bool, str]:
    """Check if a backend has all required dependencies and models.

    Returns (ready, message).
    """
    if backend_name == "dummy":
        return True, "Listo"

    if backend_name == "deepl":
        try:
            import deepl  # noqa: F401
            return True, "Paquete deepl instalado"
        except ImportError:
            return False, "Falta paquete: pip install deepl"

    if backend_name == "opus-mt":
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError:
            return False, "Faltan paquetes: pip install ctranslate2 sentencepiece transformers"
        if not _check_model_exists("Helsinki-NLP/opus-mt-tc-big-en-es"):
            return False, "Modelo Opus-MT no descargado"
        return True, "Listo"

    if backend_name == "nllb":
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError:
            return False, "Faltan paquetes: pip install ctranslate2 sentencepiece transformers"
        if not _check_model_exists("facebook/nllb-200-distilled-1.3B"):
            return False, "Modelo NLLB no descargado"
        return True, "Listo"

    if backend_name == "hybrid":
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError:
            return False, "Faltan paquetes: pip install ctranslate2 sentencepiece transformers"
        opus_ok = _check_model_exists("Helsinki-NLP/opus-mt-tc-big-en-es")
        nllb_ok = _check_model_exists("facebook/nllb-200-distilled-1.3B")
        if not opus_ok and not nllb_ok:
            return False, "Modelos Opus-MT y NLLB no descargados"
        if not opus_ok:
            return False, "Modelo Opus-MT no descargado"
        if not nllb_ok:
            return False, "Modelo NLLB no descargado"
        return True, "Listo"

    return False, f"Backend desconocido: {backend_name}"
