"""Model manager for ML backend detection, download, and CUDA support."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


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


# Pre-converted CT2 int8 models available on HuggingFace.
# NLLB uses a dedicated repo (OpenNMT); Opus-MT models use a shared repo with subdirs.
_OPUS_CT2_HF_REPO = "adamzgz/modtranslator-models"

# Map from original HF model_id → (hf_repo, subdir_or_None).
# subdir=None means the repo IS the model (files at root).
# subdir=str means files are under repo/{subdir}/.
_CT2_SOURCES: dict[str, tuple[str, str | None]] = {
    "Helsinki-NLP/opus-mt-tc-big-en-es": (_OPUS_CT2_HF_REPO, "opus-mt-tc-big-en-es-ct2-int8"),
    "Helsinki-NLP/opus-mt-tc-big-en-fr": (_OPUS_CT2_HF_REPO, "opus-mt-tc-big-en-fr-ct2-int8"),
    "Helsinki-NLP/opus-mt-tc-big-en-it": (_OPUS_CT2_HF_REPO, "opus-mt-tc-big-en-it-ct2-int8"),
    "Helsinki-NLP/opus-mt-tc-big-en-pt": (_OPUS_CT2_HF_REPO, "opus-mt-tc-big-en-pt-ct2-int8"),
    "Helsinki-NLP/opus-mt-en-de": (_OPUS_CT2_HF_REPO, "opus-mt-en-de-ct2-int8"),
    "Helsinki-NLP/opus-mt-en-ru": (_OPUS_CT2_HF_REPO, "opus-mt-en-ru-ct2-int8"),
    "facebook/nllb-200-distilled-1.3B": ("OpenNMT/nllb-200-distilled-1.3B-ct2-int8", None),
}


def _ensure_stdio() -> None:
    """Ensure sys.stdout/stderr are not None (PyInstaller console=False sets them to None)."""
    import os
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def _download_preconverted(model_id: str, ct2_dir: Path) -> tuple[bool, str]:
    """Download pre-converted CT2 int8 model from HuggingFace.

    Returns (success, error_message).
    """
    source = _CT2_SOURCES.get(model_id)
    if source is None:
        return False, f"No pre-converted model available for {model_id}"

    hf_repo, subdir = source

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub not installed"

    _ensure_stdio()

    try:
        if subdir is None:
            # Model files are at the repo root → download directly into ct2_dir
            snapshot_download(hf_repo, local_dir=str(ct2_dir))
        else:
            # Model files are under a subdirectory in a shared repo
            snapshot_download(
                hf_repo,
                allow_patterns=[f"{subdir}/*"],
                local_dir=str(ct2_dir.parent),  # ~/.modtranslator/models/
            )

        if not (ct2_dir / "model.bin").exists():
            return False, f"Downloaded files missing model.bin in {ct2_dir}"
        return True, ""
    except Exception as e:
        # Clean up partial download
        if ct2_dir.exists():
            shutil.rmtree(ct2_dir, ignore_errors=True)
        return False, str(e)


def download_model(
    model_id: str, on_progress: object = None,
) -> tuple[bool, str]:
    """Download a pre-converted CT2 model, or convert from HuggingFace if torch is available.

    Returns (success, error_message).
    """
    models_dir = Path.home() / ".modtranslator" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    short_name = model_id.split("/")[-1]
    ct2_dir = models_dir / f"{short_name}-ct2-int8"

    if (ct2_dir / "model.bin").exists():
        return True, ""

    # Strategy 1: Download pre-converted CT2 model (no torch needed)
    ok, err = _download_preconverted(model_id, ct2_dir)
    if ok:
        log.info("Downloaded pre-converted CT2 model for %s", model_id)
        return True, ""
    log.info("Pre-converted download failed (%s), trying local conversion...", err)

    # Strategy 2: Convert locally with TransformersConverter (needs torch)
    try:
        import ctranslate2  # noqa: F401
        from ctranslate2.converters.transformers import TransformersConverter
    except ImportError:
        return False, (
            "Could not download pre-converted model and ctranslate2 is not available. "
            f"Pre-converted error: {err}"
        )

    try:
        import torch  # noqa: F401
    except ImportError:
        return False, (
            "Could not download pre-converted model and torch is not installed "
            "(needed for local conversion). "
            f"Pre-converted error: {err}"
        )

    try:
        if "nllb" in model_id.lower():
            with _nllb_tokenizer_patch():
                converter = TransformersConverter(model_id, low_cpu_mem_usage=True)
                converter.convert(str(ct2_dir), quantization="int8", force=True)
        else:
            converter = TransformersConverter(model_id, low_cpu_mem_usage=True)
            converter.convert(str(ct2_dir), quantization="int8", force=True)
    except Exception as e:
        if ct2_dir.exists():
            shutil.rmtree(ct2_dir, ignore_errors=True)
        return False, str(e)
    finally:
        _delete_hf_cache(model_id)

    return True, ""


def _delete_hf_cache(hf_model_name: str) -> None:
    """Delete the HuggingFace hub cache for a model after CT2 conversion.

    The cache is only needed during conversion; once the CT2 model exists
    it is never used again. Frees several GB per model.
    Respects HF_HOME / HUGGINGFACE_HUB_CACHE env overrides.
    """
    import os

    hf_home = Path(
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        )
    )
    hub_dir = hf_home if hf_home.name == "hub" else hf_home / "hub"
    cache_dir = hub_dir / ("models--" + hf_model_name.replace("/", "--"))
    try:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    except OSError:
        pass  # Best-effort cleanup; don't crash if cache can't be deleted


@contextmanager
def _nllb_tokenizer_patch() -> Iterator[None]:
    """Temporarily patch transformers 5.0+ tokenizer for CT2 converter compat.

    transformers 5.0 replaced tokenizer classes with TokenizersBackend
    which lacks additional_special_tokens. We patch __getattr__ to return []
    for that attribute so the CT2 converter's get_vocabulary() doesn't crash.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:
        yield
        return

    original_getattr = PreTrainedTokenizerBase.__getattr__

    def _patched_getattr(self: object, key: str) -> object:
        if key == "additional_special_tokens":
            return []
        return original_getattr(self, key)  # type: ignore[no-untyped-call,arg-type]

    PreTrainedTokenizerBase.__getattr__ = _patched_getattr  # type: ignore[method-assign]
    try:
        yield
    finally:
        PreTrainedTokenizerBase.__getattr__ = original_getattr  # type: ignore[method-assign]


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

    if backend_name == "hybrid-deepl":
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError:
            return False, "Missing packages: pip install ctranslate2 sentencepiece transformers"
        try:
            import deepl  # noqa: F401
        except ImportError:
            return False, "Missing package: pip install deepl"
        if not _check_model_exists("facebook/nllb-200-distilled-1.3B"):
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
    elif backend_name == "hybrid-deepl" and not _check_model_exists(nllb[1]):
        missing.append(nllb)
    return missing
