"""Opus-MT + CTranslate2 offline translation backend."""

from __future__ import annotations

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modtranslator.backends.base import TranslationBackend

MAX_BATCH_SIZE = 256
DEFAULT_BEAM_SIZE = 2  # beam=2 is 2x faster; quality difference negligible for short strings
MAX_TOKENS = 480  # Marian models support max 512 positions; leave margin
CHAR_HEURISTIC_THRESHOLD = int(MAX_TOKENS * 2.5)  # ~1200 chars; below this, no token check needed
_NUM_TOKENIZER_WORKERS = 4
_INTER_THREADS = min(4, os.cpu_count() or 1)  # CTranslate2 workers; >4 doesn't help CT2

_MODEL_VARIANTS = {
    "base": None,       # Dynamic: opus-mt-{src}-{tgt}
    "tc-big": "tc-big", # Transformer-big: opus-mt-tc-big-{src}-{tgt} (+2.3 BLEU)
}

# Optimized decode defaults for tc-big (larger model benefits from wider beam)
_TC_BIG_DEFAULTS: dict[str, int | float] = {
    "beam_size": 4,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}

_LANG_CODES = {
    "ES": "es",
    "EN": "en",
    "FR": "fr",
    "DE": "de",
    "IT": "it",
    "PT": "pt",
    "RU": "ru",
    "JA": "ja",
    "ZH": "zh",
    "KO": "ko",
    "PL": "pl",
}

_DEFAULT_MODELS_DIR = Path.home() / ".modtranslator" / "models"


class OpusMTBackend(TranslationBackend):
    """Offline translation backend using Helsinki-NLP Opus-MT models via CTranslate2."""

    def __init__(
        self,
        device: str = "auto",
        models_dir: Path | None = None,
        beam_size: int | None = None,
        model_variant: str = "base",
    ) -> None:
        try:
            import ctranslate2  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ctranslate2 package required for Opus-MT backend. "
                "Install with: pip install -e '.[opus-mt]'"
            ) from e
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers package required for Opus-MT backend. "
                "Install with: pip install -e '.[opus-mt]'"
            ) from e

        if model_variant not in _MODEL_VARIANTS:
            raise ValueError(
                f"Unknown Opus-MT model variant '{model_variant}'. "
                f"Choose from: {', '.join(_MODEL_VARIANTS)}"
            )

        self._model_variant = model_variant

        # Apply variant-specific defaults; explicit beam_size overrides
        if model_variant == "tc-big":
            default_beam = int(_TC_BIG_DEFAULTS["beam_size"])
            self._beam_size = beam_size if beam_size is not None else default_beam
            self._repetition_penalty = float(_TC_BIG_DEFAULTS["repetition_penalty"])
            self._no_repeat_ngram_size = int(_TC_BIG_DEFAULTS["no_repeat_ngram_size"])
        else:
            self._beam_size = beam_size if beam_size is not None else DEFAULT_BEAM_SIZE
            self._repetition_penalty = 1.0
            self._no_repeat_ngram_size = 0

        self._models_dir = models_dir or _DEFAULT_MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Register nvidia pip package DLL directories so CT2 can find cuBLAS/cuDNN
        self._register_nvidia_dll_dirs()

        # Resolve device
        self._device = self._resolve_device(device)
        self._compute_type = self._resolve_compute_type(self._device)

        # Caches for loaded translators and tokenizers, keyed by (source, target)
        self._translators: dict[tuple[str, str], object] = {}
        self._tokenizers: dict[tuple[str, str], object] = {}

    @staticmethod
    def _register_nvidia_dll_dirs() -> None:
        """Add nvidia pip package bin/ dirs to DLL search path (Windows only).

        The pip packages nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, etc.
        install DLLs under site-packages/nvidia/<pkg>/bin/ which isn't in
        the system PATH.  Uses both ``os.add_dll_directory`` (for Python
        ctypes) **and** prepends to ``PATH`` (for C++ libraries like
        CTranslate2 that resolve DLLs via the system search path).
        """
        if sys.platform != "win32":
            return
        try:
            import nvidia

            nvidia_root = Path(nvidia.__path__[0])
            for pkg in ("cublas", "cuda_runtime", "cudnn", "cufft", "cusolver", "cusparse"):
                bin_dir = nvidia_root / pkg / "bin"
                if bin_dir.is_dir():
                    bin_str = str(bin_dir)
                    os.add_dll_directory(bin_str)
                    # CT2 (C++) needs DLLs on PATH, not just add_dll_directory
                    if bin_str not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
        except (ImportError, AttributeError, OSError):
            pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device in ("cpu", "cuda"):
            return device
        # auto-detect
        try:
            import ctranslate2

            supported = ctranslate2.get_supported_compute_types("cuda")
            if not supported:
                return "cpu"

            # Verify the CUDA runtime is actually installed, not just the driver.
            # The NVIDIA driver installs nvcuda.dll (driver API) which makes
            # get_supported_compute_types("cuda") succeed, but CTranslate2 needs
            # cuBLAS (from CUDA Toolkit) for actual GPU computation. Without it,
            # translate_batch() hangs indefinitely trying to load missing libraries.
            # Check if cuBLAS DLL is reachable (system PATH or nvidia pip pkg)
            if sys.platform == "win32":
                # Try to find cublas64_12.dll in PATH (includes dirs we prepended)
                import ctypes.util

                found = ctypes.util.find_library("cublas64_12")
                if not found:
                    # Direct search in nvidia pip package bin dirs
                    try:
                        import nvidia

                        dll_path = Path(nvidia.__path__[0]) / "cublas" / "bin" / "cublas64_12.dll"
                        if dll_path.is_file():
                            found = str(dll_path)
                    except (ImportError, AttributeError, OSError):
                        pass
                if found:
                    return "cuda"
                return "cpu"
            else:
                import ctypes

                try:
                    ctypes.cdll.LoadLibrary("libcublas.so.12")
                    return "cuda"
                except OSError:
                    return "cpu"

        except (RuntimeError, ImportError, OSError):
            return "cpu"

    @staticmethod
    def _resolve_compute_type(device: str) -> str:
        if device != "cuda":
            return "int8"
        try:
            import ctranslate2

            supported = ctranslate2.get_supported_compute_types("cuda")
            # Prefer quantized types for speed/memory, fall back to float
            for candidate in ("int8_float16", "int8", "float16", "bfloat16", "float32"):
                if candidate in supported:
                    return candidate
            return "float32"
        except (RuntimeError, ImportError, OSError):
            return "int8"

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        if not texts:
            return []

        src = source_lang or "EN"
        src_code = _LANG_CODES.get(src.upper(), src.lower())
        tgt_code = _LANG_CODES.get(target_lang.upper(), target_lang.lower())

        translator, tokenizer = self._get_translator_and_tokenizer(src_code, tgt_code)

        try:
            return self._translate_chunk(texts, translator, tokenizer)
        except RuntimeError:
            if self._device == "cpu":
                raise
            # CUDA runtime libs missing — fall back to CPU
            translator = self._fallback_to_cpu(src_code, tgt_code)
            return self._translate_chunk(texts, translator, tokenizer)

    def _fallback_to_cpu(self, source: str, target: str) -> object:
        """Rebuild translator on CPU after CUDA failure."""
        import ctranslate2

        self._device = "cpu"
        self._compute_type = "int8"
        key = (source, target)
        self._ensure_model(source, target)
        model_name = self._model_name(source, target)
        ct2_dir = str(self._get_ct2_dir(model_name))
        self._translators[key] = ctranslate2.Translator(
            ct2_dir, device="cpu", compute_type="int8",
            inter_threads=_INTER_THREADS,
        )
        return self._translators[key]

    def _translate_chunk(
        self,
        texts: list[str],
        translator: object,
        tokenizer: object,
    ) -> list[str]:
        # --- Phase 1: Classify by character length ---
        # Texts shorter than CHAR_HEURISTIC_THRESHOLD cannot exceed MAX_TOKENS
        # (Marian averages ~2.5-3 chars/token), so we skip the expensive encode.
        flat_segments: list[str] = []
        segment_map: list[tuple[int, int]] = []  # (start_idx, count) per original text
        long_candidates: list[tuple[int, str]] = []  # (original_index, text)

        for idx, text in enumerate(texts):
            if len(text) >= CHAR_HEURISTIC_THRESHOLD:
                long_candidates.append((idx, text))
            else:
                segment_map.append((len(flat_segments), 1))
                flat_segments.append(text)

        # --- Phase 2: Encode only long candidates to check token count ---
        if long_candidates:
            # Build segments for long texts, inserting at correct positions
            # We need to rebuild segment_map with correct ordering
            flat_segments_new: list[str] = []
            segment_map_new: list[tuple[int, int]] = []
            long_map: dict[int, list[str]] = {}

            # Encode long candidates (potentially in parallel, but these are rare)
            for orig_idx, text in long_candidates:
                ids = tokenizer.encode(text)  # type: ignore[union-attr]
                if len(ids) > MAX_TOKENS:
                    long_map[orig_idx] = self._split_long_text(text, tokenizer)
                else:
                    long_map[orig_idx] = [text]

            # Rebuild in original order
            short_pos = 0
            for idx in range(len(texts)):
                if idx in long_map:
                    parts = long_map[idx]
                    segment_map_new.append((len(flat_segments_new), len(parts)))
                    flat_segments_new.extend(parts)
                else:
                    segment_map_new.append((len(flat_segments_new), 1))
                    flat_segments_new.append(flat_segments[short_pos])
                    short_pos += 1

            flat_segments = flat_segments_new
            segment_map = segment_map_new

        # --- Phase 3: Tokenize all segments in parallel ---
        # SentencePiece releases the GIL, so threads give real parallelism.
        def _tokenize(seg: str) -> list[str]:
            ids = tokenizer.encode(seg)  # type: ignore[union-attr]
            return tokenizer.convert_ids_to_tokens(ids)  # type: ignore[union-attr]

        with ThreadPoolExecutor(max_workers=_NUM_TOKENIZER_WORKERS) as pool:
            tokenized = list(pool.map(_tokenize, flat_segments))

        # --- Phase 4: Translate + batch decode ---
        results = translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            beam_size=self._beam_size,
            max_batch_size=MAX_BATCH_SIZE,
            repetition_penalty=self._repetition_penalty,
            no_repeat_ngram_size=self._no_repeat_ngram_size,
        )

        # Collect output token IDs for batch decode
        all_token_ids: list[list[int]] = []
        fallback_indices: set[int] = set()
        for i, result in enumerate(results):
            if result.hypotheses:
                tokens = result.hypotheses[0]
                all_token_ids.append(
                    tokenizer.convert_tokens_to_ids(tokens)  # type: ignore[union-attr]
                )
            else:
                fallback_indices.add(i)
                all_token_ids.append([])  # placeholder

        # Batch decode all at once
        decoded_texts = tokenizer.batch_decode(  # type: ignore[union-attr]
            all_token_ids, skip_special_tokens=True,
        )

        # Apply fallbacks for empty hypotheses
        decoded: list[str] = [
            flat_segments[i] if i in fallback_indices else decoded_texts[i]
            for i in range(len(flat_segments))
        ]

        # Reassemble: join segments that belonged to the same original text
        translated: list[str] = []
        for start, count in segment_map:
            if count == 1:
                translated.append(decoded[start])
            else:
                translated.append(" ".join(decoded[start : start + count]))

        return translated

    @staticmethod
    def _split_long_text(text: str, tokenizer: object) -> list[str]:
        """Split text into segments that fit within MAX_TOKENS."""
        import re

        # Split by sentence boundaries, keeping the delimiter attached
        sentences = re.split(r'(?<=[.!?…])\s+', text)

        # If no sentence boundaries found, split by newlines
        if len(sentences) == 1 and "\n" in text:
            sentences = [s for s in text.split("\n") if s.strip()]

        # Group sentences into segments that fit within the token limit
        segments: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_ids = tokenizer.encode(sentence)  # type: ignore[union-attr]
            sent_len = len(sent_ids)

            # Single sentence exceeds limit — force-split by words
            if sent_len > MAX_TOKENS:
                if current:
                    segments.append(" ".join(current))
                    current = []
                    current_len = 0
                segments.extend(OpusMTBackend._split_by_words(sentence, tokenizer))
                continue

            if current_len + sent_len > MAX_TOKENS:
                segments.append(" ".join(current))
                current = [sentence]
                current_len = sent_len
            else:
                current.append(sentence)
                current_len += sent_len

        if current:
            segments.append(" ".join(current))

        return segments if segments else [text]

    @staticmethod
    def _split_by_words(text: str, tokenizer: object) -> list[str]:
        """Last-resort split: chunk by words to stay under MAX_TOKENS."""
        words = text.split()
        segments: list[str] = []
        current: list[str] = []

        for word in words:
            current.append(word)
            candidate = " ".join(current)
            if len(tokenizer.encode(candidate)) > MAX_TOKENS:  # type: ignore[union-attr]
                # Remove last word, flush current segment
                current.pop()
                if current:
                    segments.append(" ".join(current))
                current = [word]

        if current:
            segments.append(" ".join(current))

        return segments if segments else [text]

    def _get_translator_and_tokenizer(
        self, source: str, target: str
    ) -> tuple[object, object]:
        key = (source, target)
        if key not in self._translators:
            self._ensure_model(source, target)
            import ctranslate2
            from transformers import AutoTokenizer

            model_name = self._model_name(source, target)
            ct2_dir = self._get_ct2_dir(model_name)

            self._translators[key] = ctranslate2.Translator(
                str(ct2_dir),
                device=self._device,
                compute_type=self._compute_type,
                inter_threads=_INTER_THREADS if self._device == "cpu" else 1,
            )

            # Load tokenizer from local CT2 dir (offline) if available,
            # otherwise download once and save locally for future runs.
            tokenizer_marker = ct2_dir / "tokenizer_config.json"
            if tokenizer_marker.exists():
                self._tokenizers[key] = AutoTokenizer.from_pretrained(
                    str(ct2_dir), local_files_only=True,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained(str(ct2_dir))
                self._tokenizers[key] = tokenizer

        return self._translators[key], self._tokenizers[key]

    def _model_name(self, source: str, target: str) -> str:
        if self._model_variant == "tc-big":
            return f"Helsinki-NLP/opus-mt-tc-big-{source}-{target}"
        return f"Helsinki-NLP/opus-mt-{source}-{target}"

    def _ct2_model_dir(self, model_name: str) -> Path:
        # Helsinki-NLP/opus-mt-en-es → opus-mt-en-es-ct2-float16
        short_name = model_name.split("/")[-1]
        return self._models_dir / f"{short_name}-ct2-{self._compute_type}"

    def _find_existing_ct2(self, model_name: str) -> Path | None:
        """Find an existing CT2 conversion in any quantization format."""
        short_name = model_name.split("/")[-1]
        for candidate in self._models_dir.glob(f"{short_name}-ct2-*"):
            if (candidate / "model.bin").exists():
                return candidate
        return None

    def _ensure_model(self, source: str, target: str) -> None:
        model_name = self._model_name(source, target)
        ct2_dir = self._ct2_model_dir(model_name)

        if (ct2_dir / "model.bin").exists():
            return

        # Check if a conversion exists in a different quantization format
        existing = self._find_existing_ct2(model_name)
        if existing is not None:
            # Reuse existing conversion (e.g. float16 when int8 is requested)
            self._ct2_model_dir_override = existing
            return

        self._convert_model(model_name, ct2_dir)

    def _get_ct2_dir(self, model_name: str) -> Path:
        """Get the actual CT2 model directory (respects overrides)."""
        override = getattr(self, "_ct2_model_dir_override", None)
        if override is not None:
            return override
        return self._ct2_model_dir(model_name)

    def _convert_model(self, model_name: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use Python API instead of subprocess to avoid re-launching
            # the PyInstaller exe as a child process.
            from ctranslate2.converters.transformers import TransformersConverter

            converter = TransformersConverter(model_name, low_cpu_mem_usage=True)
            converter.convert(
                str(output_dir),
                quantization=self._compute_type,
                force=True,
            )
        except Exception as e:
            # Clean up partial conversion
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise RuntimeError(
                f"Failed to convert model {model_name}: {e}"
            ) from e
