"""NLLB-200 + CTranslate2 offline translation backend."""

from __future__ import annotations

import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modtranslator.backends.base import TranslationBackend

MAX_BATCH_SIZE = 256
_GPU_MAX_BATCH_SIZE = 32  # NLLB 1.3B float16 ~2.6GB; 256 overflows 8GB VRAM
DEFAULT_BEAM_SIZE = 2
MAX_TOKENS = 900  # NLLB supports 1024 positions; leave margin
CHAR_HEURISTIC_THRESHOLD = int(MAX_TOKENS * 2.5)  # ~2250 chars
_NUM_TOKENIZER_WORKERS = 4
_INTER_THREADS = min(4, os.cpu_count() or 1)

# Anti-repetition: NLLB duplicates translations of short inputs (1-3 words)
# because the decoder lacks context to emit EOS. repetition_penalty penalizes
# previously generated tokens, making "Estimulante Estimulante" unlikely.
# Value 1.2 is proven in production by Argos Translate (NLLB + CT2).
REPETITION_PENALTY = 1.2
_SHORT_INPUT_MAX_WORDS = 5  # dedup/filler-trim post-processing threshold
_SINGLE_WORD_MAX_OUTPUT = 3  # max words allowed for a 1-word input translation
_ARTICLES = frozenset({"la", "el", "los", "las", "un", "una"})

# Filler patterns: NLLB hallucinates encyclopedia-like text after short translations.
# "Blood Pack" → "Bolsa de sangre es el valor de la moneda."
# We trim at the first filler marker. No legitimate EN→ES translation of a
# 1-3 word noun phrase should contain "es el/la/un/una..." as a copula.
_FILLER_PATTERN = re.compile(
    r"\s+(?:es\s+(?:el|la|un|una|los|las)\b"
    r"|también\b|y\s+también\b"
    r"|fue\b)",
    re.IGNORECASE,
)

# FLORES-200 language codes used by NLLB
_FLORES_CODES = {
    "ES": "spa_Latn",
    "EN": "eng_Latn",
    "FR": "fra_Latn",
    "DE": "deu_Latn",
    "IT": "ita_Latn",
    "PT": "por_Latn",
    "RU": "rus_Cyrl",
    "JA": "jpn_Jpan",
    "ZH": "zho_Hans",
    "KO": "kor_Hang",
    "PL": "pol_Latn",
}

_MODEL_VARIANTS = {
    "600M": "facebook/nllb-200-distilled-600M",
    "1.3B": "facebook/nllb-200-distilled-1.3B",
}

_DEFAULT_MODELS_DIR = Path.home() / ".modtranslator" / "models"


class NLLBBackend(TranslationBackend):
    """Offline translation backend using Meta NLLB-200 models via CTranslate2."""

    def __init__(
        self,
        device: str = "auto",
        models_dir: Path | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        model_size: str = "1.3B",
    ) -> None:
        try:
            import ctranslate2  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ctranslate2 package required for NLLB backend. "
                "Install with: pip install -e '.[nllb]'"
            ) from e
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers package required for NLLB backend. "
                "Install with: pip install -e '.[nllb]'"
            ) from e

        if model_size not in _MODEL_VARIANTS:
            raise ValueError(
                f"Unknown NLLB model size '{model_size}'. "
                f"Choose from: {', '.join(_MODEL_VARIANTS)}"
            )

        self._beam_size = beam_size
        self._model_size = model_size
        self._hf_model_name = _MODEL_VARIANTS[model_size]
        self._models_dir = models_dir or _DEFAULT_MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)

        self._register_nvidia_dll_dirs()

        self._device = self._resolve_device(device)
        self._compute_type = self._resolve_compute_type(self._device)

        # Single translator + tokenizer (multilingual model)
        self._translator: object | None = None
        self._tokenizer: object | None = None

    @staticmethod
    def _register_nvidia_dll_dirs() -> None:
        """Add nvidia pip package bin/ dirs to DLL search path (Windows only).

        Uses both ``os.add_dll_directory`` (for Python ctypes) **and**
        prepends to ``PATH`` (for C++ libraries like CTranslate2 that
        resolve DLLs via the system search path at runtime).
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
        try:
            import ctranslate2

            supported = ctranslate2.get_supported_compute_types("cuda")
            if not supported:
                return "cpu"

            import ctypes

            for lib_name in ("cublas64_12", "libcublas.so.12"):
                try:
                    ctypes.cdll.LoadLibrary(lib_name)
                    return "cuda"
                except OSError:
                    continue

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
        src_flores = _FLORES_CODES.get(src.upper())
        tgt_flores = _FLORES_CODES.get(target_lang.upper())
        if src_flores is None:
            raise ValueError(f"Unsupported source language '{src}' for NLLB")
        if tgt_flores is None:
            raise ValueError(f"Unsupported target language '{target_lang}' for NLLB")

        translator, tokenizer = self._get_translator_and_tokenizer(src_flores)

        try:
            return self._translate_chunk(texts, translator, tokenizer, tgt_flores)
        except RuntimeError:
            if self._device == "cpu":
                raise
            translator = self._fallback_to_cpu()
            return self._translate_chunk(texts, translator, tokenizer, tgt_flores)

    def _fallback_to_cpu(self) -> object:
        """Rebuild translator on CPU after CUDA failure."""
        import ctranslate2

        self._device = "cpu"
        self._compute_type = "int8"
        self._ensure_model()
        ct2_dir = str(self._get_ct2_dir())
        self._translator = ctranslate2.Translator(
            ct2_dir, device="cpu", compute_type="int8",
            inter_threads=_INTER_THREADS,
        )
        return self._translator

    def _translate_chunk(
        self,
        texts: list[str],
        translator: object,
        tokenizer: object,
        tgt_flores: str,
    ) -> list[str]:
        # --- Phase 1: Classify by character length ---
        flat_segments: list[str] = []
        segment_map: list[tuple[int, int]] = []
        long_candidates: list[tuple[int, str]] = []

        for idx, text in enumerate(texts):
            if len(text) >= CHAR_HEURISTIC_THRESHOLD:
                long_candidates.append((idx, text))
            else:
                segment_map.append((len(flat_segments), 1))
                flat_segments.append(text)

        # --- Phase 2: Encode only long candidates to check token count ---
        if long_candidates:
            flat_segments_new: list[str] = []
            segment_map_new: list[tuple[int, int]] = []
            long_map: dict[int, list[str]] = {}

            for orig_idx, text in long_candidates:
                ids = tokenizer.encode(text)  # type: ignore[union-attr]
                if len(ids) > MAX_TOKENS:
                    long_map[orig_idx] = self._split_long_text(text, tokenizer)
                else:
                    long_map[orig_idx] = [text]

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
        def _tokenize(seg: str) -> list[str]:
            ids = tokenizer.encode(seg)  # type: ignore[union-attr]
            return tokenizer.convert_ids_to_tokens(ids)  # type: ignore[union-attr]

        with ThreadPoolExecutor(max_workers=_NUM_TOKENIZER_WORKERS) as pool:
            tokenized = list(pool.map(_tokenize, flat_segments))

        # --- Phase 4: Translate with target_prefix + batch decode ---
        target_prefix = [[tgt_flores]] * len(tokenized)

        batch_size = _GPU_MAX_BATCH_SIZE if self._device == "cuda" else MAX_BATCH_SIZE
        results = translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self._beam_size,
            max_batch_size=batch_size,
            repetition_penalty=REPETITION_PENALTY,
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
                all_token_ids.append([])

        decoded_texts = tokenizer.batch_decode(  # type: ignore[union-attr]
            all_token_ids, skip_special_tokens=True,
        )

        decoded: list[str] = [
            flat_segments[i] if i in fallback_indices else decoded_texts[i].strip()
            for i in range(len(flat_segments))
        ]

        # Post-process short strings (NLLB decoder artifacts):
        # 0. Strip double-space echo (NLLB duplicates EN+ES separated by "  ")
        # 1. Trim hallucinated filler ("es el/la...", "también conocido...")
        # 2. Cap single-word inputs to 1-3 output words
        # 3. Deduplicate repeated patterns (case-insensitive + bookend)
        # Order matters: echo removal first, then filler can hide duplications.
        decoded = [
            self._deduplicate_short(
                self._cap_single_word_output(
                    self._trim_hallucinated_filler(
                        self._strip_echo(decoded[i], flat_segments[i]),
                        flat_segments[i],
                    ),
                    flat_segments[i],
                ),
                flat_segments[i],
            )
            for i in range(len(decoded))
        ]

        # Reassemble segmented texts
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

        sentences = re.split(r'(?<=[.!?…])\s+', text)

        if len(sentences) == 1 and "\n" in text:
            sentences = [s for s in text.split("\n") if s.strip()]

        segments: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_ids = tokenizer.encode(sentence)  # type: ignore[union-attr]
            sent_len = len(sent_ids)

            if sent_len > MAX_TOKENS:
                if current:
                    segments.append(" ".join(current))
                    current = []
                    current_len = 0
                segments.extend(NLLBBackend._split_by_words(sentence, tokenizer))
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
                current.pop()
                if current:
                    segments.append(" ".join(current))
                current = [word]

        if current:
            segments.append(" ".join(current))

        return segments if segments else [text]

    @staticmethod
    def _strip_echo(translated: str, original: str) -> str:
        """Remove echoed original text from translation.

        NLLB sometimes outputs the original English alongside the Spanish
        translation in two ways:

        1. Double-space separated: "Switch de energía  Power switch"
        2. Concatenated: "Play Cell Juega con la celda"

        For short inputs (≤ _SHORT_INPUT_MAX_WORDS words), detect and
        remove the echoed English part.
        """
        if len(original.split()) > _SHORT_INPUT_MAX_WORDS:
            return translated

        # --- Strategy 1: double-space split ---
        if "  " in translated:
            parts = [p.strip() for p in translated.split("  ") if p.strip()]
            if len(parts) >= 2:
                orig_lower = original.lower()
                best_part = parts[0]
                best_score = 0.0
                for part in parts:
                    part_lower = part.lower()
                    if part_lower == orig_lower:
                        continue
                    orig_words = set(orig_lower.split())
                    part_words = set(part_lower.split())
                    overlap = len(orig_words & part_words)
                    score = len(part_words) - overlap
                    if score > best_score:
                        best_score = score
                        best_part = part
                return best_part

        # --- Strategy 2: original EN embedded as substring ---
        # "Play Cell Juega con la celda" → remove "Play Cell" → "Juega con la celda"
        # "Ant Vision La visión de la hormiga" → remove "Ant Vision" → "La visión de la hormiga"
        orig_lower = original.lower()
        trans_lower = translated.lower()

        # Check if original appears at the start
        if trans_lower.startswith(orig_lower) and len(translated) > len(original):
            remainder = translated[len(original):].lstrip()
            if remainder:
                return remainder

        # Check if original appears at the end
        if trans_lower.endswith(orig_lower) and len(translated) > len(original):
            remainder = translated[:len(translated) - len(original)].rstrip()
            if remainder:
                return remainder

        return translated

    @staticmethod
    def _cap_single_word_output(translated: str, original: str) -> str:
        """Cap output length for single-word inputs.

        A single English word legitimately translates to 1-3 Spanish words
        (noun, article + noun, or compound like "casilla de correos").
        If NLLB produces more, it's filler:
          "Light" → "Luz de la luz"  → "Luz"
          "Booth" → "Booth de la tienda" → "Booth"
        """
        if len(original.split()) != 1:
            return translated

        words = translated.split()
        if len(words) <= _SINGLE_WORD_MAX_OUTPUT:
            return translated

        # Allow article + noun (e.g. "La ketamina")
        if words[0].lower() in _ARTICLES:
            return " ".join(words[:2])

        return words[0]

    @staticmethod
    def _trim_hallucinated_filler(translated: str, original: str) -> str:
        """Trim hallucinated filler text from short input translations.

        NLLB generates encyclopedia-like explanations for short inputs:
          "Blood Pack" → "Bolsa de sangre es el valor de la moneda."
          "Ketamine"   → "La ketamina es una sustancia"

        For inputs ≤ _SHORT_INPUT_MAX_WORDS words, trim at common Spanish
        filler patterns ("es el/la/un/una", "también").
        """
        if len(original.split()) > _SHORT_INPUT_MAX_WORDS:
            return translated

        match = _FILLER_PATTERN.search(translated)
        if match:
            trimmed = translated[: match.start()].rstrip(" .,;:")
            if trimmed:
                return trimmed

        return translated

    @staticmethod
    def _deduplicate_short(translated: str, original: str) -> str:
        """Fix repeated output for short inputs (NLLB decoder artifact).

        NLLB generates duplicated translations for very short inputs:
          "Stimpak"    → "Estimulante Estimulante"
          "Blood Pack" → "Bolsa de sangre Bolsa de sangre"
          "Open"       → "Abierto abierto"  (case-insensitive dupe)

        Also catches "bookend" patterns for 1-word inputs where the model
        wraps the translation around a preposition:
          "Stone"  → "Piedra de piedra"  (first == last, case-insensitive)
          "Bed"    → "Cama de cama"

        Detection: if the output is an N-fold repetition of a pattern
        (case-insensitive) and the original had ≤ _SHORT_INPUT_MAX_WORDS
        words, keep one copy with original casing.
        """
        if len(original.split()) > _SHORT_INPUT_MAX_WORDS:
            return translated

        words = translated.split()
        n = len(words)
        if n < 2:
            return translated

        words_lower = [w.lower() for w in words]

        # Try pattern lengths from 1 word up to half the output
        for plen in range(1, n // 2 + 1):
            if n % plen != 0:
                continue
            pattern = words_lower[:plen]
            if all(
                words_lower[i * plen : (i + 1) * plen] == pattern
                for i in range(1, n // plen)
            ):
                return " ".join(words[:plen])

        # Bookend check for 1-word inputs: "Piedra de piedra" → "Piedra"
        # NLLB wraps the noun around a preposition, producing "X prep X".
        if len(original.split()) == 1 and n >= 3 and words_lower[0] == words_lower[-1]:
            return words[0]

        return translated

    def _get_translator_and_tokenizer(
        self, src_flores: str
    ) -> tuple[object, object]:
        if self._translator is None:
            self._ensure_model()
            import ctranslate2
            from transformers import AutoTokenizer

            ct2_dir = self._get_ct2_dir()

            self._translator = ctranslate2.Translator(
                str(ct2_dir),
                device=self._device,
                compute_type=self._compute_type,
                inter_threads=_INTER_THREADS if self._device == "cpu" else 1,
            )

            tokenizer_marker = ct2_dir / "tokenizer_config.json"
            if tokenizer_marker.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(ct2_dir), local_files_only=True, src_lang=src_flores,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self._hf_model_name, src_lang=src_flores,
                )
                tokenizer.save_pretrained(str(ct2_dir))
                self._tokenizer = tokenizer
        else:
            # Update src_lang for subsequent calls with different source language
            if hasattr(self._tokenizer, "src_lang"):
                self._tokenizer.src_lang = src_flores  # type: ignore[union-attr]

        return self._translator, self._tokenizer  # type: ignore[return-value]

    def _ct2_model_dir(self) -> Path:
        short_name = self._hf_model_name.split("/")[-1]
        return self._models_dir / f"{short_name}-ct2-{self._compute_type}"

    def _find_existing_ct2(self) -> Path | None:
        """Find an existing CT2 conversion in any quantization format."""
        short_name = self._hf_model_name.split("/")[-1]
        for candidate in self._models_dir.glob(f"{short_name}-ct2-*"):
            if (candidate / "model.bin").exists():
                return candidate
        return None

    def _get_ct2_dir(self) -> Path:
        """Get the actual CT2 model directory (respects overrides)."""
        override = getattr(self, "_ct2_model_dir_override", None)
        if override is not None:
            return override
        return self._ct2_model_dir()

    def _ensure_model(self) -> None:
        ct2_dir = self._ct2_model_dir()

        if (ct2_dir / "model.bin").exists():
            return

        # Check if a conversion exists in a different quantization format
        existing = self._find_existing_ct2()
        if existing is not None:
            self._ct2_model_dir_override = existing
            return

        self._convert_model(ct2_dir)

    def _convert_model(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use Python API instead of subprocess for transformers 5.0+ compat.
            # transformers 5.0 replaced tokenizer classes with TokenizersBackend
            # which lacks additional_special_tokens. The CT2 converter accesses
            # this attribute during get_vocabulary(). For NLLB, all language
            # tokens are already within the main vocab range, so an empty list
            # is safe — no tokens are missed.
            self._patch_tokenizer_compat()

            from ctranslate2.converters.transformers import TransformersConverter

            converter = TransformersConverter(
                self._hf_model_name, low_cpu_mem_usage=True,
            )
            converter.convert(
                str(output_dir),
                quantization=self._compute_type,
                force=True,
            )
        except Exception as e:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise RuntimeError(
                f"Failed to convert model {self._hf_model_name}: {e}"
            ) from e

    @staticmethod
    def _patch_tokenizer_compat() -> None:
        """Patch transformers 5.0+ tokenizer for CTranslate2 converter compat.

        transformers 5.0 replaced tokenizer classes with TokenizersBackend
        which lacks additional_special_tokens. The __getattr__ that raises
        the AttributeError lives on PreTrainedTokenizerBase (the parent class).
        We patch it to return [] for that specific attribute so the CT2
        converter's get_vocabulary() doesn't crash. All NLLB language tokens
        are already in the main vocab range, so nothing is lost.
        """
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        except ImportError:
            return

        if hasattr(PreTrainedTokenizerBase, "_modtranslator_patched"):
            return

        original_getattr = PreTrainedTokenizerBase.__getattr__

        def _patched_getattr(self: object, key: str) -> object:
            if key == "additional_special_tokens":
                return []
            return original_getattr(self, key)  # type: ignore[arg-type]

        PreTrainedTokenizerBase.__getattr__ = _patched_getattr  # type: ignore[assignment]
        PreTrainedTokenizerBase._modtranslator_patched = True  # type: ignore[attr-defined]
