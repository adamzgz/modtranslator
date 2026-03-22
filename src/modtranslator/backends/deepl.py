"""DeepL API translation backend."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from modtranslator.backends.base import TranslationBackend

# DeepL free tier limits
MAX_BATCH_SIZE = 50
RATE_LIMIT_RETRY_SECONDS = 1.0
MAX_RETRIES = 3


# DeepL requires specific language codes for some languages.
# Our internal codes are ISO 639-1; DeepL needs variants for PT and EN.
_DEEPL_TARGET_CODES: dict[str, str] = {
    "PT": "PT-BR",
    "EN": "EN-US",
}
_DEEPL_SOURCE_CODES: dict[str, str] = {
    "EN": "EN",  # source accepts plain "EN", unlike target
}


class DeepLBackend(TranslationBackend):
    """Translation backend using the DeepL API."""

    def __init__(self, api_key: str) -> None:
        try:
            import deepl
        except ImportError:
            raise ImportError(
                "DeepL backend requires the 'deepl' package. "
                "Install it with: pip install modtranslator[deepl]"
            ) from None
        self._deepl = deepl
        self._translator = deepl.Translator(api_key)

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        """Translate texts using DeepL, handling rate limits and batching."""
        if not texts:
            return []

        # Map to DeepL-specific language codes
        deepl_target = _DEEPL_TARGET_CODES.get(target_lang.upper(), target_lang)
        deepl_source = (
            _DEEPL_SOURCE_CODES.get(source_lang.upper(), source_lang)
            if source_lang
            else None
        )

        results: list[str] = []

        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            translated = self._translate_with_retry(batch, deepl_target, deepl_source)
            results.extend(translated)

        return results

    def _translate_with_retry(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None,
    ) -> list[str]:
        """Translate a single batch with retry on rate limit."""
        for attempt in range(MAX_RETRIES):
            try:
                result = self._translator.translate_text(
                    texts,
                    target_lang=target_lang,
                    source_lang=source_lang,
                )
                # translate_text returns a list of TextResult when given a list
                if isinstance(result, list):
                    return [r.text for r in result]
                return [result.text]

            except self._deepl.QuotaExceededException:
                raise

            except (self._deepl.DeepLException, ConnectionError, TimeoutError):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_RETRY_SECONDS * (attempt + 1))
                else:
                    raise

        return texts  # unreachable, but satisfies type checker
