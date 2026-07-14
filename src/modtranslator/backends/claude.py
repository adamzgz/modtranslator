"""Anthropic Claude translation backend."""

from __future__ import annotations

import os
import time

from modtranslator.backends.base import TranslationBackend

MAX_BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 2.0

LANG_NAMES = {
    "ES": "Spanish (Spain)",
    "EN": "English",
    "FR": "French",
    "DE": "German",
    "IT": "Italian",
    "PT": "Portuguese (Brazil)",
    "RU": "Russian",
    "PL": "Polish",
    "ZH": "Chinese (Simplified)",
    "JA": "Japanese",
    "KO": "Korean",
}


class ClaudeBackend(TranslationBackend):
    """Translation backend using the Anthropic Claude API."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Claude backend requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from None
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Use --api-key or set ANTHROPIC_API_KEY."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        if not texts:
            return []
        results: list[str] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            results.extend(self._translate_chunk(batch, target_lang, source_lang))
        return results

    def _translate_chunk(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None,
    ) -> list[str]:
        target_name = LANG_NAMES.get(target_lang.upper(), target_lang)
        source_name = LANG_NAMES.get((source_lang or "EN").upper(), source_lang or "English")

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = (
            f"Translate the following Minecraft mod UI strings "
            f"from {source_name} to {target_name}.\n"
            "Rules:\n"
            "- Keep format specifiers exactly as-is: %s, %1$s, %2$s, %d, etc.\n"
            "- Keep color codes exactly as-is: §a, §6, §r, etc.\n"
            "- Keep placeholders like FMx{0}, SCx{1} unchanged.\n"
            "- Translate only the human-readable text, nothing else.\n"
            "- Respond ONLY with numbered lines matching the input, no extra text.\n\n"
            f"{numbered}"
        )

        from anthropic.types import TextBlock

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(
                    block.text for block in response.content if isinstance(block, TextBlock)
                ).strip()
                return self._parse_response(raw, texts)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise RuntimeError(f"Claude API error after {MAX_RETRIES} attempts: {e}") from e
        return texts

    def _parse_response(self, raw: str, originals: list[str]) -> list[str]:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        results: list[str] = []
        for i, original in enumerate(originals):
            prefix = f"{i+1}."
            matched = next(
                (line[len(prefix):].strip() for line in lines if line.startswith(prefix)), None
            )
            results.append(matched if matched else original)
        return results
