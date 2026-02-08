"""SQLite cache for translations to avoid redundant API calls."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".modtranslator"
DEFAULT_CACHE_DB = DEFAULT_CACHE_DIR / "cache.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS translations (
    key TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    original TEXT NOT NULL,
    translated TEXT NOT NULL,
    backend TEXT NOT NULL DEFAULT 'deepl',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (key, target_lang)
);
"""


class TranslationCache:
    """Persistent SQLite cache mapping (key, target_lang) → translated text."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = DEFAULT_CACHE_DB
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def get(self, key: str, target_lang: str) -> str | None:
        """Look up a cached translation. Returns None if not found."""
        cursor = self._conn.execute(
            "SELECT translated FROM translations WHERE key = ? AND target_lang = ?",
            (key, target_lang),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_batch(self, keys: list[str], target_lang: str) -> dict[str, str]:
        """Look up multiple keys at once. Returns dict of found {key: translated}."""
        if not keys:
            return {}
        # SQLite has a limit of ~999 variables; chunk to stay well within it
        chunk_size = 900
        result: dict[str, str] = {}
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cursor = self._conn.execute(
                f"SELECT key, translated FROM translations "
                f"WHERE key IN ({placeholders}) AND target_lang = ?",
                [*chunk, target_lang],
            )
            result.update({row[0]: row[1] for row in cursor.fetchall()})
        return result

    def put(
        self, key: str, target_lang: str, original: str,
        translated: str, backend: str = "deepl",
    ) -> None:
        """Store a translation in the cache."""
        self._conn.execute(
            "INSERT OR REPLACE INTO translations (key, target_lang, original, translated, backend) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, target_lang, original, translated, backend),
        )
        self._conn.commit()

    def put_batch(
        self,
        entries: list[tuple[str, str, str, str]],
        backend: str = "deepl",
    ) -> None:
        """Store multiple translations. Each entry: (key, target_lang, original, translated)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO translations (key, target_lang, original, translated, backend) "
            "VALUES (?, ?, ?, ?, ?)",
            [(k, lang, orig, trans, backend) for k, lang, orig, trans in entries],
        )
        self._conn.commit()

    def count(self) -> int:
        """Return total number of cached translations."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM translations")
        return cursor.fetchone()[0]  # type: ignore[return-value]

    def clear(self) -> int:
        """Clear all cached translations. Returns number of entries deleted."""
        cursor = self._conn.execute("DELETE FROM translations")
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
