"""I/O utilities for safe file writing."""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write(target: Path, data: bytes) -> None:
    """Write *data* to *target* atomically via a temp file + os.replace()."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
