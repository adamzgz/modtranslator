"""Safe JAR (ZIP) file reading and writing for Minecraft mods.

Handles:
- Signed JAR detection (META-INF/*.SF, *.RSA, *.DSA)
- Lang file discovery inside JARs (assets/*/lang/*.json, *.lang)
- Full ZIP reconstruction (never append — always rebuild)
- Atomic write via temp file + os.replace()
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JarLangInfo:
    """Info about lang files found inside a JAR."""

    jar_path: Path
    mod_id: str  # from the asset path: assets/<mod_id>/lang/
    en_us_path: str  # internal ZIP path to en_us.json (or en_US.lang)
    en_us_content: str  # file content
    target_path: str | None = None  # internal ZIP path to existing target lang file
    target_content: str | None = None  # existing target file content
    format: str = "json"  # "json" or "lang"
    indent: str = "  "  # detected indent style


@dataclass
class JarScanResult:
    """Result of scanning a single JAR."""

    jar_path: Path
    is_signed: bool = False
    lang_entries: list[JarLangInfo] = field(default_factory=list)
    error: str | None = None


def is_jar_signed(zf: zipfile.ZipFile) -> bool:
    """Check if a JAR has signing files in META-INF/."""
    for name in zf.namelist():
        lower = name.lower()
        if lower.startswith("meta-inf/") and (
            lower.endswith(".sf") or lower.endswith(".rsa") or lower.endswith(".dsa")
        ):
            return True
    return False


def scan_jar(jar_path: Path, target_locale: str) -> JarScanResult:
    """Scan a JAR file for lang files.

    Args:
        jar_path: Path to the .jar file
        target_locale: Minecraft locale code (e.g. "es_es")

    Returns:
        JarScanResult with lang entries found.
    """
    result = JarScanResult(jar_path=jar_path)

    try:
        with zipfile.ZipFile(jar_path, "r") as zf:
            if is_jar_signed(zf):
                result.is_signed = True
                return result

            # Find all en_us.json files
            names = zf.namelist()
            en_us_files: list[str] = []
            for name in names:
                lower = name.lower()
                if (
                    "/lang/" in lower
                    and (lower.endswith("/en_us.json") or lower.endswith("/en_us.lang"))
                ):
                    en_us_files.append(name)

            for en_path in en_us_files:
                parts = en_path.split("/")
                # Expected: assets/<modid>/lang/<filename>
                if len(parts) < 4:
                    continue
                mod_id = parts[1]  # assets/<modid>/...
                is_json = en_path.lower().endswith(".json")
                fmt = "json" if is_json else "lang"

                try:
                    raw = zf.read(en_path)
                    content = raw.decode("utf-8-sig")
                except Exception:
                    continue

                # Detect indent for JSON files
                indent = "  "
                if is_json:
                    from modtranslator.core.mc_lang_parser import detect_indent
                    indent = detect_indent(content)

                # Look for existing target file
                lang_dir = "/".join(parts[:-1])  # assets/<modid>/lang
                target_name = (
                    f"{target_locale}.json" if is_json else f"{target_locale}.lang"
                )

                target_zip_path = f"{lang_dir}/{target_name}"
                target_content = None

                # Check for target (case-insensitive)
                for name in names:
                    if name.lower() == target_zip_path.lower():
                        try:
                            target_content = zf.read(name).decode("utf-8-sig")
                            target_zip_path = name  # preserve original casing
                        except Exception:
                            pass
                        break

                entry = JarLangInfo(
                    jar_path=jar_path,
                    mod_id=mod_id,
                    en_us_path=en_path,
                    en_us_content=content,
                    target_path=target_zip_path if target_content is not None else None,
                    target_content=target_content,
                    format=fmt,
                    indent=indent,
                )
                result.lang_entries.append(entry)

    except (zipfile.BadZipFile, OSError) as e:
        result.error = str(e)

    return result


def rebuild_jar_with_lang(
    jar_path: Path,
    new_files: dict[str, bytes],
    output_path: Path | None = None,
) -> None:
    """Rebuild a JAR replacing/adding lang files.

    Reads the original JAR, copies all entries, and replaces/adds entries
    from new_files. Writes to a temp file then atomically replaces.

    Args:
        jar_path: Original JAR file
        new_files: Dict of {internal_zip_path: file_bytes} to add/replace
        output_path: Where to write the result. Defaults to jar_path (in-place).
    """
    if output_path is None:
        output_path = jar_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    new_files_lower = {k.lower(): k for k in new_files}

    try:
        with (
            zipfile.ZipFile(jar_path, "r") as zf_in,
            zipfile.ZipFile(tmp_path, "w") as zf_out,
        ):
                for item in zf_in.infolist():
                    if item.filename.lower() in new_files_lower:
                        # Will be replaced
                        continue
                    # Copy with original metadata
                    data = zf_in.read(item.filename)
                    zf_out.writestr(item, data)

                # Write new/replaced files
                for zip_path, content in new_files.items():
                    zf_out.writestr(zip_path, content)

        os.replace(tmp_path, output_path)

    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
