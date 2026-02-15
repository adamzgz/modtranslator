"""Parser for compiled Papyrus script files (.pex).

Parses the PEX binary format (big-endian) used by Skyrim SE to extract
and modify string literals. Only string literals that look like real
player-visible sentences/messages are candidates for translation.

Format reference: https://en.uesp.net/wiki/Skyrim_Mod:Compiled_Script_File_Format
"""

from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PEX_MAGIC = 0xFA57C0DE


@dataclass
class PexHeader:
    """PEX file header."""

    major_version: int
    minor_version: int
    game_id: int
    compilation_time: int
    source_name: str
    username: str
    machine_name: str


@dataclass
class PexFile:
    """Parsed PEX file with string table and raw sections."""

    header: PexHeader
    string_table: list[str]
    # Raw bytes of everything after the string table (debug + objects)
    post_table_data: bytes
    # Set of string table indices that are string literals (type 0x02)
    literal_indices: set[int] = field(default_factory=set)
    # Set of string table indices used as identifiers (type 0x01)
    ident_indices: set[int] = field(default_factory=set)

    def get_translatable_strings(self) -> dict[int, str]:
        """Return string table indices -> text for player-visible string literals.

        Only returns strings that are exclusively used as type 0x02 (string
        literal) and never as type 0x01 (identifier), AND pass conservative
        heuristic filters requiring real sentences.
        """
        # Pure literals: seen as 0x02 but NEVER as 0x01
        pure = self.literal_indices - self.ident_indices
        result = {}
        for idx in pure:
            text = self.string_table[idx]
            if _is_translatable_literal(text):
                result[idx] = text
        return result


def _is_translatable_literal(text: str) -> bool:
    """Conservative heuristic to determine if a string literal is player-visible.

    Errs heavily on the side of NOT translating to avoid breaking scripts.
    A string must look like a real sentence/message to pass.
    """
    # Empty or very short
    if len(text) < 4:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    # Papyrus keywords, type names, and common state/property names
    keywords = {
        "None", "String", "Int", "Float", "Bool", "true", "false",
        "conditional", "hidden", "auto", "autoReadOnly", "Active",
        "Inactive", "Done", "Ready", "Busy", "Open", "Close", "Closed",
        "Default", "Waiting", "Running", "Stopped", "Empty", "Player",
        "Health", "Stamina", "Magicka", "Quest", "Vampire", "Werewolf",
        "Dragon", "NPC", "Actor", "Weapon", "Armor", "Potion", "Spell",
        "Item", "Object", "Script", "Event", "State", "Function",
        "Property", "Variable", "Debug", "Notification", "MessageBox",
        "Game", "Utility", "Math", "Form", "Alias", "Keyword",
    }
    if stripped in keywords:
        return False

    # MCM/config keys: ALL_CAPS_WITH_UNDERSCORES
    if stripped.replace("_", "").replace(".", "").isupper() and "_" in stripped:
        return False

    # Internal identifiers: start with :: or $ or _
    if text.startswith("::") or text.startswith("$") or text.startswith("_"):
        return False

    # File paths / model references
    if "/" in text or "\\" in text:
        return False
    if text.endswith((".nif", ".dds", ".wav", ".psc", ".pex", ".esp", ".esm")):
        return False

    # Skeleton/bone node names: "NPC R Hand", "NPC L Foot", "NPC Pelvis [Pelv]"
    if stripped.startswith("NPC ") and len(stripped) < 30:
        return False

    # Papyrus auto-generated state function comments
    if stripped.startswith("Function that ") and ("state" in stripped or "object" in stripped):
        return False

    # Animation/bone style codes: "CME BreastMagic L", "MOV WeaponBackSWP", "HDT TailBone05"
    if re.match(r"^(?:CME|MOV|BOLT|BOW|AXE|HDT)\s", stripped):
        return False

    # Warpaint texture codes: "WP 018 Grass Body"
    if re.match(r"^WP\s+\d{3}\s", stripped):
        return False

    # SkyUI/MCM internal errors: "AddTextOptionST has been called", "Option type mismatch"
    if re.match(r"^(?:Add|Set|Force)\w+Option", stripped):
        return False
    if stripped.startswith("Option type mismatch"):
        return False

    # Animation sequence traces: "Animating: SEQ:"
    if stripped.startswith("Animating:"):
        return False

    # Debug/log strings with [ModTag] prefix: "[SOS] ...", "[BCD-CLWA] ..."
    if stripped.startswith("[") and "]" in stripped[:30]:
        return False

    # Contains function call syntax: "OnUpdate()", "RestartMod() called"
    if "()" in stripped:
        return False

    # Contains code comparison operators: "== None", "&& ", "|| "
    if any(op in stripped for op in (" == ", " != ", " && ", " || ", " >= ", " <= ")):
        return False

    # Contains Papyrus variable references: kFollower1, akActor, etc.
    if re.search(r"\b[ak]{1,2}[A-Z][A-Za-z0-9]+\b", stripped):
        return False

    # ERROR: log messages
    if "ERROR:" in stripped or "ERROR :" in stripped:
        return False

    # Debug separator markers: "=== Loading Cache ===" or "*** RESET ***"
    if "===" in stripped or "***" in stripped:
        return False

    # Event handler traces: "OnInit", "OnUpdate", "OnLoad"
    if re.search(r"\bOn(?:Init|Update|Load|GameReload)\b", stripped):
        return False

    # FormID references
    if "FormID" in stripped:
        return False

    # Method call patterns: ".IsBleedingOut()", ".GetState()", "iWidgets.setZoom()"
    if re.search(r"\.[A-Z][a-zA-Z]+\(", stripped):
        return False

    # API-style calls: "iWidgets.doTransition", "iWidgets.setRGB"
    if re.search(r"\b\w+\.\w+\(", stripped):
        return False

    # Internal function/class name prefix before colon: "AnimateMyLover: Serana"
    # CamelCase word(s) with no space before the colon = code trace, not player text
    if re.match(r"^[A-Z][a-zA-Z0-9]*(?:[A-Z][a-z]+)+:", stripped):
        return False

    # Must contain at least one letter
    if not any(c.isalpha() for c in text):
        return False

    # CRITICAL: Must contain a space to be a real sentence/message.
    # Single words are almost always identifiers, property names,
    # state names, or variable values — too risky to translate.
    if " " not in text and "\n" not in text:
        return False

    # Must have at least 3 words OR be > 15 chars to be a real message
    words = text.split()
    if len(words) < 3 and len(text) < 15:
        return False

    # Reject if looks like a code identifier pattern (camelCase, PascalCase without spaces)
    return not (re.match(r"^[A-Za-z][A-Za-z0-9_.]*$", stripped) and " " not in stripped)


def _read_wstring(data: bytes, pos: int) -> tuple[str, int]:
    """Read a big-endian length-prefixed string. Returns (string, new_pos)."""
    length = struct.unpack_from(">H", data, pos)[0]
    pos += 2
    text = data[pos:pos + length].decode("utf-8", errors="replace")
    pos += length
    return text, pos


def parse_pex(data: bytes) -> PexFile:
    """Parse a compiled Papyrus script (.pex) file.

    Extracts the header, string table, and scans the post-table bytecode
    to identify which string table entries are used as string literals
    (type 0x02) vs identifiers (type 0x01) in Variable Data entries.
    """
    pos = 0

    # Magic number
    magic = struct.unpack_from(">I", data, pos)[0]
    pos += 4
    if magic != PEX_MAGIC:
        raise ValueError(f"Not a PEX file: magic {magic:#010x} != {PEX_MAGIC:#010x}")

    # Version
    major = data[pos]
    pos += 1
    minor = data[pos]
    pos += 1

    # Game ID
    game_id = struct.unpack_from(">H", data, pos)[0]
    pos += 2

    # Compilation time
    comp_time = struct.unpack_from(">Q", data, pos)[0]
    pos += 8

    # Source filename, username, machine name
    source_name, pos = _read_wstring(data, pos)
    username, pos = _read_wstring(data, pos)
    machine_name, pos = _read_wstring(data, pos)

    header = PexHeader(
        major_version=major,
        minor_version=minor,
        game_id=game_id,
        compilation_time=comp_time,
        source_name=source_name,
        username=username,
        machine_name=machine_name,
    )

    # String table
    str_count = struct.unpack_from(">H", data, pos)[0]
    pos += 2

    string_table: list[str] = []
    for _ in range(str_count):
        s, pos = _read_wstring(data, pos)
        string_table.append(s)

    table_end = pos
    post_table_data = data[table_end:]

    # Scan post-table data for Variable Data type markers
    # Type 0x01 = identifier, Type 0x02 = string literal
    # Both followed by uint16 big-endian index into string table
    literal_indices: set[int] = set()
    ident_indices: set[int] = set()

    for p in range(len(post_table_data) - 2):
        type_byte = post_table_data[p]
        if type_byte in (0x01, 0x02):
            idx = struct.unpack_from(">H", post_table_data, p + 1)[0]
            if idx < str_count:
                if type_byte == 0x02:
                    literal_indices.add(idx)
                elif type_byte == 0x01:
                    ident_indices.add(idx)

    return PexFile(
        header=header,
        string_table=string_table,
        post_table_data=post_table_data,
        literal_indices=literal_indices,
        ident_indices=ident_indices,
    )


def serialize_pex(pex: PexFile) -> bytes:
    """Serialize a PexFile back to binary format.

    Only the header and string table are rewritten; the post-table data
    (debug info + objects) is copied verbatim since it uses index-based
    references that remain valid when string contents change.
    """
    parts: list[bytes] = []

    # Magic
    parts.append(struct.pack(">I", PEX_MAGIC))

    # Version
    parts.append(struct.pack(">BB", pex.header.major_version, pex.header.minor_version))

    # Game ID
    parts.append(struct.pack(">H", pex.header.game_id))

    # Compilation time
    parts.append(struct.pack(">Q", pex.header.compilation_time))

    # Source, user, machine
    for s in (pex.header.source_name, pex.header.username, pex.header.machine_name):
        encoded = s.encode("utf-8")
        parts.append(struct.pack(">H", len(encoded)))
        parts.append(encoded)

    # String table
    parts.append(struct.pack(">H", len(pex.string_table)))
    for s in pex.string_table:
        encoded = s.encode("utf-8")
        parts.append(struct.pack(">H", len(encoded)))
        parts.append(encoded)

    # Everything else verbatim
    parts.append(pex.post_table_data)

    return b"".join(parts)


def load_pex(path: str | Path) -> PexFile:
    """Load and parse a .pex file from disk."""
    path = Path(path)
    data = path.read_bytes()
    return parse_pex(data)


def save_pex(pex: PexFile, path: str | Path) -> None:
    """Serialize and save a .pex file to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_pex(pex))
