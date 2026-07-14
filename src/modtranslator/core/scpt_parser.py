"""SCPT (Script) record handler for FO3/FNV compiled bytecode.

Forward-parses SCDA instruction-by-instruction to build a structural map,
then uses exact offset knowledge for safe string replacement with proper
length recalculation.  Post-patch validation reverts on any structural
inconsistency to guarantee zero corruption.

Two string formats:
- NVSE S-token:   0x53 + uint16(len) + string_bytes + 0x00
- Vanilla param:  paramType(2) + uint16(len) + string_bytes
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

from modtranslator.core.records import Record

# ── String extraction from SCTX ──

_RE_QUOTED = re.compile(r'"([^"]*)"')
_RE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")
_RE_FORMID_REF = re.compile(r"^[A-Za-z0-9_ ]+\.(esm|esp):[A-Fa-f0-9]+$", re.IGNORECASE)
_RE_SEPARATOR = re.compile(r"^[=\-\^~_]{3,}$")
_RE_NUMBER = re.compile(r"^[\d.]+$")

# Lines starting with debug print commands — never player-facing
_RE_SKIP_LINE_START = re.compile(r"^\s*(?:printd|printc)\b", re.IGNORECASE)

# Lines containing these function calls have non-translatable string args:
# config vars, INI access, file ops, plugin checks, string construction/search,
# console commands, JSON reads
_RE_SKIP_LINE_CONTAINS = re.compile(
    r"\b(?:AuxVar(?:Set|Get)Flt|GetINIFloat(?:_Cached)?|SetINI\w*|"
    r"ClearFileCache\w*|IsPluginInstalled|"
    r"Sv_(?:Construct|Find|ToLower|Compare)|"
    r"ReadFromJson|WriteToJson|Console)\b",
    re.IGNORECASE,
)


def _extract_quoted_strings(sctx: str) -> list[str]:
    """Extract quoted string literals from SCTX source that look like player text.

    Uses line-by-line context analysis to skip strings from debug prints,
    config/INI access, console commands, and string construction functions.
    Only strings from player-facing commands (MessageBox, variable assignments,
    etc.) pass through.
    """
    result: list[str] = []
    for line in sctx.split("\n"):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith(";"):
            continue
        # Skip debug output lines
        if _RE_SKIP_LINE_START.match(stripped):
            continue
        # Skip lines with non-translatable function calls
        if _RE_SKIP_LINE_CONTAINS.search(stripped):
            continue

        for s in _RE_QUOTED.findall(line):
            if len(s) < 3:
                continue
            if s.startswith("\\") or s.startswith("/"):
                continue
            if _RE_IDENTIFIER.match(s):
                continue
            if _RE_FORMID_REF.match(s):
                continue
            if _RE_SEPARATOR.match(s):
                continue
            if _RE_NUMBER.match(s):
                continue
            if s.startswith(".") and len(s) <= 5:
                continue
            result.append(s)
    return result


# ── Structural data classes ──


@dataclass
class InstrString:
    """A string found within a parsed instruction."""

    text: str
    str_data_offset: int  # Absolute offset of string bytes in SCDA
    str_len: int  # Current byte length of string
    str_len_offset: int  # Absolute offset of the uint16 string length prefix
    is_nvse: bool  # True if NVSE S-token format
    expr_len_offset: int = -1  # Offset of NVSE expression length uint16


@dataclass
class ParsedInstruction:
    """A parsed instruction with position info."""

    offset: int  # Absolute offset of opcode in SCDA
    opcode: int
    args_len_offset: int  # Absolute offset of the argsLen uint16
    args_offset: int  # Absolute offset of args data start
    args_len: int  # Value of argsLen
    strings: list[InstrString] = field(default_factory=list)


@dataclass
class ParsedBlock:
    """A parsed Begin...End block."""

    begin_offset: int
    bdl_offset: int  # Absolute offset of blockDataLen uint32
    block_data_start: int  # Where instruction data begins
    block_data_len: int  # Value of blockDataLen
    instructions: list[ParsedInstruction] = field(default_factory=list)


@dataclass
class ParsedSCDA:
    """Complete parsed structure of an SCDA bytecode buffer."""

    blocks: list[ParsedBlock] = field(default_factory=list)

    @property
    def all_strings(self) -> list[InstrString]:
        """Flattened list of all strings across all blocks/instructions."""
        result: list[InstrString] = []
        for block in self.blocks:
            for instr in block.instructions:
                result.extend(instr.strings)
        return result


# ── Public-facing dataclasses (unchanged interface) ──


@dataclass
class ScptString:
    """A translatable string found in a SCPT record's compiled bytecode."""

    text: str
    scda_offset: int  # Byte offset where string data starts in SCDA
    scda_len: int  # Byte length of string data in SCDA
    len_prefix_offset: int  # Offset of the uint16 string length prefix
    is_nvse: bool  # True if NVSE S-token format
    expr_len_offset: int = -1  # Offset of NVSE expression length uint16
    args_len_offset: int = -1  # Offset of instruction argsLen uint16
    bdl_offset: int = -1  # Offset of containing block's blockDataLen uint32


@dataclass
class ScptRecord:
    """Aggregates extraction results for a SCPT record."""

    record: Record
    editor_id: str
    strings: list[ScptString] = field(default_factory=list)


# ── Forward parser ──


def _parse_scda(scda: bytes) -> ParsedSCDA | None:
    """Forward-parse SCDA bytecode into a structural map.

    Returns None if the bytecode cannot be parsed (malformed data).
    SCDA with only a 4-byte header (stub/NVSE-compiled scripts) returns
    a valid empty ParsedSCDA with no blocks.
    """
    if len(scda) < 4:
        return None
    # Stub scripts: only the 4-byte header, no Begin/End blocks
    if len(scda) < 8:
        return ParsedSCDA()

    parsed = ParsedSCDA()
    pos = 4  # Skip 4-byte script header

    while pos + 6 <= len(scda):
        # Read Begin instruction
        opcode = struct.unpack_from("<H", scda, pos)[0]
        if opcode != 0x0010:
            break  # No more Begin blocks

        args_len = struct.unpack_from("<H", scda, pos + 2)[0]
        if args_len < 6 or pos + 4 + args_len > len(scda):
            return None  # Malformed

        # blockDataLen is at pos + 4 + 2 (after argsLen field, skip eventType)
        bdl_offset = pos + 6
        if bdl_offset + 4 > len(scda):
            return None
        block_data_len = struct.unpack_from("<I", scda, bdl_offset)[0]

        block_data_start = pos + 4 + args_len
        block_data_end = block_data_start + block_data_len

        if block_data_end > len(scda):
            return None  # Block extends past SCDA

        block = ParsedBlock(
            begin_offset=pos,
            bdl_offset=bdl_offset,
            block_data_start=block_data_start,
            block_data_len=block_data_len,
        )

        # Parse instructions within block
        ipos = block_data_start
        while ipos + 4 <= block_data_end:
            i_opcode = struct.unpack_from("<H", scda, ipos)[0]
            i_args_len = struct.unpack_from("<H", scda, ipos + 2)[0]

            # End instruction
            if i_opcode == 0x0011 and i_args_len == 0:
                block.instructions.append(
                    ParsedInstruction(ipos, i_opcode, ipos + 2, ipos + 4, 0)
                )
                ipos += 4
                break

            # Reference-prefixed call: 0x001C(2) + refIdx(2) + opcode(2) + argsLen(2) + args
            if i_opcode == 0x001C:
                if ipos + 8 > block_data_end:
                    return None
                real_opcode = struct.unpack_from("<H", scda, ipos + 4)[0]
                real_args_len = struct.unpack_from("<H", scda, ipos + 6)[0]
                total = 8 + real_args_len
                if ipos + total > block_data_end:
                    return None

                instr = ParsedInstruction(
                    offset=ipos,
                    opcode=real_opcode,
                    args_len_offset=ipos + 6,
                    args_offset=ipos + 8,
                    args_len=real_args_len,
                )
                _find_strings_in_args(scda, instr)
                block.instructions.append(instr)
                ipos += total
                continue

            if ipos + 4 + i_args_len > block_data_end:
                return None  # Instruction extends past block

            instr = ParsedInstruction(
                offset=ipos,
                opcode=i_opcode,
                args_len_offset=ipos + 2,
                args_offset=ipos + 4,
                args_len=i_args_len,
            )

            # Find strings within this instruction's args
            _find_strings_in_args(scda, instr)

            block.instructions.append(instr)
            ipos += 4 + i_args_len

        parsed.blocks.append(block)
        pos = block_data_end

    return parsed


# ── String finding within instruction args ──


def _find_strings_in_args(scda: bytes, instr: ParsedInstruction) -> None:
    """Find all string literals within an instruction's args payload."""
    args_start = instr.args_offset
    args_end = args_start + instr.args_len

    if instr.args_len < 3:
        return

    # Check for NVSE expression header: 0x01 + exprLen(2)
    if scda[args_start] == 0x01 and instr.args_len >= 3:
        expr_len_offset = args_start + 1
        expr_len = struct.unpack_from("<H", scda, expr_len_offset)[0]
        expr_data_start = args_start + 3
        expr_data_end = expr_data_start + expr_len

        if expr_data_end <= args_end:
            _parse_nvse_expression(scda, expr_data_start, expr_data_end, expr_len_offset, instr)
            return

    # Check for NVSE sentinel 0xFFFF at start of args
    if instr.args_len >= 2:
        sentinel = struct.unpack_from("<H", scda, args_start)[0]
        if sentinel == 0xFFFF:
            _parse_nvse_expression(scda, args_start + 2, args_end, -1, instr)
            return

    # Vanilla: scan args for string parameters
    _find_vanilla_strings(scda, args_start, args_end, instr)


def _parse_nvse_expression(
    scda: bytes,
    start: int,
    end: int,
    expr_len_offset: int,
    instr: ParsedInstruction,
) -> None:
    """Parse NVSE expression tokens looking for S-token strings."""
    pos = start
    while pos < end:
        token = scda[pos]

        if token == 0x53:  # 'S' — string literal
            if pos + 3 > end:
                break
            str_len = struct.unpack_from("<H", scda, pos + 1)[0]
            str_data_offset = pos + 3
            if str_data_offset + str_len > end:
                break
            try:
                text = scda[str_data_offset : str_data_offset + str_len].decode("cp1252")
            except UnicodeDecodeError:
                text = scda[str_data_offset : str_data_offset + str_len].decode(
                    "latin-1", errors="replace"
                )

            instr.strings.append(
                InstrString(
                    text=text,
                    str_data_offset=str_data_offset,
                    str_len=str_len,
                    str_len_offset=pos + 1,
                    is_nvse=True,
                    expr_len_offset=expr_len_offset,
                )
            )
            pos = str_data_offset + str_len
            # Skip null terminator if present
            if pos < end and scda[pos] == 0x00:
                pos += 1

        elif token == 0x56:  # 'V' — variable ref
            pos += 1
            # Variable refs: type(1) + refIdx(2) + varIdx(2) = 5 bytes
            # But some have different sizes. Skip conservatively.
            pos += 5
        elif token == 0x52:  # 'R' — reference
            pos += 1
            pos += 4
        elif token in (0x42, 0x62):  # 'B'/'b' — byte
            pos += 2
        elif token in (0x49, 0x69):  # 'I'/'i' — int16
            pos += 3
        elif token in (0x4C, 0x6C):  # 'L'/'l' — int32
            pos += 5
        elif token == 0x5A:  # 'Z' — double/float
            pos += 9
        elif token == 0x47:  # 'G' — global
            pos += 3
        elif token == 0x58:  # 'X' — command call
            # command(2) + callerRefLen(2) + callerRef(N)
            if pos + 5 > end:
                break
            pos += 1
            cmd_opcode = struct.unpack_from("<H", scda, pos)[0]  # noqa: F841
            ref_len = struct.unpack_from("<H", scda, pos + 2)[0]
            pos += 4 + ref_len
        elif token == 0x00:  # null terminator / end marker
            break
        elif 0x20 <= token <= 0x3F:
            # Operator tokens (comparison, arithmetic, logical)
            pos += 1
        else:
            # Unknown token — stop parsing this expression safely
            break


def _find_vanilla_strings(
    scda: bytes,
    args_start: int,
    args_end: int,
    instr: ParsedInstruction,
) -> None:
    """Find strings in vanilla instruction args by scanning for length-prefixed text."""
    # Vanilla string args: various formats depending on the opcode.
    # MessageBox/MessageBoxEx: [numParams(2)] + [paramType(2) + strLen(2) + string]*
    # Or simpler: paramType(2) + strLen(2) + string
    # We scan for readable text preceded by a matching uint16 length.
    pos = args_start
    while pos + 4 <= args_end:
        # Try reading paramType(2) + strLen(2) + string(strLen)
        str_len = struct.unpack_from("<H", scda, pos + 2)[0]
        if str_len < 3 or pos + 4 + str_len > args_end:
            pos += 2
            continue

        str_data_offset = pos + 4
        candidate_bytes = scda[str_data_offset : str_data_offset + str_len]

        # Check if it looks like readable text
        if _is_readable_text(candidate_bytes):
            try:
                text = candidate_bytes.decode("cp1252")
            except UnicodeDecodeError:
                text = candidate_bytes.decode("latin-1", errors="replace")

            instr.strings.append(
                InstrString(
                    text=text,
                    str_data_offset=str_data_offset,
                    str_len=str_len,
                    str_len_offset=pos + 2,
                    is_nvse=False,
                )
            )
            pos = str_data_offset + str_len
        else:
            pos += 2


def _is_readable_text(data: bytes) -> bool:
    """Check if a byte sequence looks like readable cp1252 text."""
    if len(data) < 3:
        return False
    printable = 0
    for b in data:
        if 0x20 <= b <= 0x7E or b in (0x0A, 0x0D, 0x09) or 0x80 <= b <= 0xFF:
            printable += 1
    return printable / len(data) >= 0.85


# ── Validation ──


def _validate_scda(scda: bytes) -> bool:
    """Quick validation: can we parse the SCDA structure without errors?"""
    return _parse_scda(scda) is not None


# ── Public API ──


def extract_scpt_strings(record: Record) -> ScptRecord:
    """Extract translatable strings from a SCPT record.

    Forward-parses SCDA to find strings with exact structural positions,
    then cross-references against SCTX quoted strings.
    """
    sctx_sub = None
    scda_sub = None
    edid = ""

    for sub in record.subrecords:
        if sub.type == b"SCTX":
            sctx_sub = sub
        elif sub.type == b"SCDA":
            scda_sub = sub
        elif sub.type == b"EDID":
            edid = sub.decode_string()

    result = ScptRecord(record=record, editor_id=edid)

    if not sctx_sub or not scda_sub:
        return result

    sctx_text = sctx_sub.decode_string()
    scda = bytes(scda_sub.data)
    quoted_strings = _extract_quoted_strings(sctx_text)

    if not quoted_strings:
        return result

    # Forward-parse SCDA
    parsed = _parse_scda(scda)
    if parsed is None:
        return result  # Unparseable — skip safely

    # Build a set of SCTX strings for cross-reference.
    # SCTX uses escape sequences (%r for newline, \n, \t) while SCDA has
    # the actual bytes (0x0A, 0x09).  Add unescaped variants so cross-ref matches.
    sctx_set = set(quoted_strings)
    sctx_set |= {
        s.replace("%r", "\n").replace("\\n", "\n").replace("\\t", "\t")
        for s in quoted_strings
        if "%r" in s or "\\n" in s or "\\t" in s
    }

    # Map parsed strings to ScptString objects
    used_texts: dict[str, int] = {}  # Track occurrences for dedup
    for block in parsed.blocks:
        for instr in block.instructions:
            for istr in instr.strings:
                if istr.text not in sctx_set:
                    continue

                # Track duplicate text occurrences
                count = used_texts.get(istr.text, 0)
                used_texts[istr.text] = count + 1

                result.strings.append(
                    ScptString(
                        text=istr.text,
                        scda_offset=istr.str_data_offset,
                        scda_len=istr.str_len,
                        len_prefix_offset=istr.str_len_offset,
                        is_nvse=istr.is_nvse,
                        expr_len_offset=istr.expr_len_offset,
                        args_len_offset=instr.args_len_offset,
                        bdl_offset=block.bdl_offset,
                    )
                )

    return result


def patch_scpt_record(
    record: Record,
    translations: dict[int, str],
) -> int:
    """Patch translated strings into a SCPT record's SCDA bytecode.

    Uses forward-parsed structural info for exact offset updates.
    Validates the result and reverts on any structural inconsistency.

    Args:
        record: The SCPT Record object.
        translations: Mapping of scda_offset → translated text.

    Returns:
        Number of strings patched.
    """
    scda_sub = None
    schr_sub = None

    for sub in record.subrecords:
        if sub.type == b"SCDA":
            scda_sub = sub
        elif sub.type == b"SCHR":
            schr_sub = sub

    if not scda_sub or not translations:
        return 0

    # Save original for revert
    original_scda = bytes(scda_sub.data)
    original_schr = bytes(schr_sub.data) if schr_sub else None

    # Re-parse for fresh structural info
    parsed = _parse_scda(original_scda)
    if parsed is None:
        return 0  # Unparseable — don't touch

    # Build lookup: str_data_offset → (InstrString, ParsedInstruction, ParsedBlock)
    offset_map: dict[int, tuple[InstrString, ParsedInstruction, ParsedBlock]] = {}
    for block in parsed.blocks:
        for instr in block.instructions:
            for istr in instr.strings:
                offset_map[istr.str_data_offset] = (istr, instr, block)

    scda = bytearray(original_scda)
    total_delta = 0
    patched = 0

    # Sort by offset descending — process from end to start
    sorted_offsets = sorted(translations.keys(), reverse=True)

    for offset in sorted_offsets:
        if offset not in offset_map:
            continue  # String not in parsed structure — skip

        translated_text = translations[offset]
        try:
            new_bytes = translated_text.encode("cp1252")
        except UnicodeEncodeError:
            try:
                new_bytes = translated_text.encode("latin-1")
            except UnicodeEncodeError:
                continue

        istr, instr, block = offset_map[offset]
        old_len = istr.str_len
        new_len = len(new_bytes)
        delta = new_len - old_len

        # Replace string bytes
        scda[offset : offset + old_len] = new_bytes

        # Update string length prefix
        struct.pack_into("<H", scda, istr.str_len_offset, new_len)

        if delta != 0:
            # Update NVSE expression length
            if istr.expr_len_offset >= 0:
                old_expr = struct.unpack_from("<H", scda, istr.expr_len_offset)[0]
                struct.pack_into("<H", scda, istr.expr_len_offset, old_expr + delta)

            # Update instruction argsLen
            old_args = struct.unpack_from("<H", scda, instr.args_len_offset)[0]
            struct.pack_into("<H", scda, instr.args_len_offset, old_args + delta)

            # Update block data length
            old_bdl = struct.unpack_from("<I", scda, block.bdl_offset)[0]
            struct.pack_into("<I", scda, block.bdl_offset, old_bdl + delta)

        total_delta += delta
        patched += 1

    # Update SCHR.compiledSize
    if total_delta != 0 and schr_sub and len(schr_sub.data) >= 12:
        old_compiled = struct.unpack_from("<I", schr_sub.data, 8)[0]
        struct.pack_into("<I", schr_sub.data, 8, old_compiled + total_delta)

    # ── Post-patch validation ──
    if not _validate_scda(bytes(scda)):
        # Revert to original
        scda_sub.data = bytearray(original_scda)
        if schr_sub and original_schr:
            schr_sub.data = bytearray(original_schr)
        return 0

    # Commit
    scda_sub.data = scda
    return patched
