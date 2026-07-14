"""Tests for SCPT (Script) record bytecode string extraction and patching."""

from __future__ import annotations

import struct

from modtranslator.core.records import Record, Subrecord
from modtranslator.core.scpt_parser import (
    _extract_quoted_strings,
    _parse_scda,
    extract_scpt_strings,
    patch_scpt_record,
)


# ── Helpers ──


def _make_schr(compiled_size: int) -> bytearray:
    """Build minimal SCHR: unused(4) + refCount(4) + compiledSize(4) + varCount(4) + type(2) + flags(2)."""
    return bytearray(struct.pack("<4sIIIHH", b"\x00" * 4, 0, compiled_size, 0, 0, 0))


def _make_nvse_scda(strings: list[str], event_type: int = 0x0000) -> bytearray:
    """Build synthetic SCDA with NVSE S-token strings (Let instructions)."""
    instr_data = bytearray()
    for s in strings:
        s_bytes = s.encode("cp1252")
        s_len = len(s_bytes)
        # NVSE expression: V-token(6) + S-token(1+2+N+1)
        expr_body = (
            b"\x56\x02\x00\x00\x02\x00"
            + b"\x53"
            + struct.pack("<H", s_len)
            + s_bytes
            + b"\x00"
        )
        expr_len = len(expr_body)
        args = b"\x01" + struct.pack("<H", expr_len) + expr_body
        # Let opcode: 0x1539
        instr = struct.pack("<HH", 0x1539, len(args)) + args
        instr_data += instr

    # End instruction
    instr_data += struct.pack("<HH", 0x0011, 0)

    block_data_len = len(instr_data)
    block_header = struct.pack("<HHHI", 0x0010, 6, event_type, block_data_len)
    header = struct.pack("<I", 0x0000001D)
    return bytearray(header + block_header + instr_data)


def _make_vanilla_scda(strings: list[str], opcode: int = 0x14DC) -> bytearray:
    """Build synthetic SCDA with vanilla messageboxex strings (multi-param)."""
    # All strings in ONE instruction as separate params
    params = bytearray()
    for s in strings:
        s_bytes = s.encode("cp1252")
        params += struct.pack("<HH", 0x0001, len(s_bytes)) + s_bytes
    instr = struct.pack("<HH", opcode, len(params)) + params

    end_instr = struct.pack("<HH", 0x0011, 0)
    block_data = instr + end_instr
    block_header = struct.pack("<HHHI", 0x0010, 6, 0x0000, len(block_data))
    header = struct.pack("<I", 0x0000001D)
    return bytearray(header + block_header + block_data)


def _make_sctx(strings: list[str]) -> str:
    lines = ["scn TestScript", "string_var sMSG", "begin function {}"]
    for s in strings:
        lines.append(f'    let sMSG := "{s}"')
    lines.append("end")
    return "\n".join(lines)


def _make_scpt_record(sctx: str, scda: bytearray, edid: str = "TestScript") -> Record:
    return Record(
        type=b"SCPT",
        flags=0,
        form_id=0x00001000,
        vcs1=0,
        vcs2=0,
        subrecords=[
            Subrecord(type=b"EDID", data=bytearray(edid.encode("cp1252") + b"\x00")),
            Subrecord(type=b"SCHR", data=_make_schr(len(scda))),
            Subrecord(type=b"SCDA", data=scda),
            Subrecord(type=b"SCTX", data=bytearray(sctx.encode("cp1252") + b"\x00")),
        ],
    )


# ── Tests: _extract_quoted_strings ──


class TestExtractQuotedStrings:
    def test_basic_extraction(self):
        sctx = 'let sMSG := "Hello world"\nlet sMSG2 := "Another string"'
        assert _extract_quoted_strings(sctx) == ["Hello world", "Another string"]

    def test_skips_short_strings(self):
        assert _extract_quoted_strings('let x := "ab"\nlet y := "Hello world"') == ["Hello world"]

    def test_skips_identifiers(self):
        assert _extract_quoted_strings('let x := "MyVariable"\nlet y := "Hello world"') == [
            "Hello world"
        ]

    def test_skips_formid_refs(self):
        assert _extract_quoted_strings('let x := "fallout3.esm:0C3A1D"') == []

    def test_skips_separators(self):
        assert _extract_quoted_strings('let x := "======"') == []

    def test_skips_paths(self):
        assert _extract_quoted_strings('let x := "\\path\\to\\file"') == []

    def test_empty_sctx(self):
        assert _extract_quoted_strings("scn TestScript\nbegin gamemode\nend") == []

    def test_skips_printd_lines(self):
        sctx = 'printd "Debug message here"\nMessageBoxEx "Player text"'
        assert _extract_quoted_strings(sctx) == ["Player text"]

    def test_skips_printc_lines(self):
        sctx = 'printc "Console output"\nMessageBoxEx "Visible text"'
        assert _extract_quoted_strings(sctx) == ["Visible text"]

    def test_skips_auxvar_lines(self):
        sctx = 'Player.AuxVarSetFlt "*iChance" (GetINIFloat_Cached "Main:iChance" "Lime/Enc.ini")'
        assert _extract_quoted_strings(sctx) == []

    def test_skips_getini_lines(self):
        sctx = 'GetINIFloat_Cached "Debug:bDebug" "Lime/Encounters.ini"'
        assert _extract_quoted_strings(sctx) == []

    def test_skips_console_lines(self):
        sctx = 'Console ("Player.DamageAv Health " + $val)'
        assert _extract_quoted_strings(sctx) == []

    def test_skips_sv_construct_lines(self):
        sctx = 'Sv_Construct "Check%g" iBtn'
        assert _extract_quoted_strings(sctx) == []

    def test_skips_sv_find_lines(self):
        sctx = 'if Sv_Find "damageav health" sOutcome > -1'
        assert _extract_quoted_strings(sctx) == []

    def test_skips_is_plugin_installed(self):
        sctx = 'if IsPluginInstalled "JIP NVSE Plugin"'
        assert _extract_quoted_strings(sctx) == []

    def test_skips_read_from_json(self):
        sctx = 'Let aEvent := ReadFromJson "data\\config\\file.json"'
        assert _extract_quoted_strings(sctx) == []

    def test_skips_comment_lines(self):
        sctx = '; "This is a comment with quotes"\nMessageBoxEx "Real text"'
        assert _extract_quoted_strings(sctx) == ["Real text"]

    def test_keeps_messagebox_strings(self):
        sctx = 'MessageBoxEx "You find some items."\nMessageBoxExAlt ScriptRef "^Title^Body|Ok"'
        assert _extract_quoted_strings(sctx) == ["You find some items.", "^Title^Body|Ok"]

    def test_keeps_let_assignments(self):
        sctx = 'let Game := "Dice Roll"'
        assert _extract_quoted_strings(sctx) == ["Dice Roll"]

    def test_skips_clearfilecache(self):
        sctx = 'ClearFileCacheShowOff "Lime/Encounters.ini" 0'
        assert _extract_quoted_strings(sctx) == []


# ── Tests: _parse_scda ──


class TestParseSCDA:
    def test_single_block(self):
        scda = _make_nvse_scda(["test string here"])
        parsed = _parse_scda(bytes(scda))
        assert parsed is not None
        assert len(parsed.blocks) == 1
        # Should have 2 instructions: Let + End
        assert len(parsed.blocks[0].instructions) == 2

    def test_finds_nvse_string(self):
        scda = _make_nvse_scda(["Hello from the wasteland"])
        parsed = _parse_scda(bytes(scda))
        assert parsed is not None
        strings = parsed.all_strings
        assert len(strings) == 1
        assert strings[0].text == "Hello from the wasteland"
        assert strings[0].is_nvse is True

    def test_finds_multiple_nvse_strings(self):
        texts = ["First message", "Second message"]
        scda = _make_nvse_scda(texts)
        parsed = _parse_scda(bytes(scda))
        assert parsed is not None
        strings = parsed.all_strings
        assert len(strings) == 2
        assert strings[0].text == texts[0]
        assert strings[1].text == texts[1]

    def test_finds_vanilla_strings(self):
        scda = _make_vanilla_scda(["You found the key!"])
        parsed = _parse_scda(bytes(scda))
        assert parsed is not None
        strings = parsed.all_strings
        assert len(strings) == 1
        assert strings[0].text == "You found the key!"
        assert strings[0].is_nvse is False

    def test_multi_param_vanilla(self):
        """Multiple strings in ONE messageboxex instruction."""
        texts = ["Choose an option:", "Option A button", "Option B button"]
        scda = _make_vanilla_scda(texts)
        parsed = _parse_scda(bytes(scda))
        assert parsed is not None
        strings = parsed.all_strings
        assert len(strings) == 3
        # All strings should share the same instruction (same args_len_offset)
        instrs = parsed.blocks[0].instructions
        # First instruction (messageboxex) should have all 3 strings
        assert len(instrs[0].strings) == 3

    def test_stub_scda_returns_empty(self):
        """4-byte SCDA (header only, no blocks) is a valid stub script."""
        result = _parse_scda(b"\x00\x01\x02\x03")
        assert result is not None
        assert len(result.blocks) == 0

    def test_returns_none_for_too_short(self):
        assert _parse_scda(b"\x00\x01") is None

    def test_block_data_len_correct(self):
        scda = _make_nvse_scda(["test string here"])
        parsed = _parse_scda(bytes(scda))
        assert parsed is not None
        block = parsed.blocks[0]
        bdl = struct.unpack_from("<I", scda, block.bdl_offset)[0]
        expected = len(scda) - block.block_data_start
        assert bdl == expected


# ── Tests: extract_scpt_strings ──


class TestExtractScptStrings:
    def test_nvse_s_token(self):
        strings = ["Hello world from the wasteland"]
        sctx = _make_sctx(strings)
        scda = _make_nvse_scda(strings)
        record = _make_scpt_record(sctx, scda)

        result = extract_scpt_strings(record)
        assert result.editor_id == "TestScript"
        assert len(result.strings) == 1
        ss = result.strings[0]
        assert ss.text == "Hello world from the wasteland"
        assert ss.is_nvse is True
        assert ss.args_len_offset >= 0
        assert ss.bdl_offset >= 0

    def test_filters_non_sctx_strings(self):
        """Only strings that appear in SCTX are returned."""
        sctx = 'scn Test\nbegin gamemode\n    let x := "Hello world"\nend'
        # SCDA contains extra strings not in SCTX
        scda = _make_nvse_scda(["Hello world", "Internal only string"])
        record = _make_scpt_record(sctx, scda)

        result = extract_scpt_strings(record)
        assert len(result.strings) == 1
        assert result.strings[0].text == "Hello world"

    def test_no_sctx(self):
        record = Record(
            type=b"SCPT", flags=0, form_id=0x1000, vcs1=0, vcs2=0,
            subrecords=[Subrecord(type=b"SCDA", data=bytearray(b"\x00" * 20))],
        )
        result = extract_scpt_strings(record)
        assert len(result.strings) == 0

    def test_unparseable_scda(self):
        """Malformed SCDA returns empty result safely."""
        sctx = 'scn Test\nbegin gamemode\n    let x := "Hello world"\nend'
        scda = bytearray(b"\x00" * 50)
        record = _make_scpt_record(sctx, scda)
        result = extract_scpt_strings(record)
        assert len(result.strings) == 0


# ── Tests: patch_scpt_record ──


class TestPatchScptRecord:
    def test_same_length_replacement(self):
        original = "Hello world from wasteland"
        translated = "Hola mundo de yermo  seco!"
        assert len(original) == len(translated)

        sctx = _make_sctx([original])
        scda = _make_nvse_scda([original])
        record = _make_scpt_record(sctx, scda)

        extracted = extract_scpt_strings(record)
        offset = extracted.strings[0].scda_offset

        count = patch_scpt_record(record, {offset: translated})
        assert count == 1

        scda_after = bytes(record.subrecords[2].data)
        assert translated.encode("cp1252") in scda_after
        assert original.encode("cp1252") not in scda_after

        schr = record.subrecords[1].data
        assert struct.unpack_from("<I", schr, 8)[0] == len(scda_after)

    def test_shorter_replacement(self):
        original = "Through shrewd negotiations you won"
        translated = "Ganaste las negociaciones"

        sctx = _make_sctx([original])
        scda = _make_nvse_scda([original])
        record = _make_scpt_record(sctx, scda)
        original_size = len(scda)

        extracted = extract_scpt_strings(record)
        offset = extracted.strings[0].scda_offset

        count = patch_scpt_record(record, {offset: translated})
        assert count == 1

        scda_after = bytes(record.subrecords[2].data)
        assert translated.encode("cp1252") in scda_after
        delta = len(translated.encode("cp1252")) - len(original.encode("cp1252"))
        assert len(scda_after) == original_size + delta
        assert struct.unpack_from("<I", record.subrecords[1].data, 8)[0] == len(scda_after)

    def test_longer_replacement(self):
        original = "You won the fight"
        translated = "Has ganado la pelea contra el enemigo del yermo"

        sctx = _make_sctx([original])
        scda = _make_nvse_scda([original])
        record = _make_scpt_record(sctx, scda)
        original_size = len(scda)

        extracted = extract_scpt_strings(record)
        offset = extracted.strings[0].scda_offset

        count = patch_scpt_record(record, {offset: translated})
        assert count == 1

        scda_after = bytes(record.subrecords[2].data)
        assert translated.encode("cp1252") in scda_after
        delta = len(translated.encode("cp1252")) - len(original.encode("cp1252"))
        assert len(scda_after) == original_size + delta

    def test_multiple_replacements_nvse(self):
        originals = ["First message to player", "Second message to player"]
        translateds = ["Primer mensaje al jugador", "Segundo mensaje al jugador"]

        sctx = _make_sctx(originals)
        scda = _make_nvse_scda(originals)
        record = _make_scpt_record(sctx, scda)
        original_size = len(scda)

        extracted = extract_scpt_strings(record)
        assert len(extracted.strings) == 2

        trans_map = {}
        total_delta = 0
        for ss, translated in zip(extracted.strings, translateds):
            trans_map[ss.scda_offset] = translated
            total_delta += len(translated.encode("cp1252")) - ss.scda_len

        count = patch_scpt_record(record, trans_map)
        assert count == 2

        scda_after = bytes(record.subrecords[2].data)
        for t in translateds:
            assert t.encode("cp1252") in scda_after
        assert len(scda_after) == original_size + total_delta

    def test_multi_param_vanilla_replacement(self):
        """Replace multiple strings in the same messageboxex instruction."""
        originals = ["Choose an option:", "Option A button", "Option B button"]
        translateds = ["Elige una opcion:", "Opcion A boton!!", "Opcion B boton!!"]

        sctx = (
            'scn T\nbegin gamemode\n'
            f'    messageboxex "{originals[0]}" "{originals[1]}" "{originals[2]}"\nend'
        )
        scda = _make_vanilla_scda(originals)
        record = _make_scpt_record(sctx, scda)
        original_size = len(scda)

        extracted = extract_scpt_strings(record)
        assert len(extracted.strings) == 3

        # All share the same args_len_offset
        alo = extracted.strings[0].args_len_offset
        for ss in extracted.strings:
            assert ss.args_len_offset == alo

        trans_map = {ss.scda_offset: t for ss, t in zip(extracted.strings, translateds)}
        total_delta = sum(
            len(t.encode("cp1252")) - ss.scda_len
            for ss, t in zip(extracted.strings, translateds)
        )

        count = patch_scpt_record(record, trans_map)
        assert count == 3

        scda_after = bytes(record.subrecords[2].data)
        for t in translateds:
            assert t.encode("cp1252") in scda_after
        assert len(scda_after) == original_size + total_delta

        # Verify argsLen was updated correctly (once, by total delta)
        new_args_len = struct.unpack_from("<H", scda_after, alo)[0]
        old_args_len = struct.unpack_from("<H", scda, alo)[0]
        assert new_args_len == old_args_len + total_delta

    def test_no_translations(self):
        sctx = _make_sctx(["Hello world test string"])
        scda = _make_nvse_scda(["Hello world test string"])
        record = _make_scpt_record(sctx, scda)

        count = patch_scpt_record(record, {})
        assert count == 0

    def test_nvse_s_token_length_updated(self):
        original = "Short text here"
        translated = "Texto mucho mas largo que el original de prueba"

        sctx = _make_sctx([original])
        scda = _make_nvse_scda([original])
        record = _make_scpt_record(sctx, scda)

        extracted = extract_scpt_strings(record)
        offset = extracted.strings[0].scda_offset

        patch_scpt_record(record, {offset: translated})

        scda_after = bytes(record.subrecords[2].data)
        t_bytes = translated.encode("cp1252")
        t_pos = scda_after.find(t_bytes)
        assert t_pos > 0
        assert scda_after[t_pos - 3] == 0x53
        assert struct.unpack_from("<H", scda_after, t_pos - 2)[0] == len(t_bytes)

    def test_revert_on_corruption(self):
        """If post-patch validation fails, original SCDA is preserved."""
        original = "Hello world test"
        sctx = _make_sctx([original])
        scda = _make_nvse_scda([original])
        record = _make_scpt_record(sctx, scda)
        original_scda = bytes(record.subrecords[2].data)

        # Patch with a valid-looking but wrong offset (not in parsed structure)
        count = patch_scpt_record(record, {9999: "should not work"})
        assert count == 0
        assert bytes(record.subrecords[2].data) == original_scda
