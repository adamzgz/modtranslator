"""Tests for Papyrus .pex parser and serializer."""

from __future__ import annotations

import struct

from modtranslator.core.pex_parser import (
    PEX_MAGIC,
    PexFile,
    PexHeader,
    _is_translatable_literal,
    parse_pex,
    serialize_pex,
)


def _make_pex_bytes(strings: list[str], post_data: bytes = b"") -> bytes:
    """Build a minimal .pex binary with the given string table."""
    parts: list[bytes] = []
    parts.append(struct.pack(">I", PEX_MAGIC))
    parts.append(struct.pack(">BB", 3, 2))  # version
    parts.append(struct.pack(">H", 1))  # game_id (Skyrim)
    parts.append(struct.pack(">Q", 0))  # compilation time
    # source, user, machine (empty)
    for _ in range(3):
        parts.append(struct.pack(">H", 0))
    # string table
    parts.append(struct.pack(">H", len(strings)))
    for s in strings:
        encoded = s.encode("utf-8")
        parts.append(struct.pack(">H", len(encoded)))
        parts.append(encoded)
    parts.append(post_data)
    return b"".join(parts)


class TestParsePex:
    def test_parse_header(self):
        data = _make_pex_bytes(["hello", "world"])
        pex = parse_pex(data)
        assert pex.header.major_version == 3
        assert pex.header.minor_version == 2
        assert pex.header.game_id == 1

    def test_parse_string_table(self):
        data = _make_pex_bytes(["alpha", "beta", "gamma"])
        pex = parse_pex(data)
        assert pex.string_table == ["alpha", "beta", "gamma"]

    def test_empty_string_table(self):
        data = _make_pex_bytes([])
        pex = parse_pex(data)
        assert pex.string_table == []

    def test_invalid_magic_raises(self):
        data = b"\x00\x00\x00\x00" + b"\x00" * 50
        try:
            parse_pex(data)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Not a PEX file" in str(e)

    def test_literal_detection(self):
        """Post-table data with type 0x02 markers should be detected as literals."""
        strings = ["GetState", "Hello World", "identifier"]
        post = (
            b"\x01" + struct.pack(">H", 0)
            + b"\x02" + struct.pack(">H", 1)
            + b"\x01" + struct.pack(">H", 2)
        )
        data = _make_pex_bytes(strings, post)
        pex = parse_pex(data)
        assert 1 in pex.literal_indices
        assert 0 not in pex.literal_indices
        assert 2 not in pex.literal_indices

    def test_dual_use_in_both_sets(self):
        """Indices used as both literal and identifier appear in both sets."""
        strings = ["dual_use"]
        post = (
            b"\x01" + struct.pack(">H", 0)
            + b"\x02" + struct.pack(">H", 0)
        )
        data = _make_pex_bytes(strings, post)
        pex = parse_pex(data)
        # Present in both sets, get_translatable_strings excludes it
        assert 0 in pex.literal_indices
        assert 0 in pex.ident_indices


class TestSerializePex:
    def test_roundtrip(self):
        strings = ["Hello", "World", "Test string with spaces"]
        data = _make_pex_bytes(strings, b"\x00" * 20)
        pex = parse_pex(data)
        rebuilt = serialize_pex(pex)
        assert data == rebuilt

    def test_modified_string_roundtrip(self):
        strings = ["Original text"]
        post = b"\x02" + struct.pack(">H", 0) + b"\x00" * 10
        data = _make_pex_bytes(strings, post)
        pex = parse_pex(data)
        pex.string_table[0] = "Texto traducido más largo"
        rebuilt = serialize_pex(pex)
        pex2 = parse_pex(rebuilt)
        assert pex2.string_table[0] == "Texto traducido más largo"
        assert pex2.header.major_version == 3

    def test_utf8_preservation(self):
        strings = ["Héllo wörld", "Ñoño"]
        data = _make_pex_bytes(strings)
        pex = parse_pex(data)
        rebuilt = serialize_pex(pex)
        pex2 = parse_pex(rebuilt)
        assert pex2.string_table == ["Héllo wörld", "Ñoño"]


class TestTranslatableLiteral:
    def test_empty_string(self):
        assert not _is_translatable_literal("")

    def test_short_string(self):
        assert not _is_translatable_literal("ab")

    def test_keyword(self):
        assert not _is_translatable_literal("None")
        assert not _is_translatable_literal("String")
        assert not _is_translatable_literal("Bool")
        assert not _is_translatable_literal("Player")
        assert not _is_translatable_literal("Health")

    def test_mcm_key(self):
        assert not _is_translatable_literal("TUNE_HEADER_1")
        assert not _is_translatable_literal("OPT_DEBUGGING")
        assert not _is_translatable_literal("QUEST_ENABLE")

    def test_internal_identifier(self):
        assert not _is_translatable_literal("::temp0")
        assert not _is_translatable_literal("::NoneVar")
        assert not _is_translatable_literal("$MCMLabel")

    def test_file_path(self):
        assert not _is_translatable_literal("textures/interface/book.dds")
        assert not _is_translatable_literal("scripts/source.psc")

    def test_single_word_rejected(self):
        """Single words are too risky — could be state names, properties, etc."""
        assert not _is_translatable_literal("GetState")
        assert not _is_translatable_literal("myVariable")
        assert not _is_translatable_literal("Warning")
        assert not _is_translatable_literal("Uninstall")
        assert not _is_translatable_literal("Active")
        assert not _is_translatable_literal("Vampire")

    def test_short_phrase_rejected(self):
        """Very short phrases with < 3 words and < 15 chars are too risky."""
        assert not _is_translatable_literal("Hold: NONE")
        assert not _is_translatable_literal("Remove Buff")
        assert not _is_translatable_literal("Set Stage")
        assert not _is_translatable_literal("Get Value")

    def test_three_word_phrase_accepted(self):
        """3+ word phrases are likely player-visible messages."""
        assert _is_translatable_literal("Open the door")
        assert _is_translatable_literal("You are dead")
        assert _is_translatable_literal("Press E to continue")

    def test_player_visible_sentence(self):
        assert _is_translatable_literal("You are entering a dangerous location.")
        assert _is_translatable_literal("You don't have enough dragon souls!")

    def test_long_phrase(self):
        assert _is_translatable_literal("Location Correction Started.")
        assert _is_translatable_literal("Hold: NONE\nHold Stored: NONE")

    def test_medium_length_accepted(self):
        """Strings > 15 chars with 2 words are accepted."""
        assert _is_translatable_literal("Downloading content")
        assert _is_translatable_literal("Installation complete")

    def test_underscore_prefix_rejected(self):
        assert not _is_translatable_literal("_LocationCorrectionScript")

    def test_skeleton_node_rejected(self):
        """Skeleton bone node names must not be translated."""
        assert not _is_translatable_literal("NPC R Hand")
        assert not _is_translatable_literal("NPC L Foot")
        assert not _is_translatable_literal("NPC Pelvis [Pelv]")
        assert not _is_translatable_literal("NPC R FrontThigh")

    def test_papyrus_auto_comment_rejected(self):
        """Papyrus auto-generated state function comments."""
        text = "Function that switches this object to the specified state"
        assert not _is_translatable_literal(text)
        assert not _is_translatable_literal("Function that returns the current state")

    def test_mod_tag_debug_rejected(self):
        """Debug messages with [ModTag] prefix."""
        assert not _is_translatable_literal("[SOS] RestartMod() called.")
        assert not _is_translatable_literal("[BCD-CLWA] Forcing Actor Value")
        assert not _is_translatable_literal("[MCM Recorder]")
        assert not _is_translatable_literal("[A Quality World Map] Settings applied.")

    def test_function_call_rejected(self):
        """Strings containing function call syntax ()."""
        assert not _is_translatable_literal("DisableMod() Called")
        assert not _is_translatable_literal("OnUpdate(): Flushing aliases")
        assert not _is_translatable_literal("HasAvailableBed(): Located suitable furniture:")

    def test_code_operator_rejected(self):
        """Strings with code comparison operators."""
        assert not _is_translatable_literal("sEnemyType == None")
        assert not _is_translatable_literal("bCreatureDefeat == True")
        assert not _is_translatable_literal("kFollower1.IsBleedingOut() || kFollower1.IsDead()")

    def test_error_log_rejected(self):
        """ERROR: log messages from mod internals."""
        assert not _is_translatable_literal(": ERROR: timed out waiting for")
        assert not _is_translatable_literal(": ERROR: empty sublist!")

    def test_debug_separator_rejected(self):
        """Debug marker lines with === separators."""
        assert not _is_translatable_literal("=== Loading Cache ===")
        assert not _is_translatable_literal("=== PLEASE EXIT THE MCM ===")

    def test_event_handler_rejected(self):
        """Event handler trace strings."""
        assert not _is_translatable_literal("AmorousAdvNjadaBrawlAlias: OnUpdate")
        assert not _is_translatable_literal("IN :OnInit :")

    def test_formid_reference_rejected(self):
        """Strings referencing FormIDs."""
        assert not _is_translatable_literal("DressUpLovers: FormID is Invalid!")
        assert not _is_translatable_literal("FormID is Invalid!")

    def test_variable_reference_rejected(self):
        """Strings with Papyrus variable references (kFollower, akActor)."""
        assert not _is_translatable_literal("TryToRecover: kFollower1")

    def test_animation_style_code_rejected(self):
        """Animation/bone style codes from RaceMenu etc."""
        assert not _is_translatable_literal("CME Tail L Thigh [LThg]")
        assert not _is_translatable_literal("MOV WeaponStaffLeftDefault")
        assert not _is_translatable_literal("CME BreastMagic L")
        assert not _is_translatable_literal("BOLT LeftHipBolt")
        assert not _is_translatable_literal("HDT TailBone05.2")
        assert not _is_translatable_literal("HDT TailBone01.1")

    def test_skyui_internal_error_rejected(self):
        """SkyUI/MCM internal error messages."""
        assert not _is_translatable_literal("AddTextOptionST has been called in an invalid state.")
        text = "SetSliderOptionValueST has been called in an invalid state."
        assert not _is_translatable_literal(text)
        assert not _is_translatable_literal("Option type mismatch. Expected toggle option, page \"")
        assert not _is_translatable_literal("Animating: SEQ:")

    def test_warpaint_code_rejected(self):
        """Warpaint texture codes."""
        assert not _is_translatable_literal("WP 018 Grass Body")
        assert not _is_translatable_literal("WP 011 Tentacles Head")

    def test_api_call_rejected(self):
        """API-style function calls."""
        assert not _is_translatable_literal("iWidgets.setZoom(myApple, 200, 200)")
        text = "iWidgets.doTransition(mySheep[0], 180, 300, 'rotation')"
        assert not _is_translatable_literal(text)

    def test_internal_function_trace_rejected(self):
        """CamelCase function name traces: 'AnimateMyLover: Serana'."""
        assert not _is_translatable_literal("AnimateMyLover: Serana")
        assert not _is_translatable_literal("SimulateBodyPhysicsCleanUp: Vampire Lord")
        assert not _is_translatable_literal("DisableMod: starting cleanup")

    def test_star_marker_rejected(self):
        """*** debug markers ***."""
        assert not _is_translatable_literal("iWant Widgets: ***LIBRARY RESET***")
        assert not _is_translatable_literal("*** MCM Reset ***: lsarVer=")

    def test_real_text_still_passes(self):
        """Make sure real player-visible text is NOT blocked by new filters."""
        assert _is_translatable_literal("You have won the entire game!")
        assert _is_translatable_literal("I've gained new insight from defeating the enemy.")
        assert _is_translatable_literal("Blessed of Azura")
        text = "The chance that a new quest is put up or an old one taken down."
        assert _is_translatable_literal(text)
        assert _is_translatable_literal("Your musical talent increases")
        assert _is_translatable_literal("Enable Developer Mode")
        assert _is_translatable_literal("General Settings")
        assert _is_translatable_literal("How many seconds to wait after defeat before respawning.")


class TestGetTranslatableStrings:
    def test_filters_correctly(self):
        pex = PexFile(
            header=PexHeader(3, 2, 1, 0, "", "", ""),
            string_table=[
                "GetState",                          # 0 - not a literal
                "You have won the entire game!",     # 1 - translatable
                "None",                              # 2 - keyword
                "MCM_OPTION_KEY",                    # 3 - MCM key
                "Welcome back to your home sweet home.",  # 4 - translatable
                "Active",                            # 5 - keyword
            ],
            post_table_data=b"",
            literal_indices={1, 2, 3, 4, 5},
            ident_indices={0},
        )
        result = pex.get_translatable_strings()
        assert 1 in result
        assert 4 in result
        assert 0 not in result  # not a literal
        assert 2 not in result  # keyword
        assert 3 not in result  # MCM key
        assert 5 not in result  # keyword

    def test_dual_use_excluded(self):
        """Strings used as both literal and identifier are excluded."""
        pex = PexFile(
            header=PexHeader(3, 2, 1, 0, "", "", ""),
            string_table=["This is a dual use string for testing purposes"],
            post_table_data=b"",
            literal_indices={0},
            ident_indices={0},
        )
        result = pex.get_translatable_strings()
        assert 0 not in result
