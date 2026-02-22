"""Tests for record data classes, focusing on encoding fallbacks."""

from modtranslator.core.constants import Game
from modtranslator.core.records import Subrecord
from modtranslator.translation.registry import is_translatable
from tests.conftest import make_fo4_plugin, make_plugin, make_skyrim_plugin


class TestDecodeStringFallback:
    def test_utf8_preferred(self):
        sub = Subrecord(type=b"FULL", data=bytearray(b"Hello\x00"))
        assert sub.decode_string() == "Hello"

    def test_cp1252_fallback(self):
        # 0xE9 = é in cp1252
        sub = Subrecord(type=b"FULL", data=bytearray(b"caf\xe9\x00"))
        assert sub.decode_string() == "café"

    def test_latin1_fallback_for_invalid_cp1252(self):
        # 0x90 is undefined in cp1252 but valid in latin-1 (U+0090 control char)
        data = bytearray(b"test\x90text\x00")
        sub = Subrecord(type=b"FULL", data=data)
        # Should not raise — latin-1 accepts all bytes
        result = sub.decode_string()
        assert "test" in result
        assert "text" in result

    def test_0x8d_invalid_cp1252(self):
        # 0x8D is another byte undefined in cp1252
        data = bytearray(b"hello\x8dworld\x00")
        sub = Subrecord(type=b"FULL", data=data)
        result = sub.decode_string()
        assert "hello" in result
        assert "world" in result

    def test_0x9d_invalid_cp1252(self):
        # 0x9D is undefined in cp1252
        data = bytearray(b"item\x9dname\x00")
        sub = Subrecord(type=b"FULL", data=data)
        result = sub.decode_string()
        assert "item" in result
        assert "name" in result

    def test_all_bytes_decode_without_error(self):
        # latin-1 should handle all possible byte values
        for byte_val in range(256):
            data = bytearray([byte_val, 0x00])
            sub = Subrecord(type=b"FULL", data=data)
            # Should never raise
            sub.decode_string()

    def test_no_null_terminator(self):
        sub = Subrecord(type=b"FULL", data=bytearray(b"Hello"))
        assert sub.decode_string() == "Hello"

    def test_empty_data(self):
        sub = Subrecord(type=b"FULL", data=bytearray(b"\x00"))
        assert sub.decode_string() == ""

    def test_encode_string_cp1252(self):
        sub = Subrecord(type=b"FULL", data=bytearray())
        sub.encode_string("café")
        assert sub.data == bytearray(b"caf\xe9\x00")


class TestEncodeStringMultilang:
    """encode_string() uses the correct code page per language."""

    def test_cp1251_cyrillic(self):
        """Russian text encodes as cp1251."""
        sub = Subrecord(type=b"FULL", data=bytearray())
        sub.encode_string("Привет", "cp1251")
        assert sub.data == bytearray("Привет".encode("cp1251") + b"\x00")

    def test_cp1250_polish(self):
        """Polish text encodes as cp1250."""
        sub = Subrecord(type=b"FULL", data=bytearray())
        sub.encode_string("Żółw", "cp1250")
        assert sub.data == bytearray("Żółw".encode("cp1250") + b"\x00")

    def test_encoding_for_lang_ru(self):
        from modtranslator.core.constants import encoding_for_lang
        assert encoding_for_lang("RU") == "cp1251"
        assert encoding_for_lang("ru") == "cp1251"

    def test_encoding_for_lang_pl(self):
        from modtranslator.core.constants import encoding_for_lang
        assert encoding_for_lang("PL") == "cp1250"

    def test_encoding_for_lang_default(self):
        from modtranslator.core.constants import encoding_for_lang
        assert encoding_for_lang("FR") == "cp1252"
        assert encoding_for_lang("DE") == "cp1252"
        assert encoding_for_lang("ES") == "cp1252"

    def test_roundtrip_ru(self):
        """Cyrillic survives encode → decode with explicit cp1251."""
        sub = Subrecord(type=b"FULL", data=bytearray())
        text = "мой враг"
        sub.encode_string(text, "cp1251")
        assert sub.decode_string("cp1251") == text

    def test_roundtrip_pl(self):
        """Polish chars survive encode → decode with explicit cp1250."""
        sub = Subrecord(type=b"FULL", data=bytearray())
        text = "mój wróg"
        sub.encode_string(text, "cp1250")
        assert sub.decode_string("cp1250") == text


class TestDetectGame:
    def test_detect_fo3(self):
        plugin = make_plugin(version=0.94)
        assert plugin.detect_game() == Game.FALLOUT3

    def test_detect_fo4_hedr_095(self):
        """FO4 older CK uses HEDR 0.95."""
        plugin = make_plugin(version=0.95)
        assert plugin.detect_game() == Game.FALLOUT4

    def test_detect_fo4_hedr_100(self):
        """FO4 newer CK uses HEDR 1.0."""
        plugin = make_fo4_plugin()
        assert plugin.detect_game() == Game.FALLOUT4

    def test_detect_skyrim(self):
        plugin = make_skyrim_plugin()
        assert plugin.detect_game() == Game.SKYRIM

    def test_detect_unknown(self):
        plugin = make_plugin(version=2.0)
        assert plugin.detect_game() == Game.UNKNOWN


class TestIsLocalized:
    def test_not_localized(self):
        plugin = make_skyrim_plugin(localized=False)
        assert not plugin.is_localized

    def test_localized(self):
        plugin = make_skyrim_plugin(localized=True)
        assert plugin.is_localized

    def test_fo3_never_localized(self):
        plugin = make_plugin()
        assert not plugin.is_localized

    def test_fo4_localized(self):
        plugin = make_fo4_plugin(localized=True)
        assert plugin.is_localized


class TestFo4Registry:
    """FO4-specific record types in the translatable registry."""

    def test_omod_full_translatable(self):
        assert is_translatable(b"OMOD", b"FULL")

    def test_cmpo_full_translatable(self):
        assert is_translatable(b"CMPO", b"FULL")

    def test_innr_full_translatable(self):
        assert is_translatable(b"INNR", b"FULL")

    def test_dmgt_full_translatable(self):
        assert is_translatable(b"DMGT", b"FULL")


class TestGameChoiceAndScan:
    def test_game_choice_fo4(self):
        from modtranslator.pipeline import GameChoice
        assert GameChoice.fo4 == "fo4"

    def test_scan_directory_finds_esl(self, tmp_path):
        from modtranslator.pipeline import scan_directory
        (tmp_path / "test.esp").write_bytes(b"")
        (tmp_path / "test.esl").write_bytes(b"")
        result = scan_directory(tmp_path)
        names = [f.name for f in result.esp_files]
        assert "test.esp" in names
        assert "test.esl" in names
