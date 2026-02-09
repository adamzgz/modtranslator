"""Tests for record data classes, focusing on encoding fallbacks."""

from modtranslator.core.constants import Game
from modtranslator.core.records import Subrecord
from tests.conftest import make_plugin, make_skyrim_plugin


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


class TestDetectGame:
    def test_detect_fo3(self):
        plugin = make_plugin(version=0.94)
        assert plugin.detect_game() == Game.FALLOUT3

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
