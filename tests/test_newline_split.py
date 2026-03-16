"""Tests for newline splitting/rejoining in _translate_chunks."""

import pytest

from modtranslator._pipeline_helpers import (
    _rejoin_newlines,
    _split_newlines,
    _translate_chunks,
)
from modtranslator.backends.dummy import DummyBackend


class TestSplitNewlines:
    """Unit tests for _split_newlines()."""

    def test_single_line_texts(self):
        texts = ["Hello world", "Iron Sword"]
        flat, rmap = _split_newlines(texts)
        assert flat == ["Hello world", "Iron Sword"]
        assert len(rmap) == 2
        for kind, _ in rmap:
            assert kind == "single"

    def test_multiline_text_split(self):
        texts = ["-Transmission Started-\nJanuary 28th\n\nSome text\n\n-Transmission Ended-"]
        flat, rmap = _split_newlines(texts)
        # Only non-empty lines are in flat
        assert flat == [
            "-Transmission Started-",
            "January 28th",
            "Some text",
            "-Transmission Ended-",
        ]
        assert len(rmap) == 1
        assert rmap[0][0] == "multi"

    def test_empty_lines_preserved(self):
        texts = ["Line1\n\nLine2"]
        flat, rmap = _split_newlines(texts)
        assert flat == ["Line1", "Line2"]
        # Reassemble should preserve the empty line
        result = _rejoin_newlines(flat, rmap)
        assert result == ["Line1\n\nLine2"]

    def test_mixed_single_and_multiline(self):
        texts = ["Simple text", "Line1\nLine2", "Another simple"]
        flat, rmap = _split_newlines(texts)
        assert flat == ["Simple text", "Line1", "Line2", "Another simple"]
        assert rmap[0][0] == "single"
        assert rmap[1][0] == "multi"
        assert rmap[2][0] == "single"

    def test_separator_lines_preserved(self):
        """Lines with only dashes/spaces are preserved as-is (not translated)."""
        texts = ["at the location\n--------\nis located"]
        flat, rmap = _split_newlines(texts)
        # "--------" has strip() truthy, so it IS in flat for translation
        # But the key thing is the structure is preserved
        result = _rejoin_newlines(flat, rmap)
        assert result == ["at the location\n--------\nis located"]

    def test_multiple_empty_lines(self):
        texts = ["A\n\n\nB"]
        flat, rmap = _split_newlines(texts)
        assert flat == ["A", "B"]
        result = _rejoin_newlines(flat, rmap)
        assert result == ["A\n\n\nB"]


class TestRejoinNewlines:
    """Unit tests for _rejoin_newlines()."""

    def test_identity_single_lines(self):
        flat = ["Hello", "World"]
        rmap = [("single", [("translate", 0)]), ("single", [("translate", 1)])]
        result = _rejoin_newlines(flat, rmap)
        assert result == ["Hello", "World"]

    def test_multiline_reassembly(self):
        flat = ["Translated A", "Translated B"]
        rmap = [("multi", [
            ("translate", 0),
            ("keep", ""),
            ("translate", 1),
        ])]
        result = _rejoin_newlines(flat, rmap)
        assert result == ["Translated A\n\nTranslated B"]


class TestTranslateChunksNewlines:
    """Integration: _translate_chunks preserves newlines via dummy backend.

    DummyBackend prepends ``[{LANG}] `` to each text, so assertions
    check for that prefix on translated lines while verifying structure.
    """

    def test_multiline_preserved_through_dummy(self):
        """Newlines should survive through translation pipeline."""
        backend = DummyBackend()
        texts = ["Line1\nLine2\n\nLine3"]
        translated, errors = _translate_chunks(texts, backend, "ES")
        assert errors == []
        assert "\n" in translated[0]
        lines = translated[0].split("\n")
        assert "[ES] Line1" in lines[0]
        assert "[ES] Line2" in lines[1]
        assert lines[2] == ""  # empty line preserved
        assert "[ES] Line3" in lines[3]

    def test_transmission_example(self):
        """The exact bug scenario from the report."""
        backend = DummyBackend()
        texts = [
            "-Transmission Started-\nJanuary 28th\n\n"
            "Take me off the project\n\n\n-Transmission Ended-"
        ]
        translated, errors = _translate_chunks(texts, backend, "ES")
        assert errors == []
        assert "\n" in translated[0]
        lines = translated[0].split("\n")
        assert "-Transmission Started-" in lines[0]
        assert "January 28th" in lines[1]
        assert lines[2] == ""  # empty line preserved
        assert "Take me off the project" in lines[3]
        assert lines[4] == ""  # empty lines preserved
        assert lines[5] == ""
        assert "-Transmission Ended-" in lines[6]

    def test_separator_dashes_preserved(self):
        """Dashes separator lines are preserved."""
        backend = DummyBackend()
        texts = ["at the location\n--------\nis located"]
        translated, errors = _translate_chunks(texts, backend, "ES")
        assert errors == []
        assert "--------" in translated[0]
        assert "is located" in translated[0]
        lines = translated[0].split("\n")
        assert "at the location" in lines[0]
        assert "--------" in lines[1]  # separator translated but preserved
        assert "is located" in lines[2]

    def test_mixed_single_and_multiline(self):
        backend = DummyBackend()
        texts = ["Simple", "A\nB\nC", "Also simple"]
        translated, errors = _translate_chunks(texts, backend, "ES")
        assert errors == []
        assert "Simple" in translated[0]
        assert "\n" in translated[1]
        assert "Also simple" in translated[2]
