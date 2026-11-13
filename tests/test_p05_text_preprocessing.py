import io
import itertools

import pytest

from conftest import load

m = load("p05_text_preprocessing")


# ---------- Part A: clean_text ----------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  hello   world  ", "hello world"),
        ("line1\nline2\tline3\r\nline4", "line1 line2 line3 line4"),
        ("ﬁnancial agreement", "financial agreement"),        # ligature (NFKC)
        ("ＡＣＭＥ　Corp", "ACME Corp"),                        # full-width + ideographic space
        ("sig​nature", "signature"),                      # zero-width space removed
        ("﻿Contract", "Contract"),                        # BOM removed
        ("non­binding", "nonbinding"),                    # soft hyphen removed
        ("bell\x07char\x00null", "bell char null"),            # control chars -> space
        ("", ""),
        (" \n\t ", ""),
    ],
)
def test_clean_text(raw, expected):
    assert m.clean_text(raw) == expected


def test_clean_text_lowercase():
    assert m.clean_text("Master  SERVICES\nAgreement", lowercase=True) == "master services agreement"


# ---------- Part B: chunk_document ----------

def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def chunks_of(data: bytes, **kw):
    return list(m.chunk_document(io.BytesIO(data), **kw))


def test_docstring_example():
    out = chunks_of(words(10).encode(), chunk_size=4, overlap=1)
    assert out == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_no_overlap_with_short_tail():
    out = chunks_of(words(7).encode(), chunk_size=3, overlap=0)
    assert out == ["w0 w1 w2", "w3 w4 w5", "w6"]


def test_tail_with_only_overlap_is_not_emitted():
    # 7 words, size 4, overlap 1: "w0..w3", "w3..w6" -> nothing new remains
    out = chunks_of(words(7).encode(), chunk_size=4, overlap=1)
    assert out == ["w0 w1 w2 w3", "w3 w4 w5 w6"]


def test_document_shorter_than_chunk():
    assert chunks_of(b"just three words", chunk_size=10, overlap=2) == ["just three words"]


def test_empty_and_whitespace_documents():
    assert chunks_of(b"", chunk_size=5, overlap=1) == []
    assert chunks_of(b"   \n\n\t  ", chunk_size=5, overlap=1) == []


def test_cleaning_is_applied():
    data = "Ｓign​ed  the\n\nﬁnal   copy".encode()
    assert chunks_of(data, chunk_size=10, overlap=0) == ["Signed the final copy"]


def test_words_split_across_reads_stay_whole():
    text = "indemnification limitation liability confidentiality termination"
    for read_size in (1, 2, 3, 5, 7, 64):
        out = chunks_of(text.encode(), chunk_size=10, overlap=0, read_size=read_size)
        assert out == [text], read_size


def test_multibyte_chars_split_across_reads():
    text = "São Paulo — contrato assinado çãõ 日本語 契約"
    for read_size in (1, 2, 3):
        out = chunks_of(text.encode("utf-8"), chunk_size=50, overlap=0, read_size=read_size)
        assert out == [text], read_size


def test_invalid_utf8_does_not_crash():
    data = b"valid start \xff\xfe broken \xc3 bytes end"
    out = chunks_of(data, chunk_size=50, overlap=0)
    assert len(out) == 1
    assert out[0].startswith("valid start")
    assert out[0].endswith("end")
    assert "�" in out[0]


def test_accepts_file_path(tmp_path):
    p = tmp_path / "contract.txt"
    p.write_bytes(words(5).encode())
    assert list(m.chunk_document(str(p), chunk_size=5, overlap=0)) == [words(5)]
    assert list(m.chunk_document(p, chunk_size=5, overlap=0)) == [words(5)]  # PathLike


class EndlessStream(io.RawIOBase):
    """Never ends. Tracks the largest single read request."""

    def __init__(self):
        self.pattern = itertools.cycle(b"clause ")
        self.max_request = 0

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            raise AssertionError("read() without a size would load the whole document")
        self.max_request = max(self.max_request, n)
        return bytes(next(self.pattern) for _ in range(n))


def test_streams_instead_of_reading_everything():
    stream = EndlessStream()
    gen = m.chunk_document(stream, chunk_size=5, overlap=0, read_size=16)
    assert next(gen) == "clause clause clause clause clause"
    assert next(gen) == "clause clause clause clause clause"
    assert stream.max_request <= 16


def test_large_document_chunk_count():
    n = 100_000
    out = chunks_of(words(n).encode(), chunk_size=200, overlap=20, read_size=4096)
    # starts at 0, 180, 360, ... ; last start must be < n - overlap to add new words
    assert out[0].split()[0] == "w0"
    assert out[-1].split()[-1] == f"w{n - 1}"
    assert all(len(c.split()) == 200 for c in out[:-1])
    for a, b in zip(out, out[1:]):
        assert a.split()[-20:] == b.split()[:20]


@pytest.mark.parametrize(
    "kw",
    [
        {"chunk_size": 0, "overlap": 0},
        {"chunk_size": 5, "overlap": 5},
        {"chunk_size": 5, "overlap": -1},
        {"chunk_size": 5, "overlap": 1, "read_size": 0},
    ],
)
def test_invalid_params(kw):
    with pytest.raises(ValueError):
        list(m.chunk_document(io.BytesIO(b"a b c"), **kw))
