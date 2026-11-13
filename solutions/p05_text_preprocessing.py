"""
Reference solution — Problem 5 (safe, streaming text preprocessing).

Design
- Stream bytes with read(read_size); never call read() with no size.
- codecs incremental decoder with errors="replace": handles multi-byte characters split
  across reads and turns invalid bytes into U+FFFD instead of crashing the pipeline.
- Clean each decoded piece, then keep a "carry": if the piece didn't end on whitespace,
  its last word may continue in the next read, so hold it back.
- Sliding word window via a deque; emit when it reaches chunk_size, then drop
  `stride` words. Track whether words were added since the last emit so the final
  partial chunk is only emitted if it contains new words.

Memory: O(read_size + chunk_size), independent of document size.

What else "safely" means in production (good interview talking points)
- Enforce a max byte/page limit and a timeout; reject zip bombs / huge PDFs early.
- Parse untrusted files (PDF, DOCX) in a sandboxed worker with resource limits.
- OCR quality checks (ratio of U+FFFD / non-dictionary tokens) and route bad docs to review.
- Redact PII before logs/telemetry; never log raw contract text.
- Chunk by model tokenizer tokens (not words) to respect context limits; keep offsets so
  extractions can cite the source span.
- Caveat: NFKC per read can differ from whole-document NFKC if a combining sequence is
  split exactly at a read boundary; acceptable for chunking, or carry trailing combining
  marks to the next read if exactness matters.
"""

import codecs
import os
import re
import unicodedata
from collections import deque
from typing import BinaryIO, Iterator, Union

Source = Union[str, os.PathLike, BinaryIO]

_WS = re.compile(r"\s+")


def clean_text(text: str, lowercase: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        if cat == "Cc" or ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    text = _WS.sub(" ", "".join(out)).strip()
    return text.lower() if lowercase else text


def _validate(chunk_size: int, overlap: int, read_size: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if read_size < 1:
        raise ValueError("read_size must be >= 1")


def _iter_words(stream: BinaryIO, read_size: int) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    carry = ""
    while True:
        raw = stream.read(read_size)
        final = not raw
        text = carry + decoder.decode(raw or b"", final=final)
        carry = ""
        if not text:
            if final:
                return
            continue

        # Decide if the last word might continue in the next read. Check the raw text
        # (before cleaning) so trailing whitespace/control chars count as a boundary.
        ends_clean = final or text[-1].isspace() or unicodedata.category(text[-1]) == "Cc"
        cleaned = clean_text(text)
        words = cleaned.split(" ") if cleaned else []
        if words and not ends_clean:
            carry = words.pop()
        yield from words

        if final:
            if carry:
                yield carry
            return


def chunk_document(
    source: Source,
    chunk_size: int = 200,
    overlap: int = 20,
    read_size: int = 65536,
) -> Iterator[str]:
    _validate(chunk_size, overlap, read_size)
    stride = chunk_size - overlap

    is_path = isinstance(source, (str, os.PathLike))
    stream = open(source, "rb") if is_path else source
    try:
        window: deque = deque()
        new_words = 0  # words added since the last emitted chunk
        for word in _iter_words(stream, read_size):
            window.append(word)
            new_words += 1
            if len(window) == chunk_size:
                yield " ".join(window)
                new_words = 0
                for _ in range(stride):
                    window.popleft()
        if new_words:
            yield " ".join(window)
    finally:
        if is_path:
            stream.close()
