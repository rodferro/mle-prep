"""
Problem 5 — Safely parse and preprocess large text documents

Contracts can be huge (hundreds of MB of extracted text), may contain invalid UTF-8
from bad OCR/exports, odd Unicode (ligatures, full-width chars, zero-width spaces),
and control characters. Build the first stage of an NLP pipeline.

Part A — `clean_text(text, lowercase=False) -> str`
    1. Unicode-normalize with NFKC (e.g. "ﬁ" -> "fi", full-width "Ａ" -> "A").
    2. Remove format characters (Unicode category "Cf", e.g. zero-width space U+200B,
       BOM U+FEFF, soft hyphen U+00AD).
    3. Replace every other control character (category "Cc", which includes \\n, \\t, \\r)
       and every whitespace character with a single space.
    4. Collapse runs of whitespace into one space and strip both ends.
    5. Lowercase if requested.

Part B — `chunk_document(source, chunk_size=200, overlap=20, read_size=65536)`
    A GENERATOR yielding chunks of cleaned text for downstream embedding/classification.
    - `source` is either a file path (str / os.PathLike) or a binary file-like object
      with .read(n). Open paths in binary mode yourself, and close them when done.
    - Read at most `read_size` bytes at a time. NEVER read the whole document into memory.
      (A test feeds an endless stream and only asks for the first chunk.)
    - Decode as UTF-8; invalid bytes become U+FFFD instead of raising. A multi-byte
      character split across two reads must decode correctly.
    - Apply clean_text, then split into words on spaces. A word split across two reads
      must stay one word.
    - Each chunk is `chunk_size` consecutive words joined by single spaces. Consecutive
      chunks share `overlap` words (stride = chunk_size - overlap). The final chunk may be
      shorter; never yield a chunk that adds no new words. Empty input yields nothing.
      Example: words w0..w9, chunk_size=4, overlap=1 ->
          "w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"
    - Raise ValueError if chunk_size < 1, overlap < 0, overlap >= chunk_size, or
      read_size < 1. (Raising when the generator is first advanced is fine.)

Things to explain in the interview (not tested): a max-size guard, PDF/OCR extraction,
PII redaction before logging, token-based (tokenizer) chunking instead of words,
language detection, and processing in a sandboxed worker for untrusted files.
"""

import os
from typing import BinaryIO, Iterator, Union

Source = Union[str, os.PathLike, BinaryIO]


def clean_text(text: str, lowercase: bool = False) -> str:
    raise NotImplementedError


def chunk_document(
    source: Source,
    chunk_size: int = 200,
    overlap: int = 20,
    read_size: int = 65536,
) -> Iterator[str]:
    raise NotImplementedError
