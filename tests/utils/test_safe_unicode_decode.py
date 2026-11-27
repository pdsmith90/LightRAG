"""``safe_unicode_decode`` — undoing ``\\uXXXX`` escapes in LLM replies.

A character outside the Basic Multilingual Plane (a math-italic letter, an
emoji) is escaped as a UTF-16 surrogate pair. Decoding each half on its own
leaves two lone surrogates, which UTF-8 cannot encode, so the first write of
the reply (the LLM cache) raises and fails the document.
"""

import pytest

from lightrag.utils import safe_unicode_decode

pytestmark = pytest.mark.offline


def test_bmp_escapes_are_decoded():
    assert safe_unicode_decode(b"caf\\u00e9 \\u6587") == "café 文"


def test_surrogate_pair_becomes_one_character():
    # 𝜎 is U+1D70E MATHEMATICAL ITALIC SMALL SIGMA
    decoded = safe_unicode_decode(b'{"description": "variance \\ud835\\udf0e^2"}')
    assert decoded == '{"description": "variance \U0001d70e^2"}'
    decoded.encode("utf-8")


def test_unpaired_surrogate_is_replaced():
    # e.g. a reply cut off by a token limit between the two halves
    decoded = safe_unicode_decode(b"sigma \\ud835")
    assert decoded == "sigma �"
    decoded.encode("utf-8")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (rb"\ud83d\ude00", "😀"),
        (rb"\uD834\uDD1E", "𝄞"),
        (rb"\u4f60\ud83d\ude00\u597d", "你😀好"),
        ("Already decoded: 😀".encode("utf-8"), "Already decoded: 😀"),
        (rb"Keep \utils, \n and \uZZZZ", r"Keep \utils, \n and \uZZZZ"),
        (b"", ""),
    ],
)
def test_safe_unicode_decode_keeps_utf8_text(content: bytes, expected: str):
    result = safe_unicode_decode(content)

    assert result == expected
    assert result.encode("utf-8").decode("utf-8") == expected
