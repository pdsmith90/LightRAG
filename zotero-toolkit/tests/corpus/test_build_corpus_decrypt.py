#!/usr/bin/env python3
"""Behavioural tests for build_corpus's decrypt-before-OCR step (2026-10-06).
Run:  .venv/bin/python test_build_corpus_decrypt.py     (plain asserts; pytest also collects it)
ocrmypdf exits 8 on an encrypted PDF even when it opens without a password; the OCR route then strips
the owner-password encryption with pikepdf and retries once. ocrmypdf is a fake here; the one real
decryption is skipped when pikepdf is not installed. All text is invented."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile

try:
    import corpus_testlib  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf
import build_corpus as bc
from test_build_corpus_ocrlayer import patched

LINE = "The orbit determination residuals stay below two centimetres for the whole arc."


def small_pdf(path, **save):
    doc = pymupdf.open()
    doc.new_page().insert_text((60, 80), LINE, fontsize=9)
    doc.save(path, **save)
    doc.close()


def ocr_with(original, rc, decrypt):
    """Run ocr_pdf_to_md(original) with a fake ocrmypdf that exits `rc` on the original path only."""
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        if args[-2] == original and rc:
            raise subprocess.CalledProcessError(rc, args)

    out = io.StringIO()
    with (
        patched(bc.shutil, "which", lambda name: "/usr/bin/ocrmypdf"),
        patched(bc.subprocess, "run", fake_run),
        patched(bc, "_decrypted_copy", decrypt),
        patched(bc, "pdf_to_md", lambda path: "ocr text"),
        contextlib.redirect_stdout(out),
    ):
        result = bc.ocr_pdf_to_md(original)
    return result, calls, out.getvalue()


def fake_decrypt(path, td):
    copy = os.path.join(td, "decrypted.pdf")
    with open(copy, "wb") as f:
        f.write(b"%PDF-1.4 decrypted")
    return copy


def test_encrypted_input_is_decrypted_then_ocrd():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "enc.pdf")
        small_pdf(p)
        result, calls, log = ocr_with(p, 8, fake_decrypt)
    assert result == "ocr text" and len(calls) == 2, (result, calls)
    assert (
        calls[0][-2] == p
        and calls[1][-2].endswith("decrypted.pdf")
        and calls[1][:3] == ["ocrmypdf", "--force-ocr", "--quiet"]
    )
    assert "DECRYPTED enc.pdf for OCR" in log, log


def test_other_failures_are_not_decrypted():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bad.pdf")
        small_pdf(p)
        result, calls, log = ocr_with(p, 2, fake_decrypt)
    assert (
        result == "" and len(calls) == 1 and "WARN: ocrmypdf failed on bad.pdf" in log
    ), (result, calls, log)


def test_failed_decryption_is_reported():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "enc.pdf")
        small_pdf(p)
        result, calls, log = ocr_with(p, 8, lambda path, td: "")
    assert (
        result == ""
        and len(calls) == 1
        and "WARN: ocrmypdf failed on enc.pdf" in log
        and "DECRYPTED" not in log
    )


def test_real_decryption_with_pikepdf():
    try:
        import pikepdf
    except ImportError:
        print("skip: pikepdf not installed")
        return
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "enc.pdf")
        small_pdf(
            p,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-secret",
            user_pw="",
        )
        with pikepdf.open(p) as pdf:
            assert pdf.is_encrypted
        copy = bc._decrypted_copy(p, td)
        assert copy and os.path.exists(copy)
        with pikepdf.open(copy) as pdf:
            assert not pdf.is_encrypted
        with pymupdf.open(copy) as doc:
            assert LINE.split()[1] in doc[0].get_text()


if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
