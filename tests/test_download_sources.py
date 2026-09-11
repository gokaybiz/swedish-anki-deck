from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Self

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_sources import SourceSpec, download_source, validate_source
from kelly_to_json import verified_artifacts


class Response(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class DownloadTests(unittest.TestCase):
    def test_download_is_atomic_and_existing_valid_file_is_reused(self) -> None:
        spec = SourceSpec(
            "test", "test.xls", "https://example.invalid/test.xls", "xls", 8, "test"
        )
        content = bytes.fromhex("d0cf11e0a1b11ae1") + b"payload"
        calls = 0

        def opener(request: object, timeout: int) -> Response:
            nonlocal calls
            calls += 1
            self.assertEqual(timeout, 120)
            return Response(content)

        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            first = download_source(spec, raw, opener=opener)
            second = download_source(spec, raw, opener=opener)
            self.assertEqual(calls, 1)
            self.assertEqual(first["status"], "downloaded")
            self.assertEqual(second["status"], "reused")
            self.assertEqual((raw / spec.filename).read_bytes(), content)
            self.assertFalse((raw / "test.xls.part").exists())

    def test_conversion_verifies_exact_manifested_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            kelly = raw / "kelly.xls"
            folkets = raw / "folkets.xml"
            kelly.write_bytes(b"kelly")
            folkets.write_bytes(b"folkets")
            manifest = raw / "sources.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "filename": kelly.name,
                                "url": "https://example/kelly",
                                "bytes": 5,
                                "sha256": hashlib.sha256(b"kelly").hexdigest(),
                            },
                            {
                                "filename": folkets.name,
                                "url": "https://example/folkets",
                                "bytes": 7,
                                "sha256": hashlib.sha256(b"folkets").hexdigest(),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            records = verified_artifacts(manifest, {"kelly": kelly, "folkets": folkets})
            self.assertEqual(records["folkets"]["url"], "https://example/folkets")
            folkets.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                verified_artifacts(manifest, {"kelly": kelly, "folkets": folkets})

    def test_cefrlex_tsv_validation_checks_schema_and_row_count(self) -> None:
        spec = SourceSpec(
            "test",
            "test.tsv",
            "https://example.invalid/test.tsv",
            "cefrlex-tsv",
            1,
            "test",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / spec.filename
            path.write_text(
                "word\ttag\tlevel_freq@a1\tnb_doc@a1\n" + "bok\tNN_UTR\t1\t1\n" * 1_001,
                encoding="utf-8",
            )
            validate_source(path, spec)
            path.write_text("not\ta\tresource\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CEFRLex"):
                validate_source(path, spec)

    def test_invalid_download_is_removed(self) -> None:
        spec = SourceSpec(
            "test", "test.xml", "https://example.invalid/test.xml", "xml", 1, "test"
        )
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            with self.assertRaises(ValueError):
                download_source(
                    spec, raw, opener=lambda *_args, **_kwargs: Response(b"not xml")
                )
            self.assertFalse((raw / "test.xml").exists())
            self.assertFalse((raw / "test.xml.part").exists())


if __name__ == "__main__":
    unittest.main()
