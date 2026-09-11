from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_json_frequency


def write_entry(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class FrequencyAuditTests(unittest.TestCase):
    def run_audit(self, source: Path) -> tuple[int, str]:
        argv = ["verify_json_frequency.py", "--source", str(source)]
        output = io.StringIO()
        code = 0
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            try:
                verify_json_frequency.main()
            except SystemExit as exc:
                code = int(exc.code or 0)
        return code, output.getvalue()

    def test_explainable_gaps_do_not_fail_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            write_entry(source / "a.json", {"frequency_rank": 1})
            write_entry(source / "c.json", {"frequency_rank": 3})
            code, output = self.run_audit(source)
            self.assertEqual(code, 0)
            self.assertIn("gaps: 1", output)
            self.assertIn("gaps are expected", output)

    def test_duplicate_identity_fails_but_cross_pos_homographs_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            write_entry(
                source / "first.json",
                {
                    "frequency_rank": 1,
                    "article": "en",
                    "word": "krona",
                    "tags": ["noun"],
                },
            )
            write_entry(
                source / "duplicate.json",
                {
                    "frequency_rank": 2,
                    "article": "en",
                    "word": "krona",
                    "tags": ["noun"],
                },
            )
            code, output = self.run_audit(source)
            self.assertEqual(code, 1)
            self.assertIn("duplicate identities: 1", output)

            write_entry(
                source / "duplicate.json",
                {
                    "frequency_rank": 2,
                    "article": "",
                    "word": "krona",
                    "tags": ["verb"],
                },
            )
            code, output = self.run_audit(source)
            self.assertEqual(code, 0)
            self.assertIn("duplicate identities: 0", output)

    def test_duplicate_rank_and_malformed_entry_still_fail(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            write_entry(source / "a.json", {"frequency_rank": 1})
            write_entry(source / "b.json", {"frequency_rank": 1})
            code, output = self.run_audit(source)
            self.assertEqual(code, 1)
            self.assertIn("duplicate ranks: 1", output)

            (source / "b.json").unlink()
            write_entry(source / "c.json", {"frequency_rank": 0})
            code, output = self.run_audit(source)
            self.assertEqual(code, 1)
            self.assertIn("Malformed: 1", output)


if __name__ == "__main__":
    unittest.main()
