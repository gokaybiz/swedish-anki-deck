from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from enrich_cefr_levels_local import apply_record, proposal_for
from functions.cefr import infer_learner_level, load_cefrlex, profile_for


class CefrLevelTests(unittest.TestCase):
    def test_profile_matching_is_part_of_speech_sensitive(self) -> None:
        rows = {
            "miljon": [
                self.row("miljon", "NN_UTR", A1=(12.0, 2)),
                self.row("miljon", "RG", B2=(5.0, 3)),
            ]
        }
        self.assertEqual(profile_for(rows, "miljon", "noun")["A1"]["documents"], 2)
        numeral = profile_for(rows, "miljon", "numeral")
        self.assertNotIn("A1", numeral)
        self.assertEqual(numeral["B2"]["documents"], 3)

    def test_transparent_ordinal_is_capped_without_llm(self) -> None:
        estimate = infer_learner_level(
            word="femtionde",
            part_of_speech="numeral",
            kelly_band="C2",
            receptive={},
            productive={},
        )
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertEqual(estimate.cefr, "A2")
        self.assertEqual(estimate.method, "numeral-family-v1")

    def test_unsupported_non_numeral_is_not_guessed(self) -> None:
        self.assertIsNone(
            infer_learner_level(
                word="ovanlig",
                part_of_speech="adjective",
                kelly_band="C2",
                receptive={},
                productive={},
            )
        )

    def test_proposal_apply_is_review_first_and_idempotent(self) -> None:
        entry = {
            "word": "femtionde",
            "tags": ["numeral"],
            "sources": {"frequency": {"cefr_band": "C2"}},
        }
        proposal = proposal_for(
            Path("femtionde_numeral.json"),
            entry,
            svalex_rows={},
            swellex_rows={},
        )
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertNotIn("learner_level", entry)
        self.assertTrue(apply_record(entry, proposal))
        self.assertEqual(entry["learner_level"]["cefr"], "A2")
        self.assertFalse(apply_record(entry, proposal))
        conflicting = json.loads(json.dumps(proposal))
        conflicting["learner_level"]["cefr"] = "C2"
        with self.assertRaisesRegex(ValueError, "different learner level"):
            apply_record(entry, conflicting)

    def test_cefrlex_loader_accepts_svalex_without_c2_columns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.tsv"
            fieldnames = ["word", "tag", "total_freq@total"]
            for level in ("a1", "a2", "b1", "b2", "c1"):
                fieldnames.extend((f"level_freq@{level}", f"nb_doc@{level}"))
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    {
                        **{key: "0" for key in fieldnames},
                        "word": "bok",
                        "tag": "NN_UTR",
                        "level_freq@a1": "10",
                        "nb_doc@a1": "2",
                    }
                )
            rows = load_cefrlex(path)
            self.assertEqual(profile_for(rows, "bok", "noun")["A1"]["documents"], 2)

    @staticmethod
    def row(
        word: str,
        tag: str,
        **levels: tuple[float, int],
    ) -> dict[str, str]:
        row = {"word": word, "tag": tag}
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            frequency, documents = levels.get(level, (0.0, 0))
            row[f"level_freq@{level.lower()}"] = str(frequency)
            row[f"nb_doc@{level.lower()}"] = str(documents)
        return row


if __name__ == "__main__":
    unittest.main()
