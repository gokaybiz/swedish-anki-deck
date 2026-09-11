from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from functions.enrichment_quality import (
    informative_construction,
    learner_order,
    model_sense_hints,
    model_source_hints,
    normalize_optional_note,
    note_warrants_assessment,
)


class EnrichmentQualityTests(unittest.TestCase):
    def entry(self, **lexical_info: object) -> dict[str, object]:
        return {
            "word": "absorbera",
            "lexical_info": lexical_info,
            "sources": {"frequency": {"cefr_band": "B2"}},
        }

    def test_hints_are_capped_so_prompts_stay_small(self) -> None:
        hints = model_source_hints(
            self.entry(
                compounds=[f"ord{i}" for i in range(20)],
                synonyms=["uppta", "suga", "införliva", "tillgodogöra"],
                constructions=["x & y", "x & A", "y & x"],
            ),
            purpose="definitions",
        )
        self.assertEqual(hints["synonyms"], ["uppta", "suga", "införliva"])
        self.assertEqual(hints["constructions"], ["x & y", "x & A", "y & x"])
        # Compound lists are provenance, not prompt material.
        self.assertNotIn("compounds", hints)

    def test_purpose_specific_hints_do_not_leak_irrelevant_source_data(self) -> None:
        explanation = (
            "En mycket lång svensk källförklaring som bara hjälper betydelseval."
        )
        entry = self.entry(
            explanations=[explanation],
            idioms=["absorbera en stöt"],
            kelly_hints=["usage pattern: absorbera en stöt"],
            synonyms=["uppta"],
            usage_labels=["i plur."],
        )
        examples = model_source_hints(entry, purpose="examples")
        self.assertEqual(
            examples,
            {
                "usage_labels": ["i plur."],
                "kelly_hints": ["usage pattern: absorbera en stöt"],
            },
        )
        definitions = model_source_hints(entry, purpose="definitions")
        self.assertEqual(
            definitions,
            {
                "usage_labels": ["i plur."],
                "kelly_hints": ["usage pattern: absorbera en stöt"],
                "explanations": [explanation],
                "synonyms": ["uppta"],
            },
        )
        notes = model_source_hints(entry, purpose="notes")
        self.assertEqual(
            notes,
            {
                "usage_labels": ["i plur."],
                "kelly_hints": ["usage pattern: absorbera en stöt"],
                "idioms": ["absorbera en stöt"],
                "explanations": [explanation],
            },
        )

    def test_plain_transitivity_is_not_note_worthy(self) -> None:
        for pattern in ("A & x", "A & B", "x &", "& x", "A/x & B/y"):
            self.assertFalse(
                informative_construction(pattern), f"{pattern!r} should be trivial"
            )

    def test_governed_preposition_particle_reflexive_and_complement_are_worthy(
        self,
    ) -> None:
        for pattern in (
            "A & på x",
            "A & sig + INF",
            "& (för x)",
            "A & x/att + SATS",
            "x & om y",
        ):
            self.assertTrue(
                informative_construction(pattern), f"{pattern!r} should be worthy"
            )

    def test_note_assessment_gate_uses_non_prompt_signals_only(self) -> None:
        # Synonyms and compounds are prompt noise, never a note justification.
        self.assertFalse(
            note_warrants_assessment(
                {
                    "word": "bok",
                    "lexical_info": {
                        "synonyms": ["volym"],
                        "compounds": ["skolbok"],
                        "constructions": ["A & x"],
                    },
                }
            )
        )
        for key, value in (
            ("idioms", ['a och o ("det viktigaste")']),
            ("usage_labels", ["i plur."]),
            ("translation_comments", ["used only in negative clauses"]),
            ("explanations", ["En historisk institution."]),
            ("kelly_hints", ["usage pattern: anmäla sig"]),
        ):
            self.assertTrue(
                note_warrants_assessment({"word": "ord", "lexical_info": {key: value}}),
                f"{key} should justify an assessment",
            )
        self.assertFalse(note_warrants_assessment({"word": "ord"}))

    def test_sense_hints_are_precise_bounded_and_optional(self) -> None:
        self.assertEqual(
            model_sense_hints(
                {
                    "source_hints": {
                        "sense_glosses": [
                            "the son of somebody's brother",
                            "an unused second gloss",
                        ]
                    }
                }
            ),
            {"sense_glosses": ["the son of somebody's brother"]},
        )
        self.assertEqual(
            model_sense_hints({"source_hints": {"sense_glosses": ["x" * 181]}}),
            {},
        )
        self.assertEqual(model_sense_hints({}), {})

    def test_hints_ignore_malformed_or_empty_source_values(self) -> None:
        self.assertEqual(
            model_source_hints(
                {"word": "ord", "lexical_info": {"variants": ["", "  ", 7, "form"]}}
            ),
            {"variants": ["form"]},
        )
        self.assertEqual(model_source_hints({"word": "ord"}), {})
        self.assertEqual(model_source_hints({"word": "ord", "lexical_info": []}), {})

    def test_learner_order_prefers_reviewed_level_then_kelly_rank(self) -> None:
        reviewed_late = {
            "frequency_rank": 9000,
            "learner_level": {"cefr": "A1"},
        }
        unreviewed_early = {
            "frequency_rank": 1,
            "sources": {"frequency": {"cefr_band": "C2"}},
        }
        unknown = {"frequency_rank": 5}
        self.assertLess(learner_order(reviewed_late), learner_order(unreviewed_early))
        self.assertEqual(learner_order(unknown), (6, 5))
        # A reviewed level must win over the raw Kelly band for the same entry.
        self.assertEqual(
            learner_order(
                {
                    "frequency_rank": 42,
                    "learner_level": {"cefr": "A2"},
                    "sources": {"frequency": {"cefr_band": "C1"}},
                }
            ),
            (1, 42),
        )

    def test_note_contract_is_strict_single_line(self) -> None:
        self.assertIsNone(normalize_optional_note(None))
        self.assertEqual(normalize_optional_note("  Useful.  "), "Useful.")
        for invalid in ("", "   ", "two\nlines", "x" * 181, 7):
            with self.assertRaises(ValueError):
                normalize_optional_note(invalid)


if __name__ == "__main__":
    unittest.main()
