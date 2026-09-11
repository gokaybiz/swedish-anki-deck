from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from functions.card_quality import (
    CARD_PLAN_VERSION,
    PRODUCTION_SCORE_THRESHOLD,
    card_source_digest,
    find_target_form,
    production_candidate,
    reviewed_card_plan,
)


class CardQualityTests(unittest.TestCase):
    def entry(self) -> dict[str, object]:
        return {
            "frequency_rank": 500,
            "article": "att",
            "word": "växa",
            "tags": ["verb"],
            "learner_level": {"cefr": "B1"},
            "definitions": [
                {
                    "definition": "grow",
                    "example": "Barnet har vuxit snabbt.",
                    "example_translation": "The child has grown quickly.",
                    "note": None,
                }
            ],
            "inflections": {"present": "växer", "supine": "vuxit"},
        }

    def test_exact_inflected_target_is_found_but_unrelated_pronoun_is_not(self) -> None:
        entry = self.entry()
        self.assertEqual(find_target_form(entry, "Barnet har vuxit snabbt."), "vuxit")
        pronoun = {"word": "vi", "inflections": {}}
        self.assertIsNone(find_target_form(pronoun, "Han har sett oss."))

    def test_discontinuous_construction_matches_both_visible_parts(self) -> None:
        entry = {"word": "varken…eller", "inflections": {}}
        sentence = "Jag dricker varken kaffe eller te."
        self.assertEqual(find_target_form(entry, sentence), "varken…eller")

    def test_reviewed_plan_is_invalidated_when_source_content_changes(self) -> None:
        entry = self.entry()
        plan = {
            "version": CARD_PLAN_VERSION,
            "source_digest": card_source_digest(entry),
            "reviewed_by": "test",
            "cards": [{"source_indices": [0]}],
        }
        entry["card_plan"] = plan
        self.assertEqual(reviewed_card_plan(entry), plan)
        entry["definitions"][0]["example"] = "Barnet växer."  # type: ignore[index]
        self.assertIsNone(reviewed_card_plan(entry))

    def test_production_score_rewards_frequency_and_reusable_constructions(
        self,
    ) -> None:
        entry = self.entry()
        entry["lexical_info"] = {
            "constructions": ["A & sig (till x)"],
            "kelly_hints": ["usage pattern: anmäla sig"],
        }
        candidate = production_candidate(entry)
        self.assertGreaterEqual(candidate.score, PRODUCTION_SCORE_THRESHOLD)
        self.assertIn("construction", candidate.reasons)
        self.assertIn("reflexive", candidate.reasons)
        self.assertIn("constructions", candidate.evidence)


if __name__ == "__main__":
    unittest.main()
