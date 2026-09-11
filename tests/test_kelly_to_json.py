from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kelly_to_json import article_for, clean_headword, kelly_hints


class KellyNormalizationTests(unittest.TestCase):
    def test_malformed_primary_uses_single_word_lexical_expansion(self) -> None:
        self.assertEqual(
            clean_headword("Bproper name (bruttonationalprodukt)"),
            "bruttonationalprodukt",
        )

    def test_normal_parenthetical_annotation_is_removed(self) -> None:
        self.assertEqual(clean_headword("bank (institution)"), "bank")
        self.assertEqual(clean_headword("krona (förk. kr.)"), "krona")
        self.assertEqual(
            clean_headword("Svenska kyrkan (institution)"), "Svenska kyrkan"
        )

    def test_annotations_and_usage_patterns_are_hints_not_sentences(self) -> None:
        self.assertEqual(
            kelly_hints("krona (förk. kr.)", "", "krona"),
            ["headword annotation: förk. kr."],
        )
        self.assertEqual(
            kelly_hints("anmäla", "e.g. anmäla sig", "anmäla"),
            ["usage pattern: anmäla sig"],
        )
        self.assertEqual(
            kelly_hints(
                "Bproper name (bruttonationalprodukt)",
                "",
                "bruttonationalprodukt",
            ),
            [],
        )

    def test_missing_noun_article_is_inferred_from_word_class(self) -> None:
        self.assertEqual(article_for("", "noun-en"), "en")
        self.assertEqual(article_for("", "noun-ett"), "ett")
        self.assertEqual(article_for("ett", "noun-en"), "ett")
        self.assertEqual(article_for("", "verb"), "")


if __name__ == "__main__":
    unittest.main()
