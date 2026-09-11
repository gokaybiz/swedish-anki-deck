from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from functions.inflection import extract_inflections


class InflectionTests(unittest.TestCase):
    def test_noun_forms_are_labelled_by_folkets_order(self) -> None:
        entry = {"word": "hus", "paradigm": ["huset", "hus", "husen"]}
        self.assertEqual(
            extract_inflections(entry, "noun"),
            {"definite": "huset", "plural": "hus", "definite_plural": "husen"},
        )
        pingvin = {
            "word": "pingvin",
            "paradigm": ["pingvin", "pingvinen", "pingviner", "pingvinerna"],
        }
        self.assertEqual(
            extract_inflections(pingvin, "noun"),
            {
                "definite": "pingvinen",
                "plural": "pingviner",
                "definite_plural": "pingvinerna",
            },
        )
        kravaller = {
            "word": "kravaller",
            "paradigm": ["kravall", "kravellen", "kravaller", "kravallerna"],
            "use": ["plural"],
        }
        self.assertEqual(
            extract_inflections(kravaller, "noun"),
            {
                "singular": "kravall",
                "definite": "kravellen",
                "plural": "kravaller",
                "definite_plural": "kravallerna",
            },
        )

    def test_regular_and_irregular_verbs(self) -> None:
        skriva = {
            "word": "skriva",
            "paradigm": ["skrev", "skrivit", "skriv", "skriva", "skriver"],
        }
        veta = {"word": "veta", "paradigm": ["visste", "vetat", "veta", "vet", "vet"]}
        self.assertEqual(
            extract_inflections(skriva, "verb"),
            {
                "past": "skrev",
                "supine": "skrivit",
                "present": "skriver",
                "imperative": "skriv",
            },
        )
        self.assertEqual(extract_inflections(veta, "verb")["imperative"], "vet")

    def test_expanded_and_reversed_verb_shapes(self) -> None:
        omsatta = {
            "word": "omsätta",
            "paradigm": [
                "omsätter",
                "omsatte",
                "omsatt",
                "omsätt",
                "omsätta",
                "omsätter",
            ],
        }
        burra = {"word": "burra", "paradigm": ["burra", "burrar", "burrade", "burrat"]}
        besluta = {
            "word": "besluta",
            "paradigm": [
                "beslutade",
                "beslöt",
                "beslutit",
                "beslutat",
                "besluta",
                "beslutar",
            ],
        }
        self.assertEqual(
            extract_inflections(omsatta, "verb"),
            {
                "past": "omsatte",
                "supine": "omsatt",
                "present": "omsätter",
                "imperative": "omsätt",
            },
        )
        self.assertEqual(
            extract_inflections(burra, "verb"),
            {"present": "burrar", "past": "burrade", "supine": "burrat"},
        )
        self.assertEqual(extract_inflections(besluta, "verb"), {"present": "beslutar"})

    def test_modal_and_defective_verbs_do_not_invent_imperatives(self) -> None:
        kunna = {"word": "kunna", "paradigm": ["kunde", "kunnat", "kunna", "kan"]}
        maste = {"word": "måste", "paradigm": ["måste", "måst"]}
        self.assertEqual(
            extract_inflections(kunna, "verb"),
            {"past": "kunde", "supine": "kunnat", "present": "kan"},
        )
        self.assertEqual(
            extract_inflections(maste, "verb"),
            {"past": "måste", "supine": "måst", "present": "måste"},
        )

    def test_adjective_shapes_use_usage_markers(self) -> None:
        stor = {"word": "stor", "paradigm": ["stort", "stora", "större", "störst"]}
        bra = {"word": "bra", "paradigm": ["bättre", "bäst"], "use": ["oböjligt"]}
        fa = {"word": "få", "paradigm": ["färre"], "use": ["plural; superlativ saknas"]}
        fler = {"word": "fler", "paradigm": ["flest"], "use": ["komparativ"]}
        betagen = {"word": "betagen", "paradigm": ["betagen", "betaget", "betagna"]}
        self.assertEqual(
            extract_inflections(stor, "adjective"),
            {
                "neuter": "stort",
                "plural": "stora",
                "comparative": "större",
                "superlative": "störst",
            },
        )
        self.assertEqual(
            extract_inflections(bra, "adjective"),
            {"comparative": "bättre", "superlative": "bäst"},
        )
        self.assertEqual(extract_inflections(fa, "adjective"), {"comparative": "färre"})
        self.assertEqual(
            extract_inflections(fler, "adjective"), {"superlative": "flest"}
        )
        self.assertEqual(
            extract_inflections(betagen, "adjective"),
            {"neuter": "betaget", "plural": "betagna"},
        )


if __name__ == "__main__":
    unittest.main()
