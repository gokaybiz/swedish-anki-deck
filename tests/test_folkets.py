from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from functions.folkets import FolketsLexicon
from kelly_to_json import build_entry

XML = """<?xml version="1.0" encoding="utf-8"?>
<dictionary name="Folkets" source-language="sv" target-language="en" version="test" last-changed="today" license="CC BY-SA 2.5">
  <word value="flest" lang="sv" class="jj"><translation value="most"/></word>
  <word value="fler" lang="sv" class="jj"><translation value="more"/><use value="komparativ"/><paradigm><inflection value="flest"/></paradigm></word>
  <word value="många" lang="sv" class="pn"><translation value="many"/><paradigm><inflection value="fler"/><inflection value="flest"/></paradigm></word>
  <word value="liten" lang="sv" class="jj"><translation value="small"/><paradigm><inflection value="litet"/><inflection value="lilla"/><inflection value="mindre"/><inflection value="minst"/></paradigm></word>
  <word value="små" lang="sv" class="jj"><translation value="small"/><use value="plural till &quot;liten&quot;"/></word>
  <word value="bok" lang="sv" class="nn"><translation value="book" comment="a bound publication"/><phonetic value="bu:k"/><variant value="bokform"/><grammar value="läsa &amp;"/><use value="concrete noun"/><synonym value="volym"/><compound value="skolbok"/><derivation value="boklig"/><explanation value="En tryckt eller digital publikation."/><paradigm><inflection value="boken"/><inflection value="böcker"/></paradigm><example value="Jag läser en bok."><translation value="I am reading a book."/></example></word>
  <word value="bok" lang="sv" class="vb"><translation value="book"/></word>
  <word value="bror|son" lang="sv" class="nn"><translation value="nephew"/><definition><translation value="the son of somebody's brother"/></definition></word>
  <word value="vara" lang="sv" class="vb"><translation value="last"/><paradigm><inflection value="varade"/><inflection value="varat"/><inflection value="vara"/><inflection value="varar"/></paradigm></word>
  <word value="vara" lang="sv" class="vb"><translation value="be"/><paradigm><inflection value="var"/><inflection value="varit"/><inflection value="var"/><inflection value="vara"/><inflection value="är"/></paradigm></word>
  <word value="fick" lang="sv" class=""><see value="få"/></word>
</dictionary>
"""


class FolketsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "folkets.xml"
        path.write_text(XML, encoding="utf-8")
        self.lexicon = FolketsLexicon.from_file(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_lookup_filters_pos_and_ignores_stub_entries(self) -> None:
        noun = self.lexicon.lookup("BOK", "noun")
        self.assertEqual(len(noun), 1)
        self.assertEqual(noun[0].word_class, "nn")
        self.assertEqual(self.lexicon.lookup("fick", "verb"), [])

    def test_definitions_keep_english_gloss_and_translated_example(self) -> None:
        definitions = self.lexicon.definitions(self.lexicon.lookup("bok", "noun"))
        self.assertEqual(
            definitions,
            [
                {
                    "definition": "book",
                    "example": "Jag läser en bok.",
                    "example_translation": "I am reading a book.",
                }
            ],
        )

    def test_precise_definition_translation_is_preserved_per_sense(self) -> None:
        definitions = self.lexicon.definitions(self.lexicon.lookup("brorson", "noun"))
        self.assertEqual(
            definitions,
            [
                {
                    "definition": "nephew",
                    "example": "",
                    "example_translation": "",
                    "source_hints": {
                        "sense_glosses": ["the son of somebody's brother"]
                    },
                }
            ],
        )

    def test_lexical_info_preserves_useful_non_gloss_fields(self) -> None:
        info = self.lexicon.lexical_info(self.lexicon.lookup("bok", "noun"))
        self.assertEqual(info["pronunciations"], ["bu:k"])
        self.assertEqual(info["variants"], ["bokform"])
        self.assertEqual(info["constructions"], ["läsa &"])
        self.assertEqual(info["translation_comments"], ["a bound publication"])
        self.assertEqual(info["explanations"], ["En tryckt eller digital publikation."])
        self.assertEqual(info["synonyms"], ["volym"])
        self.assertEqual(info["compounds"], ["skolbok"])
        self.assertEqual(info["derivations"], ["boklig"])

    def test_reverse_comparison_chain_improves_fler_and_flest(self) -> None:
        records = self.lexicon.lookup("flest", "adjective")
        forms, conflicts = self.lexicon.inflections("flest", "adjective", records)
        self.assertEqual(conflicts, {})
        self.assertEqual(
            forms, {"comparative": "fler", "superlative": "flest", "positive": "många"}
        )
        fler, _ = self.lexicon.inflections(
            "fler", "adjective", self.lexicon.lookup("fler", "adjective")
        )
        self.assertEqual(
            fler, {"superlative": "flest", "positive": "många", "comparative": "fler"}
        )

    def test_linked_exception_distinguishes_lilla_from_plural_sma(self) -> None:
        records = self.lexicon.lookup("liten", "adjective")
        forms, _ = self.lexicon.inflections("liten", "adjective", records)
        self.assertEqual(forms["definite"], "lilla")
        self.assertEqual(forms["plural"], "små")
        self.assertEqual(forms["comparative"], "mindre")

    def test_tied_conflicting_paradigms_are_reported_not_guessed(self) -> None:
        records = self.lexicon.lookup("vara", "verb")
        forms, conflicts = self.lexicon.inflections("vara", "verb", records)
        self.assertNotIn("past", forms)
        self.assertEqual(conflicts["past"], ["var", "varade"])

    def test_built_entry_retains_provenance(self) -> None:
        entry = build_entry("bok", "en", "noun", 10, "A1", self.lexicon)
        self.assertEqual(entry["frequency_rank"], 10)
        self.assertEqual(entry["tags"], ["noun"])
        self.assertEqual(entry["sources"]["frequency"]["cefr_band"], "A1")
        self.assertEqual(entry["inflections"]["plural"], "böcker")
        self.assertEqual(entry["sources"]["lexicon"]["version"], "test")
        self.assertEqual(entry["lexical_info"]["variants"], ["bokform"])
        self.assertEqual(entry["sources"]["lexicon"]["records"][0]["class"], "nn")


if __name__ == "__main__":
    unittest.main()
