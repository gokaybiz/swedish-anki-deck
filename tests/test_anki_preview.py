from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json_to_anki
from functions.card_quality import CARD_PLAN_VERSION, card_source_digest


class AnkiPreviewTests(unittest.TestCase):
    def test_templates_keep_answers_and_rank_off_the_recognition_front(self) -> None:
        front = json_to_anki.RECOGNITION_FRONT
        back = json_to_anki.RECOGNITION_BACK
        self.assertNotIn("{{Inflections}}", front)
        self.assertIn("{{Inflections}}", back)
        self.assertIn("{{Context Meaning}}", back)
        self.assertNotIn("{{Context Meaning}}", front)
        self.assertIn("{{Usage Note}}", back)
        self.assertNotIn("{{Usage Note}}", front)
        self.assertIn("{{Learner Level}}", back)
        self.assertNotIn("{{Learner Level}}", front)
        self.assertIn("{{Variants}}", back)
        self.assertNotIn("{{Variants}}", front)
        self.assertIn("{{Frequency Order}}", back)
        self.assertNotIn("{{Frequency Order}}", front)
        self.assertNotIn("Production Cue", json_to_anki.VOCABULARY_FIELDS)
        self.assertNotIn("Production Answer", json_to_anki.VOCABULARY_FIELDS)
        self.assertIn("{{Cue}}", json_to_anki.CHUNK_FRONT)
        self.assertIn("{{Swedish Answer}}", json_to_anki.CHUNK_BACK)

    def test_note_fields_use_the_new_semantic_names(self) -> None:
        entry = {
            "word": "överväga",
            "definitions": [
                {
                    "definition": "consider",
                    "example": "Vi överväger saken.",
                    "example_translation": "We are considering it.",
                    "note": "Often followed by a noun phrase or att-clause.",
                },
                {
                    "definition": "weigh",
                    "example": "Överväg riskerna.",
                    "example_translation": "Weigh the risks.",
                    "note": None,
                },
            ],
            "tags": ["verb"],
            "learner_level": {"cefr": "B1"},
            "lexical_info": {"variants": ["överväga", "ÖVERVÄGA", "överväga sig"]},
        }
        card = {
            "source_indices": [0],
            "context_meaning": "consider",
            "other_meanings": ["weigh carefully"],
            "sentence": "Vi överväger saken.",
            "sentence_meaning": "We are considering it.",
            "target_form": "överväger",
            "note": "Often followed by a noun phrase or att-clause.",
            "production": {
                "cue": "Say that we are considering the matter (use överväga).",
                "answer": "Vi överväger saken.",
                "target": "överväger",
            },
        }
        fields = json_to_anki.vocabulary_fields(
            entry, card, "kelly:överväga:0", 1, "Kelly List", None
        )
        self.assertEqual(fields["Context Meaning"], "consider")
        self.assertEqual(fields["Sentence Meaning"], "We are considering it.")
        self.assertEqual(
            fields["Usage Note"], "Often followed by a noun phrase or att-clause."
        )
        self.assertEqual(fields["Other Meanings"], "weigh carefully")
        self.assertIn('<span class="target">överväger</span>', fields["Sentence"])
        self.assertNotIn("Production Cue", fields)
        self.assertNotIn("Production Answer", fields)
        self.assertEqual(fields["Card ID"], "kelly:överväga:0")
        self.assertEqual(fields["Learner Level"], "B1")
        self.assertEqual(fields["Variants"], "överväga sig")
        self.assertEqual(list(fields), json_to_anki.VOCABULARY_FIELDS)
        chunk = json_to_anki.chunk_fields(
            entry, card, "kelly:överväga:0", 1, "Kelly List", None
        )
        self.assertIsNotNone(chunk)
        assert chunk is not None
        self.assertIn('<span class="target">överväger</span>', chunk["Swedish Answer"])
        self.assertEqual(list(chunk), json_to_anki.CHUNK_FIELDS)

    def test_existing_generated_model_gets_current_template_and_style(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_anki(action: str, **params: object) -> object:
            calls.append((action, params))
            if action == "modelNames":
                return [json_to_anki.VOCABULARY_MODEL_NAME]
            if action == "modelFieldNames":
                return json_to_anki.VOCABULARY_FIELDS
            return None

        with patch.object(json_to_anki, "anki", side_effect=fake_anki):
            json_to_anki.ensure_model(
                json_to_anki.VOCABULARY_MODEL_NAME,
                json_to_anki.VOCABULARY_FIELDS,
                json_to_anki.RECOGNITION_CARD_NAME,
                json_to_anki.RECOGNITION_FRONT,
                json_to_anki.RECOGNITION_BACK,
            )

        self.assertEqual(
            [action for action, _params in calls],
            [
                "modelNames",
                "modelFieldNames",
                "updateModelTemplates",
                "updateModelStyling",
            ],
        )
        template_model = calls[2][1]["model"]
        self.assertEqual(  # type: ignore[index]
            template_model["name"], json_to_anki.VOCABULARY_MODEL_NAME
        )
        self.assertEqual(
            set(template_model["templates"]),  # type: ignore[index]
            {json_to_anki.RECOGNITION_CARD_NAME},
        )

    def test_existing_model_with_other_fields_is_rejected_without_migration(
        self,
    ) -> None:
        def fake_anki(action: str, **params: object) -> object:
            if action == "modelNames":
                return [json_to_anki.VOCABULARY_MODEL_NAME]
            if action == "modelFieldNames":
                return ["Word", "Old Meaning"]
            self.fail(f"Unexpected mutating action: {action}")

        with (
            patch.object(json_to_anki, "anki", side_effect=fake_anki),
            self.assertRaisesRegex(RuntimeError, "unsupported fields"),
        ):
            json_to_anki.ensure_model(
                json_to_anki.VOCABULARY_MODEL_NAME,
                json_to_anki.VOCABULARY_FIELDS,
                json_to_anki.RECOGNITION_CARD_NAME,
                json_to_anki.RECOGNITION_FRONT,
                json_to_anki.RECOGNITION_BACK,
            )

    def test_card_id_identifies_existing_sense_note(self) -> None:
        def fake_anki(action: str, **params: object) -> object:
            if action == "findNotes":
                return [42]
            if action == "notesInfo":
                return [{"noteId": 42, "fields": {"Word": {"value": "old word"}}}]
            self.fail(f"Unexpected action: {action}")

        fields = {
            "Card ID": "kelly-list:word.json:0",
            "Frequency Order": "7",
            "Word": "corrected word",
            "Source Type": "Kelly List",
        }
        with patch.object(json_to_anki, "anki", side_effect=fake_anki):
            self.assertEqual(
                json_to_anki.existing_note_id(
                    json_to_anki.VOCABULARY_MODEL_NAME,
                    "Card ID",
                    fields["Card ID"],
                ),
                42,
            )

    def test_apply_suppresses_exact_tasks_but_keeps_distinct_same_word_senses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            core = Path(folder) / "json"
            core.mkdir()
            records = (
                (1, "determiner", "Det var jag.", "It was me."),
                (2, "pronoun", "Det var jag.", "It was me."),
                (3, "particle", "Det regnar.", "It is raining."),
            )
            for rank, pos, sentence, translation in records:
                entry = {
                    "frequency_rank": rank,
                    "word": "det",
                    "definitions": [
                        {
                            "definition": "it",
                            "example": sentence,
                            "example_translation": translation,
                            "note": None,
                        }
                    ],
                    "tags": [pos],
                }
                entry["card_plan"] = {
                    "version": CARD_PLAN_VERSION,
                    "source_digest": card_source_digest(entry),
                    "reviewed_by": "test",
                    "cards": [
                        {
                            "source_indices": [0],
                            "context_meaning": "it",
                            "other_meanings": [],
                            "sentence": sentence,
                            "sentence_meaning": translation,
                            "target_form": "Det",
                            "note": None,
                            "production": None,
                        }
                    ],
                }
                (core / f"det_{pos}.json").write_text(
                    json.dumps(entry), encoding="utf-8"
                )
            added: list[dict[str, object]] = []

            def fake_anki(action: str, **params: object) -> object:
                if action == "modelNames":
                    return []
                if action == "findNotes":
                    return []
                if action == "addNote":
                    added.append(params["note"])  # type: ignore[arg-type]
                    return 1
                return None

            argv = ["json_to_anki.py", "--core", str(core), "--apply"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(json_to_anki, "anki", side_effect=fake_anki),
                redirect_stdout(StringIO()),
            ):
                json_to_anki.main()

            self.assertEqual(len(added), 2)
            self.assertEqual(
                {note["fields"]["Sentence"] for note in added},  # type: ignore[index]
                {
                    '<span class="target">Det</span> var jag.',
                    '<span class="target">Det</span> regnar.',
                },
            )
            self.assertTrue(
                all(note["options"] == {"allowDuplicate": True} for note in added)
            )
            merged = next(
                note
                for note in added
                if "var jag" in note["fields"]["Sentence"]  # type: ignore[index]
            )
            self.assertEqual(
                merged["fields"]["Part of Speech"],  # type: ignore[index]
                "determiner / pronoun",
            )
            self.assertEqual(
                merged["fields"]["Source IDs"],  # type: ignore[index]
                "kelly:1:0 · kelly:2:0",
            )

    def test_distinct_reviewed_senses_become_separate_notes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            core = Path(folder) / "json"
            core.mkdir()
            entry = {
                "frequency_rank": 2812,
                "article": "att",
                "word": "avgå",
                "tags": ["verb"],
                "learner_level": {"cefr": "B1"},
                "definitions": [
                    {
                        "definition": "resign; retire",
                        "example": "Regeringen tvingades avgå.",
                        "example_translation": "The government was forced to resign.",
                        "note": None,
                    },
                    {
                        "definition": "leave; depart",
                        "example": "Tåget avgår klockan elva.",
                        "example_translation": "The train departs at eleven.",
                        "note": None,
                    },
                ],
                "inflections": {"present": "avgår", "past": "avgick"},
            }
            entry["card_plan"] = {
                "version": CARD_PLAN_VERSION,
                "source_digest": card_source_digest(entry),
                "reviewed_by": "test",
                "cards": [
                    {
                        "source_indices": [0],
                        "context_meaning": "resign",
                        "other_meanings": ["step down"],
                        "sentence": "Regeringen tvingades avgå.",
                        "sentence_meaning": "The government was forced to resign.",
                        "target_form": "avgå",
                        "note": None,
                        "production": None,
                    },
                    {
                        "source_indices": [1],
                        "context_meaning": "depart",
                        "other_meanings": ["leave"],
                        "sentence": "Tåget avgår klockan elva.",
                        "sentence_meaning": "The train departs at eleven.",
                        "target_form": "avgår",
                        "note": None,
                        "production": None,
                    },
                ],
            }
            (core / "avgå_verb.json").write_text(
                json.dumps(entry, ensure_ascii=False), encoding="utf-8"
            )
            added: list[dict[str, object]] = []

            def fake_anki(action: str, **params: object) -> object:
                if action == "modelNames":
                    return []
                if action == "findNotes":
                    return []
                if action == "addNote":
                    added.append(params["note"])  # type: ignore[arg-type]
                    return 1
                return None

            argv = ["json_to_anki.py", "--core", str(core), "--apply"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(json_to_anki, "anki", side_effect=fake_anki),
                redirect_stdout(StringIO()),
            ):
                json_to_anki.main()
            self.assertEqual(len(added), 2)
            self.assertEqual(
                {note["fields"]["Context Meaning"] for note in added},  # type: ignore[index]
                {"resign", "depart"},
            )

    def test_approved_production_item_becomes_a_dedicated_chunk_note(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            core = Path(folder)
            entry = {
                "frequency_rank": 120,
                "article": "att",
                "word": "anmäla",
                "tags": ["verb"],
                "learner_level": {"cefr": "A2"},
                "definitions": [
                    {
                        "definition": "register",
                        "example": "Jag anmäler mig till kursen.",
                        "example_translation": "I register for the course.",
                        "note": None,
                    }
                ],
            }
            entry["card_plan"] = {
                "version": CARD_PLAN_VERSION,
                "source_digest": card_source_digest(entry),
                "reviewed_by": "test",
                "cards": [
                    {
                        "source_indices": [0],
                        "context_meaning": "register",
                        "other_meanings": [],
                        "sentence": "Jag anmäler mig till kursen.",
                        "sentence_meaning": "I register for the course.",
                        "target_form": "anmäler",
                        "note": None,
                        "production": {
                            "cue": "Say that you are registering for the course.",
                            "answer": "Jag anmäler mig till kursen.",
                            "target": "anmäler mig till",
                        },
                    }
                ],
            }
            (core / "anmäla_verb.json").write_text(
                json.dumps(entry, ensure_ascii=False), encoding="utf-8"
            )
            added: list[dict[str, object]] = []

            def fake_anki(action: str, **params: object) -> object:
                if action == "modelNames":
                    return []
                if action == "findNotes":
                    return []
                if action == "addNote":
                    added.append(params["note"])  # type: ignore[arg-type]
                    return len(added)
                return None

            with (
                patch.object(
                    sys,
                    "argv",
                    ["json_to_anki.py", "--core", str(core), "--apply"],
                ),
                patch.object(json_to_anki, "anki", side_effect=fake_anki),
                redirect_stdout(StringIO()),
            ):
                json_to_anki.main()

            self.assertEqual(len(added), 2)
            by_model = {note["modelName"]: note for note in added}
            vocabulary = by_model[json_to_anki.VOCABULARY_MODEL_NAME]
            chunk = by_model[json_to_anki.CHUNK_MODEL_NAME]
            self.assertEqual(list(vocabulary["fields"]), json_to_anki.VOCABULARY_FIELDS)
            self.assertNotIn("Cue", vocabulary["fields"])
            self.assertEqual(list(chunk["fields"]), json_to_anki.CHUNK_FIELDS)
            self.assertEqual(
                chunk["fields"]["Cue"],
                "Say that you are registering for the course.",
            )
            self.assertEqual(chunk["deckName"], "Swedish Vocabulary::Chunks")

    def test_unreviewed_risky_entry_is_blocked_before_anki(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            core = Path(folder)
            (core / "vi.json").write_text(
                json.dumps(
                    {
                        "frequency_rank": 1,
                        "word": "vi",
                        "tags": ["pronoun"],
                        "definitions": [
                            {
                                "definition": "we",
                                "example": "Han såg oss.",
                                "example_translation": "He saw us.",
                                "note": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            argv = ["json_to_anki.py", "--core", str(core)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    json_to_anki,
                    "anki",
                    side_effect=AssertionError("AnkiConnect must not be called"),
                ),
                redirect_stderr(StringIO()) as error,
                self.assertRaises(SystemExit),
            ):
                json_to_anki.main()
            self.assertIn("require holistic card review", error.getvalue())

    def test_preview_never_calls_ankiconnect(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            core = Path(folder) / "json"
            core.mkdir()
            entry = {
                "frequency_rank": 1,
                "article": "en",
                "word": "bok",
                "definitions": [
                    {
                        "definition": "book",
                        "example": "Jag läser en bok.",
                        "example_translation": "I am reading a book.",
                        "note": None,
                    }
                ],
                "inflections": {"definite": "boken", "plural": "böcker"},
                "tags": ["A1", "noun"],
            }
            entry["card_plan"] = {
                "version": CARD_PLAN_VERSION,
                "source_digest": card_source_digest(entry),
                "reviewed_by": "test",
                "cards": [
                    {
                        "source_indices": [0],
                        "context_meaning": "book",
                        "other_meanings": [],
                        "sentence": "Jag läser en bok.",
                        "sentence_meaning": "I am reading a book.",
                        "target_form": "bok",
                        "note": None,
                        "production": None,
                    }
                ],
            }
            (core / "bok_noun.json").write_text(json.dumps(entry), encoding="utf-8")
            argv = ["json_to_anki.py", "--core", str(core)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    json_to_anki,
                    "anki",
                    side_effect=AssertionError("AnkiConnect must not be called"),
                ),
                redirect_stdout(StringIO()) as output,
            ):
                json_to_anki.main()
            self.assertIn("Preview only", output.getvalue())

    def test_manifest_backed_audio_is_selected_by_source_stem(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder)
            clip = audio / "bok_noun__0123456789abcdef.mp3"
            clip.write_bytes(b"ID3")
            # A prior content fingerprint may remain after text/voice changes;
            # the manifest, not directory order, selects the current artifact.
            (audio / "bok_noun__fedcba9876543210.mp3").write_bytes(b"old")
            (audio / "audio_manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": {
                            "bok_noun": {
                                "file": clip.name,
                                "fingerprint": "0123456789abcdef",
                                "sha256": hashlib.sha256(b"ID3").hexdigest(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(json_to_anki.audio_index([audio]), {"bok_noun": clip})


if __name__ == "__main__":
    unittest.main()
