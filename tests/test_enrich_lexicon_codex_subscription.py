from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import enrich_lexicon_codex_subscription as codex_enrichment
from enrich_lexicon_codex_subscription import (
    EXAMPLE_PROMPT,
    CardPlanTarget,
    DefinitionTarget,
    ExampleTarget,
    NoteTarget,
    apply,
    apply_cards,
    apply_definitions,
    apply_inflections,
    apply_note,
    collect_cards,
    collect_definitions,
    collect_examples,
    collect_inflections,
    collect_notes,
    collect_pending,
    ensure_codex_login,
    output_schema_for,
    propose,
    run_codex_batch,
    selected_chunks,
    validate_batch,
    validate_cards,
    validate_definitions,
    validate_examples,
    validate_notes,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class CodexExampleTests(unittest.TestCase):
    def test_example_targets_are_collected_resumably_by_kind(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            path = source / "words.json"
            write_json(
                path,
                {
                    "word": "överväga",
                    "tags": ["verb"],
                    "lexical_info": {
                        "constructions": ["& på något"],
                        "idioms": ["överväga sina ord"],
                        "explanations": ["Används om noggrann bedömning."],
                        "synonyms": ["fundera på"],
                        "pronunciations": ["ö:vervä:ga"],
                    },
                    "definitions": [
                        {
                            "definition": "consider",
                            "example": "",
                            "example_translation": "",
                        },
                        {
                            "definition": "weigh",
                            "example": "Vi överväger saken.",
                            "example_translation": "",
                            "source_hints": {"sense_glosses": ["consider carefully"]},
                        },
                    ],
                },
            )
            proposals = root / "proposals.jsonl"
            proposals.write_text(
                json.dumps(
                    {
                        "kind": "examples",
                        "id": "words.json:0",
                        "source": "words.json",
                        "word": "överväga",
                        "meaning": "consider",
                        "sentence": "Vi överväger förslaget.",
                        "sentence_meaning": "We are considering the proposal.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            targets, skipped = collect_examples(source, proposals)
            self.assertEqual(skipped, 0)
            self.assertEqual([target.id for target in targets], ["words.json:1"])
            self.assertEqual(targets[0].missing_fields, ("sentence_meaning", "note"))
            self.assertEqual(
                targets[0].request()["source_hints"],
                {
                    "constructions": ["& på något"],
                    "idioms": ["överväga sina ord"],
                    "explanations": ["Används om noggrann bedömning."],
                    "sense_glosses": ["consider carefully"],
                },
            )

    def test_all_kinds_share_one_source_and_proposal_scan(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            write_json(
                source / "word.json",
                {
                    "frequency_rank": 1,
                    "word": "bok",
                    "tags": ["noun"],
                    "learner_level": {"cefr": "C1"},
                    "definitions": [],
                    "inflections": {},
                },
            )
            write_json(
                source / "later-rank.json",
                {
                    "frequency_rank": 100,
                    "word": "hus",
                    "tags": ["noun"],
                    "learner_level": {"cefr": "A1"},
                    "definitions": [],
                    "inflections": {},
                },
            )
            proposals = root / "proposals.jsonl"
            with (
                patch.object(
                    codex_enrichment,
                    "load_json",
                    wraps=codex_enrichment.load_json,
                ) as load,
                patch.object(
                    codex_enrichment,
                    "completed_ids",
                    wraps=codex_enrichment.completed_ids,
                ) as completed,
            ):
                pending, _details = collect_pending(
                    source, proposals, codex_enrichment.ALL_KINDS
                )
            self.assertEqual(load.call_count, 2)
            self.assertEqual(completed.call_count, 1)
            self.assertEqual(
                [target.word for target in pending["definitions"]], ["hus", "bok"]
            )

    def test_definition_and_inflection_targets_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            write_json(
                source / "ett_numeral.json",
                {
                    "word": "ett",
                    "tags": ["numeral"],
                    "definitions": [],
                    "inflections": {},
                },
            )
            write_json(
                source / "abandonera_verb.json",
                {
                    "word": "abandonera",
                    "article": "",
                    "tags": ["verb"],
                    "lexical_info": {"constructions": ["A & x/att + SATS"]},
                    "definitions": [
                        {
                            "definition": "abandon",
                            "example": "De bestämde sig för att abandonera projektet.",
                            "example_translation": "They decided to abandon the project.",
                        }
                    ],
                    "inflections": {"past": "abandonerade"},
                },
            )
            # A complete, evidence-free sense must not cost a standalone request.
            write_json(
                source / "transparent_verb.json",
                {
                    "word": "transparent",
                    "tags": ["verb"],
                    "definitions": [
                        {
                            "definition": "transparent",
                            "example": "Det är transparent.",
                            "example_translation": "It is transparent.",
                        }
                    ],
                    "inflections": {"past": "transparentade"},
                },
            )
            proposals = root / "proposals.jsonl"
            self.assertEqual(
                [target.id for target in collect_definitions(source, proposals)],
                ["ett_numeral.json:definitions"],
            )
            self.assertEqual(
                [target.id for target in collect_examples(source, proposals)[0]], []
            )
            self.assertEqual(
                [target.id for target in collect_notes(source, proposals)],
                ["abandonera_verb.json:0:note", "transparent_verb.json:0:note"],
            )
            self.assertEqual(
                [
                    target.id
                    for target in collect_notes(
                        source, proposals, notes_scope="evidence"
                    )
                ],
                ["abandonera_verb.json:0:note"],
            )
            inflections = collect_inflections(source, proposals)
            self.assertEqual(
                [target.id for target in inflections],
                [
                    "abandonera_verb.json:inflections",
                    "transparent_verb.json:inflections",
                ],
            )
            self.assertEqual(
                inflections[0].missing,
                ("present", "supine", "imperative", "participle"),
            )

    def test_propose_runs_batches_concurrently_with_one_jsonl_writer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            for rank, word in enumerate(("bok", "hus"), start=1):
                write_json(
                    source / f"{word}.json",
                    {
                        "frequency_rank": rank,
                        "word": word,
                        "article": "ett" if word == "hus" else "en",
                        "tags": ["noun"],
                        "definitions": [],
                    },
                )

            active = peak = 0
            lock = threading.Lock()

            def concurrent_runner(targets: list[object], **kwargs: object) -> object:
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                target = targets[0]
                with lock:
                    active -= 1
                return {
                    target.id: {
                        "definitions": [
                            {
                                "meaning": f"meaning of {target.word}",
                                "sentence": f"Det här är {target.word}.",
                                "sentence_meaning": f"This is {target.word}.",
                                "note": None,
                            }
                        ]
                    }
                }

            proposals = root / "proposals.jsonl"
            args = argparse.Namespace(
                source=source,
                proposals=proposals,
                failures=root / "failures.jsonl",
                model="model",
                codex_command="codex",
                batch_size=1,
                workers=2,
                notes_scope="all",
                kind=["definitions"],
                max_batches=0,
                limit=None,
                timeout=60,
                dry_run=False,
            )
            with (
                patch.object(codex_enrichment, "ensure_codex_login"),
                patch.object(
                    codex_enrichment,
                    "run_codex_batch",
                    side_effect=concurrent_runner,
                ),
            ):
                self.assertEqual(propose(args), 0)

            self.assertEqual(peak, 2)
            records = [
                json.loads(line)
                for line in proposals.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual({record["word"] for record in records}, {"bok", "hus"})

    def test_login_preflight_requires_subscription_auth(self) -> None:
        def not_logged_in(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 1, "Not logged in", "")

        with self.assertRaisesRegex(RuntimeError, "codex login"):
            ensure_codex_login("codex", runner=not_logged_in)

        def api_key_login(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                command, 0, "Logged in using an API key", ""
            )

        with self.assertRaisesRegex(RuntimeError, "not a ChatGPT subscription"):
            ensure_codex_login("codex", runner=api_key_login)

        def subscription_login(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                command, 0, "Logged in using ChatGPT", ""
            )

        ensure_codex_login("codex", runner=subscription_login)

    def test_codex_invocation_parses_output_file_and_validates(self) -> None:
        targets = [
            ExampleTarget(
                id="entry.json:0",
                source="entry.json",
                definition_index=0,
                word="försiktig",
                article="",
                part_of_speech="adjective",
                meaning="careful",
                sentence="",
                sentence_meaning="",
            )
        ]

        def runner(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            self.assertIn("--ephemeral", command)
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertIn("--output-last-message", command)
            self.assertIn("--output-schema", command)
            schema_path = Path(command[command.index("--output-schema") + 1])
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self.assertEqual(schema["required"], ["entry.json:0"])
            self.assertFalse(schema["additionalProperties"])
            request = json.loads(str(kwargs["input"]))
            self.assertEqual(request[0]["word"], "försiktig")
            output = {
                "entry.json:0": {
                    "sentence": "Hon var försiktig på isen.",
                    "sentence_meaning": "She was careful on the ice.",
                }
            }
            response_path = Path(command[command.index("--output-last-message") + 1])
            response_path.write_text(
                json.dumps(output, ensure_ascii=False), encoding="utf-8"
            )
            return subprocess.CompletedProcess(
                command, 0, "non-JSON progress output", "cache warning"
            )

        response = run_codex_batch(
            targets,
            model="gpt-5.6-luna",
            codex_command="codex",
            timeout=60,
            prompt=EXAMPLE_PROMPT,
            runner=runner,
        )
        validated = validate_examples(targets, response)
        self.assertEqual(
            validated["entry.json:0"]["sentence_meaning"],
            "She was careful on the ice.",
        )

    def test_translation_only_response_cannot_return_existing_sentence(self) -> None:
        target = ExampleTarget(
            id="entry.json:0",
            source="entry.json",
            definition_index=0,
            word="ord",
            article="ett",
            part_of_speech="noun",
            meaning="word",
            sentence="Original.",
            sentence_meaning="",
        )
        response = {
            "entry.json:0": {"sentence": "Changed.", "sentence_meaning": "Changed."}
        }
        with self.assertRaisesRegex(ValueError, "response fields differ"):
            validate_examples([target], response)
        validated = validate_examples(
            [target],
            {"entry.json:0": {"sentence_meaning": "An English translation."}},
        )
        self.assertEqual(validated["entry.json:0"]["sentence"], "Original.")
        schema = output_schema_for([target])
        reference = schema["properties"][target.id]["$ref"]
        value_schema = schema["$defs"][reference.rsplit("/", 1)[-1]]
        self.assertEqual(value_schema["required"], ["sentence_meaning"])

    def test_batch_validation_salvages_valid_targets(self) -> None:
        valid = ExampleTarget(
            id="valid.json:0",
            source="valid.json",
            definition_index=0,
            word="bok",
            article="en",
            part_of_speech="noun",
            meaning="book",
            sentence="",
            sentence_meaning="",
        )
        invalid = ExampleTarget(
            id="invalid.json:0",
            source="invalid.json",
            definition_index=0,
            word="grabb",
            article="en",
            part_of_speech="noun",
            meaning="boy",
            sentence="grabbar (tag i)",
            sentence_meaning="",
        )
        result = validate_batch(
            "examples",
            [valid, invalid],
            {
                valid.id: {
                    "sentence": "Jag läser en bok.",
                    "sentence_meaning": "I am reading a book.",
                },
                invalid.id: {
                    "sentence": "Grabbar, tag i!",
                    "sentence_meaning": "Come on, guys!",
                },
            },
        )
        self.assertEqual(set(result.valid), {valid.id})
        self.assertEqual([issue.id for issue in result.issues], [invalid.id])

    def test_fenced_final_message_is_accepted(self) -> None:
        target = ExampleTarget(
            id="entry.json:0",
            source="entry.json",
            definition_index=0,
            word="ord",
            article="ett",
            part_of_speech="noun",
            meaning="word",
            sentence="",
            sentence_meaning="word",
        )

        def runner(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            output = {"entry.json:0": {"sentence": "Det är ett ord."}}
            response_path = Path(command[command.index("--output-last-message") + 1])
            response_path.write_text(
                "Here is the requested data:\n```json\n"
                + json.dumps(output, ensure_ascii=False)
                + "\n```\nDone.",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        response = run_codex_batch(
            [target],
            model="model",
            codex_command="codex",
            timeout=60,
            prompt=EXAMPLE_PROMPT,
            runner=runner,
        )
        self.assertEqual(
            validate_examples([target], response)["entry.json:0"]["sentence"],
            "Det är ett ord.",
        )

    def test_card_batches_are_capped_below_short_lexical_batches(self) -> None:
        target = CardPlanTarget(
            id="word.json:cards-v1",
            source="word.json",
            source_digest="digest",
            word="ord",
            article="ett",
            part_of_speech="noun",
            frequency_rank=1,
            learner_level="A1",
            definitions=({"index": 0},),
            allowed_target_forms=("ord",),
            production_eligible=False,
            production_score=0,
            production_reasons=(),
        )
        schema = output_schema_for([target])
        reference = schema["properties"][target.id]["$ref"]
        value_schema = schema["$defs"][reference.removeprefix("#/$defs/")]
        source_indices = value_schema["properties"]["cards"]["items"]["properties"][
            "source_indices"
        ]
        self.assertNotIn("uniqueItems", source_indices)

        selected = selected_chunks(
            {"cards": [target] * 250},
            ["cards"],
            batch_size=250,
            limit=None,
            max_batches=0,
        )
        self.assertEqual([len(batch) for _kind, batch in selected], [100, 100, 50])

    def test_card_review_can_merge_redundant_senses_and_validate_target(self) -> None:
        target = CardPlanTarget(
            id="ett_modersmål_noun.json:cards-v1",
            source="ett_modersmål_noun.json",
            source_digest="digest",
            word="modersmål",
            article="ett",
            part_of_speech="noun",
            frequency_rank=5038,
            learner_level="A2",
            definitions=(
                {"index": 0, "meaning": "mother tongue"},
                {"index": 1, "meaning": "mother tongue; native language"},
            ),
            allowed_target_forms=("modersmål", "modersmålet"),
            production_eligible=False,
            production_score=0,
            production_reasons=(),
        )
        response = {
            target.id: {
                "cards": [
                    {
                        "source_indices": [0, 1],
                        "context_meaning": "native language",
                        "other_meanings": ["mother tongue"],
                        "sentence": "Svenska är hennes modersmål.",
                        "sentence_meaning": "Swedish is her native language.",
                        "target_form": "modersmål",
                        "note": None,
                        "production": None,
                    }
                ]
            }
        }
        validated = validate_cards([target], response)
        cards = validated[target.id]["cards"]
        self.assertEqual(cards[0]["source_indices"], [0, 1])  # type: ignore[index]

        invalid = copy.deepcopy(response)
        invalid[target.id]["cards"][0]["target_form"] = "modersmålet"
        with self.assertRaisesRegex(ValueError, "does not occur"):
            validate_cards([target], invalid)

    def test_card_review_collects_even_structurally_simple_entries(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            write_json(
                source / "bok_noun.json",
                {
                    "frequency_rank": 20,
                    "article": "en",
                    "word": "bok",
                    "tags": ["noun"],
                    "definitions": [
                        {
                            "definition": "book",
                            "example": "Jag läser en bok.",
                            "example_translation": "I am reading a book.",
                            "note": None,
                        }
                    ],
                },
            )
            targets, incomplete = collect_cards(
                source, root / "proposals.jsonl", production_limit=0
            )
            self.assertEqual(incomplete, 0)
            self.assertEqual([target.source for target in targets], ["bok_noun.json"])

    def test_collect_and_apply_card_review_preserves_source_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            path = source / "vi_pronoun.json"
            entry = {
                "frequency_rank": 112,
                "word": "vi",
                "article": "",
                "tags": ["pronoun"],
                "learner_level": {"cefr": "A1"},
                "definitions": [
                    {
                        "definition": "we",
                        "example": "Han har sett oss.",
                        "example_translation": "He has seen us.",
                        "note": None,
                    }
                ],
                "inflections": {},
            }
            write_json(path, entry)
            targets, incomplete = collect_cards(
                source, root / "proposals.jsonl", production_limit=0
            )
            self.assertEqual(incomplete, 0)
            self.assertEqual(len(targets), 1)
            target = targets[0]
            response = {
                target.id: {
                    "cards": [
                        {
                            "source_indices": [0],
                            "context_meaning": "we",
                            "other_meanings": [],
                            "sentence": "Vi arbetar tillsammans.",
                            "sentence_meaning": "We work together.",
                            "target_form": "Vi",
                            "note": None,
                            "production": None,
                        }
                    ]
                }
            }
            validated = validate_cards([target], response)
            record = target.row(validated[target.id], "model")
            source_copy = copy.deepcopy(entry["definitions"])
            self.assertTrue(apply_cards(entry, record))
            self.assertEqual(entry["definitions"], source_copy)
            self.assertEqual(
                entry["card_plan"]["cards"][0]["sentence"],  # type: ignore[index]
                "Vi arbetar tillsammans.",
            )
            self.assertFalse(apply_cards(entry, record))

    def test_definitions_apply_only_when_source_has_none(self) -> None:
        entry = {"word": "hundratusen", "definitions": []}
        target = DefinitionTarget(
            id="hundratusen_numeral.json:definitions",
            source="hundratusen_numeral.json",
            word="hundratusen",
            article="",
            part_of_speech="numeral",
        )
        response = {
            target.id: {
                "definitions": [
                    {
                        "meaning": "hundred thousand",
                        "sentence": "Staden har över hundratusen invånare.",
                        "sentence_meaning": "The city has more than a hundred thousand inhabitants.",
                        "note": None,
                    }
                ]
            }
        }
        validated = validate_definitions([target], response)
        record = target.row(validated[target.id], "model")
        self.assertTrue(apply_definitions(entry, record))
        self.assertEqual(entry["definitions"][0]["definition"], "hundred thousand")
        self.assertIsNone(entry["definitions"][0]["note"])
        self.assertFalse(apply_definitions(entry, record))
        entry["definitions"][0]["definition"] = "different"
        self.assertRaises(ValueError, apply_definitions, entry, record)

    def test_note_assessment_is_sense_specific_and_records_null(self) -> None:
        target = NoteTarget(
            id="en_hela_noun.json:0:note",
            source="en_hela_noun.json",
            definition_index=0,
            word="hela",
            article="en",
            part_of_speech="noun",
            meaning="a full bottle of spirits",
            sentence="Han beställde en hela.",
            sentence_meaning="He ordered a full bottle of spirits.",
        )
        validated = validate_notes(
            [target],
            {
                target.id: {
                    "note": "A fixed noun use in traditional drinking contexts; not the adjective hela (‘whole’)."
                }
            },
        )
        record = target.row(validated[target.id], "model")
        entry = {
            "word": "hela",
            "definitions": [
                {
                    "definition": "a full bottle of spirits",
                    "example": "Han beställde en hela.",
                    "example_translation": "He ordered a full bottle of spirits.",
                }
            ],
        }
        self.assertTrue(apply_note(entry, record))
        self.assertEqual(
            entry["definitions"][0]["note"],
            "A fixed noun use in traditional drinking contexts; not the adjective hela (‘whole’).",
        )
        self.assertFalse(apply_note(entry, record))
        conflicting = dict(record, note="Different usage.")
        self.assertRaises(ValueError, apply_note, entry, conflicting)

        null_entry = {
            "word": "bok",
            "definitions": [
                {
                    "definition": "book",
                    "example": "Jag läser en bok.",
                    "example_translation": "I am reading a book.",
                }
            ],
        }
        null_target = NoteTarget(
            id="en_bok_noun.json:0:note",
            source="en_bok_noun.json",
            definition_index=0,
            word="bok",
            article="en",
            part_of_speech="noun",
            meaning="book",
            sentence="Jag läser en bok.",
            sentence_meaning="I am reading a book.",
        )
        null_record = null_target.row(
            validate_notes([null_target], {null_target.id: {"note": None}})[
                null_target.id
            ],
            "model",
        )
        self.assertTrue(apply_note(null_entry, null_record))
        self.assertIn("note", null_entry["definitions"][0])
        self.assertIsNone(null_entry["definitions"][0]["note"])

    def test_note_validator_rejects_overlong_notes(self) -> None:
        target = NoteTarget(
            id="entry.json:0:note",
            source="entry.json",
            definition_index=0,
            word="ord",
            article="ett",
            part_of_speech="noun",
            meaning="word",
            sentence="",
            sentence_meaning="",
        )
        with self.assertRaisesRegex(ValueError, "exceeds 180"):
            validate_notes([target], {target.id: {"note": "x" * 181}})

    def test_inflections_apply_only_fills_allowed_empty_keys(self) -> None:
        entry = {"word": "skriva", "tags": ["verb"], "inflections": {}}
        record = {
            "kind": "inflections",
            "word": "skriva",
            "forms": {"present": "skriver", "past": None, "nonsense": "x"},
        }
        self.assertRaises(ValueError, apply_inflections, entry, record)
        record["forms"] = {"present": "skriver", "past": "skrev", "supine": "skrivit"}
        self.assertTrue(apply_inflections(entry, record))
        self.assertEqual(entry["inflections"]["present"], "skriver")
        self.assertFalse(apply_inflections(entry, record))
        conflicting = dict(record, forms={"present": "skriver fel"})
        self.assertRaises(ValueError, apply_inflections, entry, conflicting)

    def test_apply_dispatches_examples_and_fills_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            entry_path = source / "entry.json"
            write_json(
                entry_path,
                {
                    "word": "abandonera",
                    "definitions": [
                        {
                            "definition": "abandon",
                            "example": "",
                            "example_translation": "",
                        }
                    ],
                },
            )
            proposals = root / "proposals.jsonl"
            proposals.write_text(
                json.dumps(
                    {
                        "kind": "examples",
                        "id": "entry.json:0",
                        "source": "entry.json",
                        "definition_index": 0,
                        "word": "abandonera",
                        "meaning": "abandon",
                        "sentence": "Företaget beslutade att abandonera projektet.",
                        "sentence_meaning": "The company decided to abandon the project.",
                        "note": None,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            result = apply(
                argparse.Namespace(source=source, proposals=proposals, apply=True)
            )
            self.assertEqual(result, 0)
            updated = json.loads(entry_path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated["definitions"][0]["example_translation"],
                "The company decided to abandon the project.",
            )
            self.assertIn("note", updated["definitions"][0])
            self.assertIsNone(updated["definitions"][0]["note"])
            self.assertFalse(entry_path.with_suffix(".json.part").exists())


if __name__ == "__main__":
    unittest.main()
