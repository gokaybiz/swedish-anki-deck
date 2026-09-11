from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import enrich_lexicon_openai_batch as enrich_json
from functions.enrichment_quality import ENRICHMENT_QUALITY_POLICY


class EnrichJsonTests(unittest.TestCase):
    def entry(self) -> dict[str, object]:
        return {
            "word": "hela",
            "article": "en",
            "tags": ["noun"],
            "definitions": [
                {
                    "definition": "a full bottle of spirits",
                    "example": "Han beställde en hela.",
                    "example_translation": "He ordered a full bottle of spirits.",
                }
            ],
            "inflections": {"plural": "helor"},
        }

    def test_pending_batch_entries_are_loaded_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "json"
            source.mkdir()
            path = source / "entry.json"
            entry = self.entry()
            entry["inflections"] = {}
            path.write_text(json.dumps(entry), encoding="utf-8")
            with patch.object(
                enrich_json, "load_json", wraps=enrich_json.load_json
            ) as load:
                pending = enrich_json.pending_entries(
                    source, Path(folder) / "proposals.jsonl"
                )
            self.assertEqual([item[0] for item in pending], [path])
            self.assertEqual(load.call_count, 1)

    def test_api_proposal_resumption_requires_version_and_compatible_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            proposals = Path(folder) / "proposals.jsonl"
            proposals.write_text(
                json.dumps({"source": "legacy.json", "patch": {}})
                + "\n"
                + json.dumps(
                    {
                        "quality_version": enrich_json.QUALITY_VERSION,
                        "notes_scope": "evidence",
                        "source": "evidence.json",
                        "patch": {},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "quality_version": enrich_json.QUALITY_VERSION,
                        "notes_scope": "all",
                        "source": "all.json",
                        "patch": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(enrich_json.proposed_sources(proposals), {"all.json"})
            self.assertEqual(
                enrich_json.proposed_sources(proposals, notes_scope="evidence"),
                {"all.json", "evidence.json"},
            )

    def test_api_backend_uses_the_shared_quality_contract(self) -> None:
        self.assertIn(ENRICHMENT_QUALITY_POLICY, enrich_json.SYSTEM_PROMPT)
        new_definition = enrich_json.PATCH_SCHEMA["properties"]["new_definitions"][
            "items"
        ]
        update = enrich_json.PATCH_SCHEMA["properties"]["definition_updates"]["items"]
        self.assertIn("note", new_definition["required"])
        self.assertIn("note", update["required"])

    def test_task_includes_bounded_lexical_hints_but_not_pronunciation(self) -> None:
        entry = self.entry()
        entry["definitions"][0]["source_hints"] = {  # type: ignore[index]
            "sense_glosses": ["a complete bottle of spirits"]
        }
        entry["lexical_info"] = {
            "constructions": ["& på något"],
            "translation_comments": ["informal"],
            "synonyms": ["full flaska"],
            "pronunciations": ["he:la"],
        }
        task = enrich_json.task_for(entry)
        self.assertEqual(
            task["source_hints"],
            {
                "constructions": ["& på något"],
                "translation_comments": ["informal"],
            },
        )
        self.assertEqual(
            task["definitions"][0]["source_hints"],
            {"sense_glosses": ["a complete bottle of spirits"]},
        )
        entry["definitions"] = []
        self.assertEqual(
            enrich_json.task_for(entry)["source_hints"]["synonyms"],
            ["full flaska"],
        )

    def test_model_decides_for_every_complete_sense_by_default(self) -> None:
        # The model sees the full card context and may return either a concise
        # high-value note or an explicit null.
        entry = self.entry()
        self.assertTrue(enrich_json.needs_enrichment(entry))
        task = enrich_json.task_for(entry)
        self.assertEqual(len(task["definitions"]), 1)
        self.assertTrue(task["definitions"][0]["need_note"])
        self.assertFalse(task["definitions"][0]["need_example"])

        # An explicit null records the decision and stops further work.
        entry["definitions"][0]["note"] = None  # type: ignore[index]
        self.assertFalse(enrich_json.needs_enrichment(entry))
        self.assertEqual(enrich_json.task_for(entry)["definitions"], [])

    def test_evidence_scope_is_an_optional_cost_trim(self) -> None:
        entry = self.entry()
        self.assertFalse(enrich_json.needs_enrichment(entry, notes_scope="evidence"))
        self.assertEqual(
            enrich_json.task_for(entry, notes_scope="evidence")["definitions"], []
        )
        # Source evidence still routes a sense to the model in cheap mode.
        entry["lexical_info"] = {"idioms": ['en hela ("hel flaska")']}
        self.assertTrue(enrich_json.needs_enrichment(entry, notes_scope="evidence"))
        self.assertTrue(
            enrich_json.task_for(entry, notes_scope="evidence")["definitions"][0][
                "need_note"
            ]
        )

    def test_sense_missing_an_example_always_assesses_its_note(self) -> None:
        # The example request happens anyway, so the note rides along for free.
        entry = self.entry()
        entry["definitions"][0]["example"] = ""  # type: ignore[index]
        task = enrich_json.task_for(entry)
        self.assertTrue(task["definitions"][0]["need_note"])
        self.assertTrue(task["definitions"][0]["need_example"])

    def test_evidence_scope_validates_an_inflection_only_patch(self) -> None:
        entry = self.entry()
        entry["inflections"] = {}
        task = enrich_json.task_for(entry, notes_scope="evidence")
        self.assertEqual(task["definitions"], [])
        patch = {
            "new_definitions": [],
            "definition_updates": [],
            "inflections": {"plural": "helor"},
        }
        enrich_json.validate_patch(entry, patch, notes_scope="evidence")

    def test_batch_collect_parser_always_defines_notes_scope(self) -> None:
        args = enrich_json.parser().parse_args(
            ["batch-collect", "--batch-id", "batch_123"]
        )
        self.assertEqual(args.notes_scope, "all")

    def test_batch_submit_propagates_evidence_scope_into_requests(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            entry = self.entry()
            entry["inflections"] = {}
            (source / "hela.json").write_text(json.dumps(entry), encoding="utf-8")
            proposals = root / "review" / "proposals.jsonl"
            args = enrich_json.parser().parse_args(
                [
                    "batch-submit",
                    "--source",
                    str(source),
                    "--proposals",
                    str(proposals),
                    "--notes-scope",
                    "evidence",
                ]
            )
            with (
                patch.dict(enrich_json.os.environ, {"OPENAI_API_KEY": "test"}),
                patch.object(
                    enrich_json, "upload_batch_file", return_value={"id": "file_1"}
                ),
                patch.object(
                    enrich_json,
                    "api_json",
                    return_value={"id": "batch_1", "status": "validating"},
                ),
            ):
                self.assertEqual(enrich_json.batch_submit(args), 0)

            request = json.loads(
                (proposals.parent / "batch_requests_pending.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            task = json.loads(request["body"]["input"])
            self.assertEqual(task["definitions"], [])
            manifest = json.loads(
                (proposals.parent / "batch_batch_1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["notes_scope"], "evidence")

    def test_batch_cards_mode_uses_the_shared_holistic_contract(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            (source / "vi.json").write_text(
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
            proposals = root / "review" / "proposals.jsonl"
            args = enrich_json.parser().parse_args(
                [
                    "batch-submit",
                    "--source",
                    str(source),
                    "--proposals",
                    str(proposals),
                    "--cards-only",
                    "--production-limit",
                    "0",
                ]
            )
            with (
                patch.dict(enrich_json.os.environ, {"OPENAI_API_KEY": "test"}),
                patch.object(
                    enrich_json, "upload_batch_file", return_value={"id": "file_1"}
                ),
                patch.object(
                    enrich_json,
                    "api_json",
                    return_value={"id": "batch_cards", "status": "validating"},
                ),
            ):
                self.assertEqual(enrich_json.batch_submit(args), 0)
            request = json.loads(
                (proposals.parent / "batch_requests_pending.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertIn("kortplan", request["body"]["instructions"].casefold())
            card_input = json.loads(request["body"]["input"])
            self.assertEqual(card_input[0]["word"], "vi")
            schema = request["body"]["text"]["format"]["schema"]
            self.assertEqual(schema["required"], ["vi.json:cards-v2"])

    def test_batch_cards_collection_produces_an_applicable_review_plan(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            source.mkdir()
            entry = {
                "frequency_rank": 1,
                "word": "vi",
                "article": "",
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
            (source / "vi.json").write_text(json.dumps(entry), encoding="utf-8")
            proposals = root / "proposals.jsonl"
            failures = root / "failures.jsonl"
            card_response = {
                "vi.json:cards-v2": {
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
            batch_line = json.dumps(
                {
                    "custom_id": "vi.json",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "model": "test-model",
                            "output_text": json.dumps(card_response),
                        },
                    },
                }
            )
            target = enrich_json.collect_cards(source, proposals, production_limit=0)[
                0
            ][0]
            (root / "batch_batch_cards.json").write_text(
                json.dumps(
                    {
                        "cards_only": True,
                        "card_targets": {
                            "vi.json": {
                                "id": target.id,
                                "source_digest": target.source_digest,
                                "production_eligible": False,
                                "production_score": target.production_score,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            collect_args = enrich_json.parser().parse_args(
                [
                    "batch-collect",
                    "--batch-id",
                    "batch_cards",
                    "--source",
                    str(source),
                    "--proposals",
                    str(proposals),
                    "--failures",
                    str(failures),
                    "--cards-only",
                    "--production-limit",
                    "0",
                ]
            )
            with (
                patch.dict(enrich_json.os.environ, {"OPENAI_API_KEY": "test"}),
                patch.object(
                    enrich_json,
                    "api_json",
                    return_value={
                        "status": "completed",
                        "output_file_id": "file_output",
                    },
                ),
                patch.object(enrich_json, "download_file", return_value=batch_line),
            ):
                self.assertEqual(enrich_json.batch_collect(collect_args), 0)
            proposal = json.loads(proposals.read_text(encoding="utf-8"))
            self.assertEqual(proposal["kind"], "cards")
            self.assertEqual(proposal["generator"], "openai-batch")

            apply_args = enrich_json.parser().parse_args(
                [
                    "apply",
                    "--source",
                    str(source),
                    "--proposals",
                    str(proposals),
                    "--apply",
                ]
            )
            self.assertEqual(enrich_json.apply(apply_args), 0)
            updated = json.loads((source / "vi.json").read_text(encoding="utf-8"))
            self.assertEqual(
                updated["card_plan"]["cards"][0]["sentence"],
                "Vi arbetar tillsammans.",
            )
            self.assertEqual(
                updated["card_plan"]["reviewed_by"],
                "openai-batch:test-model",
            )

    def test_validated_note_is_merged_without_replacing_source_fields(self) -> None:
        entry = self.entry()
        patch = {
            "new_definitions": [],
            "definition_updates": [
                {
                    "index": 0,
                    "example": None,
                    "example_translation": None,
                    "note": "A fixed noun use in traditional drinking contexts; not the adjective hela (‘whole’).",
                }
            ],
            "inflections": {},
        }
        original = copy.deepcopy(entry)
        enrich_json.validate_patch(entry, patch)
        updated = enrich_json.merge(entry, patch)
        self.assertEqual(entry, original)
        self.assertEqual(
            updated["definitions"][0]["note"],  # type: ignore[index]
            "A fixed noun use in traditional drinking contexts; not the adjective hela (‘whole’).",
        )
        self.assertEqual(
            updated["definitions"][0]["definition"],  # type: ignore[index]
            "a full bottle of spirits",
        )

    def test_null_note_marks_a_sense_as_assessed(self) -> None:
        entry = self.entry()
        patch = {
            "new_definitions": [],
            "definition_updates": [
                {
                    "index": 0,
                    "example": None,
                    "example_translation": None,
                    "note": None,
                }
            ],
            "inflections": {},
        }
        enrich_json.validate_patch(entry, patch)
        updated = enrich_json.merge(entry, patch)
        definition = updated["definitions"][0]  # type: ignore[index]
        self.assertIn("note", definition)
        self.assertIsNone(definition["note"])
        self.assertFalse(enrich_json.needs_enrichment(updated))

    def test_api_note_validation_matches_codex_limit(self) -> None:
        entry = self.entry()
        patch = {
            "new_definitions": [],
            "definition_updates": [
                {
                    "index": 0,
                    "example": None,
                    "example_translation": None,
                    "note": "x" * 181,
                }
            ],
            "inflections": {},
        }
        with self.assertRaisesRegex(ValueError, "exceeds 180"):
            enrich_json.validate_patch(entry, patch)

    def test_translation_only_response_never_echoes_existing_example(self) -> None:
        entry = self.entry()
        entry["definitions"][0]["example_translation"] = ""  # type: ignore[index]
        entry["definitions"][0]["note"] = None  # type: ignore[index]
        entry["definitions"][0]["source_hints"] = {  # type: ignore[index]
            "sense_glosses": ["a full bottle"]
        }
        task = enrich_json.task_for(entry)
        self.assertTrue(enrich_json.is_translation_only(task))
        payload = enrich_json.response_payload(task, "model", translation_only=True)
        compact_task = json.loads(payload["input"])
        self.assertEqual(
            compact_task["items"][0]["source_hints"],
            {"sense_glosses": ["a full bottle"]},
        )
        schema = payload["text"]["format"]["schema"]
        item_schema = schema["properties"]["translations"]["items"]
        self.assertEqual(
            item_schema["properties"]["example"]["type"], ["string", "null"]
        )
        self.assertEqual(schema["properties"]["inflections"]["properties"], {})
        self.assertIn("never echo", payload["instructions"])

        patch = enrich_json.translation_patch(
            {
                "translations": [
                    {
                        "index": 0,
                        "example": None,
                        "example_translation": "He ordered a full bottle of spirits.",
                    }
                ],
                "inflections": {},
            }
        )
        enrich_json.validate_patch(entry, patch)
        updated = enrich_json.merge(entry, patch)
        self.assertEqual(
            updated["definitions"][0]["example"],  # type: ignore[index]
            "Han beställde en hela.",
        )

    def test_patch_validation_and_merge_are_idempotent(self) -> None:
        entry = self.entry()
        patch = {
            "new_definitions": [],
            "definition_updates": [
                {
                    "index": 0,
                    "example": None,
                    "example_translation": None,
                    "note": "A traditional noun use; not the adjective hela (‘whole’).",
                }
            ],
            "inflections": {"plural": "helor"},
        }
        enrich_json.validate_patch(entry, patch)
        updated = enrich_json.merge(entry, patch)
        enrich_json.validate_patch(updated, patch)
        self.assertEqual(enrich_json.merge(updated, patch), updated)

    def test_api_schema_requests_only_missing_inflection_keys(self) -> None:
        entry = self.entry()
        entry["inflections"] = {}
        task = enrich_json.task_for(entry)
        schema = enrich_json.patch_schema_for(task)
        inflections = schema["properties"]["inflections"]
        self.assertEqual(inflections["required"], ["plural"])
        self.assertEqual(set(inflections["properties"]), {"plural"})

    def test_inflections_only_does_not_require_definition_updates(self) -> None:
        entry = self.entry()
        entry["inflections"] = {}
        patch = {
            "new_definitions": [],
            "definition_updates": [],
            "inflections": {"plural": "helor"},
        }
        with self.assertRaisesRegex(ValueError, "omitted definitions"):
            enrich_json.validate_patch(entry, patch)
        enrich_json.validate_patch(entry, patch, definitions_required=False)


if __name__ == "__main__":
    unittest.main()
