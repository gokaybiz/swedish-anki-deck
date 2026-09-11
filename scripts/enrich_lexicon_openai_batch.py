"""Propose and, only after review, apply LLM enrichments to Anki entry JSONs.

The proposal commands never change files under data/json. They write reviewable
patches to a JSONL file. A separate explicit `apply`
command applies a reviewed proposal file.

Requires only Python's standard library and OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from enrich_lexicon_codex_subscription import (
    CARD_PLAN_VERSION,
    CARDS_PROMPT,
    CardPlanTarget,
    apply_cards,
    collect_cards,
    output_schema_for,
    validate_cards,
)
from functions.enrichment_quality import (
    ENRICHMENT_QUALITY_POLICY,
    INFLECTIONS_BY_POS,
    MAX_DEFINITIONS,
    learner_order,
    model_sense_hints,
    model_source_hints,
    normalize_optional_note,
    note_warrants_assessment,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "json"
DEFAULT_PROPOSALS = ROOT / "review" / "lexicon_openai_batch_proposals.jsonl"
DEFAULT_FAILURES = ROOT / "review" / "lexicon_openai_batch_failures.jsonl"
API_URL = "https://api.openai.com/v1/responses"
API_BASE_URL = "https://api.openai.com/v1"
QUALITY_VERSION = 5

SYSTEM_PROMPT = (
    "You are a careful Swedish lexicographer preparing concise Anki data.\n\n"
    + ENRICHMENT_QUALITY_POLICY
    + """
Response contract: if definitions_are_missing is true, put the new entries in
new_definitions and leave definition_updates empty. Otherwise leave
new_definitions empty and return exactly one definition_updates item for every
input definitions item. The need_example, need_example_translation, and
need_note booleans are authoritative. For examples/translations, true requires
a non-empty string and false requires null. For note, true requires either one
high-value English note or null when no note is justified; false requires null.
Do not infer these requirements from whether another JSON key exists. In
inflections, return a string only for a requested missing form and null for
every other form.
"""
)

PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "new_definitions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "definition": {"type": "string"},
                    "example": {"type": "string"},
                    "example_translation": {"type": "string"},
                    "note": {"type": ["string", "null"]},
                },
                "required": [
                    "definition",
                    "example",
                    "example_translation",
                    "note",
                ],
            },
            "maxItems": 2,
        },
        "definition_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "example": {"type": ["string", "null"]},
                    "example_translation": {"type": ["string", "null"]},
                    "note": {"type": ["string", "null"]},
                },
                "required": ["index", "example", "example_translation", "note"],
            },
        },
        "inflections": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "plural": {"type": ["string", "null"]},
                "present": {"type": ["string", "null"]},
                "past": {"type": ["string", "null"]},
                "supine": {"type": ["string", "null"]},
                "imperative": {"type": ["string", "null"]},
                "participle": {"type": ["string", "null"]},
                "comparative": {"type": ["string", "null"]},
                "superlative": {"type": ["string", "null"]},
            },
            "required": [
                "plural",
                "present",
                "past",
                "supine",
                "imperative",
                "participle",
                "comparative",
                "superlative",
            ],
        },
    },
    "required": ["new_definitions", "definition_updates", "inflections"],
}

TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "example": {"type": ["string", "null"]},
                    "example_translation": {"type": "string", "minLength": 1},
                },
                "required": ["index", "example", "example_translation"],
            },
        },
        "inflections": copy.deepcopy(PATCH_SCHEMA["properties"]["inflections"]),
    },
    "required": ["translations", "inflections"],
}

INFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "inflections": copy.deepcopy(PATCH_SCHEMA["properties"]["inflections"]),
    },
    "required": ["inflections"],
}


def inflections_schema(keys: list[str]) -> dict[str, Any]:
    """Require exactly the missing forms, avoiding eight repeated null fields."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {key: {"type": ["string", "null"]} for key in keys},
        "required": keys,
    }


def patch_schema_for(task: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(PATCH_SCHEMA)
    schema["properties"]["inflections"] = inflections_schema(
        task["missing_inflections"]
    )
    return schema


def translation_schema_for(task: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(TRANSLATION_SCHEMA)
    schema["properties"]["inflections"] = inflections_schema(
        task["missing_inflections"]
    )
    return schema


def inflection_response_schema(missing: list[str]) -> dict[str, Any]:
    schema = copy.deepcopy(INFLECTION_SCHEMA)
    schema["properties"]["inflections"] = inflections_schema(missing)
    return schema


def pos_of(entry: dict[str, Any]) -> str:
    tags = entry.get("tags", [])
    for pos in INFLECTIONS_BY_POS:
        if pos in tags:
            return pos
    return ""


def needs_note_assessment(
    entry: dict[str, Any], item: dict[str, Any], *, notes_scope: str = "all"
) -> bool:
    """Notes are tri-state; blank means unassessed, not necessarily needed.

    By default the model judges every complete sense using its word, gloss, and
    example. ``evidence`` is an explicitly narrower pass based on local source
    signals. An item that still needs an example or translation gets its note
    assessed in that same request, so scope only governs complete senses.
    """
    if "note" in item:
        return False
    if not (item.get("example") and item.get("example_translation")):
        return True
    return notes_scope != "evidence" or note_warrants_assessment(entry)


def needs_enrichment(entry: dict[str, Any], *, notes_scope: str = "all") -> bool:
    definitions = entry.get("definitions", [])
    if not definitions:
        return True
    if any(
        needs_note_assessment(entry, item, notes_scope=notes_scope)
        for item in definitions
    ):
        return True
    pos = pos_of(entry)
    required = INFLECTIONS_BY_POS.get(pos, ())
    present = entry.get("inflections", {})
    return any(not present.get(key) for key in required)


def task_for(entry: dict[str, Any], *, notes_scope: str = "all") -> dict[str, Any]:
    """Send only data relevant to fields that are missing."""
    pos = pos_of(entry)
    definitions = entry.get("definitions", [])
    current_inflections = entry.get("inflections", {})
    missing_inflections = [
        key
        for key in INFLECTIONS_BY_POS.get(pos, ())
        if not current_inflections.get(key)
    ]
    definition_requests: list[dict[str, Any]] = []
    for index, item in enumerate(definitions):
        missing_example = not item.get("example")
        missing_translation = not item.get("example_translation")
        missing_note_assessment = needs_note_assessment(
            entry, item, notes_scope=notes_scope
        )
        if missing_example or missing_translation or missing_note_assessment:
            request: dict[str, Any] = {
                "index": index,
                "definition": item.get("definition", ""),
                "example": item.get("example", ""),
                "example_translation": item.get("example_translation", ""),
                "note": item.get("note"),
                "need_example": missing_example,
                "need_example_translation": missing_translation,
                "need_note": missing_note_assessment,
            }
            sense_hints = model_sense_hints(item)
            if sense_hints:
                request["source_hints"] = sense_hints
            definition_requests.append(request)
    task: dict[str, Any] = {
        "word": entry.get("word", ""),
        "part_of_speech": pos or entry.get("tags", []),
        "article": entry.get("article", ""),
        "missing_inflections": missing_inflections,
        "definitions": definition_requests,
        "definitions_are_missing": not definitions,
    }
    purpose = (
        "definitions"
        if not definitions
        else "notes"
        if any(item["need_note"] for item in definition_requests)
        else "examples"
    )
    hints = model_source_hints(entry, purpose=purpose)
    if hints:
        task["source_hints"] = hints
    return task


def response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    # Compatibility fallback if output_text is absent.
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def ask_model(
    task: dict[str, Any], model: str, api_key: str, timeout: int, correction: str = ""
) -> dict[str, Any]:
    payload = response_payload(task, model, correction)
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {exc.code}: {details}") from exc
    try:
        return json.loads(response_text(json.loads(response_body)))
    except json.JSONDecodeError as exc:
        raise ValueError("The API did not return valid patch JSON") from exc


def is_translation_only(task: dict[str, Any]) -> bool:
    return bool(task["definitions"]) and all(
        item["need_example_translation"] and not item["need_note"]
        for item in task["definitions"]
    )


def response_payload(
    task: dict[str, Any],
    model: str,
    correction: str = "",
    translation_only: bool = False,
) -> dict[str, Any]:
    if translation_only:
        compact_task = {
            "word": task["word"],
            "items": [
                {
                    "index": item["index"],
                    "definition": item["definition"],
                    "existing_example": item["example"],
                    "need_example": item["need_example"],
                    **(
                        {"source_hints": item["source_hints"]}
                        if item.get("source_hints")
                        else {}
                    ),
                }
                for item in task["definitions"]
            ],
            "missing_inflections": task["missing_inflections"],
        }
        if task.get("source_hints"):
            compact_task["source_hints"] = task["source_hints"]
        return {
            "model": model,
            "instructions": ENRICHMENT_QUALITY_POLICY
            + "\nFor every item, translate existing_example exactly when it is present and return null for example; never echo or rewrite existing_example. When existing_example is empty, create a short natural Swedish example and return it with its English translation. Do not omit or alter an index. Fill only the listed missing inflections; return null for all other inflection fields.",
            "input": json.dumps(
                compact_task, ensure_ascii=False, separators=(",", ":")
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "example_translations",
                    "strict": True,
                    "schema": translation_schema_for(task),
                }
            },
            "max_output_tokens": 300,
        }
    return {
        "model": model,
        "instructions": SYSTEM_PROMPT + correction,
        "input": json.dumps(task, ensure_ascii=False, separators=(",", ":")),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "anki_entry_patch",
                "strict": True,
                "schema": patch_schema_for(task),
            }
        },
        "max_output_tokens": 650,
    }


def translation_patch(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "new_definitions": [],
        "definition_updates": [
            {
                "index": item["index"],
                "example": item["example"],
                "example_translation": item["example_translation"],
                "note": None,
            }
            for item in response["translations"]
        ],
        "inflections": response["inflections"],
    }


def inflection_payload(entry: dict[str, Any], model: str) -> dict[str, Any]:
    pos = pos_of(entry)
    missing = [
        key
        for key in INFLECTIONS_BY_POS.get(pos, ())
        if not entry.get("inflections", {}).get(key)
    ]
    return {
        "model": model,
        "instructions": ENRICHMENT_QUALITY_POLICY
        + "\nReturn a non-null form for every listed missing form that genuinely exists and null for every other schema field.",
        "input": json.dumps(
            {
                "word": entry.get("word", ""),
                "article": entry.get("article", ""),
                "part_of_speech": pos,
                "definitions": [
                    x.get("definition", "") for x in entry.get("definitions", [])
                ],
                "missing_inflections": missing,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "anki_inflections",
                "strict": True,
                "schema": inflection_response_schema(missing),
            }
        },
        "max_output_tokens": 150,
    }


def card_payload(target: CardPlanTarget, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": CARDS_PROMPT,
        "input": json.dumps(
            [target.request()], ensure_ascii=False, separators=(",", ":")
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "anki_card_plan",
                "strict": True,
                "schema": output_schema_for([target]),
            }
        },
        "max_output_tokens": 1000,
    }


def inflection_patch(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "new_definitions": [],
        "definition_updates": [],
        "inflections": response["inflections"],
    }


def definition_values(item: dict[str, Any]) -> dict[str, str | None]:
    return {
        "definition": str(item.get("definition") or "").strip(),
        "example": str(item.get("example") or "").strip(),
        "example_translation": str(item.get("example_translation") or "").strip(),
        "note": normalize_optional_note(item.get("note"), "definition note"),
    }


def validate_patch(
    entry: dict[str, Any],
    patch: dict[str, Any],
    *,
    definitions_required: bool = True,
    notes_scope: str = "all",
) -> None:
    definitions = entry.get("definitions", [])
    if definitions and patch["new_definitions"]:
        existing_values = [definition_values(item) for item in definitions]
        proposed_values = [definition_values(item) for item in patch["new_definitions"]]
        if existing_values != proposed_values:
            raise ValueError("Patch attempted to replace existing definitions")
    if (
        not definitions
        and definitions_required
        and not 1 <= len(patch["new_definitions"]) <= MAX_DEFINITIONS
    ):
        raise ValueError("Patch must contain one or two new definitions")
    for item in patch["new_definitions"]:
        if not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("definition", "example", "example_translation")
        ):
            raise ValueError(
                "Every new definition needs definition, example, and translation"
            )
        normalize_optional_note(item["note"], "new definition note")
    seen: set[int] = set()
    for update in patch["definition_updates"]:
        index = update["index"]
        if index in seen or not 0 <= index < len(definitions):
            raise ValueError("Invalid or duplicate definition update index")
        seen.add(index)
        original = definitions[index]
        if (
            original.get("example")
            and update["example"] not in (None, "")
            and str(original["example"]).strip() != str(update["example"]).strip()
        ):
            raise ValueError("Patch conflicts with an existing example")
        if (
            original.get("example_translation")
            and update["example_translation"] not in (None, "")
            and str(original["example_translation"]).strip()
            != str(update["example_translation"]).strip()
        ):
            raise ValueError("Patch conflicts with an existing translation")
        if not original.get("example") and not update["example"]:
            raise ValueError("Missing proposed example")
        if (
            not original.get("example_translation")
            and not update["example_translation"]
        ):
            raise ValueError("Missing proposed example translation")
        if (
            "note" in original
            and update["note"] not in (None, "")
            and original["note"]
            != normalize_optional_note(update["note"], f"definition {index} note")
        ):
            raise ValueError("Patch conflicts with an assessed usage note")
        normalize_optional_note(update["note"], f"definition {index} note")
    expected_updates = (
        {
            index
            for index, item in enumerate(definitions)
            if not item.get("example")
            or not item.get("example_translation")
            or needs_note_assessment(entry, item, notes_scope=notes_scope)
        }
        if definitions_required
        else set()
    )
    if not expected_updates.issubset(seen):
        raise ValueError("Patch omitted definitions that still have missing fields")
    allowed = set(INFLECTIONS_BY_POS.get(pos_of(entry), ()))
    required_inflections = {
        key for key in allowed if not entry.get("inflections", {}).get(key)
    }
    if not required_inflections.issubset(patch["inflections"]):
        raise ValueError("Patch omitted requested missing inflections")
    for key, value in patch["inflections"].items():
        # The strict API schema contains every possible key; null means no proposal.
        if value is None:
            continue
        if key not in allowed:
            raise ValueError(f"Unexpected inflection key: {key}")
        existing = entry.get("inflections", {}).get(key)
        if existing and str(existing).strip() != str(value).strip():
            raise ValueError(f"Patch conflicts with existing inflection: {key}")
        if not isinstance(value, str):
            # ValueError (not TypeError) is the module-wide LLM-patch validation
            # contract: callers retry or recover on ValueError for malformed model
            # output regardless of whether the defect is a wrong type or value.
            raise ValueError(f"Invalid value for inflection: {key}")  # noqa: TRY004


def preserve_existing_values(
    entry: dict[str, Any], patch: dict[str, Any]
) -> dict[str, Any]:
    """Discard model attempts to repeat source values before validating its patch."""
    clean = copy.deepcopy(patch)
    if entry.get("definitions"):
        clean["new_definitions"] = []
    for update in clean.get("definition_updates", []):
        index = update.get("index")
        if not isinstance(index, int) or not 0 <= index < len(
            entry.get("definitions", [])
        ):
            continue
        original = entry["definitions"][index]
        if original.get("example"):
            update["example"] = None
        if original.get("example_translation"):
            update["example_translation"] = None
        if "note" in original:
            update["note"] = None
        else:
            update["note"] = normalize_optional_note(
                update["note"], f"definition {index} note"
            )
    for item in clean.get("new_definitions", []):
        item["note"] = normalize_optional_note(item["note"], "new definition note")
    for key in list(clean.get("inflections", {})):
        if key not in INFLECTIONS_BY_POS.get(pos_of(entry), ()) or entry.get(
            "inflections", {}
        ).get(key):
            clean["inflections"][key] = None
    return clean


def merge(entry: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge validated values only into empty fields; never delete source data."""
    updated = copy.deepcopy(entry)
    if not updated.get("definitions"):
        updated["definitions"] = patch["new_definitions"]
    for change in patch["definition_updates"]:
        definition = updated["definitions"][change["index"]]
        if not definition.get("example") and change["example"]:
            definition["example"] = change["example"]
        if not definition.get("example_translation") and change["example_translation"]:
            definition["example_translation"] = change["example_translation"]
        if "note" not in definition:
            # Explicit null records that this sense was assessed and needs no note.
            definition["note"] = change["note"]
    inflections = updated.setdefault("inflections", {})
    for key, value in patch["inflections"].items():
        if not inflections.get(key) and value:
            inflections[key] = value
    return updated


def iter_entries(source: Path):
    yield from sorted(source.glob("*.json"))


def propose(args: argparse.Namespace) -> int:
    selected = set(args.only or ())
    already_proposed = proposed_sources(args.proposals, notes_scope=args.notes_scope)
    pending = [
        (path, entry)
        for path in iter_entries(args.source)
        if (not selected or path.name in selected)
        and path.name not in already_proposed
        and needs_enrichment(entry := load_json(path), notes_scope=args.notes_scope)
    ]
    pending.sort(key=lambda item: (*learner_order(item[1]), item[0].name))
    if args.limit is not None:
        pending = pending[: args.limit]
    paths = [path for path, _entry in pending]
    entries = {path: entry for path, entry in pending}
    if args.dry_run:
        print(
            f"Found {len(paths)} entries that need enrichment. No API request was made."
        )
        for path in paths:
            print(path.name)
        return 0
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set; no request was made.", file=sys.stderr)
        return 2
    print(f"Found {len(paths)} entries that need enrichment.", flush=True)
    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    mode = "a" if args.proposals.exists() else "w"
    with (
        args.proposals.open(mode, encoding="utf-8") as output,
        args.failures.open("a", encoding="utf-8") as failures,
    ):
        for number, path in enumerate(paths, start=1):
            entry = entries[path]
            patch: dict[str, Any] | None = None
            task = task_for(entry, notes_scope=args.notes_scope)
            try:
                try:
                    print(
                        f"[{number}/{len(paths)}] requesting {path.name}...", flush=True
                    )
                    patch = ask_model(task, args.model, api_key, args.timeout)
                    patch = preserve_existing_values(entry, patch)
                    validate_patch(entry, patch, notes_scope=args.notes_scope)
                except ValueError as first_error:
                    # A small model may occasionally return null for a required field.
                    # Retry once with the exact validation failure instead of silently
                    # losing the entry from the review file.
                    print(
                        f"[{number}/{len(paths)}] retrying {path.name}: {first_error}",
                        flush=True,
                    )
                    correction = (
                        "\nYour previous response failed validation: "
                        + str(first_error)
                        + ". Return a corrected complete patch now. In particular, every "
                        "field marked need_example or need_example_translation true must be "
                        "a non-empty string, never null.\n"
                    )
                    patch = ask_model(
                        task, args.model, api_key, args.timeout, correction
                    )
                    patch = preserve_existing_values(entry, patch)
                    validate_patch(entry, patch, notes_scope=args.notes_scope)
                record = {
                    "quality_version": QUALITY_VERSION,
                    "notes_scope": args.notes_scope,
                    "source": path.name,
                    "task": task,
                    "patch": patch,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                print(f"[{number}/{len(paths)}] proposed {path.name}")
            except (
                urllib.error.URLError,
                TimeoutError,
                RuntimeError,
                ValueError,
                KeyError,
            ) as exc:
                print(
                    f"[{number}/{len(paths)}] skipped {path.name}: {exc}",
                    file=sys.stderr,
                )
                failure = {
                    "source": path.name,
                    "notes_scope": args.notes_scope,
                    "task": task,
                    "patch": patch,
                    "error": str(exc),
                }
                failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if args.delay:
                time.sleep(args.delay)
    print(
        f"Added {written} proposals to {args.proposals}. Source JSONs were not changed."
    )
    return 0


def proposed_sources(proposals: Path, *, notes_scope: str = "all") -> set[str]:
    """Return proposals that cover the requested note-assessment scope.

    ``all`` proposals also cover the narrower ``evidence`` scope. A missing
    scope is treated as ``all`` for compatibility with proposal rows created
    before the scope was recorded explicitly.
    """
    if not proposals.exists():
        return set()
    names: set[str] = set()
    with proposals.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = record.get("source")
            record_scope = record.get("notes_scope", "all")
            scope_matches = record_scope == "all" or record_scope == notes_scope
            if (
                record.get("quality_version") == QUALITY_VERSION
                and isinstance(record.get("patch"), dict)
                and isinstance(source, str)
                and scope_matches
            ):
                names.add(source)
    return names


def api_json(
    method: str, path: str, api_key: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        API_BASE_URL + path,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {exc.code}: {details}") from exc


def upload_batch_file(path: Path, api_key: str) -> dict[str, Any]:
    boundary = "----ankiBatchBoundary7MA4YWxkTrZu0gW"
    content = path.read_bytes()
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: application/jsonl\r\n\r\n'.encode(),
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        API_BASE_URL + "/files",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"File upload failed with HTTP {exc.code}: {details}"
        ) from exc


def pending_entries(
    source: Path, proposals: Path, *, notes_scope: str = "all"
) -> list[tuple[Path, dict[str, Any]]]:
    known = proposed_sources(proposals, notes_scope=notes_scope)
    entries = [
        (path, entry)
        for path in iter_entries(source)
        if path.name not in known
        and needs_enrichment(entry := load_json(path), notes_scope=notes_scope)
    ]
    return sorted(entries, key=lambda item: (*learner_order(item[1]), item[0].name))


def inflection_pending_entries(source: Path) -> list[tuple[Path, dict[str, Any]]]:
    entries = []
    for path in iter_entries(source):
        entry = load_json(path)
        if any(
            not entry.get("inflections", {}).get(key)
            for key in INFLECTIONS_BY_POS.get(pos_of(entry), ())
        ):
            entries.append((path, entry))
    return sorted(entries, key=lambda item: (*learner_order(item[1]), item[0].name))


def batch_submit(args: argparse.Namespace) -> int:
    if args.cards_only and args.inflections_only:
        print(
            "Choose only one of --cards-only and --inflections-only.", file=sys.stderr
        )
        return 2
    if args.production_limit < 0:
        print("--production-limit cannot be negative.", file=sys.stderr)
        return 2
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set; no batch was submitted.", file=sys.stderr)
        return 2
    card_targets: list[CardPlanTarget] = []
    entries: list[tuple[Path, dict[str, Any]]] = []
    if args.cards_only:
        card_targets, incomplete = collect_cards(
            args.source,
            args.proposals,
            production_limit=args.production_limit,
        )
        if incomplete:
            print(
                f"Deferred {incomplete} card plans until lexical fields are complete."
            )
    else:
        entries = (
            inflection_pending_entries(args.source)
            if args.inflections_only
            else pending_entries(
                args.source, args.proposals, notes_scope=args.notes_scope
            )
        )
    selected: list[CardPlanTarget] | list[tuple[Path, dict[str, Any]]] = (
        card_targets if args.cards_only else entries
    )
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        print("No pending entries to submit.")
        return 0
    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    request_file = args.proposals.parent / "batch_requests_pending.jsonl"
    with request_file.open("w", encoding="utf-8") as output:
        if args.cards_only:
            for target in card_targets[: args.limit] if args.limit else card_targets:
                line = {
                    "custom_id": target.source,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": card_payload(target, args.model),
                }
                output.write(json.dumps(line, ensure_ascii=False) + "\n")
        else:
            for path, entry in entries[: args.limit] if args.limit else entries:
                if args.inflections_only:
                    body = inflection_payload(entry, args.model)
                else:
                    task = task_for(entry, notes_scope=args.notes_scope)
                    body = response_payload(
                        task, args.model, translation_only=is_translation_only(task)
                    )
                line = {
                    "custom_id": path.name,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                output.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Prepared {len(selected)} requests. Uploading the batch file...", flush=True)
    uploaded = upload_batch_file(request_file, api_key)
    batch = api_json(
        "POST",
        "/batches",
        api_key,
        {
            "input_file_id": uploaded["id"],
            "endpoint": "/v1/responses",
            "completion_window": "24h",
        },
    )
    manifest = args.proposals.parent / f"batch_{batch['id']}.json"
    manifest.write_text(
        json.dumps(
            {
                "batch_id": batch["id"],
                "status": batch.get("status"),
                "request_file": str(request_file),
                "count": len(selected),
                "notes_scope": args.notes_scope,
                "inflections_only": args.inflections_only,
                "cards_only": args.cards_only,
                "production_limit": args.production_limit,
                "card_targets": (
                    {
                        target.source: {
                            "id": target.id,
                            "source_digest": target.source_digest,
                            "production_eligible": target.production_eligible,
                            "production_score": target.production_score,
                        }
                        for target in card_targets[: args.limit]
                        if args.limit is not None
                    }
                    if args.cards_only and args.limit is not None
                    else {
                        target.source: {
                            "id": target.id,
                            "source_digest": target.source_digest,
                            "production_eligible": target.production_eligible,
                            "production_score": target.production_score,
                        }
                        for target in card_targets
                    }
                    if args.cards_only
                    else {}
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Submitted {len(selected)} requests as {batch['id']}. Check later with: "
        "python scripts/enrich_lexicon_openai_batch.py batch-status "
        f"--batch-id {batch['id']}"
    )
    return 0


def batch_status(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    batch = api_json("GET", f"/batches/{args.batch_id}", api_key)
    counts = batch.get("request_counts", {})
    print(
        json.dumps(
            {
                "id": batch.get("id"),
                "status": batch.get("status"),
                "request_counts": counts,
                "errors": batch.get("errors"),
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def download_file(file_id: str, api_key: str) -> str:
    request = urllib.request.Request(
        API_BASE_URL + f"/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def batch_collect(args: argparse.Namespace) -> int:
    if args.cards_only and args.inflections_only:
        print(
            "Choose only one of --cards-only and --inflections-only.", file=sys.stderr
        )
        return 2
    if args.production_limit < 0:
        print("--production-limit cannot be negative.", file=sys.stderr)
        return 2
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    batch = api_json("GET", f"/batches/{args.batch_id}", api_key)
    if batch.get("status") != "completed":
        print(f"Batch is {batch.get('status')}; there is nothing to collect yet.")
        return 1
    output_id = batch.get("output_file_id")
    if not output_id:
        print("Completed batch has no output file.", file=sys.stderr)
        return 1
    known = proposed_sources(args.proposals, notes_scope=args.notes_scope)
    card_targets_by_source: dict[str, CardPlanTarget] = {}
    submitted_card_targets: dict[str, dict[str, object]] = {}
    if args.cards_only:
        manifest_path = args.proposals.parent / f"batch_{args.batch_id}.json"
        if not manifest_path.is_file():
            print(f"Card batch manifest is missing: {manifest_path}", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cards_only") is not True:
            print("Batch manifest is not a card-review batch.", file=sys.stderr)
            return 2
        submitted = manifest.get("card_targets")
        if not isinstance(submitted, dict):
            print("Card batch manifest has no target metadata.", file=sys.stderr)
            return 2
        submitted_card_targets = {
            str(source): metadata
            for source, metadata in submitted.items()
            if isinstance(metadata, dict)
        }
        card_targets, _incomplete = collect_cards(
            args.source,
            args.proposals,
            production_limit=args.production_limit,
        )
        card_targets_by_source = {target.source: target for target in card_targets}
    added = 0
    failed = 0
    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.proposals.open("a", encoding="utf-8") as output,
        args.failures.open("a", encoding="utf-8") as failures,
    ):
        for line in download_file(output_id, api_key).splitlines():
            result = json.loads(line)
            source = result.get("custom_id")
            if not isinstance(source, str):
                continue
            if args.cards_only:
                target = card_targets_by_source.get(source)
                submitted_target = submitted_card_targets.get(source)
                if target is None or submitted_target is None:
                    continue
                source_changed = (
                    submitted_target.get("source_digest") != target.source_digest
                )
                task: dict[str, Any] = target.request()
            else:
                if source in known and not args.inflections_only:
                    continue
                target = None
                task = {}
            entry = load_json(args.source / source)
            if not args.cards_only:
                task = task_for(entry, notes_scope=args.notes_scope)
            try:
                if args.cards_only and source_changed:
                    raise ValueError(
                        f"{source}: source changed after card batch submission"
                    )
                response = result.get("response") or {}
                if response.get("status_code") != 200:
                    raise ValueError(
                        f"Batch request returned HTTP {response.get('status_code')}"
                    )
                model_response = json.loads(response_text(response["body"]))
                if args.cards_only:
                    assert target is not None
                    cards = validate_cards([target], model_response)[target.id]["cards"]
                    output.write(
                        json.dumps(
                            {
                                "quality_version": QUALITY_VERSION,
                                "id": target.id,
                                "source": target.source,
                                "source_digest": target.source_digest,
                                "word": target.word,
                                "article": target.article,
                                "part_of_speech": target.part_of_speech,
                                "production_eligible": target.production_eligible,
                                "production_score": target.production_score,
                                "kind": "cards",
                                "card_plan_version": CARD_PLAN_VERSION,
                                "cards": cards,
                                "model": str(
                                    response["body"].get("model") or "openai-batch"
                                ),
                                "generator": "openai-batch",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    added += 1
                    continue
                if args.inflections_only:
                    patch = inflection_patch(model_response)
                else:
                    patch = (
                        translation_patch(model_response)
                        if "translations" in model_response
                        else model_response
                    )
                patch = preserve_existing_values(entry, patch)
                validate_patch(
                    entry,
                    patch,
                    definitions_required=not args.inflections_only,
                    notes_scope=args.notes_scope,
                )
                output.write(
                    json.dumps(
                        {
                            "quality_version": QUALITY_VERSION,
                            "notes_scope": args.notes_scope,
                            "source": source,
                            "task": task,
                            "patch": patch,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if not args.inflections_only:
                    known.add(source)
                added += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failures.write(
                    json.dumps(
                        {
                            "source": source,
                            "notes_scope": args.notes_scope,
                            "task": task,
                            "patch": None,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                failed += 1
    print(
        f"Collected {added} proposals; {failed} were recorded as failures. Source JSONs were not changed."
    )
    return 0


def recover(args: argparse.Namespace) -> int:
    """Promote valid saved failures after discarding repeated source fields."""
    if not args.failures.exists():
        print(f"No failure file found at {args.failures}.")
        return 0
    known_by_scope = {
        scope: proposed_sources(args.proposals, notes_scope=scope)
        for scope in ("all", "evidence")
    }
    recovered = 0
    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.proposals.open("a", encoding="utf-8") as output,
        args.failures.open(encoding="utf-8") as failures,
    ):
        for line_number, line in enumerate(failures, start=1):
            try:
                record = json.loads(line)
                source = record["source"]
                patch = record["patch"]
                notes_scope = record.get("notes_scope", "all")
                if notes_scope not in known_by_scope:
                    raise ValueError(f"unsupported notes_scope: {notes_scope!r}")
                if source in known_by_scope[notes_scope] or not isinstance(patch, dict):
                    continue
                entry = load_json(args.source / source)
                patch = preserve_existing_values(entry, patch)
                validate_patch(entry, patch, notes_scope=notes_scope)
            except (
                json.JSONDecodeError,
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                print(f"Line {line_number}: could not recover: {exc}", file=sys.stderr)
                continue
            output.write(
                json.dumps(
                    {
                        "quality_version": QUALITY_VERSION,
                        "notes_scope": notes_scope,
                        "source": source,
                        "task": record.get("task", {}),
                        "patch": patch,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            known_by_scope["evidence"].add(source)
            if notes_scope == "all":
                known_by_scope["all"].add(source)
            recovered += 1
    print(f"Recovered {recovered} existing proposals without calling the API.")
    return 0


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_entry(path: Path, entry: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def apply(args: argparse.Namespace) -> int:
    if not args.apply:
        print(
            "Refusing to write. Re-run with --apply after reviewing the proposal file.",
            file=sys.stderr,
        )
        return 2
    if not args.source.is_dir() or not args.proposals.is_file():
        print("Source folder or proposal file is missing.", file=sys.stderr)
        return 2
    applied = unchanged = errors = 0
    source_root = args.source.resolve()
    with args.proposals.open(encoding="utf-8") as proposals:
        for line_number, line in enumerate(proposals, start=1):
            try:
                record = json.loads(line)
                source_name = record.get("source")
                if not isinstance(source_name, str):
                    raise TypeError("proposal source must be a filename")
                path = (source_root / source_name).resolve()
                if path.parent != source_root:
                    raise ValueError("proposal source must remain inside --source")
                entry = load_json(path)
                if record.get("kind") == "cards":
                    if args.inflections_only:
                        unchanged += 1
                        continue
                    if apply_cards(entry, record):
                        write_entry(path, entry)
                        applied += 1
                    else:
                        unchanged += 1
                    continue
                patch = record["patch"]
                if args.inflections_only and (
                    patch.get("new_definitions") or patch.get("definition_updates")
                ):
                    unchanged += 1
                    continue
                if args.inflections_only:
                    patch = preserve_existing_values(entry, patch)
                notes_scope = record.get("notes_scope", "all")
                if notes_scope not in ("all", "evidence"):
                    raise ValueError(f"unsupported notes_scope: {notes_scope!r}")
                validate_patch(
                    entry,
                    patch,
                    definitions_required=not args.inflections_only,
                    notes_scope=notes_scope,
                )
                updated = merge(entry, patch)
                if updated == entry:
                    unchanged += 1
                    continue
                write_entry(path, updated)
                applied += 1
            except (
                OSError,
                TypeError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                print(f"Line {line_number}: skipped: {exc}", file=sys.stderr)
                errors += 1
    print(
        f"Applied {applied}; already identical {unchanged}; errors {errors}. "
        "No Anki operations were performed."
    )
    return 1 if errors else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    common.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)

    propose_parser = subparsers.add_parser("propose", parents=[common])
    propose_parser.add_argument("--model", default="gpt-5.4-mini")
    propose_parser.add_argument(
        "--notes-scope",
        choices=("all", "evidence"),
        default="all",
        help=(
            "all: let the model judge every complete sense using the full card "
            "context (default); evidence: cheaper, only senses whose local source "
            "data suggests a note."
        ),
    )
    propose_parser.add_argument(
        "--limit", type=int, help="Process only the first N entries."
    )
    propose_parser.add_argument(
        "--only",
        action="append",
        metavar="FILE",
        help="Process only this JSON filename; repeatable.",
    )
    propose_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List entries to process without calling the API.",
    )
    propose_parser.add_argument("--timeout", type=int, default=60)
    propose_parser.add_argument("--delay", type=float, default=0.0)
    propose_parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    propose_parser.set_defaults(handler=propose)

    apply_parser = subparsers.add_parser("apply", parents=[common])
    apply_parser.add_argument(
        "--apply",
        action="store_true",
        help="Required before any source JSON is written.",
    )
    apply_parser.add_argument(
        "--inflections-only",
        action="store_true",
        help="Apply only inflection-only proposals; useful after an inflections batch.",
    )
    apply_parser.set_defaults(handler=apply)

    recover_parser = subparsers.add_parser("recover", parents=[common])
    recover_parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    recover_parser.set_defaults(handler=recover)

    batch_submit_parser = subparsers.add_parser("batch-submit", parents=[common])
    batch_submit_parser.add_argument("--model", default="gpt-5.4-mini")
    batch_submit_parser.add_argument(
        "--notes-scope",
        choices=("all", "evidence"),
        default="all",
        help="Notes scope used when building the batch requests.",
    )
    batch_submit_parser.add_argument(
        "--limit", type=int, help="Submit only the first N pending entries."
    )
    batch_submit_parser.add_argument(
        "--inflections-only",
        action="store_true",
        help="Submit only missing inflections, even for entries that already have a proposal.",
    )
    batch_submit_parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Submit only holistic learner-card review targets.",
    )
    batch_submit_parser.add_argument(
        "--production-limit",
        type=int,
        default=1000,
        help="Maximum locally ranked entries eligible for production cards.",
    )
    batch_submit_parser.set_defaults(handler=batch_submit)

    batch_status_parser = subparsers.add_parser("batch-status")
    batch_status_parser.add_argument("--batch-id", required=True)
    batch_status_parser.set_defaults(handler=batch_status)

    batch_collect_parser = subparsers.add_parser("batch-collect", parents=[common])
    batch_collect_parser.add_argument("--batch-id", required=True)
    batch_collect_parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    batch_collect_parser.add_argument(
        "--notes-scope",
        choices=("all", "evidence"),
        default="all",
        help="Must match the scope used for batch-submit.",
    )
    batch_collect_parser.add_argument(
        "--inflections-only",
        action="store_true",
        help="Collect a batch created with --inflections-only.",
    )
    batch_collect_parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Collect a batch created with --cards-only.",
    )
    batch_collect_parser.add_argument(
        "--production-limit",
        type=int,
        default=1000,
        help="Must match the limit used for --cards-only submission.",
    )
    batch_collect_parser.set_defaults(handler=batch_collect)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.handler(arguments))
