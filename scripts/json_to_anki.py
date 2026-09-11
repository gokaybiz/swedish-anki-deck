"""Create or update a Swedish vocabulary deck in the open Anki profile.

This is a portable AnkiConnect exporter. It creates separate generated note
types for vocabulary recognition and selected production chunks. Vocabulary
notes are one-per-reviewed-sense and contain no dormant production fields.
Notes are updated by stable IDs so review histories survive later exports. It
never changes Anki unless ``--apply`` is supplied.

Typical use:
    python scripts/json_to_anki.py --core data/json --expressions data/expressions \
        --audio-dir build/audio --deck "Swedish Vocabulary" --apply

Anki must be open and the AnkiConnect add-on must be installed for --apply.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from functions.card_quality import reviewed_card_plan, target_form_spans
from functions.tts import file_sha256

ANKI_CONNECT_URL = "http://127.0.0.1:8765"
VOCABULARY_MODEL_NAME = "Swedish Vocabulary Recognition (generated)"
VOCABULARY_FIELDS = [
    "Word",
    "Card ID",
    "Source IDs",
    "Target Form",
    "Sentence",
    "Context Meaning",
    "Sentence Meaning",
    "Usage Note",
    "Other Meanings",
    "Frequency Order",
    "Learner Level",
    "Part of Speech",
    "Variants",
    "Audio",
    "Inflections",
    "Source Type",
]
CHUNK_MODEL_NAME = "Swedish Chunk Production (generated)"
CHUNK_FIELDS = [
    "Chunk ID",
    "Cue",
    "Swedish Answer",
    "Target Chunk",
    "Context Meaning",
    "Example",
    "Example Meaning",
    "Usage Note",
    "Source Word",
    "Frequency Order",
    "Learner Level",
    "Part of Speech",
    "Audio",
    "Source Type",
]
RECOGNITION_CARD_NAME = "Recognition"
CHUNK_CARD_NAME = "Chunk Production"
CSS = """
.card { font-family: Arial, sans-serif; font-size: 20px; text-align: left; color: #1f2933; background: #fbfcfe; }
.shell { max-width: 720px; margin: 0 auto; padding: 18px 22px; }
.label, .meta { color: #667085; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }
.word { font-size: 32px; font-weight: 700; margin: 3px 0 14px; }
.pos { color: #667085; font-size: 15px; font-weight: 400; }
.block { border-top: 1px solid #e4e7ec; margin-top: 14px; padding-top: 12px; }
.sentence { font-size: 22px; line-height: 1.4; }
.target { color: #9b2c2c; font-weight: 700; }
.meaning { font-size: 21px; font-weight: 650; margin-top: 4px; }
.translation { color: #475467; font-style: italic; margin-top: 5px; }
.usage-note, .other, .inflections { color: #475467; font-size: 15px; line-height: 1.45; margin-top: 8px; }
.audio { display: inline-block; margin-left: 8px; vertical-align: middle; }
.metadata { border-top: 1px solid #e4e7ec; color: #667085; font-size: 13px; margin-top: 18px; padding-top: 10px; }
.irregular { color: #1f2933; font-weight: 650; }
"""
RECOGNITION_FRONT = """<div class=\"shell\">
<div class=\"word\">{{Word}} {{#Part of Speech}}<span class=\"pos\">{{Part of Speech}}</span>{{/Part of Speech}}</div>
<div class=\"block\"><div class=\"label\">Understand the highlighted form in context</div><div class=\"sentence\">{{Sentence}} <span class=\"audio\">{{Audio}}</span></div></div>
</div>"""
RECOGNITION_BACK = """{{FrontSide}}<div class=\"shell\">
<div class=\"block\"><div class=\"label\">Meaning here</div><div class=\"meaning\">{{Context Meaning}}</div><div class=\"label\">Sentence meaning</div><div class=\"translation\">{{Sentence Meaning}}</div>{{#Usage Note}}<div class=\"label\">Usage note</div><div class=\"usage-note\">{{Usage Note}}</div>{{/Usage Note}}{{#Other Meanings}}<div class=\"other\"><span class=\"label\">Related wording</span><br>{{Other Meanings}}</div>{{/Other Meanings}}</div>
{{#Inflections}}<div class=\"block\"><div class=\"label\">Key forms</div><div class=\"inflections\">{{Inflections}}</div></div>{{/Inflections}}
{{#Variants}}<div class=\"block\"><div class=\"label\">Variants</div><div class=\"other\">{{Variants}}</div></div>{{/Variants}}
<div class=\"metadata\">#{{Frequency Order}}{{#Learner Level}} · CEFR {{Learner Level}}{{/Learner Level}} · {{Part of Speech}} · {{Source Type}}</div>
</div>"""
CHUNK_FRONT = """<div class=\"shell\"><div class=\"label\">Produce the Swedish chunk</div><div class=\"sentence\">{{Cue}}</div></div>"""
CHUNK_BACK = """{{FrontSide}}<div class=\"shell\"><div class=\"block\"><div class=\"label\">Swedish answer</div><div class=\"word\">{{Swedish Answer}}</div></div>{{#Context Meaning}}<div class=\"meaning\">{{Context Meaning}}</div>{{/Context Meaning}}{{#Example}}<div class=\"block\"><div class=\"label\">Example</div><div class=\"sentence\">{{Example}} <span class=\"audio\">{{Audio}}</span></div><div class=\"translation\">{{Example Meaning}}</div></div>{{/Example}}{{#Usage Note}}<div class=\"usage-note\">{{Usage Note}}</div>{{/Usage Note}}<div class=\"metadata\">{{Source Word}} · #{{Frequency Order}}{{#Learner Level}} · CEFR {{Learner Level}}{{/Learner Level}} · {{Part of Speech}} · {{Source Type}}</div></div>"""


def anki(action: str, **params: Any) -> Any:
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode(
        "utf-8"
    )
    request = Request(
        ANKI_CONNECT_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=90) as response:
        reply = json.loads(response.read())
    if reply.get("error"):
        raise RuntimeError(f"AnkiConnect {action}: {reply['error']}")
    return reply["result"]


def text(value: object) -> str:
    return html.escape(str(value or "").strip())


def word(entry: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(entry.get("article") or "").strip(),
            str(entry.get("word") or "").strip(),
        )
        if part
    )


def part_of_speech(entry: dict[str, Any]) -> str:
    known = {
        "noun",
        "proper noun",
        "verb",
        "adjective",
        "adverb",
        "pronoun",
        "determiner",
        "numeral",
        "preposition",
        "conjunction",
        "interjection",
        "particle",
        "partikelverb",
        "idiom",
        "expression",
    }
    return next((str(tag) for tag in entry.get("tags", []) if str(tag) in known), "")


def inflections(entry: dict[str, Any]) -> str:
    """Render one compact paradigm and emphasize structurally irregular forms."""
    forms = entry.get("inflections") or {}
    if not isinstance(forms, dict) or not forms:
        return ""
    pos = part_of_speech(entry)
    lemma = str(entry.get("word") or "").strip()
    if pos == "verb":
        stem = lemma.removesuffix("a")
        expected = {
            "present": f"{stem}ar",
            "past": f"{stem}ade",
            "supine": f"{stem}at",
        }
        order = (("present", "pres."), ("past", "pret."), ("supine", "sup."))
    elif pos == "noun":
        expected = {}
        order = (
            ("definite", "best."),
            ("plural", "pl."),
            ("definite_plural", "best. pl."),
        )
    elif pos == "adjective":
        expected = {
            "neuter": f"{lemma}t",
            "plural": f"{lemma}a",
            "comparative": f"{lemma}are",
            "superlative": f"{lemma}ast",
        }
        order = (
            ("neuter", "ett"),
            ("plural", "pl."),
            ("comparative", "komp."),
            ("superlative", "sup."),
        )
    else:
        expected = {}
        order = tuple((key, key.replace("_", " ")) for key in forms)
    rendered: list[str] = []
    for key, label in order:
        value = str(forms.get(key) or "").strip()
        if not value:
            continue
        shown = text(value)
        if key in expected and value.casefold() != expected[key].casefold():
            shown = f'<span class="irregular">{shown}</span>'
        rendered.append(f"{label} {shown}")
    return " · ".join(rendered)


def highlighted(value: object, target: object) -> str:
    """Escape text and highlight a contiguous or discontinuous target form."""
    sentence = str(value or "").strip()
    spans = target_form_spans(sentence, target)
    if not spans:
        return text(sentence)
    result: list[str] = []
    offset = 0
    for start, end in spans:
        result.append(html.escape(sentence[offset:start]))
        result.append('<span class="target">')
        result.append(html.escape(sentence[start:end]))
        result.append("</span>")
        offset = end
    result.append(html.escape(sentence[offset:]))
    return "".join(result)


def variants(entry: dict[str, Any]) -> str:
    """Show only variants that differ from the headword learners already see."""
    headword = str(entry.get("word") or "").strip().casefold()
    values = (entry.get("lexical_info") or {}).get("variants") or []
    shown = [
        str(value).strip()
        for value in values
        if isinstance(value, str) and str(value).strip().casefold() != headword
    ]
    return " · ".join(dict.fromkeys(text(value) for value in shown))


def audio_index(folders: list[Path]) -> dict[str, Path]:
    """Index manifest-backed edge-tts files plus unambiguous legacy MP3s."""
    index: dict[str, Path] = {}

    def register(source_stem: str, path: Path) -> None:
        previous = index.get(source_stem)
        if previous is not None and previous.resolve() != path.resolve():
            raise ValueError(
                f"More than one audio file matches {source_stem!r}: {previous}, {path}"
            )
        index[source_stem] = path

    for folder in folders:
        if not folder.is_dir():
            raise ValueError(f"Audio folder does not exist: {folder}")
        manifested: set[Path] = set()
        manifest_path = folder / "audio_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest.get("entries")
            if not isinstance(records, dict):
                raise ValueError(f"Malformed audio manifest: {manifest_path}")
            for source_stem, record in records.items():
                if not isinstance(record, dict) or not record.get("file"):
                    raise ValueError(
                        f"Malformed audio record for {source_stem!r}: {manifest_path}"
                    )
                path = folder / str(record["file"])
                if not path.is_file():
                    raise ValueError(f"Manifest audio file is missing: {path}")
                expected_digest = record.get("sha256")
                if (
                    not isinstance(expected_digest, str)
                    or file_sha256(path) != expected_digest
                ):
                    raise ValueError(f"Manifest audio digest does not match: {path}")
                manifested.add(path.resolve())
                register(str(source_stem), path)
        for path in folder.rglob("*.mp3"):
            if path.resolve() in manifested:
                continue
            if "__" in path.stem and len(path.stem.rsplit("__", 1)[-1]) == 16:
                if manifest_path.is_file():
                    # Superseded content-addressed clips are harmless orphans;
                    # only the current manifest record is eligible for export.
                    continue
                raise ValueError(f"Content-addressed audio has no manifest: {path}")
            source_stem = path.stem
            for prefix in ("sv_core__", "sv_partikelverb__", "sv_idioms__"):
                if source_stem.startswith(prefix):
                    source_stem = source_stem.removeprefix(prefix)
                    break
            register(source_stem, path)
    return index


def entries(
    core: Path, expressions: Path | None
) -> list[tuple[Path, dict[str, Any], str, int]]:
    if not core.is_dir():
        raise ValueError(f"Core JSON folder does not exist: {core}")
    result: list[tuple[Path, dict[str, Any], str, int]] = []
    for path in sorted(core.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        rank = entry.get("frequency_rank")
        if not isinstance(rank, int) or rank < 1:
            raise ValueError(f"{path}: frequency_rank must be a positive integer")
        result.append((path, entry, "Kelly List", rank))
    max_rank = max((item[3] for item in result), default=0)
    if expressions:
        if not expressions.is_dir():
            raise ValueError(f"Expressions JSON folder does not exist: {expressions}")
        expression_paths = sorted(
            expressions.rglob("*.json"),
            key=lambda path: (
                json.loads(path.read_text(encoding="utf-8")).get("source_order", 0),
                path.name,
            ),
        )
        for offset, path in enumerate(expression_paths, start=1):
            result.append(
                (
                    path,
                    json.loads(path.read_text(encoding="utf-8")),
                    "Expression",
                    max_rank + offset,
                )
            )
    return sorted(result, key=lambda item: item[3])


def vocabulary_fields(
    entry: dict[str, Any],
    card: dict[str, Any],
    identity: str,
    rank: int,
    source: str,
    audio: Path | None,
) -> dict[str, str]:
    target_form = str(card.get("target_form") or "").strip()
    other_meanings = card.get("other_meanings")
    return {
        "Word": text(word(entry)),
        "Card ID": text(identity),
        "Source IDs": text(identity),
        "Target Form": text(target_form),
        "Sentence": highlighted(card.get("sentence"), target_form),
        "Context Meaning": text(card.get("context_meaning")),
        "Sentence Meaning": text(card.get("sentence_meaning")),
        "Usage Note": text(card.get("note")),
        "Other Meanings": " · ".join(
            text(value) for value in other_meanings or [] if str(value).strip()
        ),
        "Frequency Order": str(rank),
        "Learner Level": text((entry.get("learner_level") or {}).get("cefr")),
        "Part of Speech": text(part_of_speech(entry)),
        "Variants": variants(entry),
        "Audio": f"[sound:{audio.name}]" if audio else "",
        "Inflections": inflections(entry),
        "Source Type": source,
    }


def chunk_fields(
    entry: dict[str, Any],
    card: dict[str, Any],
    identity: str,
    rank: int,
    source: str,
    audio: Path | None,
) -> dict[str, str] | None:
    production = card.get("production")
    if not isinstance(production, dict):
        return None
    target = str(production.get("target") or "").strip()
    return {
        "Chunk ID": text(f"{identity}:chunk"),
        "Cue": text(production.get("cue")),
        "Swedish Answer": highlighted(production.get("answer"), target),
        "Target Chunk": text(target),
        "Context Meaning": text(card.get("context_meaning")),
        "Example": highlighted(card.get("sentence"), card.get("target_form")),
        "Example Meaning": text(card.get("sentence_meaning")),
        "Usage Note": text(card.get("note")),
        "Source Word": text(word(entry)),
        "Frequency Order": str(rank),
        "Learner Level": text((entry.get("learner_level") or {}).get("cefr")),
        "Part of Speech": text(part_of_speech(entry)),
        "Audio": f"[sound:{audio.name}]" if audio else "",
        "Source Type": source,
    }


def card_id(path: Path, source: str, rank: int, card: dict[str, Any]) -> str:
    indices = "-".join(str(value) for value in card.get("source_indices") or [])
    if source == "Kelly List":
        return f"kelly:{rank}:{indices}"
    relative_name = f"{path.parent.name}/{path.name}"
    return f"expression:{relative_name}:{indices}"


def card_audio_key(path: Path, card: dict[str, Any]) -> str:
    indices = "-".join(str(value) for value in card.get("source_indices") or [])
    return f"{path.stem}__sense_{indices}"


def ensure_model(
    name: str,
    fields: list[str],
    card_name: str,
    front: str,
    back: str,
) -> None:
    if name not in anki("modelNames"):
        anki(
            "createModel",
            modelName=name,
            inOrderFields=fields,
            css=CSS,
            isCloze=False,
            cardTemplates=[{"Name": card_name, "Front": front, "Back": back}],
        )
        return
    actual = anki("modelFieldNames", modelName=name)
    if actual != fields:
        raise RuntimeError(
            f"Existing {name!r} has unsupported fields: {actual}. "
            "Expected the generated layout documented in README.md."
        )
    anki(
        "updateModelTemplates",
        model={
            "name": name,
            "templates": {card_name: {"Front": front, "Back": back}},
        },
    )
    anki("updateModelStyling", model={"name": name, "css": CSS})


def existing_note_id(model_name: str, id_field: str, identity: str) -> int | None:
    ids = anki(
        "findNotes",
        query=f'note:"{model_name}" "{id_field}:{identity}"',
    )
    if len(ids) > 1:
        raise RuntimeError(f"More than one {model_name} note has {id_field} {identity}")
    return ids[0] if ids else None


def note_tags(fields: dict[str, str], *, kind: str) -> list[str]:
    values = [
        "swedish",
        f"card::{kind}",
        f"source::{fields['Source Type'].lower().replace(' ', '-')}",
    ]
    if fields["Learner Level"]:
        values.append(f"cefr::{fields['Learner Level'].lower()}")
    if fields["Part of Speech"]:
        values.extend(
            f"pos::{value.strip().lower().replace(' ', '-')}"
            for value in fields["Part of Speech"].split("/")
            if value.strip()
        )
    return values


def merge_duplicate_metadata(
    canonical: dict[str, str], duplicate: dict[str, str]
) -> None:
    """Merge provenance/POS when two sources produce one identical retrieval task."""
    for key, separator in (("Source IDs", " · "), ("Part of Speech", " / ")):
        values = [
            value.strip()
            for fields in (canonical, duplicate)
            for value in fields[key].split(separator)
            if value.strip()
        ]
        canonical[key] = separator.join(dict.fromkeys(values))
    canonical["Frequency Order"] = str(
        min(int(canonical["Frequency Order"]), int(duplicate["Frequency Order"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core", type=Path, required=True, help="Folder of Kelly entry JSON files."
    )
    parser.add_argument(
        "--expressions",
        type=Path,
        help="Optional folder containing idiom and particle-verb JSON files.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional folder of MP3 files; may be repeated.",
    )
    parser.add_argument(
        "--deck", default="Swedish Vocabulary", help="Parent deck to create or update."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N entries; useful for a test deck.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write notes and audio to Anki. Default: preview only.",
    )
    args = parser.parse_args()

    records = entries(args.core, args.expressions)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        records = records[: args.limit]
    audio = audio_index(args.audio_dir)
    vocabulary: list[tuple[dict[str, str], Path | None]] = []
    chunks: list[tuple[dict[str, str], Path | None]] = []
    unreviewed: list[str] = []
    merged: list[tuple[str, str]] = []
    seen_tasks: dict[tuple[str, str, str, str], int] = {}
    seen_chunks: set[tuple[str, str]] = set()
    for path, entry, source, rank in records:
        plan = reviewed_card_plan(entry)
        if plan is None:
            unreviewed.append(path.name)
            continue
        for card in plan["cards"]:
            if not isinstance(card, dict):
                raise TypeError(f"{path}: card plan contains a non-object card")
            identity = card_id(path, source, rank, card)
            matching_audio = audio.get(card_audio_key(path, card))
            fields = vocabulary_fields(
                entry, card, identity, rank, source, matching_audio
            )
            task_signature = tuple(
                fields[key].casefold()
                for key in (
                    "Word",
                    "Context Meaning",
                    "Sentence",
                    "Sentence Meaning",
                )
            )
            previous_index = seen_tasks.get(task_signature)
            if previous_index is None:
                seen_tasks[task_signature] = len(vocabulary)
                vocabulary.append((fields, matching_audio))
            else:
                canonical = vocabulary[previous_index][0]
                merge_duplicate_metadata(canonical, fields)
                merged.append((identity, canonical["Card ID"]))

            chunk = chunk_fields(entry, card, identity, rank, source, matching_audio)
            if chunk is not None:
                chunk_signature = (
                    chunk["Cue"].casefold(),
                    chunk["Swedish Answer"].casefold(),
                )
                if chunk_signature not in seen_chunks:
                    seen_chunks.add(chunk_signature)
                    chunks.append((chunk, matching_audio))
    if unreviewed:
        preview = ", ".join(unreviewed[:5])
        parser.error(
            f"{len(unreviewed)} entries require holistic card review before export "
            f"({preview}{'…' if len(unreviewed) > 5 else ''}). Run Codex propose "
            "with --kind cards, review the proposals, and apply them."
        )
    print(
        f"Prepared {len(vocabulary)} recognition notes and {len(chunks)} dedicated "
        f"chunk-production notes; merged {len(merged)} exact duplicate retrieval "
        f"tasks with their provenance."
    )
    for duplicate, canonical in merged[:10]:
        print(f"  duplicate task merged: {duplicate} -> {canonical}")
    if not args.apply:
        print("Preview only. Open Anki with AnkiConnect, then re-run with --apply.")
        return

    ensure_model(
        VOCABULARY_MODEL_NAME,
        VOCABULARY_FIELDS,
        RECOGNITION_CARD_NAME,
        RECOGNITION_FRONT,
        RECOGNITION_BACK,
    )
    if chunks:
        ensure_model(
            CHUNK_MODEL_NAME,
            CHUNK_FIELDS,
            CHUNK_CARD_NAME,
            CHUNK_FRONT,
            CHUNK_BACK,
        )
    created = updated = uploaded = 0
    uploaded_files: set[Path] = set()
    planned_notes = [
        (
            fields,
            audio_path,
            VOCABULARY_MODEL_NAME,
            "Card ID",
            "recognition",
            "Vocabulary",
        )
        for fields, audio_path in vocabulary
    ] + [
        (fields, audio_path, CHUNK_MODEL_NAME, "Chunk ID", "chunk", "Chunks")
        for fields, audio_path in chunks
    ]
    for index, (fields, matching_audio, model, id_field, kind, subdeck) in enumerate(
        planned_notes, start=1
    ):
        deck = f"{args.deck}::{subdeck}"
        anki("createDeck", deck=deck)
        if matching_audio and matching_audio not in uploaded_files:
            anki(
                "storeMediaFile",
                filename=matching_audio.name,
                data=base64.b64encode(matching_audio.read_bytes()).decode("ascii"),
            )
            uploaded_files.add(matching_audio)
            uploaded += 1
        note_id = existing_note_id(model, id_field, fields[id_field])
        if note_id:
            anki("updateNoteFields", note={"id": note_id, "fields": fields})
            updated += 1
        else:
            anki(
                "addNote",
                note={
                    "deckName": deck,
                    "modelName": model,
                    "fields": fields,
                    "options": {"allowDuplicate": True},
                    "tags": note_tags(fields, kind=kind),
                },
            )
            created += 1
        if index % 100 == 0 or index == len(planned_notes):
            print(
                f"[{index}/{len(planned_notes)}] created {created}, updated {updated}",
                flush=True,
            )
    print(
        f"Finished: {created} created, {updated} updated, {uploaded} audio files uploaded."
    )


if __name__ == "__main__":
    main()
