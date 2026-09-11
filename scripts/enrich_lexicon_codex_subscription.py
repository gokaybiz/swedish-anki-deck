"""Review-first lexical enrichment through a Codex/ChatGPT subscription.

``enrich_lexicon_openai_batch.py`` uses metered OpenAI API credits instead.
This tool instead drives the Codex CLI (a Codex/ChatGPT subscription) in
resumable, inspectable batches of roughly ``--batch-size`` targets. It covers
the same learner-facing enrichment jobs under one shared quality contract:

- missing word meanings and first examples for entries with no Folkets gloss,
- missing Swedish example sentences plus their English translations,
- optional sense-specific usage notes when they prevent likely misuse, and
- missing inflection forms for nouns, verbs, and adjectives.

Proposals are appended to a review JSONL file; source JSON is never changed by
``propose``. After review, ``apply --apply`` fills empty source fields only and
refuses to overwrite or contradict existing values. The OpenAI API transport
remains available through ``enrich_lexicon_openai_batch.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from functions.card_quality import (
    CARD_PLAN_VERSION,
    PRODUCTION_SCORE_THRESHOLD,
    card_priority,
    card_source_digest,
    production_candidate,
    reviewed_card_plan,
    target_form_spans,
    target_forms,
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
DEFAULT_PROPOSALS = ROOT / "review" / "lexicon_codex_subscription_proposals.jsonl"
DEFAULT_FAILURES = ROOT / "review" / "lexicon_codex_subscription_failures.jsonl"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_BATCH_SIZE = 250
DEFAULT_WORKERS = 1
MAX_WORKERS = 8
MAX_CARD_BATCH_SIZE = 100
LEXICAL_KINDS = ("definitions", "examples", "notes", "inflections")
ALL_KINDS = (*LEXICAL_KINDS, "cards")
# Compatibility alias; the shared mapping is authoritative for both backends.
ALLOWED_INFLECTIONS = INFLECTIONS_BY_POS

EXAMPLE_PROMPT = (
    ENRICHMENT_QUALITY_POLICY
    + """

Uppgift: fyll exakt de fält som listas i missing för varje id. existing_sentence och existing_sentence_meaning är oföränderlig källkontext: returnera dem aldrig. Om endast sentence_meaning saknas ska den befintliga svenska meningen översättas exakt som den står, även om den är ett kort uttryck eller en fras. Om note listas ska du samtidigt bedöma en kort användningsnot enligt kvalitetsstandarden med den befintliga eller nyskrivna meningen som kontext; returnera null när ingen note behövs.

Returnera exakt en post för varje inkommande id och inga andra id:n. Varje post ska innehålla exakt nycklarna i missing och inga andra. Returnera endast giltig JSON, till exempel:
{"id":{"sentence":"svensk exempelmening","sentence_meaning":"engelsk översättning","note":null}}
eller
{"id":{"sentence_meaning":"engelsk översättning av existing_sentence","note":null}}
"""
)

DEFINITIONS_PROMPT = (
    ENRICHMENT_QUALITY_POLICY
    + """

Uppgift: entry saknar helt ordboksbetydelser. Ge en eller högst två betydelser med meaning, sentence, sentence_meaning och en valfri note. Note måste vara en relevant engelsk användningsnot enligt kvalitetsstandarden eller null.

Returnera exakt en post för varje inkommande id och inga andra id:n. Returnera endast giltig JSON:
{"id":{"definitions":[{"meaning":"kort engelsk glosa","sentence":"svensk exempelmening","sentence_meaning":"engelsk översättning","note":null}]}}
"""
)

NOTES_PROMPT = (
    ENRICHMENT_QUALITY_POLICY
    + """

Uppgift: avgör konservativt om den angivna betydelsen behöver en kort engelsk användningsnot. Sentence är kontext, inte text som ska ändras. Returnera null om glosan och meningen redan lär ut det viktiga eller om du är osäker på att extra information är korrekt och relevant.

Returnera exakt en post för varje inkommande id och inga andra id:n. Returnera endast giltig JSON, med en sträng eller JSON-värdet null:
{"id":{"note":"kort engelsk användningsnot"}}
eller
{"id":{"note":null}}
"""
)

INFLECTIONS_PROMPT = (
    ENRICHMENT_QUALITY_POLICY
    + """

Uppgift: fyll exakt de böjningsnycklar som listas i missing. Använd null för en form som genuint saknas.

Returnera exakt en post för varje inkommande id och inga andra id:n. Returnera endast giltig JSON; varje värde är en sträng eller JSON-värdet null:
{"id":{"forms":{"<nyckel>":"<form>"}}}
"""
)

CARDS_PROMPT = (
    ENRICHMENT_QUALITY_POLICY
    + """

Uppgift: skapa en granskad, elevsynlig kortplan från fullständiga källdefinitioner utan att ändra källdatan.

* Täck varje source index exakt en gång. Slå ihop två index i source_indices endast när de lär ut samma betydelse och skulle ge samma minnesuppgift. Behåll tydligt skilda betydelser som separata kort.
* context_meaning ska vara en enda kort engelsk betydelse som exakt passar sentence, aldrig en synonym-lista med semikolon, kommatecken eller snedstreck. Lägg användbara sekundära synonymer i other_meanings; lägg inte där en annan självständig betydelse.
* Behåll en bra befintlig sentence. Ersätt den i kortplanen endast om den inte visar uppslagsordets aktuella betydelse, är felaktig eller är olämplig för modern SFI D/SVA. target_form måste vara exakt den synliga formen i sentence och en av allowed_target_forms. Godkänn inte ett annat paradigmled eller ett avlett/relaterat ord bara för att det har samma stam eller närliggande betydelse.
* Kontrollera sentence_meaning mot hela sentence. Använd naturlig modern engelska och bevara grammatiska betydelseskillnader, men hårdkoda inte ett visst engelskt ordval när flera nutida formuleringar är korrekta.
* Granska existing_note på nytt. Behåll eller rätta den bara om den är kort, relevant och uttryckligen stöds av kortets kontext eller source_hints. Returnera null för upprepning, osäkerhet eller obelagd fakta.
* production ska vara null om production_eligible är false. När det är true får du ändå välja null. Skapa den bara för ett frekvent, återanvändbart uttryck, kollokation eller konstruktion. cue ska beskriva situation/funktion på engelska och innehålla en kort svensk ledtråd i parentes när flera naturliga svar är möjliga; den får inte bara vara en lös omvänd ordboksöversättning. answer är en kort naturlig svensk fras eller mening och target är en exakt återanvändbar flerordsdel i answer; skapa aldrig ett produktionskort som bara testar ett isolerat ord.

Returnera exakt en post per id och endast giltig JSON:
{"id":{"cards":[{"source_indices":[0],"context_meaning":"register","other_meanings":["sign up"],"sentence":"Jag anmälde mig till kursen.","sentence_meaning":"I registered for the course.","target_form":"anmälde","note":null,"production":{"cue":"Say that you registered for the course (use anmäla sig).","answer":"Jag anmälde mig till kursen.","target":"anmälde mig"}}]}}
"""
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleTarget:
    id: str
    source: str
    definition_index: int
    word: str
    article: str
    part_of_speech: str
    meaning: str
    sentence: str
    sentence_meaning: str
    assess_note: bool = False
    source_hints: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sentence and self.sentence_meaning:
            raise ValueError("ExampleTarget must have at least one missing field")

    @property
    def missing_fields(self) -> tuple[str, ...]:
        fields = [
            key
            for key, value in (
                ("sentence", self.sentence),
                ("sentence_meaning", self.sentence_meaning),
            )
            if not value
        ]
        if self.assess_note:
            fields.append("note")
        return tuple(fields)

    def request(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": self.id,
            "word": self.word,
            "article": self.article,
            "part_of_speech": self.part_of_speech,
            "meaning": self.meaning,
            "existing_sentence": self.sentence,
            "existing_sentence_meaning": self.sentence_meaning,
            "missing": list(self.missing_fields),
        }
        if self.source_hints:
            record["source_hints"] = self.source_hints
        return record

    def row(self, value: dict[str, str | None], model: str) -> dict[str, object]:
        record = asdict(self)
        record.update(value, kind="examples", model=model, generator="codex-cli")
        return record


@dataclass(frozen=True, slots=True, kw_only=True)
class DefinitionTarget:
    id: str
    source: str
    word: str
    article: str
    part_of_speech: str
    source_hints: dict[str, list[str]] = field(default_factory=dict)

    def request(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": self.id,
            "word": self.word,
            "article": self.article,
            "part_of_speech": self.part_of_speech,
        }
        if self.source_hints:
            record["source_hints"] = self.source_hints
        return record

    def row(
        self, value: dict[str, list[dict[str, str | None]]], model: str
    ) -> dict[str, object]:
        record = asdict(self)
        record.update(
            kind="definitions",
            definitions=value["definitions"],
            model=model,
            generator="codex-cli",
        )
        return record


@dataclass(frozen=True, slots=True, kw_only=True)
class NoteTarget:
    id: str
    source: str
    definition_index: int
    word: str
    article: str
    part_of_speech: str
    meaning: str
    sentence: str
    sentence_meaning: str
    source_hints: dict[str, list[str]] = field(default_factory=dict)

    def request(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": self.id,
            "word": self.word,
            "article": self.article,
            "part_of_speech": self.part_of_speech,
            "meaning": self.meaning,
            "sentence": self.sentence,
            "sentence_meaning": self.sentence_meaning,
        }
        if self.source_hints:
            record["source_hints"] = self.source_hints
        return record

    def row(self, value: dict[str, str | None], model: str) -> dict[str, object]:
        record = asdict(self)
        record.update(value, kind="notes", model=model, generator="codex-cli")
        return record


@dataclass(frozen=True, slots=True, kw_only=True)
class InflectionTarget:
    id: str
    source: str
    word: str
    article: str
    part_of_speech: str
    missing: tuple[str, ...]

    def request(self) -> dict[str, object]:
        return {
            "id": self.id,
            "word": self.word,
            "article": self.article,
            "part_of_speech": self.part_of_speech,
            "missing": list(self.missing),
        }

    def row(
        self, value: dict[str, dict[str, str | None]], model: str
    ) -> dict[str, object]:
        record = asdict(self)
        record.update(
            kind="inflections",
            missing=list(self.missing),
            forms=value["forms"],
            model=model,
            generator="codex-cli",
        )
        return record


@dataclass(frozen=True, slots=True, kw_only=True)
class CardPlanTarget:
    id: str
    source: str
    source_digest: str
    word: str
    article: str
    part_of_speech: str
    frequency_rank: int
    learner_level: str
    definitions: tuple[dict[str, object], ...]
    allowed_target_forms: tuple[str, ...]
    production_eligible: bool
    production_score: int
    production_reasons: tuple[str, ...]
    source_hints: dict[str, list[str]] = field(default_factory=dict)
    production_evidence: dict[str, list[str]] = field(default_factory=dict)

    def request(self) -> dict[str, object]:
        return {
            "id": self.id,
            "word": self.word,
            "article": self.article,
            "part_of_speech": self.part_of_speech,
            "frequency_rank": self.frequency_rank,
            "learner_level": self.learner_level,
            "definitions": list(self.definitions),
            "allowed_target_forms": list(self.allowed_target_forms),
            "production_eligible": self.production_eligible,
            "production_score": self.production_score,
            "production_reasons": list(self.production_reasons),
            "source_hints": self.source_hints,
            "production_evidence": self.production_evidence,
        }

    def row(self, value: dict[str, object], model: str) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "source_digest": self.source_digest,
            "word": self.word,
            "article": self.article,
            "part_of_speech": self.part_of_speech,
            "production_eligible": self.production_eligible,
            "production_score": self.production_score,
            "kind": "cards",
            "card_plan_version": CARD_PLAN_VERSION,
            "cards": value["cards"],
            "model": model,
            "generator": "codex-cli",
        }


EnrichmentTarget = (
    ExampleTarget | DefinitionTarget | NoteTarget | InflectionTarget | CardPlanTarget
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    id: str
    source: str
    error: str


@dataclass(frozen=True, slots=True)
class BatchValidation:
    valid: dict[str, object]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class BatchExecution:
    number: int
    kind: str
    targets: list[EnrichmentTarget]
    validation: BatchValidation
    elapsed: float


class CodexBatchCancelled(RuntimeError):
    """A concurrent Codex batch was stopped after interruption or peer failure."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


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


def proposal_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"proposal {key} must be a non-empty single line")
    return value.strip()


def clean_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text or "\n" in text:
        raise ValueError(f"{label} must be a non-empty single line")
    return text


def completed_ids(proposals: Path) -> set[str]:
    if not proposals.exists():
        return set()
    result: set[str] = set()
    with proposals.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
                target_id = record.get("id")
                if (
                    isinstance(target_id, str)
                    and isinstance(record.get("source"), str)
                    and record.get("kind") in ALL_KINDS
                ):
                    result.add(target_id)
            except json.JSONDecodeError:
                continue
    return result


def loaded_entries(source: Path) -> list[tuple[Path, dict[str, Any]]]:
    entries = [(path, load_json(path)) for path in source.glob("*.json")]
    return sorted(entries, key=lambda item: (*learner_order(item[1]), item[0].name))


def collect_definitions(
    source: Path,
    proposals: Path,
    *,
    entries: Sequence[tuple[Path, dict[str, Any]]] | None = None,
    done: set[str] | None = None,
) -> list[DefinitionTarget]:
    done = completed_ids(proposals) if done is None else done
    targets: list[DefinitionTarget] = []
    for path, entry in loaded_entries(source) if entries is None else entries:
        if entry.get("definitions"):
            continue
        word = str(entry.get("word") or "").strip()
        if not word:
            continue
        target_id = f"{path.name}:definitions"
        if target_id in done:
            continue
        targets.append(
            DefinitionTarget(
                id=target_id,
                source=path.name,
                word=word,
                article=str(entry.get("article") or "").strip(),
                part_of_speech=part_of_speech(entry),
                source_hints=model_source_hints(entry, purpose="definitions"),
            )
        )
    return targets


def collect_examples(
    source: Path,
    proposals: Path,
    *,
    entries: Sequence[tuple[Path, dict[str, Any]]] | None = None,
    done: set[str] | None = None,
) -> tuple[list[ExampleTarget], int]:
    done = completed_ids(proposals) if done is None else done
    targets: list[ExampleTarget] = []
    skipped_without_meaning = 0
    for path, entry in loaded_entries(source) if entries is None else entries:
        word = str(entry.get("word") or "").strip()
        article = str(entry.get("article") or "").strip()
        pos = part_of_speech(entry)
        for index, definition in enumerate(entry.get("definitions") or []):
            meaning = str(definition.get("definition") or "").strip()
            sentence = str(definition.get("example") or "").strip()
            sentence_meaning = str(definition.get("example_translation") or "").strip()
            if sentence and sentence_meaning:
                continue
            if not meaning:
                skipped_without_meaning += 1
                continue
            target_id = f"{path.name}:{index}"
            if target_id in done:
                continue
            source_hints = model_source_hints(
                entry,
                purpose="notes" if "note" not in definition else "examples",
            )
            source_hints.update(model_sense_hints(definition))
            targets.append(
                ExampleTarget(
                    id=target_id,
                    source=path.name,
                    definition_index=index,
                    word=word,
                    article=article,
                    part_of_speech=pos,
                    meaning=meaning,
                    sentence=sentence,
                    sentence_meaning=sentence_meaning,
                    assess_note="note" not in definition,
                    source_hints=source_hints,
                )
            )
    return targets, skipped_without_meaning


def collect_notes(
    source: Path,
    proposals: Path,
    *,
    entries: Sequence[tuple[Path, dict[str, Any]]] | None = None,
    done: set[str] | None = None,
    notes_scope: str = "all",
) -> list[NoteTarget]:
    """Collect complete, unassessed senses; explicit null counts as assessed.

    The default lets the model judge every complete sense using the word,
    gloss, and example. Batching keeps that affordable: one request covers up
    to ``--batch-size`` senses. ``evidence`` is an explicitly narrower pass
    based on local source signals. Notes for senses that still need an example
    are assessed inside that request, which happens anyway.
    """
    done = completed_ids(proposals) if done is None else done
    targets: list[NoteTarget] = []
    for path, entry in loaded_entries(source) if entries is None else entries:
        if notes_scope == "evidence" and not note_warrants_assessment(entry):
            continue
        word = str(entry.get("word") or "").strip()
        if not word:
            continue
        article = str(entry.get("article") or "").strip()
        pos = part_of_speech(entry)
        for index, definition in enumerate(entry.get("definitions") or []):
            meaning = str(definition.get("definition") or "").strip()
            sentence = str(definition.get("example") or "").strip()
            sentence_meaning = str(definition.get("example_translation") or "").strip()
            if (
                not meaning
                or not sentence
                or not sentence_meaning
                or "note" in definition
            ):
                continue
            target_id = f"{path.name}:{index}:note"
            if target_id in done:
                continue
            source_hints = model_source_hints(entry, purpose="notes")
            source_hints.update(model_sense_hints(definition))
            targets.append(
                NoteTarget(
                    id=target_id,
                    source=path.name,
                    definition_index=index,
                    word=word,
                    article=article,
                    part_of_speech=pos,
                    meaning=meaning,
                    sentence=sentence,
                    sentence_meaning=sentence_meaning,
                    source_hints=source_hints,
                )
            )
    return targets


def collect_inflections(
    source: Path,
    proposals: Path,
    *,
    entries: Sequence[tuple[Path, dict[str, Any]]] | None = None,
    done: set[str] | None = None,
) -> list[InflectionTarget]:
    done = completed_ids(proposals) if done is None else done
    targets: list[InflectionTarget] = []
    for path, entry in loaded_entries(source) if entries is None else entries:
        word = str(entry.get("word") or "").strip()
        pos = part_of_speech(entry)
        allowed = ALLOWED_INFLECTIONS.get(pos)
        if not word or not allowed:
            continue
        present = entry.get("inflections") or {}
        missing = [key for key in allowed if not present.get(key)]
        if not missing:
            continue
        target_id = f"{path.name}:inflections"
        if target_id in done:
            continue
        targets.append(
            InflectionTarget(
                id=target_id,
                source=path.name,
                word=word,
                article=str(entry.get("article") or "").strip(),
                part_of_speech=pos,
                missing=tuple(missing),
            )
        )
    return targets


def collect_cards(
    source: Path,
    proposals: Path,
    *,
    entries: Sequence[tuple[Path, dict[str, Any]]] | None = None,
    done: set[str] | None = None,
    production_limit: int = 1000,
) -> tuple[list[CardPlanTarget], int]:
    """Collect every complete entry for holistic sense and card-quality review."""
    done = completed_ids(proposals) if done is None else done
    loaded = loaded_entries(source) if entries is None else list(entries)
    ranked_candidates = sorted(
        (
            (path.name, entry)
            for path, entry in loaded
            if production_candidate(entry).score >= PRODUCTION_SCORE_THRESHOLD
        ),
        key=lambda item: card_priority(item[1]),
    )
    production_sources = {name for name, _entry in ranked_candidates[:production_limit]}

    targets: list[CardPlanTarget] = []
    incomplete = 0
    for path, entry in loaded:
        if reviewed_card_plan(entry) is not None:
            continue
        definitions = entry.get("definitions")
        if not isinstance(definitions, list) or not definitions:
            incomplete += 1
            continue
        if any(
            not isinstance(item, dict)
            or not str(item.get("definition") or "").strip()
            or not str(item.get("example") or "").strip()
            or not str(item.get("example_translation") or "").strip()
            for item in definitions
        ):
            incomplete += 1
            continue
        production = production_candidate(entry)
        production_eligible = path.name in production_sources
        target_id = f"{path.name}:cards-v{CARD_PLAN_VERSION}"
        if target_id in done:
            continue
        request_definitions: list[dict[str, object]] = []
        for index, definition in enumerate(definitions):
            item: dict[str, object] = {
                "index": index,
                "meaning": str(definition.get("definition") or "").strip(),
                "sentence": str(definition.get("example") or "").strip(),
                "sentence_meaning": str(
                    definition.get("example_translation") or ""
                ).strip(),
                "existing_note": definition.get("note"),
                "note_assessed": "note" in definition,
            }
            sense_hints = model_sense_hints(definition)
            if sense_hints:
                item["source_hints"] = sense_hints
            request_definitions.append(item)
        level = str((entry.get("learner_level") or {}).get("cefr") or "").upper()
        rank = entry.get("frequency_rank")
        forms = target_forms(entry)
        visible_forms = tuple(
            dict.fromkeys(
                form
                for form in forms
                if any(
                    target_form_spans(definition.get("example"), form)
                    for definition in definitions
                )
            )
        )
        allowed_forms = tuple(
            dict.fromkeys((str(entry.get("word") or "").strip(), *visible_forms))
        )
        targets.append(
            CardPlanTarget(
                id=target_id,
                source=path.name,
                source_digest=card_source_digest(entry),
                word=str(entry.get("word") or "").strip(),
                article=str(entry.get("article") or "").strip(),
                part_of_speech=part_of_speech(entry),
                frequency_rank=rank if isinstance(rank, int) else 0,
                learner_level=level,
                definitions=tuple(request_definitions),
                allowed_target_forms=tuple(form for form in allowed_forms if form),
                production_eligible=production_eligible,
                production_score=production.score,
                production_reasons=production.reasons,
                source_hints=model_source_hints(entry, purpose="notes"),
                production_evidence=production.evidence,
            )
        )
    return targets, incomplete


def chunks(
    items: Sequence[EnrichmentTarget], size: int
) -> list[list[EnrichmentTarget]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def validate_examples(
    targets: list[ExampleTarget], response: object
) -> dict[str, dict[str, str | None]]:
    if not isinstance(response, dict):
        raise TypeError("Codex response must be one JSON object")
    expected = {target.id for target in targets}
    if set(response) != expected:
        missing = sorted(expected - set(response))[:5]
        extra = sorted(set(response) - expected)[:5]
        raise ValueError(
            f"Codex response IDs differ (missing={missing}, extra={extra})"
        )
    by_id = {target.id: target for target in targets}
    validated: dict[str, dict[str, str | None]] = {}
    for target_id, value in response.items():
        target = by_id[target_id]
        if not isinstance(value, dict):
            raise TypeError(f"{target_id}: response value must be an object")
        expected_fields = set(target.missing_fields)
        if set(value) != expected_fields:
            raise ValueError(
                f"{target_id}: response fields differ "
                f"(expected={sorted(expected_fields)}, actual={sorted(value)})"
            )
        sentence = (
            clean_text(value["sentence"], f"{target_id}: sentence")
            if "sentence" in expected_fields
            else target.sentence
        )
        sentence_meaning = (
            clean_text(value["sentence_meaning"], f"{target_id}: sentence_meaning")
            if "sentence_meaning" in expected_fields
            else target.sentence_meaning
        )
        result: dict[str, str | None] = {
            "sentence": sentence,
            "sentence_meaning": sentence_meaning,
        }
        if "note" in expected_fields:
            result["note"] = normalize_optional_note(
                value["note"], f"{target_id}: note"
            )
        validated[target_id] = result
    return validated


def validate_definitions(
    targets: list[DefinitionTarget], response: object
) -> dict[str, dict[str, list[dict[str, str | None]]]]:
    if not isinstance(response, dict):
        raise TypeError("Codex response must be one JSON object")
    expected = {target.id for target in targets}
    if set(response) != expected:
        missing = sorted(expected - set(response))[:5]
        extra = sorted(set(response) - expected)[:5]
        raise ValueError(
            f"Codex response IDs differ (missing={missing}, extra={extra})"
        )
    validated: dict[str, dict[str, list[dict[str, str | None]]]] = {}
    required_keys = {"meaning", "sentence", "sentence_meaning", "note"}
    for target_id, value in response.items():
        if not isinstance(value, dict):
            raise TypeError(f"{target_id}: response value must be an object")
        proposed = value.get("definitions")
        if not isinstance(proposed, list) or not 1 <= len(proposed) <= MAX_DEFINITIONS:
            raise ValueError(f"{target_id}: definitions must contain one or two senses")
        cleaned: list[dict[str, str | None]] = []
        for item in proposed:
            if not isinstance(item, dict):
                raise TypeError(f"{target_id}: each definition must be an object")
            if set(item) != required_keys:
                raise ValueError(
                    f"{target_id}: each definition must contain exactly {sorted(required_keys)}"
                )
            cleaned.append(
                {
                    "meaning": clean_text(item.get("meaning"), f"{target_id}: meaning"),
                    "sentence": clean_text(
                        item.get("sentence"), f"{target_id}: sentence"
                    ),
                    "sentence_meaning": clean_text(
                        item.get("sentence_meaning"), f"{target_id}: sentence_meaning"
                    ),
                    "note": normalize_optional_note(
                        item.get("note"), f"{target_id}: note"
                    ),
                }
            )
        validated[target_id] = {"definitions": cleaned}
    return validated


def validate_notes(
    targets: list[NoteTarget], response: object
) -> dict[str, dict[str, str | None]]:
    if not isinstance(response, dict):
        raise TypeError("Codex response must be one JSON object")
    expected = {target.id for target in targets}
    if set(response) != expected:
        missing = sorted(expected - set(response))[:5]
        extra = sorted(set(response) - expected)[:5]
        raise ValueError(
            f"Codex response IDs differ (missing={missing}, extra={extra})"
        )
    validated: dict[str, dict[str, str | None]] = {}
    for target_id, value in response.items():
        if not isinstance(value, dict) or set(value) != {"note"}:
            raise ValueError(f"{target_id}: response must contain exactly note")
        validated[target_id] = {
            "note": normalize_optional_note(value["note"], f"{target_id}: note")
        }
    return validated


def validate_inflections(
    targets: list[InflectionTarget], response: object
) -> dict[str, dict[str, dict[str, str | None]]]:
    if not isinstance(response, dict):
        raise TypeError("Codex response must be one JSON object")
    expected = {target.id for target in targets}
    if set(response) != expected:
        missing = sorted(expected - set(response))[:5]
        extra = sorted(set(response) - expected)[:5]
        raise ValueError(
            f"Codex response IDs differ (missing={missing}, extra={extra})"
        )
    by_id = {target.id: target for target in targets}
    validated: dict[str, dict[str, dict[str, str | None]]] = {}
    for target_id, value in response.items():
        target = by_id[target_id]
        if not isinstance(value, dict):
            raise TypeError(f"{target_id}: response value must be an object")
        forms = value.get("forms")
        if not isinstance(forms, dict):
            raise TypeError(f"{target_id}: forms must be an object")
        allowed = set(target.missing)
        cleaned: dict[str, str | None] = {}
        for key, form in forms.items():
            if key not in allowed:
                raise ValueError(f"{target_id}: unexpected inflection key {key!r}")
            if form is None:
                cleaned[key] = None
                continue
            if not isinstance(form, str) or not form.strip() or "\n" in form:
                raise ValueError(f"{target_id}: invalid form for {key!r}")
            cleaned[key] = form.strip()
        if set(cleaned) != allowed:
            missing = sorted(allowed - set(cleaned))[:5]
            raise ValueError(f"{target_id}: missing form keys {missing}")
        validated[target_id] = {"forms": cleaned}
    return validated


def _contains_form(text: str, form: str) -> bool:
    return bool(target_form_spans(text, form))


def validate_cards(
    targets: list[CardPlanTarget], response: object
) -> dict[str, dict[str, object]]:
    if not isinstance(response, dict):
        raise TypeError("Codex response must be one JSON object")
    expected = {target.id for target in targets}
    if set(response) != expected:
        missing = sorted(expected - set(response))[:5]
        extra = sorted(set(response) - expected)[:5]
        raise ValueError(
            f"Codex response IDs differ (missing={missing}, extra={extra})"
        )
    by_id = {target.id: target for target in targets}
    card_keys = {
        "source_indices",
        "context_meaning",
        "other_meanings",
        "sentence",
        "sentence_meaning",
        "target_form",
        "note",
        "production",
    }
    production_keys = {"cue", "answer", "target"}
    validated: dict[str, dict[str, object]] = {}
    for target_id, value in response.items():
        target = by_id[target_id]
        if not isinstance(value, dict) or set(value) != {"cards"}:
            raise ValueError(f"{target_id}: response must contain exactly cards")
        proposed = value["cards"]
        if not isinstance(proposed, list) or not 1 <= len(proposed) <= len(
            target.definitions
        ):
            raise ValueError(
                f"{target_id}: cards must contain one or two reviewed senses"
            )
        cleaned_cards: list[dict[str, object]] = []
        covered: list[int] = []
        for card_number, item in enumerate(proposed, start=1):
            label = f"{target_id}: card {card_number}"
            if not isinstance(item, dict) or set(item) != card_keys:
                raise ValueError(f"{label} must contain exactly {sorted(card_keys)}")
            indices = item["source_indices"]
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < len(target.definitions)
                    for index in indices
                )
                or len(set(indices)) != len(indices)
            ):
                raise ValueError(f"{label}: invalid source_indices")
            covered.extend(indices)
            meaning = clean_text(item["context_meaning"], f"{label}: context_meaning")
            if (
                any(separator in meaning for separator in (";", ",", "/"))
                or len(meaning) > 120
            ):
                raise ValueError(
                    f"{label}: context_meaning must be one meaning under 121 characters"
                )
            other = item["other_meanings"]
            if not isinstance(other, list) or len(other) > 4:
                raise ValueError(f"{label}: other_meanings must be a short array")
            other_meanings = [
                clean_text(value, f"{label}: other_meanings") for value in other
            ]
            if any(";" in value or len(value) > 80 for value in other_meanings):
                raise ValueError(f"{label}: invalid other meaning")
            other_meanings = list(dict.fromkeys(other_meanings))
            sentence = clean_text(item["sentence"], f"{label}: sentence")
            sentence_meaning = clean_text(
                item["sentence_meaning"], f"{label}: sentence_meaning"
            )
            target_form = clean_text(item["target_form"], f"{label}: target_form")
            allowed = {form.casefold() for form in target.allowed_target_forms}
            if target_form.casefold() not in allowed:
                raise ValueError(
                    f"{label}: target_form is not the lemma or a known inflection"
                )
            if not _contains_form(sentence, target_form):
                visible_allowed = [
                    form
                    for form in target.allowed_target_forms
                    if _contains_form(sentence, form)
                ]
                if len(visible_allowed) == 1:
                    target_form = visible_allowed[0]
                else:
                    raise ValueError(f"{label}: target_form does not occur in sentence")
            note = normalize_optional_note(item["note"], f"{label}: note")
            production = item["production"]
            cleaned_production: dict[str, str] | None = None
            if production is not None:
                if not target.production_eligible:
                    raise ValueError(f"{label}: production was not locally eligible")
                if (
                    not isinstance(production, dict)
                    or set(production) != production_keys
                ):
                    raise ValueError(
                        f"{label}: production must contain exactly {sorted(production_keys)}"
                    )
                cue = clean_text(production["cue"], f"{label}: production cue")
                answer = clean_text(production["answer"], f"{label}: production answer")
                production_target = clean_text(
                    production["target"], f"{label}: production target"
                )
                if len(cue) > 240 or len(answer) > 240 or len(production_target) > 120:
                    raise ValueError(f"{label}: production text is too long")
                if len(production_target.split()) < 2:
                    raise ValueError(
                        f"{label}: production target must be a multiword unit"
                    )
                if not _contains_form(answer, production_target):
                    raise ValueError(
                        f"{label}: production target does not occur in answer"
                    )
                if answer.casefold() in cue.casefold():
                    raise ValueError(f"{label}: production cue reveals the answer")
                cleaned_production = {
                    "cue": cue,
                    "answer": answer,
                    "target": production_target,
                }
            cleaned_cards.append(
                {
                    "source_indices": sorted(indices),
                    "context_meaning": meaning,
                    "other_meanings": other_meanings,
                    "sentence": sentence,
                    "sentence_meaning": sentence_meaning,
                    "target_form": target_form,
                    "note": note,
                    "production": cleaned_production,
                }
            )
        if sorted(covered) != list(range(len(target.definitions))):
            raise ValueError(
                f"{target_id}: source indices must each be covered exactly once"
            )
        validated[target_id] = {"cards": cleaned_cards}
    return validated


def parse_response(raw_response: str) -> object:
    """Parse strict JSON while tolerating a benign prose/Markdown wrapper."""
    stripped = raw_response.strip()
    candidates = [stripped]
    candidates.extend(
        block.strip()
        for block in re.findall(
            r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL
        )
    )
    first_brace, last_brace = stripped.find("{"), stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    failures: list[tuple[str, json.JSONDecodeError]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            failures.append((candidate, exc))
    if not failures:
        raise ValueError("Codex final message was empty")

    candidate, error = max(failures, key=lambda item: len(item[0]))
    start, end = max(0, error.pos - 300), min(len(candidate), error.pos + 300)
    context = (
        candidate[start : error.pos] + "<<<PARSE ERROR>>>" + candidate[error.pos : end]
    )
    raise ValueError(
        "Codex final message was not valid JSON: "
        f"{error.msg} at line {error.lineno}, column {error.colno}, "
        f"character {error.pos}. Context: {context}"
    ) from error


def value_schema_for(target: EnrichmentTarget) -> dict[str, object]:
    if isinstance(target, ExampleTarget):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                key: {"type": ["string", "null"]}
                if key == "note"
                else {"type": "string"}
                for key in target.missing_fields
            },
            "required": list(target.missing_fields),
        }
    if isinstance(target, DefinitionTarget):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "definitions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "meaning": {"type": "string"},
                            "sentence": {"type": "string"},
                            "sentence_meaning": {"type": "string"},
                            "note": {"type": ["string", "null"]},
                        },
                        "required": [
                            "meaning",
                            "sentence",
                            "sentence_meaning",
                            "note",
                        ],
                    },
                }
            },
            "required": ["definitions"],
        }
    if isinstance(target, NoteTarget):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"note": {"type": ["string", "null"]}},
            "required": ["note"],
        }
    if isinstance(target, InflectionTarget):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "forms": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        key: {"type": ["string", "null"]} for key in target.missing
                    },
                    "required": list(target.missing),
                }
            },
            "required": ["forms"],
        }
    if isinstance(target, CardPlanTarget):
        production_schema: dict[str, object]
        if target.production_eligible:
            production_schema = {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "cue": {"type": "string"},
                            "answer": {"type": "string"},
                            "target": {"type": "string"},
                        },
                        "required": ["cue", "answer", "target"],
                    },
                ]
            }
        else:
            production_schema = {"type": "null"}
        card_properties = {
            "source_indices": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "enum": list(range(len(target.definitions))),
                },
                "minItems": 1,
                "maxItems": len(target.definitions),
            },
            "context_meaning": {"type": "string"},
            "other_meanings": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "sentence": {"type": "string"},
            "sentence_meaning": {"type": "string"},
            "target_form": {"type": "string"},
            "note": {"type": ["string", "null"]},
            "production": production_schema,
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cards": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": len(target.definitions),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": card_properties,
                        "required": list(card_properties),
                    },
                }
            },
            "required": ["cards"],
        }
    raise TypeError(f"Unsupported Codex target type: {type(target).__name__}")


def output_schema_for(targets: list[EnrichmentTarget]) -> dict[str, object]:
    """Build a compact strict schema with exactly the requested stable IDs."""
    if not targets:
        raise ValueError("Cannot build a response schema for an empty batch")
    target_type = type(targets[0])
    if any(type(target) is not target_type for target in targets):
        raise TypeError("A Codex batch cannot mix target kinds")

    required = [target.id for target in targets]
    definitions: dict[str, object] = {}
    references: dict[str, object] = {}
    names_by_schema: dict[str, str] = {}
    for target in targets:
        value_schema = value_schema_for(target)
        fingerprint = json.dumps(value_schema, sort_keys=True, separators=(",", ":"))
        name = names_by_schema.get(fingerprint)
        if name is None:
            name = f"value_{len(names_by_schema) + 1}"
            names_by_schema[fingerprint] = name
            definitions[name] = value_schema
        references[target.id] = {"$ref": f"#/$defs/{name}"}

    return {
        "$defs": definitions,
        "type": "object",
        "additionalProperties": False,
        "properties": references,
        "required": required,
    }


def run_with_heartbeat(
    command: list[str],
    payload: str,
    timeout: int,
    *,
    progress_label: str = "codex",
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    """Capture Codex output, report liveness, and reap it on interruption."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    started = time.monotonic()
    next_heartbeat = 30.0
    send_input = True
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                process.kill()
                process.communicate()
                raise CodexBatchCancelled(f"{progress_label} was cancelled")
            elapsed = time.monotonic() - started
            remaining = timeout - elapsed
            if remaining <= 0:
                process.kill()
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    command, timeout, output=stdout, stderr=stderr
                )
            try:
                stdout, stderr = process.communicate(
                    input=payload if send_input else None,
                    timeout=min(1.0, remaining),
                )
                return subprocess.CompletedProcess(
                    command, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                send_input = False
                elapsed = time.monotonic() - started
                if elapsed >= next_heartbeat:
                    print(
                        f"[{progress_label}] still running ({int(elapsed)}s elapsed)",
                        flush=True,
                    )
                    while next_heartbeat <= elapsed:
                        next_heartbeat += 30.0
    except KeyboardInterrupt:
        process.kill()
        process.communicate()
        raise


def ensure_codex_login(
    codex_command: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Fail before a large run when the subscription CLI is not authenticated."""
    try:
        completed = runner(
            [codex_command, "login", "status"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Codex CLI not found: {codex_command}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while checking Codex login status") from exc
    status = (completed.stdout + completed.stderr).strip()
    normalized = status.casefold()
    if completed.returncode != 0 or "not logged in" in normalized:
        detail = status or f"exit status {completed.returncode}"
        raise RuntimeError(
            f"Codex CLI is not subscription-authenticated ({detail}). Run: codex login"
        )
    if "api key" in normalized:
        raise RuntimeError(
            "Codex CLI is authenticated with an API key, not a ChatGPT subscription. "
            "Run `codex logout` and then `codex login`."
        )


def run_codex_batch(
    targets: list[EnrichmentTarget],
    *,
    model: str,
    codex_command: str,
    timeout: int,
    prompt: str,
    invalid_response_path: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    progress_label: str = "codex",
    cancel_event: threading.Event | None = None,
) -> object:
    payload = json.dumps(
        [target.request() for target in targets],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with tempfile.TemporaryDirectory(prefix="swedish-anki-codex-") as temporary_dir:
        response_path = Path(temporary_dir) / "last-message.json"
        schema_path = Path(temporary_dir) / "response-schema.json"
        schema_path.write_text(
            json.dumps(output_schema_for(targets), ensure_ascii=False),
            encoding="utf-8",
        )
        command = [
            codex_command,
            "exec",
            "-m",
            model,
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            prompt,
        ]
        try:
            completed = (
                run_with_heartbeat(
                    command,
                    payload,
                    timeout,
                    progress_label=progress_label,
                    cancel_event=cancel_event,
                )
                if runner is None
                else runner(
                    command,
                    input=payload,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    cwd=ROOT,
                    check=False,
                )
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Codex CLI not found: {codex_command}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Codex batch exceeded {timeout} seconds") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(
                f"Codex exited with status {completed.returncode}: {detail}"
            )
        raw_response = (
            response_path.read_text(encoding="utf-8")
            if response_path.is_file()
            else completed.stdout
        )
    try:
        return parse_response(raw_response)
    except ValueError as exc:
        if invalid_response_path is None:
            raise
        invalid_response_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_response_path.write_text(raw_response, encoding="utf-8")
        raise ValueError(
            f"{exc} Raw response saved to {invalid_response_path}."
        ) from exc


def collect_pending(
    source: Path,
    proposals: Path,
    kinds: Sequence[str],
    *,
    notes_scope: str = "all",
    production_limit: int = 1000,
) -> tuple[dict[str, list[EnrichmentTarget]], dict[str, int]]:
    pending: dict[str, list[EnrichmentTarget]] = {kind: [] for kind in kinds}
    details: dict[str, int] = {}
    entries = loaded_entries(source)
    done = completed_ids(proposals)
    if "definitions" in kinds:
        pending["definitions"] = list(
            collect_definitions(source, proposals, entries=entries, done=done)
        )
        details["definitions_skipped_meaning"] = 0
    if "examples" in kinds:
        examples, skipped_no_meaning = collect_examples(
            source, proposals, entries=entries, done=done
        )
        pending["examples"] = list(examples)
        details["examples_skipped_no_meaning"] = skipped_no_meaning
    if "notes" in kinds:
        pending["notes"] = list(
            collect_notes(
                source,
                proposals,
                entries=entries,
                done=done,
                notes_scope=notes_scope,
            )
        )
        details["notes_skipped"] = 0
    if "inflections" in kinds:
        pending["inflections"] = list(
            collect_inflections(source, proposals, entries=entries, done=done)
        )
        details["inflections_skipped"] = 0
    if "cards" in kinds:
        cards, incomplete = collect_cards(
            source,
            proposals,
            entries=entries,
            done=done,
            production_limit=production_limit,
        )
        pending["cards"] = list(cards)
        details["cards_skipped_incomplete"] = incomplete
    return pending, details


def selected_chunks(
    pending: dict[str, list[EnrichmentTarget]],
    kinds: Sequence[str],
    *,
    batch_size: int,
    limit: int | None,
    max_batches: int,
) -> list[tuple[str, list[EnrichmentTarget]]]:
    budget = limit
    result: list[tuple[str, list[EnrichmentTarget]]] = []
    for kind in kinds:
        targets = pending[kind]
        if budget is not None:
            if budget <= 0:
                break
            targets = targets[:budget]
            budget -= len(targets)
        effective_batch_size = (
            min(batch_size, MAX_CARD_BATCH_SIZE) if kind == "cards" else batch_size
        )
        for chunk in chunks(targets, effective_batch_size):
            result.append((kind, chunk))
    if max_batches:
        result = result[:max_batches]
    return result


def prompt_for(kind: str) -> str:
    return {
        "definitions": DEFINITIONS_PROMPT,
        "examples": EXAMPLE_PROMPT,
        "notes": NOTES_PROMPT,
        "inflections": INFLECTIONS_PROMPT,
        "cards": CARDS_PROMPT,
    }[kind]


def validate_for(
    kind: str, targets: list[EnrichmentTarget], response: object
) -> dict[str, object]:
    if kind == "definitions":
        return validate_definitions(list(targets), response)  # type: ignore[arg-type]
    if kind == "examples":
        return validate_examples(list(targets), response)  # type: ignore[arg-type]
    if kind == "notes":
        return validate_notes(list(targets), response)  # type: ignore[arg-type]
    if kind == "inflections":
        return validate_inflections(list(targets), response)  # type: ignore[arg-type]
    return validate_cards(list(targets), response)  # type: ignore[arg-type]


def validate_batch(
    kind: str, targets: list[EnrichmentTarget], response: object
) -> BatchValidation:
    """Validate independently so one bad target does not discard a whole batch."""
    if not isinstance(response, dict):
        raise TypeError("Codex response must be one JSON object")
    expected = {target.id for target in targets}
    extra = set(response) - expected
    if extra:
        raise ValueError(f"Codex response contains unexpected IDs: {sorted(extra)[:5]}")

    valid: dict[str, object] = {}
    issues: list[ValidationIssue] = []
    for target in targets:
        if target.id not in response:
            issues.append(
                ValidationIssue(
                    id=target.id,
                    source=target.source,
                    error="Codex response omitted this target ID",
                )
            )
            continue
        try:
            valid.update(validate_for(kind, [target], {target.id: response[target.id]}))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    id=target.id,
                    source=target.source,
                    error=str(exc),
                )
            )
    return BatchValidation(valid=valid, issues=tuple(issues))


def execute_batch(
    number: int,
    total: int,
    kind: str,
    targets: list[EnrichmentTarget],
    args: argparse.Namespace,
    cancel_event: threading.Event,
) -> BatchExecution:
    """Run and validate one isolated Codex process for a worker thread."""
    label = f"batch {number}/{total} {kind}"
    print(
        f"Running Codex batch {number}/{total} ([{kind}] {len(targets)} targets)…",
        flush=True,
    )
    started = time.monotonic()
    response = run_codex_batch(
        targets,
        model=args.model,
        codex_command=args.codex_command,
        timeout=args.timeout,
        prompt=prompt_for(kind),
        invalid_response_path=(
            args.failures.parent
            / "invalid_codex_responses"
            / f"batch_{number:03d}_{kind}.txt"
        ),
        progress_label=label,
        cancel_event=cancel_event,
    )
    return BatchExecution(
        number=number,
        kind=kind,
        targets=targets,
        validation=validate_batch(kind, targets, response),
        elapsed=time.monotonic() - started,
    )


def propose(args: argparse.Namespace) -> int:
    if not args.source.is_dir():
        print(f"Source folder does not exist: {args.source}", file=sys.stderr)
        return 2
    kinds = args.kind or list(LEXICAL_KINDS)
    pending, details = collect_pending(
        args.source,
        args.proposals,
        kinds,
        notes_scope=args.notes_scope,
        production_limit=getattr(args, "production_limit", 1000),
    )
    selected = selected_chunks(
        pending,
        kinds,
        batch_size=args.batch_size,
        limit=args.limit,
        max_batches=args.max_batches,
    )
    by_kind = {kind: len(pending[kind]) for kind in kinds}
    pending_summary = ", ".join(
        f"{count} {kind}" for kind, count in by_kind.items() if count
    )
    print(
        f"Pending targets: {pending_summary or 'none'}; this run: "
        f"{sum(len(item[1]) for item in selected)} target(s) in "
        f"{len(selected)} batch(es)."
    )
    if details.get("examples_skipped_no_meaning"):
        print(
            f"Skipped {details['examples_skipped_no_meaning']} definitions without a word meaning."
        )
    if details.get("cards_skipped_incomplete"):
        print(
            f"Deferred {details['cards_skipped_incomplete']} card plans until lexical fields are complete."
        )
    if not selected:
        print("No pending targets. Source JSON and proposal files were not changed.")
        return 0
    if args.dry_run:
        for number, (kind, batch) in enumerate(selected, start=1):
            preview = ", ".join(target.word for target in batch[:5])
            print(
                f"Batch {number}: [{kind}] {len(batch)} targets ({preview}{'…' if len(batch) > 5 else ''})"
            )
        print("Dry run only. Source JSON and proposal files were not changed.")
        return 0

    try:
        ensure_codex_login(args.codex_command)
    except RuntimeError as exc:
        print(f"Codex preflight failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"Codex preflight passed: subscription login is available; model={args.model}."
    )

    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    written = rejected = failed_batches = 0
    worker_count = min(args.workers, len(selected))
    print(
        f"Starting {len(selected)} batch(es) with {worker_count} concurrent "
        f"Codex worker(s)."
    )
    cancel_event = threading.Event()
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="codex-enrichment",
    )
    futures: dict[Future[BatchExecution], tuple[int, str, list[EnrichmentTarget]]] = {}
    interrupted = False
    try:
        for batch_number, (kind, batch) in enumerate(selected, start=1):
            future = executor.submit(
                execute_batch,
                batch_number,
                len(selected),
                kind,
                batch,
                args,
                cancel_event,
            )
            futures[future] = (batch_number, kind, batch)

        for future in as_completed(futures):
            batch_number, kind, batch = futures[future]
            try:
                execution = future.result()
            except (CancelledError, CodexBatchCancelled):
                continue
            except (RuntimeError, TypeError, ValueError) as exc:
                failure = {
                    "kind": kind,
                    "ids": [target.id for target in batch],
                    "model": args.model,
                    "error": str(exc),
                }
                with args.failures.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"Batch {batch_number} failed: {exc}", file=sys.stderr)
                failed_batches += 1
                cancel_event.set()
                for pending_future in futures:
                    pending_future.cancel()
                continue

            batch_written = 0
            with args.proposals.open("a", encoding="utf-8") as output:
                for target in execution.targets:
                    if target.id not in execution.validation.valid:
                        continue
                    record = target.row(
                        execution.validation.valid[target.id],
                        args.model,  # type: ignore[arg-type]
                    )
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    batch_written += 1
            if execution.validation.issues:
                with args.failures.open("a", encoding="utf-8") as output:
                    for issue in execution.validation.issues:
                        output.write(
                            json.dumps(
                                {
                                    "kind": execution.kind,
                                    "id": issue.id,
                                    "source": issue.source,
                                    "model": args.model,
                                    "error": issue.error,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                rejected += len(execution.validation.issues)
            print(
                f"Saved batch {execution.number} in {execution.elapsed:.1f}s: "
                f"{batch_written} proposals, "
                f"{len(execution.validation.issues)} rejected; "
                "source JSON is unchanged."
            )
    except KeyboardInterrupt:
        interrupted = True
        cancel_event.set()
        for future in futures:
            future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if interrupted:
        print(
            "Interrupted. Active Codex batches were terminated and not saved; "
            "previously saved batches remain resumable.",
            file=sys.stderr,
        )
        return 130
    print(
        f"Added {written} reviewable proposals to {args.proposals}; "
        f"{rejected} targets were rejected and {failed_batches} batch(es) failed. "
        "Unsuccessful targets remain pending."
    )
    return 1 if rejected or failed_batches else 0


def write_entry(path: Path, entry: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def apply_example(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    proposed_word = proposal_text(record, "word")
    proposed_meaning = proposal_text(record, "meaning")
    proposed_sentence = proposal_text(record, "sentence")
    proposed_sentence_meaning = proposal_text(record, "sentence_meaning")
    index = record["definition_index"]
    if not isinstance(index, int) or index < 0:
        raise ValueError("definition_index must be a non-negative integer")
    definition = entry["definitions"][index]
    if (
        str(entry.get("word") or "").strip() != proposed_word
        or str(definition.get("definition") or "").strip() != proposed_meaning
    ):
        raise ValueError("source word/meaning changed after proposal")
    current_sentence = str(definition.get("example") or "").strip()
    current_meaning = str(definition.get("example_translation") or "").strip()
    if current_sentence and current_sentence != proposed_sentence:
        raise ValueError("source sentence conflicts with proposal")
    if current_meaning and current_meaning != proposed_sentence_meaning:
        raise ValueError("source sentence meaning conflicts with proposal")
    has_note_assessment = "note" in record
    proposed_note = (
        normalize_optional_note(record["note"]) if has_note_assessment else None
    )
    if (
        has_note_assessment
        and "note" in definition
        and definition["note"] != proposed_note
    ):
        raise ValueError("source note conflicts with proposal")
    changed = False
    if not current_sentence:
        definition["example"] = proposed_sentence
        changed = True
    if not current_meaning:
        definition["example_translation"] = proposed_sentence_meaning
        changed = True
    if has_note_assessment and "note" not in definition:
        definition["note"] = proposed_note
        changed = True
    return changed


def apply_definitions(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    proposed_word = proposal_text(record, "word")
    if str(entry.get("word") or "").strip() != proposed_word:
        raise ValueError("source word changed after proposal")
    proposed = record.get("definitions")
    if not isinstance(proposed, list) or not 1 <= len(proposed) <= 2:
        raise ValueError("proposal definitions must contain one or two senses")
    definitions: list[dict[str, str | None]] = []
    for item in proposed:
        if not isinstance(item, dict):
            raise TypeError("each proposed definition must be an object")
        if "note" not in item:
            raise ValueError("each proposed definition must contain note or null")
        definitions.append(
            {
                "definition": clean_text(item.get("meaning"), "meaning"),
                "example": clean_text(item.get("sentence"), "sentence"),
                "example_translation": clean_text(
                    item.get("sentence_meaning"), "sentence_meaning"
                ),
                "note": normalize_optional_note(item.get("note")),
            }
        )
    existing = entry.get("definitions") or []
    if existing:
        existing_values = [
            {
                "definition": str(item.get("definition") or "").strip(),
                "example": str(item.get("example") or "").strip(),
                "example_translation": str(
                    item.get("example_translation") or ""
                ).strip(),
                "note": item.get("note"),
            }
            for item in existing
        ]
        if existing_values == definitions:
            return False
        raise ValueError("source already has different definitions")
    entry["definitions"] = definitions
    return True


def apply_note(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    proposed_word = proposal_text(record, "word")
    proposed_meaning = proposal_text(record, "meaning")
    index = record["definition_index"]
    if not isinstance(index, int) or index < 0:
        raise ValueError("definition_index must be a non-negative integer")
    definition = entry["definitions"][index]
    if (
        str(entry.get("word") or "").strip() != proposed_word
        or str(definition.get("definition") or "").strip() != proposed_meaning
    ):
        raise ValueError("source word/meaning changed after proposal")
    for key, source_key in (
        ("sentence", "example"),
        ("sentence_meaning", "example_translation"),
    ):
        proposed = record.get(key)
        if not isinstance(proposed, str):
            raise ValueError(f"proposal {key} must be a string")  # noqa: TRY004
        if str(definition.get(source_key) or "").strip() != proposed.strip():
            raise ValueError(f"source {source_key} changed after proposal")
    if "note" not in record:
        raise ValueError("proposal must contain note, including an explicit null")
    proposed_note = normalize_optional_note(record["note"])
    if "note" in definition:
        if definition["note"] == proposed_note:
            return False
        raise ValueError("source note conflicts with proposal")
    definition["note"] = proposed_note
    return True


def apply_inflections(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    proposed_word = proposal_text(record, "word")
    if str(entry.get("word") or "").strip() != proposed_word:
        raise ValueError("source word changed after proposal")
    forms = record.get("forms")
    if not isinstance(forms, dict):
        raise TypeError("proposal forms must be an object")
    pos = part_of_speech(entry)
    allowed = set(ALLOWED_INFLECTIONS.get(pos, ()))
    present = entry.get("inflections") or {}
    additions: dict[str, str] = {}
    for key, form in forms.items():
        if key not in allowed:
            raise ValueError(f"inflection key {key!r} is not allowed for {pos!r}")
        if form is None:
            continue
        if not isinstance(form, str) or not form.strip() or "\n" in form:
            raise ValueError(f"invalid form for {key!r}")
        normalized = form.strip()
        if present.get(key):
            if present[key] != normalized:
                raise ValueError(f"source inflection conflicts for {key!r}")
            continue
        additions[key] = normalized
    if not additions:
        return False
    entry["inflections"] = {**present, **additions}
    return True


def apply_cards(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    if record.get("card_plan_version") != CARD_PLAN_VERSION:
        raise ValueError("unsupported card plan version")
    if record.get("source_digest") != card_source_digest(entry):
        raise ValueError("source card content changed after proposal")
    for key, current in (
        ("word", str(entry.get("word") or "").strip()),
        ("part_of_speech", part_of_speech(entry)),
    ):
        if proposal_text(record, key) != current:
            raise ValueError(f"source {key} changed after proposal")
    article = record.get("article")
    if (
        not isinstance(article, str)
        or article.strip() != str(entry.get("article") or "").strip()
    ):
        raise ValueError("source article changed after proposal")
    production = production_candidate(entry)
    proposed_score = record.get("production_score")
    if not isinstance(proposed_score, int) or isinstance(proposed_score, bool):
        raise TypeError("proposal production_score must be an integer")
    if proposed_score != production.score:
        raise ValueError("production score changed after proposal")
    production_eligible = record.get("production_eligible") is True
    if production_eligible and production.score < PRODUCTION_SCORE_THRESHOLD:
        raise ValueError("entry no longer qualifies for production review")
    definitions = entry.get("definitions") or []
    target = CardPlanTarget(
        id=proposal_text(record, "id"),
        source=proposal_text(record, "source"),
        source_digest=proposal_text(record, "source_digest"),
        word=proposal_text(record, "word"),
        article=str(record.get("article") or "").strip(),
        part_of_speech=proposal_text(record, "part_of_speech"),
        frequency_rank=int(entry.get("frequency_rank") or 0),
        learner_level=str((entry.get("learner_level") or {}).get("cefr") or ""),
        definitions=tuple({"index": index} for index in range(len(definitions))),
        allowed_target_forms=target_forms(entry),
        production_eligible=production_eligible,
        production_score=production.score,
        production_reasons=production.reasons,
    )
    cards = validate_cards([target], {target.id: {"cards": record.get("cards")}})[
        target.id
    ]["cards"]
    plan = {
        "version": CARD_PLAN_VERSION,
        "source_digest": target.source_digest,
        "reviewed_by": (
            f"{proposal_text(record, 'generator')}:{proposal_text(record, 'model')}"
        ),
        "cards": cards,
    }
    existing = entry.get("card_plan")
    if existing:
        if existing == plan:
            return False
        raise ValueError("source already has a different card plan")
    entry["card_plan"] = plan
    return True


def apply(args: argparse.Namespace) -> int:
    if not args.apply:
        print(
            "Refusing to write. Review proposals, then re-run with apply --apply.",
            file=sys.stderr,
        )
        return 2
    if not args.proposals.is_file():
        print(f"Proposal file does not exist: {args.proposals}", file=sys.stderr)
        return 2
    applied = skipped = errors = 0
    source_root = args.source.resolve()
    for line_number, line in enumerate(
        args.proposals.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
            kind = record.get("kind")
            if kind not in ALL_KINDS:
                raise ValueError(f"unsupported proposal kind: {kind!r}")
            source_name = proposal_text(record, "source")
            path = (source_root / source_name).resolve()
            if path.parent != source_root:
                raise ValueError("proposal source must be a filename inside --source")
            entry = load_json(path)
            if kind == "definitions":
                changed = apply_definitions(entry, record)
            elif kind == "examples":
                changed = apply_example(entry, record)
            elif kind == "notes":
                changed = apply_note(entry, record)
            elif kind == "inflections":
                changed = apply_inflections(entry, record)
            else:
                changed = apply_cards(entry, record)
            if changed:
                write_entry(path, entry)
                applied += 1
            else:
                skipped += 1
        except (
            FileNotFoundError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(f"Line {line_number}: skipped: {exc}", file=sys.stderr)
            skipped += 1
            errors += 1
    print(
        f"Applied {applied} proposals; skipped {skipped}. No Anki operations were performed."
    )
    return 1 if errors else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    common.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)

    propose_parser = commands.add_parser("propose", parents=[common])
    propose_parser.add_argument("--model", default=DEFAULT_MODEL)
    propose_parser.add_argument("--codex-command", default="codex")
    propose_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    propose_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Concurrent Codex CLI processes. Start with 2–4; higher values consume "
            "subscription quota faster and may be throttled. Default: 1."
        ),
    )
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
        "--kind",
        action="append",
        choices=list(ALL_KINDS),
        help=(
            "Job kind(s) to propose; repeatable. Default: all lexical kinds. "
            "Run --kind cards after applying lexical proposals."
        ),
    )
    propose_parser.add_argument(
        "--production-limit",
        type=int,
        default=1000,
        help=(
            "Maximum locally ranked entries eligible for an optional production "
            "card during holistic card review. Default: 1000."
        ),
    )
    propose_parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Batches to run across all kinds; 0 means all. Default: 1.",
    )
    propose_parser.add_argument("--limit", type=int)
    propose_parser.add_argument("--timeout", type=int, default=1800)
    propose_parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    propose_parser.add_argument("--dry-run", action="store_true")
    propose_parser.set_defaults(handler=propose)

    apply_parser = commands.add_parser("apply", parents=[common])
    apply_parser.add_argument("--apply", action="store_true")
    apply_parser.set_defaults(handler=apply)
    return result


def main() -> int:
    argument_parser = parser()
    arguments = argument_parser.parse_args()
    if (
        getattr(arguments, "batch_size", 1) < 1
        or getattr(arguments, "batch_size", 1) > 500
    ):
        argument_parser.error("--batch-size must be between 1 and 500")
    if not 1 <= getattr(arguments, "workers", 1) <= MAX_WORKERS:
        argument_parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if getattr(arguments, "max_batches", 0) < 0:
        argument_parser.error("--max-batches cannot be negative")
    if getattr(arguments, "production_limit", 0) < 0:
        argument_parser.error("--production-limit cannot be negative")
    if getattr(arguments, "limit", None) is not None and arguments.limit < 1:
        argument_parser.error("--limit must be at least 1")
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
