"""Deterministic card-planning signals and safe local recognition plans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from functions.enrichment_quality import informative_construction, learner_order

CARD_PLAN_VERSION = 2
PRODUCTION_SCORE_THRESHOLD = 7
MAX_PRODUCTION_HINTS = 3


@dataclass(frozen=True, slots=True)
class ProductionCandidate:
    score: int
    evidence: dict[str, list[str]]
    reasons: tuple[str, ...]


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


def _source_values(entry: dict[str, Any], key: str, limit: int) -> list[str]:
    lexical_info = entry.get("lexical_info")
    if not isinstance(lexical_info, dict):
        return []
    values = lexical_info.get(key)
    if not isinstance(values, list):
        return []
    return list(
        dict.fromkeys(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
    )[:limit]


def target_forms(entry: dict[str, Any]) -> tuple[str, ...]:
    """Return source forms plus conservative regular forms and variants."""
    word = str(entry.get("word") or "").strip()
    inflections = entry.get("inflections")
    forms = [word]
    if isinstance(inflections, dict):
        forms.extend(
            str(value).strip()
            for value in inflections.values()
            if isinstance(value, str) and value.strip()
        )
    lexical_info = entry.get("lexical_info")
    variants = lexical_info.get("variants") if isinstance(lexical_info, dict) else None
    if isinstance(variants, list):
        forms.extend(
            value.strip()
            for value in variants
            if isinstance(value, str) and value.strip()
        )
    pos = part_of_speech(entry)
    if word and pos == "noun":
        if word.endswith("a"):
            stem = word.removesuffix("a")
            forms.extend((f"{stem}an", f"{stem}or", f"{stem}orna"))
        else:
            forms.extend(
                f"{word}{suffix}"
                for suffix in (
                    "en",
                    "et",
                    "n",
                    "t",
                    "ar",
                    "er",
                    "or",
                    "na",
                    "arna",
                    "erna",
                    "orna",
                )
            )
    elif word and pos == "adjective":
        forms.extend(f"{word}{suffix}" for suffix in ("t", "a", "are", "ast"))
    if pos == "conjunction" and len(parts := word.split()) == 2:
        forms.append(f"{parts[0]}…{parts[1]}")
    return tuple(dict.fromkeys(form for form in forms if form))


def target_form_spans(sentence: object, form: object) -> list[tuple[int, int]]:
    """Locate a contiguous form or every part of a discontinuous ellipsis form."""
    context = str(sentence or "")
    target = str(form or "").strip()
    if not context or not target:
        return []
    parts = [part.strip() for part in re.split(r"…|\.{3}", target) if part.strip()]
    if not parts:
        return []
    spans: list[tuple[int, int]] = []
    offset = 0
    for part in parts:
        match = re.search(
            rf"(?<!\w){re.escape(part)}(?!\w)",
            context[offset:],
            re.IGNORECASE,
        )
        if not match:
            return []
        start, end = offset + match.start(), offset + match.end()
        spans.append((start, end))
        offset = end
    return spans


def find_target_form(entry: dict[str, Any], sentence: object) -> str | None:
    """Find a displayed lemma or known inflection as a whole form in context."""
    context = str(sentence or "").strip()
    if not context:
        return None
    word = str(entry.get("word") or "").strip()
    forms = list(target_forms(entry))
    # Prefer the lemma when present; otherwise prefer the longest known form so
    # a contained shorter form cannot steal the match.
    ordered = ([word] if word else []) + sorted(
        (form for form in forms if form != word), key=len, reverse=True
    )
    return next((form for form in ordered if target_form_spans(context, form)), None)


def production_candidate(entry: dict[str, Any]) -> ProductionCandidate:
    """Score reusable output value using local evidence, level, and frequency."""
    score = 0
    reasons: list[str] = []
    level = str((entry.get("learner_level") or {}).get("cefr") or "").upper()
    level_points = {"A1": 5, "A2": 4, "B1": 3, "B2": 1}.get(level, 0)
    if level_points:
        score += level_points
        reasons.append(f"learner-level:{level.lower()}")

    rank = entry.get("frequency_rank")
    rank_points = (
        4
        if isinstance(rank, int) and rank <= 1000
        else 2
        if isinstance(rank, int) and rank <= 3000
        else 1
        if isinstance(rank, int) and rank <= 5000
        else 0
    )
    if rank_points:
        score += rank_points
        reasons.append("high-frequency")

    constructions = [
        value
        for value in _source_values(entry, "constructions", MAX_PRODUCTION_HINTS)
        if informative_construction(value)
    ]
    idioms = _source_values(entry, "idioms", 2)
    kelly_hints = [
        value
        for value in _source_values(entry, "kelly_hints", MAX_PRODUCTION_HINTS)
        if value.casefold().startswith("usage pattern:")
    ]
    if constructions:
        score += 4
        reasons.append("construction")
    if idioms:
        score += 5
        reasons.append("idiom")
    if kelly_hints:
        score += 4
        reasons.append("kelly-usage-pattern")

    definitions = entry.get("definitions") or []
    phrase_examples = [
        str(item.get("example") or "").strip()
        for item in definitions
        if isinstance(item, dict)
        and 2 <= len(str(item.get("example") or "").split()) <= 8
        and not str(item.get("example") or "").rstrip().endswith((".", "!", "?"))
    ][:2]
    if phrase_examples:
        score += 2
        reasons.append("compact-context")

    notes = " ".join(
        str(item.get("note") or "") for item in definitions if isinstance(item, dict)
    ).casefold()
    if any(
        marker in notes
        for marker in (
            "followed by",
            "construction",
            "takes the preposition",
            "fixed phrase",
            "fixed expression",
            "commonly used with",
            "usual expression",
        )
    ):
        score += 10
        reasons.append("reviewed-usage-note")

    evidence = {
        key: values
        for key, values in (
            ("constructions", constructions),
            ("idioms", idioms),
            ("kelly_hints", kelly_hints),
            ("compact_examples", phrase_examples),
        )
        if values
    }
    evidence_text = " ".join(value for values in evidence.values() for value in values)
    if re.search(r"(?<!\w)sig(?!\w)", evidence_text, re.IGNORECASE):
        score += 2
        reasons.append("reflexive")

    usage = " ".join(_source_values(entry, "usage_labels", 3)).casefold()
    if any(marker in usage for marker in ("ålderd", "regional", "dialekt")):
        score -= 6
        reasons.append("marked-low-transfer")

    return ProductionCandidate(
        score=max(score, 0), evidence=evidence, reasons=tuple(reasons)
    )


def card_source_digest(entry: dict[str, Any]) -> str:
    """Fingerprint source fields whose change invalidates a reviewed card plan."""
    lexical_info = entry.get("lexical_info")
    relevant_lexical_info = (
        {
            key: lexical_info.get(key)
            for key in (
                "variants",
                "constructions",
                "usage_labels",
                "translation_comments",
                "explanations",
                "idioms",
                "kelly_hints",
            )
            if lexical_info.get(key)
        }
        if isinstance(lexical_info, dict)
        else {}
    )
    payload = {
        "word": entry.get("word"),
        "article": entry.get("article"),
        "tags": entry.get("tags"),
        "frequency_rank": entry.get("frequency_rank"),
        "learner_level": entry.get("learner_level"),
        "definitions": entry.get("definitions"),
        "inflections": entry.get("inflections"),
        "lexical_info": relevant_lexical_info,
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def reviewed_card_plan(entry: dict[str, Any]) -> dict[str, object] | None:
    plan = entry.get("card_plan")
    if not isinstance(plan, dict) or plan.get("version") != CARD_PLAN_VERSION:
        return None
    if plan.get("source_digest") != card_source_digest(entry):
        return None
    cards = plan.get("cards")
    return plan if isinstance(cards, list) and cards else None


def card_priority(entry: dict[str, Any]) -> tuple[int, int, int]:
    """Sort higher production scores first, then pedagogical level and rank."""
    candidate = production_candidate(entry)
    level_index, rank = learner_order(entry)
    return -candidate.score, level_index, rank
