"""CEFRLex corpus evidence for reviewed Swedish learner-level estimates."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# Korp tags used by SVALex and SweLLex. Prefix matching includes gendered noun
# tags and multi-word variants while keeping homographs in different classes
# separate whenever possible.
_TAG_PREFIXES_BY_POS = {
    "noun": ("NN",),
    "proper noun": ("PM",),
    "verb": ("VB",),
    "adjective": ("JJ", "PC"),
    "adverb": ("AB",),
    "pronoun": ("PN", "HP", "HS", "PS"),
    "determiner": ("DT", "HD"),
    "numeral": ("RG", "RO"),
    "preposition": ("PP",),
    "conjunction": ("KN", "SN"),
    "interjection": ("IN", "IE"),
    "particle": ("PL",),
}

CorpusRows = dict[str, list[dict[str, str]]]
LevelProfile = dict[str, dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class LevelEstimate:
    cefr: str
    confidence: str
    rationale: str
    method: str


def load_cefrlex(path: Path) -> CorpusRows:
    """Load the compact CEFRLex columns needed for per-entry evidence."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CEFRLex source: {path}. Run scripts/download_sources.py "
            "--source svalex --source swellex --download."
        )
    result: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"word", "tag", "total_freq@total"}
        required.update(f"level_freq@{level.lower()}" for level in CEFR_LEVELS[:-1])
        required.update(f"nb_doc@{level.lower()}" for level in CEFR_LEVELS[:-1])
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"{path}: missing CEFRLex columns: {missing}")
        for row in reader:
            word = str(row.get("word") or "").strip().casefold()
            if word:
                result[word].append(row)
    if not result:
        raise ValueError(f"{path}: no CEFRLex lexical rows")
    return dict(result)


def profile_for(rows: CorpusRows, word: str, part_of_speech: str) -> LevelProfile:
    """Return POS-sensitive frequency/document evidence at each attested level."""
    candidates = rows.get(word.strip().casefold(), [])
    prefixes = _TAG_PREFIXES_BY_POS.get(part_of_speech, ())
    matched = [
        row
        for row in candidates
        if not prefixes or str(row.get("tag") or "").upper().startswith(prefixes)
    ]
    if not matched:
        return {}

    profile: LevelProfile = {}
    for level in CEFR_LEVELS:
        suffix = level.lower()
        frequency_key = f"level_freq@{suffix}"
        documents_key = f"nb_doc@{suffix}"
        if not any(frequency_key in row or documents_key in row for row in matched):
            continue
        frequency = sum(_number(row.get(frequency_key)) for row in matched)
        documents = max((_number(row.get(documents_key)) for row in matched), default=0)
        if frequency or documents:
            profile[level] = {
                "frequency": round(frequency, 4),
                "documents": int(documents),
            }
    return profile


def infer_learner_level(
    *,
    word: str,
    part_of_speech: str,
    kelly_band: str,
    receptive: LevelProfile,
    productive: LevelProfile,
) -> LevelEstimate | None:
    """Conservatively infer a recognition level without an LLM.

    SVALex requires occurrence in at least two coursebook documents, matching
    its distinction between shared/core and peripheral vocabulary. SweLLex
    requires three learner documents because isolated productive uses are a
    weaker recognition-level signal. Unsupported items remain unassigned.
    """
    receptive_level = _earliest_supported(receptive, minimum_documents=2)
    productive_level = _earliest_supported(productive, minimum_documents=3)

    # Numerals and ordinals are a closed, highly compositional family. Kelly's
    # corpus-rank bands put rare but transparent forms such as femtionde at C2.
    # Keep manually/basic A1 forms at A1; cap the rest at A2 unless stronger
    # corpus evidence places them at A1.
    if part_of_speech == "numeral":
        if receptive_level == "A1" or productive_level == "A1":
            return LevelEstimate(
                "A1",
                "high" if receptive_level == productive_level == "A1" else "medium",
                "Basic numeral attested at A1; exact-form frequency does not imply advanced difficulty.",
                "numeral-family-v1",
            )
        level = "A1" if kelly_band == "A1" else "A2"
        return LevelEstimate(
            level,
            "medium",
            "Transparent Swedish numeral or ordinal; exact-form rarity is capped at A2.",
            "numeral-family-v1",
        )

    supported = [level for level in (receptive_level, productive_level) if level]
    if not supported:
        return None
    level = min(supported, key=CEFR_LEVELS.index)
    if receptive_level and productive_level:
        distance = abs(
            CEFR_LEVELS.index(receptive_level) - CEFR_LEVELS.index(productive_level)
        )
        confidence = "high" if distance <= 1 else "medium"
        rationale = (
            f"SVALex supports {receptive_level}; SweLLex supports "
            f"{productive_level}. Earliest robust evidence sets {level}."
        )
        method = "svalex-swellex-v1"
    elif receptive_level:
        confidence = "medium"
        rationale = (
            f"Attested in at least two SVALex coursebook documents from "
            f"{receptive_level}; productive evidence is insufficient."
        )
        method = "svalex-v1"
    else:
        confidence = "medium"
        rationale = (
            f"Attested in at least three SweLLex learner documents from "
            f"{productive_level}; receptive evidence is insufficient."
        )
        method = "swellex-v1"
    return LevelEstimate(level, confidence, rationale, method)


def _earliest_supported(profile: LevelProfile, *, minimum_documents: int) -> str | None:
    return next(
        (
            level
            for level in CEFR_LEVELS
            if int(profile.get(level, {}).get("documents", 0)) >= minimum_documents
        ),
        None,
    )


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
