"""Propose and apply conservative, zero-LLM Swedish learner levels.

Kelly's A1-C2 values are retained as source metadata. This command derives a
separate ``learner_level`` from POS-sensitive SVALex/SweLLex corpus evidence and
a bounded rule for the transparent Swedish numeral family. Unsupported words
remain unassigned rather than receiving a guessed level.

``propose`` writes a review JSONL file and never changes source entries.
``apply --apply`` is the only mutating operation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from functions.cefr import (
    CEFR_LEVELS,
    LevelEstimate,
    infer_learner_level,
    load_cefrlex,
    profile_for,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "json"
DEFAULT_SVALEX = ROOT / "data" / "raw" / "SVALex_v2.tsv"
DEFAULT_SWELLEX = ROOT / "data" / "raw" / "SweLLex_v2.tsv"
DEFAULT_PROPOSALS = ROOT / "review" / "cefr_level_proposals.jsonl"
CONFIDENCE = {"medium", "high"}
KNOWN_POS = {
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
}


def load_entry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: entry must be a JSON object")
    return value


def part_of_speech(entry: dict[str, Any]) -> str:
    return next(
        (str(tag) for tag in entry.get("tags", []) if str(tag) in KNOWN_POS), ""
    )


def source_band(entry: dict[str, Any]) -> str:
    return str(
        entry.get("sources", {}).get("frequency", {}).get("cefr_band") or ""
    ).strip()


def learner_payload(
    estimate: LevelEstimate,
    receptive: dict[str, dict[str, float | int]],
    productive: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    return {
        "cefr": estimate.cefr,
        "confidence": estimate.confidence,
        "rationale": estimate.rationale,
        "method": estimate.method,
        "evidence": {"svalex": receptive, "swellex": productive},
    }


def proposal_for(
    path: Path,
    entry: dict[str, Any],
    *,
    svalex_rows: dict[str, list[dict[str, str]]],
    swellex_rows: dict[str, list[dict[str, str]]],
) -> dict[str, object] | None:
    if entry.get("learner_level"):
        return None
    word = str(entry.get("word") or "").strip()
    pos = part_of_speech(entry)
    band = source_band(entry)
    if not word or not pos or band not in CEFR_LEVELS:
        return None
    receptive = profile_for(svalex_rows, word, pos)
    productive = profile_for(swellex_rows, word, pos)
    estimate = infer_learner_level(
        word=word,
        part_of_speech=pos,
        kelly_band=band,
        receptive=receptive,
        productive=productive,
    )
    if estimate is None:
        return None
    return {
        "id": f"{path.name}:learner-level",
        "kind": "learner-level",
        "source": path.name,
        "word": word,
        "part_of_speech": pos,
        "kelly_cefr_band": band,
        "learner_level": learner_payload(estimate, receptive, productive),
    }


def propose(args: argparse.Namespace) -> int:
    if not args.source.is_dir():
        print(f"Source folder does not exist: {args.source}", file=sys.stderr)
        return 2
    if args.proposals.exists() and not args.replace and not args.dry_run:
        print(
            f"Proposal file already exists: {args.proposals}. Review it or use --replace.",
            file=sys.stderr,
        )
        return 2
    try:
        svalex_rows = load_cefrlex(args.svalex)
        swellex_rows = load_cefrlex(args.swellex)
        entries = [
            (path, load_entry(path)) for path in sorted(args.source.glob("*.json"))
        ]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    entries.sort(
        key=lambda item: (
            int(item[1].get("frequency_rank") or sys.maxsize),
            item[0].name,
        )
    )
    if args.limit is not None:
        entries = entries[: args.limit]

    proposals: list[dict[str, object]] = []
    already_assigned = unsupported = 0
    for path, entry in entries:
        if entry.get("learner_level"):
            already_assigned += 1
            continue
        proposal = proposal_for(
            path,
            entry,
            svalex_rows=svalex_rows,
            swellex_rows=swellex_rows,
        )
        if proposal is None:
            unsupported += 1
        else:
            proposals.append(proposal)

    counts = Counter(
        str(proposal["learner_level"]["cefr"])  # type: ignore[index]
        for proposal in proposals
    )
    print(
        f"Proposable: {len(proposals)}; unsupported/unassigned: {unsupported}; "
        f"already assigned: {already_assigned}."
    )
    print("Levels: " + ", ".join(f"{level}={counts[level]}" for level in CEFR_LEVELS))
    if args.dry_run:
        print("Dry run only. Source JSON and proposal files were not changed.")
        return 0

    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.proposals.with_suffix(args.proposals.suffix + ".part")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in proposals),
        encoding="utf-8",
    )
    temporary.replace(args.proposals)
    print(f"Wrote {len(proposals)} reviewable proposals to {args.proposals}.")
    print("Source JSON is unchanged. Review before running apply --apply.")
    return 0


def validated_learner_level(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("learner_level must be an object")
    required = {"cefr", "confidence", "rationale", "method", "evidence"}
    if set(value) != required:
        raise ValueError(f"learner_level must contain exactly {sorted(required)}")
    cefr = value["cefr"]
    confidence = value["confidence"]
    rationale = value["rationale"]
    method = value["method"]
    evidence = value["evidence"]
    if cefr not in CEFR_LEVELS:
        raise ValueError(f"unsupported learner CEFR level: {cefr!r}")
    if confidence not in CONFIDENCE:
        raise ValueError(f"unsupported learner-level confidence: {confidence!r}")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 180:
        raise ValueError("learner-level rationale must contain 1-180 characters")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("learner-level method must be non-empty")
    if not isinstance(evidence, dict) or set(evidence) != {"svalex", "swellex"}:
        raise ValueError("learner-level evidence must contain svalex and swellex")
    return {
        "cefr": cefr,
        "confidence": confidence,
        "rationale": rationale.strip(),
        "method": method.strip(),
        "evidence": evidence,
    }


def apply_record(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    if record.get("kind") != "learner-level":
        raise ValueError("unsupported proposal kind")
    if str(entry.get("word") or "").strip() != str(record.get("word") or "").strip():
        raise ValueError("source word changed after proposal")
    if part_of_speech(entry) != record.get("part_of_speech"):
        raise ValueError("source part of speech changed after proposal")
    if source_band(entry) != record.get("kelly_cefr_band"):
        raise ValueError("Kelly source band changed after proposal")
    proposed = validated_learner_level(record.get("learner_level"))
    existing = entry.get("learner_level")
    if existing:
        if existing == proposed:
            return False
        raise ValueError("source already has a different learner level")
    entry["learner_level"] = proposed
    return True


def write_entry(path: Path, entry: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def apply(args: argparse.Namespace) -> int:
    if not args.apply:
        print("Refusing to write. Review proposals, then add --apply.", file=sys.stderr)
        return 2
    if not args.source.is_dir() or not args.proposals.is_file():
        print("Source folder or proposal file is missing.", file=sys.stderr)
        return 2
    source_root = args.source.resolve()
    applied = unchanged = errors = 0
    for line_number, line in enumerate(
        args.proposals.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
            source_name = record.get("source")
            if not isinstance(source_name, str):
                raise TypeError("proposal source must be a filename")
            path = (source_root / source_name).resolve()
            if path.parent != source_root:
                raise ValueError("proposal source must remain inside --source")
            entry = load_entry(path)
            if apply_record(entry, record):
                write_entry(path, entry)
                applied += 1
            else:
                unchanged += 1
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Line {line_number}: skipped: {exc}", file=sys.stderr)
            errors += 1
    print(
        f"Applied {applied}; already identical {unchanged}; errors {errors}. "
        "No Anki operations were performed."
    )
    return 1 if errors else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    common.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)

    propose_parser = commands.add_parser("propose", parents=[common])
    propose_parser.add_argument("--svalex", type=Path, default=DEFAULT_SVALEX)
    propose_parser.add_argument("--swellex", type=Path, default=DEFAULT_SWELLEX)
    propose_parser.add_argument("--limit", type=int)
    propose_parser.add_argument("--replace", action="store_true")
    propose_parser.add_argument("--dry-run", action="store_true")
    propose_parser.set_defaults(handler=propose)

    apply_parser = commands.add_parser("apply", parents=[common])
    apply_parser.add_argument("--apply", action="store_true")
    apply_parser.set_defaults(handler=apply)
    return result


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser().error("--limit must be at least 1")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
