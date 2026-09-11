"""Convert the official Swedish Kelly list and Folkets lexikon into JSON.

Both inputs are local, inspectable raw files.  Run ``download_sources.py`` first
(or pass explicit paths).  No per-word dictionary requests are made and
existing JSON files are never overwritten unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from download_sources import sha256
from functions.folkets import FolketsLexicon

if TYPE_CHECKING:
    import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KELLY = ROOT / "data" / "raw" / "Swedish-Kelly_M3_CEFR.xls"
DEFAULT_FOLKETS = ROOT / "data" / "raw" / "folkets_sv_en_public.xml"
DEFAULT_OUTPUT = ROOT / "data" / "json"
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "sources.json"
PARENTHETICAL = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
EXAMPLE_PREFIX = re.compile(r"^e\.\s*g\.\s*", re.IGNORECASE)

POS_MAP = {
    "adjective": "adjective",
    "adverb": "adverb",
    "aux verb": "verb",
    "conj": "conjunction",
    "det": "determiner",
    "interj": "interjection",
    "noun": "noun",
    "noun-en": "noun",
    "noun-ett": "noun",
    "numeral": "numeral",
    "particle": "particle",
    "prep": "preposition",
    "pronoun": "pronoun",
    "proper name": "proper noun",
    "subj": "conjunction",
    "verb": "verb",
    "particip": "adjective",
    "noun-en/-ett": "noun",
}
CLASS_LABEL_LEAK = re.compile(
    r"^[A-ZÅÄÖ]+(?:" + "|".join(re.escape(label) for label in POS_MAP) + r")$"
)


def clean_headword(value: object) -> str:
    """Normalize Kelly display annotations and recover marked source corruption."""
    raw = str(value or "").strip()
    parenthetical = PARENTHETICAL.fullmatch(raw)
    if parenthetical:
        primary, expansion = (part.strip() for part in parenthetical.groups())
        if CLASS_LABEL_LEAK.fullmatch(primary) and expansion.isalpha():
            return expansion
    return re.sub(r"\s*\([^)]*\)", "", raw).strip()


def kelly_hints(
    source_item: object, examples: object, normalized_headword: str
) -> list[str]:
    """Preserve Kelly annotations/patterns without treating them as sentences."""
    raw_item = str(source_item or "").strip()
    result: list[str] = []
    parenthetical = PARENTHETICAL.fullmatch(raw_item)
    if parenthetical:
        primary, annotation = (part.strip() for part in parenthetical.groups())
        # A normal parenthetical annotates the normalized primary headword. The
        # malformed class-label recovery uses the parenthetical as the headword
        # itself, so it must not become a learner hint.
        if primary == normalized_headword and annotation:
            result.append(f"headword annotation: {annotation}")
    usage = EXAMPLE_PREFIX.sub("", str(examples or "").strip()).strip()
    if usage:
        result.append(f"usage pattern: {usage}")
    return result


def article_for(grammar: object, source_pos: str) -> str:
    explicit = str(grammar or "").strip().lower()
    if explicit:
        return explicit
    return {"noun-en": "en", "noun-ett": "ett"}.get(source_pos, "")


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = (
        frame.columns.str.replace("-\n", "", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )
    frame = frame.fillna("")
    frame["Kelly source item"] = frame["Swedish items for translation"].map(
        lambda value: str(value).strip()
    )
    frame["Swedish items for translation"] = frame["Kelly source item"].map(
        clean_headword
    )
    return frame


def filename(article: str, headword: str, pos: str) -> str:
    safe = re.sub(r'[<>:"/\\\\|?*]+', "_", headword.strip())
    return (
        f"{article}_{safe}_{pos}.json"
        if article in {"en", "ett"}
        else f"{safe}_{pos}.json"
    )


def verified_artifacts(
    manifest_path: Path, files: dict[str, Path]
) -> dict[str, dict[str, object]]:
    if not manifest_path.is_file():
        raise ValueError(f"Source manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("filename")): record
        for record in manifest.get("sources", [])
        if isinstance(record, dict) and record.get("filename")
    }
    result: dict[str, dict[str, object]] = {}
    for role, path in files.items():
        record = records.get(path.name)
        if not record:
            raise ValueError(f"{path.name} is not recorded in {manifest_path}")
        actual_size, actual_digest = path.stat().st_size, sha256(path)
        if record.get("bytes") != actual_size or record.get("sha256") != actual_digest:
            raise ValueError(f"{path} does not match its source manifest digest")
        result[role] = {
            "filename": path.name,
            "url": record.get("url", ""),
            "bytes": actual_size,
            "sha256": actual_digest,
        }
    return result


def local_artifacts(files: dict[str, Path]) -> dict[str, dict[str, object]]:
    return {
        role: {
            "filename": path.name,
            "url": "",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for role, path in files.items()
    }


def build_entry(
    headword: str,
    article: str,
    pos: str,
    rank: int,
    level: str,
    lexicon: FolketsLexicon,
    source_artifacts: dict[str, dict[str, object]] | None = None,
) -> dict[str, Any]:
    matches = lexicon.lookup(headword, pos)
    inflections, conflicts = lexicon.inflections(headword, pos, matches)
    entry: dict[str, Any] = {
        "frequency_rank": rank,
        "article": article,
        "word": headword,
        "definitions": lexicon.definitions(matches),
        "inflections": inflections,
        "tags": [pos],
        "sources": {
            "frequency": {
                "name": "Swedish Kelly list",
                "record_id": rank,
                "cefr_band": level,
            },
            "artifacts": source_artifacts or {},
            "lexicon": {
                "name": lexicon.metadata.get("name", "Folkets lexikon"),
                "version": lexicon.metadata.get("version", ""),
                "last_changed": lexicon.metadata.get("last-changed", ""),
                "license": lexicon.metadata.get("license", ""),
                "records": [
                    {
                        "word": record.headword,
                        "class": record.word_class,
                        "order": record.source_order,
                    }
                    for record in matches
                ],
            },
        },
    }
    lexical_info = lexicon.lexical_info(matches)
    if lexical_info:
        entry["lexical_info"] = lexical_info
    if conflicts:
        entry["source_ambiguities"] = {"inflections": conflicts}
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_KELLY,
        help=f"Kelly .xls workbook (default: {DEFAULT_KELLY})",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DEFAULT_FOLKETS,
        help=f"Folkets XML file (default: {DEFAULT_FOLKETS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Folder for entry JSON files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Source digest manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--allow-unmanifested",
        action="store_true",
        help="Use explicit local inputs without sources.json; their digests are still recorded.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N entries.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing JSON files."
    )
    args = parser.parse_args()
    if not args.input.is_file() or not args.dictionary.is_file():
        parser.error(
            "Raw source missing. Run: python3 scripts/download_sources.py --download"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    raw_files = {"kelly": args.input, "folkets": args.dictionary}
    try:
        artifacts = (
            local_artifacts(raw_files)
            if args.allow_unmanifested
            else verified_artifacts(args.manifest, raw_files)
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    try:
        import pandas as pd
    except ImportError:
        parser.error("pandas/xlrd are required; install requirements.txt")

    print(f"Indexing Folkets lexikon from {args.dictionary}…", flush=True)
    lexicon = FolketsLexicon.from_file(args.dictionary)
    print(
        f"Indexed {len(lexicon.records):,} lexical records (version {lexicon.metadata.get('version', 'unknown')}).",
        flush=True,
    )
    frame = clean_columns(pd.read_excel(args.input, sheet_name="Swedish_M3_CEFR"))
    if args.limit is not None:
        frame = frame.head(args.limit)
    args.output.mkdir(parents=True, exist_ok=True)

    skipped = unmatched = ambiguous = 0
    grouped: dict[tuple[str, str, str], list[tuple[int, Any]]] = {}
    for number, (_, row) in enumerate(frame.iterrows(), start=1):
        headword = str(row["Swedish items for translation"]).strip()
        source_pos = str(row["Word classes"]).strip().lower()
        source_article = str(row["Grammar"]).strip().lower()
        if source_pos not in POS_MAP:
            raise ValueError(
                f"Unsupported Kelly word class {source_pos!r} for {headword!r}"
            )
        if not headword:
            skipped += 1
            continue
        identity = (
            article_for(source_article, source_pos),
            headword,
            POS_MAP[source_pos],
        )
        grouped.setdefault(identity, []).append((number, row))

    for (article, headword, pos), rows in grouped.items():
        first_number, first = rows[0]
        source_article = str(first["Grammar"]).strip().lower()
        # Keep filename identity tied to the explicit Kelly grammar cell while
        # allowing noun-en/noun-ett to fill a missing learner-facing article.
        destination = args.output / filename(source_article, headword, pos)
        if destination.exists() and not args.overwrite:
            skipped += len(rows)
            continue
        print(f"[{first_number}/{len(frame)}] {headword}", flush=True)
        entry = build_entry(
            headword,
            article,
            pos,
            int(first["ID"]),
            str(first["CEFR levels"]),
            lexicon,
            artifacts,
        )
        source_item = str(first["Kelly source item"]).strip()
        if source_item != headword:
            entry["sources"]["frequency"]["source_item"] = source_item

        hints: list[str] = []
        for _number, row in rows:
            hints.extend(
                kelly_hints(
                    row["Kelly source item"],
                    row["Examples"],
                    headword,
                )
            )
        if hints:
            lexical_info = entry.setdefault("lexical_info", {})
            lexical_info["kelly_hints"] = list(dict.fromkeys(hints))

        if len(rows) > 1:
            entry["sources"]["frequency"]["alternate_records"] = [
                {
                    "record_id": int(row["ID"]),
                    "cefr_band": str(row["CEFR levels"]),
                    "source_item": str(row["Kelly source item"]).strip(),
                }
                for _number, row in rows[1:]
            ]
            skipped += len(rows) - 1
        if not entry["definitions"]:
            unmatched += 1
        if entry.get("source_ambiguities"):
            ambiguous += 1
        destination.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Finished. Skipped {skipped}; {unmatched} entries have no Folkets English definition; "
        f"{ambiguous} entries retain source ambiguity metadata."
    )


if __name__ == "__main__":
    main()
