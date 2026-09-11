"""Audit the frequency ranks in a folder of Swedish entry JSON files.

This is read-only. It reports malformed entries, duplicate ranks, duplicate
normalized lexical identities, and gaps. It exits non-zero for malformed
entries or either duplicate class.

Rank gaps are reported but do not fail the audit: the Kelly workbook contains a
few rows that share the same article, headword, and word class, and the
converter intentionally collapses them into one entry. Those collapsed rows
leave matching gaps, so a full-deck audit always has a small, explainable gap
set. Duplicate ranks in the output, by contrast, always indicate a real defect.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Folder containing entry JSON files."
    )
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"Source folder does not exist: {args.source}")

    by_rank: dict[int, list[str]] = defaultdict(list)
    by_identity: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    malformed: list[str] = []
    for path in sorted(args.source.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            rank = entry["frequency_rank"]
            if not isinstance(rank, int) or rank < 1:
                raise ValueError("frequency_rank must be a positive integer")
            by_rank[rank].append(path.name)
            word = str(entry.get("word") or "").strip().casefold()
            tags = tuple(
                sorted(str(tag).strip().casefold() for tag in entry.get("tags", []))
            )
            if word and tags:
                identity = (
                    str(entry.get("article") or "").strip().casefold(),
                    word,
                    tags,
                )
                by_identity[identity].append(path.name)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            malformed.append(f"{path.name}: {exc}")

    ranks = sorted(by_rank)
    duplicates = {rank: names for rank, names in by_rank.items() if len(names) > 1}
    duplicate_identities = {
        identity: names for identity, names in by_identity.items() if len(names) > 1
    }
    gaps = []
    if ranks:
        expected = set(range(ranks[0], ranks[-1] + 1))
        gaps = sorted(expected.difference(ranks))
    print(f"Entries checked: {sum(len(names) for names in by_rank.values())}")
    print(f"Rank range: {ranks[0]}–{ranks[-1]}" if ranks else "Rank range: none")
    print(
        f"Malformed: {len(malformed)}; duplicate ranks: {len(duplicates)}; "
        f"duplicate identities: {len(duplicate_identities)}; gaps: {len(gaps)}"
    )
    for message in malformed[:10]:
        print(f"  malformed: {message}")
    for rank, names in sorted(duplicates.items())[:10]:
        print(f"  duplicate rank {rank}: {', '.join(names)}")
    for (article, word, tags), names in sorted(duplicate_identities.items())[:10]:
        display = " ".join(part for part in (article, word) if part)
        print(f"  duplicate identity {display} ({'/'.join(tags)}): {', '.join(names)}")
    if gaps:
        preview = ", ".join(map(str, gaps[:20]))
        print(f"  first gaps: {preview}{' …' if len(gaps) > 20 else ''}")
        print("  gaps are expected where the Kelly workbook repeats one identity")
    if malformed or duplicates or duplicate_identities:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
