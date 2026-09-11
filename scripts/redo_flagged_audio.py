"""Regenerate two explicitly corrected example clips with keyless edge-tts.

This repair utility only writes content-addressed files and their manifest under
``build/audio``. It has no AnkiConnect/import capability; the user remains in
control of any later deck update.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from functions.tts import (
    DEFAULT_VOICE,
    audio_filename,
    edge_tts_version,
    file_sha256,
    load_manifest,
    save_manifest,
    spoken_fingerprint,
    synthesize_to_file,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build" / "audio"
TARGETS = (
    {
        "rank": 437,
        "word": "en ätt",
        "stem": "en_ätt_noun",
        "text": "Var honom trofast och hans ätt. Gör kronan på hans hjässa lätt. Och all din tro till honom sätt.",
    },
    {
        "rank": 3904,
        "word": "still",
        "stem": "still_adjective",
        "text": "Vattnet var stilla på morgonen.",
    },
)


async def regenerate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.output)
    entries = manifest.setdefault("entries", {})
    assert isinstance(entries, dict)
    version = edge_tts_version()
    for target in TARGETS:
        fingerprint = spoken_fingerprint(
            target["text"],
            args.voice,
            rate=args.rate,
            volume=args.volume,
            pitch=args.pitch,
            engine_version=version,
        )
        filename = audio_filename(target["stem"], fingerprint)
        destination = args.output / filename
        existing = entries.get(target["stem"])
        current = (
            isinstance(existing, dict)
            and existing.get("fingerprint") == fingerprint
            and destination.is_file()
            and existing.get("sha256") == file_sha256(destination)
        )
        if current and not args.overwrite:
            print(f"#{target['rank']} already current: {destination.name}")
            continue
        print(f"Synthesizing #{target['rank']} ({target['word']})…", flush=True)
        await synthesize_to_file(
            target["text"],
            destination,
            voice=args.voice,
            rate=args.rate,
            volume=args.volume,
            pitch=args.pitch,
            retries=args.retries,
        )
        entries[target["stem"]] = {
            "source": f"manual-repair-rank-{target['rank']}",
            "source_stem": target["stem"],
            "definition_index": 0,
            "text": target["text"],
            "voice": args.voice,
            "rate": args.rate,
            "volume": args.volume,
            "pitch": args.pitch,
            "engine": "edge-tts",
            "engine_version": version,
            "fingerprint": fingerprint,
            "file": filename,
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "role": "example_tts",
        }
        save_manifest(args.output, manifest)
        print(f"  wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Generate/replace the two corrected edge-tts clips.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--volume", default="+0%")
    parser.add_argument("--pitch", default="+0Hz")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.overwrite and not args.synthesize:
        parser.error("--overwrite requires --synthesize")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    for target in TARGETS:
        print(f"#{target['rank']}: {target['word']}\n  {target['text']}")
    if not args.synthesize:
        print(
            "Preview only. No files, network calls, or Anki operations were performed."
        )
        return
    asyncio.run(regenerate(args))


if __name__ == "__main__":
    main()
