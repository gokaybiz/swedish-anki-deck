"""Create keyless edge-tts MP3s for reviewed Swedish sense-card examples.

The program never edits JSON or contacts Anki. Use ``--dry-run`` to inspect the
plan and ``--synthesize`` for the explicit network/write step. Audio filenames
and the manifest are content-addressed so changed text or voice settings cannot
silently reuse stale speech.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from functions.card_quality import reviewed_card_plan
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
DEFAULT_SOURCE = ROOT / "data" / "json"
DEFAULT_OUTPUT = ROOT / "build" / "audio"
DEFAULT_REQUESTS_PER_MINUTE = 20


def collect_jobs(
    source: Path,
    output: Path,
    *,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    engine_version: str | None = None,
) -> list[dict[str, object]]:
    """Read reviewed sense examples in parallel and assign stable artifacts."""
    version = engine_version or edge_tts_version()

    paths = sorted(source.glob("*.json"))

    def load_one(path: Path) -> tuple[Path, dict[str, object]]:
        return path, json.loads(path.read_text(encoding="utf-8"))

    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded = list(pool.map(load_one, paths))
    plans: dict[Path, dict[str, object]] = {}
    unreviewed: list[Path] = []
    for path, entry in loaded:
        plan = reviewed_card_plan(entry)
        if plan is None:
            unreviewed.append(path)
        else:
            plans[path] = plan
    if unreviewed:
        preview = ", ".join(path.name for path in unreviewed[:5])
        raise ValueError(
            f"{len(unreviewed)} entries require holistic card review before audio "
            f"generation ({preview}{'…' if len(unreviewed) > 5 else ''})"
        )

    def read_one(item: tuple[Path, dict[str, object]]) -> list[dict[str, object]]:
        path, _entry = item
        plan = plans[path]
        jobs: list[dict[str, object]] = []
        for card in plan["cards"]:
            if not isinstance(card, dict):
                raise TypeError(f"{path}: card plan contains a non-object card")
            example = str(card.get("sentence") or "").strip()
            if not example or not any(character.isalnum() for character in example):
                continue
            indices = "-".join(str(value) for value in card.get("source_indices") or [])
            source_stem = f"{path.stem}__sense_{indices}"
            fingerprint = spoken_fingerprint(
                example,
                voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
                engine_version=version,
            )
            filename = audio_filename(source_stem, fingerprint)
            jobs.append(
                {
                    "source": path.name,
                    "source_stem": source_stem,
                    "definition_indices": card.get("source_indices") or [],
                    "text": example,
                    "characters": len(example),
                    "voice": voice,
                    "rate": rate,
                    "volume": volume,
                    "pitch": pitch,
                    "engine": "edge-tts",
                    "engine_version": version,
                    "fingerprint": fingerprint,
                    "file": filename,
                    "path": output / filename,
                }
            )
        return jobs

    with ThreadPoolExecutor(max_workers=16) as pool:
        grouped = pool.map(read_one, loaded)
        return [job for jobs in grouped for job in jobs]


def is_cached(
    job: dict[str, object], manifest: dict[str, object], output: Path
) -> bool:
    records = manifest.get("entries", {})
    if not isinstance(records, dict):
        return False
    record = records.get(str(job["source_stem"]))
    if not isinstance(record, dict) or record.get("fingerprint") != job["fingerprint"]:
        return False
    path = output / str(record.get("file", ""))
    if not path.is_file() or path.stat().st_size == 0:
        return False
    expected_digest = record.get("sha256")
    return isinstance(expected_digest, str) and file_sha256(path) == expected_digest


async def generate(
    pending: list[dict[str, object]],
    output: Path,
    manifest: dict[str, object],
    *,
    retries: int,
    requests_per_minute: int,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    interval, next_request = 60 / requests_per_minute, time.monotonic()
    entries = manifest.setdefault("entries", {})
    assert isinstance(entries, dict)
    for index, job in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {job['source']}", flush=True)
        try:
            wait = next_request - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            destination = Path(job["path"])
            await synthesize_to_file(
                str(job["text"]),
                destination,
                voice=str(job["voice"]),
                rate=str(job["rate"]),
                volume=str(job["volume"]),
                pitch=str(job["pitch"]),
                retries=retries,
            )
            next_request = max(next_request + interval, time.monotonic())
            record = {
                key: value
                for key, value in job.items()
                if key not in {"path", "characters"}
            }
            record.update(
                bytes=destination.stat().st_size,
                sha256=file_sha256(destination),
                generated_at=datetime.now(timezone.utc).isoformat(),
                role="sentence_tts",
            )
            entries[str(job["source_stem"])] = record
            save_manifest(output, manifest)
        except RuntimeError as exc:
            failures.append({"source": str(job["source"]), "error": str(exc)})
            print(f"  failed: {exc}", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Entry JSON folder (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"MP3/manifest folder (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count work without network calls or writes.",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Use edge-tts and create missing MP3 files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate current-fingerprint MP3 files.",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--volume", default="+0%")
    parser.add_argument("--pitch", default="+0Hz")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--requests-per-minute", type=int, default=DEFAULT_REQUESTS_PER_MINUTE
    )
    args = parser.parse_args()
    if args.dry_run == args.synthesize:
        parser.error("Choose exactly one: --dry-run or --synthesize")
    if not args.source.is_dir():
        parser.error(f"Source folder does not exist: {args.source}")
    if args.requests_per_minute < 1 or args.retries < 0:
        parser.error(
            "--requests-per-minute must be positive and --retries non-negative"
        )

    print("Reading JSON entries…", flush=True)
    try:
        jobs = collect_jobs(
            args.source,
            args.output,
            voice=args.voice,
            rate=args.rate,
            volume=args.volume,
            pitch=args.pitch,
        )
    except ValueError as exc:
        parser.error(str(exc))
    manifest = load_manifest(args.output)
    pending = (
        jobs
        if args.overwrite
        else [job for job in jobs if not is_cached(job, manifest, args.output)]
    )
    characters = sum(int(job["characters"]) for job in pending)
    print(f"Found {len(jobs)} reviewed sense-card examples.")
    print(f"Pending audio files: {len(pending)} ({characters:,} characters).")
    print(
        f"Engine: edge-tts {edge_tts_version()}; voice: {args.voice}; output: {args.output}"
    )
    if args.dry_run:
        print(
            f"Minimum pacing time at {args.requests_per_minute}/minute: {len(pending) * 60 / args.requests_per_minute / 3600:.1f} hours."
        )
        return

    args.output.mkdir(parents=True, exist_ok=True)
    failures = asyncio.run(
        generate(
            pending,
            args.output,
            manifest,
            retries=args.retries,
            requests_per_minute=args.requests_per_minute,
        )
    )
    if failures:
        report = args.output / "failures.jsonl"
        report.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
            encoding="utf-8",
        )
        print(f"Finished with {len(failures)} failures; see {report}.")
    else:
        print(f"Finished: {len(pending)} MP3 files are current in {args.output}.")


if __name__ == "__main__":
    main()
