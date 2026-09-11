"""Download the official raw lexical inputs into ``data/raw`` safely.

The command is a preview unless ``--download`` is supplied. Existing valid
files are reused; ``--force`` is required to replace them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "swedish-anki-deck-source-fetcher/1.0"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    filename: str
    url: str
    kind: str
    minimum_size: int
    license: str


SOURCES = {
    "kelly": SourceSpec(
        key="kelly",
        filename="Swedish-Kelly_M3_CEFR.xls",
        url="https://svn.spraakbanken.gu.se/sb-arkiv/pub/lexikon/kelly/Swedish-Kelly_M3_CEFR.xls",
        kind="xls",
        minimum_size=100_000,
        license="See the official Språkbanken Kelly resource page and workbook metadata.",
    ),
    "folkets": SourceSpec(
        key="folkets",
        filename="folkets_sv_en_public.xml",
        url="https://folkets-lexikon.csc.kth.se/folkets/folkets_sv_en_public.xml",
        kind="xml",
        minimum_size=1_000_000,
        license="CC BY-SA 2.5 (declared in the XML root metadata)",
    ),
    "svalex": SourceSpec(
        key="svalex",
        filename="SVALex_v2.tsv",
        url="https://cental.uclouvain.be/cefrlex/static/resources/sv/SVALex_v2.tsv",
        kind="cefrlex-tsv",
        minimum_size=3_000_000,
        license="CC BY-NC-SA 4.0",
    ),
    "swellex": SourceSpec(
        key="swellex",
        filename="SweLLex_v2.tsv",
        url="https://cental.uclouvain.be/cefrlex/static/resources/sv/SweLLex_v2.tsv",
        kind="cefrlex-tsv",
        minimum_size=7_000_000,
        license="CC BY-NC-SA 4.0",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(path: Path, spec: SourceSpec) -> None:
    if not path.is_file() or path.stat().st_size < spec.minimum_size:
        raise ValueError(f"{path} is missing or smaller than expected")
    if spec.kind == "xls":
        with path.open("rb") as source:
            if source.read(8) != bytes.fromhex("d0cf11e0a1b11ae1"):
                raise ValueError(f"{path} is not an OLE2 .xls workbook")
    elif spec.kind == "xml":
        root: ET.Element | None = None
        words = 0
        try:
            for event, element in ET.iterparse(path, events=("start", "end")):
                if root is None and event == "start":
                    root = element
                if event == "end" and element.tag == "word":
                    words += 1
                    element.clear()
        except ET.ParseError as exc:
            raise ValueError(f"{path} is not complete, readable XML") from exc
        if (
            root is None
            or root.tag != "dictionary"
            or root.get("source-language") != "sv"
            or words == 0
        ):
            raise ValueError(f"{path} is not the Swedish-source Folkets dictionary")
    elif spec.kind == "cefrlex-tsv":
        with path.open(encoding="utf-8-sig") as source:
            header = source.readline().rstrip("\n").split("\t")
            required = {"word", "tag", "level_freq@a1", "nb_doc@a1"}
            if not required.issubset(header):
                raise ValueError(f"{path} is not a CEFRLex TSV export")
            if sum(1 for _ in source) < 1_000:
                raise ValueError(f"{path} has too few CEFRLex lexical rows")


def secure_open(request: Request, timeout: int):
    """Use certifi when installed, while retaining normal TLS verification."""
    try:
        import certifi
    except ImportError:
        context = ssl.create_default_context()
    else:
        context = ssl.create_default_context(cafile=certifi.where())
    return urlopen(request, timeout=timeout, context=context)


def download_source(
    spec: SourceSpec,
    raw_dir: Path,
    *,
    force: bool = False,
    opener: Callable[..., object] = secure_open,
) -> dict[str, object]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / spec.filename
    if destination.exists() and not force:
        validate_source(destination, spec)
        status = "reused"
    else:
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        request = Request(spec.url, headers={"User-Agent": USER_AGENT})
        try:
            with (
                opener(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):  # type: ignore[attr-defined]
                while chunk := response.read(1024 * 1024):  # type: ignore[attr-defined]
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            validate_source(temporary, spec)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        status = "downloaded"
    return {
        "key": spec.key,
        "filename": spec.filename,
        "url": spec.url,
        "license": spec.license,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": status,
    }


def write_manifest(raw_dir: Path, records: list[dict[str, object]]) -> Path:
    manifest = raw_dir / "sources.json"
    existing: dict[str, dict[str, object]] = {}
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        for record in previous.get("sources", []):
            if isinstance(record, dict) and isinstance(record.get("key"), str):
                existing[str(record["key"])] = record
    existing.update({str(record["key"]): record for record in records})
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [existing[key] for key in sorted(existing)],
    }
    temporary = manifest.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES),
        help="Download one source; repeatable. Default: both.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Perform downloads and write the source manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing source files (requires --download).",
    )
    args = parser.parse_args()
    if args.force and not args.download:
        parser.error("--force requires --download")
    selected = args.source or list(SOURCES)
    if not args.download:
        for key in selected:
            spec = SOURCES[key]
            destination = args.raw_dir / spec.filename
            print(
                f"{key}: {spec.url}\n  -> {destination} ({'present' if destination.exists() else 'missing'})"
            )
        print(
            "Preview only. Re-run with --download to fetch/reuse files and write sources.json."
        )
        return

    records = []
    for key in selected:
        record = download_source(SOURCES[key], args.raw_dir, force=args.force)
        records.append(record)
        print(
            f"{key}: {record['status']} {record['bytes']:,} bytes, sha256 {record['sha256']}"
        )
    manifest = write_manifest(args.raw_dir, records)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
