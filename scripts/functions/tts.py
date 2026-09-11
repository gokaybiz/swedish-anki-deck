"""Shared, keyless edge-tts helpers with content-addressed audio metadata."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_VOICE = "sv-SE-SofieNeural"
MANIFEST_NAME = "audio_manifest.json"


def edge_tts_version() -> str:
    try:
        return metadata.version("edge-tts")
    except metadata.PackageNotFoundError:
        return "not-installed"


def spoken_fingerprint(
    text: str,
    voice: str,
    *,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    engine_version: str | None = None,
) -> str:
    payload = {
        "engine": "edge-tts",
        "engine_version": engine_version or edge_tts_version(),
        "text": unicodedata.normalize("NFC", text).strip(),
        "voice": voice,
        "rate": rate,
        "volume": volume,
        "pitch": pitch,
        "format": "audio-24khz-48kbitrate-mono-mp3",
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ascii_stem(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-").lower()
    return normalized or "entry"


def audio_filename(source_stem: str, fingerprint: str) -> str:
    return f"{ascii_stem(source_stem)}__{fingerprint[:16]}.mp3"


async def synthesize_to_file(
    text: str,
    destination: Path,
    *,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    retries: int = 3,
    communicator_factory: Callable[..., Any] | None = None,
) -> None:
    """Generate one MP3 atomically, retrying transient edge service failures."""
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if communicator_factory is None:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed; install requirements.txt"
            ) from exc
        communicator_factory = edge_tts.Communicate

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries + 1):
        temporary.unlink(missing_ok=True)
        try:
            communicator = communicator_factory(
                unicodedata.normalize("NFC", text).strip(),
                voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
            )
            await communicator.save(str(temporary))
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("edge-tts returned no audio")
            temporary.replace(destination)
            return
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(
                    f"edge-tts failed after {retries + 1} attempt(s): {exc}"
                ) from exc
            await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(folder: Path) -> dict[str, Any]:
    path = folder / MANIFEST_NAME
    if not path.exists():
        return {"version": 1, "engine": "edge-tts", "entries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("entries"), dict):
        raise ValueError(f"Unsupported audio manifest: {path}")
    return data


def save_manifest(folder: Path, manifest: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MANIFEST_NAME
    temporary = path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
