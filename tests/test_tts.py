from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from functions.card_quality import CARD_PLAN_VERSION, card_source_digest
from functions.tts import (
    ascii_stem,
    audio_filename,
    spoken_fingerprint,
    synthesize_to_file,
)
from json_to_audio import collect_jobs


class FakeCommunicator:
    attempts = 0

    def __init__(self, text: str, voice: str, **settings: str):
        self.text = text
        self.voice = voice
        self.settings = settings

    async def save(self, path: str) -> None:
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise OSError("temporary failure")
        Path(path).write_bytes(b"ID3-test-audio")


class TtsTests(unittest.TestCase):
    def test_fingerprint_tracks_spoken_content_and_settings(self) -> None:
        first = spoken_fingerprint("Hej", "sv-SE-SofieNeural", engine_version="test")
        same = spoken_fingerprint("Hej", "sv-SE-SofieNeural", engine_version="test")
        changed = spoken_fingerprint("Hej!", "sv-SE-SofieNeural", engine_version="test")
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)
        self.assertEqual(ascii_stem("en_ätt_noun"), "en_att_noun")
        self.assertRegex(
            audio_filename("en_ätt_noun", first), r"^en_att_noun__[0-9a-f]{16}\.mp3$"
        )

    def test_synthesis_retries_and_replaces_atomically(self) -> None:
        FakeCommunicator.attempts = 0
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "clip.mp3"
            with patch("functions.tts.asyncio.sleep", new=AsyncMock()) as sleep:
                asyncio.run(
                    synthesize_to_file(
                        "Hej",
                        destination,
                        retries=1,
                        communicator_factory=FakeCommunicator,
                    )
                )
            self.assertEqual(destination.read_bytes(), b"ID3-test-audio")
            self.assertFalse(destination.with_suffix(".mp3.part").exists())
            sleep.assert_awaited_once()

    def test_collect_jobs_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json"
            output = root / "audio"
            source.mkdir()
            entry = {
                "word": "ätt",
                "tags": ["noun"],
                "definitions": [
                    {
                        "definition": "family line",
                        "example": "Ätten samlades.",
                        "example_translation": "The family assembled.",
                        "note": None,
                    }
                ],
                "inflections": {"definite": "ätten"},
            }
            entry["card_plan"] = {
                "version": CARD_PLAN_VERSION,
                "source_digest": card_source_digest(entry),
                "reviewed_by": "test",
                "cards": [
                    {
                        "source_indices": [0],
                        "context_meaning": "family line",
                        "other_meanings": [],
                        "sentence": "Ätten samlades.",
                        "sentence_meaning": "The family assembled.",
                        "target_form": "Ätten",
                        "note": None,
                        "production": None,
                    }
                ],
            }
            (source / "en_ätt_noun.json").write_text(
                json.dumps(entry, ensure_ascii=False), encoding="utf-8"
            )
            jobs = collect_jobs(source, output, engine_version="test")
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["source_stem"], "en_ätt_noun__sense_0")
            self.assertRegex(
                str(jobs[0]["file"]),
                r"^en_att_noun__sense_0__[0-9a-f]{16}\.mp3$",
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
