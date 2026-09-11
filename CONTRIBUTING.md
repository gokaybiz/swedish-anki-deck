# Contributing

Corrections and focused improvements are welcome, especially for Swedish
examples, translations, meanings, inflections, frequency ranks, and reproducible
source/audio handling.

Please include the headword, proposed change, and a reliable source. Preserve
source provenance: Folkets-derived lexical content is governed by the CC BY-SA
2.5 declaration in its XML; SVALex/SweLLex evidence is CC BY-NC-SA 4.0; Kelly
reuse/citation terms should be checked against the official Språkbanken
resource. Preserve attribution and applicable ShareAlike handling in releases.

Before opening a pull request, run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
ruff check scripts tests
ruff format --check scripts tests
```

Formatting follows Ruff defaults; do not introduce unrelated formatting-only
churn in untouched modules.

Tests must not call live dictionary/TTS services, invoke Codex, or make AnkiConnect requests.
Do not run `json_to_anki.py --apply` as validation. Never commit API keys,
downloaded raw corpora without explicit licence review, generated media, Anki
collections, or personal study data.
