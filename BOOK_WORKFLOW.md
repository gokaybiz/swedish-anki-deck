# Book-based Swedish Anki deck

This is the primary workflow when the vocabulary source is `words.csv`, rather than the Kelly list. The legacy Kelly/Folkets tools remain available but are **not required** here. The book-based workflow creates a separate Anki note type and deck so it will not overwrite older Kelly notes.

## Local inputs

- `words.csv`: exact extracted word forms, exact-form frequencies, PDF page references, estimated printed pages, surface variants, first-occurrence IDs, and sentences from the book. The complete source file stays outside Git.
- `prompts/wordnote_generation_prompt.md`: the book-context lexicography prompt for Swedish forms, contextual meanings, English/Turkish translations, inflections, usage notes, lexical ambiguities and source provenance. The program appends book-specific spoiler protection when preparing batches.

`words.csv` has **9,075 rows**, in first-occurrence order. It contains **66,141 sentence references**. It does not provide a full occurrence-ID map, so an individual form's frequency must never be described as its lemma/sense frequency.

## Run

From the repository root:

```bash
# No Kelly files, Folkets download, pandas, or API key needed to import.
python3 scripts/book_pipeline.py import --csv /path/to/words.csv

# Start with a small inspectable pilot: 10 words, up to 12 contexts per word.
python3 scripts/book_pipeline.py prepare --limit 10

# Inspect review/book/batches/*.prompt.txt before invoking a model.
# Uses authenticated Codex CLI; run 'codex login' separately first.
python3 scripts/book_pipeline.py generate --limit 2

# Review response JSON in review/book/batches/.
# Dry run validates IDs, values, and source integrity.
python3 scripts/book_pipeline.py apply
python3 scripts/book_pipeline.py apply --apply

# With Anki open and AnkiConnect installed:
python3 scripts/book_pipeline.py anki             # read-only preview
python3 scripts/book_pipeline.py anki --apply     # create/update notes
```

After the pilot, omit `--limit` in `prepare` and `generate` to process the remaining records. Preparation uses small batches (five words by default) and a distributed sample of up to 12 source sentences for each word, so thousands of examples for a common function word are not passed to the model at once. When context sampling is incomplete, the prompt explicitly forbids pretending to have exhaustive sense coverage. To increase the limit for a difficult word, use `--max-contexts 30`. Generated batch names derive from their first and last source IDs, avoiding collisions when resuming.

The `generate` command runs `codex exec` with the configured `--model` and submits the selected book passages to the model. This **transmits the selected text to your model provider**. If you do not want that, stop at `prepare` and process response JSON using a local service that follows the prompt instead.

`apply` is read-only without `--apply`, and `anki` is read-only without `--apply`. Import skips existing record JSON, protecting prior note work. `import --overwrite` is destructive, so use it only when intentionally starting again.

## How the note generation works

For each original word form, the pipeline preserves the exact source frequency, pages, first-occurrence ID, surface variants, and book sentences in local JSON. It gives representative contexts to the supplied generation prompt. The model returns a disposition plus zero or more **sense-specific notes**, and each note stores its own lemma, POS, article, English and Turkish meanings, one spoiler-safe Swedish example, both example translations, inflections, register, and sidenotes.

The exported Anki card has the Swedish lemma and teaching sentence on the front. The back adds the English and Turkish explanations, inflections, usage notes, book PDF pages and **exact-form** frequencies. Raw book sentences, source contexts, and predicted plot details are never automatically inserted into Anki cards. Exact quotation of a source sentence and copied runs of eight words are blocked during `apply`; this does **not** guarantee that all spoilers are caught. Review generated output before `anki --apply`.

The exporter collapses records with identical stable `note_key` only when their lemma, POS and English sense agree; otherwise it refuses the conflicting key. Different meanings (e.g. `tänder` as teeth vs lights) remain separate notes. Unknown, uncertain, nonlexical and unreviewed items remain in local data and do not silently become cards. A generated record with `quality_status: needs_review` is withheld until reviewed or corrected.

## Data and security

Do not commit `words.csv`, `data/book/`, `review/book/`, generated Anki packages, or book excerpts. The public repository intentionally contains the processing code and note prompt, **not copyrighted book text**. The model-generated notes still require human checking for Swedish accuracy, Turkish translation accuracy, OCR confusion, homographs, and spoiler leakage. This is a separate book pipeline; legacy Kelly CEFR enrichment, chunk production and TTS scripts have not been repurposed to imply corpus-verified book frequencies or senses.