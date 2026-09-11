# Swedish Vocabulary A1–C2 for Anki

Python tools for building a Swedish vocabulary deck ordered by the Kelly List.
The source inventory contains **8,420 Kelly lexical entries**. The final Anki
note count is determined by reviewed sense splitting and merging rather than
equated with the entry count.

The repository keeps source acquisition, lexical conversion, generated speech,
and Anki export as separate inspectable steps. Generated JSON, raw downloads,
MP3 files, Anki collections, and deck exports are not committed.

## Pipeline

```text
official raw sources → local JSON → reviewed lexical enrichment → reviewed card plans → edge-tts audio → Anki preview/export
```

| Script | Purpose |
| --- | --- |
| `download_sources.py` | Downloads Kelly, Folkets, SVALex, and SweLLex atomically into `data/raw/`. |
| `kelly_to_json.py` | Joins Kelly frequency metadata to the local Folkets lexicon. |
| `enrich_cefr_levels_local.py` | Proposes conservative learner levels locally from SVALex/SweLLex evidence; uses no LLM tokens. |
| `enrich_lexicon_codex_subscription.py` | Proposes missing lexical fields and holistic learner-card plans through an authenticated Codex/ChatGPT subscription. |
| `enrich_lexicon_openai_batch.py` | Provides the same lexical and card-review contracts through the metered OpenAI API and Batch API. |
| `json_to_audio.py` | Generates content-addressed TTS for every reviewed sense-card sentence with `edge-tts`; no key required. |
| `json_to_anki.py` | Previews or, only with `--apply`, creates/updates separate recognition and chunk-production note types. |
| `verify_json_frequency.py` | Audits ranks without changing data. |

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The preferred enrichment workflow uses an authenticated `codex` CLI and does
not read `OPENAI_API_KEY`. The functionally equivalent
`enrich_lexicon_openai_batch.py` workflow uses that key. Both use the same
learner-facing quality and usage-note contract. Source downloads and `edge-tts`
need network access but no API credentials.

## Quick start: build your own deck

The two Codex stages are intentionally separate. The first fills missing
lexical data; the second (`--kind cards`, plural) reviews every complete entry
and creates the learner-facing sense plan. Applying lexical proposals does not
apply card proposals that have not been generated yet.

```bash
# 1. Sources and local lexical JSON
python3 scripts/download_sources.py --download
python3 scripts/kelly_to_json.py

# 2. Optional evidence-based CEFR estimates (no LLM)
python3 scripts/enrich_cefr_levels_local.py propose
# inspect review/cefr_level_proposals.jsonl
python3 scripts/enrich_cefr_levels_local.py apply --apply

# 3. Missing definitions, examples, notes, and inflections
python3 scripts/enrich_lexicon_codex_subscription.py propose \
  --workers 3 --max-batches 0
# inspect review/lexicon_codex_subscription_proposals.jsonl
python3 scripts/enrich_lexicon_codex_subscription.py apply --apply

# 4. AI review and generation of one learner plan per lexical entry
python3 scripts/enrich_lexicon_codex_subscription.py propose \
  --kind cards --production-limit 1000 --workers 3 --max-batches 0
# inspect the newly appended cards-v2 rows, then apply again
python3 scripts/enrich_lexicon_codex_subscription.py apply --apply

# 5. Audio, read-only Anki preview, then explicit export
python3 scripts/json_to_audio.py --dry-run
python3 scripts/json_to_audio.py --synthesize
python3 scripts/json_to_anki.py --core data/json --audio-dir build/audio
python3 scripts/json_to_anki.py --core data/json --audio-dir build/audio --apply
```

Both `apply --apply` commands mutate local source JSON but never contact Anki.
Only the final `json_to_anki.py --apply` command imports notes and media. Omit
that flag to keep the exporter read-only.

## Reproducible build

### 1. Acquire raw inputs

Preview the official URLs and destinations:

```bash
python3 scripts/download_sources.py
```

Download them and write `data/raw/sources.json` with URLs, sizes, and SHA-256
digests:

```bash
python3 scripts/download_sources.py --download
```

Existing valid files are reused. Use `--force --download` only when you
intentionally want the current upstream versions. Downloads use normal TLS
verification and atomic `.part` files.

### 2. Build lexical JSON locally

```bash
python3 scripts/kelly_to_json.py
```

Defaults:

- `data/raw/Swedish-Kelly_M3_CEFR.xls`
- `data/raw/folkets_sv_en_public.xml`
- output under `data/json/`

The converter verifies both raw files against `data/raw/sources.json` before
reading them and embeds their URLs, sizes, and SHA-256 digests in entry
provenance. Kelly's source-provided A1–C2 value is retained as
`sources.frequency.cefr_band`, not as a generic learner-level tag: it is a
corpus/frequency-informed band and is not treated as independently validated
item difficulty. Frequency rank remains the deck-order signal; the local
estimator below writes reviewed results to a separate `learner_level` field.
For deliberate custom inputs,
`--allow-unmanifested` records local digests without claiming an upstream URL.
The Folkets XML is indexed once;
there is no dictionary request per word. Definitions, translated examples,
source record positions, dictionary version, and source-declared licence are
retained where available. Folkets pronunciation notation, spelling variants,
constructions, usage labels, translation comments, explanations, synonyms,
compounds, derivations, and idioms are preserved separately under `lexical_info`
instead of being flattened into definitions. Kelly's `Examples` column is also
lexical guidance rather than sentence data: its 245 values are predominantly
patterns such as `e.g. anmäla sig`. They are stored as bounded
`lexical_info.kelly_hints`, never copied into `definitions[].example`, and may
guide the model when it is already generating an example or assessing a usage
note. Parenthetical headword annotations such as `förk. kr.` receive the same
treatment.

Missing learner-facing `en`/`ett` articles are inferred only from Kelly's
explicit `noun-en`/`noun-ett` class. Rows that normalize to the same article,
headword, and part of speech are one lexical identity: for example, `krona` and
`krona (förk. kr.)` produce one `en krona` entry. The earliest frequency rank is
kept, later source rows remain under `sources.frequency.alternate_records`, and
their annotations remain available as hints. A lexical parenthetical expansion
replaces the primary text only when a Kelly word-class label has demonstrably
leaked into that primary text. Positional paradigms are labelled
conservatively. Tied or structurally ambiguous forms are omitted or retained
under `source_ambiguities` rather than guessed.

Use a bounded sample while inspecting changes:

```bash
python3 scripts/kelly_to_json.py --limit 25 --output build/sample-json
python3 scripts/verify_json_frequency.py --source build/sample-json
```

A sample starting at the first Kelly row has a contiguous rank range and no
gaps; the full deck legitimately has five gaps from collapsed duplicate rows.

### 3. Propose learner levels locally (no LLM)

Kelly's six equal-sized frequency bands are retained only as
`sources.frequency.cefr_band`. A separate learner estimate can be proposed from
POS-sensitive evidence in SVALex (receptive coursebook exposure) and SweLLex
(productive learner writing):

```bash
python3 scripts/enrich_cefr_levels_local.py propose --dry-run
python3 scripts/enrich_cefr_levels_local.py propose
```

The estimator requires evidence in at least two SVALex coursebook documents or
three SweLLex learner documents. Transparent numerals and ordinals use a bounded
family rule: basic Kelly A1 forms remain A1 and other unsupported forms are
capped at A2. This changes `femtionde` from Kelly's source band C2 to a proposed
learner level A2. Unsupported words remain unassigned rather than being guessed.
No LLM or API is contacted.

Review `review/cefr_level_proposals.jsonl`, then apply explicitly:

```bash
python3 scripts/enrich_cefr_levels_local.py apply --apply
```

Application creates a separate `learner_level` object with confidence,
rationale, method, and compact corpus evidence. It never changes Kelly's source
band and rejects stale or conflicting proposals.

SVALex/SweLLex tag common-gender and neuter nouns, so they also carry corpus
evidence for the 76 nouns where Kelly supplies no `en`/`ett` at all. That
evidence is deliberately **not** applied automatically: the same group includes
plural-only and idiom-bound nouns such as `kläder`, `anor`, `förhand`, and
`vägnar`, for which an automatic `en` would produce a wrong card. Kelly
omitting the article there is meaningful information rather than a gap to fill
blindly, so those entries stay reviewable by hand.

### 4. Generate missing content in Codex batches (subscription)

`enrich_lexicon_codex_subscription.py` is the preferred, subscription-based
alternative to `enrich_lexicon_openai_batch.py`. It drives the Codex CLI (a Codex/ChatGPT login) instead of
OpenAI API credits and covers the same four enrichment jobs:

- `definitions` — English word meaning plus a Swedish example and English
  translation for entries where Folkets supplied no gloss;
- `examples` — one natural Swedish example sentence plus its English
  translation for each existing definition that lacks one;
- `notes` — a conservative, sense-specific usage-note assessment;
- `inflections` — missing modern standard Swedish forms per part of speech.

A definition's optional `note` is supplementary, not another gloss. It is used
only for a likely misuse involving a fixed/non-compositional expression,
register or pragmatics, a necessary cultural/institutional reference, a
required construction/collocation, or a common false friend. Notes are English,
one sentence, normally at most 25 words and always at most 180 characters.
They must not repeat the meaning, hide essential meaning, add trivia, or make
unsupported cultural claims. Missing `note` means unassessed; explicit `null`
means assessed and unnecessary, so transparent words are not regenerated on
every run.

Standalone `notes` targets default to `--notes-scope all`: the model decides
for every complete sense. The reason is not that every word needs a note — most
do not — but that the judgment needs the word, gloss, and example in hand. A
gate over source metadata decides *before* seeing those, so it cannot tell a
false friend that still confuses from one the example sentence already
resolves.

In practice the example sentence is the main disambiguator, and it is why plain
lookalikes such as `glass` (ice cream), `prick` (dot), `slut` (end), or `fart`
(speed) normally need no note at all: the policy requires a lookalike note only
when confusion survives the gloss and the example, and otherwise demands
`null`. Sparse notes are the intended outcome, and an explicit `null` is the
model's recorded decision rather than an omission.

Batching keeps full coverage affordable: one `codex exec` call covers up to
`--batch-size` senses, so roughly 4,900 note targets is about 20 requests.

`--notes-scope evidence` is an optional cheaper pass that asks only about
senses whose Folkets data hints at a note-worthy issue (an idiom, a usage or
register label, a translation comment, a cultural explanation, or an
informative valency pattern such as `A & på x`). Plain transitivity (`A & x`)
and near-synonym or compound lists never trigger a request. On the current data
that currently selects 2,352 senses instead of 4,872. Because it decides
without seeing the card, it is a coverage trade rather than the default.

Notes are never gated when a sense still needs an example or translation: that
request happens anyway, so the note is assessed inside it for free. New
definitions likewise assess their note in the same response, so no sense is
assessed twice.

Preview all pending work without running Codex or writing anything:

```bash
python3 scripts/enrich_lexicon_codex_subscription.py propose --source data/json --dry-run
```

Process one resumable batch of up to 250 targets across the default lexical
kinds with the authenticated Codex CLI:

```bash
python3 scripts/enrich_lexicon_codex_subscription.py propose --source data/json \
  --model gpt-5.6-luna --batch-size 250 --max-batches 1
```

Run several independent subscription-backed batches concurrently when latency
matters:

```bash
python3 scripts/enrich_lexicon_codex_subscription.py propose --source data/json \
  --model gpt-5.6-luna --batch-size 250 --workers 3 --max-batches 12
```

`--workers` starts separate ephemeral `codex exec` processes; 2–4 is the
recommended starting range. The default remains 1 because parallel work uses
subscription quota faster and may encounter account throttling. Set
`--max-batches 0` to process all pending batches. Each worker has an isolated
schema/output directory, while the parent process alone appends validated rows
to proposal and failure JSONL files. Heartbeats identify their batch. A
transport-level batch failure cancels queued and peer work; already saved
batches remain resumable.

Restrict to a single job kind with a repeatable `--kind` (for example
`--kind examples`), or steer volume with `--limit` and `--max-batches`.
Targets carry a stable ID and the Swedish word, article, and part of speech.
Bounded relevant Folkets hints are supplied to both LLM transports to improve
sense choice, constructions, register, and usage notes. When Folkets provides a
more precise explanation behind a broad translation gloss, one short
sense-specific `sense_glosses` value is retained and sent only with that sense;
pronunciation and large compound lists are deliberately excluded from prompts.
Pending work is ordered
by the reviewed learner level (falling back to Kelly's band), then Kelly rank,
so useful beginner material is completed first. Examples and definitions prioritize short, reusable modern Swedish from adult
SFI D/SVA contexts—everyday life, work, study, and civic life—while preserving
particles, reflexive forms, collocations, and the requested sense. The
subprocess is ephemeral and read-only. Results are appended to
`review/lexicon_codex_subscription_proposals.jsonl`; source JSON is unchanged. Subsequent
runs skip completed target IDs. Each target requests only fields that are
actually missing, so Codex never has to echo protected source text. Responses
use a compact strict JSON schema. Validation is target-local: valid results are
saved even if another result in the same model response is rejected; rejected
IDs remain pending for a later retry. Inspect the proposal file, then
explicitly apply it:

```bash
python3 scripts/enrich_lexicon_codex_subscription.py apply --source data/json --apply
```

Application fills empty lexical source fields only and rejects proposals when
the source word, meaning, existing sentence, or an already-filled inflection
would be overwritten or contradicted. Reapplying the same accepted proposal is
a safe no-op.

After lexical fields are complete, run the holistic learner-card review:

```bash
python3 scripts/enrich_lexicon_codex_subscription.py propose --source data/json \
  --kind cards --production-limit 1000 --workers 3 --max-batches 0
python3 scripts/enrich_lexicon_codex_subscription.py apply --source data/json --apply
```

This pass does not rewrite definitions. It stores a versioned `card_plan`
projection beside them. Every complete entry goes through the semantic review;
the exporter and audio generator reject entries without a current reviewed
plan. This avoids treating a merely well-shaped example as semantically
verified when historical per-field generation provenance is unavailable. On
the current data that is 8,420 targets. Holistic card output is larger than
lexical output, so card batches are capped internally at 100 targets (85
batches currently) even when the general `--batch-size` is 250.

Every source sense must be covered exactly once. Redundant senses may merge;
distinct senses remain separate. Each retained sense receives one contextual
meaning plus optional closely related wording instead of an untested
semicolon-separated gloss list. The model must also produce a target form that
is both accepted for the entry and literally present in the reviewed sentence,
and a revalidated note or null. Existing definitions/examples remain immutable
provenance even when the learner-facing plan corrects a weak example such as an
object-form sentence for `vi`.

Target matching covers explicit source paradigms plus conservative productive
forms that dictionaries commonly omit: Swedish s-passives, noun genitives,
reflexive person realizations for lexical `... sig` constructions, and common
possessive/determiner agreement forms. Validation may repair the model's chosen
`target_form` only when exactly one other accepted form occurs as a complete
form in the final sentence. It still rejects ambiguous matches, unrelated
compounds/derivations, and examples that demonstrate another paradigm member
instead of the requested target. The accepted span is stored and highlighted
on the recognition card.

Production eligibility is local-first and capped. A score rewards reviewed
A1–B2 evidence, Kelly frequency, informative Folkets constructions, idioms,
Kelly usage patterns, compact collocations, reflexives, and existing
construction notes; marked archaic/regional material is penalized. Only the
top `--production-limit` eligible entries reach the model, which may still
return null. Approved production cues describe a situation or function and add
a bounded Swedish target-family hint when needed; the pipeline never reverses
all vocabulary cards mechanically. The approved item becomes a separate chunk
note rather than adding empty production fields to every vocabulary note.
Use `--production-limit 0` for a recognition-only deck. Choose the limit before
the first card-plan run and keep it unchanged across retries; completed stable
IDs are intentionally not regenerated merely because a later command uses a
different limit.

### Card-review options and retries

`--kind cards` is not part of the default lexical enrichment run. Run it only
after applying lexical proposals, then run `apply --apply` again after reviewing
the card proposals. Useful options are:

- `--production-limit N` — maximum locally ranked entries that may receive a
  dedicated chunk-production note; all retained senses still receive
  recognition notes;
- `--workers N` — concurrent Codex CLI processes (2–4 recommended);
- `--batch-size N` — requested targets per call; card batches are capped at 100
  because their responses are larger than lexical responses;
- `--max-batches N` — stop after N batches; `0` means all pending batches;
- `--limit N` — process only the first N pending targets;
- `--dry-run` — show pending counts and batches without calling Codex or writing
  proposals.

Proposal JSONL files are resumability state; do not delete them between runs.
Successful stable IDs are skipped, while rejected targets remain pending. Retry
with the same command. As the easy entries complete, the remainder may be
concentrated in exact-form validation cases and therefore show a higher
rejection percentage. Smaller batches give those entries more attention:

```bash
python3 scripts/enrich_lexicon_codex_subscription.py propose \
  --kind cards --production-limit 1000 --batch-size 20 \
  --workers 3 --max-batches 0
```

The failure JSONL is an append-only diagnostic history; old failure rows do not
block a later success. Card-plan v2 IDs coexist safely with older lexical
proposal rows. A source digest prevents a stale plan from being applied after
its lexical source changes.

The equivalent metered OpenAI API workflow remains available but is not needed
for the standard subscription pipeline:

```bash
python3 scripts/enrich_lexicon_openai_batch.py batch-submit --source data/json
python3 scripts/enrich_lexicon_openai_batch.py batch-collect --batch-id YOUR_BATCH_ID --source data/json
python3 scripts/enrich_lexicon_openai_batch.py apply --source data/json --apply
```

The equivalent holistic card review is also available through Batch:

```bash
python3 scripts/enrich_lexicon_openai_batch.py batch-submit --source data/json \
  --cards-only --production-limit 1000
python3 scripts/enrich_lexicon_openai_batch.py batch-collect \
  --batch-id YOUR_BATCH_ID --source data/json --cards-only --production-limit 1000
python3 scripts/enrich_lexicon_openai_batch.py apply --source data/json --apply
```

If `batch-submit` uses the optional `--notes-scope evidence` cost/coverage
trade-off, repeat that option on `batch-collect`; proposal rows and the batch
manifest record the scope for review and resumability.

### 5. Generate keyless Swedish TTS

Preview without importing `edge-tts`, making network calls, or writing files:

```bash
python3 scripts/json_to_audio.py --source data/json --output build/audio --dry-run
```

Generate missing clips:

```bash
python3 scripts/json_to_audio.py --source data/json --output build/audio --synthesize
```

The default voice is `sv-SE-SofieNeural`; override it with `--voice`. Filenames
include a fingerprint of the normalized text, voice, synthesis settings, and
installed `edge-tts` version. `build/audio/audio_manifest.json` records the
`sentence_tts` role, source definition indices, and file digest for every
retained sense, preventing stale text, sense merges, or voice settings from
silently reusing an old clip.

`edge-tts` uses Microsoft Edge's online speech service without an API key. It is
not an offline engine or a supported Azure Speech API, so availability and
voice names can change. Review a few generated clips and applicable service and
redistribution terms before publishing audio.

### 6. Validate without touching Anki

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
ruff check scripts tests
python3 scripts/verify_json_frequency.py --source data/json
python3 scripts/json_to_anki.py --core data/json --audio-dir build/audio
```

The exporter command is preview-only without `--apply`. The automated tests
block Anki access and never import notes or media. This repository does not use
a temporary Anki profile as a test mechanism. `ruff` is used as the lint gate
(`ruff format --check scripts tests` for formatting). The frequency audit fails
on malformed entries, duplicate ranks, or duplicate normalized lexical
identities; rank gaps are reported but accepted
because the Kelly workbook repeats a few article/headword/class identities that
the converter intentionally collapses into one entry.

When you have reviewed the JSON, audio manifest, and preview, perform the final
Anki step yourself with the documented exporter options. `json_to_anki.py`
changes Anki only when you explicitly supply `--apply`.

The recognition note type uses these learner-facing/content-metadata fields:

```text
Word | Card ID | Source IDs | Target Form | Sentence | Context Meaning |
Sentence Meaning | Usage Note | Other Meanings | Frequency Order |
Learner Level | Part of Speech | Variants | Audio | Inflections | Source Type
```

Only approved reusable constructions create a separate chunk note:

```text
Chunk ID | Cue | Swedish Answer | Target Chunk | Context Meaning | Example |
Example Meaning | Usage Note | Source Word | Frequency Order | Learner Level |
Part of Speech | Audio | Source Type
```

One Anki note represents one retained sense, so a second meaning can no longer
appear untested on the first sense's answer. The exact target form is highlighted
inside its sentence. A context-specific meaning is primary; closely related
wording appears only as optional `Other Meanings`. Revalidated usage notes,
variants, and compact paradigms remain answer-side information. Predictable
verb/adjective forms are visually quiet while irregular forms receive emphasis.

Frequency rank is removed from the question but retained on the answer and in
`Frequency Order`. CEFR, POS, source, and rank remain separate fields. Notes are
tagged with stable `cefr::...`, `pos::...`, `source::...`, and card-type tags for
filtered decks. `Word` remains first, while `Card ID` gives each sense a stable
update identity based on Kelly rank plus source indices. Exact duplicate
learning tasks are merged across files while retaining all source IDs and POS
metadata; same-spelling entries remain separate when their reviewed meaning or
context differs. The exact generated models'
templates and CSS are synchronized on `--apply`; other field layouts are
rejected rather than migrated.

Every retained sense creates a focused **recognition-in-context** card: Swedish
word, highlighted form, reviewed context, and sentence TTS on the question;
meaning, translation, note, variants, compact forms, CEFR, POS, source, and rank
appear after reveal. A dedicated **chunk-production note** is created only when
the local shortlist and semantic reviewer both approve a reusable
chunk/construction. It lives in the `Chunks` subdeck and uses a
situation/function cue rather than an unconstrained reverse translation.

## Source and licence notes

- **Swedish Kelly list**, Språkbanken Text, University of Gothenburg: frequency,
  CEFR level, headword, and grammatical metadata. See the official resource
  page and workbook metadata for citation and current reuse terms:
  <https://spraakbanken.gu.se/en/resources/kelly>.
- **SVALex v2** and **SweLLex v2**, CEFRLex: descriptive receptive and
  productive Swedish L2 frequency distributions. Both downloads declare
  **CC BY-NC-SA 4.0** and are used as evidence rather than definitive word
  assignments: <https://cental.uclouvain.be/cefrlex/svalex/> and
  <https://cental.uclouvain.be/cefrlex/swellex/>.
- **Folkets lexikon**, KTH: Swedish-English glosses, examples, and paradigms.
  The downloaded XML declares **CC BY-SA 2.5**, its version, last-change date,
  origin URL, and source/target languages in the root element. Derived lexical
  content must retain appropriate attribution and ShareAlike handling.
- **Codex CLI** and the optional **OpenAI API** path generate reviewable
  definitions, examples/translations, sense-specific usage-note assessments,
  inflections, and reviewed learner-card projections under the same quality
  contract. Neither path silently replaces populated source values.
- **edge-tts** generates `sentence_tts` audio. Generated speech is distinct from
  original sentence recordings and dictionary pronunciation audio.

Generated and source-derived material can contain errors. Corrections are
welcome with the Swedish headword, proposed correction, and a reliable source.

## Safety

Never commit API keys, downloaded source corpora without a deliberate licence
review, generated media, Anki collections, or personal review data. The code is
MIT licensed; source data and generated deck content remain subject to their
respective source/service terms.
