"""Book-word CSV -> spoiler-safe bilingual notes -> AnkiConnect (no Kelly dependency).

Requires the user's local words.csv and wordnote_generation_prompt.md. Neither the
copyrighted sentences nor model results are committed to the public repository.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data' / 'book' / 'entries'
DEFAULT_BATCHES = ROOT / 'review' / 'book' / 'batches'
DEFAULT_PROMPT = ROOT / 'prompts' / 'wordnote_generation_prompt.md'
MODEL = 'Swedish Book Vocabulary (generated)'
FIELDS = ['Book Key', 'Swedish', 'Example', 'English', 'Turkish', 'Example English',
          'Example Turkish', 'Part of Speech', 'Inflections', 'Pattern', 'Usage English',
          'Usage Turkish', 'Register', 'Book Forms', 'PDF Pages', 'Exact Frequencies']
FRONT = '''<div class="head">{{Swedish}}</div><div class="example">{{Example}}</div>'''
BACK = '''{{FrontSide}}<hr><div class="meaning">{{English}} / {{Turkish}}</div>
<div>{{Example English}}</div><div>{{Example Turkish}}</div>
{{#Inflections}}<div class="detail">{{Inflections}}</div>{{/Inflections}}
{{#Pattern}}<div class="detail">{{Pattern}}</div>{{/Pattern}}
{{#Usage English}}<div class="detail">{{Usage English}}</div>{{/Usage English}}
{{#Usage Turkish}}<div class="detail">{{Usage Turkish}}</div>{{/Usage Turkish}}
<div class="meta">{{Part of Speech}} · {{Register}} · PDF {{PDF Pages}} · {{Exact Frequencies}}</div>'''
CSS = '.card{text-align:left;font:20px Arial,sans-serif;background:#fafafa;color:#222;max-width:700px;margin:auto;padding:20px}.head{font-size:30px;font-weight:bold}.example{margin:15px 0;font-size:24px}.meaning{font-weight:bold;font-size:22px}.detail{margin:12px 0;color:#555}.meta{font-size:12px;color:#777;margin-top:24px}'
SPOILER_RULES = '''
BOOK-SPECIFIC OVERRIDES (apply together with the supplied prompt):
- Source sentences are private reading context, NOT examples to publish on a card.
- Create a fresh standalone Swedish example or adapt a mundane fragment by changing
  the characters, setting and situation. A slightly modified book sentence is fine
  only if it cannot reveal events, relationships, dialogue, surprises or outcomes.
- Do not repeat distinctive book dialogue, proper names, plot events or identifiable
  combinations of details in example_sv, translations, meanings or sidenotes.
- Preserve the lexical sense and register including colloquial or vulgar language.
- Input may have sampled contexts, not all book occurrences. Do not claim complete
  sense/lemma frequency or assign guessed occurrence IDs. Use null and incomplete.
- Only first_occurrence_id is available per surface form; sentence_id is NOT an
  occurrence ID. pdf_pages are exact; printed_pages_estimate is estimated.
- Mark ambiguous sense, probable OCR artifacts and inadequate context needs_review.
- Use original_teaching_example or adapted_teaching_example as example_type.
- Never quote original source sentences in learner-visible card content.
'''


def dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return data


def read_entries(folder: Path) -> list[tuple[Path, dict]]:
    if not folder.is_dir():
        raise ValueError(f'Entries not found: {folder}; run import first')
    return sorted(((p, load(p)) for p in folder.glob('*.json')),
                  key=lambda pair: pair[1]['source_order'])


def do_import(args: argparse.Namespace) -> None:
    csv.field_size_limit(sys.maxsize)
    with args.csv.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'word', 'frequency', 'pdf_pages', 'printed_pages_estimate',
                    'first_pdf_page', 'first_occurrence_id', 'surface_variants',
                    'source_sentence_count', 'source_sentences'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing CSV columns: {required - set(reader.fieldnames or [])}')
        used = set()
        created = skipped = 0
        for order, row in enumerate(reader, 1):
            if args.limit and order > args.limit:
                break
            uid = row['first_occurrence_id'].strip()
            if not re.fullmatch(r'o\d+', uid) or uid in used:
                raise ValueError(f'Invalid/duplicate first occurrence ID: {uid!r}')
            used.add(uid)
            frequency = int(row['frequency'])
            pages = json.loads(row['pdf_pages'])
            printed = json.loads(row['printed_pages_estimate'])
            variants = json.loads(row['surface_variants'])
            contexts = json.loads(row['source_sentences'])
            if frequency < 1 or not isinstance(contexts, list) or not isinstance(pages, list):
                raise ValueError(f'Invalid source statistics: {uid}')
            if len(contexts) != int(row['source_sentence_count']):
                raise ValueError(f'Sentence count disagrees for {uid}')
            path = args.entries / f'{uid}.json'
            if path.exists() and not args.overwrite:
                skipped += 1
                continue
            data = {'source_id': uid, 'source_order': order, 'source_form': row['word'],
                    'exact_form_frequency': frequency, 'pdf_pages': pages,
                    'printed_pages_estimate': printed, 'first_pdf_page': int(row['first_pdf_page']),
                    'surface_variants': variants, 'first_occurrence_id': uid,
                    'source_sentence_count': len(contexts), 'source_sentences': contexts,
                    'disposition': 'not_processed', 'notes': []}
            dump(path, data)
            created += 1
    print(f'Imported {created} records, kept {skipped} existing; source text remains local.')


def sample_contexts(contexts: list[dict], limit: int) -> list[dict]:
    if len(contexts) <= limit:
        return contexts
    # Spread coverage across the book; take beginning/end as well as interior.
    indices = sorted({round(i * (len(contexts)-1) / (limit-1)) for i in range(limit)})
    return [contexts[i] for i in indices]


def prepare(args: argparse.Namespace) -> None:
    prompt = args.prompt.read_text(encoding='utf-8')
    if '{{INPUT_RECORDS_JSON}}' not in prompt:
        raise ValueError('Prompt must contain {{INPUT_RECORDS_JSON}}')
    pending = [(p, x) for p, x in read_entries(args.entries)
               if x.get('disposition') == 'not_processed']
    if args.limit:
        pending = pending[:args.limit]
    count = 0
    for start in range(0, len(pending), args.batch_size):
        group = pending[start:start+args.batch_size]
        inputs = []
        for _, x in group:
            contexts = sample_contexts(x['source_sentences'], args.max_contexts)
            inputs.append({'id': x['source_id'], 'word': x['source_form'],
                           'frequency': x['exact_form_frequency'],
                           'pdf_pages': x['pdf_pages'],
                           'printed_pages_estimate': x['printed_pages_estimate'],
                           'surface_variants': x['surface_variants'],
                           'first_occurrence_id': x['first_occurrence_id'],
                           'source_sentence_count': x['source_sentence_count'],
                           'contexts_sampled': len(contexts) < x['source_sentence_count'],
                           'sampled_sentence_count': len(contexts),
                           'source_sentences': contexts,
                           'available_occurrence_ids': [x['first_occurrence_id']]})
        batch = {'records': inputs}
        serial = f'batch_{group[0][1]["source_id"]}_{group[-1][1]["source_id"]}'
        request_path = args.batches / f'{serial}.request.json'
        instruction_path = args.batches / f'{serial}.prompt.txt'
        dump(request_path, batch)
        instruction_path.write_text(prompt.replace('{{INPUT_RECORDS_JSON}}',
            json.dumps(batch, ensure_ascii=False)) + '\n' + SPOILER_RULES, encoding='utf-8')
        count += 1
    print(f'Prepared {count} batches for {len(pending)} records in {args.batches}.')


def generate(args: argparse.Namespace) -> None:
    requests = sorted(args.batches.glob('*.request.json'))
    if args.limit:
        requests = requests[:args.limit]
    if not requests:
        raise ValueError('No batches found; run prepare first')
    for i, path in enumerate(requests, 1):
        output = path.with_name(path.name.replace('.request.json', '.response.json'))
        if output.exists() and not args.overwrite:
            continue
        prompt = path.with_name(path.name.replace('.request.json', '.prompt.txt'))
        with tempfile.TemporaryDirectory() as temp:
            response = Path(temp) / 'response.json'
            command = [args.codex, 'exec', '-m', args.model, '--ephemeral',
                       '--skip-git-repo-check', '--sandbox', 'read-only',
                       '--output-last-message', str(response), '-']
            completed = subprocess.run(command, input=prompt.read_text(encoding='utf-8'),
                                       text=True, capture_output=True, timeout=args.timeout)
            if completed.returncode:
                raise RuntimeError(f'Batch {i}: Codex failed: {completed.stderr[-1200:]}')
            result = json.loads(response.read_text(encoding='utf-8'))
            if not isinstance(result, dict) or not isinstance(result.get('records'), list):
                raise ValueError(f'Batch {i}: malformed model JSON')
            dump(output, result)
            print(f'Generated {output}', flush=True)
    print('Review the responses before running apply.')


def apply(args: argparse.Namespace) -> None:
    files = sorted(args.batches.glob('*.response.json'))
    if not files:
        raise ValueError('No response files; run generate or write reviewed responses')
    updated = 0
    for response_path in files:
        request_path = response_path.with_name(response_path.name.replace('.response.json', '.request.json'))
        inputs = {r['id']: r for r in load(request_path)['records']}
        output = load(response_path)
        records = output.get('records')
        if not isinstance(records, list) or any(not isinstance(x, dict) for x in records):
            raise ValueError(f'{response_path}: records must be a list of objects')
        by_id = {record.get('input_id'): record for record in records}
        if len(by_id) != len(records) or set(by_id) != set(inputs):
            raise ValueError(f'{response_path}: missing, repeated or unexpected IDs')
        changes = []
        for uid, result in by_id.items():
            path = args.entries / f'{uid}.json'
            entry = load(path)
            if entry['source_form'] != inputs[uid]['word']:
                raise ValueError(f'{uid}: source mismatch; prepare fresh batches')
            disposition = result.get('disposition')
            if disposition not in {'notes_created', 'merged_into_other_record',
                'source_review_required', 'proper_name', 'nonlexical_artifact'}:
                raise ValueError(f'{uid}: invalid disposition {disposition!r}')
            notes = result.get('notes')
            if not isinstance(notes, list) or (disposition == 'notes_created' and not notes):
                raise ValueError(f'{uid}: invalid notes')
            for note in notes:
                required = ['note_key', 'word_sv', 'part_of_speech', 'word_en', 'word_tr',
                            'example_sv', 'example_en', 'example_tr', 'example_type',
                            'inflections', 'pdf_pages', 'quality_status']
                if any(key not in note for key in required):
                    raise ValueError(f'{uid}: missing note keys')
                if note['quality_status'] not in {'generated_checked', 'needs_review',
                    'independently_reviewed'}:
                    raise ValueError(f'{uid}: invalid quality status')
                if note['example_type'] not in {'original_teaching_example',
                                                'adapted_teaching_example'}:
                    raise ValueError(f'{uid}: example origin not declared')
                if not isinstance(note['inflections'], dict):
                    raise ValueError(f'{uid}: invalid inflections')
                example = note['example_sv'].strip().casefold()
                for context in entry['source_sentences']:
                    source = context['text'].strip().casefold()
                    if example == source:
                        raise ValueError(f'{uid}: verbatim book quotation cannot become a card')
                    tokens = re.findall(r'\w+', example)
                    original = re.findall(r'\w+', source)
                    if len(tokens) >= 8 and any(tokens[i:i+8] == original[j:j+8]
                        for i in range(len(tokens)-7) for j in range(len(original)-7)):
                        raise ValueError(f'{uid}: lengthy unchanged book passage in teaching example')
            entry['disposition'] = disposition
            entry['reason'] = result.get('reason')
            entry['merged_into_input_id'] = result.get('merged_into_input_id')
            entry['notes'] = notes
            changes.append((path, entry))
        if not args.apply:
            print(f'Validated {response_path.name}: {len(changes)} records (dry run)')
            continue
        for path, entry in changes:
            dump(path, entry)
            updated += 1
    print(f'Applied {updated} source records; no Anki changes performed.')


def api(action: str, **params):
    body = json.dumps({'action': action, 'version': 6, 'params': params}).encode()
    req = Request('http://127.0.0.1:8765', body, {'Content-Type': 'application/json'})
    with urlopen(req, timeout=45) as reply:
        result = json.load(reply)
    if result.get('error'):
        raise RuntimeError(f'AnkiConnect {action}: {result["error"]}')
    return result['result']


def field(value) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value if value is not None else '').strip())


def anki_notes(folder: Path) -> list[dict]:
    cards = {}
    for path, entry in read_entries(folder):
        if entry.get('disposition') != 'notes_created':
            continue
        for note in entry.get('notes', []):
            if note.get('quality_status') not in {'generated_checked', 'independently_reviewed'}:
                continue
            key = 'book:' + str(note['note_key']).strip()
            if key == 'book:':
                raise ValueError(f'{path}: empty note_key')
            content = (note['word_sv'], note['part_of_speech'], note['word_en'])
            if key in cards:
                previous = cards[key]
                if previous['_identity'] != content:
                    raise ValueError(f'{key}: conflicting lexical notes')
                previous['_forms'][entry['source_form']] = entry['exact_form_frequency']
                previous['_pages'].update(entry['pdf_pages'])
                continue
            cards[key] = {'_identity': content,
                '_forms': {entry['source_form']: entry['exact_form_frequency']},
                '_pages': set(entry['pdf_pages']),
                '_note': note}
    output = []
    for key, item in cards.items():
        note = item['_note']
        forms = item['_forms']
        output.append({'Book Key': key, 'Swedish': field(('en ' if note.get('article') == 'en' else
            'ett ' if note.get('article') == 'ett' else '') + note['word_sv']),
            'Example': field(note['example_sv']), 'English': field(note['word_en']),
            'Turkish': field(note['word_tr']), 'Example English': field(note['example_en']),
            'Example Turkish': field(note['example_tr']),
            'Part of Speech': field(note['part_of_speech']),
            'Inflections': field(note.get('inflections') or ''),
            'Pattern': field(note.get('pattern')), 'Usage English': field(note.get('sidenote_en')),
            'Usage Turkish': field(note.get('sidenote_tr')),
            'Register': field(note.get('register') or 'neutral'),
            'Book Forms': field(', '.join(sorted(forms))),
            'PDF Pages': field(', '.join(map(str, sorted(item['_pages'])))),
            'Exact Frequencies': field(', '.join(f'{form}: {n}' for form, n in sorted(forms.items())))})
    return sorted(output, key=lambda x: x['Book Key'])


def export(args: argparse.Namespace) -> None:
    cards = anki_notes(args.entries)
    print(f'Prepared {len(cards)} spoiler-safe bilingual notes; excluded missing/unreviewed entries.')
    if not args.apply:
        print('Preview only: add --apply with Anki open and AnkiConnect installed.')
        return
    models = api('modelNames')
    if MODEL not in models:
        api('createModel', modelName=MODEL, inOrderFields=FIELDS, css=CSS,
            isCloze=False, cardTemplates=[{'Name':'Recognition', 'Front':FRONT, 'Back':BACK}])
    else:
        actual = api('modelFieldNames', modelName=MODEL)
        if actual != FIELDS:
            raise ValueError(f'Existing {MODEL} has incompatible fields; refusing to overwrite')
    api('createDeck', deck=args.deck)
    made = revised = 0
    for fields in cards:
        key = fields['Book Key']
        # Fetch the exact stable key rather than relying on headword uniqueness.
        escaped = key.replace('"', '\\"')
        ids = api('findNotes', query=f'note:"{MODEL}" "Book Key:{escaped}"')
        if len(ids) > 1:
            raise ValueError(f'Ambiguous existing Anki notes: {key}')
        if ids:
            api('updateNoteFields', note={'id': ids[0], 'fields': fields})
            revised += 1
        else:
            api('addNote', note={'deckName': args.deck, 'modelName': MODEL,
                'fields': fields, 'options': {'allowDuplicate': True},
                'tags': ['swedish', 'book-vocabulary']})
            made += 1
    print(f'Created {made}, updated {revised}.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('import', help='Import book word CSV locally')
    sub.add_argument('--csv', type=Path, required=True)
    sub.add_argument('--entries', type=Path, default=DEFAULT_DATA)
    sub.add_argument('--overwrite', action='store_true')
    sub.add_argument('--limit', type=int, help='Import only first N forms (for testing)')
    sub.set_defaults(fn=do_import)
    sub = commands.add_parser('prepare', help='Prepare prompt batches')
    sub.add_argument('--entries', type=Path, default=DEFAULT_DATA)
    sub.add_argument('--prompt', type=Path, default=DEFAULT_PROMPT)
    sub.add_argument('--batches', type=Path, default=DEFAULT_BATCHES)
    sub.add_argument('--batch-size', type=int, default=5)
    sub.add_argument('--max-contexts', type=int, default=12)
    sub.add_argument('--limit', type=int)
    sub.set_defaults(fn=prepare)
    sub = commands.add_parser('generate', help='Run prompt batches with Codex subscription')
    sub.add_argument('--batches', type=Path, default=DEFAULT_BATCHES)
    sub.add_argument('--codex', default='codex')
    sub.add_argument('--model', default='gpt-5.6-luna')
    sub.add_argument('--timeout', type=int, default=1800)
    sub.add_argument('--limit', type=int)
    sub.add_argument('--overwrite', action='store_true')
    sub.set_defaults(fn=generate)
    sub = commands.add_parser('apply', help='Review/apply generated JSON; read-only by default')
    sub.add_argument('--entries', type=Path, default=DEFAULT_DATA)
    sub.add_argument('--batches', type=Path, default=DEFAULT_BATCHES)
    sub.add_argument('--apply', action='store_true')
    sub.set_defaults(fn=apply)
    sub = commands.add_parser('anki', help='Preview/export to AnkiConnect')
    sub.add_argument('--entries', type=Path, default=DEFAULT_DATA)
    sub.add_argument('--deck', default='Swedish Book Vocabulary')
    sub.add_argument('--apply', action='store_true')
    sub.set_defaults(fn=export)
    args = parser.parse_args()
    if getattr(args, 'batch_size', 1) < 1 or getattr(args, 'max_contexts', 12) < 2:
        parser.error('batch-size must be >=1 and max-contexts must be >=2')
    if getattr(args, 'limit', None) is not None and args.limit < 1:
        parser.error('--limit must be >=1')
    args.fn(args)


if __name__ == '__main__':
    main()