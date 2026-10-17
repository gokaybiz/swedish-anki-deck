import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'book_pipeline.py'
spec = importlib.util.spec_from_file_location('book_pipeline', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BookPipelineTests(unittest.TestCase):
    def test_samples_across_contexts(self):
        self.assertEqual([x['id'] for x in module.sample_contexts([{'id': i} for i in range(21)], 3)], [0, 10, 20])

    def test_csv_import_prepare_apply_anki(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            word_file = root / 'words.csv'
            with word_file.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['word','frequency','pdf_pages',
                    'printed_pages_estimate','first_pdf_page','first_occurrence_id',
                    'surface_variants','source_sentence_count','source_sentences'])
                writer.writeheader()
                writer.writerow({'word':'brallor','frequency':'2',
                    'pdf_pages':'[11,22]','printed_pages_estimate':'[7,18]',
                    'first_pdf_page':'11','first_occurrence_id':'o000001',
                    'surface_variants':'["Brallor", "brallor"]',
                    'source_sentence_count':'2',
                    'source_sentences':json.dumps([{'sentence_id':'s1','text':'Det här är inga brallor man vill ha.', 'pdf_pages':[11]},
                                                   {'sentence_id':'s2','text':'Hans brallor var alldeles för stora.', 'pdf_pages':[22]}])})
            entries = root/'entries'
            batches = root/'batches'
            prompt = root/'prompt.md'
            prompt.write_text('PROMPT\n{{INPUT_RECORDS_JSON}}', encoding='utf-8')
            cmd = [sys.executable, str(SCRIPT)]
            def run(*args):
                return subprocess.run(cmd+list(map(str,args)), text=True, capture_output=True, check=True)
            self.assertIn('Imported 1', run('import','--csv',word_file,'--entries',entries).stdout)
            self.assertIn('kept 1 existing', run('import','--csv',word_file,'--entries',entries).stdout)
            self.assertIn('Prepared 1', run('prepare','--entries',entries,'--prompt',prompt,'--batches',batches).stdout)
            request = list(batches.glob('*.request.json'))[0]
            batch = json.loads(request.read_text())
            self.assertEqual(batch['records'][0]['frequency'], 2)
            self.assertEqual(batch['records'][0]['pdf_pages'], [11,22])
            self.assertIn('BOOK-SPECIFIC OVERRIDES', list(batches.glob('*.prompt.txt'))[0].read_text())
            note = {'note_key':'brallor:noun:trousers','word_sv':'brallor','article':None,
                    'part_of_speech':'noun','word_en':'trousers','word_tr':'pantolon',
                    'example_sv':'Jag hittade mina gamla brallor.',
                    'example_en':'I found my old trousers.',
                    'example_tr':'Eski pantolonumu buldum.',
                    'example_type':'original_teaching_example','inflections':{},
                    'pdf_pages':[11,22],'quality_status':'generated_checked',
                    'register':'informal','book_forms':['brallor']}
            response_path = request.with_name(request.name.replace('.request.json','.response.json'))
            def response(notes):
                return {'records':[{'input_id':'o000001','disposition':'notes_created',
                    'reason':None,'merged_into_input_id':None,'notes':notes}]}
            response_path.write_text(json.dumps(response([note])))
            self.assertIn('dry run', run('apply','--entries',entries,'--batches',batches).stdout)
            self.assertEqual(json.loads((entries/'o000001.json').read_text())['disposition'], 'not_processed')
            self.assertIn('Applied 1', run('apply','--entries',entries,'--batches',batches,'--apply').stdout)
            self.assertIn('1 spoiler-safe bilingual notes', run('anki','--entries',entries).stdout)
            fields = module.anki_notes(entries)[0]
            self.assertEqual(fields['Turkish'], 'pantolon')
            self.assertEqual(fields['Exact Frequencies'], 'brallor: 2')
            self.assertEqual(fields['PDF Pages'], '11, 22')
            # Guard against verbatim quotation from the novel.
            note['example_sv'] = 'Det här är inga brallor man vill ha.'
            response_path.write_text(json.dumps(response([note])))
            with self.assertRaises(subprocess.CalledProcessError):
                run('apply','--entries',entries,'--batches',batches,'--apply')

    def test_no_false_lemma_merge(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for i, meaning in enumerate(['to light', 'teeth']):
                module.dump(directory/f'o00000{i+1}.json',
                    {'source_order': i+1, 'source_form':'tänder', 'exact_form_frequency':3,
                     'pdf_pages':[1], 'disposition':'notes_created', 'notes':[
                        {'note_key':f'tänder:{meaning}', 'word_sv':'tänder',
                         'part_of_speech':'noun' if i else 'verb', 'word_en': meaning,
                         'word_tr':'...', 'example_sv':'...', 'example_en':'...', 'example_tr':'...',
                         'quality_status':'generated_checked'}]})
            self.assertEqual(len(module.anki_notes(directory)), 2)

if __name__ == '__main__':
    unittest.main()