import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passport_ocr import (
    AS_OF, FIELDS, lower_mrz_is_valid, norm, normalize_mixed_word,
    parse_date, read_jsonl, repair_reference_line, validity_reason
)
from postprocess import PassportProcessor, process_file


class DomainTests(unittest.TestCase):
    def test_twenty_year_renewal_grace(self):
        from datetime import date
        fields = {'Birth_date': '01.01.2000', 'Issue_date': '01.02.2014'}
        self.assertIsNone(validity_reason(fields, date(2020, 3, 31)))
        self.assertEqual(validity_reason(fields, date(2020, 4, 1)), 'expired_20')
        fields['Issue_date'] = '01.01.2020'
        self.assertIsNone(validity_reason(fields))

    def test_forty_five_year_renewal_grace(self):
        from datetime import date
        fields = {'Birth_date': '01.01.1970', 'Issue_date': '02.01.1990'}
        self.assertIsNone(validity_reason(fields, date(2015, 4, 1)))
        self.assertEqual(validity_reason(fields, date(2015, 4, 2)), 'expired_45')
        fields['Issue_date'] = '01.01.2015'
        self.assertIsNone(validity_reason(fields))

    def test_dates_use_task_date(self):
        self.assertIsNone(parse_date('31.02.2020'))
        self.assertIsNotNone(parse_date('29.02.2020'))
        self.assertEqual(validity_reason({'Birth_date': '01.01.2000', 'Issue_date': '20.10.2026'}), 'issue_date_or_age')
        self.assertEqual(validity_reason({'Birth_date': '01.01.2000', 'Issue_date': '01.01.2013'}), 'issue_date_or_age')
        self.assertEqual(str(AS_OF), '2026-10-01')

    def test_mrz_corrupted_digit_fails_checksum(self):
        doc = read_jsonl(ROOT / 'data/train_predictions.jsonl')[0]
        lower = doc['mrz']['lines'][1]['text']
        self.assertTrue(lower_mrz_is_valid(lower))
        corrupted = str((int(lower[0]) + 1) % 10) + lower[1:]
        self.assertFalse(lower_mrz_is_valid(corrupted))

    def test_reference_repair_requires_unique_low_confidence_edit(self):
        line = {'text': 'СБЛАСТИ', 'char_conf': [0.5, 1, 1, 1, 1, 1, 1]}
        self.assertEqual(repair_reference_line(line, {'ОБЛАСТИ'}), ('ОБЛАСТИ', 1))
        self.assertEqual(repair_reference_line(line, {'ОБЛАСТИ', 'ВБЛАСТИ'}), ('СБЛАСТИ', 0))
        line['char_conf'][0] = 1.0
        self.assertEqual(repair_reference_line(line, {'ОБЛАСТИ'}), ('СБЛАСТИ', 0))

    def test_preserves_latin_place_names_and_letter_yo(self):
        import re
        self.assertEqual(re.sub('[A-ZА-ЯЁ]+', normalize_mixed_word, 'DIMITROVO'), 'DIMITROVO')
        self.assertEqual(re.sub('[A-ZА-ЯЁ]+', normalize_mixed_word, 'ОБЛАCТИ'), 'ОБЛАСТИ')
        self.assertNotEqual(norm('Е'), norm('Ё'))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = read_jsonl(ROOT / 'data/test_predictions.jsonl')
        cls.answers = read_jsonl(ROOT / 'answers.jsonl')
        cls.processor = PassportProcessor()
        cls.temporary = tempfile.TemporaryDirectory(prefix='.test-', dir=ROOT)
        cls.work = Path(cls.temporary.name)
        assert cls.work.resolve().parent == ROOT.resolve()

    @classmethod
    def tearDownClass(cls):
        assert cls.work.resolve().parent == ROOT.resolve()
        cls.temporary.cleanup()

    def test_all_answers_and_fields_match_model(self):
        actual = self.processor.predict(self.documents)
        self.assertEqual(actual, self.answers)
        self.assertEqual(len(actual), 2000)
        self.assertEqual(len({row['doc_id'] for row in actual}), 2000)
        for row in actual:
            self.assertIn(row['decision'], ['accept', 'reject'])
            if row['decision'] == 'accept':
                self.assertEqual(set(row['fields']), set(FIELDS))
                self.assertIsNone(validity_reason(row['fields']))
            else:
                self.assertNotIn('fields', row)

    def test_batching_does_not_change_decisions(self):
        accepted_ids = {row['doc_id'] for row in self.answers if row['decision'] == 'accept'}
        selected = [doc for doc in self.documents if doc['doc_id'] in accepted_ids]
        selected += self.documents[:40]
        expected = self.processor.predict(selected)
        actual = []
        for start in range(0, len(selected), 7):
            actual.extend(self.processor.predict(selected[start:start + 7]))
        self.assertEqual(actual, expected)

    def test_unknown_source_and_malformed_metadata_are_rejected(self):
        doc = copy.deepcopy(self.documents[0])
        doc['source'] = 'new_capture_source'
        self.assertEqual(self.processor.predict([doc])[0]['decision'], 'reject')
        doc['source'] = 'web'
        doc['quality']['glare'] = float('nan')
        self.assertEqual(self.processor.predict([doc])[0]['decision'], 'reject')
        doc = copy.deepcopy(self.documents[0])
        doc['viz']['Birth_date'] = 'invalid'
        self.assertEqual(self.processor.predict([doc])[0]['decision'], 'reject')

    def test_duplicate_ids_leave_existing_output_untouched(self):
        source = self.work / 'duplicate.jsonl'
        target = self.work / 'existing.jsonl'
        text = json.dumps(self.documents[0], ensure_ascii=False)
        source.write_text(text + '\n' + text + '\n', encoding='utf-8')
        target.write_text('keep-existing-output', encoding='utf-8')
        with self.assertRaises(ValueError):
            process_file(source, target)
        self.assertEqual(target.read_text(encoding='utf-8'), 'keep-existing-output')
        with self.assertRaises(ValueError):
            process_file(source, source)

    def test_cli_needs_no_labels_and_makes_no_network_connections(self):
        isolated = self.work / 'isolated'
        isolated.mkdir()
        for name in ['postprocess.py', 'passport_ocr.py']:
            shutil.copy2(ROOT / name, isolated / name)
        (isolated / 'artifacts').mkdir()
        (isolated / 'reference').mkdir()
        shutil.copy2(ROOT / 'artifacts/models.joblib', isolated / 'artifacts/models.joblib')
        shutil.copy2(ROOT / 'reference/subdivisions.csv', isolated / 'reference/subdivisions.csv')
        shutil.copy2(ROOT / 'data/test_predictions.jsonl', isolated / 'input.jsonl')
        command = '''import runpy, socket, sys
def blocked(*args, **kwargs):
    raise RuntimeError('Network access is disabled in the offline test')
socket.create_connection = blocked
socket.socket.connect = blocked
sys.path.insert(0, sys.argv[1])
script = sys.argv[1] + '/postprocess.py'
sys.argv = [script, sys.argv[1] + '/input.jsonl', sys.argv[1] + '/output.jsonl']
runpy.run_path(script, run_name='__main__')'''
        result = subprocess.run(
            [sys.executable, '-c', command, str(isolated)],
            cwd=self.work, capture_output=True, text=True, encoding='utf-8', timeout=180
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_jsonl(isolated / 'output.jsonl'), self.answers)
        self.assertFalse((isolated / 'data/train_labels.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
