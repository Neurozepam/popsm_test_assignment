import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
import train
from passport_ocr import FIELDS, feature_table, load_reference, norm, read_zones, validity_reason, visual_fields, wrong_fields

data = train.prepare_data(root)
validation_manifest = json.loads((root / 'artifacts/validation_manifest.json').read_text(encoding='utf-8'))
if validation_manifest['input_sha256'] != train.hashes(root):
    raise ValueError('Validation cache differs from inputs; run python train.py --validate')
labels = data['labels']
documents = data['documents']
candidates = data['candidates']
reference, words = load_reference(root / 'reference/subdivisions.csv')
raw = [visual_fields(doc) for doc in documents]
zones = [read_zones(doc, reference) for doc in documents]
recovery = []
for name, fields in [
    ('OCR: объединение строк', raw),
    ('MRZ и форматы', [candidate['fields'] for candidate in zones]),
    ('Полная постобработка', [candidate['fields'] for candidate in candidates])
]:
    exact = safe = 0
    for doc, predicted in zip(documents, fields):
        expected = labels[doc['doc_id']]
        correct = not wrong_fields(predicted, expected)
        exact += correct
        safe += correct and validity_reason(expected) is None
    recovery.append({'stage': name, 'exact_fields': int(exact), 'safe': int(safe)})
pd.DataFrame(recovery).to_csv(root / 'artifacts/recovery_summary.csv', index=False)

field_rows = []
for field in FIELDS:
    before = after = 0
    for doc, original, candidate in zip(documents, raw, candidates):
        expected = labels[doc['doc_id']][field]
        before += norm(original[field]) != norm(expected)
        after += norm(candidate['fields'][field]) != norm(expected)
    field_rows.append({'field': field, 'before': before, 'after': after})
pd.DataFrame(field_rows).to_csv(root / 'artifacts/field_errors.csv', index=False)

evidence = []
for c in candidates:
    expected = labels[c['doc_id']]
    for field in FIELDS[:3]:
        if field in c['supported'] and field not in c['changed']:
            if norm(c['fields'][field]) != norm(expected[field]):
                evidence.append({
                    'doc_id': c['doc_id'], 'source': c['source'], 'field': field,
                    'viz_and_mrz': c['fields'][field], 'label': expected[field],
                    'viz_char_min': c['stats'][field]['char_min'],
                    'mrz_char_min': min(c['mrz_lines'][0]['char_conf'], default=0)
                })
pd.DataFrame(evidence).to_csv(root / 'artifacts/zone_label_disagreements.csv', index=False)

strict_oof = pd.read_csv(root / 'artifacts/strict_oof.csv')
strict_oof['FP'] = strict_oof['accept'] & (strict_oof['safe'] == 0)
strict_oof['FN'] = ~strict_oof['accept'] & (strict_oof['safe'] == 1)
strict = strict_oof.groupby(['seed', 'members']).agg(
    n=('doc_id', 'size'), accepted=('accept', 'sum'), FP=('FP', 'sum'), FN=('FN', 'sum')
).reset_index()
strict['coverage'] = strict['accepted'] / strict['n']
strict.to_csv(root / 'artifacts/strict_summary.csv', index=False)

economic_oof = pd.read_csv(root / 'artifacts/economic_oof.csv')
economic_oof['FP'] = economic_oof['accept'] & (economic_oof['safe'] == 0)
economic_oof['FN'] = ~economic_oof['accept'] & (economic_oof['safe'] == 1)
economic = economic_oof.groupby('ratio').agg(
    n=('doc_id', 'size'), accepted=('accept', 'sum'), FP=('FP', 'sum'), FN=('FN', 'sum')
).reset_index()
economic['cost'] = economic['ratio'] * economic['FP'] + economic['FN']
economic['coverage'] = economic['accepted'] / economic['n']
economic['risk'] = economic['FP'] / economic['accepted']
economic['reject_all_cost'] = int(data['document_targets']['field'].sum())
economic.to_csv(root / 'artifacts/economic_summary.csv', index=False)

main = strict_oof[(strict_oof['seed'] == 42) & (strict_oof['members'] == 9)]
by_source = main.groupby('source').agg(accepted=('accept', 'sum'), FP=('FP', 'sum'))
answers = json.loads('[]')
with open(root / 'answers.jsonl', encoding='utf-8') as file:
    answers = [json.loads(line) for line in file if line.strip()]
accepted_ids = {row['doc_id'] for row in answers if row['decision'] == 'accept'}
test = train.read_jsonl(root / 'data/test_predictions.jsonl')
rng = np.random.default_rng(42)
simulated = np.zeros(20000, dtype=int)
forecast_rows = []
for source in ['mobile', 'mobile_v2', 'web']:
    count = sum(doc['source'] == source and doc['doc_id'] in accepted_ids for doc in test)
    n = int(by_source.loc[source, 'accepted'])
    bad = int(by_source.loc[source, 'FP'])
    a, b = bad + .5, n - bad + .5
    simulated += rng.binomial(count, rng.beta(a, b, size=len(simulated)))
    forecast_rows.append({'source': source, 'test_accepted': count, 'OOF_accepted': n, 'OOF_FP': bad, 'expected_errors': count * a / (a + b)})
forecast = pd.DataFrame(forecast_rows)
forecast.to_csv(root / 'artifacts/strict_forecast.csv', index=False)
primary = strict[(strict['seed'] == 42) & (strict['members'] == 9)].iloc[0]
baseline_safe = recovery[1]['safe']
new_safe = recovery[2]['safe']
summary = {
    'input_sha256': train.hashes(root),
    'train_documents': len(documents), 'test_documents': len(test),
    'unique_passports': len(set(data['groups'])),
    'feature_columns': len(data['tables']['field'].columns),
    'safe_before': baseline_safe, 'safe_after': new_safe,
    'repaired_documents': sum(train.wrong_fields(a['fields'], labels[a['doc_id']]) != [] and train.wrong_fields(b['fields'], labels[b['doc_id']]) == [] for a,b in zip(zones,candidates)),
    'damaged_documents': sum(train.wrong_fields(a['fields'], labels[a['doc_id']]) == [] and train.wrong_fields(b['fields'], labels[b['doc_id']]) != [] for a,b in zip(zones,candidates)),
    'zone_label_disagreements': len(evidence),
    'strict_primary_accepted': int(primary['accepted']),
    'strict_primary_FP': int(primary['FP']),
    'strict_primary_upper95': float(beta.ppf(.95, int(primary['FP']) + 1, int(primary['accepted']) - int(primary['FP']))),
    'strict_test_accepted': len(accepted_ids),
    'strict_expected_errors': float(forecast['expected_errors'].sum()),
    'strict_modal_errors': int(np.bincount(simulated).argmax()),
    'strict_prediction_interval95': np.quantile(simulated, [.025,.975]).astype(int).tolist(),
    'validation': 'Grouped internal CV; all design iterations use the supplied train, no independent labeled test.',
    'forecast_assumption': 'Transfer of risk within capture source; beta-binomial approximation, not a guarantee.'
}
(root / 'artifacts/research_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
manifest_path = root / 'artifacts/model_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['model_sha256'] = hashlib.sha256((root / 'artifacts/models.joblib').read_bytes()).hexdigest()
manifest['release_source_sha256'] = {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in ['passport_ocr.py', 'postprocess.py', 'train.py', 'analyze.py']
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

with open(root / 'artifacts/oof_strict_answers.jsonl', 'w', encoding='utf-8') as file:
    candidate_by_id = {candidate['doc_id']: candidate for candidate in candidates}
    for row in main.itertuples():
        answer = {'doc_id': row.doc_id, 'decision': 'accept' if row.accept else 'reject'}
        if row.accept:
            answer['fields'] = candidate_by_id[row.doc_id]['fields']
        file.write(json.dumps(answer, ensure_ascii=False) + '\n')

from postprocess import PassportProcessor
processor = PassportProcessor()
policy_rows = []
for policy in ['strict', 'cost5', 'cost10', 'cost25']:
    result = processor.predict(test, policy)
    if policy == 'strict' and result != answers:
        raise ValueError('answers.jsonl differs from the final model; regenerate answers first')
    policy_rows.append({'policy': policy, 'documents': len(result), 'accepted': sum(row['decision'] == 'accept' for row in result)})
pd.DataFrame(policy_rows).to_csv(root / 'artifacts/test_policy_summary.csv', index=False)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(economic.to_string(index=False))
print(strict.to_string(index=False))
