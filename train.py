"""Reproduce final models and optional grouped validation without external services."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from passport_ocr import (
    AS_OF, FIELDS, aggregate_scores, document_features, feature_table,
    norm, read_jsonl, recover_documents, validity_reason, wrong_fields
)

SEED = 42
COST_RATIOS = [5, 10, 25]
PARTITION_SEEDS = [142, 143, 144]


def prepare_data(root):
    documents = read_jsonl(root / 'data/train_predictions.jsonl')
    label_rows = read_jsonl(root / 'data/train_labels.jsonl')
    labels = {row['doc_id']: row['fields'] for row in label_rows}
    ids = [row['doc_id'] for row in documents]
    if set(ids) != set(labels):
        raise ValueError('Training documents and labels have different IDs')
    candidates = recover_documents(documents, root / 'reference/subdivisions.csv')
    safe = []
    field_targets = []
    for candidate in candidates:
        expected = labels[candidate['doc_id']]
        correct = not wrong_fields(candidate['fields'], expected)
        safe.append(int(correct and validity_reason(expected) is None))
        for field in FIELDS:
            field_targets.append(int(norm(candidate['fields'][field]) == norm(expected[field])))
    y = np.array(safe)
    allowed = np.array([not candidate['reasons'] for candidate in candidates])
    field_table = feature_table(candidates)
    doc_table = pd.DataFrame([document_features(candidate) for candidate in candidates])
    return {
        'documents': documents, 'labels': labels, 'candidates': candidates,
        'tables': {'doc': doc_table, 'field': field_table},
        'targets': {'doc': y, 'field': np.array(field_targets)},
        'document_targets': {'doc': y, 'field': y},
        'allowed': {'doc': allowed, 'field': allowed},
        'groups': np.array([''.join(labels[doc_id]['Licence_number'].split()) for doc_id in ids])
    }


def make_risk_pipeline(table, model):
    categorical = [name for name in ['source', 'field'] if name in table.columns]
    numeric = [name for name in table.columns if name not in categorical]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])
    return Pipeline([('preprocess', preprocess), ('model', model)])

def expanded_indices(indices, kind):
    if kind != 'field':
        return indices
    rows = []
    for index in indices:
        rows.extend(range(index * len(FIELDS), (index + 1) * len(FIELDS)))
    return np.array(rows)

def threshold_frontier(scores, targets, allowed):
    table = pd.DataFrame({'score': scores[allowed], 'safe': targets[allowed]})
    table = table.groupby('score')['safe'].agg(['size', 'sum'])
    table = table.sort_index(ascending=False)
    table['TP'] = table['sum'].cumsum()
    table['FP'] = table['size'].cumsum() - table['TP']
    table['FN'] = int(targets.sum()) - table['TP']
    table = table.reset_index().rename(columns={'score': 'threshold'})
    table = table[['threshold', 'TP', 'FP', 'FN']]
    reject_all = pd.DataFrame([{
        'threshold': float(np.nextafter(scores.max(), np.inf)),
        'TP': 0, 'FP': 0, 'FN': int(targets.sum())
    }])
    return pd.concat([reject_all, table], ignore_index=True)

def choose_economic_threshold(frontier, ratio):
    options = frontier.copy()
    options['cost'] = ratio * options['FP'] + options['FN']
    options = options.sort_values(['cost', 'FP', 'FN'], kind='stable')
    best = options.iloc[0]
    return {
        'threshold': float(best['threshold']),
        'cal_TP': int(best['TP']),
        'cal_FP': int(best['FP']),
        'cal_FN': int(best['FN']),
        'cal_cost': int(best['cost'])
    }

def economic_metrics(targets, accepted, ratio):
    tp = int(((targets == 1) & accepted).sum())
    fp = int(((targets == 0) & accepted).sum())
    fn = int(((targets == 1) & ~accepted).sum())
    tn = int(((targets == 0) & ~accepted).sum())
    accepted_count = tp + fp
    return {
        'n': len(targets), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'accepted': accepted_count,
        'coverage': accepted_count / len(targets),
        'risk': fp / accepted_count if accepted_count else np.nan,
        'cost': ratio * fp + fn,
        'cost_per_document': (ratio * fp + fn) / len(targets)
    }

def economic_model_grid():
    trials = []
    for kind in ['doc', 'field']:
        for strength in [0.1, 1.0, 10.0]:
            trials.append({
                'name': f'{kind}_logistic_C{strength}', 'kind': kind,
                'model': LogisticRegression(C=strength, max_iter=3000, random_state=SEED)
            })
        for depth, leaf in [(6, 20), (10, 10), (None, 5)]:
            trials.append({
                'name': f'{kind}_forest_depth{depth}_leaf{leaf}', 'kind': kind,
                'model': RandomForestClassifier(
                    n_estimators=120, max_depth=depth, min_samples_leaf=leaf,
                    max_features=0.8, random_state=SEED, n_jobs=1
                )
            })
        for leaves, penalty in [(7, 5), (15, 2), (31, 1)]:
            trials.append({
                'name': f'{kind}_hist_leaves{leaves}_l2_{penalty}', 'kind': kind,
                'model': HistGradientBoostingClassifier(
                    max_iter=150, max_leaf_nodes=leaves, min_samples_leaf=50,
                    l2_regularization=penalty, random_state=SEED
                )
            })
    return trials

def fit_economic_models(data, training_indices, calibration_indices):
    tables = data['tables']
    targets = data['targets']
    document_targets = data['document_targets']
    allowed_documents = data['allowed']
    best = {}
    trials = []
    for trial in economic_model_grid():
        kind = trial['kind']
        fit_rows = expanded_indices(training_indices, kind)
        cal_rows = expanded_indices(calibration_indices, kind)
        pipeline = make_risk_pipeline(tables[kind], trial['model'])
        pipeline.fit(tables[kind].iloc[fit_rows], targets[kind][fit_rows])
        raw_scores = pipeline.predict_proba(tables[kind].iloc[cal_rows])[:, 1]
        aggregations = ['minimum', 'product'] if kind == 'field' else ['document']
        for aggregation in aggregations:
            scores = aggregate_scores(raw_scores, aggregation)
            target = document_targets[kind][calibration_indices]
            allowed = allowed_documents[kind][calibration_indices]
            frontier = threshold_frontier(scores, target, allowed)
            for ratio in COST_RATIOS:
                tuning = choose_economic_threshold(frontier, ratio)
                candidate = {
                    'ratio': ratio, 'policy': trial['name'] + ':' + aggregation,
                    'kind': kind, 'aggregation': aggregation,
                    'pipeline': pipeline, **tuning
                }
                trials.append({
                    key: value for key, value in candidate.items()
                    if key != 'pipeline'
                })
                key = (candidate['cal_cost'], candidate['cal_FP'], candidate['cal_FN'])
                if ratio not in best or key < best[ratio]['selection_key']:
                    candidate['selection_key'] = key
                    best[ratio] = candidate
    return best, trials

def strict_score_threshold(scores, targets, allowed):
    bad_scores = scores[(targets == 0) & allowed]
    if len(bad_scores) == 0:
        return 0.0
    return float(bad_scores.max() + 1e-12)

def fit_strict_ensemble(data, document_indices, partition_seeds=PARTITION_SEEDS, feature_table=None):
    X_field = data['tables']['field']
    y_field = data['targets']['field']
    y_v2 = data['document_targets']['field']
    allowed_v2 = data['allowed']['field']
    groups = data['groups']
    if feature_table is None:
        feature_table = X_field
    members = []
    calibration_rows = []
    for partition_seed in partition_seeds:
        calibration_cv = GroupKFold(
            n_splits=3, shuffle=True, random_state=partition_seed
        )
        for fold, (fit_local, cal_local) in enumerate(
            calibration_cv.split(document_indices, groups=groups[document_indices]), 1
        ):
            training = document_indices[fit_local]
            calibration = document_indices[cal_local]
            assert not set(groups[training]) & set(groups[calibration])
            model = RandomForestClassifier(
                n_estimators=160, max_depth=10, min_samples_leaf=10,
                max_features=0.8, random_state=SEED, n_jobs=min(4, os.cpu_count() or 1)
            )
            pipeline = make_risk_pipeline(feature_table, model)
            fit_rows = expanded_indices(training, 'field')
            cal_rows = expanded_indices(calibration, 'field')
            pipeline.fit(feature_table.iloc[fit_rows], y_field[fit_rows])
            raw = pipeline.predict_proba(feature_table.iloc[cal_rows])[:, 1]
            scores = aggregate_scores(raw, 'product')
            threshold = strict_score_threshold(scores, y_v2[calibration], allowed_v2[calibration])
            accepted = allowed_v2[calibration] & (scores >= threshold)
            assert not (accepted & (y_v2[calibration] == 0)).any()
            members.append({'pipeline': pipeline, 'threshold': threshold})
            calibration_rows.append({
                'partition_seed': partition_seed, 'fold': fold,
                'train_documents': len(training), 'cal_documents': len(calibration),
                'threshold': threshold, 'cal_accepted': int(accepted.sum())
            })
    return members, calibration_rows

def predict_member(member, table, aggregation):
    raw = member['pipeline'].predict_proba(table)[:, 1]
    return aggregate_scores(raw, aggregation) >= member['threshold']


def predict_ensemble(members, table, allowed, aggregation='product'):
    return allowed & np.logical_and.reduce([
        predict_member(member, table, aggregation) for member in members
    ])


def hashes(root):
    paths = [
        'data/train_predictions.jsonl', 'data/train_labels.jsonl',
        'data/test_predictions.jsonl', 'reference/subdivisions.csv'
    ]
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def train_final(root, data):
    indices = np.arange(len(data['documents']))
    groups = data['groups']
    remaining, audit = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED
    ).split(indices, groups=groups))
    fit_local, cal_local = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=SEED + 1
    ).split(remaining, groups=groups[remaining]))
    training, calibration = remaining[fit_local], remaining[cal_local]
    print('Fit final strict ensemble', flush=True)
    members, calibration_rows = fit_strict_ensemble(data, indices)
    print('Tune final economic policies', flush=True)
    economic, trials = fit_economic_models(data, training, calibration)
    policies = {
        'strict': {
            'aggregation': 'product',
            'members': [{'model': row['pipeline'], 'threshold': row['threshold']} for row in members]
        }
    }
    selected = []
    for ratio, candidate in economic.items():
        if candidate['kind'] != 'field':
            raise ValueError('A document model wins on changed data; extend inference before exporting')
        policies[f'cost{ratio}'] = {
            'aggregation': candidate['aggregation'],
            'members': [{'model': candidate['pipeline'], 'threshold': candidate['threshold']}]
        }
        selected.append({key: value for key, value in candidate.items() if key != 'pipeline'})
    bundle = {
        'format_version': 1, 'as_of': AS_OF.isoformat(), 'fields': FIELDS,
        'feature_columns': list(data['tables']['field'].columns), 'policies': policies
    }
    artifacts = root / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    temporary_model = artifacts / 'models.joblib.tmp'
    joblib.dump(bundle, temporary_model, compress=3)
    os.replace(temporary_model, artifacts / 'models.joblib')
    manifest = {
        'as_of': AS_OF.isoformat(), 'seed': SEED,
        'strict_partition_seeds': PARTITION_SEEDS,
        'training_documents': len(indices),
        'unique_passports': len(set(groups)),
        'economic_train_documents': len(training),
        'economic_calibration_documents': len(calibration),
        'versions': {
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
            'joblib': joblib.__version__
        },
        'input_sha256': hashes(root),
        'model_sha256': hashlib.sha256((artifacts / 'models.joblib').read_bytes()).hexdigest(),
        'source_sha256': {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ['passport_ocr.py', 'postprocess.py', 'train.py']
        },
        'economic_selected': selected,
        'strict_calibration': calibration_rows,
        'caveat': 'Internal grouped CV on the supplied train; no independent labeled test is available.'
    }
    (artifacts / 'model_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    pd.DataFrame(trials).to_csv(artifacts / 'final_tuning.csv', index=False)
    return bundle


def validate(root, data):
    all_indices = np.arange(len(data['documents']))
    groups = data['groups']
    y = data['document_targets']['field']
    allowed = data['allowed']['field']
    strict_rows, economic_rows, tuning_rows = [], [], []
    for seed in [42, 123, 2026]:
        cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (remaining, check) in enumerate(cv.split(all_indices, groups=groups), 1):
            members, _ = fit_strict_ensemble(data, remaining)
            check_table = data['tables']['field'].iloc[expanded_indices(check, 'field')]
            for count in [3, 6, 9]:
                accepted = predict_ensemble(members[:count], check_table, allowed[check])
                for position, index in enumerate(check):
                    strict_rows.append({
                        'seed': seed, 'fold': fold, 'members': count,
                        'doc_id': data['documents'][index]['doc_id'],
                        'source': data['documents'][index]['source'],
                        'safe': int(y[index]), 'accept': bool(accepted[position])
                    })
            if seed == 42:
                inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + fold)
                fit_local, cal_local = next(inner.split(remaining, groups=groups[remaining]))
                choices, trials = fit_economic_models(data, remaining[fit_local], remaining[cal_local])
                tuning_rows.extend([{'fold': fold, **row} for row in trials])
                for ratio, chosen in choices.items():
                    kind = chosen['kind']
                    table = data['tables'][kind].iloc[expanded_indices(check, kind)]
                    accepted = allowed[check] & predict_member(chosen, table, chosen['aggregation'])
                    for position, index in enumerate(check):
                        economic_rows.append({
                            'ratio': ratio, 'fold': fold,
                            'doc_id': data['documents'][index]['doc_id'],
                            'source': data['documents'][index]['source'],
                            'safe': int(y[index]), 'accept': bool(accepted[position]),
                            'policy': chosen['policy']
                        })
            print('Validate', seed, fold, flush=True)
    artifacts = root / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    pd.DataFrame(strict_rows).to_csv(artifacts / 'strict_oof.csv', index=False)
    pd.DataFrame(economic_rows).to_csv(artifacts / 'economic_oof.csv', index=False)
    pd.DataFrame(tuning_rows).to_csv(artifacts / 'cv_tuning.csv', index=False)
    (artifacts / 'validation_manifest.json').write_text(json.dumps({
        'input_sha256': hashes(root),
        'seed': SEED,
        'strict_outer_seeds': [42, 123, 2026],
        'strict_partition_seeds': PARTITION_SEEDS,
        'economic_cost_ratios': COST_RATIOS,
        'scikit_learn': sklearn.__version__
    }, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true', help='Recompute expensive grouped CV before final fitting')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    started = time.perf_counter()
    data = prepare_data(root)
    if args.validate:
        validate(root, data)
    train_final(root, data)
    if args.validate:
        from postprocess import process_file
        process_file(root / 'data/test_predictions.jsonl', root / 'answers.jsonl')
        subprocess.run([sys.executable, str(root / 'analyze.py')], check=True)
    print('Finished in', round(time.perf_counter() - started, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()

