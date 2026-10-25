"""CPU-only passport OCR postprocessing with bundled trained models."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np

from passport_ocr import (
    AS_OF, FIELDS, aggregate_scores, feature_table, load_reference,
    read_jsonl, recover_document, valid_input
)

POLICIES = ['strict', 'cost5', 'cost10', 'cost25']
ROOT = Path(__file__).resolve().parent


class PassportProcessor:
    def __init__(self, model_path=None, reference_path=None):
        model_path = Path(model_path or ROOT / 'artifacts/models.joblib')
        reference_path = Path(reference_path or ROOT / 'reference/subdivisions.csv')
        self.bundle = joblib.load(model_path)
        if self.bundle.get('format_version') != 1:
            raise ValueError('Unsupported model format')
        if self.bundle.get('as_of') != AS_OF.isoformat() or self.bundle.get('fields') != FIELDS:
            raise ValueError('Model schema/date differs from the postprocessor')
        self.reference, self.words = load_reference(reference_path)

    def predict(self, documents, policy='strict'):
        if policy not in self.bundle['policies']:
            raise ValueError(f'Unknown policy: {policy}')
        answers = [{'doc_id': doc['doc_id'], 'decision': 'reject'} for doc in documents]
        candidates = []
        positions = []
        for position, doc in enumerate(documents):
            if not valid_input(doc):
                continue
            try:
                candidate = recover_document(doc, self.reference, self.words)
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                continue
            if not candidate['reasons']:
                candidates.append(candidate)
                positions.append(position)
        if not candidates:
            return answers
        table = feature_table(candidates)
        if list(table.columns) != self.bundle['feature_columns']:
            raise ValueError('Feature schema differs from the trained model')
        settings = self.bundle['policies'][policy]
        votes = []
        for member in settings['members']:
            raw = member['model'].predict_proba(table)[:, 1]
            scores = aggregate_scores(raw, settings['aggregation'])
            votes.append(scores >= member['threshold'])
        accepted = np.logical_and.reduce(votes)
        for index, passed in enumerate(accepted):
            if passed:
                answers[positions[index]] = {
                    'doc_id': candidates[index]['doc_id'],
                    'decision': 'accept',
                    'fields': candidates[index]['fields']
                }
        return answers


def process_file(input_path, output_path, policy='strict', batch_size=512, model_path=None, reference_path=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output must be different files')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    documents = read_jsonl(input_path)
    processor = PassportProcessor(model_path, reference_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='\n',
            prefix='.answers-', suffix='.jsonl', dir=output_path.parent, delete=False
        ) as file:
            temporary_path = Path(file.name)
            for start in range(0, len(documents), batch_size):
                batch = documents[start:start + batch_size]
                for answer in processor.predict(batch, policy):
                    accepted += answer['decision'] == 'accept'
                    file.write(json.dumps(answer, ensure_ascii=False) + '\n')
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {'documents': len(documents), 'accepted': accepted, 'rejected': len(documents) - accepted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--policy', choices=POLICIES, default='strict')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    try:
        stats = process_file(args.input, args.output, args.policy, args.batch_size, args.model, args.reference)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({'policy': args.policy, **stats}, ensure_ascii=False), file=sys.stderr)


if __name__ == '__main__':
    main()
