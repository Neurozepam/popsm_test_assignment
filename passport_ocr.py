"""Shared OCR recovery and feature extraction used in research and inference."""

import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path
import re

import numpy as np
import pandas as pd

FIELDS = [
    'Last_name_ru', 'First_name_ru', 'Middle_name_ru', 'Sex_ru',
    'Birth_date', 'Birth_place_ru', 'Issue_date',
    'Issue_organisation_code', 'Licence_number', 'Issue_organization_ru'
]
AS_OF = date(2026, 10, 1)
CYR = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
LAT = 'ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678'
TRANSLIT = dict(zip(CYR, LAT))
REVERSE = dict(zip(LAT, CYR))
LOOKALIKES = str.maketrans('ABCEHKMOPTXY', 'АВСЕНКМОРТХУ')
TEXT_FIELDS = [field for field in FIELDS if field.endswith('_ru')]


def norm(value):
    return ' '.join(str(value or '').split()).upper()

def visual_fields(doc):
    result = {}
    for field in FIELDS:
        value = doc['viz'][field] or {}
        texts = [line['text'] for line in value.get('lines', [])]
        result[field] = norm(' '.join(texts))
    return result

def wrong_fields(predicted, expected):
    errors = []
    for field in FIELDS:
        if norm(predicted[field]) != norm(expected[field]):
            errors.append(field)
    return errors

def parse_date(text):
    if not re.fullmatch(r'\d{2}\.\d{2}\.\d{4}', text):
        return None
    try:
        day, month, year = map(int, text.split('.'))
        return date(year, month, day)
    except ValueError:
        return None

def birthday(birth, age):
    try:
        return birth.replace(year=birth.year + age)
    except ValueError:
        return date(birth.year + age, 2, 28)

def validity_reason(fields, as_of=AS_OF):
    birth = parse_date(fields['Birth_date'])
    issued = parse_date(fields['Issue_date'])
    if birth is None or issued is None:
        return 'invalid_date'
    if birth < date(1900, 1, 1) or birth > as_of:
        return 'birth_out_of_range'
    if issued < birthday(birth, 14) or issued > as_of:
        return 'issue_date_or_age'
    for age in (20, 45):
        renewal = birthday(birth, age)
        if issued < renewal and as_of > renewal + timedelta(days=90):
            return f'expired_{age}'
    return None

def format_reasons(fields):
    reasons = []
    for field in FIELDS:
        if not fields[field] and field != 'Middle_name_ru':
            reasons.append('missing:' + field)
    if fields['Sex_ru'] not in ['МУЖ.', 'ЖЕН.']:
        reasons.append('sex_format')
    if not re.fullmatch(r'\d{3}-\d{3}', fields['Issue_organisation_code']):
        reasons.append('code_format')
    if not re.fullmatch(r'\d{2} \d{2} \d{6}', fields['Licence_number']):
        reasons.append('number_format')
    reason = validity_reason(fields)
    if reason:
        reasons.append(reason)
    return reasons

def translit(text):
    result = []
    for char in norm(text):
        if char in ' -':
            result.append('<')
        else:
            result.append(TRANSLIT.get(char, char))
    return ''.join(result)

def check_digit(text):
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    total = 0
    for index, char in enumerate(text):
        value = 0 if char == '<' else alphabet.index(char)
        total += value * [7, 3, 1][index % 3]
    return str(total % 10)

def lower_mrz_is_valid(text):
    if not re.fullmatch(r'[0-9A-Z<]{44}', text):
        return False
    if text[10:13] != 'RUS' or text[20] not in 'MF':
        return False
    if text[21:28] not in ['<<<<<<<', '<<<<<<0'] or text[41] != '<':
        return False
    if not text[:9].isdigit() or not text[13:19].isdigit() or not text[28:41].isdigit():
        return False
    checks = [
        (text[:9], text[9]),
        (text[13:19], text[19]),
        (text[21:27], text[27]),
        (text[28:42], text[42]),
        (text[:10] + text[13:20] + text[21:43], text[43])
    ]
    for value, digit in checks:
        expected = '0' if digit == '<' else digit
        if check_digit(value) != expected:
            return False
    return True

def decode_lower_mrz(text, visual):
    digits = text[:3] + text[28] + text[3:9]
    issue = text[29:35]
    issue_year = int(issue[:2])
    issue_year += 2000 if issue_year <= 26 else 1900
    fields = {
        'Licence_number': f'{digits[:2]} {digits[2:4]} {digits[4:]}',
        'Issue_organisation_code': text[35:38] + '-' + text[38:41],
        'Issue_date': f'{issue[4:6]}.{issue[2:4]}.{issue_year}',
        'Sex_ru': 'МУЖ.' if text[20] == 'M' else 'ЖЕН.'
    }
    birth = text[13:19]
    years = []
    for century in [1900, 2000]:
        year = century + int(birth[:2])
        candidate = f'{birth[4:6]}.{birth[2:4]}.{year}'
        parsed = parse_date(candidate)
        if parsed and date(1900, 1, 1) <= parsed <= AS_OF:
            if birthday(parsed, 14) <= AS_OF:
                years.append(year)
    visual_birth = parse_date(visual['Birth_date'])
    if visual_birth and visual_birth.year in years:
        suffix = f'{visual_birth.year % 100:02d}{visual_birth.month:02d}{visual_birth.day:02d}'
        if suffix == birth:
            years = [visual_birth.year]
    if len(years) == 1:
        fields['Birth_date'] = f'{birth[4:6]}.{birth[2:4]}.{years[0]}'
    return fields

def decode_upper_mrz(text):
    if len(text) != 44 or not text.startswith('PNRUS') or not text.endswith('<'):
        return None
    parts = text[5:].split('<<', 1)
    if len(parts) != 2:
        return None
    names = [parts[0]] + parts[1].rstrip('<').split('<')
    if len(names) not in [2, 3]:
        return None
    if len(names) == 2:
        names.append('')
    for name in names:
        if any(char not in REVERSE for char in name):
            return None
    return dict(zip(FIELDS[:3], names))

def field_stats(doc, field):
    value = doc['viz'][field] or {}
    lines = value.get('lines', [])
    chars = []
    line_height = 0
    alignment_ok = True
    for line in lines:
        chars.extend(line.get('char_conf', []))
        line_height += line['box'][3]
        if len(line.get('char_conf', [])) != len(line['text']):
            alignment_ok = False
    height = value.get('box', [0, 0, 0, 0])[3]
    ratio = height / line_height if line_height else 10.0
    return {
        'conf': value.get('conf', 0.0),
        'char_min': min(chars, default=0.0),
        'char_mean': float(np.mean(chars)) if chars else 0.0,
        'saturated': sum(char == 1 for char in chars) / max(1, len(chars)),
        'height_ratio': min(ratio, 10.0),
        'line_count': len(lines),
        'alignment_ok': alignment_ok
    }

def reference_features(text, unit_code, reference):
    options = reference.get(unit_code, [])
    compact = re.sub(r'\s+', '', text)
    scores = []
    for name in options:
        target = re.sub(r'\s+', '', name)
        score = SequenceMatcher(None, compact, target, autojunk=False).ratio()
        scores.append(score)
    scores = sorted(scores, reverse=True)
    best = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else 0.0
    return {
        'reference_known_code': bool(options),
        'reference_exact_name': text in options,
        'reference_similarity': best,
        'reference_margin': best - second
    }

def read_zones(doc, reference):
    fields = visual_fields(doc)
    stats = {field: field_stats(doc, field) for field in FIELDS}
    supported = set()
    changed = []
    conflicts = []
    lines = (doc.get('mrz') or {}).get('lines', [])
    mrz_valid = False

    if len(lines) == 2:
        upper = lines[0]['text'].upper()
        lower = lines[1]['text'].upper()
        mrz_valid = lower_mrz_is_valid(lower)
        if mrz_valid:
            lower_fields = decode_lower_mrz(lower, fields)
            chars = lines[1].get('char_conf', [])
            for field, value in lower_fields.items():
                if fields[field] == value:
                    supported.add(field)
                elif len(chars) == 44 and min(chars) >= 0.97:
                    fields[field] = value
                    changed.append(field)
                    supported.add(field)
                else:
                    conflicts.append(field)
        upper_fields = decode_upper_mrz(upper)
        if upper_fields is not None:
            chars = lines[0].get('char_conf', [])
            for field, encoded in upper_fields.items():
                if translit(fields[field]) == encoded:
                    supported.add(field)
                elif encoded and len(chars) == 44 and min(chars) >= 0.99:
                    fields[field] = ''.join(REVERSE[char] for char in encoded)
                    changed.append(field)
                    supported.add(field)
                else:
                    conflicts.append(field)

    for field, length in [('Issue_organisation_code', 6), ('Licence_number', 10)]:
        digits = re.sub(r'[\s-]', '', fields[field])
        if digits.isdigit() and len(digits) == length:
            if field == 'Licence_number':
                fields[field] = f'{digits[:2]} {digits[2:4]} {digits[4:]}'
            else:
                fields[field] = digits[:3] + '-' + digits[3:]

    reasons = format_reasons(fields)
    if conflicts:
        reasons.append('mrz_conflict')
    for field in FIELDS:
        if field in supported:
            continue
        if field == 'Middle_name_ru' and not fields[field]:
            reasons.append('unconfirmed_empty_middle_name')
            continue
        if stats[field]['height_ratio'] > 1.4:
            reasons.append('incomplete_lines:' + field)
        if not stats[field]['alignment_ok']:
            reasons.append('char_alignment:' + field)

    return {
        'doc_id': doc['doc_id'], 'source': doc['source'],
        'fields': fields, 'stats': stats, 'supported': supported,
        'changed': changed, 'conflicts': conflicts, 'mrz_valid': mrz_valid,
        'mrz_lines': lines, 'quality': doc['quality'],
        'reference': reference_features(fields['Issue_organization_ru'], fields['Issue_organisation_code'], reference),
        'reasons': sorted(set(reasons))
    }

def normalize_mixed_word(match):
    word = match.group()
    latin = re.findall('[A-Z]', word)
    cyrillic = re.search('[А-ЯЁ]', word)
    if latin and cyrillic and all(char in 'ABCEHKMOPTXY' for char in latin):
        return word.translate(LOOKALIKES)
    return word

def prepare_visual(doc):
    clean = deepcopy(doc)
    for field, value in clean['viz'].items():
        if not value:
            continue
        value['lines'].sort(key=lambda line: (line['box'][1], line['box'][0]))
        if field in TEXT_FIELDS:
            for line in value['lines']:
                text = line['text'].upper()
                line['text'] = re.sub('[A-ZА-ЯЁ]+', normalize_mixed_word, text)
    return clean

def repair_reference_line(line, words):
    text = line['text']
    chars = line['char_conf']
    repairs = 0
    if len(chars) != len(text):
        return text, repairs
    for match in re.finditer(r'[А-ЯЁ]{4,}', text):
        word = match.group()
        if word in words:
            continue
        options = []
        for target_word in words:
            if len(target_word) != len(word):
                continue
            differences = []
            for position, (left, right) in enumerate(zip(word, target_word)):
                if left != right:
                    differences.append(position)
            if len(differences) == 1:
                offset = differences[0]
                position = match.start() + offset
                if chars[position] < 0.9:
                    options.append((position, target_word[offset]))
        if len(options) == 1:
            position, char = options[0]
            text = text[:position] + char + text[position + 1:]
            repairs += 1
    return text, repairs

def recover_document(doc, reference, reference_words):
    clean = prepare_visual(doc)
    candidate = read_zones(clean, reference)
    old_format_reasons = format_reasons(candidate['fields'])
    candidate['reference_repairs'] = 0
    candidate['format_repairs'] = []

    for field in ['Birth_date', 'Issue_date']:
        text = candidate['fields'][field]
        digits = re.sub(r'[.\s/-]', '', text)
        if digits.isdigit() and len(digits) == 8:
            formatted = f'{digits[:2]}.{digits[2:4]}.{digits[4:]}'
            if parse_date(formatted) and formatted != text:
                candidate['fields'][field] = formatted
                candidate['format_repairs'].append(field)

    unit_code = candidate['fields']['Issue_organisation_code']
    words = reference_words.get(unit_code, set())
    lines = (clean['viz']['Issue_organization_ru'] or {}).get('lines', [])
    texts = []
    for line in lines:
        text, repairs = repair_reference_line(line, words)
        texts.append(text)
        candidate['reference_repairs'] += repairs
    if texts:
        candidate['fields']['Issue_organization_ru'] = norm(' '.join(texts))

    candidate['reference'] = reference_features(
        candidate['fields']['Issue_organization_ru'], unit_code, reference
    )
    reasons = []
    for reason in candidate['reasons']:
        if reason not in old_format_reasons:
            reasons.append(reason)
    reasons.extend(format_reasons(candidate['fields']))
    candidate['reasons'] = sorted(set(reasons))
    return candidate

def field_features(candidate, field):
    features = dict(candidate['stats'][field])
    features.update(candidate['quality'])
    features['field'] = field
    features['source'] = candidate['source']
    features['supported'] = field in candidate['supported']
    features['changed'] = field in candidate['changed']
    features['format_repair'] = field in candidate['format_repairs']
    features['mrz_conflict'] = field in candidate['conflicts']
    features['mrz_valid'] = candidate['mrz_valid']
    features['length'] = len(candidate['fields'][field])
    features['empty'] = not candidate['fields'][field]
    features['reference_repairs'] = 0
    if field == 'Issue_organization_ru':
        features['reference_repairs'] = candidate['reference_repairs']
    for name, value in candidate['reference'].items():
        features[name] = value if field == 'Issue_organization_ru' else 0
    return features

def make_features(candidate):
    features = dict(candidate['quality'])
    features['source'] = candidate['source']
    features['mrz_valid'] = int(candidate['mrz_valid'])
    features['mrz_changes'] = len(candidate['changed'])
    features['mrz_conflicts'] = len(candidate['conflicts'])
    features.update(candidate['reference'])
    for field in FIELDS:
        for name, value in candidate['stats'][field].items():
            features[field + '__' + name] = value
        features[field + '__supported'] = int(field in candidate['supported'])
        features[field + '__length'] = len(candidate['fields'][field])
    for index in range(2):
        value = 0.0
        if len(candidate['mrz_lines']) > index:
            chars = candidate['mrz_lines'][index].get('char_conf', [])
            value = min(chars, default=0.0)
        features['mrz_char_min_' + str(index)] = value
    return features

def document_features(candidate):
    features = make_features(candidate)
    stats = []
    for field in FIELDS:
        if field not in candidate['supported']:
            stats.append(candidate['stats'][field])
    features['reference_repairs'] = candidate['reference_repairs']
    features['format_repairs'] = len(candidate['format_repairs'])
    features['unconfirmed_fields'] = len(stats)
    features['unconfirmed_char_min'] = min([item['char_min'] for item in stats], default=1)
    features['unconfirmed_conf_min'] = min([item['conf'] for item in stats], default=1)
    return features

def load_reference(path):
    reference = defaultdict(set)
    with open(path, encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            reference[row['code']].add(norm(row['name']))
    reference = {code: sorted(names) for code, names in reference.items()}
    words = {}
    for code, names in reference.items():
        vocabulary = set()
        for name in names:
            vocabulary.update(re.findall(r'[А-ЯЁ]{4,}', name))
        words[code] = vocabulary
    return reference, words


def read_jsonl(path):
    rows = []
    ids = set()
    with open(path, encoding='utf-8-sig') as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get('doc_id') if isinstance(row, dict) else None
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f'Invalid doc_id at line {line_number}')
            if doc_id in ids:
                raise ValueError(f'Duplicate doc_id: {doc_id}')
            ids.add(doc_id)
            rows.append(row)
    return rows


def feature_table(candidates):
    rows = []
    for candidate in candidates:
        for field in FIELDS:
            rows.append(field_features(candidate, field))
    return pd.DataFrame(rows)


def aggregate_scores(raw, aggregation):
    if aggregation == 'document':
        return np.asarray(raw)
    matrix = np.asarray(raw).reshape(-1, len(FIELDS))
    if aggregation == 'minimum':
        return matrix.min(axis=1)
    if aggregation == 'product':
        return matrix.prod(axis=1)
    raise ValueError(f'Unknown aggregation: {aggregation}')


def recover_documents(documents, reference_path):
    reference, words = load_reference(reference_path)
    return [recover_document(doc, reference, words) for doc in documents]


def valid_input(doc):
    if not isinstance(doc, dict) or doc.get('source') not in {'web', 'mobile', 'mobile_v2'}:
        return False
    quality = doc.get('quality')
    if not isinstance(quality, dict):
        return False
    for key in ['sharpness', 'doc_px', 'glare']:
        value = quality.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
    if not isinstance(doc.get('viz'), dict) or not set(FIELDS).issubset(doc['viz']):
        return False
    for value in doc['viz'].values():
        if value is None:
            continue
        if not isinstance(value, dict):
            return False
        box = value.get('box')
        if not isinstance(box, list) or len(box) != 4:
            return False
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            return False
        if not isinstance(value.get('conf'), (int, float)) or not 0 <= value['conf'] <= 1:
            return False
        if not isinstance(value.get('lines'), list):
            return False
        for line in value['lines']:
            if not valid_line(line, needs_box=True):
                return False
    mrz = doc.get('mrz')
    if mrz is not None:
        if not isinstance(mrz, dict) or not isinstance(mrz.get('lines'), list):
            return False
        if not all(valid_line(line, needs_box=False) for line in mrz['lines']):
            return False
    return True


def valid_line(line, needs_box):
    if not isinstance(line, dict) or not isinstance(line.get('text'), str):
        return False
    chars = line.get('char_conf')
    if not isinstance(chars, list) or len(chars) != len(line['text']):
        return False
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in chars):
        return False
    if needs_box:
        box = line.get('box')
        if not isinstance(box, list) or len(box) != 4:
            return False
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            return False
    return True

