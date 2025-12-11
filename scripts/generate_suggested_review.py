"""Generate a review JSONL by attaching `suggested_gold` to flagged items.

Reads `results/label_disagreements.json` and `data/MedQA-USMLE_formatted.jsonl`.
Writes `data/MedQA-USMLE_flagged_review.jsonl` with an added field `suggested_gold` for
each flagged item (majority-model suggestion).
"""
import json
import os
from collections import Counter


def load_flagged(path='results/label_disagreements.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(items, path):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def main():
    flagged = load_flagged()
    flagged_map = {}
    for f in flagged:
        idx = f.get('index')
        preds = []
        for m, info in f.get('model_predictions', {}).items():
            p = info.get('pred','')
            if p:
                preds.append(p)
        if not preds:
            continue
        # majority vote
        cnt = Counter(preds)
        most_common = cnt.most_common()
        suggested = most_common[0][0]
        flagged_map[idx] = {
            'suggested_gold': suggested,
            'model_votes': dict(cnt)
        }

    src_path = 'data/MedQA-USMLE_formatted.jsonl'
    out_path = 'data/MedQA-USMLE_flagged_review.jsonl'
    items = load_jsonl(src_path)

    new_items = []
    for i, it in enumerate(items):
        new = dict(it)
        if i in flagged_map:
            new['suggested_gold'] = flagged_map[i]['suggested_gold']
            new['suggested_votes'] = flagged_map[i]['model_votes']
        new_items.append(new)

    write_jsonl(new_items, out_path)
    print(f'Wrote {len(new_items)} items to {out_path}; flagged: {len(flagged_map)}')


if __name__ == '__main__':
    main()
