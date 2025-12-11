import json
import csv
import sys
import os

DS_PATH = 'data/MedQA-USMLE_formatted.jsonl'
RES_PATH = 'results/baseline_results.json'
OUT_PATH = 'results/medqa_usmle_per_example.csv'

try:
    with open(DS_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f if l.strip()]
except Exception as e:
    print('Error reading dataset:', e)
    sys.exit(1)

try:
    with open(RES_PATH, 'r', encoding='utf-8') as f:
        res = json.load(f)
except Exception as e:
    print('Error reading results:', e)
    res = {}

preds = res.get('medqa_usmle', {})
models = sorted(preds.keys())
rows = []
for i, item in enumerate(data):
    gold = item.get('gold_label', '')
    q = item.get('question', '').replace('\n', ' ')
    row = {'index': i, 'question': q, 'gold': gold}
    for m in models:
        p_list = preds.get(m, {}).get('predictions', [])
        p = p_list[i] if i < len(p_list) else ''
        row[m] = p
    rows.append(row)

for r in rows:
    print(f"{r['index']}: gold={r['gold']}, " + ", ".join([f"{m}={r.get(m,'')}" for m in models]))

os.makedirs('results', exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['index', 'question', 'gold'] + models)
    writer.writeheader()
    writer.writerows(rows)

print('\nWrote', OUT_PATH)
