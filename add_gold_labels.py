import json

# Update PubMedQA
with open('data/PubMedQA_formatted.json', 'r') as f:
    data = json.load(f)

for item in data:
    item['gold_label'] = item.get('answer_type')

with open('data/PubMedQA_formatted.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✓ PubMedQA updated with gold_label field")

# Update MedQA-USMLE
with open('data/MedQA-USMLE_formatted.jsonl', 'r') as f:
    lines = f.readlines()

with open('data/MedQA-USMLE_formatted.jsonl', 'w') as f:
    for line in lines:
        item = json.loads(line)
        item['gold_label'] = item.get('answer')
        f.write(json.dumps(item) + '\n')

print("✓ MedQA-USMLE updated with gold_label field")

# Update MedMCQA
with open('data/MedMCQA_formatted.json', 'r') as f:
    data = json.load(f)

for item in data:
    item['gold_label'] = item.get('correct_answer')

with open('data/MedMCQA_formatted.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✓ MedMCQA updated with gold_label field")
