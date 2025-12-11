from src.utils.config import ConfigLoader
from src.llm_clients.gpt4_client import GPT4Client
import json

# Load config and dataset
cfg = ConfigLoader()
api_keys = cfg.get_api_keys()
openai_cfg = api_keys.get('openai', {})
api_key = openai_cfg.get('api_key')

# Read first MedQA-USMLE example
with open('data/MedQA-USMLE_formatted.jsonl','r',encoding='utf-8') as f:
    lines=[json.loads(l) for l in f if l.strip()]
if not lines:
    raise SystemExit('No MedQA-USMLE examples found')
item = lines[0]

# Normalize options (dataset stores options as dict mapping 'A'->text)
raw_options = item.get('options', [])
if isinstance(raw_options, dict):
    options = [raw_options.get(k, '') for k in sorted(raw_options.keys())]
else:
    options = raw_options

# Prepare client
client = GPT4Client(api_key, openai_cfg)
question = item.get('question')
print('\n--- NORMALIZED OPTIONS (for prompt) ---')
print(repr(options))


# Format prompt and query
prompt = client.format_question(question, options)
print('--- PROMPT ---\n')
print(prompt)
print('\n--- QUERYING MODEL ---\n')
resp = client.query(prompt)
raw = resp.get('text','')
print('--- RAW RESPONSE ---\n')
print(raw)

# Extracted answer
extracted = client.extract_answer(raw, options)
print('\n--- EXTRACTED ANSWER ---')
print(extracted)
