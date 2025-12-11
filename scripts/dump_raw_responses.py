from src.utils.config import ConfigLoader
from src.llm_clients.gpt4_client import GPT4Client
from src.llm_clients.medical_chatgpt_client import MedicalChatGPTClient
import json
import os


def load_medqa(path='data/MedQA-USMLE_formatted.jsonl'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines


def normalize_options(raw_options):
    if isinstance(raw_options, dict):
        return [raw_options.get(k, '') for k in sorted(raw_options.keys())]
    return raw_options or []


def main():
    cfg = ConfigLoader()
    api_keys = cfg.get_api_keys()
    openai_cfg = api_keys.get('openai', {})
    api_key = openai_cfg.get('api_key')

    data = load_medqa()
    indices = [1, 2]

    gpt = GPT4Client(api_key, openai_cfg)
    med = MedicalChatGPTClient(api_key, openai_cfg)
    clients = [
        ("gpt4", gpt),
        ("medical_ai_4o", med),
    ]

    results = []
    for idx in indices:
        if idx >= len(data):
            print(f"Index {idx} out of range (len={len(data)})")
            continue
        item = data[idx]
        question = item.get('question')
        options = normalize_options(item.get('options', []))

        for name, client in clients:
            prompt = client.format_question(question, options)
            resp = client.query(prompt)
            raw = resp.get('text', '')
            extracted = client.extract_answer(raw, options)

            out = {
                'index': idx,
                'dataset': 'MedQA-USMLE',
                'model': name,
                'prompt': prompt,
                'raw_response': raw,
                'extracted': extracted,
                'gold_label': item.get('gold_label')
            }
            results.append(out)
            print('---')
            print(f"Index: {idx}  Model: {name}")
            print('Prompt:')
            print(prompt)
            print('\nRaw response:')
            print(raw)
            print('\nExtracted:', extracted)

    os.makedirs('results', exist_ok=True)
    with open('results/raw_medqa_responses.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\nSaved to results/raw_medqa_responses.json')


if __name__ == '__main__':
    main()
