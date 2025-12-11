"""Diagnostic: flag dataset items where all models disagree with gold label.

Saves results to `results/label_disagreements.json` and prints a brief summary.
"""
from src.utils.config import ConfigLoader
from src.llm_clients.gpt4_client import GPT4Client
from src.llm_clients.medical_chatgpt_client import MedicalChatGPTClient
import importlib
import json
import os
from typing import List, Dict


def load_items(path: str):
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(l) for l in f if l.strip()]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def normalize_options(raw_options):
    if isinstance(raw_options, dict):
        return [raw_options.get(k, '') for k in sorted(raw_options.keys())]
    return raw_options or []


def try_instantiate_llama(cfg):
    try:
        mod = importlib.import_module('src.llm_clients.llama2_client')
        Llama = getattr(mod, 'Llama2Client', None)
        if Llama is None:
            return None
        # no API key required for local LLaMA; pass empty config
        return Llama(None, cfg)
    except Exception:
        return None


def main(dataset_path='data/MedQA-USMLE_formatted.jsonl'):
    cfg = ConfigLoader()
    api_keys = cfg.get_api_keys()
    openai_cfg = api_keys.get('openai', {})
    api_key = openai_cfg.get('api_key')

    # instantiate clients
    clients = []
    try:
        gpt = GPT4Client(api_key, openai_cfg)
        clients.append(('gpt4', gpt))
    except Exception as e:
        print('Warning: GPT4 client init failed:', e)

    try:
        med = MedicalChatGPTClient(api_key, openai_cfg)
        clients.append(('medical_ai_4o', med))
    except Exception as e:
        print('Warning: Medical AI client init failed:', e)

    llama = try_instantiate_llama(openai_cfg)
    if llama:
        clients.append(('llama2', llama))

    if not clients:
        print('No LLM clients available; aborting.')
        return

    items = load_items(dataset_path)
    flagged: List[Dict] = []

    for idx, item in enumerate(items):
        question = item.get('question')
        options = normalize_options(item.get('options', []))
        gold = (item.get('gold_label') or item.get('answer') or '').strip().upper()
        if not gold or len(gold) != 1 or not gold.isalpha():
            # skip non-MCQ or malformed gold
            continue

        model_preds = {}
        all_preds = []
        for name, client in clients:
            try:
                prompt = client.format_question(question, options)
                resp = client.query(prompt, temperature=0.0)
                raw = resp.get('text', '')
                pred = client.extract_answer(raw, options).strip().upper()
            except Exception as e:
                raw = f'<error: {e}>'
                pred = ''

            model_preds[name] = {'raw': raw, 'pred': pred}
            if pred:
                all_preds.append(pred)

        # consider disagreement when no model predicted the gold label
        agrees = any(p == gold for p in all_preds)
        if not agrees:
            flagged.append({
                'index': idx,
                'question': question,
                'options': options,
                'gold_label': gold,
                'model_predictions': model_preds
            })

    os.makedirs('results', exist_ok=True)
    out_path = 'results/label_disagreements.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(flagged, f, indent=2, ensure_ascii=False)

    print(f'Processed {len(items)} items; flagged {len(flagged)} potential label disagreements.')
    if flagged:
        print('Sample flagged indices:', [f['index'] for f in flagged[:10]])
        print('Saved details to', out_path)


if __name__ == '__main__':
    main()
