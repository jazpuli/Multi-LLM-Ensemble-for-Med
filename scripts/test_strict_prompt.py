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


def build_few_shot(example_item):
    opts = normalize_options(example_item.get('options', []))
    q = example_item.get('question')
    gold = example_item.get('gold_label', '').upper()
    prompt = "Example:\n"
    prompt += q + "\n"
    for i, o in enumerate(opts):
        prompt += f"{chr(65+i)}. {o}\n"
    prompt += f"Answer: {gold}\n\n"
    return prompt


def main():
    cfg = ConfigLoader()
    api_keys = cfg.get_api_keys()
    openai_cfg = api_keys.get('openai', {})
    api_key = openai_cfg.get('api_key')

    data = load_medqa()
    idx = 2
    if idx >= len(data):
        print(f"Index {idx} out of range (len={len(data)})")
        return

    item = data[idx]
    question = item.get('question')
    options = normalize_options(item.get('options', []))

    # Prepare clients
    gpt = GPT4Client(api_key, openai_cfg)
    med = MedicalChatGPTClient(api_key, openai_cfg)
    clients = [("gpt4", gpt), ("medical_ai_4o", med)]

    # Strict instruction (stronger than existing formatter)
    strict_tail = "\nSTRICT INSTRUCTION: Respond with exactly one UPPERCASE letter (A-{}) and nothing else. Do NOT include punctuation, option text, explanations, or any other characters.\n".format(chr(65 + max(0, len(options) - 1)))

    # Build base prompt using existing formatter, then append stricter tail
    base_prompts = {}
    for name, client in clients:
        base = client.format_question(question, options)
        base_prompts[name] = base + strict_tail

    # Build few-shot version using first available example as a shot
    few_shot_prefix = ""
    if len(data) > 0:
        few_shot_prefix = build_few_shot(data[0])

    results = []
    for name, client in clients:
        for variant in ("strict", "few_shot_strict"):
            if variant == 'strict':
                prompt = base_prompts[name]
            else:
                prompt = few_shot_prefix + base_prompts[name]

            # Force deterministic behavior by setting temperature to 0.0
            resp = client.query(prompt, temperature=0.0)
            raw = resp.get('text', '')
            extracted = client.extract_answer(raw, options)

            out = {
                'index': idx,
                'variant': variant,
                'model': name,
                'prompt': prompt,
                'raw_response': raw,
                'extracted': extracted,
                'gold_label': item.get('gold_label')
            }
            results.append(out)
            print('---')
            print(f"Model: {name}  Variant: {variant}")
            print('Raw response:')
            print(raw)
            print('Extracted:', extracted)

    os.makedirs('results', exist_ok=True)
    with open('results/strict_prompt_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\nSaved to results/strict_prompt_results.json')


if __name__ == '__main__':
    main()
