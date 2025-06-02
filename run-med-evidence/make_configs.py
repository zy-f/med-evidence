import json

model_list = [
    "llama70b_r1distill",
    "llama70b_instruct",
    "llama3_1_405b",
    "llama3_1_70b",
    "llama3_1_8b",
    "llama4scout",
    "llama4mav",
    "llama3_8b",
    "qwen3_235b",
    "qwen72b",
    "qwen7b",
    "qwq32b",
    "deepseekR1",
    "gpto1",
    "gpt4_1mini",
    "gpt4_1",
    "openthinker2_32b",
    "qwen32b",
    "huatuo7b",
    "openbiollm_8b"
    "deepseekV3"
    "mmed_llama3_8b"
    "openbiollm_70b"
    "huatuo70b",
    "llama3_70b"
]

for content_type in ['abstract', 'fulltext']:
    filename_template = f"configs/{{MODEL}}_closed_barebones_{content_type}.json"
    config = {
        "dataset_name": f"med-evidence-{content_type}",
        "model_type": None,
        "prompt_set": "basic",
        "evidence": {
            "summarize": False,
            "llm_filter": False,
        },
        "rag_setup": {
            "prompt_buffer_size": 1000
        }
    }
    for model in model_list:
        filename = filename_template.format(MODEL=model)
        config['model_type'] = model
        with open(filename, 'w') as fh:
            json.dump(config, fh, indent=4)