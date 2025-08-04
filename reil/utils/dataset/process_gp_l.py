import json
from datasets import Dataset
import random
from tqdm import tqdm
"""
This script is to process the gp-l dataset provided by the author of SFTvsRL paper.
"""
def parse_answer(answer):
    answer = answer.strip()
    # print(answer)
    
    # Fix malformed JSON by replacing single quotes with double quotes
    import re
    
    # Replace single quotes around strings with double quotes
    fixed_answer = re.sub(r"'([^']*)'", r'"\1"', answer)
    
    # Also handle the case where the entire string might be wrapped in single quotes
    if fixed_answer.startswith("'") and fixed_answer.endswith("'"):
        fixed_answer = fixed_answer[1:-1]
    
    # Remove trailing commas that are invalid in JSON
    fixed_answer = re.sub(r',(\s*[}\]])', r'\1', fixed_answer)
    
    # Remove any trailing whitespace and newlines
    fixed_answer = fixed_answer.strip()
    
    # print("Fixed JSON:", fixed_answer)
    
    try:
        dict = json.loads(fixed_answer)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Error at position: {e.pos}")
        print(f"Line: {e.lineno}, Column: {e.colno}")
        # Print the problematic part of the JSON
        if e.pos < len(fixed_answer):
            print(f"Problematic part: {fixed_answer[max(0, e.pos-10):e.pos+10]}")
        raise
    
    card_str = dict["cards"]
    number = dict["number"]
    formula = dict["formula"]
    return {
        "cards": card_str,
        "display_cards": number,
        "solution": formula,
        "target": 24,
        "treat_face_cards_as_10": True,
    }

def convert_data(json_path, dataset_size=10000):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    sft_datapoints = []
    rl_datapoints = []
    index = 0
    random.seed(42)
    random.shuffle(raw_data)
    if len(raw_data) > dataset_size:
        raw_data = raw_data[:dataset_size]
    else:
        print(f'Warning: dataset size is less than {dataset_size}, using all data')
        dataset_size = len(raw_data)
    
    for item in tqdm(raw_data):
        question = None
        answer = None
        for convo in item.get("conversations", []):
            if convo["from"] == "human":
                question = convo["value"].strip()
            elif convo["from"] == "gpt":
                answer = convo["value"].strip()
        if question is not None and answer is not None:
            meta_info = parse_answer(answer)
            meta_info["index"] = index
            sft_datapoints.append({
                "data_source": "gp-l",
                "extra_info": meta_info,
                "question": question,
                "answer": answer
            })
            rl_datapoints.append({
                "data_source": "gp-l",
                "extra_info": meta_info,
                "question": [{"role": "user", "content": question}],
            })
            index += 1
    return sft_datapoints, rl_datapoints


if __name__ == "__main__":
    # === Configuration ===
    json_path = "./data/gp-l-only/SFT_Data/gp-l/data.json"  # Your input data
    sft_parquet_path = "./data/gp-l-only/sft/train-10k.parquet"
    rl_parquet_path = "./data/gp-l-only/rl/train-10k.parquet"
    dataset_id = "Xiaofeng77/gp-l-only-10k"  # Change this to your HF namespace

    # === Execution ===
    sft_datapoints, rl_datapoints = convert_data(json_path)
    print(f'sft_datapoints={len(sft_datapoints)}')
    print(f'rl_datapoints={len(rl_datapoints)}')
    # randomly shuffle datapoints and take the first 10000
    sft_dataset = Dataset.from_list(sft_datapoints)
    sft_dataset.to_parquet(sft_parquet_path)
    rl_dataset = Dataset.from_list(rl_datapoints)
    rl_dataset.to_parquet(rl_parquet_path)
    # dataset.push_to_hub(dataset_id, split="train")

