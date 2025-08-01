import json
from datasets import Dataset
import random

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

def convert_data(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    datapoints = []
    index = 0
    for item in raw_data:
        question = None
        answer = None
        for convo in item.get("conversations", []):
            if convo["from"] == "human":
                question = convo["value"].strip()
            elif convo["from"] == "gpt":
                answer = convo["value"].strip()
        if question is not None and answer is not None:
            meta_info = parse_answer(answer)
            meta_info["task_id"] = index
            datapoints.append({
                "meta_info": meta_info,
                "question": question,
                "answer": answer
            })
            index += 1
    return datapoints


if __name__ == "__main__":
    # === Configuration ===
    json_path = "./data/gp-l-only/SFT_Data/gp-l/data.json"  # Your input data
    parquet_path = "./data/gp-l-only/sft/train-10k.parquet"
    dataset_id = "Xiaofeng77/gp-l-only-10k"  # Change this to your HF namespace

    # === Execution ===
    datapoints = convert_data(json_path)
    print(f'datapoints={len(datapoints)}')
    # randomly shuffle datapoints and take the first 10000
    random.seed(42)
    random.shuffle(datapoints)
    datapoints = datapoints[:10000]
    dataset = Dataset.from_list(datapoints)
    dataset.to_parquet(parquet_path)
    dataset.push_to_hub(dataset_id, split="train")

