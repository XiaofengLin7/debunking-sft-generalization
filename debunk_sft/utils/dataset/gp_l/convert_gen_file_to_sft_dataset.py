import pandas as pd
import copy
from pprint import pprint
from datasets import Dataset

def strip_chat_template(question: str):
    return question.split("<|im_start|>user\n")[1].split("<|im_end|>\n<|im_start|>assistant\n")[0]

def convert_jsonl_to_sft_data(data_file: str):
    df = pd.read_json(data_file, lines=True)
    instance_template = {
            "data_source": None,
            "question": None,
            "answer": None,
            "extra_info": {"index": None, "cards": None, "display_cards": None, "solution": None, "target": None, "treat_face_cards_as_10": None}
        }
    instances = []
    for idx, row in df.iterrows():
        pprint(row)
        instance = copy.deepcopy(instance_template)
        instance['data_source'] = row['data_source'] if 'data_source' in row else 'sokoban'
        instance['question'] = strip_chat_template(row['prompt'])
        instance['answer'] = row['response']
        instance['extra_info']['index'] = row['answer']['index']
        instance['extra_info']['cards'] = row['answer']['cards']
        instance['extra_info']['display_cards'] = row['answer']['display_cards']
        instance['extra_info']['solution'] = row['answer']['solution']
        instance['extra_info']['target'] = row['answer']['target']
        instance['extra_info']['face_card_mapping'] = row['answer']['face_card_mapping']
        instances.append(instance)
    return instances

if __name__ == "__main__":
    instances = convert_jsonl_to_sft_data("./results/RLed_qwen3-8b-diverse-train-temp_1.0-top_p_1.0-top_k_-1-20250919_052544.jsonl")
    train_dataset = Dataset.from_list(instances)
    train_dataset.to_parquet("./data/gp-l-only/10k-mixed/cot-sft/train.parquet")
    train_dataset.push_to_hub("Xiaofeng77/Qwen3-8b-cot-gp-l-only-10k-mixed", split="train")