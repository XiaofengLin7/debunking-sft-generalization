import json
from datasets import Dataset

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
            datapoints.append({
                "index": index,
                "question": question,
                "answer": answer
            })
            index += 1
    return datapoints


if __name__ == "__main__":
    # === Configuration ===
    json_path = "./data/gp-l-only/SFT_Data/gp-l/data.json"  # Your input data
    parquet_path = "./data/gp-l-only/sft/train.parquet"
    dataset_id = "Xiaofeng77/gp-l-only"  # Change this to your HF namespace

    # === Execution ===
    datapoints = convert_data(json_path)
    dataset = Dataset.from_list(datapoints)
    dataset.to_parquet(parquet_path)
    dataset.push_to_hub(dataset_id, split="train")

