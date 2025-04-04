import pandas as pd

def print_test_output_parquet(path: str):
    df = pd.read_parquet(path)
    for idx, row in df.iterrows():
        print(f"\n=== Test {idx} ===")
        print("[Prompt]:")
        print(row['prompt'][0]['content'])
        for i, response in enumerate(row['responses']):
            print(f"[Response {i}]:")
            print(response)

if __name__ == "__main__":
    print_test_output_parquet("./data/sokoban/base_model_0.5b/test_generation.parquet")
