from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--save_dir", type=str, default="./models/rlft")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.save_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.save_dir)

    print(f"You saved the model {args.model_name} in {args.save_dir}")


    


if __name__ == "__main__":
    main()

