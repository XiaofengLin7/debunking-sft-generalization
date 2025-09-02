import argparse
from vllm import LLM, SamplingParams
import os
from datasets import load_dataset
from tqdm import tqdm
import json
from reil.utils.reward_score.sokoban import compute_score_with_action_sequence

def extract_thought_n_answer(response):
    if "Assistant:" in response:
        processed_str = response.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in response:
        processed_str = response.split("<|im_start|>assistant", 1)[1]
    else:
        return None, None
    
    thought = processed_str.split("<think>")[1].split("</think>")[0]
    thought = "<think>" + thought + "</think>"
    final_answer = processed_str.split("<answer>")[1].split("</answer>")[0]
    final_answer = "<answer>" + final_answer + "</answer>"
    return thought, final_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--num_generation", type=int, default=10)
    # parser.add_argument("--verifier_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--extract_thought_n_answer", action="store_true", default=False)
    parser.add_argument("--rejection_sampling", action="store_true", default=False)

    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.model_path):
        print(f"Model {args.model_path} not found. Skip.")
        return

    # Load the model and tokenizer
    print(f"Loading model {args.model_path}")
    llm = LLM(args.model_path, tensor_parallel_size=args.num_gpus, dtype="bfloat16", gpu_memory_utilization=0.6, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=args.num_generation, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    datasets = args.datasets.split(",")
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name, split=args.split)
        if "sokoban" in dataset_name.lower():
            answer_key = "reward_model"
            prompt_key = "prompt"
        else:
            answer_key = "answer"
            prompt_key = "prompt"
        import datetime
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            dataset_name.split("/")[-1]
            + '-'
            + args.split
            + '-temp_'
            + str(args.temperature)
            + "-top_p_"
            + str(args.top_p)
            + "-top_k_"
            + str(args.top_k)
            + '-'
            + now_str
            + '.jsonl'
        )
        output_dir = args.output_dir
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if local_rank == 0 and os.path.exists(os.path.join(output_dir, output_file)):
            raise FileExistsError(f"Output file {output_file} already exists.")
        # Create a JSONL file to store the output
        with open(os.path.join(output_dir, output_file), 'w') as f:
            for i in tqdm(range(0, len(dataset), args.batch_size)):
                batch = dataset[i:i + args.batch_size]
                print(batch[prompt_key][0][0])
                inputs = [batch[prompt_key][j][0]["content"] for j in range(len(batch[prompt_key]))]
                answers = batch[answer_key]

                # Generate the answer
                outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
                results = [[_.outputs[l].text for l in range(len(_.outputs))] for _ in outputs]
                assert len(results[0]) == args.num_generation, f"Number of generations is not equal to {args.num_generation}, got {len(results[0])}"

                # Prepare all outputs for batch tokenization
                flat_outputs = []
                output_mapping = []  # To map back to original indices
                
                for j in range(len(results)):
                    for k in range(args.num_generation):
                        flat_outputs.append(results[j][k])
                        output_mapping.append((j, k))

                # Process the results, store each generation result as a separate qa pair
                # if extract_thought_n_answer is True, store the thought and final answer in the qa pair
                output_idx = 0
                for j, (inp, q, a, r) in enumerate(zip(inputs, batch[prompt_key], answers, results)):
                    for k in range(args.num_generation):
                        qa_pair = {
                            "prompt": inp,
                            "answer": a,
                            "question_id": i + j,
                            "generation_id": k,
                        }
                        qa_pair["response"] = r[k]
                        if "sokoban" in dataset_name.lower():
                            qa_pair["score"] = compute_score_with_action_sequence(qa_pair["prompt"]+qa_pair["response"], a['ground_truth'], data_source='sokoban', format_score=0.1, score=1.0)
                            if args.rejection_sampling:
                                if qa_pair["score"] == 1:
                                    output_idx += 1
                                    f.write(json.dumps(qa_pair) + '\n')
                                break
                            if args.extract_thought_n_answer:
                                if qa_pair["score"] == 0:
                                    qa_pair["thought"] = None
                                    qa_pair["final_answer"] = None
                                    qa_pair["label"] = 0
                                else:
                                    qa_pair["thought"], qa_pair["final_answer"] = extract_thought_n_answer(qa_pair["prompt"] + qa_pair["response"])
                                    qa_pair["label"] = 1 if qa_pair["score"] == 1 else 0
                        output_idx += 1
                        f.write(json.dumps(qa_pair) + '\n')
                f.flush()        
if __name__ == "__main__":
    main()