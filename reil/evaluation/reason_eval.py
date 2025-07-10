import argparse
from vllm import LLM, SamplingParams
import os
import json
from collections import defaultdict
import re
verifier_prompt = \
"""<|im_start|>user\nYou are a meticulous scientific evaluator.  
Your task is to decide which of two categories best describes the
another model's *entire* response to the prompt shown below.
Decide whether the agent's entire response to the prompt shows REASONING,
GUESSING. The response will have <think></think> and <answer></answer> tags.
We call the content between <think></think> tags the "thought" and the content between <answer></answer> tags the "answer".

Prompt: {prompt}

Response: {response}

CATEGORIES
----------
REASONING : Does the "thought" contain step-by-step derivation with ≥ 2 substantive intermediate
            steps; a reader can reproduce the answer from reading the thought.
GUESSING  : The “thought” is decorative / irrelevant to the answer; a reader cannot reproduce the answer from reading the thought.

Let's think step by step and output your final judgment within <result> ... </result> tags. You have 1024 tokens to use.
<|im_end|>\n<|im_start|>assistant\n
"""
def post_process_prompt(prompt):
    """
    replace the <|im_start|>user\n and <|im_end|>\n with space
    replace the <|im_start|>assistant\n and <|im_end|>\n with space
    delete the last "<think>" in prompt
    """
    # Find the last occurrence of "<think>" and preserve only content before it
    last_think_index = prompt.rfind("<think>")
    if last_think_index != -1:
        prompt = prompt[:last_think_index]

    return prompt.replace("<|im_start|>user\n", "").replace("<|im_end|>\n", "").replace("<|im_start|>assistant\n", "")

def extract_answer(response):
    """
    extract the answer from the response, and use regex to check if it is REASONING, return 1, otherwise return 0
    """
    # Check if response contains both <answer> and </answer> and <answer> comes before </answer>
    if "<result>" not in response or "</result>" not in response:
        return -1
    if response.find("<result>") > response.find("</result>"):
        return -1
    
    answer = response.split("<result>")[1].split("</result>")[0]
    if re.search(r"REASONING", answer, re.IGNORECASE):
        return 1
    elif re.search(r"GUESSING", answer, re.IGNORECASE):
        return 0
    else:
        return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONLines file")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for LLM generation")
    parser.add_argument("--num_generations", type=int, default=1, help="Number of generations for each example")
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.verifier_path):
        print(f"Model {args.verifier_path} not found. Skip.")
        return

    # Load the model and tokenizer
    print(f"Loading verifier model {args.verifier_path}")
    llm = LLM(args.verifier_path, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.9, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=1.0, 
        top_k=-1,
        max_tokens=2048,
        n=args.num_generations,
    )
    
    data = []
    print(f"Loading data from {args.file_path}")
    with open(args.file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Processing {len(data)} examples with verifier...")
    
    # Process examples in batches
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i + args.batch_size]
        batch_prompts = []
        batch_indices = []
        
        # Prepare batch of formatted prompts only for examples with thought
        for j, example in enumerate(batch):
            if example.get("thought") is not None:
                prompt = post_process_prompt(example["prompt"])
                response = '<think>' + example["response"]
                formatted_prompt = verifier_prompt.format(prompt=prompt, response=response)
                batch_prompts.append(formatted_prompt)
                batch_indices.append(j)
        
        # Generate verifier analysis for the batch only if there are examples with thought
        if batch_prompts:
            outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
            
            # Store the verifier analysis in each example with thought
            for k, output in enumerate(outputs):
                j = batch_indices[k]
                if args.num_generations == 1:
                    verifier_analysis = output.outputs[0].text.strip()
                    batch[j]["verifier_analysis"] = verifier_analysis
                else:
                    verifier_analysis = [output.outputs[l].text.strip() for l in range(len(output.outputs))]
                    batch[j]["verifier_analysis"] = verifier_analysis
                    # Majority voting
                    reasoning_count = 0
                    guessing_count = 0
                    for analysis in verifier_analysis:
                        answer = extract_answer(analysis)
                        if answer == 1:
                            reasoning_count += 1
                        elif answer == 0:
                            guessing_count += 1
                    valid_count = reasoning_count + guessing_count
                    if valid_count == 0:
                        batch[j]["verifier_majority"] = "UNDECIDED"
                        continue
                    if reasoning_count > valid_count * 0.7:
                        batch[j]["verifier_majority"] = "REASONING"
                    elif guessing_count > valid_count * 0.7:
                        batch[j]["verifier_majority"] = "GUESSING"
                    else:
                        batch[j]["verifier_majority"] = "UNDECIDED"
        
        if (i + len(batch)) % (args.batch_size * 5) == 0:
            print(f"Processed {i + len(batch)}/{len(data)} examples")
    
    # Save the results with verifier analysis
    output_file = args.file_path.replace('.jsonl', '_with_verifier.jsonl')
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    
    print("Verifier analysis complete!")

    # Calculate statistics
    num_reasoning = 0
    num_guessing = 0
    correct_reasoning = 0
    correct_guessing = 0
    for example in data:
        majority = example.get("verifier_majority")
        label = example.get("label")
        if majority == "REASONING":
            num_reasoning += 1
            if label == 1:
                correct_reasoning += 1
        elif majority == "GUESSING":
            num_guessing += 1
            if label == 1:
                correct_guessing += 1
    acc_reasoning = correct_reasoning / num_reasoning if num_reasoning > 0 else 0
    acc_guessing = correct_guessing / num_guessing if num_guessing > 0 else 0
    overall_accuracy = (correct_reasoning + correct_guessing) / (num_reasoning + num_guessing) if (num_reasoning + num_guessing) > 0 else 0
    print(f"Number of REASONING: {num_reasoning}")
    print(f"Number of GUESSING: {num_guessing}")
    print(f"Accuracy of REASONING: {acc_reasoning:.4f}")
    print(f"Accuracy of GUESSING: {acc_guessing:.4f}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")


if __name__ == "__main__":
    main()