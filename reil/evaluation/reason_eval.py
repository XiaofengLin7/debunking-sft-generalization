import argparse
from vllm import LLM, SamplingParams
import os
import json
from collections import defaultdict

verifier_prompt = \
"""<|im_start|>user\nYou are a meticulous scientific evaluator.  
Your task is to decide which of three categories best describes the
another model's *entire* response to the prompt shown below.
Decide whether the agent's entire response to the prompt shows REASONING,
GUESSING, or NONSENSE.  

Prompt: {prompt}

Response: {response}

CATEGORIES
----------
REASONING : Step-by-step derivation with ≥ 2 substantive intermediate
            steps; a reader can reproduce the answer.
GUESSING  : Gives an answer but shows little or no supporting logic,
            or the “reasoning” is decorative / irrelevant.
NONSENSE  : Fails to answer, is off-topic, incoherent, or merely
            repeats the prompt.

DECISION RUBRIC
---------------
1. Traceability : are steps sufficient to compute the answer? can the reader reproduce the answer? if not, it's not REASONING.
2. Coherence    : are steps logically connected and plausible?
3. Density      : fewer than two meaningful steps ⇒ not REASONING.

Let's think step by step and output your final judgment within <answer></answer> tags.
<|im_end|>\n<|im_start|>assistant\n
"""
def post_process_prompt(prompt):
    """
    replace the <|im_start|>user\n and <|im_end|>\n with space
    replace the <|im_start|>assistant\n and <|im_end|>\n with space
    """
    return prompt.replace("<|im_start|>user\n", "").replace("<|im_end|>\n", "").replace("<|im_start|>assistant\n", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONLines file")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for LLM generation")
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.verifier_path):
        print(f"Model {args.verifier_path} not found. Skip.")
        return

    # Load the model and tokenizer
    print(f"Loading verifier model {args.verifier_path}")
    llm = LLM(args.verifier_path, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.6, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=1.0, 
        top_k=-1,
        max_tokens=1024,
    )
    
    data = []
    print(f"Loading data from {args.file_path}")
    with open(args.file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Processing {len(data)} examples with verifier...")
    
    # Process examples in batches
    for i in range(0, int(len(data)*0.1), args.batch_size):
        batch = data[i:i + args.batch_size]
        batch_prompts = []
        
        # Prepare batch of formatted prompts
        for example in batch:
            prompt = post_process_prompt(example["prompt"])
            response = example["response"]
            formatted_prompt = verifier_prompt.format(prompt=prompt, response=response)
            batch_prompts.append(formatted_prompt)
        
        # Generate verifier analysis for the batch
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        
        # Store the verifier analysis in each example
        for j, example in enumerate(batch):
            verifier_analysis = outputs[j].outputs[0].text.strip()
            example["verifier_analysis"] = verifier_analysis
        
        if (i + len(batch)) % (args.batch_size * 5) == 0:
            print(f"Processed {i + len(batch)}/{len(data)} examples")
    
    # Save the results with verifier analysis
    output_file = args.file_path.replace('.jsonl', '_with_verifier.jsonl')
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    
    print("Verifier analysis complete!")


if __name__ == "__main__":
    main()