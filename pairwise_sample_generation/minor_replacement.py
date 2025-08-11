import re
import os
import sys
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
import copy
from vllm import LLM, SamplingParams
from collections import defaultdict
import ray

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def extract_completion_only(answer):
    pattern = ""
    results = []
    for an in answer:
        per_results = []
        for per_an in an.outputs:
            parts = per_an.text[len(pattern):].strip('\n')
            per_results.append(parts)
        results.append(per_results)
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument("--request_batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--begin_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()

def main():
    args = parse_args()
    print(vars(args))
    ray.init(address="local", ignore_reinit_error=True)
    print("Cluster resources:", ray.cluster_resources())
    random.seed(args.seed)

    author_grouped_pairs = defaultdict(list)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_list = []

    with open(args.input_file, 'r', encoding='utf-8') as file:
        author_data = json.load(file)
        for author_id, samples in author_data.items():
            if isinstance(samples, list):
                for sample in samples:
                    sample["author"] = author_id
                    data_list.append(sample)

    if args.begin_idx != -1 and args.end_idx != -1:
        data_list = data_list[args.begin_idx: args.end_idx]

    print(f"Loaded {len(data_list)} samples")

    with open(os.path.join(script_dir, "min_change_generation.txt"), "r") as file:
        generate_prompt = file.read()

    prompt_template = (
                    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n{instruction}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    )

    llm = LLM(model=args.model, tensor_parallel_size=args.gpus)
    tokenizer = llm.get_tokenizer()

    prompt_samples = []
    for data in data_list:
        prompt_sample = [
            generate_prompt.format(paragraph=data["output"]),
            data["prompt"], data["output"], data["author"]
        ]
        prompt_samples.append(prompt_sample)

    print(f"Processed {len(prompt_samples)} prompts")

    for i in tqdm(range((len(prompt_samples) // args.request_batch_size) + 1)):
        batch_prompt_samples = prompt_samples[i * args.request_batch_size : (i + 1) * args.request_batch_size]
        if not batch_prompt_samples:
            break

        prompt_samples_for_model = [
            prompt_template.format(instruction=s[0]) for s in batch_prompt_samples
        ]

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            n=args.n
        )

        outputs = llm.generate(prompt_samples_for_model, sampling_params)
        outputs = extract_completion_only(outputs)

        for idx_in_batch, generated_outputs in enumerate(outputs):
            _, prompt_text, gt_output, author_id = batch_prompt_samples[idx_in_batch]
            for llm_output in generated_outputs:
                pair = {
                    "prompt": prompt_text,
                    "response_A": gt_output,
                    "response_B": llm_output,
                }
                author_grouped_pairs[author_id].append(pair)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(author_grouped_pairs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()





