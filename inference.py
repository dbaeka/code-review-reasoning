import argparse
import logging
import multiprocessing as mp
import os
import random
from functools import partial

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.common.tools import shard_dataset, init_logging
from src.service.anthropic import review_comment_generation as rcg_anthropic
from src.service.groq import review_comment_generation as rcg_groq
from src.service.ollama import review_comment_generation as rcg_ollama
from src.service.openai import review_comment_generation as rcg_openai
from src.service.vllm import review_comment_generation as rcg_vllm

# from src.service.unsloth import review_comment_generation as rcg_unsloth

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

if __name__ == "__main__":
    load_dotenv()

    # parse arguments
    parser = argparse.ArgumentParser(description="Generate review comments")
    parser.add_argument("--model_name", type=str, help="Model name to use for inference")
    parser.add_argument("--test_name", type=str, help="Test name to use for inference")
    parser.add_argument("--batch_size", type=int, help="Batch size to use for inference", default=32)
    parser.add_argument("--base_drive_dir", type=str, help="Base drive directory to store results")
    parser.add_argument("--shard_size", type=int, help="Shard size to use for inference", default=32)
    parser.add_argument("--seed", type=int, help="Seed value to use for random number generator", default=0)
    parser.add_argument("--temperature", type=float, help="Temperature value to use for sampling", default=0.7)
    parser.add_argument("--num_of_results", type=int, help="Number of results to generate", default=1)
    parser.add_argument("--num_of_few_shot", type=int, help="Number of few shot examples to use", default=1)
    parser.add_argument("--with_summary", type=int, help="Flag to include summary in context", default=0)
    parser.add_argument("--with_callgraph", type=int, help="Flag to include callgraph in context", default=0)
    parser.add_argument("--is_reasoning_model", type=int, help="Flag to include reasoning model", default=0)
    parser.add_argument("--pause_duration", type=int, help="Pause duration between requests", default=0.5)
    parser.add_argument("--batch_call", type=int, help="Flag to use batch call for inference", default=0)
    parser.add_argument("--num_instances", type=int, help="Number of instances to process", default=1)
    parser.add_argument("--provider", type=str, help="Provider to use for inference", default="vllm")

    args = parser.parse_args()

    model_name = args.model_name
    test_name = args.test_name
    batch_size = args.batch_size
    base_drive_dir = args.base_drive_dir
    shard_size = args.shard_size
    seed = args.seed
    temperature = args.temperature
    num_of_results = args.num_of_results
    with_summary = args.with_summary
    with_callgraph = args.with_callgraph
    is_reasoning_model = args.is_reasoning_model
    pause_duration = args.pause_duration
    batch_call = args.batch_call
    num_instances = args.num_instances
    provider = args.provider
    num_of_few_shot = args.num_of_few_shot

    if seed is not None:
        torch.manual_seed(seed)

    # Set up Logging
    init_logging(base_drive_dir)

    base_results_dir = os.path.join(base_drive_dir, "soen691/results/")

    if num_of_few_shot > 0:
        datasets_path = "few_shot_" + test_name
        datasets_path += "_base" if not with_summary and not with_callgraph else ""
        local_path_dir = "_few" if not with_summary and not with_callgraph else "_few"
        if with_summary:
            datasets_path += "_summary"
            local_path_dir += "_summary"
        if with_callgraph:
            datasets_path += "_callgraph"
            local_path_dir += "_callgraph"
    else:
        datasets_path = "zero_shot_" + test_name
        local_path_dir = "_zero"

    test_results_dir = os.path.join(base_results_dir, f"{test_name}{local_path_dir}_input")

    shard_dataset(f"dbaeka/soen_691_{datasets_path}_hashed", test_results_dir, shard_size)

    # Get total number of shards from test results directory
    total_shards = len(os.listdir(test_results_dir))

    n_jobs = min(mp.cpu_count(), num_instances)

    provider = provider.lower()

    # choose provider
    if provider == "groq":
        rcg_fn = rcg_groq
    elif provider == "ollama":
        n_jobs = 3
        rcg_fn = rcg_ollama
    elif provider == "vllm":
        rcg_fn = rcg_vllm
        n_jobs = 1
    elif provider == "openai":
        rcg_fn = rcg_openai
        n_jobs = 1
    elif provider == "anthropic":
        rcg_fn = rcg_anthropic
        n_jobs = 1
    else:
        # rcg_fn = rcg_unsloth
        rcg_fn = None
        n_jobs = 1

    with mp.Pool(processes=n_jobs) as pool:
        shard_indices = list(range(total_shards))
        random.shuffle(shard_indices)

        progress_bar = tqdm(total=total_shards, desc="Processing Shards")

        while shard_indices:
            # Pop n_jobs shards for processing, keeping the rest
            current_shard_indices, shard_indices = shard_indices[:n_jobs], shard_indices[n_jobs:]
            for _ in pool.imap(
                    partial(
                        rcg_fn,
                        model_name=model_name,
                        test_name=test_name,
                        shard_indices=current_shard_indices,
                        base_dir=base_results_dir,
                        batch_size=batch_size,
                        temperature=temperature,
                        pause_duration=pause_duration,
                        batch_call=batch_call,
                        num_of_results=num_of_results,
                        seed=seed,
                        is_reasoning_model=is_reasoning_model,
                        dir_prefix=local_path_dir
                    ),
                    range(n_jobs)
            ):
                progress_bar.update(1)

    progress_bar.close()
    print("Finished processing all shards")
    logging.info("Finished processing all shards")
