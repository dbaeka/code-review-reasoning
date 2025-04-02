import argparse
import logging
import os

import torch
from dotenv import load_dotenv

from src.common.tools import shard_dataset, init_logging
from src.service.vllm import budget_force_infer

if __name__ == "__main__":
    load_dotenv()

    # parse arguments
    parser = argparse.ArgumentParser(description="Generate review comments")
    parser.add_argument("--model_name", type=str, help="Model name to use for inference")
    parser.add_argument("--batch_size", type=int, help="Batch size to use for inference", default=32)
    parser.add_argument("--base_drive_dir", type=str, help="Base drive directory to store results")
    parser.add_argument("--seed", type=int, help="Seed value to use for random number generator", default=0)
    parser.add_argument("--temperature", type=float, help="Temperature value to use for sampling", default=0.7)
    parser.add_argument("--num_of_results", type=int, help="Number of results to generate", default=1)
    parser.add_argument("--num_of_few_shot", type=int, help="Number of few shot examples to use", default=1)
    parser.add_argument("--with_summary", type=int, help="Flag to include summary in context", default=0)
    parser.add_argument("--with_callgraph", type=int, help="Flag to include callgraph in context", default=0)
    parser.add_argument("--pause_duration", type=int, help="Pause duration between requests", default=0.5)
    parser.add_argument("--provider", type=str, help="Provider to use for inference", default="vllm")
    parser.add_argument("--budget", type=int, help="Number of time to budget for thinking", default=1)

    args = parser.parse_args()

    model_name = args.model_name
    test_name = "test_500"
    batch_size = args.batch_size
    base_drive_dir = args.base_drive_dir
    seed = args.seed
    temperature = args.temperature
    num_of_results = args.num_of_results
    with_summary = args.with_summary
    with_callgraph = args.with_callgraph
    pause_duration = args.pause_duration
    num_of_few_shot = args.num_of_few_shot
    budget = args.budget

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

    local_path_dir += "_budget_force_{budget}"
    test_results_dir = os.path.join(base_results_dir, f"{test_name}{local_path_dir}_input")

    shard_dataset(f"dbaeka/soen_691_{datasets_path}_hashed", test_results_dir, 500)

    budget_force_infer(
        model_name=model_name,
        test_name=test_name,
        shard_index=0,
        base_dir=base_results_dir,
        batch_size=batch_size,
        temperature=temperature,
        pause_duration=pause_duration,
        num_of_results=num_of_results,
        budget=args.budget,
        seed=seed,
        dir_prefix=local_path_dir
    )

    print("Finished processing all shards")
    logging.info("Finished processing all shards")
