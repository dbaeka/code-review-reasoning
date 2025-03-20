import argparse
import logging
import multiprocessing as mp
import os
import random
from functools import partial

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.common.tools import shard_dataset, get_bm25_scores, init_logging
from src.service.groq import review_comment_generation
from src.utils.string import prettify

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
    parser.add_argument("--batch_size", type=int, help="Batch size to use for inference")
    parser.add_argument("--base_drive_dir", type=str, help="Base drive directory to store results")
    parser.add_argument("--shard_size", type=int, help="Shard size to use for inference")
    parser.add_argument("--total_shards", type=int, help="Total number of shards to process")
    parser.add_argument("--seed", type=int, help="Seed value to use for random number generator")
    parser.add_argument("--temperature", type=float, help="Temperature value to use for sampling")
    parser.add_argument("--num_of_results", type=int, help="Number of results to generate")
    parser.add_argument("--num_of_few_shot", type=int, help="Number of few shot examples to use")
    parser.add_argument("--with_summary", type=int, help="Flag to include summary in context")
    parser.add_argument("--with_callgraph", type=int, help="Flag to include callgraph in context")
    parser.add_argument("--is_reasoning_model", type=int, help="Flag to include reasoning model")
    parser.add_argument("--pause_duration", type=int, help="Pause duration between requests")
    parser.add_argument("--batch_call", type=int, help="Flag to use batch call for inference")
    parser.add_argument("--log_file", type=str, help="Log file to store logs")
    parser.add_argument("--num_instances", type=int, help="Number of instances to process", default=1)

    args = parser.parse_args()

    model_name = args.model_name
    test_name = args.test_name
    batch_size = args.batch_size
    base_drive_dir = args.base_drive_dir
    shard_size = args.shard_size
    total_shards = args.total_shards
    seed = args.seed
    temperature = args.temperature
    num_of_results = args.num_of_results
    num_of_few_shot = args.num_of_few_shot
    with_summary = args.with_summary
    with_callgraph = args.with_callgraph
    is_reasoning_model = args.is_reasoning_model
    pause_duration = args.pause_duration
    batch_call = args.batch_call
    num_instances = args.num_instances

    if seed is not None:
        torch.manual_seed(seed)

    # Set up Logging
    init_logging(base_drive_dir)

    base_results_dir = os.path.join(base_drive_dir, "soen691/results/")
    test_results_dir = os.path.join(base_results_dir, prettify(model_name), f"{test_name}_input")

    shard_dataset(f"dbaeka/soen_691_msg_{test_name}_hashed", test_results_dir, shard_size)

    train_dataset = load_dataset("dbaeka/soen_691_msg_train")['train']
    print(f"Train data length: {len(train_dataset)}")
    print("\n-----------------\n")

    # Set up BM25
    tokenized_corpus = [doc["patch"].split(" ") for doc in train_dataset]
    bm25 = BM25Okapi(tokenized_corpus)

    # Query example
    query = "CrossProduct"
    tokenized_query = query.split(" ")
    sorted_indices, scores = get_bm25_scores(bm25, tokenized_query, top_k=3)
    # Show top 3 matches
    top_k = 3
    print("Top Retrieved Documents:")
    for i in range(top_k):
        index = sorted_indices[i]
        patch = train_dataset["patch"]
        print(f"Rank {i + 1}: Score {scores[index]:.4f} - {patch[index]}")

    n_jobs = min(mp.cpu_count(), num_instances)
    with mp.Pool(processes=n_jobs) as pool:
        shard_indices = list(range(total_shards))
        random.shuffle(shard_indices)

        progress_bar = tqdm(total=total_shards, desc="Processing Shards")

        while shard_indices:
            # Pop n_jobs shards for processing, keeping the rest
            current_shard_indices, shard_indices = shard_indices[:n_jobs], shard_indices[n_jobs:]

            for _ in pool.imap(
                    partial(
                        review_comment_generation,
                        model_name=model_name,
                        test_name=test_name,
                        shard_indices=current_shard_indices,
                        base_dir=base_results_dir,
                        train_dataset=train_dataset,
                        bm25=bm25,
                        batch_size=batch_size,
                        num_of_few_shot=num_of_few_shot,
                        with_summary=with_summary,
                        with_callgraph=with_callgraph,
                        temperature=temperature,
                        pause_duration=pause_duration,
                        batch_call=batch_call,
                        num_of_results=num_of_results,
                        seed=seed,
                        is_reasoning_model=is_reasoning_model
                    ),
                    range(n_jobs)
            ):
                progress_bar.update(1)  # Manually update tqdm for each completed shard

        progress_bar.close()
        print("Finished processing all shards")
        logging.info("Finished processing all shards")
