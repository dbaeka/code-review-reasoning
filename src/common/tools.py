import logging
import os
import time

import numpy as np
from datasets import load_dataset

from src.utils.ioutils import jdump


def shard_dataset(repo_name, output_dir, chunk_size: int = 10_000):
    dataset = load_dataset(repo_name)['test']
    for i in range(0, len(dataset), chunk_size):
        shard = dataset[i:i + chunk_size]
        jdump(shard, f"{output_dir}/shard_{i // chunk_size}_input.json")


def get_bm25_scores(bm25, tokenized_query, top_k: int = 1):
    scores = bm25.get_scores(tokenized_query)
    scores_arr = np.array(scores)
    sorted_indices = scores_arr.argsort()[-top_k:][::-1]
    return sorted_indices, scores_arr


def init_logging(base_drive_dir: str):
    log_folder_path = os.path.join(base_drive_dir, 'soen691/logs')
    os.makedirs(log_folder_path, exist_ok=True)
    # use the current time to create a unique log file
    log_file_path = os.path.join(log_folder_path, f"review_comment_generation_{time.strftime('%Y%m%d')}.log")
    logging.basicConfig(
        filename=log_file_path,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
