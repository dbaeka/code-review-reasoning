import multiprocessing as mp

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.common.tools import init_logging

local_drive_mount = "/Users/dbaeka/Library/CloudStorage/GoogleDrive-dbaekajnr@gmail.com/My Drive/"

train_dataset = load_dataset("dbaeka/soen_691_msg_train")['train']
print(f"Train data length: {len(train_dataset)}")
print("\n-----------------\n")

tokenized_corpus = [doc["patch"].split(" ") for doc in train_dataset]
bm25 = BM25Okapi(tokenized_corpus)

# # Query example
# query = "CrossProduct"
# tokenized_query = query.split(" ")
#
# # Get scores
# scores = bm25.get_scores(tokenized_query)
#
# # Sort documents by score (descending order)
# sorted_indices = np.argsort(scores)[::-1]  # Get indices sorted by highest score
#
# # Show top 3 matches
# top_k = 3
# print("Top Retrieved Documents:")
# for i in range(top_k):
#     index = sorted_indices[i]
#     patch = train_dataset["patch"]
#     print(f"Rank {i + 1}: Score {scores[index]:.4f} - {patch[index]}")


NUM_OF_FEW_SHOT = 5
SEED = 0


def get_bm25_indices(example, num_shot: int = 1):
    tokenized_query = example.split(" ")
    scores = bm25.get_scores(tokenized_query)
    scores_arr = np.array(scores)
    sorted_indices = scores_arr.argsort()[-num_shot:][::-1]
    return sorted_indices


def build_dataset(dataset):
    new_dataset = []
    i = 0
    for sample in zip(dataset['hash'], dataset['value']):
        print(f"Processing {i}th sample")
        data = {"hash": sample[0], "value": sample[1]}
        context_msg, bm_indices = get_bm25_indices(data["value"]["patch"], num_shot=NUM_OF_FEW_SHOT)
        data["bm_indices"] = bm_indices
        new_dataset.append(data)
        i += 1
    return new_dataset


if __name__ == "__main__":
    init_logging(local_drive_mount)

    test_name = "test_5000"

    n_jobs = min(8, mp.cpu_count())

    dataset = load_dataset(f"dbaeka/soen_691_msg_{test_name}_hashed")['test']

    chunk_size = len(dataset) // (n_jobs - 1)
    dataset_chunks = dataset.batch(chunk_size)

    # Process in parallel with tqdm
    with mp.Pool(n_jobs) as pool:
        new_dataset = list(tqdm(
            pool.imap(build_dataset, dataset_chunks),
            total=len(dataset_chunks),
            desc="Building dataset"
        ))
    new_dataset = [item for sublist in new_dataset for item in sublist]
    hf_dataset = Dataset.from_list(new_dataset)
    hf_dataset = DatasetDict({"test": hf_dataset})
    hf_dataset.push_to_hub(f"dbaeka/soen_691_{test_name}_bm_25_indices_hashed")
