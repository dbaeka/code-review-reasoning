import json
import multiprocessing as mp
import os
from functools import partial

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from src.common.tools import init_logging

local_drive_mount = "/Users/dbaeka/Library/CloudStorage/GoogleDrive-dbaekajnr@gmail.com/My Drive/"


def combine_results(instance, dataset, chunks):
    shards = chunks[instance]
    new_dataset = {}

    dataset_dict = {hash: sample for hash, sample in zip(dataset["hash"], dataset)}

    for shard in tqdm(shards):
        file_path, value = next(iter(shard.items()))
        model = value["model"]
        result_type = value["type"]

        with open(file_path, 'r') as f:
            data = json.loads(f.read())

        if not data:
            continue

        for hash, result in data.items():
            if hash not in dataset_dict:
                continue

            sample = dataset_dict[hash]

            new_sample = new_dataset.get(hash)
            if new_sample is None:
                new_sample = {
                    "hash": hash,
                    "prompt": sample["prompt_base"],
                    f"{model}_{result_type}": result
                }
                new_dataset[hash] = new_sample
            else:
                # check if the result already exists
                new_dataset[hash][f"{model}_{result_type}"] = result
    return new_dataset


if __name__ == "__main__":
    init_logging(local_drive_mount)

    test_name = "test_5000"

    n_jobs = min(8, mp.cpu_count())

    base_results_dir = os.path.join(local_drive_mount, "soen691/results/")

    shards_dict = {}
    models = [x for x in os.listdir(base_results_dir) if "_input" not in x]
    for model in models:
        if not os.path.isdir(os.path.join(base_results_dir, model)):
            continue
        output_dir = os.path.join(base_results_dir, model, f"{test_name}_output")
        shards = [x for x in os.listdir(output_dir) if x.endswith(".json")]
        for shard in shards:
            path = os.path.join(output_dir, shard)
            if "_summary_callgraph" in shard:
                shards_dict[path] = {"type": "summary_callgraph", "model": model}
            elif "_summary" in shard:
                shards_dict[path] = {"type": "summary", "model": model}
            else:
                shards_dict[path] = {"type": "base", "model": model}

    dataset = load_dataset(f"dbaeka/soen_691_few_shot_{test_name}_base_hashed")['test']

    shards_dict_chunks = {}
    for i in range(n_jobs):
        shards_dict_chunks[i] = []
    for i, (key, value) in enumerate(shards_dict.items()):
        shards_dict_chunks[i % n_jobs].append({key: value})

    with mp.Pool(n_jobs) as pool:
        new_datasets = list(tqdm(
            pool.imap(partial(combine_results, dataset=dataset, chunks=shards_dict_chunks), range(n_jobs)),
            total=len(shards_dict_chunks.items()),
            desc="Combining dataset"
        ))

    new_dataset_dict = {}
    for i, new_dataset in tqdm(enumerate(new_datasets)):
        for hash, sample in new_dataset.items():
            if hash not in new_dataset_dict:
                new_dataset_dict[hash] = sample
            else:
                for key, value in sample.items():
                    if key not in new_dataset_dict[hash]:
                        new_dataset_dict[hash][key] = value

    hf_dataset = Dataset.from_list(list(new_dataset_dict.values()))
    hf_dataset = DatasetDict({"test": hf_dataset})

    hf_dataset.push_to_hub(f"dbaeka/soen_691_{test_name}_hashed_with_results")
