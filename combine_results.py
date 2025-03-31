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
                    "few_shot_prompt": sample["prompt_base"],
                    f"{result_type}__{model}": result
                }
                new_dataset[hash] = new_sample
            else:
                # check if the result already exists
                new_dataset[hash][f"{result_type}__{model}"] = result
    return new_dataset


if __name__ == "__main__":
    init_logging(local_drive_mount)

    test_name = "test_5000"

    n_jobs = min(8, mp.cpu_count())

    base_results_dir = os.path.join(local_drive_mount, "soen691/results/output")

    shards_dict = {}
    models = [x for x in os.listdir(base_results_dir)]
    for model in models:
        if not os.path.isdir(os.path.join(base_results_dir, model)):
            continue

        # go through zero shot results
        zero_shot_dir = os.path.join(base_results_dir, model, "zero")
        if os.path.isdir(zero_shot_dir):
            shards = [x for x in os.listdir(zero_shot_dir) if x.endswith(".json")]
            for shard in shards:
                path = os.path.join(zero_shot_dir, shard)
                shards_dict[path] = {"type": "zero", "model": model}

        # go through few shot results
        few_shot_dir = os.path.join(base_results_dir, model, "few")
        if os.path.isdir(few_shot_dir) and "deepseek" in model:
            shards = [x for x in os.listdir(few_shot_dir) if x.endswith(".json")]
            for shard in shards:
                path = os.path.join(few_shot_dir, shard)
                if "summary_callgraph" in shard:
                    shards_dict[path] = {"type": "few_summary_callgraph", "model": model}
                elif "summary" in shard:
                    shards_dict[path] = {"type": "few_summary", "model": model}
                elif "callgraph" in shard:
                    shards_dict[path] = {"type": "few_callgraph", "model": model}
                else:
                    shards_dict[path] = {"type": "few_without", "model": model}

    few_shot_dataset = load_dataset(f"dbaeka/soen_691_few_shot_{test_name}_base_hashed")['test']
    zero_shot_dataset = load_dataset(f"dbaeka/soen_691_zero_shot_{test_name}_hashed")['test']

    shards_dict_chunks = {}
    for i in range(n_jobs):
        shards_dict_chunks[i] = []
    for i, (key, value) in enumerate(shards_dict.items()):
        shards_dict_chunks[i % n_jobs].append({key: value})

    # with mp.Pool(n_jobs) as pool:
    #     new_datasets = list(tqdm(
    #         pool.imap(partial(combine_results, dataset=few_shot_dataset, chunks=shards_dict_chunks), range(n_jobs)),
    #         total=len(shards_dict_chunks.items()),
    #         desc="Combining dataset"
    #     ))

    # zero_shot_dataset_dict = {hash: sample for hash, sample in zip(zero_shot_dataset["hash"], zero_shot_dataset)}
    # new_dataset_dict = {}
    # for i, new_dataset in tqdm(enumerate(new_datasets)):
    #     for hash, sample in new_dataset.items():
    #         if hash not in new_dataset_dict:
    #             new_dataset_dict[hash] = sample
    #         else:
    #             for key, value in sample.items():
    #                 if key not in new_dataset_dict[hash]:
    #                     new_dataset_dict[hash][key] = value
    #         zero_shot_prompt = zero_shot_dataset_dict.get(hash, None)
    #         new_dataset_dict[hash]['zero_shot_prompt'] = zero_shot_prompt["prompt_base"] if zero_shot_prompt else None
    #
    # # Ensure all dictionaries have the same keys
    # all_keys = set()
    # for sample in new_dataset_dict.values():
    #     all_keys.update(sample.keys())
    #
    # for sample in new_dataset_dict.values():
    #     for key in all_keys:
    #         if key not in sample:
    #             sample[key] = None
    #
    # hf_dataset = Dataset.from_list(list(new_dataset_dict.values()))
    #
    # # reorder the columns put hash first, few_shot_prompt second, and zero_shot_prompt third, and then the rest
    # columns = ["hash", "few_shot_prompt", "zero_shot_prompt"]
    # columns += [key for key in hf_dataset.column_names if key not in columns]
    #
    # hf_dataset = hf_dataset.select_columns(columns)
    #
    # hf_dataset = DatasetDict({"test": hf_dataset})
    #
    # hf_dataset.push_to_hub(f"dbaeka/soen_691_{test_name}_hashed_with_results")

    test_name = "test_500"

    shards_dict = {}
    models = [x for x in os.listdir(base_results_dir)]
    for model in models:
        if not os.path.isdir(os.path.join(base_results_dir, model)):
            continue

        # go through few shot results
        few_shot_dir = os.path.join(base_results_dir, model, "few")
        if os.path.isdir(few_shot_dir):
            shards = [x for x in os.listdir(few_shot_dir) if x.endswith(".json")]
            for shard in shards:
                path = os.path.join(few_shot_dir, shard)
                if "summary_callgraph" in shard:
                    shards_dict[path] = {"type": "few_summary_callgraph", "model": model}
                elif "summary" in shard:
                    pass
                elif "callgraph" in shard:
                    pass
                else:
                    shards_dict[path] = {"type": "few_without", "model": model}

    few_shot_dataset = load_dataset(f"dbaeka/soen_691_few_shot_{test_name}_base_hashed")['test']

    shards_dict_chunks = {}
    for i in range(n_jobs):
        shards_dict_chunks[i] = []
    for i, (key, value) in enumerate(shards_dict.items()):
        shards_dict_chunks[i % n_jobs].append({key: value})

    with mp.Pool(n_jobs) as pool:
        new_datasets = list(tqdm(
            pool.imap(partial(combine_results, dataset=few_shot_dataset, chunks=shards_dict_chunks), range(n_jobs)),
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

    # Ensure all dictionaries have the same keys
    all_keys = set()
    for sample in new_dataset_dict.values():
        all_keys.update(sample.keys())

    for sample in new_dataset_dict.values():
        for key in all_keys:
            if key not in sample:
                sample[key] = None

    hf_dataset = Dataset.from_list(list(new_dataset_dict.values()))

    # reorder the columns put hash first, few_shot_prompt second, and then the rest
    columns = ["hash", "few_shot_prompt"]
    columns += [key for key in hf_dataset.column_names if key not in columns]

    hf_dataset = hf_dataset.select_columns(columns)

    hf_dataset = DatasetDict({"test": hf_dataset})

    hf_dataset.push_to_hub(f"dbaeka/soen_691_{test_name}_hashed_with_results")
