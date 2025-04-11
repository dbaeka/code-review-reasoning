import json
import logging
import multiprocessing as mp
import os
from functools import partial
from hashlib import sha256

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from src.common.eval import calculate_bert_score
from src.common.tools import init_logging

local_drive_mount = "/Users/dbaeka/Library/CloudStorage/GoogleDrive-dbaekajnr@gmail.com/My Drive/"


def get_programming_language(diff, code_table):
    val = sha256(f"{diff}".encode()).hexdigest()
    lang = code_table[val]
    if lang is None:
        lang = "Unknown"
    return lang


def build_results(instance, code_table, dataset, chunks):
    finals = chunks[instance]
    new_dataset = {}

    dataset_dict = {hash: sample for hash, sample in zip(dataset["hash"], dataset)}

    for pred in tqdm(zip(*finals.values()), total=len(finals['hash']), desc=f"Processing sample {instance}"):
        final = dict(zip(finals.keys(), pred))

        hash = final["hash"]

        if hash not in dataset_dict:
            continue

        sample = dataset_dict[hash]
        gold = sample["value"]["msg"]
        code_patch = sample["value"]["patch"]
        pred_value = final["few_without__Anthropic_Claude_3_7_Sonnet_20250219"]
        if pred_value is None:
            pred_value = {"answer": "", "cot": ""}

        language = get_programming_language(code_patch, code_table)
        bertscore = calculate_bert_score(pred_value['answer'], gold)
        logging.info(f"BERT score ({hash}): {bertscore}")
        new_sample = {
            "hash": hash,
            "gold": gold,
            "pred": pred_value["answer"],
            "cot": pred_value["cot"],
            "lang": language,
            "summary": sample["value"]["summary"],
            "bert_score": bertscore,
            "patch": code_patch,
        }
        new_dataset[hash] = new_sample
    return list(new_dataset.values())


def build_code_index(file_path):
    lines = []
    for line in open(file_path, 'r', encoding="utf-8"):
        lines.append(json.loads(line))
    hash_table = {}
    for line in lines:
        val = sha256(f"{line['patch']}".encode()).hexdigest()
        hash_table[val] = line['lang']
    return hash_table


if __name__ == "__main__":
    init_logging(local_drive_mount)

    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_path, "codereviewer/msg-test.jsonl")
    code_table = build_code_index(file_path)

    n_jobs = min(8, mp.cpu_count())

    test_name = "test_500"
    base_dataset = load_dataset(f"dbaeka/soen_691_msg_{test_name}_hashed")['test']
    final_dataset = load_dataset(f"dbaeka/soen_691_{test_name}_final_selected_results")['test']

    chunk_size = len(final_dataset) // (n_jobs - 1)
    final_dataset_chunks = final_dataset.batch(chunk_size)

    with mp.Pool(n_jobs) as pool:
        new_datasets = list(tqdm(
            pool.imap(partial(build_results, dataset=base_dataset, code_table=code_table, chunks=final_dataset_chunks),
                      range(n_jobs)),
            desc=f"Building {test_name} dataset"
        ))

    new_dataset = [item for sublist in new_datasets for item in sublist]
    hf_dataset = Dataset.from_list(new_dataset)
    hf_dataset = DatasetDict({"test": hf_dataset})

    hf_dataset.push_to_hub(f"dbaeka/soen_691_{test_name}_meta_set")
