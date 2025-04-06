import logging
import multiprocessing as mp
from functools import partial

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from src.common.eval import calculate_bleu
from src.common.tools import init_logging

local_drive_mount = "/Users/dbaeka/Library/CloudStorage/GoogleDrive-dbaekajnr@gmail.com/My Drive/"


def build_results(instance, dataset, chunks):
    preds = chunks[instance]
    new_dataset = {}

    dataset_dict = {hash: sample for hash, sample in zip(dataset["hash"], dataset)}

    columns_to_ignore = ['hash', 'few_shot_prompt', 'zero_shot_prompt', 'value']

    for pred in tqdm(zip(*preds.values()), total=len(preds['hash']), desc=f"Processing sample {instance}"):
        pred = dict(zip(preds.keys(), pred))

        hash = pred["hash"]

        if hash not in dataset_dict:
            continue

        sample = dataset_dict[hash]
        gold = sample["value"]["msg"]

        for column in pred.keys():
            if column in columns_to_ignore:
                continue
            values = pred[column]
            if values is None:
                values = [{"answer": "", "value": ""}]
            bleu_score = -1
            final_response = ""
            for value in values:
                new_bleu_score = calculate_bleu(value['answer'], gold)
                logging.info(f"BLEU score: {new_bleu_score}")
                if new_bleu_score > bleu_score:
                    bleu_score = new_bleu_score
                    final_response = value
            new_sample = new_dataset.get(hash)
            final_response["bleu"] = bleu_score
            if new_sample is None:
                new_sample = {
                    "hash": hash,
                    "few_shot_prompt": pred["few_shot_prompt"],
                    "gold": gold,
                    column: final_response
                }
                if "zero_shot_prompt" in pred:
                    new_sample["zero_shot_prompt"] = pred["zero_shot_prompt"]
                new_dataset[hash] = new_sample
            else:
                new_dataset[hash][column] = final_response
    return list(new_dataset.values())


if __name__ == "__main__":
    init_logging(local_drive_mount)

    n_jobs = min(8, mp.cpu_count())

    test_name = "test_500"
    base_dataset = load_dataset(f"dbaeka/soen_691_msg_{test_name}_hashed")['test']
    pred_dataset = load_dataset(f"dbaeka/soen_691_{test_name}_hashed_with_results")['test']

    chunk_size = len(pred_dataset) // (n_jobs - 1)
    pred_dataset_chunks = pred_dataset.batch(chunk_size)

    with mp.Pool(n_jobs) as pool:
        new_datasets = list(tqdm(
            pool.imap(partial(build_results, dataset=base_dataset, chunks=pred_dataset_chunks), range(n_jobs)),
            desc=f"Building {test_name} dataset"
        ))

    new_dataset = [item for sublist in new_datasets for item in sublist]
    hf_dataset = Dataset.from_list(new_dataset)
    hf_dataset = DatasetDict({"test": hf_dataset})

    hf_dataset.push_to_hub(f"dbaeka/soen_691_{test_name}_final_selected_results")
