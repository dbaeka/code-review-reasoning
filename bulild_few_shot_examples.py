import multiprocessing as mp
from functools import partial

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from src.common.tools import init_logging

local_drive_mount = "/Users/dbaeka/Library/CloudStorage/GoogleDrive-dbaekajnr@gmail.com/My Drive/"

train_dataset = load_dataset("dbaeka/soen_691_msg_train")['train']

INSTRUCTION_PROMPT = (
    "Please Give FORMAL Codereview for software developers in one sentence from the given diff hunk. "
    "Don’t start with Codereview/review. Just give the answer. "
    "Five examples are given before with code and code review and the code to work on is given after.")

WITH_SUMMARY = True
WITH_CALLGRAPH = True
NUM_OF_FEW_SHOT = 5
SEED = 0


def get_bm25_review_context(indices, train_data, num_shot: int = 1):
    indices = indices[-num_shot:]
    msg = []
    for i in indices:
        context = ""
        context = context + "Code: \t" + train_data["patch"][i] + "\n"
        if WITH_SUMMARY:
            context = context + "Summary: \t" + train_data["summary"][i] + "\n"
        if WITH_CALLGRAPH:
            context = context + "Callgraph: \t" + train_data["callgraph"][i] + "\n"
        context = context + "Codereview: "
        msg.append({"role": "user", "content": context})
        context = train_data["msg"][i] + " </s>" + "\n\n"
        msg.append({"role": "assistant", "content": context})
    return msg


def build_dataset(instance, dataset_chunks):
    dataset = dataset_chunks[instance]
    new_dataset = []
    for sample in tqdm(zip(dataset['hash'], dataset['value'], dataset['bm_indices']), total=len(dataset['hash']),
                       desc=f"Processing sample {instance}"):
        data = {"hash": sample[0], "value": sample[1]}
        dialog = [{"role": "user", "content": INSTRUCTION_PROMPT}]

        context_msg = get_bm25_review_context(sample[2], train_dataset,
                                              num_shot=NUM_OF_FEW_SHOT)
        dialog.extend(context_msg)

        test_code = data["value"]["patch"]
        test_summary = data["value"]["summary"]
        test_callgraph = data["value"]["callgraph"]

        context = ""
        context = context + "Code: \t" + test_code + "\n"
        if WITH_SUMMARY:
            context = context + "Summary: \t" + test_summary + "\n"
        if WITH_CALLGRAPH:
            context = context + "Callgraph: \t" + test_callgraph + "\n"
        context = context + "Codereview: "

        dialog.append({"role": "user", "content": context})
        data["prompt"] = dialog
        new_dataset.append(data)
    return new_dataset


if __name__ == "__main__":
    init_logging(local_drive_mount)

    test_name = "test_500"

    n_jobs = min(8, mp.cpu_count())

    dataset = load_dataset(f"dbaeka/soen_691_{test_name}0_bm_25_indices_hashed")['test']
    dataset = dataset.select(range(0, 500))

    chunk_size = len(dataset) // (n_jobs - 1)
    dataset_chunks = dataset.batch(chunk_size)

    with mp.Pool(n_jobs) as pool:
        new_dataset = list(tqdm(
            pool.imap(partial(build_dataset, dataset_chunks=dataset_chunks), range(n_jobs)),
            total=len(dataset_chunks),
            desc="Building dataset"
        ))
    new_dataset = [item for sublist in new_dataset for item in sublist]
    hf_dataset = Dataset.from_list(new_dataset)
    hf_dataset = DatasetDict({"test": hf_dataset})

    output_dir = "_base" if not WITH_SUMMARY and not WITH_CALLGRAPH else ""
    if WITH_SUMMARY:
        output_dir += "_summary"
    if WITH_CALLGRAPH:
        output_dir += "_callgraph"
    hf_dataset.push_to_hub(f"dbaeka/soen_691_few_shot_{test_name}{output_dir}_hashed")
