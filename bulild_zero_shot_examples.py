import multiprocessing as mp
from functools import partial

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from src.common.tools import init_logging

local_drive_mount = "/Users/dbaeka/Library/CloudStorage/GoogleDrive-dbaekajnr@gmail.com/My Drive/"

train_dataset = load_dataset("dbaeka/soen_691_msg_train")['train']

INSTRUCTION_PROMPT = ("Please GIVE FORMAL Codereview for software developers in ONE SENTENCE for testcase, "
                      "implementing Few Shot Learning from example. Dont start with Codereview/review. Just give the "
                      "answer.")

NUM_OF_FEW_SHOT = 0
SEED = 0


def build_dataset(instance, dataset_chunks):
    dataset = dataset_chunks[instance]
    new_dataset = []
    for sample in tqdm(zip(dataset['hash'], dataset['value'], dataset['bm_indices']), total=len(dataset['hash']),
                       desc=f"Processing sample {instance}"):
        data = {"hash": sample[0], "value": sample[1]}
        dialog = [{"role": "user", "content": INSTRUCTION_PROMPT}]
        dialog_thinking = [{"role": "user", "content": INSTRUCTION_PROMPT}]

        test_code = data["value"]["patch"]

        context = ""
        context = context + "Code: \t" + test_code + "\n"
        context = context + "Codereview: "

        dialog.append({"role": "user", "content": context})
        dialog_thinking.append({"role": "user", "content": context})
        data["prompt_base"] = dialog
        data["prompt_thinking"] = dialog_thinking
        new_dataset.append(data)
    return new_dataset


if __name__ == "__main__":
    init_logging(local_drive_mount)

    test_name = "test_5000"

    n_jobs = min(8, mp.cpu_count())

    dataset = load_dataset(f"dbaeka/soen_691_{test_name}_bm_25_indices_hashed")['test']

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

    hf_dataset.push_to_hub(f"dbaeka/soen_691_zero_shot_{test_name}_hashed")
