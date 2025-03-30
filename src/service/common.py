import os
import re

from src.utils.ioutils import jload, jdump
from src.utils.string import prettify


def extract_cot_and_answer(response, is_reasoning_model: bool = False):
    think_start_token = "<think>"
    think_end_token = "</think>"

    if is_reasoning_model and think_end_token not in response:
        return {"cot": "", "answer": response}

    # Add a start token if it's missing to keep compatibility.
    if is_reasoning_model and think_start_token not in response:
        response = f"{think_start_token}{response}"

    # Extract content within <think>...</think>
    cot_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    cot = cot_match.group(1).strip() if cot_match else ""

    if not is_reasoning_model:
        cot = "NO THINKING"
        answer = response
    else:
        # Extract content after </think>
        answer_match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""
    return {"cot": cot, "answer": answer}


def get_unprocessed_examples(
        base_dir: str,
        model_name: str,
        test_name: str,
        shard_index: int,
        num_of_results: int,
        is_reasoning_model: bool,
        dir_prefix: str = ""
):
    input_dir = os.path.join(base_dir, f"{test_name}{dir_prefix}_input")
    input_path = os.path.join(input_dir, f"shard_{shard_index}_input.json")
    input_data = jload(input_path)
    prompt_path = "prompt_thinking" if is_reasoning_model else "prompt_base"
    input_list = [{"hash": h, "value": v, "prompt": p} for h, v, p in
                  zip(input_data["hash"], input_data["value"], input_data[prompt_path])]

    output_dir = os.path.join(base_dir, prettify(model_name), f"{test_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"shard_{shard_index}{dir_prefix}_output.json")

    # Load existing results if they exist
    existing_results = jload(output_path) if os.path.exists(output_path) else {}

    # Filter out already processed hashes with 5 results
    filtered_input = [
        sample for sample in input_list
        if sample["hash"] not in existing_results or len(existing_results[sample["hash"]]) != num_of_results
    ]

    return filtered_input, existing_results, output_path


def save_results(
        batch,
        results,
        existing_results,
        output_path
):
    for sample, result in zip(batch, results):
        filtered_result = [
            r for r in result
            if r.get("cot", "").strip() != "" and r.get("answer", "").strip() != "" and
               r.get("cot", "").strip() != "...some explanation here..."
        ]
        if len(filtered_result) > 4:
            existing_results[sample["hash"]] = filtered_result
            jdump(existing_results, output_path)
