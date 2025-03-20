import logging
import os
import re
from time import sleep

from groq import Groq
from tqdm import tqdm

from src.common.tools import get_bm25_scores
from src.utils.ioutils import jdump, jload
from src.utils.string import prettify

INSTRUCTION_PROMPT = ("Please GIVE FORMAL Codereview for software developers in ONE SENTENCE for testcase, "
                      "implementing Few Shot Learning from example. Dont start with Codereview/review. Just give the "
                      "answer.")

groq_model_map = {
    "Qwen/QwQ-32B": "qwen-qwq-32b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-32b",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen-2.5-coder-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-r1-distill-qwen-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b-specdec"
}


def extract_cot_and_answer(response, is_reasoning_model: bool = False):
    # Extract content within <think>...</think>
    cot_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    cot = cot_match.group(1).strip() if cot_match else ""

    if not is_reasoning_model:
        cot = "NO THINKING"

    # Extract content after </think>
    answer_match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    return {"cot": cot, "answer": answer}


def get_response(
        messages,
        model_name: str,
        num_of_results: int,
        max_new_tokens: int,
        temperature: float,
        seed: int = None,
        is_reasoning_model: bool = False,
        instance_index: int = None
):
    if instance_index is None:
        api_key = os.environ.get("GROQ_API_KEY_0")
    else:
        api_key = os.environ.get(f"GROQ_API_KEY_{instance_index}")
    client = Groq(
        api_key=api_key,
    )
    mapped_model = groq_model_map.get(model_name)
    logging.debug(f"Model: {mapped_model}")
    response = client.chat.completions.create(
        model=mapped_model,
        messages=messages[0],
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=num_of_results,
        stop=["</s>"],
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=seed
    )
    logging.debug(f"Fingerprints: {response.system_fingerprint}")
    result = []
    for choice in response.choices:
        logging.debug(f"Model Response: {choice.message.content}")
        logging.debug("_" * 70)
        result.append(extract_cot_and_answer(choice.message.content, is_reasoning_model))
    return result


def forward(
        messages,
        model_name: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.05,
        batch_call: bool = True,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False,
        instance_index: int = None
):
    logging.debug("Generating model response")
    results = []
    if not batch_call:
        for i in range(num_of_results):
            logging.debug(f"Result {i + 1}")
            logging.debug("_" * 70)
            result = get_response(
                messages,
                model_name,
                1,
                max_new_tokens,
                temperature,
                seed,
                is_reasoning_model
            )
            results.append(result[0])
            sleep(1)
        results = [results]
    else:
        logging.debug("Result")
        logging.debug("_" * 70)
        result = get_response(messages, model_name, num_of_results, max_new_tokens, temperature, seed,
                              is_reasoning_model, instance_index)
        results.append(result)
    return results


def get_bm25_review_context(bm25, example, train_data, num_shot: int = 1, with_summary: bool = False,
                            with_callgraph: bool = False):
    tokenized_query = example.split(" ")
    sorted_indices, _ = get_bm25_scores(bm25, tokenized_query, top_k=num_shot)
    msg = []
    for i in sorted_indices:
        context = ""
        context = context + "Code: \t" + train_data["patch"][i] + "\n"
        if with_summary:
            context = context + "Summary: \t" + train_data["summary"][i] + "\n"
        if with_callgraph:
            context = context + "Callgraph: \t" + train_data["callgraph"][i] + "\n"
        context = context + "Codereview: "
        msg.append({"role": "user", "content": context})
        context = "<think>\n...some explantion here...\n</think>\n\n" + train_data["msg"][i] + " </s>" + "\n\n"
        msg.append({"role": "assistant", "content": context})
    return msg


def review_comment_generation(
        instance_index,
        model_name: str,
        test_name: str,
        shard_indices: list,
        base_dir: str,
        train_dataset,
        bm25,
        batch_size: int = 32,
        num_of_few_shot: int = 1,
        with_summary: bool = False,
        with_callgraph: bool = False,
        temperature: float = 0.7,
        pause_duration: int = 4,
        batch_call: bool = False,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False
):
    shard_index = shard_indices[instance_index]
    print(f"Processing instance {instance_index} for shard {shard_index}")
    logging.info(f"Processing instance {instance_index} for shard {shard_index}")

    input_dir = os.path.join(base_dir, prettify(model_name), f"{test_name}_input")
    input_path = os.path.join(input_dir, f"shard_{shard_index}_input.json")
    input_data = jload(input_path)
    input_list = [{"hash": h, "value": v} for h, v in zip(input_data["hash"], input_data["value"])]

    output_dir = os.path.join(base_dir, prettify(model_name), f"{test_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"shard_{shard_index}_output.json")

    # Load existing results if they exist
    existing_results = jload(output_path) if os.path.exists(output_path) else {}

    # Filter out already processed hashes with 5 results
    filtered_input = [
        sample for sample in input_list
        if sample["hash"] not in existing_results or len(existing_results[sample["hash"]]) != num_of_results
    ]

    for i in tqdm(range(0, len(filtered_input), batch_size)):
        end_index = min(i + batch_size, len(filtered_input))
        batch = filtered_input[i:end_index]

        print(f"Processing batch {i} to {end_index}")
        logging.debug(f"Processing batch {i} to {end_index}")

        prompts = []
        for j in range(len(batch)):
            dialog = [{"role": "user", "content": INSTRUCTION_PROMPT}]
            context_msg = get_bm25_review_context(bm25, batch[j]["value"]["patch"], train_dataset,
                                                  num_shot=num_of_few_shot)
            dialog.extend(context_msg)

            test_code = batch[j]["value"]["patch"]
            test_summary = batch[j]["value"]["summary"]
            test_callgraph = batch[j]["value"]["callgraph"]

            context = ""
            context = context + "Code: \t" + test_code + "\n"
            if with_summary:
                context = context + "Summary: \t" + test_summary + "\n"
            if with_callgraph:
                context = context + "Callgraph: \t" + test_callgraph + "\n"
            context = context + "Codereview: "

            dialog.append({"role": "user", "content": context})
            prompts.append(dialog)

            logging.debug("################context ####################")
            logging.debug(dialog)
        try:
            results = forward(
                prompts,
                temperature=temperature,
                batch_call=batch_call,
                num_of_results=num_of_results,
                seed=seed,
                model_name=model_name,
                is_reasoning_model=is_reasoning_model,
                instance_index=instance_index
            )
            for sample, result in zip(batch, results):
                filtered_result = [
                    r for r in result
                    if r.get("cot", "").strip() != "" or r.get("answer", "").strip() != ""
                ]
                if filtered_result:
                    existing_results[sample["hash"]] = filtered_result
            jdump(existing_results, output_path)

            sleep(pause_duration)
        except Exception as e:
            logging.error("Error: ", e)
            sleep(3)

    logging.info(f"Completed processing shard {shard_index}")
