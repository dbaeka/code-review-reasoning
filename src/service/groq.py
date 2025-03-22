import logging
import os
from time import sleep

from groq import Groq
from tqdm import tqdm

from src.service.common import extract_cot_and_answer, get_unprocessed_examples, save_results

groq_model_map = {
    "Qwen/QwQ-32B": "qwen-qwq-32b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-32b",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen-2.5-coder-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-r1-distill-qwen-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b-specdec"
}


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


def review_comment_generation(
        instance_index,
        model_name: str,
        test_name: str,
        shard_indices: list,
        base_dir: str,
        batch_size: int = 32,
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

    filtered_input, existing_results, output_path = get_unprocessed_examples(
        base_dir, model_name, test_name,
        shard_index, num_of_results,
        is_reasoning_model
    )

    for i in tqdm(range(0, len(filtered_input), batch_size)):
        end_index = min(i + batch_size, len(filtered_input))
        batch = filtered_input[i:end_index]

        prompts = [v["prompt"] for v in batch]

        print(f"Processing batch {i} to {end_index}")
        logging.debug(f"Processing batch {i} to {end_index}")
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
            save_results(batch, results, existing_results, output_path)

            sleep(pause_duration)
        except Exception as e:
            logging.error("Error: ", e)
            sleep(3)

    logging.info(f"Completed processing shard {shard_index}")
