import asyncio
import logging
import os
from datetime import datetime

import toml
from ollama import AsyncClient
from tqdm import tqdm

from src.service.common import extract_cot_and_answer, get_unprocessed_examples, save_results

ollama_model_map = {
    # "Qwen/QwQ-32B": "qwen-qwq-32b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5:1.5b-instruct",
    # "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen-2.5-coder-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-r1:1.5b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-r1:7b"
}


async def get_response(
        messages,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        seed: int = None,
        is_reasoning_model: bool = False,
        host: str = None,
        gpu_index: int = None
):
    try:
        # Start timing
        start_time = datetime.now()

        client = AsyncClient(
            host=os.environ.get(host),
        )
        mapped_model = ollama_model_map.get(model_name)
        logging.debug(f"Model: {mapped_model}")
        response = await client.chat(
            model=mapped_model,
            messages=messages,
            keep_alive=-1,
            stream=False,
            options={
                # "num_predict": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
                "num_ctx": 4096,
                "stop": ["</s>"],
                "top_p": 1,
            }
        )

        # Stop timing
        end_time = datetime.now()
        # Calculate duration and word count
        duration = (end_time - start_time).total_seconds()
        word_count = len(response.message.content.split())

        # Calculate words per second (WPS)
        wps = word_count / duration if duration > 0 else 0

        # Log the GPU index, word count, duration, and words per second
        logging.debug(f"Host: {host}, GPU: {gpu_index}, Words: {word_count}, Duration: {duration:.2f}s, WPS: {wps:.2f}")

        logging.debug(f"Model Response: {response.message.content}")
        logging.debug("_" * 70)
        return [extract_cot_and_answer(response.message.content, is_reasoning_model)]
    except Exception as e:
        logging.error(f"Error on host {host}: {e}")


async def forward(
        messages,
        model_name: str,
        max_new_tokens: int = 4089,
        temperature: float = 0.05,
        batch_call: bool = True,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False,
        host: str = None,
        gpu_index: int = None
):
    logging.debug("Generating model response")
    results = []
    if not batch_call:
        for i in range(num_of_results):
            logging.debug(f"Result {i + 1}")
            logging.debug("_" * 70)
            result = await get_response(
                messages,
                model_name,
                max_new_tokens,
                temperature,
                seed,
                is_reasoning_model,
                host=host,
                gpu_index=gpu_index
            )
            if result and len(result) > 0:
                results.append(result[0])
        results = [results]
    else:
        logging.debug("Result")
        logging.debug("_" * 70)
        result = await get_response(messages, model_name, max_new_tokens, temperature, seed,
                                    is_reasoning_model, host=host, gpu_index=gpu_index)
        results.append(result)
    return results


async def worker(host, gpu_index, task_queue, temperature, batch_call, num_of_results, seed, model_name,
                 is_reasoning_model, existing_results, output_path):
    """Worker function to process tasks using the specified host."""
    progress_bar = tqdm(total=task_queue.qsize(), desc=f"Processing on {host}")
    while not task_queue.empty():
        inputs = await task_queue.get()
        results = await forward(inputs['prompt'],
                                model_name=model_name,
                                temperature=temperature, batch_call=batch_call,
                                num_of_results=num_of_results, seed=seed, is_reasoning_model=is_reasoning_model,
                                host=host, gpu_index=gpu_index
                                )
        save_results([inputs], results, existing_results, output_path)
        progress_bar.update(1)
        task_queue.task_done()


async def task_to_run(
        instance_index,
        model_name: str,
        test_name: str,
        shard_indices: list,
        base_dir: str,
        temperature: float = 0.7,
        batch_call: bool = False,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False,
):
    config_path = os.path.join(os.path.dirname(__file__), "../../ollama_config.toml")
    config = toml.load(config_path)
    gpus = config["ollama_instances"]

    shard_index = shard_indices[instance_index]
    print(f"Processing instance {instance_index} for shard {shard_index}")
    logging.info(f"Processing instance {instance_index} for shard {shard_index}")

    filtered_input, existing_results, output_path = get_unprocessed_examples(
        base_dir, model_name, test_name,
        shard_index, num_of_results,
        is_reasoning_model
    )

    task_queue = asyncio.Queue()
    for inputs in filtered_input:
        task_queue.put_nowait(inputs)

    # Create a list of worker tasks, one for each Ollama instance
    tasks = []
    for host, gpu_index in gpus.items():
        tasks.append(worker(
            task_queue=task_queue,
            temperature=temperature,
            batch_call=batch_call,
            num_of_results=num_of_results,
            seed=seed,
            model_name=model_name,
            is_reasoning_model=is_reasoning_model,
            host=host,
            gpu_index=gpu_index,
            existing_results=existing_results,
            output_path=output_path
        ))

    # Await the completion of all worker tasks
    await asyncio.gather(*tasks)

    logging.info(f"Completed processing shard {shard_index}")


def review_comment_generation(
        instance_index,
        model_name: str,
        test_name: str,
        shard_indices: list,
        base_dir: str,
        temperature: float = 0.7,
        batch_call: bool = False,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False,
        batch_size: int = 32,
        pause_duration: int = 4
):
    asyncio.run(task_to_run(
        instance_index,
        model_name,
        test_name,
        shard_indices,
        base_dir,
        temperature,
        batch_call,
        num_of_results,
        seed,
        is_reasoning_model
    ))
