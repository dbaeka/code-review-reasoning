import asyncio
import logging
import os
from datetime import datetime

import toml
from openai import AsyncClient
from tqdm import tqdm

from src.service.common import get_unprocessed_examples, save_results


async def get_response(
        messages,
        model_name: str,
        max_new_tokens: int,
        num_of_results: int,
        temperature: float,
        seed: int = None,
        host: str = None,
        gpu_index: int = None
):
    try:
        # Start timing
        start_time = datetime.now()

        client = AsyncClient(
            base_url=f"http://{host}/v1",
        )
        logging.debug(f"Model: {model_name}")

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=2,
            stop=["</s>"],
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=seed
        )

        # Stop timing
        end_time = datetime.now()
        # Calculate duration and word count
        duration = (end_time - start_time).total_seconds()

        # Log the GPU index, word count, duration, and words per second
        logging.debug(f"Host: {host}, GPU: {gpu_index}, Duration: {duration:.2f}s")
        return response
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
    response = await get_response(messages, model_name, max_new_tokens, num_of_results, temperature, seed,
                                  host=host, gpu_index=gpu_index)
    all_results = []
    if response is not None:
        for idx, choice in enumerate(response.choices):
            result = []
            for seq_idx, multi_result_output in enumerate(output.outputs):
                thinking = choice.message.reasoning_content if is_reasoning_model else "NO THINKING"
                logging.debug(
                    f"Prompt {idx + 1} - Sequence {seq_idx + 1}: {thinking + "\n\n" + choice.message.content}")
                logging.debug("_" * 70)
                final_output = {"cot": thinking, "answer": choice.message.content}
                result.append(final_output)
            all_results.append(result)
    logging.debug(f"Total number of results for the batch: {len(all_results)}")
    return all_results


async def worker(host, gpu_index, task_queue, temperature, batch_call, num_of_results, seed, model_name,
                 is_reasoning_model, existing_results, output_path):
    """Worker function to process tasks using the specified host."""
    progress_bar = tqdm(total=task_queue.qsize(), desc=f"Processing on {host}")
    batch_size = 2
    while not task_queue.empty():
        prompts = []
        batch = []
        for _ in range(batch_size):
            if task_queue.empty():
                break
            inputs = await task_queue.get()
            prompts.append(inputs['prompt'])
            batch.append(inputs)
        results = await forward(prompts,
                                model_name=model_name,
                                temperature=temperature, batch_call=batch_call,
                                num_of_results=num_of_results, seed=seed, is_reasoning_model=is_reasoning_model,
                                host=host, gpu_index=gpu_index
                                )
        save_results(batch, results, existing_results, output_path)
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
    config_path = os.path.join(os.path.dirname(__file__), "../../config.toml")
    config = toml.load(config_path)
    gpus = config["instances"]

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
