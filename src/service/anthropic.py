import json
import logging
import os
import uuid
from time import sleep

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from tqdm import tqdm

from src.service.common import get_unprocessed_examples


def build_requests():
    pass


def forward(
        batches,
        num_of_results: int = 1,
):
    logging.debug("Building batch for response")

    # generate unique custom_id for each request
    requests = []
    for batch in batches:
        prompt = batch["prompt"]
        hash = batch["hash"]
        for i in range(num_of_results):
            request = {
                "custom_id": str(uuid.uuid4()),
                "hash": hash,
                "system": {
                    "text": prompt[0]["content"],
                    "type": "text"
                },
                "messages": prompt[1:],
                "stop_sequence": ["</s>"]
            }
            requests.append(request)

    # create a batch of requests
    batch_requests = []
    for request in requests:
        batch_requests.append(
            Request(
                custom_id=request["custom_id"],
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=20000,
                    system=[request["system"]],
                    messages=request["messages"],
                    stop_sequences=request["stop_sequence"],
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 2048
                    },
                )
            )
        )

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message_batch = client.messages.batches.create(requests=batch_requests)

    logging.debug("Processing batch response: %s", message_batch)

    return {
        "id": message_batch.id,
        "response": message_batch.to_json(),
        "requests": requests
    }


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
        is_reasoning_model: bool = False,
        dir_prefix: str = ""
):
    shard_index = shard_indices[instance_index]
    print(f"Processing instance {instance_index} for shard {shard_index}")
    logging.info(f"Processing instance {instance_index} for shard {shard_index}")

    filtered_input, existing_results, output_path = get_unprocessed_examples(
        base_dir, model_name, test_name,
        shard_index, num_of_results,
        is_reasoning_model, dir_prefix
    )

    for i in tqdm(range(0, len(filtered_input), batch_size)):
        end_index = min(i + batch_size, len(filtered_input))
        batch = filtered_input[i:end_index]

        prompts = [{"prompt": v["prompt"], "hash": v["hash"]} for v in batch]

        print(f"Processing batch {i} to {end_index}")
        logging.debug(f"Processing batch {i} to {end_index}")
        try:
            batch_info = forward(
                prompts,
                num_of_results=num_of_results,
            )
            # save batch info
            # create directory if it doesn't exist
            os.makedirs(f"{base_dir}/anthropic", exist_ok=True)
            with open(f"{base_dir}/anthropic/{shard_index}_batch.json", "a") as f:
                f.write(json.dumps(batch_info, indent=4))
        except Exception as e:
            logging.error("Error: ", e)
            sleep(pause_duration)

    logging.info(f"Completed processing shard {shard_index}")
