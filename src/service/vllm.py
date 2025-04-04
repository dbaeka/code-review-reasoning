import logging
from time import sleep

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.service.common import extract_cot_and_answer, get_unprocessed_examples, save_results
from vllm import LLM, SamplingParams

vllm_model = None
tokenizer = None

USE_BNB = False

MAX_ATTEMPT = 3
MAX_NEW_TOKENS = 2048

KEEP_INSTRUCTION = False


def load_model(model_name: str):
    global vllm_model, tokenizer
    if vllm_model is not None:
        return vllm_model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if USE_BNB:
        vllm_model = LLM(
            model_name,
            dtype=torch.bfloat16,
            quantization="bitsandbytes",
            load_format="bitsandbytes"
        )
    else:
        vllm_model = LLM(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return vllm_model, tokenizer


def forward(
        inputs,
        model,
        tokenizer,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.05,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False
):
    logging.debug("Generating model response")

    if not is_reasoning_model:
        max_new_tokens = 250

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=num_of_results,
        seed=seed if seed is not None else None,
        top_p=1,
        stop=["</s>"]
    )
    outputs = model.generate(
        inputs,
        sampling_params=sampling_params,
    )
    all_results = []
    for idx, output in enumerate(outputs):
        result = []
        for seq_idx, multi_result_output in enumerate(output.outputs):
            logging.debug(
                f"Prompt {idx + 1} - Sequence {seq_idx + 1}: {multi_result_output.text.replace(tokenizer.pad_token, '')}")
            logging.debug("_" * 70)
            result.append(extract_cot_and_answer(multi_result_output.text, is_reasoning_model))
        all_results.append(result)
    logging.debug(f"Total number of results for the batch: {len(all_results)}")
    return all_results


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

    attempt = 0
    while len(filtered_input) > 0 and attempt < MAX_ATTEMPT:
        model, tokenizer = load_model(model_name)
        max_model_len = tokenizer.model_max_length
        max_new_tokens = MAX_NEW_TOKENS
        for i in tqdm(range(0, len(filtered_input), batch_size)):
            end_index = min(i + batch_size, len(filtered_input))
            batch = filtered_input[i:end_index]

            prompts = [v["prompt"] for v in batch]

            if KEEP_INSTRUCTION:
                instruction_prompt = prompts[0][0]
                prompts = [v[1:] for v in prompts]
                instruction_text = tokenizer.apply_chat_template([instruction_prompt], tokenize=False)
                instruction_tokens = tokenizer(instruction_text)["input_ids"]
                instruction_tokens_len = len(instruction_tokens)
            else:
                instruction_tokens = []
                instruction_tokens_len = 0

            texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)

            tokens = tokenizer(texts)["input_ids"]
            clipped_tokens = [token[-(max_model_len - instruction_tokens_len - max_new_tokens):] for token in tokens]

            clipped_tokens = [instruction_tokens + token for token in clipped_tokens]
            texts = tokenizer.batch_decode(clipped_tokens, skip_special_tokens=True)

            print(f"Processing batch {i} to {end_index}")
            logging.debug(f"Processing batch {i} to {end_index}")
            try:
                results = forward(
                    texts,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    num_of_results=num_of_results,
                    seed=seed,
                    is_reasoning_model=is_reasoning_model
                )
                save_results(batch, results, existing_results, output_path)
            except Exception as e:
                logging.error("Error: ", e)
                sleep(pause_duration)

        logging.info(f"Completed processing shard {shard_index}")
        logging.info(f"Attempt {attempt} to process shard {shard_index}")
        attempt += 1
        filtered_input, existing_results, output_path = get_unprocessed_examples(
            base_dir, model_name, test_name,
            shard_index, num_of_results,
            is_reasoning_model, dir_prefix
        )
