import copy
import logging
import re
from time import sleep

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.service.common import extract_cot_and_answer, get_unprocessed_examples, save_results
from vllm import LLM, SamplingParams

vllm_model = None
tokenizer = None

USE_BNB = True

BUDGET_THINKING_TEMP = 0.0

MAX_ATTEMPT = 8
MAX_NEW_TOKENS = 2048
MAX_TOKENS_THINKING = 32000


def load_model(model_name: str, seed: int = None):
    global vllm_model, tokenizer
    if vllm_model is not None:
        return vllm_model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if USE_BNB:
        vllm_model = LLM(
            model_name,
            dtype=torch.bfloat16,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            seed=seed
        )
    else:
        vllm_model = LLM(model_name, seed=seed)
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


def forward_with_budget(
        prompts,
        model,
        tokenizer,
        max_new_tokens: int = 2048,
        temperature: float = 0.05,
        num_of_results: int = 1,
        seed: int = None,
        budget: int = 1
):
    logging.debug(f"Generating model response with budget forcing budget: {budget}")

    stop_token_ids = tokenizer("</think>")["input_ids"]

    final_results = []
    texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        n=1,
        seed=seed if seed is not None else None,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=BUDGET_THINKING_TEMP,
        top_p=1,
    )
    outputs = model.generate(
        texts,
        sampling_params=sampling_params,
    )
    max_tokens_thinking_length = float('-inf')
    ignore_str = "Wait"

    for idx, output in enumerate(outputs):
        for seq_idx, multi_result_output in enumerate(output.outputs):
            result = multi_result_output.text
            max_tokens_thinking_length = max(max_tokens_thinking_length, len(result))
            result = result.replace("</think>", "")
            logging.debug(f"Pre-Result:  {result}")
            logging.debug("_" * 70)
            prompts[idx].append({"role": "assistant", "content": "<think>\n" + result + ignore_str})
            final_results.append({"cot": result + ignore_str})

    max_tokens_thinking_tmp = MAX_TOKENS_THINKING
    if max_tokens_thinking_tmp > 0:
        for i in range(budget):
            max_tokens_thinking_tmp -= max_tokens_thinking_length
            sampling_params = SamplingParams(
                max_tokens=min(max_new_tokens, max_tokens_thinking_tmp),
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                seed=seed if seed is not None else None,
                n=1,
                top_p=1,
                temperature=BUDGET_THINKING_TEMP,
            )
            texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
            texts = [re.sub(r"<｜end▁of▁sentence｜><｜Assistant｜>(<think>\n)?", "", item) for item in texts]
            outputs = model.generate(
                texts,
                sampling_params=sampling_params
            )
            for idx, output in enumerate(outputs):
                for seq_idx, multi_result_output in enumerate(output.outputs):
                    result = multi_result_output.text
                    result = result.replace("</think>", "")
                    logging.debug(f"Result {i}:  {result}")
                    logging.debug("_" * 70)
                    prompts[idx].append(
                        {"role": "assistant", "content": result + (ignore_str if (i + 1) != budget else "")})
                    final_results[idx]["cot"] = final_results[idx]["cot"] + result + (
                        ignore_str if (i + 1) != budget else "")

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=1,
        seed=seed if seed is not None else None,
        top_p=1,
        stop=["</s>"]
    )

    texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
    texts = [re.sub(r"<｜end▁of▁sentence｜><｜Assistant｜>(<think>\n)?", "", item) for item in texts]
    texts = [item + "</think>" for item in texts]
    outputs = model.generate(
        texts,
        sampling_params=sampling_params,
    )
    for idx, output in enumerate(outputs):
        for seq_idx, multi_result_output in enumerate(output.outputs):
            final = extract_cot_and_answer(multi_result_output.text, True)
            logging.debug(f"Final Result: {final}")
            logging.debug("_" * 70)
            final_results[idx]["cot"] = final_results[idx]["cot"] + final["cot"]
            final_results[idx]["answer"] = final["answer"]

    all_results = []
    for idx in range(0, len(final_results), num_of_results):
        result = []
        for j in range(num_of_results):
            if idx + j < len(final_results):
                logging.debug(
                    f"Prompt {idx + j} - Sequence {j + 1} With Budget Forcing: {final_results[idx + j]}")
                logging.debug("_" * 70)
                result.append(final_results[idx + j])
        all_results.append(result)
    logging.debug(f"Total number of results for the batch: {len(all_results)}")
    return all_results


def forward_with_budget_single(
        prompts,
        model,
        tokenizer,
        max_new_tokens: int = 2048,
        temperature: float = 0.05,
        num_of_results: int = 1,
        seed: int = None,
        budget: int = 1
):
    logging.debug(f"Generating model response with budget forcing budget: {budget}")

    stop_token_ids = tokenizer("</think>")["input_ids"]

    results = []
    for i, prompt in tqdm(enumerate(prompts)):
        text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            n=1,
            seed=seed if seed is not None else None,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0,
        )
        outputs = model.generate(
            text,
            sampling_params=sampling_params,
        )
        ignore_str = "Wait"
        max_tokens_thinking_tmp = MAX_TOKENS_THINKING
        if max_tokens_thinking_tmp > 0:
            for j in range(budget):  # Num of times to skip stop token
                max_tokens_thinking_tmp -= len(outputs[0].outputs[0].token_ids)
                result = outputs[0].outputs[0].text
                result = result.replace("</think>", "")
                prompt.append({"role": "assistant", "content": "<think>\n" + result + ignore_str})
                sampling_params = SamplingParams(
                    max_tokens=min(max_new_tokens, max_tokens_thinking_tmp),
                    min_tokens=1,
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    seed=seed if seed is not None else None,
                    n=1,
                    top_p=1,
                    temperature=0.0,
                )
                text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
                text = re.sub(r"<｜end▁of▁sentence｜><｜Assistant｜>(<think>\n)?", "", text)
                outputs = model.generate(
                    text,
                    sampling_params=sampling_params
                )

        result = outputs[0].outputs[0].text
        result = result.replace("</think>", "")
        prompt.append({"role": "assistant", "content": result})

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=1,
            seed=seed if seed is not None else None,
            top_p=1,
            stop=["</s>"]
        )

        text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        text = re.sub(r"<｜end▁of▁sentence｜><｜Assistant｜>(<think>\n)?", "", text)
        text += "</think>"
        outputs = model.generate(
            text,
            sampling_params=sampling_params,
        )
        multi_result_output = outputs[0].outputs[0]
        logging.debug(
            f"Prompt {i + 1} - With Budget Forcing: {multi_result_output.text.replace(tokenizer.pad_token, '')}")
        prior_thinking_tokens = "\n".join(item["content"] for item in prompt[-(budget + 1):])
        final = extract_cot_and_answer(multi_result_output.text, True)
        final["cot"] = prior_thinking_tokens.replace("<think>", "") + "\n" + final["cot"]
        results.append(final)
        logging.debug(f"Result {i + 1}: {final}")
        logging.debug("_" * 70)

    all_results = []
    for idx in range(0, len(results), num_of_results):
        result = []
        for j in range(num_of_results):
            if idx + j < len(results):
                result.append(results[idx + j])
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
        model, tokenizer = load_model(model_name, seed)
        for i in tqdm(range(0, len(filtered_input), batch_size)):
            end_index = min(i + batch_size, len(filtered_input))
            batch = filtered_input[i:end_index]

            prompts = [v["prompt"] for v in batch]

            texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)

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


def budget_force_infer(
        model_name: str,
        test_name: str,
        shard_index: int,
        base_dir: str,
        batch_size: int = 32,
        temperature: float = 0.7,
        pause_duration: int = 4,
        num_of_results: int = 1,
        budget: int = 1,
        seed: int = None,
        dir_prefix: str = ""
):
    print(f"Processing shard {shard_index}")
    logging.info(f"Processing shard {shard_index}")

    filtered_input, existing_results, output_path = get_unprocessed_examples(
        base_dir, model_name, test_name,
        shard_index, num_of_results,
        True, dir_prefix
    )
    attempt = 0
    while len(filtered_input) > 0 and attempt < MAX_ATTEMPT:
        model, tokenizer = load_model(model_name)
        for i in tqdm(range(0, len(filtered_input), batch_size)):
            end_index = min(i + batch_size, len(filtered_input))
            batch = filtered_input[i:end_index]
            prompts = [v["prompt"] for v in batch]
            final_prompts = []
            for prompt in prompts:
                for i in range(num_of_results):
                    final_prompts.append(copy.deepcopy(prompt))
            prompts = final_prompts
            print(f"Processing batch {i} to {end_index}")
            logging.debug(f"Processing batch {i} to {end_index}")
            try:
                results = forward_with_budget(
                    prompts,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    num_of_results=num_of_results,
                    seed=seed,
                    budget=budget,
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
            True, dir_prefix
        )
