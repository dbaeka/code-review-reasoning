import logging
from time import sleep

import torch
from tqdm import tqdm

from src.service.common import extract_cot_and_answer, get_unprocessed_examples, save_results
from unsloth import FastLanguageModel

unsloth_model = None
tokenizer = None
max_seq_length = 4096


def load_model(model_name: str):
    global unsloth_model, tokenizer
    if unsloth_model is not None:
        return unsloth_model, tokenizer
    unsloth_model, tokenizer = FastLanguageModel.from_pretrained(model_name, max_seq_length=max_seq_length,
                                                                 load_in_4bit=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    FastLanguageModel.for_inference(unsloth_model)
    return unsloth_model, tokenizer


def forward(
        inputs,
        model,
        tokenizer,
        max_new_tokens: int = 2048,
        temperature: float = 0.05,
        batch_size: int = 32,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False
):
    logging.debug("Generating model response")
    outputs = model.generate(
        **inputs,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        stop_strings=["</s>"],
        temperature=temperature,
        use_cache=True,
        num_return_sequences=num_of_results,
        do_sample=True,
        top_p=1,
        seed=[seed, seed] if seed is not None else None
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logging.debug("Total decoded: " + str(len(decoded)))
    all_results = []
    for idx in range(0, batch_size):
        result = []
        prompt_results = decoded[idx * num_of_results: (idx + 1) * num_of_results]
        for seq_idx, text in enumerate(prompt_results):
            logging.debug(f"Prompt {idx + 1} - Decoded Sequence {seq_idx + 1}: {text.replace(tokenizer.pad_token, '')}")
            logging.debug("_" * 70)
            result.append(extract_cot_and_answer(text, is_reasoning_model))
        all_results.append(result)
    logging.debug(f"Total number of results for the batch: {len(all_results)}")
    print(len(all_results))
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

    model, tokenizer = load_model(model_name)

    for i in tqdm(range(0, len(filtered_input), batch_size)):
        end_index = min(i + batch_size, len(filtered_input))
        batch = filtered_input[i:end_index]

        prompts = [v["prompt"] for v in batch]

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(texts, padding_side="left", padding="longest", return_tensors="pt").to(device)

        print(f"Processing batch {i} to {end_index}")
        logging.debug(f"Processing batch {i} to {end_index}")
        try:
            results = forward(
                inputs,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                batch_size=batch_size,
                num_of_results=num_of_results,
                seed=seed,
                is_reasoning_model=is_reasoning_model
            )
            save_results(batch, results, existing_results, output_path)
        except Exception as e:
            logging.error("Error: ", e)
            sleep(pause_duration)

    logging.info(f"Completed processing shard {shard_index}")
